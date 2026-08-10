from pathlib import Path

import torch
from torch.ao.quantization import (
    disable_observer,
    enable_observer,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)


def _freeze_teacher(teacher):
    """Freeze all teacher parameters and switch it to inference semantics.

    Teachers that expose return_all_feats keep BaseModel.forward's training
    branch (full output dict) while every BatchNorm stays in eval mode so the
    teacher produces deterministic running-stat features. Wrapper teachers are
    set to eval so all BatchNorm layers use running statistics; CTCHead keeps
    its training flag so raw logits (not Softmax) are emitted, matching the
    student training graph used by CTCLoss and KD.
    """
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    if getattr(teacher, "return_all_feats", False):
        teacher.train()
        for module in teacher.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return
    teacher.eval()
    model = getattr(teacher, "model", teacher)
    head = getattr(model, "head", None)
    if head is not None:
        ctc_head = getattr(head, "ctc_head", None)
        if ctc_head is not None:
            ctc_head.training = True


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler=None,
        device="cpu",
        amp=False,
        grad_clip_norm=None,
        teacher=None,
        kd_criterion=None,
        kd_weight=1.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.grad_clip_norm = grad_clip_norm
        self.teacher = teacher
        self.kd_criterion = kd_criterion
        self.kd_weight = float(kd_weight)
        if (teacher is None) != (kd_criterion is None):
            raise ValueError(
                "Knowledge distillation requires both a teacher and a KD criterion."
            )
        if teacher is not None:
            if self.kd_weight < 0:
                raise ValueError("KD weight must be non-negative.")
            _freeze_teacher(teacher)
            self.teacher.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.epoch = 0
        self.global_step = 0
        self.observers_frozen = False
        self.best_validation_loss = float("inf")
        self.best_validation_metric_name = None
        self.best_validation_metric_value = None
        self._exported_mode = None
        if isinstance(self.model, torch.fx.GraphModule):
            move_exported_model_to_train(self.model)
            self._exported_mode = "train"
        self.model.to(self.device)

    def _set_model_mode(self, training):
        if not isinstance(self.model, torch.fx.GraphModule):
            if training:
                self.model.train()
            return
        requested_mode = "train" if training else "eval"
        if self._exported_mode == requested_mode:
            return
        if training:
            move_exported_model_to_train(self.model)
        else:
            move_exported_model_to_eval(self.model)
        # PT2E mode replacement can introduce fresh CPU control tensors.
        self.model.to(self.device)
        self._exported_mode = requested_mode

    def _forward(self, images, targets):
        is_full_recognition = (
            getattr(self.model, "graph_role", None) == "pretrained_train"
            and getattr(self.model, "model_type", None) == "rec"
        )
        if is_full_recognition:
            required = ("targets", "gtc_targets", "target_lengths", "valid_ratio")
            missing = [name for name in required if name not in targets]
            if missing:
                raise ValueError(f"Full recognition targets are missing: {missing}")
            data = [
                targets["targets"],
                targets["gtc_targets"],
                targets["target_lengths"],
                targets["valid_ratio"],
            ]
            if isinstance(self.model, torch.fx.GraphModule):
                outputs = self.model(images, targets["gtc_targets"])
                if isinstance(outputs, (tuple, list)):
                    output_names = getattr(
                        self.model,
                        "output_names",
                        ("ctc", "ctc_neck", "gtc"),
                    )
                    if len(outputs) != len(output_names):
                        raise ValueError(
                            "Full recognition prepared output count does not match "
                            f"its schema: {len(outputs)} != {len(output_names)}"
                        )
                    return dict(zip(output_names, outputs))
                return outputs
            return self.model(images, data=data)
        return self.model(images)

    def _teacher_forward(self, images, targets):
        if self.teacher is None:
            raise RuntimeError("No teacher is configured.")
        is_full_recognition = (
            getattr(self.teacher, "graph_role", None) == "pretrained_train"
            and getattr(self.teacher, "model_type", None) == "rec"
        )
        if is_full_recognition:
            required = ("targets", "gtc_targets", "target_lengths", "valid_ratio")
            missing = [name for name in required if name not in targets]
            if missing:
                raise ValueError(f"Teacher full recognition targets are missing: {missing}")
            data = [
                targets["targets"],
                targets["gtc_targets"],
                targets["target_lengths"],
                targets["valid_ratio"],
            ]
            return self.teacher(images, data=data)
        return self.teacher(images)

    def _kd_losses(self, outputs, teacher_outputs):
        kd_losses = self.kd_criterion(outputs, teacher_outputs)
        if not isinstance(kd_losses, dict) or "kd_loss" not in kd_losses:
            raise ValueError("KD criterion must return a dict with key 'kd_loss'.")
        return kd_losses

    def _combine_losses(self, task_losses, kd_losses):
        if self.teacher is None:
            return task_losses
        combined = dict(task_losses)
        combined.update(kd_losses)
        combined["loss"] = (
            task_losses["loss"] + self.kd_weight * kd_losses["kd_loss"]
        )
        return combined

    def train_step(self, images, targets):
        self._set_model_mode(training=True)
        images = images.to(self.device, non_blocking=True)
        targets = self._to_device(targets)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.amp,
        ):
            outputs = self._forward(images, targets)
            task_losses = self.criterion(outputs, targets)
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_outputs = self._teacher_forward(images, targets)
                kd_losses = self._kd_losses(outputs, teacher_outputs)
            else:
                kd_losses = None
            losses = self._combine_losses(task_losses, kd_losses)
            loss = losses["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {float(loss)}")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("Training step produced no parameter gradients.")
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Training step produced non-finite gradients.")
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm,
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1
        return {
            name: float(value.detach())
            for name, value in losses.items()
            if torch.is_tensor(value) and value.numel() == 1
        }

    def evaluate(self, loader, metric=None):
        if metric is not None:
            metric.reset()
        batch_counters = {
            name: buffer.detach().clone()
            for name, buffer in self.model.named_buffers()
            if name.endswith("num_batches_tracked")
        }
        if isinstance(self.model, torch.fx.GraphModule):
            self._set_model_mode(training=False)
        else:
            set_validation_mode = getattr(
                self.model,
                "set_validation_mode",
                None,
            )
            if callable(set_validation_mode):
                set_validation_mode()
            else:
                self.model.eval()
        self.model.apply(disable_observer)
        totals = {}
        steps = 0
        try:
            with torch.no_grad():
                for images, targets in loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = self._to_device(targets)
                    with torch.amp.autocast(
                        device_type=self.device.type,
                        enabled=self.amp,
                    ):
                        outputs = self._forward(images, targets)
                        task_losses = self.criterion(outputs, targets)
                        if self.teacher is not None:
                            with torch.no_grad():
                                teacher_outputs = self._teacher_forward(
                                    images,
                                    targets,
                                )
                            kd_losses = self._kd_losses(outputs, teacher_outputs)
                        else:
                            kd_losses = None
                        losses = self._combine_losses(task_losses, kd_losses)
                    if metric is not None:
                        metric.update(outputs, targets)
                    for name, value in losses.items():
                        if torch.is_tensor(value) and value.numel() == 1:
                            totals[name] = totals.get(name, 0.0) + float(value)
                    steps += 1
        finally:
            for name, value in batch_counters.items():
                owner_name, _, buffer_name = name.rpartition(".")
                owner = (
                    self.model.get_submodule(owner_name)
                    if owner_name
                    else self.model
                )
                getattr(owner, buffer_name).copy_(value)
            if not self.observers_frozen:
                self.model.apply(enable_observer)
        if steps == 0:
            raise ValueError("Validation loader is empty.")
        results = {name: value / steps for name, value in totals.items()}
        if metric is not None:
            metric_results = metric.compute()
            duplicate_names = set(results).intersection(metric_results)
            if duplicate_names:
                raise ValueError(
                    f"Validation loss and metric names overlap: {sorted(duplicate_names)}"
                )
            results.update(metric_results)
        return results

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def freeze_observers(self):
        self.model.apply(disable_observer)
        self.observers_frozen = True

    def state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "observers_frozen": self.observers_frozen,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_metric_name": self.best_validation_metric_name,
            "best_validation_metric_value": self.best_validation_metric_value,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        self.best_validation_loss = float(
            state.get("best_validation_loss", float("inf"))
        )
        self.best_validation_metric_name = state.get("best_validation_metric_name")
        best_metric_value = state.get("best_validation_metric_value")
        self.best_validation_metric_value = (
            None if best_metric_value is None else float(best_metric_value)
        )
        if state.get("observers_frozen", False):
            self.freeze_observers()

    def save_checkpoint(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        state["metadata"] = metadata or {}
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state)
        return state.get("metadata", {})

    def _to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        return value

import math

import torch

from .data import DetectionDataset, RecognitionDataset
from .losses import CTCLoss, DBLoss, MultiLoss
from .model_builder import resolve_config_path


def config_value(value, fallback):
    return fallback if value is None else value


def build_dataset(
    task,
    model_config,
    config,
    image_shape,
    label_file,
    data_dir,
    *,
    return_polygons=False,
    rec_multi_head=False,
    augmentation="none",
):
    if task == "det":
        return DetectionDataset(
            label_file,
            data_dir=data_dir,
            image_shape=image_shape,
            return_polygons=return_polygons,
            augmentation=augmentation,
        )
    global_config = config["Global"]
    dictionary_path = resolve_config_path(
        global_config["character_dict_path"],
        config_path=model_config,
    )
    return RecognitionDataset(
        label_file,
        dictionary_path=dictionary_path,
        data_dir=data_dir,
        image_shape=image_shape,
        max_text_length=int(global_config.get("max_text_length", 25)),
        use_space_char=bool(global_config.get("use_space_char", False)),
        multi_head=rec_multi_head,
        augmentation=augmentation,
    )


def build_criterion(task, config, *, rec_multi_head=False):
    if task == "rec":
        if rec_multi_head:
            loss_config = config.get("Loss", {})
            if loss_config.get("name") not in (None, "MultiLoss"):
                raise ValueError("Full recognition training requires MultiLoss.")
            return MultiLoss(
                weight_1=float(loss_config.get("weight_1", 1.0)),
                weight_2=float(loss_config.get("weight_2", 1.0)),
            )
        return CTCLoss()
    loss_config = config.get("Loss", {})
    return DBLoss(
        balance_loss=bool(loss_config.get("balance_loss", True)),
        main_loss_type=str(loss_config.get("main_loss_type", "DiceFocalLoss")),
        alpha=float(loss_config.get("alpha", 5.0)),
        beta=float(loss_config.get("beta", 10.0)),
        ohem_ratio=float(loss_config.get("ohem_ratio", 3.0)),
        focal_alpha=float(loss_config.get("focal_alpha", 0.25)),
        focal_gamma=float(loss_config.get("focal_gamma", 2.0)),
        dice_weight=float(loss_config.get("dice_weight", 1.0)),
        focal_weight=float(loss_config.get("focal_weight", 1.0)),
        aux_weight_p4=float(loss_config.get("aux_weight_p4", 0.0)),
        aux_weight_p3=float(loss_config.get("aux_weight_p3", 0.0)),
        aux_weight_p2=float(loss_config.get("aux_weight_p2", 0.0)),
    )


def build_optimizer(
    model,
    config,
    learning_rate=None,
    weight_decay=None,
    lab_lr_multiplier=None,
    ctc_fc_weight_decay=None,
    optimizer_name=None,
    momentum=None,
):
    optimizer_config = config.get("Optimizer", {})
    lr_config = optimizer_config.get("lr", {})
    learning_rate = config_value(
        learning_rate,
        float(lr_config.get("learning_rate", 1.0e-3)),
    )
    regularizer = optimizer_config.get("regularizer", {})
    weight_decay = config_value(
        weight_decay,
        float(regularizer.get("factor", 0.0)),
    )
    name = str(
        config_value(optimizer_name, optimizer_config.get("name", "Adam"))
    ).lower()
    parameter_groups = {}
    for parameter_name, parameter in model.named_parameters():
        qualified_name = f".{parameter_name}."
        configured_lr_multiplier = (
            lab_lr_multiplier
            if lab_lr_multiplier is not None and ".lab." in qualified_name
            else None
        )
        lr_multiplier = config_value(
            configured_lr_multiplier,
            getattr(parameter, "_paddle_lr_multiplier", 1.0),
        )
        is_ctc_fc = ".ctc_head.fc" in qualified_name
        configured_weight_decay = (
            ctc_fc_weight_decay
            if ctc_fc_weight_decay is not None
            and is_ctc_fc
            else None
        )
        parameter_weight_decay = config_value(
            configured_weight_decay,
            getattr(parameter, "_paddle_weight_decay", weight_decay),
        )
        group_name = (
            "ctc_fc"
            if is_ctc_fc
            else "paddle_no_decay"
            if parameter_weight_decay == 0.0
            else "lab"
            if lr_multiplier != 1.0
            else "default"
        )
        key = (group_name, lr_multiplier, parameter_weight_decay)
        parameter_groups.setdefault(key, []).append(parameter)
    parameters = [
        {
            "params": group_parameters,
            "group_name": group_name,
            "lr": learning_rate * lr_multiplier,
            "weight_decay": parameter_weight_decay,
        }
        for (
            group_name,
            lr_multiplier,
            parameter_weight_decay,
        ), group_parameters in parameter_groups.items()
    ]
    if name == "adam":
        betas = (
            float(optimizer_config.get("beta1", 0.9)),
            float(optimizer_config.get("beta2", 0.999)),
        )
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name in ("sgd", "momentum"):
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(
                config_value(momentum, optimizer_config.get("momentum", 0.9))
            ),
            weight_decay=weight_decay,
        )
    unsupported = optimizer_name or optimizer_config.get("name")
    raise ValueError(f"Unsupported optimizer: {unsupported}")


def build_scheduler(
    optimizer,
    config,
    epochs,
    warmup_epochs=None,
    final_factor=None,
):
    lr_config = config.get("Optimizer", {}).get("lr", {})
    scheduler_name = str(lr_config.get("name", "Cosine")).lower()
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported learning-rate scheduler: {scheduler_name}")
    warmup_epochs = int(config_value(warmup_epochs, lr_config.get("warmup_epoch", 0)))
    final_factor = float(config_value(final_factor, 0.0))
    if warmup_epochs < 0:
        raise ValueError("Warmup epochs must be non-negative.")
    if not 0 <= final_factor <= 1:
        raise ValueError("Final learning-rate factor must be in [0, 1].")

    def schedule(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        cosine_epochs = max(epochs - warmup_epochs, 1)
        progress = min(max(epoch - warmup_epochs, 0), cosine_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress / cosine_epochs))
        return final_factor + (1.0 - final_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def update_best_validation(trainer, validation, metric=None):
    loss_improved = (
        "loss" in validation
        and validation["loss"] < trainer.best_validation_loss
    )
    if loss_improved:
        trainer.best_validation_loss = validation["loss"]
    if metric is None:
        return loss_improved

    metric_name = metric.main_indicator
    if metric_name not in validation:
        raise ValueError(
            f"Validation did not produce main indicator {metric_name!r}."
        )
    metric_value = float(validation[metric_name])
    is_best = (
        trainer.best_validation_metric_name != metric_name
        or trainer.best_validation_metric_value is None
        or metric_value > trainer.best_validation_metric_value
    )
    if is_best:
        trainer.best_validation_metric_name = metric_name
        trainer.best_validation_metric_value = metric_value
    return is_best

import copy

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    move_exported_model_to_eval,
)

from pytorchocr.training import CTCLoss, build_validation_metric

from .errors import ctc_collapse


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def evaluate_recognition_qat(
    model,
    loader,
    config,
    metadata,
    device,
    *,
    fake_quant,
):
    model = copy.deepcopy(model)
    full_training_graph = metadata.get("rec_graph") == "pretrained_train"
    model.apply(disable_observer)
    model.apply(enable_fake_quant if fake_quant else disable_fake_quant)
    move_exported_model_to_eval(model)
    model.to(device)
    criterion = CTCLoss()
    metric = build_validation_metric(
        "rec",
        config,
        config_path=metadata["model_config"],
    )
    loss_sum = 0.0
    batches = 0
    blank_values = 0
    argmax_values = 0
    predicted_length_sum = 0
    target_length_sum = 0
    logits_sum = 0.0
    logits_square_sum = 0.0
    logits_abs_max = 0.0
    logits_values = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            if full_training_graph:
                outputs = model(images, targets["gtc_targets"].to(device))
                if isinstance(outputs, dict):
                    logits = outputs.get(
                        "ctc",
                        outputs.get("head_out", {}).get("ctc"),
                    )
                else:
                    logits = outputs[0]
            else:
                logits = model(images)
            losses = criterion(logits, _to_device(targets, device))
            loss_sum += float(losses["loss"])
            batches += 1
            metric.update(logits, targets)
            indices = logits.argmax(dim=-1)
            blank_values += int((indices == 0).sum())
            argmax_values += indices.numel()
            predicted_length_sum += sum(
                len(ctc_collapse(sequence)) for sequence in indices
            )
            target_length_sum += int(targets["target_lengths"].sum())
            float_logits = logits.float()
            logits_sum += float(float_logits.sum())
            logits_square_sum += float(float_logits.square().sum())
            logits_abs_max = max(logits_abs_max, float(float_logits.abs().max()))
            logits_values += float_logits.numel()
    task_metrics = metric.compute()
    mean = logits_sum / logits_values
    variance = max(logits_square_sum / logits_values - mean * mean, 0.0)
    return {
        "fake_quant_enabled": fake_quant,
        "loss": loss_sum / batches,
        **task_metrics,
        "blank_argmax_fraction": blank_values / argmax_values,
        "mean_predicted_length": predicted_length_sum / metric.samples,
        "mean_target_length": target_length_sum / metric.samples,
        "logits_mean": mean,
        "logits_std": variance**0.5,
        "logits_abs_max": logits_abs_max,
    }

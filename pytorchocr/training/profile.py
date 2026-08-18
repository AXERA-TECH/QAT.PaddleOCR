from dataclasses import dataclass
from pathlib import Path

import yaml

from pytorchocr.utils.hashing import file_sha256


_PROFILE_KEYS = {"name", "task", "model_name", "qat", "qat_config", "training"}
_TRAINING_KEYS = {
    "epochs",
    "batch_size",
    "workers",
    "learning_rate",
    "weight_decay",
    "image_shape",
    "amp",
    "grad_clip_norm",
    "observer_freeze_epoch",
    "save_every",
    "reparameterize",
    "warmup_epochs",
    "lr_final_factor",
    "seed",
    "rec_ctc_backbone_grad",
    "rec_graph",
    "lab_lr_multiplier",
    "ctc_fc_weight_decay",
    "optimizer",
    "momentum",
    "dynamic_heights",
    "augmentation",
    "multi_scale_training",
    "float_accuracy_baseline",
    "epoch2_max_accuracy_drop",
    "lsq",
    "insert_identity_bn",
    "bn_statistics_steps",
    "bn_training_momentum",
    "freeze_bn_stats",
    "kd",
    "teacher_weights",
    "teacher_model_config",
    "kd_mode",
    "kd_weight",
    "kd_temperature",
    "kd_neck_weight",
    "kd_backbone_weight",
}
_RESUME_CONTRACT_KEYS = (
    "task",
    "model_config",
    "training_profile_sha256",
    "qat",
    "qat_config_sha256",
    "torch_version",
    "reparameterized",
    "lsq",
    "insert_identity_bn",
    "bn_statistics_steps",
    "bn_training_momentum",
    "freeze_bn_stats",
    "kd",
    "teacher_model_config",
    "kd_mode",
    "kd_weight",
    "kd_temperature",
    "kd_neck_weight",
    "kd_backbone_weight",
    "rec_ctc_backbone_grad",
    "rec_graph",
    "optimizer",
    "optimizer_momentum",
    "dynamic_heights",
    "augmentation",
    "multi_scale_training",
    "float_accuracy_baseline",
    "epoch2_max_accuracy_drop",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LEGACY_PROJECT_ROOTS = (
    Path("/home/heqi/project/PaddleOCR/route2/PaddleOCR2Pytorch-QAT"),
)


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    path: Path
    sha256: str
    task: str | None
    model_name: str | None
    qat: bool | None
    qat_config: Path | None
    training: dict

    def validate_contract(self, task, model_name):
        if self.task is not None and self.task != task:
            raise ValueError(
                f"Training profile task is {self.task!r}, but --task is {task!r}."
            )
        if self.model_name is not None and self.model_name != model_name:
            raise ValueError(
                "Training profile model_name is "
                f"{self.model_name!r}, but model YAML declares {model_name!r}."
            )


def _validate_training_values(training, qat=False):
    positive_ints = ("epochs", "batch_size", "save_every")
    for key in positive_ints:
        if key in training and (
            not isinstance(training[key], int) or training[key] <= 0
        ):
            raise ValueError(f"Training profile {key} must be a positive integer.")
    for key in ("workers", "warmup_epochs", "observer_freeze_epoch", "seed"):
        if key in training and training[key] is not None and (
            not isinstance(training[key], int) or training[key] < 0
        ):
            raise ValueError(f"Training profile {key} must be a non-negative integer.")
    for key in ("learning_rate", "grad_clip_norm"):
        if key in training and training[key] is not None and training[key] <= 0:
            raise ValueError(f"Training profile {key} must be positive.")
    if "weight_decay" in training and training["weight_decay"] < 0:
        raise ValueError("Training profile weight_decay must be non-negative.")
    if (
        "ctc_fc_weight_decay" in training
        and training["ctc_fc_weight_decay"] is not None
        and training["ctc_fc_weight_decay"] < 0
    ):
        raise ValueError("Training profile ctc_fc_weight_decay must be non-negative.")
    if "lab_lr_multiplier" in training and training["lab_lr_multiplier"] <= 0:
        raise ValueError("Training profile lab_lr_multiplier must be positive.")
    if "optimizer" in training and str(training["optimizer"]).lower() not in (
        "adam",
        "adamw",
        "sgd",
        "momentum",
    ):
        raise ValueError(
            "Training profile optimizer must be Adam, AdamW, SGD, or Momentum."
        )
    if qat and "optimizer" in training and str(training["optimizer"]).lower() in (
        "adam",
    ):
        raise ValueError(
            "QAT training profile optimizer must be AdamW or SGD; Adam's L2 "
            "weight decay corrupts LSQ learnable scale/zero_point parameters."
        )
    if "momentum" in training and (
        not isinstance(training["momentum"], (int, float))
        or not 0 <= training["momentum"] < 1
    ):
        raise ValueError("Training profile momentum must be in [0, 1).")
    if "lr_final_factor" in training and not 0 <= training["lr_final_factor"] <= 1:
        raise ValueError("Training profile lr_final_factor must be in [0, 1].")
    for key in ("float_accuracy_baseline", "epoch2_max_accuracy_drop"):
        if key in training and (
            not isinstance(training[key], (int, float))
            or not 0 <= training[key] <= 1
        ):
            raise ValueError(f"Training profile {key} must be in [0, 1].")
    accuracy_guard_keys = {
        "float_accuracy_baseline",
        "epoch2_max_accuracy_drop",
    }
    if len(accuracy_guard_keys.intersection(training)) == 1:
        raise ValueError(
            "Training profile float_accuracy_baseline and "
            "epoch2_max_accuracy_drop must be configured together."
        )
    if "image_shape" in training:
        image_shape = training["image_shape"]
        if (
            not isinstance(image_shape, list)
            or len(image_shape) != 3
            or any(not isinstance(value, int) or value <= 0 for value in image_shape)
        ):
            raise ValueError(
                "Training profile image_shape must be three positive integers."
            )
    if "dynamic_heights" in training:
        heights = training["dynamic_heights"]
        if (
            not isinstance(heights, list)
            or len(heights) < 2
            or any(not isinstance(value, int) or value <= 0 for value in heights)
            or sorted(set(heights)) != heights
        ):
            raise ValueError(
                "Training profile dynamic_heights must contain at least two "
                "sorted unique positive integers."
            )
        step = heights[1] - heights[0]
        if heights != list(range(heights[0], heights[-1] + step, step)):
            raise ValueError(
                "Training profile dynamic_heights must form an arithmetic sequence."
            )
        if "image_shape" in training and training["image_shape"][1] not in heights:
            raise ValueError(
                "Training profile image_shape height must be in dynamic_heights."
            )
    for key in (
        "amp",
        "reparameterize",
        "rec_ctc_backbone_grad",
        "multi_scale_training",
        "lsq",
        "insert_identity_bn",
        "freeze_bn_stats",
        "kd",
    ):
        if key in training and not isinstance(training[key], bool):
            raise ValueError(f"Training profile {key} must be boolean.")
    for key in ("kd_weight", "kd_neck_weight", "kd_backbone_weight"):
        if key in training and training[key] is not None and training[key] < 0:
            raise ValueError(f"Training profile {key} must be non-negative.")
    if "bn_statistics_steps" in training and training["bn_statistics_steps"] <= 0:
        raise ValueError("Training profile bn_statistics_steps must be positive.")
    if "bn_training_momentum" in training:
        value = training["bn_training_momentum"]
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError(
                "Training profile bn_training_momentum must be in (0, 1]."
            )
    if "kd_temperature" in training and training["kd_temperature"] <= 0:
        raise ValueError("Training profile kd_temperature must be positive.")
    if "kd_mode" in training and training["kd_mode"] not in (
        "logits",
        "logits_mse",
        "maps",
    ):
        raise ValueError(
            "Training profile kd_mode must be 'logits', 'logits_mse', or 'maps'."
        )
    if "rec_graph" in training and training["rec_graph"] not in (
        "deploy",
        "pretrained_train",
    ):
        raise ValueError(
            "Training profile rec_graph must be 'deploy' or 'pretrained_train'."
        )
    if "augmentation" in training and training["augmentation"] not in (
        "none",
        "paddle",
    ):
        raise ValueError(
            "Training profile augmentation must be 'none' or 'paddle'."
        )


def load_training_profile(path):
    if path is None:
        return None
    profile_path = Path(path).resolve()
    with open(profile_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Training profile must contain a YAML mapping.")
    unknown = set(config) - _PROFILE_KEYS
    if unknown:
        raise ValueError(f"Unknown training profile keys: {sorted(unknown)}")
    training = config.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("Training profile 'training' must be a mapping.")
    unknown_training = set(training) - _TRAINING_KEYS
    if unknown_training:
        raise ValueError(
            f"Unknown training settings: {sorted(unknown_training)}"
        )
    _validate_training_values(training, qat=bool(config.get("qat", False)))

    qat_config = config.get("qat_config")
    if qat_config is not None:
        qat_config = (profile_path.parent / qat_config).resolve()
        if not qat_config.is_file():
            raise FileNotFoundError(
                f"Training profile QAT config does not exist: {qat_config}"
            )
    qat = config.get("qat")
    if qat is not None and not isinstance(qat, bool):
        raise ValueError("Training profile qat must be boolean.")
    return TrainingProfile(
        name=str(config.get("name", profile_path.stem)),
        path=profile_path,
        sha256=file_sha256(profile_path),
        task=config.get("task"),
        model_name=config.get("model_name"),
        qat=qat,
        qat_config=qat_config,
        training=dict(training),
    )


def profile_value(cli_value, profile, key, fallback=None):
    if cli_value is not None:
        return cli_value
    if profile is not None and key in profile.training:
        return profile.training[key]
    return fallback


def copy_run_configs(output_dir, *, model_config, training_profile=None, qat_config=None, teacher_model_config=None):
    """Copy the training, network and quantization configs into the run output directory.

    Configs are copied verbatim (``copy2``) so each run directory is
    self-contained and reproducible without relying on the source tree.
    Returns a list of ``(source, destination)`` pairs that were copied.
    """
    import shutil

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "model_config": model_config,
        "training_profile": training_profile,
        "qat_config": qat_config,
        "teacher_model_config": teacher_model_config,
    }
    copied = []
    for name, source in sources.items():
        if source is None:
            continue
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Cannot copy {name} for the run: {source_path}"
            )
        destination = output_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied.append((str(source_path), str(destination)))
    return copied


def epoch2_accuracy_guard(
    epoch,
    validation,
    *,
    float_accuracy_baseline=None,
    max_accuracy_drop=None,
):
    if float_accuracy_baseline is None and max_accuracy_drop is None:
        return None
    if float_accuracy_baseline is None or max_accuracy_drop is None:
        raise ValueError(
            "float_accuracy_baseline and max_accuracy_drop are required together."
        )
    if epoch != 2:
        return None
    if "acc" not in validation:
        raise ValueError("The epoch-2 accuracy guard requires validation metric 'acc'.")
    baseline = float(float_accuracy_baseline)
    current = float(validation["acc"])
    threshold = float(max_accuracy_drop)
    drop = baseline - current
    return {
        "epoch": int(epoch),
        "float_accuracy_baseline": baseline,
        "validation_accuracy": current,
        "accuracy_drop": drop,
        "max_accuracy_drop": threshold,
        "triggered": drop + 1.0e-12 >= threshold,
    }


def _project_relative_path(value):
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if not path.is_absolute():
        return None
    for root in (*_LEGACY_PROJECT_ROOTS, _PROJECT_ROOT):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return None


def relocate_project_path(value):
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value)
    for root in _LEGACY_PROJECT_ROOTS:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) == 1 and relative.suffix in (".pth", ".pdparams"):
            relative = Path("weights") / relative
        return str(_PROJECT_ROOT / relative)
    return str(path)


def relocate_checkpoint_metadata(metadata):
    relocated = dict(metadata)
    for key in ("model_config", "qat_config", "weights"):
        if relocated.get(key) is not None:
            relocated[key] = relocate_project_path(relocated[key])
    return relocated


def _same_relocated_model_config(saved, current):
    saved_relative = _project_relative_path(saved)
    current_relative = _project_relative_path(current)
    return (
        saved_relative is not None
        and current_relative is not None
        and saved_relative == current_relative
    )


def validate_resume_contract(saved_metadata, current_metadata):
    mismatches = []
    for key in _RESUME_CONTRACT_KEYS:
        if key in ("rec_ctc_backbone_grad", "lsq", "insert_identity_bn", "kd"):
            saved = bool(saved_metadata.get(key, False))
            current = bool(current_metadata.get(key, False))
        elif key == "rec_graph":
            saved = saved_metadata.get(
                key,
                "deploy" if saved_metadata.get("task") == "rec" else None,
            )
            current = current_metadata.get(
                key,
                "deploy" if current_metadata.get("task") == "rec" else None,
            )
        elif key == "optimizer":
            saved = str(saved_metadata.get(key, "adam")).lower()
            current = str(current_metadata.get(key, "adam")).lower()
        elif key == "optimizer_momentum":
            saved = float(saved_metadata.get(key, 0.0))
            current = float(current_metadata.get(key, 0.0))
        elif key == "dynamic_heights":
            saved = list(saved_metadata.get(key) or [])
            current = list(current_metadata.get(key) or [])
        elif key == "augmentation":
            saved = saved_metadata.get(key, "none")
            current = current_metadata.get(key, "none")
        elif key == "multi_scale_training":
            saved = bool(saved_metadata.get(key, False))
            current = bool(current_metadata.get(key, False))
        else:
            saved = saved_metadata.get(key)
            current = current_metadata.get(key)
        if saved != current and not (
            key == "model_config"
            and _same_relocated_model_config(saved, current)
        ):
            mismatches.append(f"{key}: checkpoint={saved!r}, current={current!r}")
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Resume checkpoint contract mismatch: {details}")

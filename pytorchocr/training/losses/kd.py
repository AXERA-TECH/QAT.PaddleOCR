"""Knowledge-distillation losses for v5/v6 det/rec float and QAT training.

KD contract: a frozen float teacher and the trainable student share the same
graph role, so their outputs align structurally. KD consumes named outputs
extracted through normalize_outputs(): task heads (CTC logits / DB maps) and,
when enabled, intermediate features (backbone_out / neck_out).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "KDLogitsLoss",
    "KDMapsLoss",
    "KDFeatureLoss",
    "KDCompositeLoss",
    "build_kd_criterion",
    "normalize_outputs",
]

_CTC_KEYS = ("ctc", "ctc_neck", "gtc", "backbone_out", "neck_out")
_DET_KEYS = ("maps", "backbone_out", "neck_out")


def _dict_value(outputs, keys):
    for key in keys:
        if key in outputs:
            return outputs[key]
    raise KeyError(f"Outputs are missing any of {keys}; got {sorted(outputs)}")


def normalize_outputs(outputs, task, rec_graph="deploy"):
    """Normalize student/teacher outputs to a keyed dict.

    rec deploy:  RecCTCWrapper tensor -> {"ctc": logits}
    rec full:    FullRecTrainingWrapper dict {ctc, ctc_neck, gtc, ...} or
                 BaseModel return_all_feats dict {backbone_out, neck_out,
                 head_out: {ctc, ...}, ...}
    det:         DetTrainingWrapper tuple (shrink, threshold, binary) or
                 expose_intermediates dict / BaseModel dict {maps, neck_out, ...}
    """
    if task == "rec":
        if isinstance(outputs, dict):
            normalized = {}
            if "head_out" in outputs:
                head = outputs["head_out"]
                normalized["ctc"] = head.get("ctc", head)
                if "ctc_neck" in head:
                    normalized["ctc_neck"] = head["ctc_neck"]
            for key in _CTC_KEYS:
                if key in outputs:
                    normalized[key] = outputs[key]
            if "neck_out" in outputs and "ctc_neck" not in normalized:
                normalized["ctc_neck"] = outputs["neck_out"]
            if "ctc" not in normalized:
                raise KeyError(
                    f"Recognition outputs have no CTC logits: {sorted(outputs)}"
                )
            return normalized
        if isinstance(outputs, (tuple, list)):
            ctc = outputs[0]
            normalized = {"ctc": ctc}
            if len(outputs) > 1:
                normalized["ctc_neck"] = outputs[1]
            if len(outputs) > 2:
                normalized["gtc"] = outputs[2]
            return normalized
        return {"ctc": outputs}
    if task == "det":
        if isinstance(outputs, dict):
            normalized = dict(outputs)
            if "maps" not in normalized and "head_out" in outputs:
                normalized["maps"] = outputs["head_out"]
            if "neck_out" in outputs and "neck_out" not in normalized:
                normalized["neck_out"] = outputs["neck_out"]
            if "maps" not in normalized:
                raise KeyError(
                    f"Detection outputs have no maps: {sorted(outputs)}"
                )
            return normalized
        if isinstance(outputs, (tuple, list)):
            return {"maps": tuple(outputs)}
        return {"maps": outputs}
    raise ValueError(f"Unsupported KD task: {task}")


class KDLogitsLoss(nn.Module):
    """CTC-logits KD for recognition.

    mode='kl': temperature-scaled KL divergence
        loss = T^2 * mean(KL(softmax(s/T) || softmax(t/T)))
    mode='mse': mean squared error on logits.
    """

    def __init__(self, mode="kl", temperature=4.0):
        super().__init__()
        if mode not in ("kl", "mse"):
            raise ValueError(f"KDLogitsLoss mode must be 'kl' or 'mse', got {mode!r}")
        self.mode = mode
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError("KD temperature must be positive.")

    def forward(self, student, teacher):
        if not torch.is_tensor(student) or not torch.is_tensor(teacher):
            raise TypeError("KDLogitsLoss expects tensor CTC logits.")
        if student.shape != teacher.shape:
            raise ValueError(
                "Student/teacher CTC logits shape mismatch: "
                f"{tuple(student.shape)} != {tuple(teacher.shape)}"
            )
        if self.mode == "mse":
            return F.mse_loss(student, teacher.detach())
        temperature = self.temperature
        teacher_log_prob = F.log_softmax(teacher.detach() / temperature, dim=-1)
        student_log_prob = F.log_softmax(student / temperature, dim=-1)
        kld = F.kl_div(
            student_log_prob,
            teacher_log_prob,
            log_target=True,
            reduction="batchmean",
        )
        return temperature * temperature * kld


class KDMapsLoss(nn.Module):
    """DB-map KD for detection: MSE over shrink/threshold/binary maps."""

    def forward(self, student, teacher):
        student_maps = student if isinstance(student, (tuple, list)) else (student,)
        teacher_maps = teacher if isinstance(teacher, (tuple, list)) else (teacher,)
        if len(student_maps) != len(teacher_maps):
            raise ValueError(
                "Student/teacher detection map count mismatch: "
                f"{len(student_maps)} != {len(teacher_maps)}"
            )
        total = 0.0
        for index, (s_map, t_map) in enumerate(zip(student_maps, teacher_maps)):
            if not torch.is_tensor(s_map) or not torch.is_tensor(t_map):
                raise TypeError("KDMapsLoss expects tensor maps.")
            if s_map.shape != t_map.shape:
                raise ValueError(
                    f"Student/teacher map {index} shape mismatch: "
                    f"{tuple(s_map.shape)} != {tuple(t_map.shape)}"
                )
            total = total + F.mse_loss(s_map, t_map.detach())
        if isinstance(total, float):
            raise RuntimeError("KDMapsLoss received no maps.")
        return total


class KDFeatureLoss(nn.Module):
    """Feature-level KD for intermediate layers (MSE by default).

    Accepts a single tensor or a list/tuple of tensors (multi-level detection
    backbone features); list inputs are reduced by element-wise sum.
    """

    def __init__(self, mode="mse"):
        super().__init__()
        if mode not in ("mse", "l1"):
            raise ValueError(f"KDFeatureLoss mode must be 'mse' or 'l1', got {mode!r}")
        self.mode = mode

    def forward(self, student, teacher):
        student_features = student if isinstance(student, (list, tuple)) else (student,)
        teacher_features = teacher if isinstance(teacher, (list, tuple)) else (teacher,)
        if len(student_features) != len(teacher_features):
            raise ValueError(
                "Student/teacher feature level count mismatch: "
                f"{len(student_features)} != {len(teacher_features)}"
            )
        total = 0.0
        for index, (s_feature, t_feature) in enumerate(
            zip(student_features, teacher_features)
        ):
            if not torch.is_tensor(s_feature) or not torch.is_tensor(t_feature):
                raise TypeError("KDFeatureLoss expects tensor features.")
            if s_feature.shape != t_feature.shape:
                raise ValueError(
                    f"Student/teacher feature {index} shape mismatch: "
                    f"{tuple(s_feature.shape)} != {tuple(t_feature.shape)}"
                )
            if self.mode == "l1":
                total = total + F.l1_loss(s_feature, t_feature.detach())
            else:
                total = total + F.mse_loss(s_feature, t_feature.detach())
        if isinstance(total, float):
            raise RuntimeError("KDFeatureLoss received no features.")
        return total


_LOSS_FACTORIES = {
    "kl": lambda temperature: KDLogitsLoss(mode="kl", temperature=temperature),
    "logits_mse": lambda temperature: KDLogitsLoss(mode="mse"),
    "maps": lambda temperature: KDMapsLoss(),
    "mse": lambda temperature: KDFeatureLoss(mode="mse"),
    "l1": lambda temperature: KDFeatureLoss(mode="l1"),
}


class KDCompositeLoss(nn.Module):
    """Weighted KD over multiple named layers.

    layers: mapping of output key -> (loss mode, weight), e.g.
        {"ctc": ("kl", 1.0), "ctc_neck": ("mse", 0.0)}
    Keys with weight 0 are computed but reported; keys not listed are skipped.
    """

    def __init__(self, layers, temperature=4.0, task=None, rec_graph="deploy"):
        super().__init__()
        if not isinstance(layers, dict) or not layers:
            raise ValueError("KD layers must be a non-empty mapping.")
        self.layers = {}
        for key, (mode, weight) in layers.items():
            if mode not in _LOSS_FACTORIES:
                raise ValueError(f"Unsupported KD loss mode: {mode!r}")
            if float(weight) < 0:
                raise ValueError(f"KD weight for {key!r} must be non-negative.")
            self.layers[key] = (
                _LOSS_FACTORIES[mode](temperature),
                float(weight),
            )
        self.task = task
        self.rec_graph = rec_graph
        self._criterion_keys = frozenset(layers)

    def forward(self, student_outputs, teacher_outputs):
        student = normalize_outputs(
            student_outputs,
            self.task,
            rec_graph=self.rec_graph,
        )
        teacher = normalize_outputs(
            teacher_outputs,
            self.task,
            rec_graph=self.rec_graph,
        )
        total = 0.0
        losses = {}
        for key, (criterion, weight) in self.layers.items():
            if key not in student:
                raise KeyError(
                    f"KD layer {key!r} is missing from student outputs: "
                    f"{sorted(student)}"
                )
            if key not in teacher:
                raise KeyError(
                    f"KD layer {key!r} is missing from teacher outputs: "
                    f"{sorted(teacher)}"
                )
            value = criterion(student[key], teacher[key])
            losses[f"kd_{key}"] = value
            total = total + weight * value
        return {"kd_loss": total, **losses}


def default_kd_layers(task, kd_mode=None, head_weight=1.0):
    """Default KD layer mapping: task head enabled, intermediates disabled.

    kd_mode uses the CLI vocabulary: 'logits' (temperature KL), 'logits_mse',
    or 'maps'; it is normalized to the internal loss-mode vocabulary.
    """
    if task == "rec":
        mode = kd_mode or "kl"
        if mode == "logits":
            mode = "kl"
        if mode not in ("kl", "logits_mse"):
            raise ValueError(
                f"Recognition KD mode must be 'logits' or 'logits_mse', got {kd_mode!r}"
            )
        return {
            "ctc": (mode, float(head_weight)),
            "ctc_neck": ("mse", 0.0),
            "backbone_out": ("mse", 0.0),
        }
    if task == "det":
        return {
            "maps": ("maps", float(head_weight)),
            "neck_out": ("mse", 0.0),
            "backbone_out": ("mse", 0.0),
        }
    raise ValueError(f"Unsupported KD task: {task}")


def build_kd_criterion(
    task,
    kd_layers=None,
    kd_mode=None,
    kd_weight=1.0,
    kd_temperature=4.0,
    rec_graph="deploy",
):
    """Build the KD composite criterion.

    kd_layers: optional full mapping {key: (mode, weight)}; defaults to
    default_kd_layers() with intermediate layers disabled.
    """
    if kd_layers is None:
        kd_layers = default_kd_layers(task, kd_mode=kd_mode, head_weight=kd_weight)
    else:
        if not isinstance(kd_layers, dict) or not kd_layers:
            raise ValueError("--kd-layers must be a non-empty mapping.")
    return KDCompositeLoss(
        kd_layers,
        temperature=kd_temperature,
        task=task,
        rec_graph=rec_graph,
    )

import copy
import warnings
from pathlib import Path

import torch
import yaml

from pytorchocr.modeling.architectures import build_model
from pytorchocr.quantization import (
    DetInferenceWrapper,
    DetTrainingWrapper,
    RecCTCWrapper,
    reparameterize_for_deploy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_ocr_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path, "rb") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict) or "Architecture" not in config:
        raise ValueError(f"Invalid PaddleOCR model config: {config_path}")
    return config


def resolve_config_path(path, config_path=None):
    path = Path(path)
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [PROJECT_ROOT / path, Path.cwd() / path]
        # Paddle model YAMLs keep paths relative to the source PaddleOCR tree.
        candidates.append(PROJECT_ROOT / "references/PaddleOCR" / path)
        if config_path is not None:
            candidates.append(Path(config_path).resolve().parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {path}; searched: {searched}")


def rec_out_channels(config, config_path=None):
    dictionary_path = resolve_config_path(
        config["Global"]["character_dict_path"],
        config_path=config_path,
    )
    with open(dictionary_path, encoding="utf-8") as dictionary_file:
        character_count = sum(1 for _ in dictionary_file)
    character_count += int(config["Global"].get("use_space_char", False))
    ctc_channels = character_count + 1
    return {
        "CTCLabelDecode": ctc_channels,
        "SARLabelDecode": ctc_channels + 2,
        "NRTRLabelDecode": ctc_channels + 3,
    }


_REC_AUXILIARY_PREFIXES = ("head.before_gtc.", "head.gtc_head.")


def _load_recognition_state_dict(model, weights_path, graph_role):
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if graph_role == "pretrained_train":
        model.load_state_dict(state_dict, strict=True)
        return

    target_state = model.state_dict()
    if set(state_dict) == set(target_state):
        model.load_state_dict(state_dict, strict=True)
        return

    # Historical CTC-only exports saved a random, incompatible NRTR module.
    # Keep that path readable for deployment diagnostics, but never for training.
    deploy_state = {
        name: value
        for name, value in state_dict.items()
        if not name.startswith(_REC_AUXILIARY_PREFIXES)
    }
    incompatible = model.load_state_dict(deploy_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(_REC_AUXILIARY_PREFIXES)
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            "Recognition weights are incompatible: "
            f"missing={invalid_missing}, unexpected={unexpected}"
        )
    warnings.warn(
        "Loaded legacy CTC-only recognition weights for deployment; auxiliary "
        "NRTR parameters were not loaded. Use full weights for training.",
        stacklevel=2,
    )


def _load_state_dict(model, weights_path, task, graph_role="deploy"):
    if task == "det":
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return
    _load_recognition_state_dict(model, weights_path, graph_role)


def build_det_model(
    config_path,
    weights_path=None,
    reparameterize=True,
    graph_mode="training",
    insert_identity_bn=False,
    keep_bn=False,
):
    config = load_ocr_config(config_path)
    if config["Architecture"].get("model_type") != "det":
        raise ValueError("The model config is not a detection architecture.")
    if graph_mode not in ("training", "inference", "pretrained_train"):
        raise ValueError(f"Unsupported detection graph mode: {graph_mode}")
    model = build_model(config["Architecture"])
    if weights_path:
        _load_state_dict(model, weights_path, task="det")
    if reparameterize:
        reparameterize_for_deploy(
            model, insert_identity_bn=insert_identity_bn, keep_bn=keep_bn
        )
    if graph_mode == "pretrained_train":
        model.graph_role = graph_mode
        model.train()
        return model
    wrapper_type = (
        DetTrainingWrapper if graph_mode == "training" else DetInferenceWrapper
    )
    return wrapper_type(model).set_qat_capture_mode()


def build_rec_model(
    config_path,
    weights_path=None,
    reparameterize=True,
    ctc_backbone_grad=False,
    graph_role="deploy",
    insert_identity_bn=False,
    keep_bn=False,
):
    config = load_ocr_config(config_path)
    architecture = copy.deepcopy(config["Architecture"])
    if architecture.get("model_type") != "rec":
        raise ValueError("The model config is not a recognition architecture.")
    if graph_role not in ("pretrained_train", "deploy"):
        raise ValueError(f"Unsupported recognition graph role: {graph_role}")
    if graph_role == "pretrained_train" and ctc_backbone_grad:
        raise ValueError(
            "Full pretrained training keeps the configured guide detach and NRTR branch."
        )
    architecture["Head"]["out_channels_list"] = rec_out_channels(
        config,
        config_path=config_path,
    )
    model = build_model(architecture)
    if weights_path:
        _load_state_dict(model, weights_path, task="rec", graph_role=graph_role)
    if reparameterize:
        reparameterize_for_deploy(
            model, insert_identity_bn=insert_identity_bn, keep_bn=keep_bn
        )
    if ctc_backbone_grad:
        encoder = model.head.ctc_encoder.encoder
        if not hasattr(encoder, "use_guide"):
            raise ValueError("Recognition CTC encoder does not expose use_guide.")
        encoder.use_guide = False
    if graph_role == "pretrained_train":
        model.graph_role = graph_role
        return model
    wrapped = RecCTCWrapper(model).set_qat_capture_mode()
    wrapped.graph_role = graph_role
    return wrapped


def build_task_model(
    task,
    config_path,
    weights_path=None,
    reparameterize=True,
    det_graph="training",
    rec_ctc_backbone_grad=False,
    rec_graph="deploy",
    insert_identity_bn=False,
    keep_bn=False,
):
    if task == "det":
        return build_det_model(
            config_path,
            weights_path=weights_path,
            reparameterize=reparameterize,
            graph_mode=det_graph,
            insert_identity_bn=insert_identity_bn,
            keep_bn=keep_bn,
        )
    if task == "rec":
        return build_rec_model(
            config_path,
            weights_path=weights_path,
            reparameterize=reparameterize,
            ctc_backbone_grad=rec_ctc_backbone_grad,
            graph_role=rec_graph,
            insert_identity_bn=insert_identity_bn,
            keep_bn=keep_bn,
        )
    raise ValueError(f"Unsupported OCR task: {task}")

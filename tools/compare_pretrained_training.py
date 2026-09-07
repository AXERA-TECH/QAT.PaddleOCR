import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = Path(
    os.environ.get(
        "PPOCR_RUNTIME_DIR",
        Path(tempfile.gettempdir()) / f"ppocr_qat_{os.getpid()}",
    )
).resolve()
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(RUNTIME_DIR)
os.environ.setdefault("TMPDIR", str(RUNTIME_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_DIR / "xdg_cache"))
os.environ.setdefault("PADDLE_HOME", str(RUNTIME_DIR / "paddle"))
os.environ.setdefault("PADDLE_EXTENSION_DIR", str(RUNTIME_DIR / "paddle_extensions"))

import numpy as np
import paddle
import torch


PADDLEOCR_ROOT = ROOT_DIR / "references/PaddleOCR"
for source_root in (ROOT_DIR, PADDLEOCR_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from ppocr.losses import build_loss as build_paddle_loss
from ppocr.modeling.architectures import build_model as build_paddle_model
from pytorchocr.diagnostics import compare_gradient_maps, tensor_difference
from pytorchocr.training import (
    MultiLoss,
    build_optimizer,
    build_rec_model,
    load_ocr_config,
    rec_out_channels,
)
from pytorchocr.utils.hashing import file_sha256


GRADIENT_OWNERS = (
    "backbone.",
    "head.ctc_encoder.",
    "head.ctc_head.",
    "head.before_gtc.",
    "head.gtc_head.",
)
LINEAR_TRANSPOSE_SUFFIXES = (
    "fc1.weight",
    "fc2.weight",
    "fc.weight",
    "qkv.weight",
    "proj.weight",
    "out_proj.weight",
    "q.weight",
    "kv.weight",
    "tgt_word_prj.weight",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Paddle/PyTorch PP-OCR full recognition outputs, losses, "
            "gradients, and PyTorch deployment projection."
        )
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--paddle-weights", required=True)
    parser.add_argument("--torch-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-shape", nargs=3, type=int, default=[3, 48, 320])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eval_bn", "train_bn"),
        default=("eval_bn", "train_bn"),
    )
    return parser.parse_args()


def git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT_DIR,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def array_hash(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_batch(batch_size, image_shape, seed):
    if batch_size < 2:
        raise ValueError("Full-training parity requires batch_size >= 2 for BN.")
    rng = np.random.default_rng(seed)
    images = rng.uniform(-1.0, 1.0, [batch_size, *image_shape]).astype(np.float32)
    ctc = np.zeros([batch_size, 25], dtype=np.int64)
    gtc = np.zeros([batch_size, 25], dtype=np.int64)
    lengths = np.empty([batch_size], dtype=np.int64)
    valid_ratio = np.linspace(1.0, 0.75, batch_size, dtype=np.float32)
    for index in range(batch_size):
        length = 3 if index % 2 == 0 else 2
        labels = np.arange(1 + index, 1 + index + length, dtype=np.int64)
        ctc[index, :length] = labels
        gtc[index, : length + 2] = np.concatenate(
            [np.asarray([2]), labels + 3, np.asarray([3])]
        )
        lengths[index] = length
    targets = {
        "targets": ctc,
        "gtc_targets": gtc,
        "target_lengths": lengths,
        "valid_ratio": valid_ratio,
    }
    return images, targets


def paddle_to_torch_name(name):
    return name.replace("._mean", ".running_mean").replace(
        "._variance", ".running_var"
    )


def selected_parameter(name):
    return any(name.startswith(owner) for owner in GRADIENT_OWNERS)


def configure_paddle_mode(model, mode):
    if mode == "train_bn":
        model.train()
    else:
        model.eval()
        # Select MultiHead's training outputs without recursively moving the
        # CTC encoder or backbone BN layers back to training mode.
        model.head.training = True
        model.head.ctc_head.training = True
        model.head.gtc_head.training = True
    dropout_count = 0
    for layer in model.sublayers():
        if isinstance(layer, paddle.nn.Dropout):
            layer.eval()
            dropout_count += 1
    return dropout_count


def configure_torch_mode(model, mode):
    if mode == "train_bn":
        model.train()
    else:
        model.eval()
        model.head.training = True
        model.head.ctc_head.training = True
        model.head.gtc_head.training = True
    dropout_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
            dropout_count += 1
    return dropout_count


def build_paddle(config, config_path, weights_path):
    architecture = copy.deepcopy(config["Architecture"])
    architecture["Head"]["out_channels_list"] = rec_out_channels(
        config,
        config_path=config_path,
    )
    model = build_paddle_model(architecture)
    model.set_state_dict(paddle.load(str(weights_path)))
    return model


def paddle_tensor(value):
    return value.detach().cpu().numpy().copy()


def paddle_optimizer(model, config):
    optimizer_config = config["Optimizer"]
    regularizer = optimizer_config.get("regularizer", {})
    return paddle.optimizer.Adam(
        learning_rate=float(optimizer_config["lr"]["learning_rate"]),
        beta1=float(optimizer_config.get("beta1", 0.9)),
        beta2=float(optimizer_config.get("beta2", 0.999)),
        epsilon=1.0e-8,
        weight_decay=float(regularizer.get("factor", 0.0)),
        parameters=[parameter for parameter in model.parameters() if parameter.trainable],
    )


def run_paddle(config, config_path, weights_path, images, targets, mode):
    model = build_paddle(config, config_path, weights_path)
    dropout_count = configure_paddle_mode(model, mode)
    batch = [
        paddle.to_tensor(images),
        paddle.to_tensor(targets["targets"]),
        paddle.to_tensor(targets["gtc_targets"]),
        paddle.to_tensor(targets["target_lengths"]),
        paddle.to_tensor(targets["valid_ratio"]),
    ]
    model.clear_gradients()
    outputs = model(batch[0], data=batch[1:])
    if not isinstance(outputs, dict) or not {"ctc", "ctc_neck", "gtc"}.issubset(
        outputs
    ):
        raise RuntimeError(f"Unexpected Paddle outputs: {type(outputs)}")
    criterion = build_paddle_loss(config["Loss"])
    losses = criterion(outputs, batch)
    losses["loss"].backward()

    gradients = {}
    optimizer_before = {}
    gradient_none = []
    transpose_reference = []
    for source_name, parameter in model.named_parameters():
        target_name = paddle_to_torch_name(source_name)
        if parameter.stop_gradient or not selected_parameter(target_name):
            continue
        if parameter.grad is None:
            gradient_none.append(target_name)
            continue
        gradients[target_name] = paddle_tensor(parameter.grad)
        if mode == "train_bn":
            optimizer_before[target_name] = paddle_tensor(parameter)
        if source_name.endswith(LINEAR_TRANSPOSE_SUFFIXES):
            transpose_reference.append(target_name)
    optimizer_delta = None
    if mode == "train_bn":
        optimizer = paddle_optimizer(model, config)
        optimizer.step()
        optimizer_delta = {
            paddle_to_torch_name(name): paddle_tensor(parameter)
            - optimizer_before[paddle_to_torch_name(name)]
            for name, parameter in model.named_parameters()
            if paddle_to_torch_name(name) in optimizer_before
        }
    result = {
        "outputs": {
            name: paddle_tensor(outputs[name])
            for name in ("ctc", "ctc_neck", "gtc")
        },
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "gradients": gradients,
        "optimizer_delta": optimizer_delta,
        "gradient_none": sorted(gradient_none),
        "transpose_reference": sorted(transpose_reference),
        "dropout_modules_disabled": dropout_count,
    }
    del losses, outputs, criterion, batch, model
    gc.collect()
    if paddle.device.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()
    return result


def torch_tensor(value):
    return value.detach().cpu().numpy().copy()


def run_torch(config, config_path, weights_path, images, targets, mode, device):
    model = build_rec_model(
        config_path,
        weights_path=weights_path,
        reparameterize=False,
        graph_role="pretrained_train",
    ).to(device)
    dropout_count = configure_torch_mode(model, mode)
    images_tensor = torch.from_numpy(images).to(device)
    torch_targets = {
        name: torch.from_numpy(value).to(device) for name, value in targets.items()
    }
    data = [
        torch_targets["targets"],
        torch_targets["gtc_targets"],
        torch_targets["target_lengths"],
        torch_targets["valid_ratio"],
    ]
    model.zero_grad(set_to_none=True)
    outputs = model(images_tensor, data=data)
    if not isinstance(outputs, dict) or not {"ctc", "ctc_neck", "gtc"}.issubset(
        outputs
    ):
        raise RuntimeError(f"Unexpected PyTorch outputs: {type(outputs)}")
    loss_config = config.get("Loss", {})
    criterion = MultiLoss(
        weight_1=float(loss_config.get("weight_1", 1.0)),
        weight_2=float(loss_config.get("weight_2", 1.0)),
    ).to(device)
    losses = criterion(outputs, torch_targets)
    losses["loss"].backward()

    gradients = {}
    optimizer_before = {}
    gradient_none = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not selected_parameter(name):
            continue
        if parameter.grad is None:
            gradient_none.append(name)
            continue
        gradients[name] = torch_tensor(parameter.grad)
        if mode == "train_bn":
            optimizer_before[name] = torch_tensor(parameter)
    optimizer_delta = None
    if mode == "train_bn":
        optimizer_config = config["Optimizer"]
        regularizer = optimizer_config.get("regularizer", {})
        optimizer = build_optimizer(
            model,
            config,
            learning_rate=float(optimizer_config["lr"]["learning_rate"]),
            weight_decay=float(regularizer.get("factor", 0.0)),
        )
        optimizer.step()
        optimizer_delta = {
            name: torch_tensor(parameter) - optimizer_before[name]
            for name, parameter in model.named_parameters()
            if name in optimizer_before
        }
    result = {
        "outputs": {
            name: torch_tensor(outputs[name])
            for name in ("ctc", "ctc_neck", "gtc")
        },
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "gradients": gradients,
        "optimizer_delta": optimizer_delta,
        "gradient_none": sorted(gradient_none),
        "dropout_modules_disabled": dropout_count,
    }
    del losses, outputs, criterion, data, torch_targets, images_tensor, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compare_mode(paddle_result, torch_result):
    outputs = {
        name: tensor_difference(
            paddle_result["outputs"][name],
            torch_result["outputs"][name],
        )
        for name in ("ctc", "ctc_neck", "gtc")
    }
    for name in ("ctc", "gtc"):
        paddle_logits = paddle_result["outputs"][name]
        torch_logits = torch_result["outputs"][name]
        paddle_prob = torch.softmax(torch.from_numpy(paddle_logits), dim=-1).numpy()
        torch_prob = torch.softmax(torch.from_numpy(torch_logits), dim=-1).numpy()
        outputs[name]["probability"] = tensor_difference(paddle_prob, torch_prob)
        outputs[name]["argmax_agreement"] = float(
            np.mean(np.argmax(paddle_logits, axis=-1) == np.argmax(torch_logits, axis=-1))
        )
    losses = {}
    for name in ("CTCLoss", "NRTRLoss", "loss"):
        reference = paddle_result["losses"][name]
        candidate = torch_result["losses"][name]
        losses[name] = {
            "paddle": reference,
            "torch": candidate,
            "abs": abs(reference - candidate),
            "relative": abs(reference - candidate) / max(abs(reference), 1.0e-30),
        }
    gradients = compare_gradient_maps(
        paddle_result["gradients"],
        torch_result["gradients"],
        owners=GRADIENT_OWNERS,
        transpose_reference=paddle_result["transpose_reference"],
    )
    gradients["paddle_none"] = paddle_result["gradient_none"]
    gradients["torch_none"] = torch_result["gradient_none"]
    optimizer_delta = {"status": "not_run"}
    if paddle_result["optimizer_delta"] is not None:
        optimizer_delta = {
            "status": "compared",
            **compare_gradient_maps(
                paddle_result["optimizer_delta"],
                torch_result["optimizer_delta"],
                owners=GRADIENT_OWNERS,
                transpose_reference=paddle_result["transpose_reference"],
            ),
        }
    return {
        "dropout_modules_disabled": {
            "paddle": paddle_result["dropout_modules_disabled"],
            "torch": torch_result["dropout_modules_disabled"],
        },
        "outputs": outputs,
        "losses": losses,
        "gradients": gradients,
        "optimizer_delta": optimizer_delta,
    }


def deployment_projection(config_path, weights_path, images, device):
    images_tensor = torch.from_numpy(images).to(device)
    full = build_rec_model(
        config_path,
        weights_path=weights_path,
        reparameterize=False,
        graph_role="pretrained_train",
    ).to(device)
    full.set_validation_mode()
    with torch.no_grad():
        full_output = torch_tensor(full(images_tensor))
    del full
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    deploy = build_rec_model(
        config_path,
        weights_path=weights_path,
        reparameterize=False,
        graph_role="deploy",
    ).to(device)
    deploy.set_validation_mode()
    with torch.no_grad():
        deploy_output = torch_tensor(deploy(images_tensor))
    del deploy, images_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tensor_difference(full_output, deploy_output)


def main(args):
    started = time.time()
    config_path = Path(args.model_config).resolve()
    paddle_weights = Path(args.paddle_weights).resolve()
    torch_weights = Path(args.torch_weights).resolve()
    output_path = Path(args.output).resolve()
    config = load_ocr_config(config_path)
    architecture = config.get("Architecture", {})
    if (
        architecture.get("model_type") != "rec"
        or architecture.get("Head", {}).get("name") != "MultiHead"
    ):
        raise ValueError("Full-training parity requires a MultiHead recognition model.")
    paddle_device = args.device.replace("cuda", "gpu")
    paddle.set_device(paddle_device)
    torch_device = torch.device(args.device)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    torch.manual_seed(args.seed)
    images, targets = make_batch(args.batch_size, args.image_shape, args.seed)

    mode_results = {}
    for mode in args.modes:
        paddle_result = run_paddle(
            config,
            config_path,
            paddle_weights,
            images,
            targets,
            mode,
        )
        torch_result = run_torch(
            config,
            config_path,
            torch_weights,
            images,
            targets,
            mode,
            torch_device,
        )
        mode_results[mode] = compare_mode(paddle_result, torch_result)
        del paddle_result, torch_result
        gc.collect()

    result = {
        "schema_version": 1,
        "model_name": config["Global"]["model_name"],
        "git": git_state(),
        "environment": {
            "python": platform.python_version(),
            "paddle": paddle.__version__,
            "torch": torch.__version__,
            "device": args.device,
        },
        "contract": {
            "model_config": str(config_path),
            "model_config_sha256": file_sha256(config_path),
            "paddle_weights": str(paddle_weights),
            "paddle_weights_sha256": file_sha256(paddle_weights),
            "torch_weights": str(torch_weights),
            "torch_weights_sha256": file_sha256(torch_weights),
            "image_shape": [args.batch_size, *args.image_shape],
            "seed": args.seed,
            "images_sha256": array_hash({"images": images}),
            "targets_sha256": array_hash(targets),
            "dropout_policy": "disabled_in_both_frameworks",
        },
        "modes": mode_results,
        "deployment_projection": deployment_projection(
            config_path,
            torch_weights,
            images,
            torch_device,
        ),
        "elapsed_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())

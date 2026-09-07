import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import paddle
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
PADDLEOCR_ROOT = ROOT_DIR / "references/PaddleOCR"
for source_root in (ROOT_DIR, PADDLEOCR_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from ppocr.losses import build_loss as build_paddle_loss
from ppocr.modeling.architectures import build_model as build_paddle_model
from pytorchocr.training import (
    build_criterion,
    build_det_model,
    build_optimizer,
    file_sha256,
    load_ocr_config,
)


OWNER_PREFIXES = {
    "backbone": "backbone.",
    "neck": "neck.",
    "binarize": "head.binarize.",
    "threshold": "head.thresh.",
}
AUX_OWNER_PREFIXES = {
    f"aux_{kind}_{level}": f"head.aux_{kind}_{level}."
    for level in ("p4", "p3", "p2")
    for kind in ("binarize", "thresh")
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Paddle/PyTorch DB training maps, loss, and gradients."
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--paddle-weights", required=True)
    parser.add_argument("--torch-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-shape", nargs=3, type=int, default=[3, 64, 64])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eval_bn", "train_bn"),
        default=["eval_bn", "train_bn"],
    )
    return parser.parse_args()


def array_hash(values):
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def difference(reference, actual):
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {actual.shape}")
    delta = np.abs(reference - actual)
    reference_norm = np.linalg.norm(reference.reshape(-1))
    return {
        "shape": list(reference.shape),
        "mae": float(delta.mean()),
        "max_abs": float(delta.max(initial=0.0)),
        "relative_l2": float(
            np.linalg.norm((reference - actual).reshape(-1))
            / max(reference_norm, 1.0e-12)
        ),
    }


def vector_difference(reference, actual):
    report = difference(reference, actual)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(reference) * np.linalg.norm(actual)
    report["cosine"] = float(
        np.dot(reference, actual) / max(denominator, 1.0e-30)
    )
    return report


def make_inputs(args):
    rng = np.random.default_rng(args.seed)
    images = rng.standard_normal(
        [args.batch_size, *args.image_shape], dtype=np.float32
    )
    height, width = args.image_shape[1:]
    shrink_map = np.zeros((args.batch_size, height, width), dtype=np.float32)
    shrink_map[:, height // 4 : 3 * height // 4, width // 5 : 4 * width // 5] = 1
    threshold_map = rng.uniform(
        0.3, 0.7, size=(args.batch_size, height, width)
    ).astype(np.float32)
    shrink_mask = np.ones_like(shrink_map)
    threshold_mask = np.ones_like(shrink_map)
    shrink_mask[:, :2] = 0
    threshold_mask[:, :, -2:] = 0
    targets = {
        "threshold_map": threshold_map,
        "threshold_mask": threshold_mask,
        "shrink_map": shrink_map,
        "shrink_mask": shrink_mask,
    }
    return images, targets


def build_models(config, args):
    paddle_model = build_paddle_model(copy.deepcopy(config["Architecture"]))
    paddle_model.set_state_dict(paddle.load(args.paddle_weights))
    torch_model = build_det_model(
        args.model_config,
        weights_path=args.torch_weights,
        reparameterize=False,
        graph_mode="pretrained_train",
    )
    return paddle_model, torch_model


def set_mode(paddle_model, torch_model, mode):
    if mode == "train_bn":
        paddle_model.train()
        torch_model.train()
        return
    paddle_model.eval()
    paddle_model.head.training = True
    if getattr(paddle_model.head, "aux_in_channels", 0) > 0:
        paddle_model.neck.training = True
    torch_model.set_validation_mode()


def paddle_targets(images, targets):
    return [
        paddle.to_tensor(images),
        paddle.to_tensor(targets["threshold_map"]),
        paddle.to_tensor(targets["threshold_mask"]),
        paddle.to_tensor(targets["shrink_map"]),
        paddle.to_tensor(targets["shrink_mask"]),
    ]


def torch_targets(targets):
    return {name: torch.from_numpy(value) for name, value in targets.items()}


def comparable_owner_prefixes(torch_model):
    parameter_names = tuple(name for name, _ in torch_model.named_parameters())
    return {
        owner: prefix
        for owner, prefix in {**OWNER_PREFIXES, **AUX_OWNER_PREFIXES}.items()
        if any(name.startswith(prefix) for name in parameter_names)
    }


def gradient_groups(paddle_model, torch_model):
    paddle_parameters = dict(paddle_model.named_parameters())
    torch_parameters = dict(torch_model.named_parameters())
    groups = {}
    for owner, prefix in comparable_owner_prefixes(torch_model).items():
        paddle_values = []
        torch_values = []
        names = []
        for name, torch_parameter in torch_parameters.items():
            if not name.startswith(prefix) or name not in paddle_parameters:
                continue
            paddle_gradient = paddle_parameters[name].grad
            torch_gradient = torch_parameter.grad
            if paddle_gradient is None or torch_gradient is None:
                continue
            paddle_array = paddle_gradient.numpy()
            torch_array = torch_gradient.detach().cpu().numpy()
            if paddle_array.shape != torch_array.shape:
                continue
            paddle_values.append(paddle_array.reshape(-1))
            torch_values.append(torch_array.reshape(-1))
            names.append(name)
        if not names:
            raise RuntimeError(f"No comparable gradients for {owner}.")
        groups[owner] = {
            "parameter_tensors": len(names),
            "elements": int(sum(value.size for value in paddle_values)),
            **vector_difference(
                np.concatenate(paddle_values),
                np.concatenate(torch_values),
            ),
        }
    return groups


def optimizer_step_report(paddle_model, torch_model, config):
    optimizer_config = config["Optimizer"]
    lr_config = optimizer_config["lr"]
    regularizer_config = optimizer_config.get("regularizer", {})
    paddle_optimizer = paddle.optimizer.Adam(
        learning_rate=float(lr_config["learning_rate"]),
        beta1=float(optimizer_config.get("beta1", 0.9)),
        beta2=float(optimizer_config.get("beta2", 0.999)),
        weight_decay=paddle.regularizer.L2Decay(
            float(regularizer_config.get("factor", 0.0))
        ),
        parameters=[
            parameter
            for parameter in paddle_model.parameters()
            if not parameter.stop_gradient
        ],
    )
    torch_optimizer = build_optimizer(torch_model, config)
    paddle_before = {
        name: parameter.numpy().copy()
        for name, parameter in paddle_model.named_parameters()
        if not parameter.stop_gradient
    }
    torch_before = {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in torch_model.named_parameters()
    }
    paddle_gradients = {
        name: parameter.grad.numpy().copy()
        for name, parameter in paddle_model.named_parameters()
        if parameter.grad is not None
    }
    torch_gradients = {
        name: parameter.grad.detach().cpu().numpy().copy()
        for name, parameter in torch_model.named_parameters()
        if parameter.grad is not None
    }
    paddle_optimizer.step()
    torch_optimizer.step()

    paddle_parameters = dict(paddle_model.named_parameters())
    torch_parameters = dict(torch_model.named_parameters())
    reports = {}
    for owner, prefix in {
        "all": "",
        **comparable_owner_prefixes(torch_model),
    }.items():
        paddle_deltas = []
        torch_deltas = []
        names = []
        for name, torch_parameter in torch_parameters.items():
            if not name.startswith(prefix) or name not in paddle_before:
                continue
            if name not in paddle_parameters or name not in torch_before:
                continue
            paddle_delta = paddle_parameters[name].numpy() - paddle_before[name]
            torch_delta = (
                torch_parameter.detach().cpu().numpy() - torch_before[name]
            )
            if paddle_delta.shape != torch_delta.shape:
                continue
            paddle_deltas.append(paddle_delta.reshape(-1))
            torch_deltas.append(torch_delta.reshape(-1))
            names.append(name)
        if not names:
            raise RuntimeError(f"No comparable optimizer deltas for {owner}.")
        paddle_vector = np.concatenate(paddle_deltas)
        torch_vector = np.concatenate(torch_deltas)
        report = vector_difference(paddle_vector, torch_vector)
        report.update(
            {
                "parameter_tensors": len(names),
                "elements": int(paddle_vector.size),
                "direction_disagreement_fraction": float(
                    np.mean(np.signbit(paddle_vector) != np.signbit(torch_vector))
                ),
            }
        )
        reports[owner] = report
    parameter_to_group = {
        id(parameter): group
        for group in torch_optimizer.param_groups
        for parameter in group["params"]
    }
    parameter_reports = []
    stability_values = []
    for name, torch_parameter in torch_parameters.items():
        if (
            name not in paddle_before
            or name not in paddle_parameters
            or name not in torch_before
        ):
            continue
        paddle_delta = paddle_parameters[name].numpy() - paddle_before[name]
        torch_delta = torch_parameter.detach().cpu().numpy() - torch_before[name]
        if paddle_delta.shape != torch_delta.shape:
            continue
        delta_report = vector_difference(paddle_delta, torch_delta)
        paddle_gradient = paddle_gradients.get(name)
        torch_gradient = torch_gradients.get(name)
        optimizer_group = parameter_to_group[id(torch_parameter)]
        if paddle_gradient is not None and torch_gradient is not None:
            weight_decay = float(optimizer_group["weight_decay"])
            stability_values.append(
                (
                    paddle_gradient + weight_decay * paddle_before[name],
                    torch_gradient + weight_decay * torch_before[name],
                    paddle_delta,
                    torch_delta,
                )
            )
        parameter_reports.append(
            {
                "name": name,
                "optimizer_group": optimizer_group.get("group_name", "default"),
                "weight_norm": float(np.linalg.norm(torch_before[name].reshape(-1))),
                "paddle_gradient_present": paddle_gradient is not None,
                "torch_gradient_present": torch_gradient is not None,
                "paddle_gradient_norm": (
                    float(np.linalg.norm(paddle_gradient.reshape(-1)))
                    if paddle_gradient is not None
                    else None
                ),
                "torch_gradient_norm": (
                    float(np.linalg.norm(torch_gradient.reshape(-1)))
                    if torch_gradient is not None
                    else None
                ),
                "paddle_delta_norm": float(np.linalg.norm(paddle_delta.reshape(-1))),
                "torch_delta_norm": float(np.linalg.norm(torch_delta.reshape(-1))),
                "direction_disagreement_fraction": float(
                    np.mean(np.signbit(paddle_delta) != np.signbit(torch_delta))
                ),
                **delta_report,
            }
        )
    reports["largest_parameter_differences"] = sorted(
        parameter_reports,
        key=lambda report: (report["max_abs"], report["relative_l2"]),
        reverse=True,
    )[:25]
    reports["largest_parameter_mae"] = sorted(
        parameter_reports,
        key=lambda report: (report["mae"], report["max_abs"]),
        reverse=True,
    )[:25]
    reports["largest_parameter_direction_disagreement"] = sorted(
        parameter_reports,
        key=lambda report: (
            report["direction_disagreement_fraction"],
            report["mae"],
        ),
        reverse=True,
    )[:25]
    paddle_effective_gradient = np.concatenate(
        [values[0].reshape(-1) for values in stability_values]
    )
    torch_effective_gradient = np.concatenate(
        [values[1].reshape(-1) for values in stability_values]
    )
    paddle_delta = np.concatenate(
        [values[2].reshape(-1) for values in stability_values]
    )
    torch_delta = np.concatenate(
        [values[3].reshape(-1) for values in stability_values]
    )
    stable_reports = {}
    for threshold in (0.0, 1.0e-10, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4):
        stable = np.minimum(
            np.abs(paddle_effective_gradient),
            np.abs(torch_effective_gradient),
        ) >= threshold
        if not np.any(stable):
            continue
        stable_reports[f"min_abs_effective_gradient_{threshold:.0e}"] = {
            "elements": int(np.sum(stable)),
            "fraction": float(np.mean(stable)),
            "effective_gradient_direction_disagreement_fraction": float(
                np.mean(
                    np.signbit(paddle_effective_gradient[stable])
                    != np.signbit(torch_effective_gradient[stable])
                )
            ),
            "delta_direction_disagreement_fraction": float(
                np.mean(
                    np.signbit(paddle_delta[stable])
                    != np.signbit(torch_delta[stable])
                )
            ),
            "delta": vector_difference(
                paddle_delta[stable],
                torch_delta[stable],
            ),
        }
    reports["effective_gradient_stability"] = stable_reports
    reports["torch_param_groups"] = [
        {
            "name": group.get("group_name", "default"),
            "parameters": len(group["params"]),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in torch_optimizer.param_groups
    ]
    return reports


def deployment_projection(args, images):
    images_tensor = torch.from_numpy(images)
    full_model = build_det_model(
        args.model_config,
        weights_path=args.torch_weights,
        reparameterize=False,
        graph_mode="pretrained_train",
    )
    full_model.set_validation_mode()
    deploy_model = build_det_model(
        args.model_config,
        weights_path=args.torch_weights,
        reparameterize=False,
        graph_mode="inference",
    )
    deploy_model.eval()
    with torch.no_grad():
        full_maps = full_model(images_tensor)["maps"]
        full_shrink = full_maps[:, :1]
        deploy_shrink = deploy_model(images_tensor)
    return {
        "full_maps_shape": list(full_maps.shape),
        "full_shrink_shape": list(full_shrink.shape),
        "deploy_shrink_shape": list(deploy_shrink.shape),
        "full_shrink_to_deploy": difference(
            full_shrink.numpy(),
            deploy_shrink.numpy(),
        ),
    }


def run_mode(config, args, images, targets, mode):
    paddle_model, torch_model = build_models(config, args)
    set_mode(paddle_model, torch_model, mode)
    paddle_criterion = build_paddle_loss(copy.deepcopy(config["Loss"]))
    torch_criterion = build_criterion("det", config)

    paddle_images = paddle.to_tensor(images)
    torch_images = torch.from_numpy(images)
    paddle_output = paddle_model(paddle_images)
    torch_output = torch_model(torch_images)
    paddle_losses = paddle_criterion(
        paddle_output,
        paddle_targets(images, targets),
    )
    torch_losses = torch_criterion(torch_output, torch_targets(targets))
    paddle_losses["loss"].backward()
    torch_losses["loss"].backward()

    common_map_names = sorted(
        name
        for name in set(paddle_output).intersection(torch_output)
        if name == "maps" or name.startswith("aux_maps_")
    )
    map_reports = {}
    for name in common_map_names:
        paddle_maps = paddle_output[name].numpy()
        torch_maps = torch_output[name].detach().cpu().numpy()
        map_reports[name] = {
            "all": difference(paddle_maps, torch_maps),
            "shrink": difference(paddle_maps[:, 0], torch_maps[:, 0]),
            "threshold": difference(paddle_maps[:, 1], torch_maps[:, 1]),
            "binary": difference(paddle_maps[:, 2], torch_maps[:, 2]),
        }
    common_losses = sorted(set(paddle_losses).intersection(torch_losses))
    result = {
        "maps": map_reports.pop("maps"),
        "losses": {
            name: {
                "paddle": float(paddle_losses[name]),
                "torch": float(torch_losses[name].detach()),
                "abs": abs(
                    float(paddle_losses[name])
                    - float(torch_losses[name].detach())
                ),
            }
            for name in common_losses
        },
        "gradients": gradient_groups(paddle_model, torch_model),
    }
    if map_reports:
        result["auxiliary_maps"] = map_reports
    if mode == "train_bn":
        result["adam_step"] = optimizer_step_report(
            paddle_model,
            torch_model,
            config,
        )
    return result


def main(args):
    if args.batch_size <= 0 or len(args.image_shape) != 3:
        raise ValueError("Invalid batch size or image shape.")
    paddle.set_device("cpu")
    paddle.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_ocr_config(args.model_config)
    images, targets = make_inputs(args)
    result = {
        "task": "det",
        "model_name": config["Global"].get("model_name"),
        "device": "cpu",
        "batch_size": args.batch_size,
        "image_shape": list(args.image_shape),
        "seed": args.seed,
        "model_config": str(Path(args.model_config).resolve()),
        "model_config_sha256": file_sha256(args.model_config),
        "paddle_weights": str(Path(args.paddle_weights).resolve()),
        "paddle_weights_sha256": file_sha256(args.paddle_weights),
        "torch_weights": str(Path(args.torch_weights).resolve()),
        "torch_weights_sha256": file_sha256(args.torch_weights),
        "inputs_sha256": array_hash([images]),
        "targets_sha256": array_hash(list(targets.values())),
        "deployment_projection": deployment_projection(args, images),
        "modes": {
            mode: run_mode(config, args, images, targets, mode)
            for mode in args.modes
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())

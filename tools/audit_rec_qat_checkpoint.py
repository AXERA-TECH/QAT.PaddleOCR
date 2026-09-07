import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.diagnostics import (
    build_diagnostic_dataset,
    build_prepared_qat_checkpoint,
    evaluate_recognition_qat,
    load_qat_checkpoint,
)
from pytorchocr.training import load_ocr_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit trained recognition QAT checkpoints with fake quant off/on."
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-file")
    parser.add_argument("--data-dir")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _state_drift(prepared, initial_state, checkpoint_state):
    parameter_names = {name for name, _ in prepared.named_parameters()}
    buffer_names = {
        name
        for name, _ in prepared.named_buffers()
        if "activation_post_process" not in name
    }

    def records(names):
        result = []
        for name in names:
            if name not in initial_state or name not in checkpoint_state:
                continue
            reference = initial_state[name].detach().to(torch.float64)
            actual = checkpoint_state[name].detach().to(torch.float64)
            if reference.shape != actual.shape:
                continue
            difference = actual - reference
            reference_norm = float(torch.linalg.vector_norm(reference))
            difference_norm = float(torch.linalg.vector_norm(difference))
            result.append(
                {
                    "name": name,
                    "max_abs": float(difference.abs().max()),
                    "relative_l2": difference_norm / max(reference_norm, 1.0e-12),
                }
            )
        return sorted(result, key=lambda item: item["relative_l2"], reverse=True)

    parameter_records = records(parameter_names)
    buffer_records = records(buffer_names)
    return {
        "parameters_compared": len(parameter_records),
        "changed_parameters": sum(item["max_abs"] != 0.0 for item in parameter_records),
        "top_parameter_drift": parameter_records[:20],
        "buffers_compared": len(buffer_records),
        "changed_buffers": sum(item["max_abs"] != 0.0 for item in buffer_records),
        "top_buffer_drift": buffer_records[:20],
    }


def _observer_summary(model):
    scales = []
    zero_points = []
    for module in model.modules():
        scale = getattr(module, "scale", None)
        zero_point = getattr(module, "zero_point", None)
        if torch.is_tensor(scale):
            scales.append(scale.detach().to(torch.float64).reshape(-1).cpu())
        if torch.is_tensor(zero_point):
            zero_points.append(zero_point.detach().to(torch.int64).reshape(-1).cpu())
    all_scales = torch.cat(scales) if scales else torch.empty(0)
    all_zero_points = torch.cat(zero_points) if zero_points else torch.empty(0)
    return {
        "scale_values": int(all_scales.numel()),
        "scale_min": float(all_scales.min()) if all_scales.numel() else None,
        "scale_max": float(all_scales.max()) if all_scales.numel() else None,
        "zero_point_values": int(all_zero_points.numel()),
        "zero_point_min": int(all_zero_points.min()) if all_zero_points.numel() else None,
        "zero_point_max": int(all_zero_points.max()) if all_zero_points.numel() else None,
    }


def main(args):
    if args.samples < 0 or args.batch_size <= 0:
        raise ValueError("--samples must be non-negative and --batch-size positive.")
    loaded = [load_qat_checkpoint(path) for path in args.checkpoint]
    checkpoints = [item[0] for item in loaded]
    metadatas = [item[1] for item in loaded]
    metadata = metadatas[0]
    if metadata.get("task") != "rec":
        raise ValueError("Expected a recognition QAT checkpoint.")
    contract = (
        metadata["model_config"],
        metadata["weights"],
        metadata["qat_config"],
        tuple(metadata["image_shape"]),
        bool(metadata.get("reparameterized", True)),
        bool(metadata.get("rec_ctc_backbone_grad", False)),
    )
    for candidate in metadatas[1:]:
        candidate_contract = (
            candidate.get("model_config"),
            candidate.get("weights"),
            candidate.get("qat_config"),
            tuple(candidate.get("image_shape", [])),
            bool(candidate.get("reparameterized", True)),
            bool(candidate.get("rec_ctc_backbone_grad", False)),
        )
        if candidate_contract != contract:
            raise ValueError("Checkpoints do not share one graph contract.")

    label_file = args.label_file or metadata.get("val_label_file")
    data_dir = args.data_dir or metadata.get("val_data_dir")
    dataset = build_diagnostic_dataset("rec", metadata, label_file, data_dir)
    if args.samples:
        dataset = Subset(dataset, range(min(args.samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    prepared, _ = build_prepared_qat_checkpoint(metadata)
    initial_state = copy.deepcopy(prepared.state_dict())
    parameter_names = {name for name, _ in prepared.named_parameters()}
    buffer_names = {
        name
        for name, _ in prepared.named_buffers()
        if "activation_post_process" not in name
    }
    config = load_ocr_config(metadata["model_config"])
    device = torch.device(args.device)
    results = []
    for path, checkpoint in zip(args.checkpoint, checkpoints):
        prepared.load_state_dict(checkpoint["model"], strict=True)
        observer_only_state = {
            name: (
                value
                if "activation_post_process" in name
                else initial_state[name]
            )
            for name, value in checkpoint["model"].items()
        }
        observer_only_model = copy.deepcopy(prepared)
        observer_only_model.load_state_dict(observer_only_state, strict=True)
        checkpoint_parameters_state = {
            name: (
                value
                if name in parameter_names or "activation_post_process" in name
                else initial_state[name]
            )
            for name, value in checkpoint["model"].items()
        }
        checkpoint_parameters_model = copy.deepcopy(prepared)
        checkpoint_parameters_model.load_state_dict(
            checkpoint_parameters_state,
            strict=True,
        )
        checkpoint_buffers_state = {
            name: (
                value
                if name in buffer_names or "activation_post_process" in name
                else initial_state[name]
            )
            for name, value in checkpoint["model"].items()
        }
        checkpoint_buffers_model = copy.deepcopy(prepared)
        checkpoint_buffers_model.load_state_dict(
            checkpoint_buffers_state,
            strict=True,
        )
        checkpoint_result = {
            "checkpoint": str(Path(path).resolve()),
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
            "observers_frozen": bool(checkpoint.get("observers_frozen", False)),
            "observer_qparams": _observer_summary(prepared),
            "state_drift": _state_drift(
                prepared,
                initial_state,
                checkpoint["model"],
            ),
            "fake_off": evaluate_recognition_qat(
                prepared, loader, config, metadata, device, fake_quant=False
            ),
            "fake_on": evaluate_recognition_qat(
                prepared, loader, config, metadata, device, fake_quant=True
            ),
            "initial_model_checkpoint_observers": evaluate_recognition_qat(
                observer_only_model,
                loader,
                config,
                metadata,
                device,
                fake_quant=True,
            ),
            "checkpoint_parameters_initial_buffers": evaluate_recognition_qat(
                checkpoint_parameters_model,
                loader,
                config,
                metadata,
                device,
                fake_quant=True,
            ),
            "initial_parameters_checkpoint_buffers": evaluate_recognition_qat(
                checkpoint_buffers_model,
                loader,
                config,
                metadata,
                device,
                fake_quant=True,
            ),
        }
        parameter_groups = {
            "lab_parameters_only": lambda name: ".lab." in name,
            "backbone_parameters_only": lambda name: name.startswith(
                "model.backbone."
            ),
            "ctc_parameters_only": lambda name: name.startswith(
                ("model.head.ctc_", "model.head.ctc_head.")
            ),
            "gtc_parameters_only": lambda name: name.startswith(
                ("model.head.gtc_head.", "model.head.before_gtc.")
            ),
            "checkpoint_parameters_without_lab": lambda name: ".lab." not in name,
        }
        for result_name, include_parameter in parameter_groups.items():
            variant_state = {
                name: (
                    value
                    if "activation_post_process" in name
                    or (name in parameter_names and include_parameter(name))
                    else initial_state[name]
                )
                for name, value in checkpoint["model"].items()
            }
            variant = copy.deepcopy(prepared)
            variant.load_state_dict(variant_state, strict=True)
            checkpoint_result[result_name] = evaluate_recognition_qat(
                variant,
                loader,
                config,
                metadata,
                device,
                fake_quant=True,
            )
        results.append(checkpoint_result)

    report = {
        "sample_count": len(dataset),
        "batch_size": args.batch_size,
        "device": str(device),
        "checkpoints": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())

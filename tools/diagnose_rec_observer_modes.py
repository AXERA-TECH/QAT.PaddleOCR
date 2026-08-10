import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    enable_fake_quant,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
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
        description="Compare recognition observer updates in PT2E train/eval modes."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-label-file")
    parser.add_argument("--train-data-dir")
    parser.add_argument("--val-label-file")
    parser.add_argument("--val-data-dir")
    parser.add_argument("--update-batches", type=int, default=69)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def _loader(dataset, batch_size, workers, *, samples=None, shuffle=False, seed=0):
    if samples is not None:
        dataset = Subset(dataset, range(min(samples, len(dataset))))
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        generator=generator,
    )


def _run_updates(model, loader, mode, batches, device, fake_quant):
    model.apply(enable_fake_quant if fake_quant else disable_fake_quant)
    if mode == "train":
        move_exported_model_to_train(model)
    else:
        move_exported_model_to_eval(model)
    model.to(device)
    completed = 0
    with torch.no_grad():
        while completed < batches:
            for images, _ in loader:
                outputs = model(images.to(device, non_blocking=True))
                if not torch.isfinite(outputs).all():
                    raise RuntimeError(f"Non-finite output in {mode}-mode update.")
                completed += 1
                if completed >= batches:
                    break
    model.cpu()
    if not fake_quant:
        _refresh_fake_quant_qparams(model)
    return model, completed


def _refresh_fake_quant_qparams(model):
    with torch.no_grad():
        for module in model.modules():
            scale_buffer = getattr(module, "scale", None)
            zero_point_buffer = getattr(module, "zero_point", None)
            calculate_qparams = getattr(module, "calculate_qparams", None)
            if (
                not torch.is_tensor(scale_buffer)
                or not torch.is_tensor(zero_point_buffer)
                or not callable(calculate_qparams)
            ):
                continue
            scale, zero_point = calculate_qparams()
            scale = scale.detach().to(scale_buffer)
            zero_point = zero_point.detach().to(zero_point_buffer)
            scale_buffer.resize_(scale.shape).copy_(scale)
            zero_point_buffer.resize_(zero_point.shape).copy_(zero_point)


def _observer_sites(model):
    sites = {}
    for node in model.graph.nodes:
        if node.op != "call_module" or not str(node.target).startswith(
            "activation_post_process"
        ):
            continue
        producer = node.args[0] if node.args else None
        if not isinstance(producer, torch.fx.Node):
            continue
        module_stack = producer.meta.get("nn_module_stack") or {}
        module_path = None
        if module_stack:
            module_path = list(module_stack.values())[-1][0]
        sites.setdefault(str(node.target), []).append(
            {
                "producer": producer.name,
                "operator": str(producer.target),
                "source_fn_stack": str(producer.meta.get("source_fn_stack")),
                "module_path": module_path,
                "users": [user.name for user in node.users],
                "is_parameter": producer.op == "get_attr",
            }
        )
    return sites


def _scale_drift(reference, current, graph_model):
    sites = _observer_sites(graph_model)
    records = []
    for name, callsites in sites.items():
        if callsites and all(site["is_parameter"] for site in callsites):
            continue
        reference_module = reference.get_submodule(name)
        current_module = current.get_submodule(name)
        reference_scale = getattr(reference_module, "scale", None)
        current_scale = getattr(current_module, "scale", None)
        if not torch.is_tensor(reference_scale) or not torch.is_tensor(current_scale):
            continue
        if reference_scale.numel() != 1 or current_scale.numel() != 1:
            continue
        reference_value = float(reference_scale)
        current_value = float(current_scale)
        ratio = current_value / max(reference_value, torch.finfo(torch.float32).tiny)
        records.append(
            {
                "observer": name,
                "dtype": str(getattr(current_module, "dtype", None)),
                "qmin": getattr(current_module, "quant_min", None),
                "qmax": getattr(current_module, "quant_max", None),
                "reference_scale": reference_value,
                "updated_scale": current_value,
                "scale_ratio": ratio,
                "sites": callsites,
            }
        )
    return sorted(records, key=lambda record: record["scale_ratio"], reverse=True)


def _non_observer_buffer_drift(initial, updated):
    initial_buffers = dict(initial.named_buffers())
    records = []
    for name, value in updated.named_buffers():
        if "activation_post_process" in name or name not in initial_buffers:
            continue
        reference = initial_buffers[name]
        if reference.shape != value.shape:
            continue
        difference = value.detach().to(torch.float64) - reference.detach().to(torch.float64)
        max_abs = float(difference.abs().max()) if difference.numel() else 0.0
        if max_abs:
            records.append({"name": name, "max_abs": max_abs})
    return sorted(records, key=lambda record: record["max_abs"], reverse=True)


def _observer_only(initial, updated):
    state = initial.state_dict()
    updated_state = updated.state_dict()
    merged = {
        name: updated_state[name] if "activation_post_process" in name else value
        for name, value in state.items()
    }
    result = copy.deepcopy(initial)
    result.load_state_dict(merged, strict=True)
    return result


def main(args):
    if args.update_batches <= 0 or args.calibration_samples <= 0:
        raise ValueError("Update batches and calibration samples must be positive.")
    _, metadata = load_qat_checkpoint(args.checkpoint)
    if metadata.get("task") != "rec":
        raise ValueError("Expected a recognition QAT checkpoint.")

    train_label = args.train_label_file or metadata["label_file"]
    train_data = args.train_data_dir or metadata["data_dir"]
    val_label = args.val_label_file or metadata["val_label_file"]
    val_data = args.val_data_dir or metadata["val_data_dir"]
    train_dataset = build_diagnostic_dataset(
        "rec", metadata, train_label, train_data
    )
    val_dataset = build_diagnostic_dataset("rec", metadata, val_label, val_data)
    batch_size = args.batch_size or int(metadata["batch_size"])
    eval_dataset = Subset(val_dataset, range(min(args.eval_samples, len(val_dataset))))
    eval_loader = _loader(
        eval_dataset,
        args.eval_batch_size,
        args.workers,
    )
    config = load_ocr_config(metadata["model_config"])
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    initial, _ = build_prepared_qat_checkpoint(metadata)
    calibration_loader = _loader(
        val_dataset,
        args.calibration_batch_size,
        args.workers,
        samples=args.calibration_samples,
    )

    calibrations = {}
    calibration_metrics = {}
    for fake_quant in (True, False):
        calibration = copy.deepcopy(initial)
        calibration, calibration_batches = _run_updates(
            calibration,
            calibration_loader,
            "eval",
            len(calibration_loader),
            device,
            fake_quant,
        )
        name = "fake_on" if fake_quant else "fake_off"
        calibrations[name] = calibration
        calibration_metrics[name] = evaluate_recognition_qat(
            calibration, eval_loader, config, metadata, device, fake_quant=True
        )

    results = {}
    scenarios = (
        ("train_fake_on", "train", True),
        ("eval_fake_on", "eval", True),
        ("train_fake_off", "train", False),
        ("eval_fake_off", "eval", False),
    )
    for scenario, mode, fake_quant in scenarios:
        torch.manual_seed(args.seed)
        updated = copy.deepcopy(initial)
        update_loader = _loader(
            train_dataset,
            batch_size,
            args.workers,
            shuffle=True,
            seed=args.seed,
        )
        updated, completed = _run_updates(
            updated,
            update_loader,
            mode,
            args.update_batches,
            device,
            fake_quant,
        )
        results[scenario] = {
            "update_batches": completed,
            "update_mode": mode,
            "fake_quant_during_update": fake_quant,
            "fake_on": evaluate_recognition_qat(
                updated, eval_loader, config, metadata, device, fake_quant=True
            ),
            "observer_only_fake_on": evaluate_recognition_qat(
                _observer_only(initial, updated),
                eval_loader,
                config,
                metadata,
                device,
                fake_quant=True,
            ),
            "non_observer_buffer_drift": _non_observer_buffer_drift(
                initial, updated
            ),
            "top_scale_expansion_vs_fake_off_calibration": _scale_drift(
                calibrations["fake_off"], updated, initial
            )[:30],
        }

    report = {
        "checkpoint_contract": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "batch_size": batch_size,
        "calibration_samples": min(args.calibration_samples, len(val_dataset)),
        "calibration_batch_size": args.calibration_batch_size,
        "calibration_batches": calibration_batches,
        "calibration": calibration_metrics,
        "modes": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(parse_args())

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    move_exported_model_to_eval,
)
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.quantization import (
    build_qat_dynamic_shapes,
    convert_prepared_model,
    load_axera_quantizer,
    prepare_qat_model,
)
from pytorchocr.diagnostics import (
    compute_recognition_pair,
    fake_quant_state,
    new_recognition_pair,
    observer_qparams,
    update_recognition_pair,
)
from pytorchocr.training import (
    build_dataset,
    build_task_model,
    build_validation_metric,
    file_sha256,
    load_ocr_config,
)


STAGES = ("prepared_fake_off", "prepared_fake_on", "converted")
PAIRS = (
    ("fake_off_to_fake_on", "prepared_fake_off", "prepared_fake_on"),
    ("fake_on_to_converted", "prepared_fake_on", "converted"),
    ("fake_off_to_converted", "prepared_fake_off", "converted"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PP-OCR recognition prepared fake-quant off/on and "
            "converted PT2E using one frozen observer state."
        )
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--qat-config", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates all samples.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-shape", nargs=3, type=int, default=[3, 48, 320])
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--calibration-batch-size", type=int, default=8)
    parser.add_argument(
        "--calibration-label-file",
        help=(
            "Optional label file for observer calibration, independent of the "
            "evaluation set. Default: calibrate on the first samples of the "
            "evaluation set (legacy behavior — note that this overlaps "
            "calibration with evaluation data)."
        ),
    )
    parser.add_argument(
        "--calibration-data-dir",
        default=None,
        help="Data directory for --calibration-label-file (default: --data-dir).",
    )
    parser.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-bn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the native post-sum BN of v6 RepDWConv during "
            "reparameterization (matches train.py --keep-bn QAT contract)."
        ),
    )
    return parser.parse_args()


def _calibrate(prepared, dataset, args, device):
    sample_count = min(args.calibration_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=args.calibration_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    prepared.to(device)
    move_exported_model_to_eval(prepared)
    with torch.no_grad():
        for images, _ in loader:
            outputs = prepared(images.to(device, non_blocking=True))
            if not torch.isfinite(outputs).all():
                raise RuntimeError("Observer calibration produced non-finite output.")
    return sample_count


def main(args):
    if args.samples < 0 or args.batch_size <= 0:
        raise ValueError("--samples must be non-negative and --batch-size must be positive.")
    if args.calibration_samples <= 0 or args.calibration_batch_size <= 0:
        raise ValueError("Calibration sample and batch counts must be positive.")
    torch.manual_seed(20260805)
    config = load_ocr_config(args.model_config)
    capture_model = build_task_model(
        "rec",
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
        rec_graph="deploy",
        keep_bn=args.keep_bn,
    )
    capture_images = torch.randn(2, *args.image_shape)
    dynamic_shapes = ({0: torch.export.Dim("batch", min=1)},)
    prepared, _ = prepare_qat_model(
        copy.deepcopy(capture_model),
        (capture_images,),
        load_axera_quantizer(args.qat_config),
        dynamic_shapes=dynamic_shapes,
    )
    dataset = build_dataset(
        "rec",
        args.model_config,
        config,
        tuple(args.image_shape),
        args.label_file,
        args.data_dir,
    )
    if args.samples:
        dataset = Subset(dataset, range(min(args.samples, len(dataset))))
    device = torch.device(args.device)
    if args.calibration_label_file is not None:
        calibration_dataset = build_dataset(
            "rec",
            args.model_config,
            config,
            tuple(args.image_shape),
            args.calibration_label_file,
            args.calibration_data_dir or args.data_dir,
        )
    else:
        calibration_dataset = dataset
    calibration_samples = _calibrate(prepared, calibration_dataset, args, device)
    qparams = observer_qparams(prepared)
    if qparams["modules"] == 0 or qparams["nonfinite"]:
        raise RuntimeError(f"Observer qparams are incomplete: {qparams['nonfinite'][:10]}")

    prepared.cpu()
    prepared_fake_off = copy.deepcopy(prepared)
    prepared_fake_on = prepared
    for model in (prepared_fake_off, prepared_fake_on):
        model.apply(disable_observer)
        move_exported_model_to_eval(model)
    prepared_fake_off.apply(disable_fake_quant)
    prepared_fake_on.apply(enable_fake_quant)
    converted = convert_prepared_model(prepared_fake_on)
    states = {
        "prepared_fake_off": fake_quant_state(prepared_fake_off),
        "prepared_fake_on": fake_quant_state(prepared_fake_on),
    }
    if states["prepared_fake_off"]["observer_enabled"]:
        raise RuntimeError("Observer remains enabled in fake-off model.")
    if states["prepared_fake_off"]["fake_quant_enabled"]:
        raise RuntimeError("Fake quant remains enabled in fake-off model.")
    if states["prepared_fake_on"]["observer_enabled"]:
        raise RuntimeError("Observer remains enabled in fake-on model.")
    if states["prepared_fake_on"]["fake_quant_enabled"] != qparams["modules"]:
        raise RuntimeError("Fake quant is not enabled for every fake-on module.")

    models = {
        "prepared_fake_off": prepared_fake_off.to(device),
        "prepared_fake_on": prepared_fake_on.to(device),
        "converted": converted.to(device),
    }
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    metrics = {
        stage: build_validation_metric("rec", config, config_path=args.model_config)
        for stage in STAGES
    }
    pair_stats = {name: new_recognition_pair() for name, _, _ in PAIRS}
    finite = {stage: True for stage in STAGES}
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = {stage: model(images) for stage, model in models.items()}
            for stage, output in outputs.items():
                finite[stage] &= bool(torch.isfinite(output).all())
                metrics[stage].update(output, targets)
            for name, reference_stage, actual_stage in PAIRS:
                update_recognition_pair(
                    pair_stats[name],
                    outputs[reference_stage],
                    outputs[actual_stage],
                )

    result = {
        "task": "rec",
        "model_config": str(Path(args.model_config).resolve()),
        "model_config_sha256": file_sha256(args.model_config),
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": file_sha256(args.weights),
        "qat_config": str(Path(args.qat_config).resolve()),
        "qat_config_sha256": file_sha256(args.qat_config),
        "label_file": str(Path(args.label_file).resolve()),
        "label_file_sha256": file_sha256(args.label_file),
        "device": str(device),
        "sample_count": len(dataset),
        "batch_size": args.batch_size,
        "calibration_samples": calibration_samples,
        "calibration_batch_size": args.calibration_batch_size,
        "image_shape": list(args.image_shape),
        "reparameterized": args.reparameterize,
        "nodes": {stage: len(list(model.graph.nodes)) for stage, model in models.items()},
        "fake_quant_state": states,
        "observer_qparams": qparams,
        "finite": finite,
        "metrics": {stage: metric.compute() for stage, metric in metrics.items()},
        "comparisons": {
            name: compute_recognition_pair(stats)
            for name, stats in pair_stats.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "sample_count": result["sample_count"],
                "calibration_samples": calibration_samples,
                "nodes": result["nodes"],
                "fake_quant_state": {
                    name: {
                        key: value
                        for key, value in state.items()
                        if key != "module_states"
                    }
                    for name, state in states.items()
                },
                "observer_qparams": {
                    "modules": qparams["modules"],
                    "nonfinite": qparams["nonfinite"],
                },
                "finite": finite,
                "metrics": result["metrics"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    main(parse_args())

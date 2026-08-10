import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.training import (
    build_dataset,
    build_validation_metric,
    detection_collate,
    load_ocr_config,
)
from pytorchocr.diagnostics import resolve_onnx_input_shape


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a PP-OCR QuantONNX on a PaddleOCR-format dataset."
    )
    parser.add_argument("--task", choices=["det", "rec"], required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--image-shape",
        nargs=3,
        type=int,
        help="CHW shape used to resolve a dynamic-spatial ONNX input.",
    )
    parser.add_argument(
        "--ort-optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable ONNX Runtime graph optimizations (default: disabled for QDQ).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Evaluate only the first N samples; default evaluates the full set.",
    )
    return parser.parse_args()


def main(args):
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers non-negative.")
    if args.samples is not None and args.samples <= 0:
        raise ValueError("--samples must be positive.")
    session_options = ort.SessionOptions()
    if not args.ort_optimize:
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
    session = ort.InferenceSession(
        str(Path(args.onnx).resolve()),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    config = load_ocr_config(args.model_config)
    configured_shape = tuple(config["Global"].get("d2s_train_image_shape", ()))
    requested_shape = tuple(args.image_shape or configured_shape)
    input_shape = resolve_onnx_input_shape(session, requested_shape)
    onnx_batch = input_shape[0]
    if isinstance(onnx_batch, int) and onnx_batch != args.batch_size:
        raise ValueError(
            f"ONNX batch is {onnx_batch}, but --batch-size is {args.batch_size}."
        )
    image_shape = tuple(input_shape[1:])
    dataset = build_dataset(
        args.task,
        args.model_config,
        config,
        image_shape,
        args.label_file,
        args.data_dir,
        return_polygons=args.task == "det",
    )
    sample_count = len(dataset)
    if args.samples is not None:
        sample_count = min(args.samples, sample_count)
        dataset = Subset(dataset, range(sample_count))
    if isinstance(onnx_batch, int) and sample_count % onnx_batch != 0:
        raise ValueError(
            "Static-batch ONNX requires a validation sample count divisible by its batch."
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=detection_collate if args.task == "det" else None,
    )
    metric = build_validation_metric(
        args.task,
        config,
        config_path=args.model_config,
    )
    metric.reset()
    input_name = session.get_inputs()[0].name
    output_finite = True
    start = time.perf_counter()
    for images, targets in loader:
        output = session.run(
            None,
            {input_name: images.numpy()},
        )[0]
        output_finite &= bool(np.isfinite(output).all())
        metric.update(torch.from_numpy(output), targets)
    elapsed = time.perf_counter() - start
    result = {
        "task": args.task,
        "onnx": str(Path(args.onnx).resolve()),
        "label_file": str(Path(args.label_file).resolve()),
        "sample_count": sample_count,
        "input_shape": list(input_shape),
        "providers": session.get_providers(),
        "ort_optimized": args.ort_optimize,
        "output_finite": output_finite,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / elapsed,
        **metric.compute(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main(parse_args())

"""Evaluate PP-OCR float/QAT models from PyTorch checkpoints (.pt) or ONNX.

Three modes:
  --onnx <file>                    evaluate a QuantONNX/Float ONNX on a dataset
  --pt <checkpoint.pt> [--stage prepared|converted]
                                   evaluate a PT2E QAT checkpoint on a dataset
  --onnx <file> --pt <checkpoint>  alignment check: same inputs through ONNX
                                   Runtime and the PT2E model, reporting
                                   cosine similarity, MAE, MSE and argmax
                                   agreement (export alignment verification).

Dataset metrics reuse the PaddleOCR-format validation pipeline used by
train.py; alignment mode reports per-batch and aggregate error statistics.
"""

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

from pytorchocr.diagnostics import (
    build_prepared_qat_checkpoint,
    resolve_onnx_input_shape,
)
from pytorchocr.quantization import convert_prepared_model
from pytorchocr.quantization.validation import numpy_error_stats, sequence_error_stats
from pytorchocr.training import (
    build_dataset,
    build_validation_metric,
    detection_collate,
    load_ocr_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PP-OCR models from PyTorch QAT checkpoints and/or ONNX; "
            "with both, verify ONNX-vs-PT2E export alignment."
        )
    )
    parser.add_argument("--task", choices=["det", "rec"], required=True)
    parser.add_argument("--model-config", required=True)
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
        "--onnx",
        help="ONNX model path (QuantONNX or Float ONNX).",
    )
    parser.add_argument(
        "--pt",
        help="PyTorch QAT checkpoint path (.pt).",
    )
    parser.add_argument(
        "--weights",
        help="Float weights used to rebuild the PT2E graph (default: checkpoint metadata).",
    )
    parser.add_argument(
        "--stage",
        choices=["prepared", "converted"],
        default="converted",
        help="PT2E stage to evaluate (default: converted).",
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


def load_pt2e(args, config):
    checkpoint = torch.load(args.pt, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    if not metadata.get("qat", False):
        raise ValueError("Checkpoint is not marked as an Axera QAT checkpoint.")
    prepared, _ = build_prepared_qat_checkpoint(
        metadata,
        weights_path=args.weights,
        model_config=args.model_config,
    )
    prepared.load_state_dict(checkpoint["model"], strict=True)
    if args.stage == "converted":
        return convert_prepared_model(prepared), metadata
    return prepared, metadata


def run_onnx_session(args, input_shape):
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
    return session


def evaluate_onnx(args, config, session, input_shape):
    onnx_batch = input_shape[0]
    if isinstance(onnx_batch, int) and onnx_batch != args.batch_size:
        raise ValueError(
            f"ONNX batch is {onnx_batch}, but --batch-size is {args.batch_size}."
        )
    image_shape = tuple(input_shape[1:])
    full_count = None
    loader = _build_loader(args, config, image_shape, full_rec=False, sample_count=args.samples)
    sample_count = len(loader.dataset)
    if isinstance(onnx_batch, int) and sample_count % onnx_batch != 0:
        raise ValueError(
            "Static-batch ONNX requires a validation sample count divisible by its batch."
        )
    metric = build_validation_metric(args.task, config, config_path=args.model_config)
    metric.reset()
    input_name = session.get_inputs()[0].name
    output_finite = True
    start = time.perf_counter()
    for images, targets in loader:
        output = session.run(None, {input_name: images.numpy()})[0]
        output_finite &= bool(np.isfinite(output).all())
        metric.update(torch.from_numpy(output), targets)
    elapsed = time.perf_counter() - start
    return {
        "onnx": str(Path(args.onnx).resolve()),
        "sample_count": sample_count,
        "input_shape": list(input_shape),
        "providers": session.get_providers(),
        "ort_optimized": args.ort_optimize,
        "output_finite": output_finite,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / elapsed,
        **metric.compute(),
    }



def _build_loader(args, config, image_shape, full_rec, sample_count):
    dataset = build_dataset(
        args.task,
        args.model_config,
        config,
        image_shape,
        args.label_file,
        args.data_dir,
        return_polygons=args.task == "det",
        rec_multi_head=full_rec,
    )
    if sample_count is not None:
        dataset = Subset(dataset, range(sample_count))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=detection_collate if args.task == "det" else None,
    )


def _pt2e_forward(model, images, targets, full_rec):
    """Run a PT2E (prepared/converted) model and return the logits tensor."""
    device = images.device
    if full_rec:
        gtc = targets["gtc_targets"].to(device)
        outputs = model(images, gtc)
    else:
        outputs = model(images)
    if isinstance(outputs, (tuple, list)):
        output = outputs[0]
    elif isinstance(outputs, dict):
        output = outputs.get("logits", outputs.get("ctc"))
        if output is None:
            raise KeyError(f"No logits in PT2E outputs: {sorted(outputs)}")
    else:
        output = outputs
    return output


def _is_full_rec_graph(model):
    """Full recognition graphs take (images, gtc_targets) placeholders."""
    if getattr(model, "graph_role", None) == "pretrained_train":
        return True
    return sum(1 for node in model.graph.nodes if node.op == "placeholder") >= 2


def _move_pt2e_to_device(model, device):
    model = model.to(device)
    from torch.ao.quantization import move_exported_model_to_eval

    move_exported_model_to_eval(model)
    return model


def evaluate_pt2e(args, config, model):
    image_shape = tuple(args.image_shape or config["Global"].get("d2s_train_image_shape", ()))
    model = _move_pt2e_to_device(model, "cuda" if torch.cuda.is_available() else "cpu")
    full_rec = _is_full_rec_graph(model)
    loader = _build_loader(args, config, image_shape, full_rec=full_rec, sample_count=args.samples)
    sample_count = len(loader.dataset)
    metric = build_validation_metric(args.task, config, config_path=args.model_config)
    metric.reset()
    output_finite = True
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(next(model.parameters()).device)
            targets = {
                key: (
                    value.to(images.device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in targets.items()
            }
            output = _pt2e_forward(model, images, targets, full_rec)
            output = output.detach().cpu().numpy()
            output_finite &= bool(np.isfinite(output).all())
            metric.update(torch.from_numpy(output), targets)
    elapsed = time.perf_counter() - start
    return {
        "pt": str(Path(args.pt).resolve()),
        "stage": args.stage,
        "sample_count": sample_count,
        "output_finite": output_finite,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count / elapsed,
        **metric.compute(),
    }


def collect_aligned_outputs(args, config, session, pt_model, input_shape):
    onnx_batch = input_shape[0]
    if isinstance(onnx_batch, int) and onnx_batch != args.batch_size:
        raise ValueError(
            f"ONNX batch is {onnx_batch}, but --batch-size is {args.batch_size}."
        )
    image_shape = tuple(input_shape[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_model = _move_pt2e_to_device(pt_model, device)
    full_rec = _is_full_rec_graph(pt_model)
    loader = _build_loader(args, config, image_shape, full_rec=full_rec, sample_count=args.samples)
    sample_count = len(loader.dataset)
    if isinstance(onnx_batch, int) and sample_count % onnx_batch != 0:
        raise ValueError(
            "Static-batch ONNX requires a validation sample count divisible by its batch."
        )
    input_name = session.get_inputs()[0].name
    onnx_outputs = []
    pt_outputs = []
    with torch.no_grad():
        for images, targets in loader:
            ort_output = session.run(None, {input_name: images.numpy()})[0]
            pt_images = images.to(device)
            pt_targets = {
                key: (
                    value.to(device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in targets.items()
            }
            pt_output = _pt2e_forward(pt_model, pt_images, pt_targets, full_rec)
            onnx_outputs.append(ort_output)
            pt_outputs.append(pt_output.detach().cpu().numpy())
    return (
        np.concatenate(onnx_outputs, axis=0),
        np.concatenate(pt_outputs, axis=0),
    )


def alignment_report(onnx_outputs, pt_outputs):
    stats = numpy_error_stats(pt_outputs, onnx_outputs)
    sequence = sequence_error_stats(pt_outputs, onnx_outputs)
    return {
        "samples": int(onnx_outputs.shape[0]),
        "mae": stats["mae"],
        "mse": stats["mse"],
        "max_abs": stats["max_abs"],
        "cosine_similarity": stats["cosine_similarity"],
        "argmax_agreement": sequence["argmax_agreement"],
        "probability_mae": sequence["probability_mae"],
    }


def main(args):
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("--batch-size must be positive and --workers non-negative.")
    if args.samples is not None and args.samples <= 0:
        raise ValueError("--samples must be positive.")
    if not args.onnx and not args.pt:
        raise ValueError("At least one of --onnx / --pt is required.")
    config = load_ocr_config(args.model_config)

    session = None
    pt_model = None
    input_shape = None
    if args.onnx:
        session = run_onnx_session(args, None)
        configured_shape = tuple(config["Global"].get("d2s_train_image_shape", ()))
        requested_shape = tuple(args.image_shape or configured_shape)
        input_shape = resolve_onnx_input_shape(session, requested_shape)
    if args.pt:
        pt_model, _ = load_pt2e(args, config)

    result = {}
    if args.onnx:
        result["onnx_eval"] = evaluate_onnx(args, config, session, input_shape)
    if args.pt:
        result["pt_eval"] = evaluate_pt2e(args, config, pt_model)
    if args.onnx and args.pt:
        onnx_outputs, pt_outputs = collect_aligned_outputs(
            args, config, session, pt_model, input_shape
        )
        result["alignment"] = alignment_report(onnx_outputs, pt_outputs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main(parse_args())

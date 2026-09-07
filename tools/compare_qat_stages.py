import argparse
import json
import sys
from pathlib import Path

import torch
from torch.ao.quantization import disable_observer, move_exported_model_to_eval
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.quantization import (
    convert_prepared_model,
)
from pytorchocr.diagnostics import (
    ErrorAccumulator,
    build_diagnostic_dataset,
    build_prepared_qat_checkpoint,
    ctc_collapse,
    load_qat_checkpoint,
    outputs_as_tuple,
    sample_ids,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare prepared QAT and converted PT2E outputs on real images."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-file")
    parser.add_argument("--data-dir")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--onnx",
        help="Also compare the converted deployment output with this QuantONNX.",
    )
    parser.add_argument(
        "--ort-optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable ORT graph optimizations for --onnx (default: disabled).",
    )
    return parser.parse_args()


def main(args):
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be positive.")
    checkpoint, metadata = load_qat_checkpoint(args.checkpoint)
    task = metadata["task"]
    label_file = args.label_file or metadata.get("val_label_file")
    data_dir = args.data_dir or metadata.get("val_data_dir")
    if not label_file or not data_dir:
        raise ValueError("Validation label_file and data_dir are required.")

    dataset = build_diagnostic_dataset(task, metadata, label_file, data_dir)
    sample_count = min(args.samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    captured_batch = int(metadata.get("batch_size", 1))
    dynamic_batch = bool(metadata.get("dynamic_batch", False))
    if not dynamic_batch and (
        args.batch_size != captured_batch or sample_count % captured_batch != 0
    ):
        raise ValueError(
            "Static-batch checkpoint comparison requires --batch-size equal to "
            "the captured batch and a divisible sample count."
        )
    prepared, _ = build_prepared_qat_checkpoint(metadata)
    prepared.load_state_dict(checkpoint["model"], strict=True)
    prepared.apply(disable_observer)
    move_exported_model_to_eval(prepared)
    converted = convert_prepared_model(prepared)
    device = torch.device(args.device)
    prepared.to(device)
    converted.to(device)

    ort_session = None
    ort_input_name = None
    if args.onnx:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        if not args.ort_optimize:
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        ort_session = ort.InferenceSession(
            str(Path(args.onnx).resolve()),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        ort_input = ort_session.get_inputs()[0]
        ort_input_name = ort_input.name
        onnx_batch = ort_input.shape[0]
        if isinstance(onnx_batch, int) and onnx_batch != args.batch_size:
            raise ValueError(
                f"ONNX batch is {onnx_batch}, but --batch-size is {args.batch_size}."
            )

    names = ("shrink", "threshold", "binary") if task == "det" else ("logits",)
    errors = {name: ErrorAccumulator() for name in names}
    probability_error = ErrorAccumulator() if task == "rec" else None
    centered_logit_error = ErrorAccumulator() if task == "rec" else None
    argmax_matches = 0
    argmax_values = 0
    sequence_matches = 0
    sequence_values = 0
    onnx_error = ErrorAccumulator() if ort_session is not None else None
    onnx_probability_error = (
        ErrorAccumulator() if ort_session is not None and task == "rec" else None
    )
    onnx_argmax_matches = 0
    onnx_argmax_values = 0
    onnx_sequence_matches = 0
    onnx_sequence_values = 0
    onnx_finite = True
    prepared_finite = True
    converted_finite = True
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            prepared_outputs = outputs_as_tuple(prepared(images))
            converted_outputs = outputs_as_tuple(converted(images))
            if len(prepared_outputs) != len(names):
                raise ValueError(
                    f"Expected {len(names)} outputs, got {len(prepared_outputs)}."
                )
            for name, prepared_output, converted_output in zip(
                names,
                prepared_outputs,
                converted_outputs,
            ):
                errors[name].update(prepared_output, converted_output)
                prepared_finite &= bool(torch.isfinite(prepared_output).all())
                converted_finite &= bool(torch.isfinite(converted_output).all())
            if task == "rec":
                prepared_logits = prepared_outputs[0].float()
                converted_logits = converted_outputs[0].float()
                probability_error.update(
                    prepared_logits.softmax(dim=-1),
                    converted_logits.softmax(dim=-1),
                )
                centered_logit_error.update(
                    prepared_logits - prepared_logits.mean(dim=-1, keepdim=True),
                    converted_logits - converted_logits.mean(dim=-1, keepdim=True),
                )
                prepared_indices = prepared_logits.argmax(dim=-1)
                converted_indices = converted_logits.argmax(dim=-1)
                argmax_matches += int((prepared_indices == converted_indices).sum())
                argmax_values += prepared_indices.numel()
                for prepared_sequence, converted_sequence in zip(
                    prepared_indices,
                    converted_indices,
                ):
                    sequence_matches += int(
                        ctc_collapse(prepared_sequence)
                        == ctc_collapse(converted_sequence)
                    )
                    sequence_values += 1
            if ort_session is not None:
                ort_array = ort_session.run(
                    None,
                    {ort_input_name: images.detach().cpu().numpy()},
                )[0]
                onnx_output = torch.from_numpy(ort_array).to(device)
                converted_deployment_output = converted_outputs[0]
                onnx_error.update(converted_deployment_output, onnx_output)
                onnx_finite &= bool(torch.isfinite(onnx_output).all())
                if task == "rec":
                    converted_logits = converted_deployment_output.float()
                    onnx_logits = onnx_output.float()
                    onnx_probability_error.update(
                        converted_logits.softmax(dim=-1),
                        onnx_logits.softmax(dim=-1),
                    )
                    converted_indices = converted_logits.argmax(dim=-1)
                    onnx_indices = onnx_logits.argmax(dim=-1)
                    onnx_argmax_matches += int(
                        (converted_indices == onnx_indices).sum()
                    )
                    onnx_argmax_values += converted_indices.numel()
                    for converted_sequence, onnx_sequence in zip(
                        converted_indices,
                        onnx_indices,
                    ):
                        onnx_sequence_matches += int(
                            ctc_collapse(converted_sequence)
                            == ctc_collapse(onnx_sequence)
                        )
                        onnx_sequence_values += 1

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "task": task,
        "sample_count": sample_count,
        "sample_ids": sample_ids(dataset, sample_count),
        "prepared_finite": prepared_finite,
        "converted_finite": converted_finite,
        "outputs": {name: accumulator.compute() for name, accumulator in errors.items()},
    }
    if task == "rec":
        result["probabilities"] = probability_error.compute()
        result["centered_logits"] = centered_logit_error.compute()
        result["argmax_agreement"] = argmax_matches / argmax_values
        result["ctc_sequence_agreement"] = sequence_matches / sequence_values
    if ort_session is not None:
        onnx_result = {
            "path": str(Path(args.onnx).resolve()),
            "ort_optimized": args.ort_optimize,
            "finite": onnx_finite,
            "output": onnx_error.compute(),
        }
        if task == "rec":
            onnx_result.update(
                {
                    "probabilities": onnx_probability_error.compute(),
                    "argmax_agreement": (
                        onnx_argmax_matches / onnx_argmax_values
                    ),
                    "ctc_sequence_agreement": (
                        onnx_sequence_matches / onnx_sequence_values
                    ),
                }
            )
        result["converted_to_onnx"] = onnx_result
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main(parse_args())

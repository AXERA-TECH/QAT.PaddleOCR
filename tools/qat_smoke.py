import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.quantization import (
    build_qat_dynamic_shapes,
    convert_prepared_model,
    load_axera_quantizer,
    prepare_qat_model,
    run_onnx_reference,
    run_ort,
    run_smoke_step,
    sequence_error_stats,
    validate_qdq_graph,
)
from pytorchocr.diagnostics import outputs_as_tuple
from pytorchocr.quantization.onnx_export import export_qat_onnx
from pytorchocr.training import build_task_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PP-OCR Axera QAT smoke test.")
    parser.add_argument("--task", choices=["det", "rec"], default="det")
    parser.add_argument(
        "--det-graph",
        choices=["inference", "training"],
        default="inference",
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--qat-config", required=True)
    parser.add_argument(
        "--axera-root",
        help="Deprecated compatibility option; the quantizer is vendored.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-shape", nargs=3, type=int, default=[3, 128, 128])
    parser.add_argument(
        "--dynamic-heights",
        nargs="+",
        type=int,
        help="Allowed recognition heights for the prepared PT2E training graph only.",
    )
    parser.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse reparameterizable inference branches before QAT capture.",
    )
    parser.add_argument(
        "--keep-bn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the native post-sum BN of v6 RepDWConv during "
            "reparameterization (QARepVGG-style conv+bn deploy graph)."
        ),
    )
    parser.add_argument(
        "--onnx-optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNXProgram.optimize() before saving (enabled by default).",
    )
    parser.add_argument(
        "--onnx-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the slow ONNX Python ReferenceEvaluator (disabled by default).",
    )
    parser.add_argument(
        "--ort-optimizer-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run ORT graph-optimization A/B as a separate diagnostic.",
    )
    return parser.parse_args()


def build_model(args):
    return build_task_model(
        args.task,
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
        det_graph=args.det_graph,
        keep_bn=args.keep_bn,
    )


def output_mae(first, second):
    pairs = zip(outputs_as_tuple(first), outputs_as_tuple(second))
    errors = [torch.mean(torch.abs(left - right)) for left, right in pairs]
    return float(torch.stack(errors).mean())


def main():
    args = parse_args()
    if args.dynamic_heights and args.task != "rec":
        raise ValueError("--dynamic-heights is only valid for recognition.")
    torch.manual_seed(20260728)
    model = build_model(args)
    images = torch.randn([1, *args.image_shape], dtype=torch.float32)
    quantizer = load_axera_quantizer(args.qat_config)
    dynamic_shapes = build_qat_dynamic_shapes(
        images,
        dynamic_heights=args.dynamic_heights,
    )
    prepared, float_node_count = prepare_qat_model(
        model,
        (images,),
        quantizer,
        dynamic_shapes=dynamic_shapes,
    )
    del model
    _, loss, gradient_count = run_smoke_step(prepared, images)
    height_samples = {}
    prepared_height_outputs = {}
    heights = list(args.dynamic_heights or [args.image_shape[1]])
    # Keep the capture height last so converted qparams match the main report.
    ordered_heights = [height for height in heights if height != args.image_shape[1]]
    ordered_heights.append(args.image_shape[1])
    with torch.no_grad():
        for height in ordered_heights:
            sample = torch.randn(
                [1, args.image_shape[0], height, args.image_shape[2]],
                dtype=torch.float32,
            )
            height_samples[height] = sample
            prepared_height_outputs[height] = prepared(sample)
    qat_outputs = prepared_height_outputs[args.image_shape[1]]
    prepared_node_count = len(list(prepared.graph.nodes))
    converted = convert_prepared_model(prepared)
    del prepared
    gc.collect()
    with torch.no_grad():
        converted_outputs = converted(images)
    if not all(
        torch.isfinite(output).all()
        for output in outputs_as_tuple(converted_outputs)
    ):
        raise RuntimeError("Converted QAT model produced non-finite outputs.")
    output_name = "maps" if args.task == "det" else "logits"
    output_index = 0 if args.task == "det" and args.det_graph == "training" else None
    onnx_model = export_qat_onnx(
        converted,
        (images,),
        args.output,
        [output_name],
        optimize=args.onnx_optimize,
        output_index=output_index,
        static_axis_values={
            0: {
                1: args.image_shape[0],
                2: args.image_shape[1],
                3: args.image_shape[2],
            }
        },
    )
    with torch.no_grad():
        post_export_outputs = converted(images)
    converted_node_count = len(list(converted.graph.nodes))
    reference_outputs = (
        run_onnx_reference(onnx_model, images)
        if args.onnx_reference
        else None
    )
    exported_torch_output = (
        post_export_outputs[output_index]
        if output_index is not None
        else post_export_outputs
    )
    onnx_stats = validate_qdq_graph(onnx_model)
    onnx_input_shape = [
        dimension.dim_param or dimension.dim_value
        for dimension in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    prepared_height_checks = {
        str(height): {
            "output_shapes": [
                list(output.shape)
                for output in outputs_as_tuple(prepared_height_outputs[height])
            ]
        }
        for height in heights
    }
    prepared_converted_mae = output_mae(qat_outputs, converted_outputs)
    converted_pre_post_export_mae = output_mae(
        converted_outputs,
        post_export_outputs,
    )
    output_shape = list(exported_torch_output.shape)
    pytorch_output_abs_mean = float(exported_torch_output.abs().mean())
    torch_array = exported_torch_output.detach().cpu().numpy().copy()

    del converted
    del converted_outputs
    del post_export_outputs
    del prepared_height_outputs
    del qat_outputs
    del onnx_model
    gc.collect()

    ort_outputs = run_ort(args.output, images, optimize=False)
    optimized_ort_outputs = (
        run_ort(args.output, images, optimize=True)
        if args.ort_optimizer_check
        else None
    )
    result = {
        "float_nodes": float_node_count,
        "prepared_nodes": prepared_node_count,
        "converted_nodes": converted_node_count,
        "loss": loss,
        "gradient_tensors": gradient_count,
        "reparameterized": args.reparameterize,
        "task": args.task,
        "onnx_optimized": args.onnx_optimize,
        "det_graph": args.det_graph if args.task == "det" else None,
        "prepared_dynamic_heights": args.dynamic_heights or [],
        "prepared_height_checks": prepared_height_checks,
        "quantonnx_static_image_shape": args.image_shape,
        "prepared_converted_mae": prepared_converted_mae,
        "converted_pre_post_export_mae": converted_pre_post_export_mae,
        "output_shape": output_shape,
        "pytorch_output_abs_mean": pytorch_output_abs_mean,
        "ort_output_abs_mean": float(np.mean(np.abs(ort_outputs))),
        "pytorch_ort_mae": float(
            np.mean(np.abs(torch_array - ort_outputs))
        ),
        "pytorch_ort_max_abs": float(
            np.max(np.abs(torch_array - ort_outputs))
        ),
        "onnx": onnx_stats,
        "onnx_input_shape": onnx_input_shape,
    }
    if optimized_ort_outputs is not None:
        result["ort_optimizer_mae"] = float(
            np.mean(np.abs(ort_outputs - optimized_ort_outputs))
        )
        result["ort_optimizer_max_abs"] = float(
            np.max(np.abs(ort_outputs - optimized_ort_outputs))
        )
    if reference_outputs is not None:
        reference_difference = np.abs(torch_array - reference_outputs)
        result["pytorch_onnx_reference_mae"] = float(
            np.mean(reference_difference)
        )
        result["pytorch_onnx_reference_max_abs"] = float(
            np.max(reference_difference)
        )
    if args.task == "rec":
        if reference_outputs is not None:
            result["rec_onnx_reference"] = sequence_error_stats(
                torch_array,
                reference_outputs,
            )
        result["rec_ort"] = sequence_error_stats(torch_array, ort_outputs)
        if optimized_ort_outputs is not None:
            result["rec_ort_optimized"] = sequence_error_stats(
                torch_array,
                optimized_ort_outputs,
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

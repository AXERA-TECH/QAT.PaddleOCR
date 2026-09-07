import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pytorchocr.quantization import (
    convert_prepared_model,
    load_axera_quantizer,
    prepare_qat_model,
    run_smoke_step,
    validate_qdq_graph,
)
from pytorchocr.diagnostics import outputs_as_tuple
from pytorchocr.quantization.onnx_export import export_qat_onnx
from pytorchocr.training import build_task_model


STAGES = ("build", "forward", "prepare", "backward", "convert", "onnx", "qdq")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report staged float/PT2E/QuantONNX compatibility for one OCR model."
    )
    parser.add_argument("--task", choices=["det", "rec"], required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--qat-config")
    parser.add_argument("--weights")
    parser.add_argument("--image-shape", nargs=3, type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        default="qdq",
    )
    return parser.parse_args()


def main(args):
    torch.manual_seed(20260803)
    result = {
        "task": args.task,
        "model_config": str(Path(args.model_config).resolve()),
        "image_shape": [1, *args.image_shape],
        "stages": {},
    }

    def complete(stage, **details):
        result["stages"][stage] = {"status": "passed", **details}
        return STAGES.index(stage) >= STAGES.index(args.stop_after)

    try:
        model = build_task_model(
            args.task,
            args.model_config,
            weights_path=args.weights,
            reparameterize=True,
            det_graph="training",
        )
        if complete("build"):
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        images = torch.randn([1, *args.image_shape])
        with torch.no_grad():
            float_outputs = model(images)
        shapes = [list(output.shape) for output in outputs_as_tuple(float_outputs)]
        if complete("forward", output_shapes=shapes):
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        if not args.qat_config:
            raise ValueError("--qat-config is required after the forward stage.")
        quantizer = load_axera_quantizer(args.qat_config)
        prepared, float_nodes = prepare_qat_model(model, (images,), quantizer)
        if complete(
            "prepare",
            float_nodes=float_nodes,
            prepared_nodes=len(list(prepared.graph.nodes)),
        ):
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        _, loss, gradients = run_smoke_step(prepared, images)
        if complete("backward", loss=loss, gradient_tensors=gradients):
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        converted = convert_prepared_model(prepared)
        with torch.no_grad():
            converted_outputs = converted(images)
        finite = all(
            bool(torch.isfinite(output).all())
            for output in outputs_as_tuple(converted_outputs)
        )
        if complete(
            "convert",
            converted_nodes=len(list(converted.graph.nodes)),
            output_finite=finite,
        ):
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        output_path = Path(args.output or f"/tmp/{Path(args.model_config).stem}.onnx")
        output_index = 0 if args.task == "det" else None
        onnx_model = export_qat_onnx(
            converted,
            (images,),
            output_path,
            ["maps" if args.task == "det" else "logits"],
            output_index=output_index,
        )
        if complete("onnx", output=str(output_path.resolve())):
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        stats = validate_qdq_graph(onnx_model)
        complete("qdq", stats=stats)
    except Exception as error:
        failed_stage = next(stage for stage in STAGES if stage not in result["stages"])
        result["stages"][failed_stage] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        result["status"] = "failed"
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(1)

    result["status"] = "passed"
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())

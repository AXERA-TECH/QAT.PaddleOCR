#!/usr/bin/env python3
"""Generate a generic Axera Pulsar2 config from an exact QuantONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pulsar2_config import (
    choose_input,
    load_layer_configs,
    model_contract,
    validate_config,
    validate_layer_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("generic", "ppocrv5-rec"),
        default="generic",
        help="Optional model-specific rules; generic only creates the base config.",
    )
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-hardware", default="AX650")
    parser.add_argument("--npu-mode", default="NPU3")
    parser.add_argument("--calibration-dataset", required=True)
    parser.add_argument("--input-name")
    parser.add_argument("--layer-configs", type=Path)
    parser.add_argument("--frontend-onnx", type=Path)
    parser.add_argument("--input-layout", default="NCHW")
    parser.add_argument("--src-layout", default="NCHW")
    parser.add_argument("--src-dtype", default="FP32")
    parser.add_argument("--mean", nargs="+", type=float, default=[0, 0, 0])
    parser.add_argument("--std", nargs="+", type=float, default=[1, 1, 1])
    parser.add_argument("--conv-bias-data-type", default="FP32")
    parser.add_argument("--precision-analysis", action="store_true")
    parser.add_argument("--precision-analysis-method", default="PerLayer")
    parser.add_argument("--precision-analysis-mode", default="NPUBackend")
    parser.add_argument("--expected-input-shape", nargs=4, type=int, default=[1, 3, 48, 320])
    parser.add_argument("--expected-classes", type=int, default=18385)
    parser.add_argument("--expected-attention", type=int, default=2)
    parser.add_argument("--expected-requant", type=int, default=1)
    parser.add_argument("--expected-silu", type=int, default=7)
    parser.add_argument(
        "--attention-dtype",
        choices=("S8", "S16"),
        default="S16",
        help="Signed dtype inside the PP-OCRv5-rec Attention profile.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing config: {args.output}")
    if args.frontend_onnx is not None and not args.frontend_onnx.is_file():
        raise SystemExit(f"Frontend ONNX does not exist: {args.frontend_onnx}")
    if args.profile == "ppocrv5-rec":
        if args.layer_configs is not None:
            raise SystemExit("--layer-configs cannot be combined with --profile ppocrv5-rec.")
        from profiles.ppocrv5_rec import build_config, inspect_model

        inspection = inspect_model(
            args.onnx,
            args.expected_input_shape,
            args.expected_classes,
            args.expected_attention,
            args.expected_requant,
            args.expected_silu,
            args.attention_dtype,
        )
        config = build_config(args, inspection)
        if args.frontend_onnx is not None:
            raise SystemExit(
                "--frontend-onnx is only used with generic explicit layer_configs; "
                "validate v5-rec source names before frontend remapping."
            )
    else:
        contract = model_contract(args.onnx)
        input_info = choose_input(contract, args.input_name)
        layer_configs = load_layer_configs(args.layer_configs)
        validation_names = set(contract["node_names"])
        if args.frontend_onnx is not None:
            validation_names = set(model_contract(args.frontend_onnx)["node_names"])
        validate_layer_configs(layer_configs, validation_names)
        config = None
    if len(args.mean) != len(args.std):
        raise SystemExit("--mean and --std must have the same number of values.")
    if config is None:
        processor = {
            "tensor_name": input_info["name"],
            "tensor_layout": args.input_layout,
            "src_layout": args.src_layout,
            "src_dtype": args.src_dtype,
            "mean": args.mean,
            "std": args.std,
        }
        quant = {
            "input_configs": [
                {
                    "tensor_name": input_info["name"],
                    "calibration_dataset": args.calibration_dataset,
                }
            ],
            "layer_configs": layer_configs,
            "conv_bias_data_type": args.conv_bias_data_type,
            "precision_analysis": bool(args.precision_analysis),
        }
        if args.precision_analysis:
            quant["precision_analysis_method"] = args.precision_analysis_method
            quant["precision_analysis_mode"] = args.precision_analysis_mode
        config = {
            "input": str(args.onnx.resolve()),
            "output_dir": str(
                (args.output_dir or Path(f"./output_{args.onnx.stem}")).resolve()
            ),
            "model_type": "QuantONNX",
            "target_hardware": args.target_hardware,
            "npu_mode": args.npu_mode,
            "quant": quant,
            "input_processors": [processor],
            "output_processors": [],
            "compiler": {"check": 2},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary = validate_config(
        args.onnx,
        args.output,
        args.frontend_onnx if args.profile == "generic" else None,
        args.target_hardware,
        args.npu_mode,
    )
    if args.profile == "ppocrv5-rec":
        from profiles.ppocrv5_rec import validate_profile_config

        validate_profile_config(
            config,
            inspection,
            args.target_hardware,
            args.npu_mode,
            args.attention_dtype,
        )
        summary["profile"] = args.profile
        summary["attention_regions"] = len(inspection["attention"])
        summary["requants"] = len(inspection["requants"])
        summary["silu"] = inspection["silu"]["total"]
    print(json.dumps({"config": str(args.output.resolve()), **summary}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate a PP-OCRv5 rec Pulsar2 config against its exact QuantONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_ppocrv5_rec_pulsar2_config import inspect_model, json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-input-shape", nargs=4, type=int, default=[1, 3, 48, 320])
    parser.add_argument("--expected-classes", type=int, default=18385)
    parser.add_argument("--expected-attention", type=int, default=2)
    parser.add_argument("--expected-requant", type=int, default=1)
    parser.add_argument("--expected-silu", type=int, default=7)
    parser.add_argument("--expected-target-hardware", default="AX650")
    parser.add_argument("--expected-npu-mode", default="NPU3")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def find_rule(rules: list[dict[str, Any]], names: list[str], role: str) -> dict[str, Any]:
    expected = set(names)
    matches = [rule for rule in rules if set(rule.get("layer_names", [])) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {role} rule for {names}, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    inspection = inspect_model(
        args.onnx,
        args.expected_input_shape,
        args.expected_classes,
        args.expected_attention,
        args.expected_requant,
        args.expected_silu,
    )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    configured_input = config.get("input")
    require(isinstance(configured_input, str), "Config input must be an ONNX path string")
    require(
        Path(configured_input).resolve() == args.onnx.resolve(),
        "Config input does not match the requested ONNX path",
    )
    require(config.get("model_type") == "QuantONNX", "model_type must be QuantONNX")
    require(
        config.get("target_hardware") == args.expected_target_hardware,
        f"target_hardware must be {args.expected_target_hardware}",
    )
    require(
        config.get("npu_mode") == args.expected_npu_mode,
        f"npu_mode must be {args.expected_npu_mode}",
    )
    require(config.get("compiler") == {"check": 2}, "compiler.check must be 2")
    require(config.get("output_processors") == [], "output_processors must be empty")

    quant = config.get("quant", {})
    input_configs = quant.get("input_configs", [])
    require(len(input_configs) == 1, "Exactly one quant.input_configs item is required")
    require(input_configs[0].get("tensor_name") == "DEFAULT", "input_configs tensor_name must be DEFAULT")
    require(bool(input_configs[0].get("calibration_dataset")), "calibration_dataset field is required by Pulsar2")
    processors = config.get("input_processors", [])
    expected_processor = {
        "tensor_name": "DEFAULT",
        "tensor_layout": "NCHW",
        "src_layout": "NCHW",
        "src_dtype": "FP32",
        "mean": [0, 0, 0],
        "std": [1, 1, 1],
    }
    require(processors == [expected_processor], "Input processor must remain FP32/NCHW identity")

    regions = inspection["attention"]
    qkv_names = [region["qkv_output"] for region in regions]
    core_names = [name for region in regions for name in region["core"]]
    second_names = [region["second_matmul"] for region in regions]
    rules = quant.get("layer_configs", [])
    require(len(rules) == 3, "Exactly three generated layer_configs groups are required")
    qkv_rule = find_rule(rules, qkv_names, "QKV output S16")
    core_rule = find_rule(rules, core_names, "attention core S16")
    second_rule = find_rule(rules, second_names, "second MatMul S16 input")
    require(qkv_rule == {"layer_names": qkv_names, "output_data_type": "S16"}, "Invalid QKV S16 rule")
    require(
        core_rule == {"layer_names": core_names, "data_type": "S16", "output_data_type": "S16"},
        "Invalid attention core S16 rule",
    )
    require(second_rule == {"layer_names": second_names, "data_type": "S16"}, "Invalid second MatMul S16 rule")

    if args.report is not None:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        require(report.get("onnx", {}).get("sha256") == inspection["sha256"], "Report ONNX SHA256 is stale")
        require(
            report.get("pulsar2", {}).get("config_sha256") == json_sha256(config),
            "Report config SHA256 is stale",
        )
    print(
        json.dumps(
            {
                "status": "pass",
                "onnx": str(args.onnx),
                "config": str(args.config),
                "onnx_sha256": inspection["sha256"],
                "attention_regions": len(regions),
                "requants": len(inspection["requants"]),
                "silu": inspection["silu"]["total"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

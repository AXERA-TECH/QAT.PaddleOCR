#!/usr/bin/env python3
"""Validate a generic Axera Pulsar2 config against QuantONNX and frontend graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pulsar2_config import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("generic", "ppocrv5-rec"),
        default="generic",
        help="Optional model-specific validation profile.",
    )
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frontend-onnx", type=Path)
    parser.add_argument("--target-hardware")
    parser.add_argument("--npu-mode")
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
    if not args.config.is_file():
        raise SystemExit(f"Config does not exist: {args.config}")
    summary = validate_config(
        args.onnx,
        args.config,
        None if args.profile == "ppocrv5-rec" else args.frontend_onnx,
        args.target_hardware,
        args.npu_mode,
    )
    if args.profile == "ppocrv5-rec":
        if args.frontend_onnx is not None:
            raise SystemExit(
                "--frontend-onnx is not supported with --profile ppocrv5-rec."
            )
        from profiles.ppocrv5_rec import inspect_model, validate_profile_config

        inspection = inspect_model(
            args.onnx,
            args.expected_input_shape,
            args.expected_classes,
            args.expected_attention,
            args.expected_requant,
            args.expected_silu,
            args.attention_dtype,
        )
        config = json.loads(args.config.read_text(encoding="utf-8"))
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
    print(json.dumps({"status": "pass", **summary}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Discover PP-OCR det/rec FX regions and generate an Axera QAT config."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from pytorchocr.training import build_task_model, load_ocr_config


LINEAR_OPS = {torch.ops.aten.linear.default}
MATMUL_OPS = {torch.ops.aten.matmul.default}
MUL_OPS = {torch.ops.aten.mul.Tensor}
SOFTMAX_OPS = {torch.ops.aten.softmax.int}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("auto", "det", "rec"), default="auto")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--image-shape", nargs=3, type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--expected-attention",
        type=int,
        help="Optional exact Attention-region count; zero is valid for det models.",
    )
    parser.add_argument(
        "--attention-dtype",
        choices=("auto", "S8", "S16"),
        default="auto",
        help=(
            "Signed activation dtype inside Attention. auto maps global U8 to S8 "
            "and global U16 to S16; use an explicit value for mixed-width regions."
        ),
    )
    parser.add_argument(
        "--reparameterize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def module_entries(node: torch.fx.Node) -> list[tuple[str, str]]:
    stack = node.meta.get("nn_module_stack") or {}
    return [
        (str(value[0]), str(value[1]))
        for value in stack.values()
        if isinstance(value, tuple) and len(value) >= 2
    ]


def attention_owner(node: torch.fx.Node) -> str | None:
    for path, module_type in reversed(module_entries(node)):
        if module_type.endswith(".Attention"):
            return path
    return None


def module_paths(node: torch.fx.Node) -> list[str]:
    return [path for path, _ in module_entries(node)]


def node_inputs(value: Any):
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from node_inputs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from node_inputs(item)


def direct_node_inputs(node: torch.fx.Node) -> list[torch.fx.Node]:
    return list(node_inputs((node.args, node.kwargs)))


def exactly_one(nodes: list[torch.fx.Node], role: str, owner: str) -> torch.fx.Node:
    if len(nodes) != 1:
        raise RuntimeError(f"{owner} expected one {role}, found {[node.name for node in nodes]}")
    return nodes[0]


def discover(
    gm: torch.fx.GraphModule,
    expected_attention: int | None,
) -> dict[str, Any]:
    nodes = list(gm.graph.nodes)
    order = {node: position for position, node in enumerate(nodes)}
    groups: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for node in nodes:
        owner = attention_owner(node)
        if owner is not None:
            groups[owner].append(node)

    regions = []
    for owner, owned in groups.items():
        qkv = exactly_one(
            [
                node
                for node in owned
                if node.op == "call_function"
                and node.target in LINEAR_OPS
                and any(path.endswith(".mixer.qkv") for path in module_paths(node))
            ],
            "QKV Linear",
            owner,
        )
        matmuls = sorted(
            [node for node in owned if node.op == "call_function" and node.target in MATMUL_OPS],
            key=order.__getitem__,
        )
        if len(matmuls) != 2:
            raise RuntimeError(f"{owner} expected two MatMul nodes, found {[node.name for node in matmuls]}")
        first, second = matmuls
        softmax = exactly_one(
            [node for node in owned if node.op == "call_function" and node.target in SOFTMAX_OPS],
            "Softmax",
            owner,
        )
        scale_mul = exactly_one(
            [
                node
                for node in owned
                if node.op == "call_function"
                and node.target in MUL_OPS
                and order[qkv] < order[node] < order[first]
            ],
            "Q scale Mul",
            owner,
        )
        first_inputs = set(direct_node_inputs(first))
        second_inputs = set(direct_node_inputs(second))
        if scale_mul not in first_inputs:
            raise RuntimeError(f"{owner} scale Mul does not feed the first MatMul")
        if first not in set(direct_node_inputs(softmax)):
            raise RuntimeError(f"{owner} first MatMul does not feed Softmax directly")
        if softmax not in second_inputs:
            # Training-mode dropout may remain between Softmax and second MatMul.
            if not any(
                input_node.target == torch.ops.aten.dropout.default
                and softmax in direct_node_inputs(input_node)
                for input_node in second_inputs
                if input_node.op == "call_function"
            ):
                raise RuntimeError(f"{owner} Softmax does not feed the second MatMul")
        regions.append(
            {
                "owner": owner,
                "qkv_linear": qkv.name,
                "scale_mul": scale_mul.name,
                "first_matmul": first.name,
                "softmax": softmax.name,
                "second_matmul": second.name,
                "source_fn_stack": {
                    role: [str(item) for item in node.meta.get("source_fn_stack", [])]
                    for role, node in {
                        "qkv_linear": qkv,
                        "scale_mul": scale_mul,
                        "first_matmul": first,
                        "softmax": softmax,
                        "second_matmul": second,
                    }.items()
                },
            }
        )
    regions.sort(key=lambda item: item["owner"])
    if expected_attention is not None and len(regions) != expected_attention:
        raise RuntimeError(
            f"Expected {expected_attention} complete Attention regions, "
            f"found {len(regions)}: {[item['owner'] for item in regions]}"
        )
    return {"attention": regions, "fx_nodes": len(nodes)}


def signed_qspec(dtype: str) -> dict[str, Any]:
    if dtype == "S8":
        return {"dtype": "S8", "qmin": -127, "qmax": 127}
    if dtype == "S16":
        return {"dtype": "S16", "qmin": -32767, "qmax": 32767}
    raise ValueError(f"Unsupported signed activation dtype: {dtype}")


def resolve_attention_dtype(
    requested: str,
    global_activation: dict[str, Any],
) -> str:
    if requested != "auto":
        signed_qspec(requested)
        return requested
    activation_dtype = global_activation.get("dtype")
    inferred = {"U8": "S8", "U16": "S16"}.get(activation_dtype)
    if inferred is None:
        raise RuntimeError(
            f"Cannot infer Attention dtype from global activation {activation_dtype!r}"
        )
    return inferred


def validate_global_qspec(
    global_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    activation = copy.deepcopy(global_config.get("input", {}))
    weight = copy.deepcopy(global_config.get("weight", {}))
    expected_ranges = {
        "U8": (0, 255),
        "U16": (0, 65535),
        "S8": (-127, 127),
        "S16": (-32767, 32767),
    }
    for role, qspec, allowed in (
        ("activation", activation, {"U8", "U16"}),
        ("weight", weight, {"S8", "S16"}),
    ):
        dtype = qspec.get("dtype")
        if dtype not in allowed:
            raise RuntimeError(
                f"PP-OCR global {role} dtype must be one of {sorted(allowed)}"
            )
        if (qspec.get("qmin"), qspec.get("qmax")) != expected_ranges[dtype]:
            raise RuntimeError(f"Invalid {dtype} qmin/qmax for global {role}")
    return activation, weight


def update_config(
    template: dict[str, Any],
    discovery: dict[str, Any],
    attention_dtype: str,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    global_config = config.get("global_config", {})
    global_activation, global_weight = validate_global_qspec(global_config)

    regions = discovery["attention"]
    if not regions:
        config["regional_configs"] = []
        return config
    resolved_attention_dtype = resolve_attention_dtype(
        attention_dtype,
        global_activation,
    )
    signed = signed_qspec(resolved_attention_dtype)
    config["regional_configs"] = [
        {
            "module_names": [item["qkv_linear"] for item in regions],
            "module_type": "linear",
            "module_config": {
                "is_symmetric": False,
                "output_is_symmetric": True,
                "input": global_activation,
                "output": signed,
                "weight": global_weight,
            },
        },
        {
            "module_names": [item["scale_mul"] for item in regions],
            "module_type": "mul",
            "module_config": {
                "is_symmetric": True,
                "output_is_symmetric": True,
                "input": signed,
                "output": signed,
            },
        },
        {
            "module_names": [item["first_matmul"] for item in regions],
            "module_type": "matmul",
            "module_config": {
                "is_symmetric": True,
                "output_is_symmetric": True,
                "input": signed,
                "output": signed,
            },
        },
        {
            "module_names": [item["softmax"] for item in regions],
            "module_type": "softmax",
            "module_config": {
                "is_symmetric": True,
                "output_is_symmetric": True,
                "input": signed,
                "output": signed,
            },
        },
        {
            "module_names": [item["second_matmul"] for item in regions],
            "module_type": "matmul",
            "module_config": {
                "is_symmetric": True,
                "output_is_symmetric": False,
                "input": signed,
                "output": global_activation,
            },
        },
    ]
    return config


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.expected_attention is not None and args.expected_attention < 0:
        raise ValueError("--expected-attention must be non-negative")
    if not args.check and args.output is None:
        raise ValueError("--output is required unless --check is used")
    if args.output is not None:
        if args.output.resolve() == args.base_config.resolve():
            raise ValueError("--output must not overwrite --base-config")
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    model_config = load_ocr_config(args.model_config)
    detected_task = model_config["Architecture"].get("model_type")
    if detected_task not in ("det", "rec"):
        raise ValueError(f"Unsupported PP-OCR model_type: {detected_task}")
    if args.task != "auto" and args.task != detected_task:
        raise ValueError(
            f"--task {args.task} does not match model config task {detected_task}"
        )
    task = detected_task
    model = build_task_model(
        task,
        args.model_config,
        weights_path=args.weights,
        reparameterize=args.reparameterize,
        det_graph="inference",
        rec_graph="deploy",
    )
    images = torch.zeros([args.batch_size, *args.image_shape], dtype=torch.float32)
    gm = torch.export.export_for_training(model, (images,)).module()
    discovery = discover(gm, args.expected_attention)
    template = json.loads(args.base_config.read_text(encoding="utf-8"))
    global_activation, _ = validate_global_qspec(template.get("global_config", {}))
    resolved_attention_dtype = resolve_attention_dtype(
        args.attention_dtype,
        global_activation,
    )
    generated = update_config(template, discovery, resolved_attention_dtype)
    if args.check:
        if generated != template:
            raise SystemExit(
                "ERROR: QAT regional module names do not match the current export_for_training graph"
            )
        print("QAT config FX node structure: PASS")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")
    report = {
        "task": task,
        "model_config": str(Path(args.model_config).resolve()),
        "weights": str(Path(args.weights).resolve()) if args.weights else None,
        "base_config": str(args.base_config.resolve()),
        "output": str(args.output.resolve()) if args.output else None,
        "image_shape": [args.batch_size, *args.image_shape],
        "reparameterized": args.reparameterize,
        "attention_dtype_requested": args.attention_dtype,
        "attention_dtype": resolved_attention_dtype,
        **discovery,
    }
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for position, region in enumerate(discovery["attention"], start=1):
        print(f"attention {position}: {region['owner']}")
        print(
            "  "
            f"qkv={region['qkv_linear']} scale={region['scale_mul']} "
            f"matmul={region['first_matmul']} softmax={region['softmax']} "
            f"matmul_out={region['second_matmul']}"
        )
    if not discovery["attention"]:
        print("attention regions: 0 (global QAT config only)")
    else:
        print(f"attention dtype: {resolved_attention_dtype}")
    if args.output is not None:
        print(f"generated config: {args.output}")


if __name__ == "__main__":
    main()

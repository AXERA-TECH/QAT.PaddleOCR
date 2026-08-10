#!/usr/bin/env python3
"""Generate an AXERA Pulsar2 config from a PP-OCRv5 rec QuantONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


DTYPE_NAMES = {
    TensorProto.INT8: "S8",
    TensorProto.UINT8: "U8",
    TensorProto.INT16: "S16",
    TensorProto.UINT16: "U16",
}
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
REGION_OPS = {
    "Div",
    "Gather",
    "Identity",
    "MatMul",
    "Mul",
    "Reshape",
    "Slice",
    "Softmax",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Defaults to <output>.report.json.")
    parser.add_argument("--output-dir", help="Pulsar2 build output directory.")
    parser.add_argument("--target-hardware", default="AX650")
    parser.add_argument("--npu-mode", default="NPU3")
    parser.add_argument(
        "--calibration-dataset",
        default="/path/to/dataset",
        help="Required Pulsar2 field; QuantONNX Q/DQ does not use PTQ calibration.",
    )
    parser.add_argument("--expected-input-shape", nargs=4, type=int, default=[1, 3, 48, 320])
    parser.add_argument("--expected-classes", type=int, default=18385)
    parser.add_argument("--expected-attention", type=int, default=2)
    parser.add_argument("--expected-requant", type=int, default=1)
    parser.add_argument("--expected-silu", type=int, default=7)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def tensor_contract(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "shape": tensor_shape(value_info),
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
    }


class GraphIndex:
    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.nodes = list(model.graph.node)
        names = [node.name for node in self.nodes]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise RuntimeError("Every ONNX node must have a unique non-empty name")
        self.producers = {output: node for node in self.nodes for output in node.output}
        self.consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
        for node in self.nodes:
            for input_name in node.input:
                self.consumers[input_name].append(node)
        self.initializers = {item.name: item for item in model.graph.initializer}

    def qdtype(self, node: onnx.NodeProto | None) -> str | None:
        if node is None or node.op_type not in QDQ_OPS or len(node.input) < 3:
            return None
        zero_point = self.initializers.get(node.input[2])
        if zero_point is None:
            return None
        return DTYPE_NAMES.get(zero_point.data_type)

    def input_qdtype(self, node: onnx.NodeProto, index: int) -> str | None:
        producer = self.producers.get(node.input[index])
        while producer is not None and producer.op_type == "Identity":
            producer = self.producers.get(producer.input[0])
        return self.qdtype(producer) if producer is not None and producer.op_type == "DequantizeLinear" else None

    def output_qdtypes(self, node: onnx.NodeProto) -> set[str]:
        found: set[str] = set()
        queue = deque(node.output)
        visited: set[str] = set()
        while queue:
            tensor = queue.popleft()
            if tensor in visited:
                continue
            visited.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                dtype = self.qdtype(consumer)
                if consumer.op_type == "QuantizeLinear" and dtype is not None:
                    found.add(dtype)
                elif consumer.op_type == "Identity":
                    queue.extend(consumer.output)
        return found

    def nearest_upstream(self, tensor: str, op_type: str, max_depth: int = 12) -> list[onnx.NodeProto]:
        queue = deque([(tensor, 0)])
        visited: set[str] = set()
        found: dict[str, onnx.NodeProto] = {}
        while queue:
            current, depth = queue.popleft()
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            producer = self.producers.get(current)
            if producer is None:
                continue
            if producer.op_type == op_type:
                found[producer.name] = producer
                continue
            if producer.op_type in QDQ_OPS | {"Identity", "Mul", "Div"}:
                for input_name in producer.input:
                    if input_name not in self.initializers:
                        queue.append((input_name, depth + 1))
        return list(found.values())

    def nearest_downstream(self, tensor: str, op_type: str, max_depth: int = 12) -> list[onnx.NodeProto]:
        queue = deque([(tensor, 0)])
        visited: set[str] = set()
        found: dict[str, onnx.NodeProto] = {}
        while queue:
            current, depth = queue.popleft()
            if current in visited or depth > max_depth:
                continue
            visited.add(current)
            for consumer in self.consumers.get(current, []):
                if consumer.op_type == op_type:
                    found[consumer.name] = consumer
                elif consumer.op_type in QDQ_OPS | {"Identity", "Transpose"}:
                    for output_name in consumer.output:
                        queue.append((output_name, depth + 1))
        return list(found.values())

    def ancestors(self, tensors: Iterable[str], max_depth: int = 40) -> dict[str, onnx.NodeProto]:
        queue = deque((tensor, 0) for tensor in tensors)
        visited: set[str] = set()
        result: dict[str, onnx.NodeProto] = {}
        while queue:
            tensor, depth = queue.popleft()
            if tensor in visited or depth > max_depth:
                continue
            visited.add(tensor)
            producer = self.producers.get(tensor)
            if producer is None:
                continue
            result[producer.name] = producer
            for input_name in producer.input:
                if input_name not in self.initializers:
                    queue.append((input_name, depth + 1))
        return result

    def region_between(self, source: onnx.NodeProto, sink: onnx.NodeProto) -> list[onnx.NodeProto]:
        sink_ancestors = self.ancestors(sink.input, max_depth=50)
        relevant = set(sink_ancestors) | {sink.name}
        queue = deque(source.output)
        visited_tensors: set[str] = set()
        reached: dict[str, onnx.NodeProto] = {}
        while queue:
            tensor = queue.popleft()
            if tensor in visited_tensors:
                continue
            visited_tensors.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.name not in relevant:
                    continue
                reached[consumer.name] = consumer
                if consumer.name != sink.name:
                    queue.extend(consumer.output)
        if sink.name not in reached:
            raise RuntimeError(f"{source.name} does not reach {sink.name}")
        return [node for node in self.nodes if node.name in reached]


def exactly_one(nodes: Iterable[onnx.NodeProto], role: str) -> onnx.NodeProto:
    items = list(nodes)
    if len(items) != 1:
        raise RuntimeError(f"Expected one {role}, found {[node.name for node in items]}")
    return items[0]


def require_output_dtype(index: GraphIndex, node: onnx.NodeProto, expected: str) -> None:
    actual = index.output_qdtypes(node)
    if actual != {expected}:
        raise RuntimeError(f"{node.name} output must be {expected}, got {sorted(actual)}")


def discover_attention(index: GraphIndex, expected: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for softmax in (node for node in index.nodes if node.op_type == "Softmax"):
        first = exactly_one(index.nearest_upstream(softmax.input[0], "MatMul"), f"first MatMul for {softmax.name}")
        second = exactly_one(index.nearest_downstream(softmax.output[0], "MatMul"), f"second MatMul for {softmax.name}")
        ancestors = index.ancestors(list(first.input) + list(second.input), max_depth=40)
        qkv_candidates = []
        for node in ancestors.values():
            if node.op_type != "Add":
                continue
            bias_names = [name for name in node.input if name in index.initializers]
            if any(name.endswith(".mixer.qkv.bias") for name in bias_names):
                qkv_candidates.append(node)
        qkv = exactly_one(qkv_candidates, f"QKV Linear Add for {softmax.name}")
        region_nodes = index.region_between(qkv, second)
        core = [
            node
            for node in region_nodes
            if node.name not in {qkv.name, second.name} and node.op_type not in QDQ_OPS
        ]
        unsupported = [node for node in core if node.op_type not in REGION_OPS]
        if unsupported:
            raise RuntimeError(
                f"Unsupported operators in {softmax.name} S16 region: "
                f"{[(node.name, node.op_type) for node in unsupported]}"
            )
        if first.name not in {node.name for node in core} or softmax.name not in {node.name for node in core}:
            raise RuntimeError(f"Incomplete S16 core for {softmax.name}")
        require_output_dtype(index, qkv, "S16")
        for node in core:
            require_output_dtype(index, node, "S16")
        first_inputs = [index.input_qdtype(first, position) for position in range(2)]
        second_inputs = [index.input_qdtype(second, position) for position in range(2)]
        if first_inputs != ["S16", "S16"]:
            raise RuntimeError(f"{first.name} inputs must be S16/S16, got {first_inputs}")
        if second_inputs != ["S16", "S16"]:
            raise RuntimeError(f"{second.name} inputs must be S16/S16, got {second_inputs}")
        require_output_dtype(index, second, "U16")
        regions.append(
            {
                "qkv_output": qkv.name,
                "core": [node.name for node in core],
                "first_matmul": first.name,
                "softmax": softmax.name,
                "second_matmul": second.name,
                "dtypes": {
                    "qkv_output": "S16",
                    "first_matmul_inputs": first_inputs,
                    "first_matmul_output": "S16",
                    "softmax_output": "S16",
                    "second_matmul_inputs": second_inputs,
                    "second_matmul_output": "U16",
                },
            }
        )
    if len(regions) != expected:
        raise RuntimeError(f"Expected {expected} SVTR attention regions, found {len(regions)}")
    return regions


def qparam(index: GraphIndex, node: onnx.NodeProto) -> tuple[Any, ...] | None:
    if node.op_type not in QDQ_OPS or len(node.input) < 3:
        return None
    scale = index.initializers.get(node.input[1])
    zero_point = index.initializers.get(node.input[2])
    if scale is None or zero_point is None:
        return None
    axis = next((int(attribute.i) for attribute in node.attribute if attribute.name == "axis"), None)
    scale_array = numpy_helper.to_array(scale)
    zero_point_array = numpy_helper.to_array(zero_point)
    return (
        int(zero_point.data_type),
        axis,
        scale_array.dtype.str,
        scale_array.shape,
        scale_array.tobytes(),
        zero_point_array.dtype.str,
        zero_point_array.shape,
        zero_point_array.tobytes(),
    )


def scalar_value(index: GraphIndex, tensor_name: str) -> float | int | list[float] | list[int] | None:
    initializer = index.initializers.get(tensor_name)
    if initializer is None:
        return None
    value = np.asarray(numpy_helper.to_array(initializer))
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return value.reshape(-1).tolist()


def discover_requants(index: GraphIndex) -> list[dict[str, Any]]:
    requants: list[dict[str, Any]] = []
    for quantize in (node for node in index.nodes if node.op_type == "QuantizeLinear"):
        producer = index.producers.get(quantize.input[0])
        identity = None
        if producer is not None and producer.op_type == "Identity":
            identity = producer
            producer = index.producers.get(producer.input[0])
        if producer is None or producer.op_type != "DequantizeLinear":
            continue
        if qparam(index, producer) == qparam(index, quantize):
            continue
        requants.append(
            {
                "dequantize": producer.name,
                "identity": identity.name if identity is not None else None,
                "quantize": quantize.name,
                "from": {
                    "dtype": index.qdtype(producer),
                    "scale": scalar_value(index, producer.input[1]),
                    "zero_point": scalar_value(index, producer.input[2]),
                },
                "to": {
                    "dtype": index.qdtype(quantize),
                    "scale": scalar_value(index, quantize.input[1]),
                    "zero_point": scalar_value(index, quantize.input[2]),
                },
            }
        )
    return requants


def discover_silu(index: GraphIndex) -> dict[str, Any]:
    patterns: list[dict[str, Any]] = []
    for sigmoid in (node for node in index.nodes if node.op_type == "Sigmoid"):
        if not sigmoid.input or not sigmoid.output:
            continue
        source = sigmoid.input[0]
        queue = deque(index.consumers.get(sigmoid.output[0], []))
        visited: set[str] = set()
        mul = None
        crossed_qdq = False
        while queue:
            node = queue.popleft()
            if node.name in visited:
                continue
            visited.add(node.name)
            if node.op_type == "Mul" and source in node.input:
                mul = node
                break
            if node.op_type in QDQ_OPS:
                crossed_qdq = True
                for output_name in node.output:
                    queue.extend(index.consumers.get(output_name, []))
        if mul is None:
            continue
        input_dequantize = index.producers.get(source)
        output_quantizes = [
            node
            for node in index.consumers.get(mul.output[0], [])
            if node.op_type == "QuantizeLinear"
        ]
        boundary_quantized = (
            input_dequantize is not None
            and input_dequantize.op_type == "DequantizeLinear"
            and len(output_quantizes) == 1
            and not crossed_qdq
        )
        patterns.append(
            {
                "sigmoid": sigmoid.name,
                "mul": mul.name,
                "input_dequantize": (
                    input_dequantize.name
                    if input_dequantize is not None
                    and input_dequantize.op_type == "DequantizeLinear"
                    else None
                ),
                "output_quantize": (
                    output_quantizes[0].name if len(output_quantizes) == 1 else None
                ),
                "boundary_quantized": boundary_quantized,
                "internal_qdq": crossed_qdq,
            }
        )
    return {
        "total": len(patterns),
        "quantized": sum(pattern["boundary_quantized"] for pattern in patterns),
        "internal_qdq": sum(pattern["internal_qdq"] for pattern in patterns),
        "patterns": patterns,
    }


def inspect_model(
    model_path: Path,
    expected_input_shape: list[int],
    expected_classes: int,
    expected_attention: int,
    expected_requant: int,
    expected_silu: int,
) -> dict[str, Any]:
    model = onnx.load(model_path)
    onnx.checker.check_model(model, full_check=True)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError("PP-OCRv5 rec deployment graph must have one input and one output")
    input_info = tensor_contract(model.graph.input[0])
    output_info = tensor_contract(model.graph.output[0])
    if input_info["dtype"] != "FLOAT" or input_info["shape"] != expected_input_shape:
        raise RuntimeError(
            f"Expected FP32 input shape {expected_input_shape}, got {input_info['dtype']} {input_info['shape']}"
        )
    if output_info["dtype"] != "FLOAT" or len(output_info["shape"]) != 3:
        raise RuntimeError(f"Expected rank-3 FP32 CTC logits, got {output_info}")
    if output_info["shape"][-1] != expected_classes:
        raise RuntimeError(f"Expected {expected_classes} CTC classes, got {output_info['shape'][-1]}")
    index = GraphIndex(model)
    attention = discover_attention(index, expected_attention)
    requants = discover_requants(index)
    if len(requants) != expected_requant:
        raise RuntimeError(f"Expected {expected_requant} necessary requants, found {len(requants)}")
    silu = discover_silu(index)
    if silu["total"] != expected_silu:
        raise RuntimeError(f"Expected {expected_silu} SiLU patterns, found {silu['total']}")
    if silu["quantized"] != expected_silu or silu["internal_qdq"]:
        raise RuntimeError(
            "SiLU patterns must use boundary-only QDQ: "
            f"{silu['quantized']} / {silu['total']} quantized, "
            f"{silu['internal_qdq']} with internal QDQ"
        )
    zero_point_counts = Counter(
        index.qdtype(node)
        for node in index.nodes
        if node.op_type == "QuantizeLinear" and index.qdtype(node) is not None
    )
    return {
        "path": str(model_path.resolve()),
        "sha256": sha256(model_path),
        "inputs": [input_info],
        "outputs": [output_info],
        "nodes": len(index.nodes),
        "quantize_linear": sum(node.op_type == "QuantizeLinear" for node in index.nodes),
        "dequantize_linear": sum(node.op_type == "DequantizeLinear" for node in index.nodes),
        "activation_zero_point_dtypes": dict(sorted(zero_point_counts.items())),
        "attention": attention,
        "requants": requants,
        "silu": silu,
    }


def build_config(args: argparse.Namespace, inspection: dict[str, Any]) -> dict[str, Any]:
    regions = inspection["attention"]
    qkv = [region["qkv_output"] for region in regions]
    core = [name for region in regions for name in region["core"]]
    second = [region["second_matmul"] for region in regions]
    return {
        "input": str(args.onnx),
        "output_dir": args.output_dir or f"./output_{args.onnx.stem}",
        "model_type": "QuantONNX",
        "target_hardware": args.target_hardware,
        "npu_mode": args.npu_mode,
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "DEFAULT",
                    "calibration_dataset": args.calibration_dataset,
                }
            ],
            "layer_configs": [
                {"layer_names": qkv, "output_data_type": "S16"},
                {"layer_names": core, "data_type": "S16", "output_data_type": "S16"},
                {"layer_names": second, "data_type": "S16"},
            ],
            "conv_bias_data_type": "FP32",
            "precision_analysis": True,
            "precision_analysis_method": "PerLayer",
            "precision_analysis_mode": "NPUBackend",
        },
        "input_processors": [
            {
                "tensor_name": "DEFAULT",
                "tensor_layout": "NCHW",
                "src_layout": "NCHW",
                "src_dtype": "FP32",
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            }
        ],
        "output_processors": [],
        "compiler": {"check": 2},
    }


def main() -> None:
    args = parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX does not exist: {args.onnx}")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing config: {args.output}")
    report_path = args.report or Path(f"{args.output}.report.json")
    if report_path.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {report_path}")
    inspection = inspect_model(
        args.onnx,
        args.expected_input_shape,
        args.expected_classes,
        args.expected_attention,
        args.expected_requant,
        args.expected_silu,
    )
    config = build_config(args, inspection)
    report = {
        "schema_version": 1,
        "generator": "ppocrv5-rec-pulsar2-config",
        "onnx": inspection,
        "pulsar2": {
            "config": str(args.output.resolve()),
            "config_sha256": json_sha256(config),
            "target_hardware": args.target_hardware,
            "npu_mode": args.npu_mode,
        },
        "preprocessing": {
            "location": "host",
            "resize": "preserve aspect ratio to height 48",
            "padding": "right zero padding to width 320",
            "normalization": "[-1, 1] before Pulsar2 FP32/NCHW identity processor",
            "channel_order": "BGR (OpenCV route2 preprocessing contract)",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "config": str(args.output),
                "report": str(report_path),
                "onnx_sha256": inspection["sha256"],
                "attention_regions": len(inspection["attention"]),
                "requants": len(inspection["requants"]),
                "silu": inspection["silu"]["total"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

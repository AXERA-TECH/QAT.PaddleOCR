#!/usr/bin/env python3
"""PP-OCRv5-rec profile for the generic Pulsar2 config skill."""

from __future__ import annotations

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


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            param = dimension.dim_param
            result.append(int(param) if param.isdigit() else param)
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


def discover_attention(index: GraphIndex, expected: int, attention_dtype: str = "S16") -> list[dict[str, Any]]:
    unsigned_output = "U8" if attention_dtype == "S8" else "U16"
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
                f"Unsupported operators in {softmax.name} {attention_dtype} region: "
                f"{[(node.name, node.op_type) for node in unsupported]}"
            )
        if first.name not in {node.name for node in core} or softmax.name not in {node.name for node in core}:
            raise RuntimeError(f"Incomplete {attention_dtype} core for {softmax.name}")
        require_output_dtype(index, qkv, attention_dtype)
        for node in core:
            require_output_dtype(index, node, attention_dtype)
        first_inputs = [index.input_qdtype(first, position) for position in range(2)]
        second_inputs = [index.input_qdtype(second, position) for position in range(2)]
        if first_inputs != [attention_dtype, attention_dtype]:
            raise RuntimeError(
                f"{first.name} inputs must be {attention_dtype}/{attention_dtype}, got {first_inputs}"
            )
        if second_inputs != [attention_dtype, attention_dtype]:
            raise RuntimeError(
                f"{second.name} inputs must be {attention_dtype}/{attention_dtype}, got {second_inputs}"
            )
        require_output_dtype(index, second, unsigned_output)
        regions.append(
            {
                "qkv_output": qkv.name,
                "core": [node.name for node in core],
                "first_matmul": first.name,
                "softmax": softmax.name,
                "second_matmul": second.name,
                "dtypes": {
                    "qkv_output": attention_dtype,
                    "first_matmul_inputs": first_inputs,
                    "first_matmul_output": attention_dtype,
                    "softmax_output": attention_dtype,
                    "second_matmul_inputs": second_inputs,
                    "second_matmul_output": unsigned_output,
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
    attention_dtype: str = "S16",
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
    attention = discover_attention(index, expected_attention, attention_dtype)
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


def build_config(args: Any, inspection: dict[str, Any]) -> dict[str, Any]:
    regions = inspection["attention"]
    qkv = [region["qkv_output"] for region in regions]
    core = [name for region in regions for name in region["core"]]
    second = [region["second_matmul"] for region in regions]
    attention_dtype = args.attention_dtype
    return {
        "input": str(args.onnx.resolve()),
        "output_dir": str(
            (args.output_dir or Path(f"./output_{args.onnx.stem}")).resolve()
        ),
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
                {"layer_names": qkv, "output_data_type": attention_dtype},
                {
                    "layer_names": core,
                    "data_type": attention_dtype,
                    "output_data_type": attention_dtype,
                },
                {"layer_names": second, "data_type": attention_dtype},
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _find_rule(
    rules: list[dict[str, Any]], names: list[str], role: str
) -> dict[str, Any]:
    expected = set(names)
    matches = [rule for rule in rules if set(rule.get("layer_names", [])) == expected]
    _require(
        len(matches) == 1,
        f"Expected one {role} rule for {names}, found {len(matches)}",
    )
    return matches[0]


def validate_profile_config(
    config: dict[str, Any],
    inspection: dict[str, Any],
    target_hardware: str | None = None,
    npu_mode: str | None = None,
    attention_dtype: str = "S16",
) -> None:
    """Validate the v5-rec Attention contract after generic validation."""
    _require(config.get("model_type") == "QuantONNX", "model_type must be QuantONNX")
    if target_hardware is not None:
        _require(
            config.get("target_hardware") == target_hardware,
            f"target_hardware must be {target_hardware}",
        )
    if npu_mode is not None:
        _require(config.get("npu_mode") == npu_mode, f"npu_mode must be {npu_mode}")
    _require(config.get("compiler") == {"check": 2}, "compiler.check must be 2")
    _require(config.get("output_processors") == [], "output_processors must be empty")
    quant = config.get("quant", {})
    input_configs = quant.get("input_configs", [])
    _require(
        len(input_configs) == 1,
        "Exactly one quant.input_configs item is required",
    )
    _require(
        bool(input_configs[0].get("calibration_dataset")),
        "calibration_dataset field is required by Pulsar2",
    )
    _require(
        config.get("input_processors") == [
            {
                "tensor_name": "DEFAULT",
                "tensor_layout": "NCHW",
                "src_layout": "NCHW",
                "src_dtype": "FP32",
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            }
        ],
        "Input processor must remain FP32/NCHW identity",
    )
    regions = inspection["attention"]
    qkv_names = [region["qkv_output"] for region in regions]
    core_names = [name for region in regions for name in region["core"]]
    second_names = [region["second_matmul"] for region in regions]
    rules = quant.get("layer_configs", [])
    _require(len(rules) == 3, "Exactly three generated layer_configs groups are required")
    qkv_rule = _find_rule(rules, qkv_names, f"QKV output {attention_dtype}")
    core_rule = _find_rule(rules, core_names, f"attention core {attention_dtype}")
    second_rule = _find_rule(rules, second_names, f"second MatMul {attention_dtype} input")
    _require(
        qkv_rule == {"layer_names": qkv_names, "output_data_type": attention_dtype},
        f"Invalid QKV {attention_dtype} rule",
    )
    _require(
        core_rule
        == {
            "layer_names": core_names,
            "data_type": attention_dtype,
            "output_data_type": attention_dtype,
        },
        f"Invalid attention core {attention_dtype} rule",
    )
    _require(
        second_rule == {"layer_names": second_names, "data_type": attention_dtype},
        f"Invalid second MatMul {attention_dtype} rule",
    )

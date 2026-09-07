#!/usr/bin/env python3
"""Verify an ONNX QDQ model against the repository ax quant spec."""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import onnx
from onnx import TensorProto, numpy_helper
from rich.console import Console
from rich.table import Table


PASSIVE_OPS = {
    "DepthToSpace",
    "Expand",
    "Flatten",
    "Gather",
    "Identity",
    "Reshape",
    "Slice",
    "SpaceToDepth",
    "Split",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
    "MaxPool",
    "ReduceMax",
    "Resize",
    "Tile",
    "GatherElements",
    "GatherND",
    "ReduceMin",
    "GlobalMaxPool",
}

NON_QUANT_OPS = {
    "Abs",
    "Ceil",
    "Floor",
    "Neg",
    "Round",
    "And",
    "ArgMax",
    "ArgMin",
    "Cast",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Not",
}

UNSUPPORTED_OPS = {"AffineGrid", "MaxRoiPool", "GRU", "LSTM"}

QUANT_UNARY_OPS = {
    "Cos",
    "Elu",
    "Erf",
    "Exp",
    "HardSigmoid",
    "HardSwish",
    "Log",
    "Mish",
    "Sin",
    "Softplus",
    "Sqrt",
    "Tanh",
    "Reciprocal",
    "Sigmoid",
    "Gelu",
    "LeakyRelu",
    "Softmax",
    "AveragePool",
    "GlobalAveragePool",
    "LayerNormalization",
    "GroupNormalization",
    "InstanceNormalization",
    "BatchNormalization",
    "LogSoftmax",
    "LpNormalization",
    "CumSum",
    "ReduceL2",
    "ReduceMean",
    "ReduceSum",
}

QUANT_ARITHMETIC_OPS = {"Add", "Sub", "Mul", "Div", "Max", "Min"}

QUANT_SPECIAL_OPS = {
    "Conv",
    "ConvTranspose",
    "Gemm",
    "MatMul",
    "GridSample",
    "Pow",
    "PRelu",
    "RoiAlign",
    "ScatterElements",
    "ScatterND",
    "Where",
    "Relu",
    "Clip",
    "Concat",
    "Pad",
    "TopK",
}

SUPPORTED_OPS = (
    PASSIVE_OPS
    | NON_QUANT_OPS
    | UNSUPPORTED_OPS
    | QUANT_UNARY_OPS
    | QUANT_ARITHMETIC_OPS
    | QUANT_SPECIAL_OPS
    | {"QuantizeLinear", "DequantizeLinear", "Constant"}
)

INTEGER_TYPES = {
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
}

QPARAM_TYPES = {
    TensorProto.UINT4,
    TensorProto.INT4,
    TensorProto.UINT8,
    TensorProto.INT8,
    TensorProto.UINT16,
    TensorProto.INT16,
    TensorProto.INT32,
}

ACTIVATION_TYPES = {
    TensorProto.UINT4,
    TensorProto.INT8,
    TensorProto.UINT8,
    TensorProto.INT16,
    TensorProto.UINT16,
}

WEIGHT_TYPES = {
    TensorProto.INT4,
    TensorProto.INT8,
    TensorProto.INT16,
}

TYPE_NAMES = {
    TensorProto.FLOAT: "FP32",
    TensorProto.UINT4: "U4",
    TensorProto.INT4: "S4",
    TensorProto.UINT8: "U8",
    TensorProto.INT8: "S8",
    TensorProto.UINT16: "U16",
    TensorProto.INT16: "S16",
    TensorProto.INT32: "S32",
}

CAST_FUNCTIONS = {
    TensorProto.FLOAT: float,
    TensorProto.DOUBLE: float,
    TensorProto.INT8: int,
    TensorProto.INT16: int,
    TensorProto.INT32: int,
    TensorProto.INT64: int,
    TensorProto.UINT8: int,
    TensorProto.UINT16: int,
    TensorProto.UINT32: int,
    TensorProto.UINT64: int,
    TensorProto.BOOL: bool,
}


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    message: str
    location: str


@dataclass
class TensorData:
    data_type: Optional[int]
    shape: Optional[Tuple[Optional[int], ...]]
    values: Optional[List[Any]]


@dataclass
class QuantInfo:
    node_index: int
    op_type: str
    scale_name: Optional[str]
    zero_point_name: Optional[str]
    scale: TensorData
    zero_point: TensorData
    axis_present: bool
    axis: int
    normalized_axis: int
    axis_size: Optional[int]
    granularity: Optional[str]

    @property
    def storage_type(self) -> Optional[int]:
        return self.zero_point.data_type


@dataclass
class Pattern:
    kind: str
    nodes: List[int]
    inputs: List[str]
    outputs: List[str]


def fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def get_attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return default


def tensor_shape(value_info: onnx.ValueInfoProto) -> Tuple[Optional[int], ...]:
    return tuple(
        int(dimension.dim_value) if dimension.HasField("dim_value") else None
        for dimension in value_info.type.tensor_type.shape.dim
    )


def clean_text(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").replace("|", "/").strip()


def checker_result(model_path: Path) -> subprocess.CompletedProcess[str]:
    code = (
        "import onnx,sys;"
        "model=onnx.load(sys.argv[1]);"
        "onnx.checker.check_model(model)"
    )
    return subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def load_result(model_path: Path) -> subprocess.CompletedProcess[str]:
    code = "import onnx,sys;onnx.load(sys.argv[1])"
    return subprocess.run(
        [sys.executable, "-c", code, str(model_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def subprocess_error(result: subprocess.CompletedProcess[str]) -> str:
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    return clean_text(lines[-1]) if lines else "The ONNX subprocess failed."


class Verifier:
    def __init__(self, model: onnx.ModelProto, check_value: bool):
        self.model = model
        self.graph = model.graph
        self.nodes = list(self.graph.node)
        self.check_value = check_value
        self.initializers = {item.name: item for item in self.graph.initializer}
        self.graph_inputs = {item.name for item in self.graph.input}
        self.graph_outputs = {item.name for item in self.graph.output}
        self.producers: Dict[str, Tuple[int, onnx.NodeProto]] = {}
        self.consumers: Dict[str, List[Tuple[int, onnx.NodeProto, int]]] = defaultdict(list)
        self.value_types: Dict[str, int] = {}
        self.value_shapes: Dict[str, Tuple[Optional[int], ...]] = {}
        self.tensor_cache: Dict[str, TensorData] = {}
        self.quant_cache: Dict[int, QuantInfo] = {}
        self.q_to_dq: Dict[int, int] = {}
        self.dq_to_q: Dict[int, int] = {}
        self.findings: List[Finding] = []
        self.pattern_nodes: Set[int] = set()
        self.patterns: List[Pattern] = []
        self.passive_origins: Dict[str, QuantInfo] = {}
        self.passive_split_sources: Dict[str, int] = {}
        self.split_inputs: Dict[int, QuantInfo] = {}
        self.split_boundaries: Dict[int, List[Tuple[QuantInfo, str]]] = defaultdict(
            list
        )
        self._index_graph()
        self._collect_value_metadata()
        self._propagate_value_types()

    def _index_graph(self) -> None:
        for index, node in enumerate(self.nodes):
            for output in node.output:
                if output:
                    self.producers[output] = (index, node)
            for input_index, input_name in enumerate(node.input):
                if input_name:
                    self.consumers[input_name].append((index, node, input_index))

    def _collect_value_metadata(self) -> None:
        for name, initializer in self.initializers.items():
            self.value_types[name] = initializer.data_type
            self.value_shapes[name] = tuple(int(value) for value in initializer.dims)
        value_infos = list(self.graph.input) + list(self.graph.output) + list(self.graph.value_info)
        for value_info in value_infos:
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type:
                self.value_types[value_info.name] = tensor_type.elem_type
                self.value_shapes[value_info.name] = tensor_shape(value_info)

    def _propagate_value_types(self) -> None:
        passthrough = PASSIVE_OPS | QUANT_ARITHMETIC_OPS | {
            "Concat",
            "Pad",
            "TopK",
            "Relu",
            "Clip",
        }
        for node in self.nodes:
            if not node.output:
                continue
            output_type = None
            if node.op_type == "Cast":
                output_type = int(get_attribute(node, "to", 0))
            elif node.op_type == "QuantizeLinear" and len(node.input) > 2:
                output_type = self.tensor_data(node.input[2]).data_type
            elif node.op_type == "DequantizeLinear":
                output_type = TensorProto.FLOAT
            elif node.op_type in passthrough and node.input:
                output_type = self.value_types.get(node.input[0])
            elif node.op_type in QUANT_UNARY_OPS | QUANT_SPECIAL_OPS:
                output_type = TensorProto.FLOAT
            if output_type:
                for output in node.output:
                    if output:
                        self.value_types[output] = output_type
            if node.input:
                input_shape = self.value_shapes.get(node.input[0])
                if input_shape is not None and node.op_type in {"Cast", "Identity"}:
                    for output in node.output:
                        if output:
                            self.value_shapes[output] = input_shape

    def add(self, severity: str, rule: str, message: str, location: str) -> None:
        self.findings.append(
            Finding(severity, rule, clean_text(message), clean_text(location))
        )

    def label(self, node_index: int) -> str:
        node = self.nodes[node_index]
        return node.name if node.name else f"{node.op_type}[{node_index}]"

    def tensor_data(self, name: str) -> TensorData:
        cached = self.tensor_cache.get(name)
        if cached is not None:
            return cached
        initializer = self.initializers.get(name)
        if initializer is not None:
            array = numpy_helper.to_array(initializer)
            data = TensorData(
                initializer.data_type,
                tuple(int(value) for value in initializer.dims),
                array.reshape(-1).tolist(),
            )
            self.tensor_cache[name] = data
            return data
        producer = self.producers.get(name)
        if producer is not None and producer[1].op_type in {"Cast", "Identity"}:
            node = producer[1]
            source = self.tensor_data(node.input[0])
            if node.op_type == "Identity":
                data = TensorData(source.data_type, source.shape, source.values)
            else:
                target_type = int(get_attribute(node, "to", 0))
                converter = CAST_FUNCTIONS.get(target_type)
                values = None
                if source.values is not None and converter is not None:
                    values = [converter(value) for value in source.values]
                data = TensorData(target_type or None, source.shape, values)
            self.tensor_cache[name] = data
            return data
        data = TensorData(
            self.value_types.get(name),
            self.value_shapes.get(name),
            None,
        )
        self.tensor_cache[name] = data
        return data

    def data_shape_for_quant_node(self, node: onnx.NodeProto) -> Optional[Tuple[Optional[int], ...]]:
        data_name = node.input[0] if node.input else ""
        if node.op_type == "DequantizeLinear":
            producer = self.producers.get(data_name)
            if producer is not None and producer[1].op_type == "QuantizeLinear":
                data_name = producer[1].input[0]
        return self.tensor_data(data_name).shape

    def quant_info(self, node_index: int) -> QuantInfo:
        cached = self.quant_cache.get(node_index)
        if cached is not None:
            return cached
        node = self.nodes[node_index]
        scale_name = node.input[1] if len(node.input) > 1 and node.input[1] else None
        zero_point_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
        scale = self.tensor_data(scale_name) if scale_name else TensorData(None, None, None)
        zero_point = (
            self.tensor_data(zero_point_name)
            if zero_point_name
            else TensorData(None, None, None)
        )
        axis_present = any(attribute.name == "axis" for attribute in node.attribute)
        axis = int(get_attribute(node, "axis", 1))
        data_shape = self.data_shape_for_quant_node(node)
        normalized_axis = axis
        if data_shape is not None and normalized_axis < 0:
            normalized_axis += len(data_shape)
        axis_size = None
        if data_shape is not None and 0 <= normalized_axis < len(data_shape):
            axis_size = data_shape[normalized_axis]
        granularity = None
        if scale.shape == () and zero_point.shape == ():
            granularity = "per-tensor"
        elif (
            scale.shape is not None
            and zero_point.shape is not None
            and len(scale.shape) == 1
            and len(zero_point.shape) == 1
            and scale.shape == zero_point.shape
            and scale.shape[0] is not None
        ):
            size = int(scale.shape[0])
            if size > 1:
                granularity = "per-channel"
            elif axis_present and axis_size == 1:
                granularity = "per-channel"
            else:
                granularity = "per-tensor"
        info = QuantInfo(
            node_index,
            node.op_type,
            scale_name,
            zero_point_name,
            scale,
            zero_point,
            axis_present,
            axis,
            normalized_axis,
            axis_size,
            granularity,
        )
        self.quant_cache[node_index] = info
        return info

    def tensor_structure_equal(self, left: TensorData, right: TensorData) -> bool:
        return left.shape == right.shape and left.data_type == right.data_type

    def tensor_values_equal(self, left: TensorData, right: TensorData) -> bool:
        if not self.tensor_structure_equal(left, right):
            return False
        if left.values is None or right.values is None:
            return False
        return left.values == right.values

    def signatures_structurally_equal(
        self, left: QuantInfo, right: QuantInfo
    ) -> bool:
        if left.storage_type != right.storage_type:
            return False
        if left.granularity != right.granularity:
            return False
        if (
            left.granularity == "per-channel"
            and left.normalized_axis != right.normalized_axis
        ):
            return False
        return self.tensor_structure_equal(
            left.scale, right.scale
        ) and self.tensor_structure_equal(left.zero_point, right.zero_point)

    def signatures_equal(self, left: QuantInfo, right: QuantInfo) -> bool:
        if not self.signatures_structurally_equal(left, right):
            return False
        scale_equal = self.tensor_values_equal(left.scale, right.scale)
        if left.zero_point_name == right.zero_point_name and left.zero_point_name is not None:
            zero_point_equal = True
        else:
            zero_point_equal = self.tensor_values_equal(left.zero_point, right.zero_point)
        return scale_equal and zero_point_equal

    def quant_info_from_dq_output(self, tensor_name: str) -> Optional[QuantInfo]:
        producer = self.producers.get(tensor_name)
        if producer is not None and producer[1].op_type == "DequantizeLinear":
            return self.quant_info(producer[0])
        return None

    def quant_info_from_q_input(self, tensor_name: str) -> Optional[QuantInfo]:
        for node_index, node, input_index in self.consumers.get(tensor_name, []):
            if node.op_type == "QuantizeLinear" and input_index == 0:
                dq_index = self.q_to_dq.get(node_index)
                if dq_index is not None:
                    return self.quant_info(dq_index)
        return None

    def paired_quant_info(self, q_index: int) -> Optional[QuantInfo]:
        dq_index = self.q_to_dq.get(q_index)
        return self.quant_info(dq_index) if dq_index is not None else None

    def is_quantized_input(self, tensor_name: str) -> bool:
        return self.quant_info_from_dq_output(tensor_name) is not None

    def all_zero(self, values: Optional[List[Any]]) -> bool:
        return values is not None and all(float(value) == 0.0 for value in values)

    def check_zero_point_range(self, info: QuantInfo, location: str) -> None:
        if not self.check_value or info.zero_point.values is None:
            return
        limits = {
            TensorProto.UINT4: (0, 15),
            TensorProto.INT4: (-8, 7),
            TensorProto.UINT8: (0, 255),
            TensorProto.INT8: (-128, 127),
            TensorProto.UINT16: (0, 65535),
            TensorProto.INT16: (-32768, 32767),
            TensorProto.INT32: (-2147483648, 2147483647),
        }
        limit = limits.get(info.storage_type)
        if limit is not None and any(
            int(value) < limit[0] or int(value) > limit[1]
            for value in info.zero_point.values
        ):
            self.add(
                "warn",
                "3.2",
                "zero_point contains a value outside the storage type range.",
                location,
            )

    def check_zero_point_zero(
        self,
        info: QuantInfo,
        rule: str,
        message: str,
        location: str,
    ) -> None:
        if self.check_value and info.zero_point.values is not None:
            if not self.all_zero(info.zero_point.values):
                self.add("warn", rule, message, location)

    def check_model_scope(self, checker: subprocess.CompletedProcess[str]) -> None:
        if checker.returncode != 0:
            self.add("error", "2.2", subprocess_error(checker), "graph")
        has_qdq = any(
            node.op_type in {"QuantizeLinear", "DequantizeLinear"}
            for node in self.nodes
        )
        if not has_qdq:
            self.add(
                "error",
                "2.1",
                "The model has no QDQ edge owned by an operator rule or pattern.",
                "graph",
            )
        for index, node in enumerate(self.nodes):
            if node.domain not in {"", "ai.onnx"}:
                self.add(
                    "error",
                    "2.3",
                    f"Operator domain '{node.domain}' is not covered by the spec.",
                    self.label(index),
                )
            if node.op_type == "Constant":
                self.add(
                    "error",
                    "2.2",
                    "Static constants must use initializers instead of Constant nodes.",
                    self.label(index),
                )
            elif node.op_type not in SUPPORTED_OPS:
                self.add(
                    "error",
                    "2.3",
                    f"Operator '{node.op_type}' is not covered by an operator rule or pattern.",
                    self.label(index),
                )
            if node.op_type in UNSUPPORTED_OPS:
                self.add(
                    "error",
                    "5.17",
                    f"Operator '{node.op_type}' is not accepted by the current spec.",
                    self.label(index),
                )

    def check_qdq(self) -> None:
        for index, node in enumerate(self.nodes):
            if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
                continue
            info = self.quant_info(index)
            location = self.label(index)
            if info.scale_name not in self.initializers:
                self.add(
                    "error",
                    "3.1",
                    "scale must be a directly referenced initializer.",
                    location,
                )
            if info.zero_point_name is None:
                self.add(
                    "error",
                    "3.1",
                    "zero_point must be explicitly present.",
                    location,
                )
            if info.scale.data_type not in {None, TensorProto.FLOAT}:
                self.add("error", "3.1", "scale must use FP32.", location)
            if info.scale.values is not None and any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in info.scale.values
            ):
                self.add(
                    "error",
                    "3.1",
                    "scale values must be finite and strictly positive.",
                    location,
                )
            if info.storage_type is not None and info.storage_type not in QPARAM_TYPES:
                self.add(
                    "error",
                    "3.2",
                    "zero_point uses an unsupported storage type.",
                    location,
                )
            if info.granularity is None:
                self.add(
                    "error",
                    "3.3",
                    "scale and zero_point do not form per-tensor or per-channel parameters.",
                    location,
                )
            if info.granularity == "per-channel" and not info.axis_present:
                self.add(
                    "error",
                    "3.3",
                    "Per-channel quantization must explicitly set axis.",
                    location,
                )
            if (
                info.granularity == "per-channel"
                and info.axis_size is not None
                and info.scale.shape is not None
                and info.axis_size != info.scale.shape[0]
            ):
                self.add(
                    "error",
                    "2.2",
                    "The per-channel parameter length does not match the data axis.",
                    location,
                )
            self.check_zero_point_range(info, location)
            if node.op_type == "QuantizeLinear":
                input_type = self.value_types.get(node.input[0]) if node.input else None
                if input_type not in {None, TensorProto.FLOAT}:
                    self.add(
                        "error",
                        "3.1",
                        "QuantizeLinear data input must use FP32.",
                        location,
                    )
                output = node.output[0] if node.output else ""
                consumers = self.consumers.get(output, [])
                if output in self.graph_outputs:
                    self.add(
                        "error",
                        "3.4",
                        "QuantizeLinear output cannot be a graph output.",
                        location,
                    )
                if (
                    len(consumers) != 1
                    or consumers[0][1].op_type != "DequantizeLinear"
                    or consumers[0][2] != 0
                ):
                    names = ", ".join(self.label(item[0]) for item in consumers)
                    names = names if names else "none"
                    self.add(
                        "error",
                        "3.4",
                        f"Q output must have one paired DQ consumer; actual consumers: {names}.",
                        location,
                    )
                else:
                    dq_index = consumers[0][0]
                    self.q_to_dq[index] = dq_index
                    self.dq_to_q[dq_index] = index
            else:
                data_name = node.input[0] if node.input else ""
                producer = self.producers.get(data_name)
                valid_source = (
                    data_name in self.initializers
                    or (
                        producer is not None
                        and producer[1].op_type == "QuantizeLinear"
                    )
                )
                if not valid_source:
                    self.add(
                        "error",
                        "3.4",
                        "DQ integer data must come from a paired Q or an allowed initializer.",
                        location,
                    )
                output = node.output[0] if node.output else ""
                if not self.consumers.get(output) and output not in self.graph_outputs:
                    self.add(
                        "error",
                        "2.3",
                        "DQ output has no rule owner.",
                        location,
                    )

        for q_index, dq_index in self.q_to_dq.items():
            q_info = self.quant_info(q_index)
            dq_info = self.quant_info(dq_index)
            location = f"{self.label(q_index)} -> {self.label(dq_index)}"
            if not self.signatures_structurally_equal(q_info, dq_info):
                self.add(
                    "error",
                    "3.4",
                    "Paired Q and DQ parameter structures are not semantically identical.",
                    location,
                )
                continue
            if not self.check_value:
                continue
            if (
                q_info.scale.values is not None
                and dq_info.scale.values is not None
                and q_info.scale.values != dq_info.scale.values
            ):
                self.add(
                    "warn",
                    "3.4",
                    "Paired Q and DQ scale values are not identical.",
                    location,
                )
            if (
                q_info.zero_point_name != dq_info.zero_point_name
                and q_info.zero_point.values is not None
                and dq_info.zero_point.values is not None
                and q_info.zero_point.values != dq_info.zero_point.values
            ):
                self.add(
                    "warn",
                    "3.4",
                    "Paired Q and DQ zero_point values are not identical.",
                    location,
                )

        for q_index, node in enumerate(self.nodes):
            if node.op_type != "QuantizeLinear" or q_index not in self.q_to_dq:
                continue
            producer = self.producers.get(node.input[0])
            if producer is None or producer[1].op_type != "DequantizeLinear":
                continue
            upstream = self.quant_info(producer[0])
            downstream = self.quant_info(self.q_to_dq[q_index])
            if not self.signatures_equal(upstream, downstream):
                self.add(
                    "error",
                    "3.5",
                    "Different consecutive QDQ parameters require a requantize Identity.",
                    f"{self.label(producer[0])} -> {self.label(q_index)}",
                )

    def quant_roles(self) -> Dict[int, str]:
        roles: Dict[int, str] = {}
        for index, node in enumerate(self.nodes):
            if node.op_type != "DequantizeLinear":
                continue
            output = node.output[0]
            detected: Set[str] = set()
            for _, consumer, input_index in self.consumers.get(output, []):
                if consumer.op_type in {"Conv", "ConvTranspose", "Gemm"}:
                    if input_index == 1:
                        detected.add("weight")
                    elif input_index == 2:
                        detected.add("bias")
                    else:
                        detected.add("activation")
                else:
                    detected.add("activation")
            if output in self.graph_outputs:
                detected.add("activation")
            info = self.quant_info(index)
            if detected == {"activation"} and info.granularity == "per-channel":
                # torch.onnx lowers nn.Linear weight as DQ -> Transpose -> MatMul.
                # The passive Transpose hides the weight role from the direct consumer.
                detected = {"weight"}
            if len(detected) > 1:
                self.add(
                    "error",
                    "2.3",
                    "One DQ output is assigned to conflicting quantization roles.",
                    self.label(index),
                )
            roles[index] = next(iter(detected), "activation")
        return roles

    def check_quant_roles(self) -> None:
        roles = self.quant_roles()
        for dq_index, role in roles.items():
            info = self.quant_info(dq_index)
            location = self.label(dq_index)
            storage_name = TYPE_NAMES.get(info.storage_type, str(info.storage_type))
            if role == "activation":
                if info.storage_type is not None and info.storage_type not in ACTIVATION_TYPES:
                    self.add(
                        "error",
                        "4",
                        f"Activation storage type '{storage_name}' is not allowed.",
                        location,
                    )
                if info.granularity not in {None, "per-tensor"}:
                    self.add(
                        "error",
                        "4",
                        "Activation quantization must be per-tensor.",
                        location,
                    )
                if info.storage_type == TensorProto.INT16:
                    self.check_zero_point_zero(
                        info,
                        "4",
                        "Symmetric S16 activation zero_point should be zero.",
                        location,
                    )
            elif role == "weight":
                if info.storage_type is not None and info.storage_type not in WEIGHT_TYPES:
                    self.add(
                        "error",
                        "4",
                        f"Weight storage type '{storage_name}' must be S4, S8, or S16.",
                        location,
                    )
                if info.granularity not in {None, "per-channel"}:
                    self.add(
                        "error",
                        "4",
                        "Weight quantization must be per-channel.",
                        location,
                    )
                self.check_zero_point_zero(
                    info,
                    "4",
                    "Symmetric weight zero_point should be zero.",
                    location,
                )
                for _, consumer, _ in self.consumers.get(
                    self.nodes[dq_index].output[0], []
                ):
                    expected_axis = 1 if consumer.op_type == "ConvTranspose" else 0
                    if (
                        info.granularity == "per-channel"
                        and info.axis != expected_axis
                    ):
                        self.add(
                            "error",
                            "3.3",
                            f"{consumer.op_type} weight axis must be {expected_axis}.",
                            location,
                        )
            else:
                if info.storage_type not in {None, TensorProto.INT32}:
                    self.add(
                        "error",
                        "4",
                        f"Quantized bias storage type '{storage_name}' must be S32.",
                        location,
                    )
                if info.granularity not in {None, "per-channel"}:
                    self.add(
                        "error",
                        "4",
                        "Quantized bias must be per-channel.",
                        location,
                    )
                self.check_zero_point_zero(
                    info,
                    "4",
                    "Symmetric quantized bias zero_point should be zero.",
                    location,
                )

        if not self.check_value:
            return
        for index, node in enumerate(self.nodes):
            if node.op_type not in {"Conv", "ConvTranspose", "Gemm"}:
                continue
            if len(node.input) < 3 or not node.input[2]:
                continue
            input_info = self.quant_info_from_dq_output(node.input[0])
            weight_info = self.quant_info_from_dq_output(node.input[1])
            bias_info = self.quant_info_from_dq_output(node.input[2])
            if input_info is None or weight_info is None or bias_info is None:
                continue
            scales = (
                input_info.scale.values,
                weight_info.scale.values,
                bias_info.scale.values,
            )
            if any(values is None for values in scales):
                continue
            input_scale = float(scales[0][0])
            expected = [
                fp32(fp32(input_scale) * fp32(float(value)))
                for value in scales[1]
            ]
            actual = [fp32(float(value)) for value in scales[2]]
            mismatch = len(expected) != len(actual)
            if not mismatch:
                mismatch = any(
                    abs(left - right) > 1e-7 + 1e-5 * abs(right)
                    for left, right in zip(actual, expected)
                )
            if mismatch:
                self.add(
                    "warn",
                    "4",
                    "bias_scale differs from input_scale * weight_scale beyond the configured thresholds.",
                    self.label(index),
                )

    def mark_patterns(self) -> None:
        for sigmoid_index, sigmoid in enumerate(self.nodes):
            if sigmoid.op_type != "Sigmoid" or not sigmoid.output:
                continue
            output = sigmoid.output[0]
            consumers = self.consumers.get(output, [])
            if len(consumers) != 1 or consumers[0][1].op_type != "Mul":
                continue
            mul_index, mul, _ = consumers[0]
            source = sigmoid.input[0]
            if source in mul.input and output in mul.input:
                self.pattern_nodes.update({sigmoid_index, mul_index})
                self.patterns.append(
                    Pattern(
                        "SiLU",
                        [sigmoid_index, mul_index],
                        [source],
                        [mul.output[0]],
                    )
                )

        for split_index, split in enumerate(self.nodes):
            if split.op_type != "Split":
                continue
            for gated in split.output:
                consumers = self.consumers.get(gated, [])
                if len(consumers) != 1 or consumers[0][1].op_type != "Sigmoid":
                    continue
                sigmoid_index, sigmoid, _ = consumers[0]
                sigmoid_output = sigmoid.output[0]
                sigmoid_consumers = self.consumers.get(sigmoid_output, [])
                if (
                    len(sigmoid_consumers) != 1
                    or sigmoid_consumers[0][1].op_type != "Mul"
                ):
                    continue
                mul_index, mul, _ = sigmoid_consumers[0]
                direct = [
                    item
                    for item in split.output
                    if item != gated and item in mul.input
                ]
                if not direct or sigmoid_output not in mul.input:
                    continue
                if len(self.consumers.get(direct[0], [])) != 1:
                    continue
                owned = {split_index, sigmoid_index, mul_index}
                if owned & self.pattern_nodes:
                    continue
                self.pattern_nodes.update(owned)
                self.patterns.append(
                    Pattern(
                        "GLU",
                        sorted(owned),
                        [split.input[0]],
                        [mul.output[0]],
                    )
                )

        for transpose_index, transpose in enumerate(self.nodes):
            if transpose.op_type != "Transpose" or not transpose.input or not transpose.output:
                continue
            weight_info = self.quant_info_from_dq_output(transpose.input[0])
            if weight_info is None or weight_info.granularity != "per-channel":
                continue
            consumers = self.consumers.get(transpose.output[0], [])
            if len(consumers) != 1 or consumers[0][1].op_type != "MatMul":
                continue
            matmul_index, matmul, weight_input_index = consumers[0]
            if weight_input_index != 1 or len(matmul.input) < 2:
                continue
            matmul_consumers = self.consumers.get(matmul.output[0], [])
            if len(matmul_consumers) != 1 or matmul_consumers[0][1].op_type != "Add":
                continue
            add_index, add, _ = matmul_consumers[0]
            bias_inputs = [name for name in add.input if name != matmul.output[0]]
            if len(bias_inputs) != 1:
                continue
            bias = self.initializers.get(bias_inputs[0])
            if bias is None or bias.data_type != TensorProto.FLOAT:
                continue
            owned = {transpose_index, matmul_index, add_index}
            if owned & self.pattern_nodes:
                continue
            self.pattern_nodes.update(owned)
            self.patterns.append(
                Pattern(
                    "Linear",
                    sorted(owned),
                    [matmul.input[0], transpose.input[0]],
                    [add.output[0]],
                )
            )

        for softmax_index, softmax in enumerate(self.nodes):
            if softmax.op_type != "Softmax":
                continue
            current = softmax.input[0]
            scale_index = None
            producer = self.producers.get(current)
            if producer is not None and producer[1].op_type in {"Mul", "Div"}:
                scale_index = producer[0]
                data_inputs = [
                    name
                    for name in producer[1].input
                    if name not in self.initializers
                ]
                if len(data_inputs) != 1:
                    continue
                current = data_inputs[0]
                producer = self.producers.get(current)
            if producer is None or producer[1].op_type != "MatMul":
                continue
            first_index, first = producer
            consumers = self.consumers.get(softmax.output[0], [])
            if len(consumers) != 1 or consumers[0][1].op_type != "MatMul":
                continue
            second_index, second, softmax_input_index = consumers[0]
            if softmax_input_index != 0:
                continue
            internal_outputs = [first.output[0], softmax.output[0]]
            if scale_index is not None:
                internal_outputs.append(self.nodes[scale_index].output[0])
            if any(len(self.consumers.get(name, [])) != 1 for name in internal_outputs):
                continue
            owned = {first_index, softmax_index, second_index}
            if scale_index is not None:
                owned.add(scale_index)
            if owned & self.pattern_nodes:
                continue
            self.pattern_nodes.update(owned)
            self.patterns.append(
                Pattern(
                    "SDPA",
                    sorted(owned),
                    [first.input[0], first.input[1], second.input[1]],
                    [second.output[0]],
                )
            )

        for batch_index, batch_norm in enumerate(self.nodes):
            if batch_norm.op_type != "BatchNormalization" or not batch_norm.input:
                continue
            producer = self.producers.get(batch_norm.input[0])
            if producer is None or producer[1].op_type not in {
                "Conv",
                "ConvTranspose",
            }:
                continue
            conv_index, conv = producer
            if len(self.consumers.get(conv.output[0], [])) != 1:
                continue
            owned = {conv_index, batch_index}
            if owned & self.pattern_nodes:
                continue
            self.pattern_nodes.update(owned)
            self.patterns.append(
                Pattern(
                    f"{conv.op_type}+BatchNormalization",
                    [conv_index, batch_index],
                    list(conv.input),
                    [batch_norm.output[0]],
                )
            )

    def require_quant_input(
        self, node_index: int, input_index: int, role_name: str = "data input"
    ) -> None:
        node = self.nodes[node_index]
        if input_index >= len(node.input) or not node.input[input_index]:
            return
        input_name = node.input[input_index]
        if not self.is_quantized_input(input_name):
            producer = self.producers.get(input_name)
            source = (
                self.label(producer[0])
                if producer is not None
                else ("initializer" if input_name in self.initializers else "graph input")
            )
            self.add(
                "error",
                "5",
                f"{role_name} must come from a complete QDQ edge.",
                f"{self.label(node_index)} input[{input_index}] from {source}",
            )

    def require_quant_output(self, node_index: int, output_index: int = 0) -> None:
        node = self.nodes[node_index]
        if output_index >= len(node.output) or not node.output[output_index]:
            return
        output = node.output[output_index]
        if not self.consumers.get(output) and output not in self.graph_outputs:
            return
        invalid_destinations = []
        for consumer_index, consumer, input_index in self.consumers.get(output, []):
            if consumer.op_type == "QuantizeLinear" and input_index == 0:
                continue
            if consumer.op_type in {"Relu", "Clip"} and input_index == 0:
                continue
            invalid_destinations.append(self.label(consumer_index))
        if output in self.graph_outputs:
            invalid_destinations.append("graph output")
        if invalid_destinations:
            destinations = ", ".join(invalid_destinations[:4])
            if len(invalid_destinations) > 4:
                destinations += f", plus {len(invalid_destinations) - 4} more"
            self.add(
                "error",
                "5.1",
                "A used quantized output branch has no QDQ edge.",
                f"{self.label(node_index)} output[{output_index}] to {destinations}",
            )
        elif self.quant_info_from_q_input(output) is None and not any(
            consumer.op_type in {"Relu", "Clip"}
            for _, consumer, _ in self.consumers.get(output, [])
        ):
            self.add(
                "error",
                "5.1",
                "A used quantized output has no complete QDQ edge.",
                f"{self.label(node_index)} output[{output_index}]",
            )

    def check_pattern_boundaries(self) -> None:
        for pattern in self.patterns:
            if pattern.kind in {
                "Conv+BatchNormalization",
                "ConvTranspose+BatchNormalization",
            }:
                conv_index, batch_index = pattern.nodes
                conv = self.nodes[conv_index]
                batch_norm = self.nodes[batch_index]
                self.require_quant_input(conv_index, 0, "pattern activation input")
                self.require_quant_input(conv_index, 1, "pattern weight")
                if len(conv.input) > 2 and conv.input[2]:
                    if (
                        conv.input[2] not in self.initializers
                        and not self.is_quantized_input(conv.input[2])
                    ):
                        self.add(
                            "error",
                            "6.6",
                            "Pattern bias must use an allowed representation.",
                            self.label(conv_index),
                        )
                for input_index in range(1, min(5, len(batch_norm.input))):
                    name = batch_norm.input[input_index]
                    initializer = self.initializers.get(name)
                    if initializer is None or initializer.data_type != TensorProto.FLOAT:
                        self.add(
                            "error",
                            "6.6",
                            "BatchNormalization parameters must be FP32 initializers.",
                            f"{self.label(batch_index)} input[{input_index}]",
                        )
                self.require_quant_output(batch_index)
                continue
            for input_index, input_name in enumerate(pattern.inputs):
                if not self.is_quantized_input(input_name):
                    self.add(
                        "error",
                        "6",
                        f"{pattern.kind} pattern input must have a complete QDQ edge.",
                        f"{self.label(pattern.nodes[0])} input[{input_index}]",
                    )
            output_producer = self.producers[pattern.outputs[0]][0]
            self.require_quant_output(output_producer)
            if pattern.kind == "SDPA":
                inputs = [
                    self.quant_info_from_dq_output(name) for name in pattern.inputs
                ]
                storage_types = {
                    item.storage_type for item in inputs if item is not None
                }
                if len(storage_types) > 1:
                    self.add(
                        "error",
                        "6.4",
                        "SDPA Q, K, and V storage types must match.",
                        self.label(pattern.nodes[0]),
                    )

    def check_conv_like(self, index: int, node: onnx.NodeProto) -> None:
        self.require_quant_input(index, 0, "activation input")
        self.require_quant_input(index, 1, "weight")
        if len(node.input) > 2 and node.input[2]:
            bias_name = node.input[2]
            if bias_name in self.initializers:
                if self.initializers[bias_name].data_type != TensorProto.FLOAT:
                    self.add(
                        "error",
                        "5.2",
                        "A directly used bias initializer must use FP32.",
                        self.label(index),
                    )
            elif not self.is_quantized_input(bias_name):
                self.add(
                    "error",
                    "5.2",
                    "bias must use an allowed FP32 or quantized representation.",
                    self.label(index),
                )
        self.require_quant_output(index)
        if node.op_type == "Gemm":
            canonical = (
                int(get_attribute(node, "transA", 0)) == 0
                and int(get_attribute(node, "transB", 0)) == 1
                and float(get_attribute(node, "alpha", 1.0)) == 1.0
                and float(get_attribute(node, "beta", 1.0)) == 1.0
            )
            if not canonical:
                self.add(
                    "error",
                    "5.2",
                    "Gemm must use canonical fully connected attributes.",
                    self.label(index),
                )
        if (
            node.op_type == "ConvTranspose"
            and int(get_attribute(node, "group", 1)) > 1
            and len(node.input) > 2
            and self.quant_info_from_dq_output(node.input[2]) is not None
        ):
            self.add(
                "error",
                "4",
                "ConvTranspose with group > 1 cannot use quantized bias.",
                self.label(index),
            )

    def check_matmul(self, index: int, node: onnx.NodeProto) -> None:
        self.require_quant_input(index, 0)
        self.require_quant_input(index, 1)
        infos = [
            self.quant_info_from_dq_output(node.input[0]),
            self.quant_info_from_dq_output(node.input[1]),
        ]
        if all(item is not None for item in infos):
            storage_types = {item.storage_type for item in infos if item is not None}
            if storage_types not in ({TensorProto.INT8}, {TensorProto.INT16}):
                self.add(
                    "error",
                    "5.3",
                    "MatMul inputs must both use S8 or both use S16.",
                    self.label(index),
                )
            if any(item.granularity != "per-tensor" for item in infos if item is not None):
                self.add(
                    "error",
                    "5.3",
                    "MatMul inputs must use per-tensor quantization.",
                    self.label(index),
                )
            for item in infos:
                if item is not None:
                    self.check_zero_point_zero(
                        item,
                        "5.3",
                        "Symmetric MatMul input zero_point should be zero.",
                        self.label(index),
                    )
        self.require_quant_output(index)

    def check_grid_sample(self, index: int, node: onnx.NodeProto) -> None:
        self.require_quant_input(index, 0, "feature")
        grid_name = node.input[1]
        grid_info = self.quant_info_from_dq_output(grid_name)
        if grid_info is not None:
            if (
                grid_info.storage_type != TensorProto.INT16
                or grid_info.granularity != "per-tensor"
            ):
                self.add(
                    "error",
                    "5.4",
                    "A quantized grid must use per-tensor S16.",
                    self.label(index),
                )
            self.check_zero_point_zero(
                grid_info,
                "5.4",
                "A symmetric S16 grid zero_point should be zero.",
                self.label(index),
            )
        elif self.value_types.get(grid_name) != TensorProto.FLOAT:
            self.add(
                "error",
                "5.4",
                "GridSample grid must use FP32 or symmetric per-tensor S16.",
                self.label(index),
            )
        self.require_quant_output(index)

    def check_relu_or_clip(
        self, index: int, node: onnx.NodeProto, is_relu: bool
    ) -> None:
        rule = "5.15" if is_relu else "5.16"
        name = "Relu" if is_relu else "Clip"
        input_info = self.quant_info_from_dq_output(node.input[0])
        output_info = self.quant_info_from_q_input(node.output[0])
        if output_info is None:
            self.add(
                "error",
                rule,
                f"{name} output must have a complete QDQ edge.",
                self.label(index),
            )
        if not is_relu:
            for input_index in (1, 2):
                if input_index < len(node.input) and node.input[input_index]:
                    initializer = self.initializers.get(node.input[input_index])
                    if initializer is None or initializer.data_type != TensorProto.FLOAT:
                        self.add(
                            "error",
                            rule,
                            "Clip min and max must be omitted or directly referenced FP32 initializers.",
                            f"{self.label(index)} input[{input_index}]",
                        )
        if input_info is None:
            producer = self.producers.get(node.input[0])
            invalid_upstream = (
                producer is None
                or producer[1].op_type
                in PASSIVE_OPS
                | NON_QUANT_OPS
                | {"QuantizeLinear", "DequantizeLinear"}
            )
            if invalid_upstream:
                self.add(
                    "error",
                    rule,
                    f"{name} has no input QDQ and cannot be a local fusion tail.",
                    self.label(index),
                )
            if (
                output_info is not None
                and output_info.storage_type == TensorProto.INT16
            ):
                self.add(
                    "error",
                    rule,
                    f"{name} local fusion tail requires an asymmetric activation mode.",
                    self.label(index),
                )
        if is_relu and output_info is not None:
            self.check_zero_point_zero(
                output_info,
                rule,
                "Relu output zero_point should be zero.",
                self.label(index),
            )

    def check_operator_rules(self) -> None:
        self.check_pattern_boundaries()
        for index, node in enumerate(self.nodes):
            operation = node.op_type
            if (
                index in self.pattern_nodes
                or operation in {"QuantizeLinear", "DequantizeLinear", "Constant"}
                or operation in PASSIVE_OPS
                or operation in NON_QUANT_OPS
                or operation in UNSUPPORTED_OPS
            ):
                continue
            if operation in {"Conv", "ConvTranspose", "Gemm"}:
                self.check_conv_like(index, node)
            elif operation == "MatMul":
                self.check_matmul(index, node)
            elif operation == "GridSample":
                self.check_grid_sample(index, node)
            elif operation in QUANT_ARITHMETIC_OPS:
                native_integer = operation in {"Add", "Sub", "Mul"} and all(
                    self.value_types.get(name) in INTEGER_TYPES
                    for name in node.input
                    if name
                )
                if not native_integer:
                    for input_index, _ in enumerate(node.input):
                        self.require_quant_input(index, input_index)
                    self.require_quant_output(index)
            elif operation in QUANT_UNARY_OPS:
                self.require_quant_input(index, 0)
                self.require_quant_output(index)
                if operation in {
                    "LayerNormalization",
                    "GroupNormalization",
                    "InstanceNormalization",
                }:
                    for input_index in range(1, len(node.input)):
                        name = node.input[input_index]
                        if not name:
                            continue
                        initializer = self.initializers.get(name)
                        if initializer is None or initializer.data_type != TensorProto.FLOAT:
                            self.add(
                                "error",
                                "5.7",
                                "Affine scale and bias must be FP32 initializers.",
                                f"{self.label(index)} input[{input_index}]",
                            )
                if operation == "BatchNormalization":
                    for input_index in range(1, min(5, len(node.input))):
                        initializer = self.initializers.get(node.input[input_index])
                        if initializer is None or initializer.data_type != TensorProto.FLOAT:
                            self.add(
                                "error",
                                "5.7",
                                "BatchNormalization parameters must be FP32 initializers.",
                                f"{self.label(index)} input[{input_index}]",
                            )
            elif operation == "Pow":
                self.require_quant_input(index, 0, "base")
                self.require_quant_output(index)
                if len(node.input) > 1 and self.quant_info_from_dq_output(node.input[1]):
                    self.add(
                        "error",
                        "5.5",
                        "Pow exponent must not be quantized.",
                        self.label(index),
                    )
            elif operation == "PRelu":
                self.require_quant_input(index, 0)
                self.require_quant_output(index)
                slope = self.initializers.get(node.input[1]) if len(node.input) > 1 else None
                if slope is None or slope.data_type != TensorProto.FLOAT:
                    self.add(
                        "error",
                        "5.8",
                        "PRelu slope must be an FP32 initializer.",
                        self.label(index),
                    )
            elif operation == "RoiAlign":
                self.require_quant_input(index, 0, "feature")
                self.require_quant_output(index)
            elif operation in {"ScatterElements", "ScatterND"}:
                self.require_quant_input(index, 0, "data")
                self.require_quant_input(index, 2, "updates")
                data_info = self.quant_info_from_dq_output(node.input[0])
                update_info = self.quant_info_from_dq_output(node.input[2])
                if (
                    data_info is not None
                    and update_info is not None
                    and data_info.storage_type != update_info.storage_type
                ):
                    self.add(
                        "error",
                        "5.8",
                        "Scatter data and updates storage types must match.",
                        self.label(index),
                    )
                self.require_quant_output(index)
            elif operation == "Where":
                self.require_quant_input(index, 1, "X")
                self.require_quant_input(index, 2, "Y")
                x_info = self.quant_info_from_dq_output(node.input[1])
                y_info = self.quant_info_from_dq_output(node.input[2])
                if (
                    x_info is not None
                    and y_info is not None
                    and x_info.storage_type != y_info.storage_type
                ):
                    self.add(
                        "error",
                        "5.8",
                        "Where X and Y storage types must match.",
                        self.label(index),
                    )
                self.require_quant_output(index)
            elif operation == "Concat":
                native_integer = all(
                    self.value_types.get(name) in INTEGER_TYPES for name in node.input
                )
                if not native_integer:
                    for input_index, _ in enumerate(node.input):
                        self.require_quant_input(index, input_index)
                    infos = [
                        self.quant_info_from_dq_output(name) for name in node.input
                    ]
                    if all(item is not None for item in infos):
                        storage_types = {
                            item.storage_type for item in infos if item is not None
                        }
                        if len(storage_types) > 1:
                            self.add(
                                "error",
                                "5.12",
                                "Concat input storage types must match.",
                                self.label(index),
                            )
                    self.require_quant_output(index)
            elif operation == "Pad":
                input_info = self.quant_info_from_dq_output(node.input[0])
                if input_info is not None:
                    self.require_quant_output(index)
                    output_info = self.quant_info_from_q_input(node.output[0])
                    if (
                        output_info is not None
                        and not self.signatures_equal(input_info, output_info)
                    ):
                        self.add(
                            "error",
                            "5.13",
                            "Quantized Pad must preserve complete quantization parameters.",
                            self.label(index),
                        )
                    if len(node.input) > 2 and node.input[2]:
                        initializer = self.initializers.get(node.input[2])
                        if initializer is None or initializer.data_type != TensorProto.FLOAT:
                            self.add(
                                "error",
                                "5.13",
                                "Pad constant_value must be an FP32 initializer.",
                                self.label(index),
                            )
            elif operation == "TopK":
                input_info = self.quant_info_from_dq_output(node.input[0])
                if input_info is not None and node.output:
                    values_output = node.output[0]
                    if (
                        self.consumers.get(values_output)
                        or values_output in self.graph_outputs
                    ):
                        output_info = self.quant_info_from_q_input(values_output)
                        if output_info is None or not self.signatures_equal(
                            input_info, output_info
                        ):
                            self.add(
                                "error",
                                "5.14",
                                "TopK Values must preserve the complete X quantization parameters.",
                                self.label(index),
                            )
            elif operation == "Relu":
                self.check_relu_or_clip(index, node, True)
            elif operation == "Clip":
                self.check_relu_or_clip(index, node, False)

    def check_passive_subgraphs(self) -> None:
        for index, node in enumerate(self.nodes):
            if node.op_type not in PASSIVE_OPS or index in self.pattern_nodes:
                continue
            if node.op_type == "Identity":
                input_info = self.quant_info_from_dq_output(node.input[0])
                output_info = self.quant_info_from_q_input(node.output[0])
                if (
                    input_info is not None
                    and output_info is not None
                    and not self.signatures_equal(input_info, output_info)
                ):
                    continue
            input_name = node.input[0] if node.input else ""
            origin = self.quant_info_from_dq_output(
                input_name
            ) or self.passive_origins.get(input_name)
            if origin is None:
                continue
            split_source = self.passive_split_sources.get(input_name)
            if node.op_type == "Split" and split_source is None:
                split_source = index
                self.split_inputs[index] = origin
            data_outputs = list(node.output)
            if node.op_type == "MaxPool" and len(data_outputs) > 1:
                data_outputs = data_outputs[:1]
            for output in data_outputs:
                if not output:
                    continue
                self.passive_origins[output] = origin
                if split_source is not None:
                    self.passive_split_sources[output] = split_source
                for consumer_index, consumer, input_index in self.consumers.get(
                    output, []
                ):
                    if consumer.op_type in PASSIVE_OPS and input_index == 0:
                        continue
                    if consumer.op_type == "QuantizeLinear" and input_index == 0:
                        output_info = self.paired_quant_info(consumer_index)
                        location = (
                            f"{self.label(index)} -> {self.label(consumer_index)}"
                        )
                        if split_source is not None:
                            if output_info is not None:
                                self.split_boundaries[split_source].append(
                                    (output_info, location)
                                )
                            continue
                        if (
                            output_info is not None
                            and not self.signatures_equal(origin, output_info)
                        ):
                            self.add(
                                "error",
                                "5.10",
                                "Passive output does not preserve complete quantization parameters.",
                                location,
                            )
                        continue
                    self.add(
                        "error",
                        "5.10",
                        "A quantized passive output has no matching QDQ boundary.",
                        f"{self.label(index)} -> {self.label(consumer_index)}",
                    )
                if output in self.graph_outputs:
                    self.add(
                        "error",
                        "5.10",
                        "A quantized passive output reaches a graph output without QDQ.",
                        f"{self.label(index)} output",
                    )
        self.check_split_boundary_values()

    def check_split_boundary_values(self) -> None:
        if not self.check_value:
            return
        for split_index, boundaries in self.split_boundaries.items():
            input_info = self.split_inputs[split_index]
            input_type = input_info.storage_type
            for output_info, location in boundaries:
                if output_info.storage_type != input_type:
                    continue
                if (
                    input_info.scale.values is None
                    or input_info.zero_point.values is None
                    or output_info.scale.values is None
                    or output_info.zero_point.values is None
                ):
                    self.add(
                        "warn",
                        "5.11",
                        "Split output and same-type input QDQ require statically known scale and zero_point values.",
                        location,
                    )
                if (
                    input_info.scale.values is not None
                    and output_info.scale.values is not None
                    and input_info.scale.values != output_info.scale.values
                ):
                    self.add(
                        "warn",
                        "5.11",
                        "Split output QDQ scale differs from the same-type input QDQ scale.",
                        location,
                    )
                if (
                    input_info.zero_point.values is not None
                    and output_info.zero_point.values is not None
                    and input_info.zero_point.values
                    != output_info.zero_point.values
                ):
                    self.add(
                        "warn",
                        "5.11",
                        "Split output QDQ zero_point differs from the same-type input QDQ zero_point.",
                        location,
                    )

            groups: Dict[Optional[int], List[Tuple[QuantInfo, str]]] = defaultdict(
                list
            )
            for output_info, location in boundaries:
                groups[output_info.storage_type].append((output_info, location))
            for storage_type, group in groups.items():
                if storage_type is None or len(group) < 2:
                    continue
                type_name = TYPE_NAMES.get(storage_type, str(storage_type))
                group_location = (
                    f"{self.label(split_index)} {type_name} outputs: "
                    + "; ".join(location for _, location in group)
                )
                known_scales = [
                    item
                    for item in group
                    if item[0].scale.values is not None
                ]
                known_zero_points = [
                    item
                    for item in group
                    if item[0].zero_point.values is not None
                ]
                if (
                    len(known_scales) != len(group)
                    or len(known_zero_points) != len(group)
                ):
                    self.add(
                        "warn",
                        "5.11",
                        "Same-type Split output QDQ pairs require statically known scale and zero_point values.",
                        group_location,
                    )
                if len(known_scales) >= 2 and any(
                    item.scale.values != known_scales[0][0].scale.values
                    for item, _ in known_scales[1:]
                ):
                    self.add(
                        "warn",
                        "5.11",
                        "Same-type Split output QDQ pairs have inconsistent scale values.",
                        group_location,
                    )
                if len(known_zero_points) >= 2 and any(
                    item.zero_point.values
                    != known_zero_points[0][0].zero_point.values
                    for item, _ in known_zero_points[1:]
                ):
                    self.add(
                        "warn",
                        "5.11",
                        "Same-type Split output QDQ pairs have inconsistent zero_point values.",
                        group_location,
                    )

    def run(self, checker: subprocess.CompletedProcess[str]) -> List[Finding]:
        self.check_model_scope(checker)
        self.check_qdq()
        self.check_quant_roles()
        self.mark_patterns()
        self.check_operator_rules()
        self.check_passive_subgraphs()
        return self.findings


def grouped_findings(findings: Sequence[Finding]) -> List[Tuple[str, str, int, str, str]]:
    groups: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for finding in findings:
        groups[(finding.severity, finding.rule, finding.message)].append(
            finding.location
        )
    severity_order = {"error": 0, "warn": 1, "info": 2}
    rows = []
    for (severity, rule, message), locations in sorted(
        groups.items(),
        key=lambda item: (
            severity_order.get(item[0][0], 9),
            item[0][1],
            item[0][2],
        ),
    ):
        examples = "; ".join(locations[:4])
        if len(locations) > 4:
            examples += f"; plus {len(locations) - 4} more"
        rows.append((severity, rule, len(locations), message, examples))
    return rows


def severity_style(severity: str) -> str:
    return {
        "error": "bold red",
        "warn": "bold yellow",
        "info": "cyan",
    }.get(severity, "white")


def print_finding_table(
    console: Console,
    title: str,
    severity: str,
    rows: Sequence[Tuple[str, str, int, str, str]],
) -> None:
    table = Table(
        title=title,
        title_style=severity_style(severity),
        header_style="bold",
    )
    table.add_column("Rule", no_wrap=True)
    table.add_column("Count", justify="right", no_wrap=True)
    table.add_column("Reason", ratio=3)
    table.add_column("Locations", ratio=2)
    matching_rows = [row for row in rows if row[0] == severity]
    if not matching_rows:
        return
    for _, rule, count, message, locations in matching_rows:
        table.add_row(rule, str(count), message, locations)
    console.print(table)


def print_report(
    console: Console,
    model_path: Path,
    model: onnx.ModelProto,
    check_value: bool,
    findings: Sequence[Finding],
    patterns: Sequence[Pattern],
) -> bool:
    counts = Counter(finding.severity for finding in findings)
    compliant = counts["error"] == 0
    summary = Table(title="QAT ONNX Verification", show_header=False)
    summary.add_column("Field", style="bold cyan", no_wrap=True)
    summary.add_column("Value")
    summary.add_row("Model", str(model_path))
    summary.add_row("check_value", str(check_value).lower())
    summary.add_row("Errors", str(counts["error"]))
    summary.add_row("Warnings", str(counts["warn"]))
    summary.add_row("Info", str(counts["info"]))
    summary.add_row("Nodes", str(len(model.graph.node)))
    summary.add_row(
        "Q / DQ",
        f"{sum(node.op_type == 'QuantizeLinear' for node in model.graph.node)} / "
        f"{sum(node.op_type == 'DequantizeLinear' for node in model.graph.node)}",
    )
    pattern_counts = Counter(pattern.kind for pattern in patterns)
    summary.add_row(
        "Patterns",
        ", ".join(f"{name} x {count}" for name, count in sorted(pattern_counts.items()))
        if pattern_counts
        else "none",
    )
    console.print(summary)

    operator_table = Table(title="Operator Statistics", header_style="bold")
    operator_table.add_column("Operator")
    operator_table.add_column("Count", justify="right")
    operator_counts = Counter(node.op_type for node in model.graph.node)
    for name, count in sorted(operator_counts.items()):
        operator_table.add_row(name, str(count))
    console.print(operator_table)

    rows = grouped_findings(findings)
    print_finding_table(console, "Errors", "error", rows)
    print_finding_table(console, "Warnings", "warn", rows)
    if any(row[0] == "info" for row in rows):
        print_finding_table(console, "Information", "info", rows)
    console.print(
        "[bold green]CHECK PASS[/bold green]"
        if compliant
        else "[bold red]CHECK FAILED[/bold red]"
    )
    return compliant


def print_load_failure(console: Console, model_path: Path, message: str) -> None:
    summary = Table(title="QAT ONNX Verification", show_header=False)
    summary.add_column("Field", style="bold cyan", no_wrap=True)
    summary.add_column("Value")
    summary.add_row("Model", str(model_path))
    summary.add_row("Status", "[red]LOAD FAILED[/red]")
    summary.add_row("Reason", message)
    console.print(summary)
    console.print("[bold red]CHECK FAILED[/bold red]")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify an ONNX QDQ model against the repository ax quant spec."
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=Path,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "-c",
        "--check-value",
        action="store_true",
        help="Enable QDQ pair, zero_point, and bias scale value checks.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    console = Console()
    model_path = arguments.model.expanduser()
    if not model_path.is_file():
        print_load_failure(console, model_path, "The model file does not exist.")
        return 2
    load_check = load_result(model_path)
    if load_check.returncode != 0:
        print_load_failure(console, model_path, subprocess_error(load_check))
        return 2
    model = onnx.load(model_path)
    checker = checker_result(model_path)
    verifier = Verifier(model, arguments.check_value)
    findings = verifier.run(checker)
    compliant = print_report(
        console,
        model_path,
        model,
        arguments.check_value,
        findings,
        verifier.patterns,
    )
    return 0 if compliant else 1


if __name__ == "__main__":
    sys.exit(main())

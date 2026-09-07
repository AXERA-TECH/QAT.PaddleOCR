#!/usr/bin/env python3
"""Shared inspection and validation helpers for generic Pulsar2 configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    shape: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def tensor_contract(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "shape": tensor_shape(value_info),
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
    }


def load_model(model_path: Path) -> onnx.ModelProto:
    if not model_path.is_file():
        raise ValueError(f"ONNX does not exist: {model_path}")
    model = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def model_contract(model_path: Path) -> dict[str, Any]:
    model = load_model(model_path)
    return {
        "path": str(model_path.resolve()),
        "inputs": [tensor_contract(value) for value in model.graph.input],
        "outputs": [tensor_contract(value) for value in model.graph.output],
        "nodes": len(model.graph.node),
        "node_names": [node.name for node in model.graph.node if node.name],
    }


def choose_input(contract: dict[str, Any], input_name: str | None) -> dict[str, Any]:
    inputs = contract["inputs"]
    if input_name is None:
        if len(inputs) != 1:
            raise ValueError(
                "QuantONNX has multiple inputs; pass --input-name explicitly."
            )
        return inputs[0]
    matches = [item for item in inputs if item["name"] == input_name]
    if len(matches) != 1:
        raise ValueError(
            f"Input {input_name!r} was not found; available inputs are "
            f"{[item['name'] for item in inputs]}"
        )
    return matches[0]


def load_layer_configs(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"Layer config file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Layer config file must contain a JSON array.")
    return payload


def validate_layer_configs(
    layer_configs: list[dict[str, Any]],
    node_names: set[str],
) -> None:
    for index, rule in enumerate(layer_configs):
        if not isinstance(rule, dict):
            raise ValueError(f"layer_configs[{index}] must be an object.")
        names = rule.get("layer_names")
        if not isinstance(names, list) or not names or not all(
            isinstance(name, str) and name for name in names
        ):
            raise ValueError(
                f"layer_configs[{index}].layer_names must be a non-empty string array."
            )
        unknown = sorted(set(names) - node_names)
        if unknown:
            raise ValueError(
                f"layer_configs[{index}] contains nodes absent from the selected graph: "
                f"{unknown}"
            )


def validate_config(
    onnx_path: Path,
    config_path: Path,
    frontend_path: Path | None = None,
    expected_target_hardware: str | None = None,
    expected_npu_mode: str | None = None,
) -> dict[str, Any]:
    contract = model_contract(onnx_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Pulsar2 config must be a JSON object.")
    configured_input = config.get("input")
    if not isinstance(configured_input, str):
        raise ValueError("Config input must be an ONNX path string.")
    configured_path = Path(configured_input)
    if not configured_path.is_absolute():
        configured_path = config_path.parent / configured_path
    if configured_path.resolve() != onnx_path.resolve():
        raise ValueError("Config input does not match the requested ONNX path.")
    if config.get("model_type") != "QuantONNX":
        raise ValueError("model_type must be QuantONNX.")
    if expected_target_hardware and config.get("target_hardware") != expected_target_hardware:
        raise ValueError(f"target_hardware must be {expected_target_hardware}.")
    if expected_npu_mode and config.get("npu_mode") != expected_npu_mode:
        raise ValueError(f"npu_mode must be {expected_npu_mode}.")

    input_configs = config.get("quant", {}).get("input_configs", [])
    if not isinstance(input_configs, list) or len(input_configs) != 1:
        raise ValueError("quant.input_configs must contain exactly one item.")
    input_config = input_configs[0]
    if not isinstance(input_config, dict) or not input_config.get("calibration_dataset"):
        raise ValueError("quant.input_configs must specify calibration_dataset.")
    input_names = {item["name"] for item in contract["inputs"]}
    config_input_name = input_config.get("tensor_name")
    if config_input_name not in input_names and config_input_name != "DEFAULT":
        raise ValueError(
            f"quant.input_configs tensor_name {config_input_name!r} is not an ONNX input."
        )

    processors = config.get("input_processors", [])
    if not isinstance(processors, list):
        raise ValueError("input_processors must be an array.")
    for processor in processors:
        if not isinstance(processor, dict):
            raise ValueError("Every input processor must be an object.")
        if processor.get("tensor_name") not in input_names | {"DEFAULT"}:
            raise ValueError("Input processor tensor_name is not an ONNX input.")
        for field in ("tensor_layout", "src_layout", "src_dtype"):
            if not isinstance(processor.get(field), str):
                raise ValueError(f"Input processor field {field} must be a string.")
        for field in ("mean", "std"):
            values = processor.get(field)
            if not isinstance(values, list) or not all(
                isinstance(value, (int, float)) for value in values
            ):
                raise ValueError(f"Input processor field {field} must be numeric array.")

    frontend_contract = None
    validation_graph = contract
    if frontend_path is not None:
        frontend_contract = model_contract(frontend_path)
        validation_graph = frontend_contract
    layer_configs = config.get("quant", {}).get("layer_configs", [])
    if not isinstance(layer_configs, list):
        raise ValueError("quant.layer_configs must be an array.")
    validate_layer_configs(layer_configs, set(validation_graph["node_names"]))
    compiler = config.get("compiler")
    if not isinstance(compiler, dict) or compiler.get("check") != 2:
        raise ValueError("compiler.check must be 2.")
    if config.get("output_processors") != []:
        raise ValueError("output_processors must be an empty array.")
    return {
        "onnx": contract,
        "frontend": frontend_contract,
        "target_hardware": config.get("target_hardware"),
        "npu_mode": config.get("npu_mode"),
        "layer_configs": len(layer_configs),
        "input_processors": len(processors),
    }

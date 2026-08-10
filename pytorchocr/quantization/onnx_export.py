import gc
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from onnxscript import evaluator
from onnxscript.function_libs.torch_lib.tensor_typing import TFloat
from onnxscript.onnx_opset import opset18 as op
from torch.onnx._internal.exporter import _schemas
from torch import nn

from .validation import constant_tensors, qparam_key


_BATCH_NORM_SIGNATURE = _schemas.OpSignature.from_opschema(
    onnx.defs.get_schema("BatchNormalization", max_inclusive_version=18)
)
_BATCH_NORM_SIGNATURE.outputs = _BATCH_NORM_SIGNATURE.outputs[:1]


def _batch_normalization(
    input,
    weight,
    bias,
    running_mean,
    running_var,
    momentum,
    eps,
):
    recorder = evaluator.default()
    outputs = recorder._call_op(
        _BATCH_NORM_SIGNATURE,
        {
            "X": input,
            "scale": weight,
            "B": bias,
            "input_mean": running_mean,
            "input_var": running_var,
        },
        {
            "epsilon": eps,
            "momentum": 1.0 - momentum,
            "training_mode": 0,
        },
    )
    return outputs[0]


class OutputSelector(nn.Module):
    def __init__(self, model, output_index):
        super().__init__()
        self.model = model
        self.output_index = output_index

    def forward(self, *inputs):
        return self.model(*inputs)[self.output_index]


class RecTrainingDeploymentProjection(nn.Module):
    """Project a full recognition QAT graph to its deployment CTC output."""

    def __init__(self, model, max_text_length):
        super().__init__()
        self.model = model
        self.max_text_length = int(max_text_length)

    def forward(self, images):
        gtc_targets = images.new_zeros(
            (images.shape[0], self.max_text_length),
            dtype=torch.int64,
        )
        return self.model(images, gtc_targets)[0]


def fold_constant_zero_point_casts(model):
    """Fold exact integer Casts used only as Q/DQ zero-point constants."""
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    consumers = {}
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            consumers.setdefault(input_name, []).append((node, input_index))
    graph_outputs = {output.name for output in model.graph.output}
    existing_names = set(initializers)
    folded_nodes = set()
    folded_sources = set()
    replacements = []

    for node in model.graph.node:
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            continue
        source = initializers.get(node.input[0])
        output_name = node.output[0]
        output_consumers = consumers.get(output_name, [])
        if source is None or not output_consumers or output_name in graph_outputs:
            continue
        if not all(
            consumer.op_type in ("QuantizeLinear", "DequantizeLinear")
            and input_index == 2
            for consumer, input_index in output_consumers
        ):
            continue
        target_type = next(
            (
                attribute.i
                for attribute in node.attribute
                if attribute.name == "to"
            ),
            None,
        )
        if target_type is None or output_name in existing_names:
            continue
        try:
            target_dtype = helper.tensor_dtype_to_np_dtype(target_type)
        except (KeyError, TypeError, ValueError):
            continue
        source_array = numpy_helper.to_array(source)
        if not (
            np.issubdtype(source_array.dtype, np.integer)
            and np.issubdtype(target_dtype, np.integer)
        ):
            continue
        target_limits = np.iinfo(target_dtype)
        if source_array.size and (
            source_array.min() < target_limits.min
            or source_array.max() > target_limits.max
        ):
            continue
        cast_array = source_array.astype(target_dtype, copy=True)
        if not np.array_equal(
            cast_array.astype(source_array.dtype),
            source_array,
        ):
            continue
        replacements.append(numpy_helper.from_array(cast_array, output_name))
        existing_names.add(output_name)
        folded_nodes.add(id(node))
        folded_sources.add(node.input[0])

    if not folded_nodes:
        return 0

    kept_nodes = [node for node in model.graph.node if id(node) not in folded_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.initializer.extend(replacements)

    remaining_inputs = {
        input_name for node in model.graph.node for input_name in node.input
    }
    graph_inputs = {input_value.name for input_value in model.graph.input}
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in folded_sources
        or initializer.name in remaining_inputs
        or initializer.name in graph_inputs
        or initializer.name in graph_outputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    return len(folded_nodes)


def remove_exact_redundant_dq_q(model):
    """Remove only direct DQ->Q pairs with bit-identical qparams."""
    values = constant_tensors(model)
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }
    graph_outputs = {output.name for output in model.graph.output}
    nodes_to_remove = set()
    removed = 0
    for quantize in model.graph.node:
        if quantize.op_type != "QuantizeLinear" or not quantize.input:
            continue
        dequantize = producers.get(quantize.input[0])
        if dequantize is None or dequantize.op_type != "DequantizeLinear":
            continue
        if qparam_key(quantize, values) != qparam_key(dequantize, values):
            continue
        for consumer in model.graph.node:
            for index, input_name in enumerate(consumer.input):
                if input_name == quantize.output[0]:
                    consumer.input[index] = dequantize.input[0]
        nodes_to_remove.add((quantize.name, tuple(quantize.output)))
        removed += 1

        remaining_users = [
            node
            for node in model.graph.node
            if node is not quantize and dequantize.output[0] in node.input
        ]
        if not remaining_users and dequantize.output[0] not in graph_outputs:
            nodes_to_remove.add((dequantize.name, tuple(dequantize.output)))
    if nodes_to_remove:
        kept = [
            node
            for node in model.graph.node
            if (node.name, tuple(node.output)) not in nodes_to_remove
        ]
        del model.graph.node[:]
        model.graph.node.extend(kept)
    return removed


def insert_identity_for_requantize_dq_q(model):
    """Make non-identical direct DQ -> Q requantization boundaries explicit."""
    values = constant_tensors(model)
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }
    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    rewritten = []
    inserted = 0
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and node.input:
            dequantize = producers.get(node.input[0])
            is_requantize = (
                dequantize is not None
                and dequantize.op_type == "DequantizeLinear"
                and qparam_key(node, values) is not None
                and qparam_key(dequantize, values) is not None
                and qparam_key(node, values)
                != qparam_key(dequantize, values)
            )
            if is_requantize:
                base_name = (
                    f"{dequantize.name or dequantize.output[0]}"
                    f"_to_{node.name or node.output[0]}_identity"
                )
                identity_name = base_name
                suffix = 1
                while identity_name in existing_names:
                    identity_name = f"{base_name}_{suffix}"
                    suffix += 1
                identity_output = f"{node.input[0]}_identity"
                suffix = 1
                while identity_output in existing_names:
                    identity_output = f"{node.input[0]}_identity_{suffix}"
                    suffix += 1
                rewritten.append(
                    helper.make_node(
                        "Identity",
                        [node.input[0]],
                        [identity_output],
                        name=identity_name,
                    )
                )
                existing_names.update((identity_name, identity_output))
                node.input[0] = identity_output
                inserted += 1
        rewritten.append(node)
    if inserted:
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
    return inserted


def name_dynamic_input_axes(model, dynamic_axis_names):
    """Give already-symbolic ONNX input dimensions stable public names."""
    renamed = 0
    for input_index, axes in (dynamic_axis_names or {}).items():
        if input_index >= len(model.graph.input):
            raise ValueError(f"ONNX input index does not exist: {input_index}")
        dimensions = model.graph.input[input_index].type.tensor_type.shape.dim
        for axis, name in axes.items():
            if axis >= len(dimensions):
                raise ValueError(
                    f"ONNX input {input_index} axis does not exist: {axis}"
                )
            dimension = dimensions[axis]
            if not dimension.dim_param:
                raise RuntimeError(
                    f"ONNX input {input_index} axis {axis} is not dynamic."
                )
            dimension.dim_param = str(name)
            renamed += 1
    return renamed


def set_static_input_axes(model, static_axis_values):
    """Specialize selected ONNX input dimensions for deployment."""
    specialized = 0
    for input_index, axes in (static_axis_values or {}).items():
        if input_index >= len(model.graph.input):
            raise ValueError(f"ONNX input index does not exist: {input_index}")
        dimensions = model.graph.input[input_index].type.tensor_type.shape.dim
        for axis, value in axes.items():
            if axis >= len(dimensions):
                raise ValueError(
                    f"ONNX input {input_index} axis does not exist: {axis}"
                )
            value = int(value)
            if value <= 0:
                raise ValueError("Static ONNX dimensions must be positive.")
            dimension = dimensions[axis]
            if dimension.dim_value and dimension.dim_value != value:
                raise RuntimeError(
                    f"ONNX input {input_index} axis {axis} is "
                    f"{dimension.dim_value}, expected {value}."
                )
            dimension.ClearField("dim_param")
            dimension.dim_value = value
            specialized += 1
    return specialized


def prepared_activation_qparams(prepared):
    records = []
    visited = set()
    for node in prepared.graph.nodes:
        if node.op != "call_module":
            continue
        module = prepared.get_submodule(str(node.target))
        if id(module) in visited or not hasattr(module, "scale"):
            continue
        source = node.args[0] if node.args else None
        if isinstance(source, torch.fx.Node) and source.op == "get_attr":
            try:
                value = prepared.get_parameter(str(source.target))
            except AttributeError:
                try:
                    value = prepared.get_buffer(str(source.target))
                except AttributeError:
                    value = None
            if (
                isinstance(value, torch.Tensor)
                and value.dtype.is_floating_point
                and value.numel() == 1
            ):
                continue
        qscheme = getattr(module, "qscheme", None)
        if qscheme not in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            continue
        scale = module.scale.detach().cpu().numpy()
        zero_point = module.zero_point.detach().cpu().numpy()
        records.append(
            {
                "observer": str(node.target),
                "scale_is_one": bool(np.all(scale == 1.0)),
                "zero_point_is_zero": bool(np.all(zero_point == 0)),
            }
        )
        visited.add(id(module))
    return records


def onnx_activation_qparams(model):
    values = constant_tensors(model)
    records = []
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        if node.input[0] in values:
            continue
        scale = values[node.input[1]]
        zero_point = values[node.input[2]]
        records.append(
            {
                "node": node.name,
                "scale_is_one": bool(np.all(scale == 1.0)),
                "zero_point_is_zero": bool(np.all(zero_point == 0)),
                "scale_shape": list(scale.shape),
                "zero_point_dtype": str(zero_point.dtype),
            }
        )
    return records


def require_default_qparams(records, source):
    mismatches = [
        record
        for record in records
        if not record["scale_is_one"] or not record["zero_point_is_zero"]
    ]
    if mismatches:
        names = [item.get("observer", item.get("node")) for item in mismatches]
        raise RuntimeError(
            f"{source} contains non-default activation qparams: {names[:10]}"
        )


def aten_batch_norm_inference(
    input: TFloat,
    weight: TFloat,
    bias: TFloat,
    running_mean: TFloat,
    running_var: TFloat,
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
) -> TFloat:
    if training:
        raise ValueError("QAT ONNX export requires BatchNorm in inference mode.")
    return _batch_normalization(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        momentum,
        eps,
    )


def aten_native_batch_norm_inference(
    input: TFloat,
    weight: TFloat,
    bias: TFloat,
    running_mean: TFloat,
    running_var: TFloat,
    momentum: float,
    eps: float,
) -> Tuple[TFloat, TFloat, TFloat]:
    normalized = _batch_normalization(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        momentum,
        eps,
    )
    inverse_std = op.Div(1.0, op.Sqrt(op.Add(running_var, eps)))
    return normalized, op.Identity(running_mean), inverse_std


def export_onnx(
    model,
    example_inputs,
    output_path,
    output_names,
    input_names=None,
    optimize=True,
    output_index=None,
    dynamic_shapes=None,
    dynamic_axis_names=None,
    static_axis_values=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = (
        OutputSelector(model, output_index) if output_index is not None else model
    )
    onnx_program = torch.onnx.export(
        export_model,
        example_inputs,
        input_names=list(input_names) if input_names is not None else None,
        output_names=list(output_names),
        custom_translation_table={
            torch.ops.aten.batch_norm.default: aten_batch_norm_inference,
            torch.ops.aten._native_batch_norm_legit_no_training.default:
                aten_native_batch_norm_inference,
        },
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        opset_version=21,
    )
    if optimize:
        onnx_program.optimize()
    onnx_program.save(str(output_path))
    del onnx_program
    gc.collect()
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    renamed_dynamic_axes = name_dynamic_input_axes(
        onnx_model,
        dynamic_axis_names,
    )
    specialized_static_axes = set_static_input_axes(
        onnx_model,
        static_axis_values,
    )
    folded_zero_point_casts = fold_constant_zero_point_casts(onnx_model)
    removed_requant = remove_exact_redundant_dq_q(onnx_model)
    inserted_requant_identity = insert_identity_for_requantize_dq_q(onnx_model)
    if (
        renamed_dynamic_axes
        or specialized_static_axes
        or folded_zero_point_casts
        or removed_requant
        or inserted_requant_identity
    ):
        onnx.save(onnx_model, str(output_path))
    if renamed_dynamic_axes:
        print(f"named {renamed_dynamic_axes} dynamic ONNX input axis/axes")
    if specialized_static_axes:
        print(f"specialized {specialized_static_axes} static ONNX input axis/axes")
    if folded_zero_point_casts:
        print(
            f"folded {folded_zero_point_casts} constant zero-point Cast node(s)"
        )
    if removed_requant:
        print(f"removed {removed_requant} exact redundant DQ -> Q pair(s)")
    if inserted_requant_identity:
        print(
            "inserted "
            f"{inserted_requant_identity} Identity node(s) for non-identical DQ -> Q"
        )
    onnx.checker.check_model(onnx_model, full_check=True)
    return onnx_model


def export_qat_onnx(
    model,
    example_inputs,
    output_path,
    output_names,
    input_names=None,
    optimize=True,
    output_index=None,
    dynamic_shapes=None,
    dynamic_axis_names=None,
    static_axis_values=None,
):
    """Backward-compatible QuantONNX entry point."""
    return export_onnx(
        model,
        example_inputs,
        output_path,
        output_names,
        input_names=input_names,
        optimize=optimize,
        output_index=output_index,
        dynamic_shapes=dynamic_shapes,
        dynamic_axis_names=dynamic_axis_names,
        static_axis_values=static_axis_values,
    )

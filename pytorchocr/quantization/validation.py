import numpy as np
import onnxruntime as ort
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator


def constant_tensors(model):
    values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        value = next(
            (attribute for attribute in node.attribute if attribute.name == "value"),
            None,
        )
        if value is not None:
            values[node.output[0]] = numpy_helper.to_array(value.t)
    return values


def qparam_key(node, values):
    if len(node.input) < 3:
        return None
    scale = values.get(node.input[1])
    zero_point = values.get(node.input[2])
    if scale is None or zero_point is None:
        return None
    axis = next(
        (attribute.i for attribute in node.attribute if attribute.name == "axis"),
        1,
    )
    return (
        scale.tobytes(),
        str(scale.dtype),
        zero_point.tobytes(),
        str(zero_point.dtype),
        axis,
    )


def merge_domain_stats(model, producers, values, op_type):
    quantized = 0
    shared = 0
    for node in model.graph.node:
        if node.op_type != op_type:
            continue
        input_producers = [producers.get(input_name) for input_name in node.input]
        if not all(
            producer is not None and producer.op_type == "DequantizeLinear"
            for producer in input_producers
        ):
            continue
        quantized += 1
        qparams = [qparam_key(producer, values) for producer in input_producers]
        shared += len(set(qparams)) == 1
    return {"quantized": quantized, "shared_qparams": shared}


def hard_activation_stats(model, producers, users):
    standard = 0
    paddle_hsigmoid = 0
    quantized = 0
    for node in model.graph.node:
        if node.op_type == "HardSigmoid":
            standard += 1
            input_producer = producers.get(node.input[0])
            if (
                input_producer is not None
                and input_producer.op_type == "DequantizeLinear"
                and any(
                    user.op_type == "QuantizeLinear"
                    for user in users.get(node.output[0], [])
                )
            ):
                quantized += 1
            continue
        if node.op_type != "Clip":
            continue
        div_users = [
            user
            for user in users.get(node.output[0], [])
            if user.op_type == "Div"
        ]
        if not div_users:
            continue
        paddle_hsigmoid += len(div_users)
        add_node = producers.get(node.input[0])
        input_dequantizer = (
            producers.get(add_node.input[0])
            if add_node is not None and add_node.op_type == "Add"
            else None
        )
        input_quantizer = (
            producers.get(input_dequantizer.input[0])
            if input_dequantizer is not None
            and input_dequantizer.op_type == "DequantizeLinear"
            else None
        )
        scale_node = (
            producers.get(input_quantizer.input[0])
            if input_quantizer is not None
            and input_quantizer.op_type == "QuantizeLinear"
            else None
        )
        for div_node in div_users:
            if (
                input_dequantizer is not None
                and input_dequantizer.op_type == "DequantizeLinear"
                and scale_node is not None
                and scale_node.op_type == "Mul"
                and any(
                    user.op_type == "QuantizeLinear"
                    for user in users.get(div_node.output[0], [])
                )
            ):
                quantized += 1
    return {
        "total": standard + paddle_hsigmoid,
        "quantized": quantized,
        "standard": standard,
        "paddle_hsigmoid": paddle_hsigmoid,
    }


def silu_activation_stats(model, producers, users):
    total = 0
    quantized = 0
    internal_qdq = 0
    for sigmoid in model.graph.node:
        if sigmoid.op_type != "Sigmoid" or not sigmoid.input or not sigmoid.output:
            continue
        source = sigmoid.input[0]
        queue = list(users.get(sigmoid.output[0], []))
        visited = set()
        mul = None
        crossed_qdq = False
        while queue:
            node = queue.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            if node.op_type == "Mul" and source in node.input:
                mul = node
                break
            if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                crossed_qdq = True
                for output_name in node.output:
                    queue.extend(users.get(output_name, []))
        if mul is None:
            continue
        total += 1
        internal_qdq += int(crossed_qdq)
        input_producer = producers.get(source)
        has_input_dq = (
            input_producer is not None
            and input_producer.op_type == "DequantizeLinear"
        )
        has_output_q = any(
            user.op_type == "QuantizeLinear"
            for user in users.get(mul.output[0], [])
        )
        quantized += int(has_input_dq and has_output_q and not crossed_qdq)
    return {
        "total": total,
        "quantized": quantized,
        "internal_qdq": internal_qdq,
    }


def qdq_stats(model):
    producers = {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
    }
    users = {}
    for node in model.graph.node:
        for input_name in node.input:
            users.setdefault(input_name, []).append(node)
    values = constant_tensors(model)
    quantize = [node for node in model.graph.node if node.op_type == "QuantizeLinear"]
    dequantize = [
        node for node in model.graph.node if node.op_type == "DequantizeLinear"
    ]
    direct_dq_q = 0
    redundant_dq_q = 0
    for node in quantize:
        producer = producers.get(node.input[0])
        if producer is not None and producer.op_type == "DequantizeLinear":
            direct_dq_q += 1
            redundant_dq_q += qparam_key(producer, values) == qparam_key(node, values)
    quantize_zero_point_dtypes = {}
    for node in quantize:
        dtype = str(values[node.input[2]].dtype)
        quantize_zero_point_dtypes[dtype] = (
            quantize_zero_point_dtypes.get(dtype, 0) + 1
        )
    quantized_weight_dtypes = {}
    for node in dequantize:
        quantized_value = values.get(node.input[0])
        if quantized_value is None or quantized_value.ndim < 2:
            continue
        dtype = str(quantized_value.dtype)
        quantized_weight_dtypes[dtype] = (
            quantized_weight_dtypes.get(dtype, 0) + 1
        )
    conv_qdq_batchnorm = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        for quantizer_node in users.get(node.output[0], []):
            if quantizer_node.op_type != "QuantizeLinear":
                continue
            for dequantizer_node in users.get(quantizer_node.output[0], []):
                if dequantizer_node.op_type != "DequantizeLinear":
                    continue
                conv_qdq_batchnorm += sum(
                    user.op_type == "BatchNormalization"
                    for user in users.get(dequantizer_node.output[0], [])
                )
    unquantized_conv_outputs = []
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvTranspose"):
            continue
        output_users = users.get(node.output[0], [])
        if any(user.op_type == "QuantizeLinear" for user in output_users):
            continue
        fused_activations = [
            user for user in output_users if user.op_type in ("Relu", "Clip")
        ]
        if fused_activations and all(
            any(
                activation_user.op_type == "QuantizeLinear"
                for activation_user in users.get(activation.output[0], [])
            )
            for activation in fused_activations
        ):
            continue
        unquantized_conv_outputs.append(node.name)
    return {
        "nodes": len(model.graph.node),
        "batch_normalization": sum(
            node.op_type == "BatchNormalization" for node in model.graph.node
        ),
        "quantize_linear": len(quantize),
        "dequantize_linear": len(dequantize),
        "direct_dq_q": direct_dq_q,
        "redundant_dq_q": redundant_dq_q,
        "requantize_dq_q": direct_dq_q - redundant_dq_q,
        "conv_qdq_batchnorm": conv_qdq_batchnorm,
        "unquantized_conv_outputs": unquantized_conv_outputs,
        "quantize_zero_point_dtypes": quantize_zero_point_dtypes,
        "quantized_weight_dtypes": quantized_weight_dtypes,
        "add_domains": merge_domain_stats(model, producers, values, "Add"),
        "concat_domains": merge_domain_stats(model, producers, values, "Concat"),
        "hard_activation_domains": hard_activation_stats(model, producers, users),
        "silu_domains": silu_activation_stats(model, producers, users),
    }


def validate_qdq_graph(model):
    stats = qdq_stats(model)
    if stats["batch_normalization"]:
        raise RuntimeError("ONNX graph contains BatchNormalization nodes.")
    if stats["redundant_dq_q"]:
        raise RuntimeError("ONNX graph contains a redundant DQ -> Q pair.")
    if stats["conv_qdq_batchnorm"]:
        raise RuntimeError("ONNX graph contains a Conv -> QDQ -> BatchNorm pattern.")
    if stats["unquantized_conv_outputs"]:
        names = ", ".join(stats["unquantized_conv_outputs"])
        raise RuntimeError(f"ONNX Conv outputs are missing QDQ: {names}")
    concat_domains = stats["concat_domains"]
    if concat_domains["quantized"] != concat_domains["shared_qparams"]:
        raise RuntimeError("ONNX Concat inputs do not share one quantization domain.")
    hard_domains = stats["hard_activation_domains"]
    if hard_domains["total"] != hard_domains["quantized"]:
        raise RuntimeError(
            "ONNX HardSigmoid patterns are not fully quantized: "
            f"{hard_domains['quantized']} / {hard_domains['total']}"
        )
    silu_domains = stats["silu_domains"]
    if silu_domains["total"] != silu_domains["quantized"]:
        raise RuntimeError(
            "ONNX SiLU patterns must have QDQ only at the activation boundary: "
            f"{silu_domains['quantized']} / {silu_domains['total']} quantized, "
            f"{silu_domains['internal_qdq']} with internal QDQ"
        )
    return stats


def run_ort(model_path, images, optimize=False):
    session_options = ort.SessionOptions()
    if not optimize:
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: images.detach().cpu().numpy()})[0]


def run_onnx_reference(model, images):
    evaluator = ReferenceEvaluator(model)
    input_name = model.graph.input[0].name
    return evaluator.run(
        None,
        {input_name: images.detach().cpu().numpy()},
    )[0]


def numpy_error_stats(reference, actual):
    difference = np.abs(reference - actual)
    return {
        "mae": float(np.mean(difference)),
        "max_abs": float(np.max(difference)),
    }


def sequence_error_stats(reference, actual):
    reference_probabilities = _softmax(reference)
    actual_probabilities = _softmax(actual)
    probability_difference = np.abs(
        reference_probabilities - actual_probabilities
    )
    return {
        "argmax_agreement": float(
            np.mean(
                np.argmax(reference, axis=-1) == np.argmax(actual, axis=-1)
            )
        ),
        "probability_mae": float(np.mean(probability_difference)),
        "probability_max_abs": float(np.max(probability_difference)),
    }


def _softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)

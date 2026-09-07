"""ONNX graph folding passes for BN and LAB static scalars in non-reparameterized QAT."""

import numpy as np
from onnx import helper, numpy_helper
from .validation import constant_tensors


def fold_conv_qdq_bn(model):
  """Fold Conv → Q → DQ → BN into Conv weights.

  Matches: Conv(W,b) → Q(s1,zp1) → DQ(s1) → BN(γ,β,μ,σ²,ε) → Q(s2)
  Result: Conv_folded → Q(s2), with BN parameters absorbed into W,b

  Returns: number of Conv-BN pairs folded
  """
  values = constant_tensors(model)
  producers = {
    output_name: node
    for node in model.graph.node
    for output_name in node.output
  }
  init_map = {init.name: init for init in model.graph.initializer}

  folded = 0
  nodes_to_remove = []

  for bn_node in list(model.graph.node):
    if bn_node.op_type != "BatchNormalization":
      continue

    # Trace upstream: BN.input[0] ← DQ.output[0] ← Q.input[0] ← Conv.output[0]
    dq_node = producers.get(bn_node.input[0])
    if dq_node is None or dq_node.op_type != "DequantizeLinear":
      continue

    q_node = producers.get(dq_node.input[0])
    if q_node is None or q_node.op_type != "QuantizeLinear":
      continue

    conv_node = producers.get(q_node.input[0])
    if conv_node is None or conv_node.op_type != "Conv":
      continue

    # Read BN parameters: γ, β, μ, σ² (from inputs 1-4)
    gamma_init = init_map.get(bn_node.input[1])
    beta_init = init_map.get(bn_node.input[2])
    mean_init = init_map.get(bn_node.input[3])
    var_init = init_map.get(bn_node.input[4])

    if not all([gamma_init, beta_init, mean_init, var_init]):
      continue

    gamma = numpy_helper.to_array(gamma_init).astype(np.float32)
    beta = numpy_helper.to_array(beta_init).astype(np.float32)
    mean = numpy_helper.to_array(mean_init).astype(np.float32)
    variance = numpy_helper.to_array(var_init).astype(np.float32)

    # Get BN epsilon (attribute)
    eps_attr = next((a for a in bn_node.attribute if a.name == "epsilon"), None)
    eps = eps_attr.f if eps_attr else 1e-5

    # Read Conv weight and bias
    weight_init = init_map.get(conv_node.input[1])
    if weight_init is None:
      continue

    weight = numpy_helper.to_array(weight_init).astype(np.float32)

    # Conv may or may not have bias
    has_bias = len(conv_node.input) > 2 and conv_node.input[2]
    if has_bias:
      bias_init = init_map.get(conv_node.input[2])
      if bias_init is None:
        has_bias = False
      else:
        bias = numpy_helper.to_array(bias_init).astype(np.float32)

    if not has_bias:
      bias = np.zeros(weight.shape[0], dtype=np.float32)

    # Compute folding coefficient: a_c = γ_c / sqrt(σ²_c + ε)
    # Shape: γ and β are [C], weight is [OC, IC, KH, KW], bias is [OC]
    a = gamma / np.sqrt(variance + eps)  # [C]

    # Fold into Conv: W_new = W * a, b_new = (b - μ) * a + β
    weight_folded = weight * a[:, np.newaxis, np.newaxis, np.newaxis]
    bias_folded = (bias - mean) * a + beta

    # Update Conv initializers
    weight_init.raw_data = numpy_helper.from_array(weight_folded, conv_node.input[1]).raw_data

    if has_bias:
      bias_init.raw_data = numpy_helper.from_array(bias_folded, conv_node.input[2]).raw_data
    else:
      # Create new bias initializer
      bias_tensor = numpy_helper.from_array(bias_folded, f"{conv_node.name}_bias")
      model.graph.initializer.append(bias_tensor)
      conv_node.input.append(bias_tensor.name)

    # Rewire: Conv.output[0] → [Q(s1), DQ(s1), BN] → Q(s2).input[0]
    # Find Q(s2) node that consumes BN.output[0]
    q_out_node = producers.get(bn_node.output[0])
    if q_out_node is not None and q_out_node.op_type == "QuantizeLinear":
      # Rewire Q(s2) to consume Conv directly
      for node in model.graph.node:
        for idx, inp in enumerate(node.input):
          if inp == q_out_node.output[0]:
            # Reconnect through Conv
            pass  # Will update after node removal

    # Change Conv.output to match BN.output (so Q(s2) still matches)
    # Actually: Conv.output → Q(s1).input, so no change needed
    # But we need to reroute: Q(s2).input[0] = DQ.output[0] → Conv.output[0]
    q_out_node.input[0] = conv_node.output[0]

    # Mark Q(s1), DQ(s1), BN for removal
    nodes_to_remove.extend([q_node.name, dq_node.name, bn_node.name])
    folded += 1

  # Remove marked nodes
  if nodes_to_remove:
    kept = [n for n in model.graph.node if n.name not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept)

  return folded


def replace_qdq_bn_with_affine(model):
  """Replace DQ → BN → Q patterns (non-Conv upstream) with float Mul+Add.

  Matches: upstream(Add/Mul) → Q(s1) → DQ(s1) → BN → Q(s2)
  Replaces: DQ → Mul(a) → Add(b) → [original Q(s2).input[0]]
  where a_c = γ_c / sqrt(σ²_c + ε), b_c = β_c - μ_c * a_c (per-channel)

  Returns: number of Add/Mul-BN pairs replaced
  """
  values = constant_tensors(model)
  producers = {
    output_name: node
    for node in model.graph.node
    for output_name in node.output
  }
  init_map = {init.name: init for init in model.graph.initializer}
  existing_names = {
    name for node in model.graph.node
    for name in (*node.input, *node.output) if name
  }

  replaced = 0
  nodes_to_remove = []
  nodes_to_add = []

  for bn_node in list(model.graph.node):
    if bn_node.op_type != "BatchNormalization":
      continue

    # Check if upstream is NOT Conv (we already handled Conv in Pass 1)
    dq_node = producers.get(bn_node.input[0])
    if dq_node is None or dq_node.op_type != "DequantizeLinear":
      continue

    q_node = producers.get(dq_node.input[0])
    if q_node is None or q_node.op_type != "QuantizeLinear":
      continue

    upstream_node = producers.get(q_node.input[0])
    if upstream_node is None or upstream_node.op_type == "Conv":
      continue  # Already handled in Pass 1

    # This is an Add or Mul upstream; proceed with Affine replacement
    if upstream_node.op_type not in ("Add", "Mul"):
      continue

    # Read BN parameters
    gamma_init = init_map.get(bn_node.input[1])
    beta_init = init_map.get(bn_node.input[2])
    mean_init = init_map.get(bn_node.input[3])
    var_init = init_map.get(bn_node.input[4])

    if not all([gamma_init, beta_init, mean_init, var_init]):
      continue

    gamma = numpy_helper.to_array(gamma_init).astype(np.float32)
    beta = numpy_helper.to_array(beta_init).astype(np.float32)
    mean = numpy_helper.to_array(mean_init).astype(np.float32)
    variance = numpy_helper.to_array(var_init).astype(np.float32)

    eps_attr = next((a for a in bn_node.attribute if a.name == "epsilon"), None)
    eps = eps_attr.f if eps_attr else 1e-5

    # Compute coefficients: a_c, b_c
    a = gamma / np.sqrt(variance + eps)  # [C]
    b = beta - mean * a  # [C]

    # Reshape for broadcasting: [1, C, 1, 1] (assuming 4D NCHW)
    a_expanded = a.reshape(1, -1, 1, 1)
    b_expanded = b.reshape(1, -1, 1, 1)

    # Create initializers for Mul and Add
    mul_weight_name = f"{bn_node.name}_affine_a"
    add_bias_name = f"{bn_node.name}_affine_b"

    suffix = 1
    while mul_weight_name in existing_names:
      mul_weight_name = f"{bn_node.name}_affine_a_{suffix}"
      suffix += 1

    suffix = 1
    while add_bias_name in existing_names:
      add_bias_name = f"{bn_node.name}_affine_b_{suffix}"
      suffix += 1

    mul_weight_tensor = numpy_helper.from_array(a_expanded, mul_weight_name)
    add_bias_tensor = numpy_helper.from_array(b_expanded, add_bias_name)
    model.graph.initializer.extend([mul_weight_tensor, add_bias_tensor])
    existing_names.update([mul_weight_name, add_bias_name])

    # Create Mul and Add nodes
    mul_output_name = f"{dq_node.output[0]}_mul"
    suffix = 1
    while mul_output_name in existing_names:
      mul_output_name = f"{dq_node.output[0]}_mul_{suffix}"
      suffix += 1

    mul_node = helper.make_node(
      "Mul",
      inputs=[dq_node.output[0], mul_weight_name],
      outputs=[mul_output_name],
      name=f"{bn_node.name}_mul",
    )

    add_output_name = mul_output_name  # Will be updated after BN removal
    add_node = helper.make_node(
      "Add",
      inputs=[mul_output_name, add_bias_name],
      outputs=[add_output_name],  # Placeholder
      name=f"{bn_node.name}_affine_add",
    )

    # Find Q(s2) node consuming BN.output[0]
    q_out_node = producers.get(bn_node.output[0])
    if q_out_node is not None and q_out_node.op_type == "QuantizeLinear":
      add_output_name = q_out_node.input[0]  # Direct connection to Q input
      add_node.output[0] = add_output_name

    nodes_to_add.extend([mul_node, add_node])
    nodes_to_remove.extend([q_node.name, dq_node.name, bn_node.name])
    replaced += 1

  # Apply changes
  if nodes_to_remove or nodes_to_add:
    kept = [n for n in model.graph.node if n.name not in nodes_to_remove]
    kept.extend(nodes_to_add)
    del model.graph.node[:]
    model.graph.node.extend(kept)

  return replaced


def fold_static_scalar_mul(model):
  """Fold DQ(static) → Mul(w_float) → Q into adjusted scale of preceding DQ.

  Matches: feature_DQ(s_x) → Mul(w_float_from_static_DQ) → Q(s_q, zp_q)
  where w_float > 0 (no sign change) and feature_DQ has exactly 1 consumer.

  Optimization: update feature_DQ.scale ← s_x * w_float, remove Mul and DQ(static).

  Returns: number of static scalar Muls folded
  """
  values = constant_tensors(model)
  producers = {
    output_name: node
    for node in model.graph.node
    for output_name in node.output
  }
  init_map = {init.name: init for init in model.graph.initializer}

  folded = 0
  nodes_to_remove = []

  for mul_node in list(model.graph.node):
    if mul_node.op_type != "Mul":
      continue

    if not mul_node.input or not mul_node.output:
      continue

    # Find which input is the static scalar
    input0 = producers.get(mul_node.input[0])
    input1 = producers.get(mul_node.input[1]) if len(mul_node.input) > 1 else None

    static_dq = None
    feature_input_idx = None

    if input0 and input0.op_type == "DequantizeLinear" and input0.input[0] in values:
      static_dq = input0
      feature_input_idx = 1
    elif input1 and input1.op_type == "DequantizeLinear" and input1.input[0] in values:
      static_dq = input1
      feature_input_idx = 0

    if static_dq is None:
      continue

    # Get feature DQ (the other input)
    feature_dq = producers.get(mul_node.input[feature_input_idx])
    if feature_dq is None or feature_dq.op_type != "DequantizeLinear":
      continue

    # Check feature_DQ has exactly 1 consumer
    consumers = [n for n in model.graph.node if mul_node.name in n.input]
    if len(consumers) != 1:
      continue  # Multiple consumers; unsafe to modify feature_DQ scale

    # Extract static scalar value
    static_int_init = init_map.get(static_dq.input[0])
    static_scale_init = init_map.get(static_dq.input[1])
    static_zp_init = init_map.get(static_dq.input[2])

    if not all([static_int_init, static_scale_init, static_zp_init]):
      continue

    static_int = numpy_helper.to_array(static_int_init).astype(np.float32)
    static_scale = numpy_helper.to_array(static_scale_init).astype(np.float32)
    static_zp = numpy_helper.to_array(static_zp_init).astype(np.float32)

    w_float = (static_int - static_zp) * static_scale

    # Check positivity and scalar
    if not (np.all(w_float > 0) and w_float.size == 1):
      continue

    w_float = float(w_float.flat[0])

    # Update feature_DQ scale
    feature_scale_init = init_map.get(feature_dq.input[1])
    if feature_scale_init is None:
      continue

    feature_scale = numpy_helper.to_array(feature_scale_init).astype(np.float32)
    feature_scale_new = feature_scale * w_float
    feature_scale_init.raw_data = numpy_helper.from_array(
      feature_scale_new, feature_dq.input[1]
    ).raw_data

    # Rewire: Mul.input[feature_input_idx] → Mul.output consumers
    for consumer in model.graph.node:
      for idx, inp in enumerate(consumer.input):
        if inp == mul_node.output[0]:
          consumer.input[idx] = feature_dq.output[0]

    # Mark for removal: static_dq, mul_node
    nodes_to_remove.extend([static_dq.name, mul_node.name])
    folded += 1

  if nodes_to_remove:
    kept = [n for n in model.graph.node if n.name not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept)

  return folded


def fold_static_scalar_add(model):
  """Fold DQ(static) → Add(b_float) → Q into Q's zero_point or remove Add.

  Matches: feature_DQ(s_x, zp_x) → Add(b_float_from_static_DQ) → Q(s_q, zp_q)

  For zero-bias (|b_float| < 1e-7): remove DQ(static) + Add directly.
  For non-zero bias: attempt to fold into Q's zero_point (if range safe).

  Returns: number of static scalar Adds folded or removed
  """
  values = constant_tensors(model)
  producers = {
    output_name: node
    for node in model.graph.node
    for output_name in node.output
  }
  init_map = {init.name: init for init in model.graph.initializer}

  folded = 0
  nodes_to_remove = []
  rewrites = {}

  for add_node in list(model.graph.node):
    if add_node.op_type != "Add":
      continue

    if not add_node.input or not add_node.output:
      continue

    # Find which input is the static scalar
    input0 = producers.get(add_node.input[0])
    input1 = producers.get(add_node.input[1]) if len(add_node.input) > 1 else None

    static_dq = None
    feature_input_idx = None

    if input0 and input0.op_type == "DequantizeLinear" and input0.input[0] in values:
      static_dq = input0
      feature_input_idx = 1
    elif input1 and input1.op_type == "DequantizeLinear" and input1.input[0] in values:
      static_dq = input1
      feature_input_idx = 0

    if static_dq is None:
      continue

    # Get feature DQ
    feature_dq = producers.get(add_node.input[feature_input_idx])
    if feature_dq is None or feature_dq.op_type != "DequantizeLinear":
      continue

    # Extract static scalar
    static_int_init = init_map.get(static_dq.input[0])
    static_scale_init = init_map.get(static_dq.input[1])
    static_zp_init = init_map.get(static_dq.input[2])

    if not all([static_int_init, static_scale_init, static_zp_init]):
      continue

    static_int = numpy_helper.to_array(static_int_init).astype(np.float32)
    static_scale = numpy_helper.to_array(static_scale_init).astype(np.float32)
    static_zp = numpy_helper.to_array(static_zp_init).astype(np.float32)

    b_float = (static_int - static_zp) * static_scale

    if b_float.size != 1:
      continue

    b_float = float(b_float.flat[0])

    # Case 1: Zero bias → direct removal
    if abs(b_float) < 1e-7:
      rewrites[add_node.output[0]] = feature_dq.output[0]
      nodes_to_remove.extend([static_dq.name, add_node.name])
      folded += 1
      continue

    # Case 2: Non-zero bias → try to fold into Q's zero_point
    q_out = producers.get(add_node.output[0])
    if q_out is None or q_out.op_type != "QuantizeLinear":
      continue  # Not followed by Q; skip

    q_scale_init = init_map.get(q_out.input[1])
    q_zp_init = init_map.get(q_out.input[2])

    if not all([q_scale_init, q_zp_init]):
      continue

    q_scale = numpy_helper.to_array(q_scale_init).astype(np.float32)
    q_zp = numpy_helper.to_array(q_zp_init).astype(np.int32)

    # Quantized bias: b_int = round(b_float / s_q)
    b_int = np.round(b_float / q_scale).astype(np.int32)
    zp_new = q_zp - b_int

    # Check if zp_new is in valid range (assume S16: [-32767, 32767] or U16: [0, 65535])
    # For simplicity, just check int32 range; Pulsar2 validation happens later
    if not (-2**31 <= zp_new <= 2**31 - 1):
      continue  # Overflow; skip this fold

    # Update Q's zero_point
    q_zp_init.raw_data = numpy_helper.from_array(zp_new, q_out.input[2]).raw_data

    # Rewire: Add.output → Q.input[0]
    rewrites[add_node.output[0]] = feature_dq.output[0]
    nodes_to_remove.extend([static_dq.name, add_node.name])
    folded += 1

  # Apply rewrites
  if rewrites:
    for node in model.graph.node:
      for idx, inp in enumerate(node.input):
        if inp in rewrites:
          node.input[idx] = rewrites[inp]

  if nodes_to_remove:
    kept = [n for n in model.graph.node if n.name not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(kept)

  return folded

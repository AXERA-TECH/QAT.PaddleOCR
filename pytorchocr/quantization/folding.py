"""Quantized-domain folding public API.

A non-reparameterized QAT checkpoint stores LSQ-quantized-domain weights:
each branch weight is only meaningful through its fake-quant (fq) node
(per-channel, may clip). Folding the branches into a single deployment conv
must therefore sum the *effective* quantized-domain weights:

    W_fused = sum_i pad(fq_i(fold_i(w_i))),   b_fused = sum_i fold_bias_i

This module provides:

- ``collect_quantized_weight_map``: extract effective weights from a prepared
  graph (conv2d/linear fq chains, including derived bias fq).
- ``build_folded_state``: build the folded single-branch state dict from a
  non-reparameterized QAT checkpoint.
- ``apply_folded_state``: load the folded state onto a
  ``reparameterize_for_deploy``'d eager model (the finetune starting point).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.ao.quantization import move_exported_model_to_eval

from pytorchocr.modeling.backbones.rec_lcnetv3 import ConvBNLayer, LearnableRepLayer


def attr_value(gm, target):
    value = gm
    if not isinstance(target, str):
        target = target.target
    for part in target.split("."):
        value = getattr(value, part)
    return value


def eval_expr(gm, node):
    if not hasattr(node, "op"):
        return torch.tensor(node, dtype=torch.float32)
    if node.op == "get_attr":
        return attr_value(gm, node).float()
    target = str(node.target)
    if target.endswith("div.Tensor"):
        return eval_expr(gm, node.args[0]) / eval_expr(gm, node.args[1])
    if target.endswith("sqrt.default"):
        return torch.sqrt(eval_expr(gm, node.args[0]))
    if target.endswith("add.Tensor"):
        return eval_expr(gm, node.args[0]) + eval_expr(gm, node.args[1])
    if target.endswith("reshape.default"):
        return eval_expr(gm, node.args[0])
    raise ValueError(f"unsupported expr: {target}")


def collect_quantized_weight_map(prepared):
    """Return {attr_target: effective_weight_tensor} for conv2d and linear."""
    gm = prepared
    result = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        is_conv = target.endswith("conv2d.default")
        is_linear = target.endswith("linear.default") or target.endswith("addmm.default")
        if not (is_conv or is_linear):
            continue
        warg = node.args[1]
        if warg is None or warg.op != "call_module":
            continue
        fq = gm.get_submodule(warg.target)
        if not hasattr(fq, "scale"):
            continue
        for nn in gm.graph.nodes:
            if nn.target == warg.target:
                src = nn.args[0]
                break
        if src is None:
            continue
        if src.op == "get_attr":
            raw = attr_value(gm, src)
            with torch.no_grad():
                eff = fq(raw.detach().float()).to(raw.dtype)
            result[src.target] = eff
        elif src.op == "call_function" and str(src.target).endswith("mul.Tensor"):
            wattr = src.args[0] if src.args[0].op == "get_attr" else src.args[1]
            other = src.args[1] if src.args[0].op == "get_attr" else src.args[0]
            if wattr.op != "get_attr":
                continue
            raw = attr_value(gm, wattr).float()
            if other.op == "get_attr":
                scale_vec = attr_value(gm, other).float()
            elif other.op == "call_function":
                scale_vec = eval_expr(gm, other)
            else:
                continue
            folded = raw * scale_vec.reshape(-1, 1, 1, 1)
            with torch.no_grad():
                eff = fq(folded.detach()).to(raw.dtype)
            result[wattr.target] = eff
        if len(node.args) > 2 and node.args[2] is not None:
            barg = node.args[2]
            if barg.op == "call_module":
                try:
                    bfq = gm.get_submodule(barg.target)
                    for nn in gm.graph.nodes:
                        if nn.target == barg.target:
                            bsrc = nn.args[0]
                            break
                    if bsrc is not None and bsrc.op == "get_attr":
                        raw_b = attr_value(gm, bsrc).float()
                        with torch.no_grad():
                            eff_b = bfq(raw_b.detach())
                        result[bsrc.target] = eff_b.to(raw_b.dtype)
                except (AttributeError, RuntimeError):
                    pass
    return result


def fuse_branch(branch):
    if branch is None:
        return None
    if isinstance(branch, ConvBNLayer):
        kernel = branch.conv.weight
        norm = getattr(branch, "norm", None) or getattr(branch, "bn", None)
        if norm is None:
            return None
        std = (norm.running_var + norm.eps).sqrt()
        t = (norm.weight / std).reshape((-1, 1, 1, 1))
        return kernel * t, norm.bias - norm.running_mean * norm.weight / std
    return None


def fuse_identity(block):
    bn = block.identity
    input_dim = block.in_channels // block.groups
    id_tensor = bn.weight.new_zeros(
        (block.in_channels, input_dim, block.kernel_size, block.kernel_size))
    for i in range(block.in_channels):
        id_tensor[i, i % input_dim, block.kernel_size // 2, block.kernel_size // 2] = 1
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape((-1, 1, 1, 1))
    return id_tensor * t, bn.bias - bn.running_mean * bn.weight / std


def checkpoint_qat_config(metadata):
    """Resolve the qspec used at training time.

    Prefers the archived copy from metadata ``copied_configs`` (the live
    config may have evolved after training, which would rebuild a different
    prepared graph and break strict load).
    """
    qat_config = metadata.get("qat_config")
    copied = metadata.get("copied_configs") or {}
    for name, path in copied.items():
        if name.endswith(".json") and any(
            token in name.lower() for token in ("qat", "lsq", "attn", "s16", "u16")
        ):
            candidate = Path(path)
            if candidate.exists():
                qat_config = str(candidate)
                break
    return qat_config


def strip_state_prefix(state_dict, prefix="model."):
    """Strip a leading module prefix from state dict keys and drop QAT
    auxiliary keys (fake quant buffers, tensor constants)."""
    out = {}
    for k, v in state_dict.items():
        if "activation_post_process" in k or k.startswith("_tensor_constant"):
            continue
        stripped = k[len(prefix):] if k.startswith(prefix) else k
        out[stripped] = v
    return out


def build_folded_state(ckpt, model_config, weights):
    """Build the quantized-domain folded state from a non-reparameterized QAT
    checkpoint. The state loads strict=True into a model that has been
    ``reparameterize_for_deploy``'d."""
    from pytorchocr.diagnostics import build_prepared_qat_checkpoint
    from pytorchocr.training import build_task_model

    metadata = ckpt["metadata"]
    build_metadata = {
        **metadata,
        "qat_config": checkpoint_qat_config(metadata),
    }
    prepared, _ = build_prepared_qat_checkpoint(
        build_metadata, weights_path=None, model_config=model_config
    )
    loaded = prepared.load_state_dict(ckpt["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            "Checkpoint graph does not match the current quantizer build "
            f"(missing={len(loaded.missing_keys)}, unexpected={len(loaded.unexpected_keys)}). "
            "This is expected when the QAT JSON/quantizer changed after training "
            "(e.g. output/output_is_symmetric support). Use a checkpoint trained "
            "with the current quantizer, or reuse the folded state archived from "
            "the original extraction."
        )
    move_exported_model_to_eval(prepared)
    eff_map = collect_quantized_weight_map(prepared)

    state = strip_state_prefix(ckpt["model"])
    model = build_task_model(
        "rec", model_config, weights_path=weights,
        reparameterize=False, rec_graph="pretrained_train",
    )
    model.load_state_dict(state, strict=True)

    folded_state = {}

    # 1) LearnableRepLayer: branch folding summed in the quantized domain
    for module_name, module in model.named_modules():
        if not isinstance(module, LearnableRepLayer):
            continue
        pad = module.kernel_size // 2
        kernel = None
        bias = None
        for idx, conv in enumerate(module.conv_kxk):
            attr = f"model.{module_name}.conv_kxk.{idx}.conv.weight"
            fused = fuse_branch(conv)
            w, b = fused
            if attr in eff_map:
                w = eff_map[attr]
            kernel = w if kernel is None else kernel + w
            bias = b if bias is None else bias + b
        if module.conv_1x1 is not None:
            attr = f"model.{module_name}.conv_1x1.conv.weight"
            fused = fuse_branch(module.conv_1x1)
            w, b = fused
            if attr in eff_map:
                w = eff_map[attr]
            w = F.pad(w, [pad, pad, pad, pad])
            kernel = w if kernel is None else kernel + w
            bias = b if bias is None else bias + b
        if module.identity is not None:
            w, b = fuse_identity(module)
            kernel = w if kernel is None else kernel + w
            bias = b if bias is None else bias + b
        folded_state[f"{module_name}.reparam_conv.weight"] = kernel
        folded_state[f"{module_name}.reparam_conv.bias"] = bias

    # 2) Plain ConvBNLayer (conv1/neck/head convs)
    for module_name, module in model.named_modules():
        if not isinstance(module, ConvBNLayer):
            continue
        fused = fuse_branch(module)
        if fused is None:
            continue
        w, b = fused
        attr = f"model.{module_name}.conv.weight"
        if attr in eff_map:
            w = eff_map[attr]
        folded_state[f"{module_name}.conv.weight"] = w
        folded_state[f"{module_name}.conv.bias"] = b
        norm = getattr(module, "norm", None) or getattr(module, "bn", None)
        if norm is None:
            continue
        norm_prefix = "norm" if hasattr(module, "norm") else "bn"
        folded_state[f"{module_name}.{norm_prefix}.weight"] = torch.ones_like(norm.weight)
        folded_state[f"{module_name}.{norm_prefix}.bias"] = torch.zeros_like(norm.bias)
        folded_state[f"{module_name}.{norm_prefix}.running_mean"] = torch.zeros_like(norm.running_mean)
        folded_state[f"{module_name}.{norm_prefix}.running_var"] = torch.ones_like(norm.running_var)
        folded_state[f"{module_name}.{norm_prefix}.num_batches_tracked"] = torch.tensor(0)

    # 3) Linear weights + bare SE Conv2d
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            attr = f"model.{module_name}.weight"
            if attr in eff_map:
                folded_state[f"{module_name}.weight"] = eff_map[attr].to(module.weight.dtype)
        elif isinstance(module, torch.nn.Conv2d) and not isinstance(module, ConvBNLayer):
            parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
            parent = model.get_submodule(parent_name) if parent_name else None
            if isinstance(parent, ConvBNLayer) or "se" not in module_name:
                continue
            attr = f"model.{module_name}.weight"
            if attr in eff_map:
                folded_state[f"{module_name}.weight"] = eff_map[attr].to(module.weight.dtype)
                if module.bias is not None:
                    bias_attr = f"model.{module_name}.bias"
                    if bias_attr in eff_map:
                        folded_state[f"{module_name}.bias"] = eff_map[bias_attr].to(module.bias.dtype)
                    else:
                        folded_state[f"{module_name}.bias"] = module.bias.detach().clone()

    return folded_state


def apply_folded_state(model, folded_state):
    """Overwrite a reparameterize_for_deploy'd eager model with the folded
    state. Returns (applied_count, skipped_count)."""
    current = model.state_dict()
    applied = 0
    skipped = 0
    for k, v in folded_state.items():
        if k in current:
            if current[k].shape == v.shape:
                current[k] = v
                applied += 1
            else:
                skipped += 1
        else:
            skipped += 1
    model.load_state_dict(current, strict=True)
    return applied, skipped


def folded_eager_model(checkpoint_path, model_config, weights, folded_state):
    """Build the folded finetune-starting eager model: source checkpoint raw
    weights -> eager multi-branch -> reparameterize_for_deploy -> apply folded
    state."""
    from pytorchocr.quantization import reparameterize_for_deploy
    from pytorchocr.training import build_task_model

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_task_model(
        "rec", model_config, weights_path=weights,
        reparameterize=False, rec_graph="pretrained_train",
    )
    state = strip_state_prefix(ckpt["model"])
    model.load_state_dict(state, strict=True)
    reparameterize_for_deploy(model)
    model.eval()
    applied, skipped = apply_folded_state(model, folded_state)
    return model, applied, skipped

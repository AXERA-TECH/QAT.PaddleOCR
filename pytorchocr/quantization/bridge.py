import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import (
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer import (
    QuantizationSpec,
    Quantizer,
    SharedQuantizationSpec,
)
from torch.fx import Node


_PER_CHANNEL_QSCHEMES = {
    torch.per_channel_affine,
    torch.per_channel_symmetric,
    torch.per_channel_affine_float_qparams,
}


class _SignedScalarMovingAverageObserver(MovingAverageMinMaxObserver):
    """S16 symmetric observer for one-element trainable parameters."""

    def calculate_qparams(self):
        if self.min_val == float("inf") and self.max_val == float("-inf"):
            return (
                torch.ones(1, dtype=torch.float32, device=self.min_val.device),
                torch.zeros(1, dtype=torch.int32, device=self.min_val.device),
            )
        denominator = float(max(abs(self.quant_min), abs(self.quant_max)))
        max_abs = torch.maximum(self.min_val.abs(), self.max_val.abs())
        scale = torch.maximum(
            max_abs.to(torch.float32) / denominator,
            self.eps.to(device=max_abs.device, dtype=torch.float32),
        ).reshape(-1)
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        return scale, zero_point


def _get_exported_attribute(module, target):
    try:
        return module.get_parameter(target)
    except AttributeError:
        try:
            return module.get_buffer(target)
        except AttributeError:
            owner_name, _, attribute_name = target.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            return getattr(owner, attribute_name)


def _resolve_static_fx_value(module, value, cache):
    if not isinstance(value, Node):
        if isinstance(value, tuple):
            return tuple(_resolve_static_fx_value(module, item, cache) for item in value)
        if isinstance(value, list):
            return [_resolve_static_fx_value(module, item, cache) for item in value]
        if isinstance(value, dict):
            return {
                key: _resolve_static_fx_value(module, item, cache)
                for key, item in value.items()
            }
        return value
    if value in cache:
        return cache[value]
    if value.op == "get_attr":
        result = _get_exported_attribute(module, str(value.target))
    elif value.op == "call_function":
        args = _resolve_static_fx_value(module, value.args, cache)
        kwargs = _resolve_static_fx_value(module, value.kwargs, cache)
        result = value.target(*args, **kwargs)
    elif value.op == "call_method":
        args = _resolve_static_fx_value(module, value.args, cache)
        kwargs = _resolve_static_fx_value(module, value.kwargs, cache)
        result = getattr(args[0], str(value.target))(*args[1:], **kwargs)
    else:
        raise RuntimeError(
            f"FX node {value.name} ({value.op}) is not a static parameter expression."
        )
    cache[value] = result
    return result


class AxeraQuantizerAdapter(Quantizer):
    """Apply project-side graph-domain fixes around QAT.axera."""

    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer

    def transform_for_annotation(self, model):
        return self.quantizer.transform_for_annotation(model)

    def annotate(self, model):
        model = self.quantizer.annotate(model)
        self._use_weight_qspec_for_dyadic_static_scalars(model)
        self._keep_nonfinite_attention_masks_float(model)
        self._remove_non_floating_qspecs(model)
        self._share_concat_domains(model)
        return model

    def validate(self, model):
        return self.quantizer.validate(model)

    def _use_weight_qspec_for_dyadic_static_scalars(self, model):
        """Quantize static scalar Mul/Add operands as weights, not activations.

        The scalar width follows the node's activation domain: S16-domain
        dyadic ops (e.g. the S16 downsampling chain) get an S16 scalar so the
        AX650 TENG pixel-repeat packing (p.n * p.ksize) % 256 == 0 holds for
        W*C=240 layouts (240 * 16 % 256 == 0, while 240 * 8 fails); everything
        else keeps the global weight qspec (S8).
        """
        global_config = getattr(self.quantizer, "global_config", None)
        weight_qspec = getattr(global_config, "weight", None)
        if weight_qspec is None:
            return
        scalar_weight_qspec = QuantizationSpec(
            dtype=weight_qspec.dtype,
            quant_min=weight_qspec.quant_min,
            quant_max=weight_qspec.quant_max,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=weight_qspec.is_dynamic,
            observer_or_fake_quant_ctr=FakeQuantize.with_args(
                observer=_SignedScalarMovingAverageObserver,
                eps=2**-12,
            ),
        )
        scalar_s16_qspec = QuantizationSpec(
            dtype=torch.int16,
            quant_min=-32767,
            quant_max=32767,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=weight_qspec.is_dynamic,
            observer_or_fake_quant_ctr=FakeQuantize.with_args(
                observer=_SignedScalarMovingAverageObserver,
                eps=2**-12,
            ),
        )
        dyadic_ops = {
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.mul_.Tensor,
        }
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in dyadic_ops:
                continue
            annotation = node.meta.get("quantization_annotation")
            if annotation is None or not annotation._annotated:
                continue
            s16_domain = any(
                getattr(spec, "dtype", None) == torch.int16
                for input_node, spec in annotation.input_qspec_map.items()
                if input_node.op != "get_attr"
            )
            scalar_qspec = scalar_s16_qspec if s16_domain else scalar_weight_qspec
            for input_node in tuple(annotation.input_qspec_map):
                if input_node.op != "get_attr":
                    continue
                try:
                    value = _get_exported_attribute(model, str(input_node.target))
                except AttributeError:
                    continue
                if (
                    not isinstance(value, torch.Tensor)
                    or not value.dtype.is_floating_point
                    or value.numel() != 1
                ):
                    continue
                annotation.input_qspec_map[input_node] = scalar_qspec

    @staticmethod
    def _keep_nonfinite_attention_masks_float(model):
        """Keep causal masks float until Softmax produces finite probabilities."""
        new_full = torch.ops.aten.new_full.default
        mask_path_ops = {
            new_full,
            torch.ops.aten.triu.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
            torch.ops.aten.unsqueeze.default,
        }
        softmax_ops = {
            torch.ops.aten.softmax.int,
            torch.ops.aten._softmax.default,
        }
        mask_nodes = set()
        queue = []
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target != new_full:
                continue
            fill_value = (
                node.args[2]
                if len(node.args) > 2
                else node.kwargs.get("fill_value")
            )
            if not isinstance(fill_value, (int, float)):
                continue
            if math.isfinite(float(fill_value)):
                continue
            mask_nodes.add(node)
            queue.append(node)

        softmax_nodes = set()
        while queue:
            node = queue.pop(0)
            for user in node.users:
                if user.op != "call_function":
                    continue
                if user.target in softmax_ops:
                    softmax_nodes.add(user)
                    continue
                if user.target not in mask_path_ops or user in mask_nodes:
                    continue
                mask_nodes.add(user)
                queue.append(user)

        if not softmax_nodes:
            return
        for node in mask_nodes:
            annotation = node.meta.get("quantization_annotation")
            if annotation is None:
                continue
            annotation.input_qspec_map = {
                input_node: qspec
                for input_node, qspec in annotation.input_qspec_map.items()
                if input_node not in mask_nodes
            }
            annotation.output_qspec = None
        for node in softmax_nodes:
            annotation = node.meta.get("quantization_annotation")
            if annotation is None:
                continue
            annotation.input_qspec_map = {
                input_node: qspec
                for input_node, qspec in annotation.input_qspec_map.items()
                if input_node not in mask_nodes
            }

    @staticmethod
    def _node_has_non_floating_tensor(node):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return not value.dtype.is_floating_point
        if isinstance(value, (tuple, list)) and value:
            tensors = [item for item in value if isinstance(item, torch.Tensor)]
            return bool(tensors) and all(
                not item.dtype.is_floating_point for item in tensors
            )
        return False

    @classmethod
    def _remove_non_floating_qspecs(cls, model):
        """Keep token indices, masks, and shape tensors outside QAT domains."""
        for node in model.graph.nodes:
            annotation = node.meta.get("quantization_annotation")
            if annotation is None:
                continue
            annotation.input_qspec_map = {
                input_node: qspec
                for input_node, qspec in annotation.input_qspec_map.items()
                if not cls._node_has_non_floating_tensor(input_node)
            }
            if cls._node_has_non_floating_tensor(node):
                annotation.output_qspec = None

    @staticmethod
    def _share_concat_domains(model):
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target != torch.ops.aten.cat.default:
                continue
            inputs = node.args[0]
            if not inputs or not all(
                isinstance(input_node, Node) for input_node in inputs
            ):
                continue
            annotation = node.meta.get("quantization_annotation")
            if annotation is None or not annotation._annotated:
                continue
            first_input = inputs[0]
            first_qspec = annotation.input_qspec_map.get(first_input)
            if first_qspec is None:
                continue
            # PT2E QAT rewrites Conv-BN nodes after quantizer annotation. An
            # edge-root that names a BN input becomes stale during that pass.
            # Use the stable Concat output node as the root only for this case;
            # preserve the existing edge-root layout (and checkpoint names)
            # for already-supported v6/mobile graphs.
            has_fusable_bn_input = any(
                input_node.op == "call_function"
                and "batch_norm" in str(input_node.target)
                for input_node in inputs
            )
            if has_fusable_bn_input:
                shared_qspec = SharedQuantizationSpec(node)
                annotation.input_qspec_map = {
                    input_node: shared_qspec for input_node in inputs
                }
                annotation.output_qspec = first_qspec
                continue
            shared_qspec = SharedQuantizationSpec((first_input, node))
            annotation.input_qspec_map = {
                input_node: first_qspec if index == 0 else shared_qspec
                for index, input_node in enumerate(inputs)
            }
            annotation.output_qspec = shared_qspec

class DetInferenceWrapper(nn.Module):
    """Expose the DB inference map as a stable tensor-only QAT interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)["maps"]

    def set_qat_capture_mode(self):
        self.model.train()
        # Keep child BN layers in training mode while selecting tensor-only
        # neck output and DBHead's inference-only shrink-map branch.
        self.model.neck.training = False
        self.model.head.training = False
        self.training = True
        return self


class DetTrainingWrapper(nn.Module):
    """Expose the three DB training maps without auxiliary detection heads."""

    def __init__(self, model, expose_intermediates=False):
        super().__init__()
        self.model = model
        self.expose_intermediates = bool(expose_intermediates)

    def forward(self, images):
        backbone_out = self.model.backbone(images)
        neck_outputs = self.model.neck(backbone_out)
        # v6 RepLKFPN returns auxiliary maps in a dict; v4/v5 RSEFPN/LKPAN
        # return the fused tensor directly. Keep one training graph contract.
        features = (
            neck_outputs["fuse"]
            if isinstance(neck_outputs, dict)
            else neck_outputs
        )
        shrink = self.model.head.binarize(features)
        threshold = self.model.head.thresh(features)
        binary = self.model.head.step_function(shrink, threshold)
        if not self.expose_intermediates:
            return shrink, threshold, binary
        return {
            "maps": (shrink, threshold, binary),
            "backbone_out": backbone_out,
            "neck_out": features,
        }

    def set_qat_capture_mode(self):
        self.model.train()
        self.training = True
        return self

    def set_validation_mode(self):
        self.eval()
        # Keep the training-graph return branches while child BN modules stay
        # in eval mode. The wrapper explicitly selects all three DB maps.
        self.model.neck.training = True
        self.model.head.training = True
        return self


class FullDetTrainingWrapper(nn.Module):
    """Expose every native DB training output in a stable tuple order."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.has_auxiliary_maps = getattr(model.head, "aux_in_channels", 0) > 0

    @property
    def output_names(self):
        names = ["maps"]
        if self.has_auxiliary_maps:
            names.extend(("aux_maps_p4", "aux_maps_p3", "aux_maps_p2"))
        return tuple(names)

    def forward(self, images):
        outputs = self.model(images)
        return tuple(outputs[name] for name in self.output_names)

    def set_qat_capture_mode(self):
        self.train()
        return self

    def set_validation_mode(self):
        self.eval()
        # Select the full DB training branches while keeping child BN modules
        # in inference mode for deterministic ONNX execution.
        self.model.head.training = True
        if self.has_auxiliary_maps:
            self.model.neck.training = True
        return self


class RecCTCWrapper(nn.Module):
    """Expose only the CTC path from a recognition MultiHead model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        features = images
        if self.model.use_transform:
            features = self.model.transform(features)
        features = self.model.backbone(features)
        if self.model.use_neck:
            features = self.model.neck(features)
        features = self.model.head.ctc_encoder(features)
        return self.model.head.ctc_head(features)

    def set_qat_capture_mode(self):
        self.model.train()
        self.training = True
        return self

    def set_validation_mode(self):
        self.eval()
        # PaddleOCR's CTCHead applies softmax when its own training flag is
        # false. Loss and metric validation require the same raw logits as QAT.
        self.model.head.ctc_head.training = True
        return self


class FullRecTrainingWrapper(nn.Module):
    """Expose CTC, CTC neck, and fixed-length NRTR training outputs.

    PaddleOCR trims the NRTR decoder to the longest label in each eager
    batch. Export uses the configured padded target length instead. The extra
    padding positions are ignored by NRTRLoss, while the graph remains static
    and exportable by PT2E and ONNX.
    """

    output_names = ("ctc", "ctc_neck", "gtc")

    def __init__(self, model, max_text_length, expose_intermediates=False):
        super().__init__()
        self.model = model
        self.graph_role = "pretrained_train"
        self.model_type = "rec"
        self.max_text_length = int(max_text_length)
        self.expose_intermediates = bool(expose_intermediates)
        if self.max_text_length < 2:
            raise ValueError("max_text_length must be at least 2.")
        if isinstance(model.head.gtc_head, str):
            raise ValueError("Full recognition export requires an NRTR head.")

    def forward(self, images, gtc_targets):
        features = images
        if self.model.use_transform:
            features = self.model.transform(features)
        backbone_out = self.model.backbone(features)
        features = backbone_out
        if self.model.use_neck:
            features = self.model.neck(features)

        ctc_neck = self.model.head.ctc_encoder(features)
        ctc = self.model.head.ctc_head(ctc_neck)
        padded_targets = gtc_targets[:, : self.max_text_length]
        gtc = self.model.head.gtc_head.forward_train(
            self.model.head.before_gtc(features),
            padded_targets,
        )
        if not self.expose_intermediates:
            return ctc, ctc_neck, gtc
        return {
            "ctc": ctc,
            "ctc_neck": ctc_neck,
            "gtc": gtc,
            "backbone_out": backbone_out,
            "neck_out": features,
        }

    def set_qat_capture_mode(self):
        self.train()
        return self

    def set_validation_mode(self):
        self.eval()
        # CTCHead otherwise applies Softmax in eval mode. Training ONNX keeps
        # raw logits, matching CTCLoss and the native full-training graph.
        self.model.head.ctc_head.training = True
        return self


def reparameterize_for_deploy(model, insert_identity_bn=False):
    model.eval()
    for module_name in ("backbone", "neck"):
        module = getattr(model, module_name, None)
        rep = getattr(module, "rep", None)
        if callable(rep):
            try:
                rep(insert_identity_bn=insert_identity_bn)
            except TypeError:
                rep()
    return model


def load_axera_quantizer(config_path, legacy_config_path=None):
    # Preserve the former (axera_root, config_path) API while callers migrate
    # to the self-contained vendored quantizer.
    if legacy_config_path is not None:
        config_path = legacy_config_path
    from . import quantized_decomposed_dequantize_per_channel  # noqa: F401

    config_file = str(Path(config_path).resolve())
    lsq = _qat_config_flag(config_file, "lsq", False)
    if lsq:
        from .ax_quantizer_lsq import AXQuantizer
    else:
        from .ax_quantizer import AXQuantizer

    quantizer = AXQuantizer(config_file)
    return AxeraQuantizerAdapter(quantizer)


def _qat_config_flag(config_file, key, default=False):
    """Read a top-level boolean flag from an Axera QAT JSON config."""
    try:
        with open(config_file, encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, ValueError):
        return default
    value = config.get(key, default)
    return bool(value)


def is_lsq_config(config_path):
    """Whether the QAT JSON enables the LSQ quantizer (top-level "lsq": true)."""
    return _qat_config_flag(str(config_path), "lsq", False)


def enable_learn(module):
    """Enable learnable-scale QAT on a module, if applicable.

    Scale-only LSQ: the zero_point parameter stays frozen at its static value.
    Learning both scale and zero_point from a zero-initialized zero_point
    diverges quickly for affine U16 domains (loss explodes within a few
    training steps).
    """
    from torch.ao.quantization._learnable_fake_quantize import (
        _LearnableFakeQuantize,
    )

    if isinstance(module, _LearnableFakeQuantize):
        module.enable_param_learning()
        # module.zero_point.requires_grad = False


def disable_learn(module):
    """Disable learnable-scale QAT on a module, if applicable."""
    from torch.ao.quantization._learnable_fake_quantize import (
        _LearnableFakeQuantize,
    )

    if isinstance(module, _LearnableFakeQuantize):
        module.toggle_qparam_learning(False)


def build_qat_dynamic_shapes(
    images,
    *,
    dynamic_batch=False,
    dynamic_heights=None,
    batch_aligned_inputs=0,
    max_batch=None,
):
    """Build the dynamic shape contract for the prepared PT2E training graph."""
    dimensions = {}
    batch_dimension = None
    if dynamic_batch:
        if images.shape[0] > 1:
            batch_kwargs = {"min": 1}
            if max_batch is not None:
                if int(max_batch) < int(images.shape[0]):
                    raise ValueError("Dynamic max batch cannot be below capture batch.")
                batch_kwargs["max"] = int(max_batch)
            batch_dimension = torch.export.Dim("batch", **batch_kwargs)
        else:
            batch_dimension = torch.export.Dim.AUTO
        dimensions[0] = batch_dimension

    if dynamic_heights:
        heights = tuple(int(height) for height in dynamic_heights)
        if len(heights) < 2 or tuple(sorted(set(heights))) != heights:
            raise ValueError(
                "Dynamic heights must be at least two sorted unique values."
            )
        if images.ndim != 4 or images.shape[2] not in heights:
            raise ValueError(
                "The capture image height must be included in dynamic heights."
            )
        step = heights[1] - heights[0]
        if step <= 0 or heights != tuple(
            range(heights[0], heights[-1] + step, step)
        ):
            raise ValueError("Dynamic heights must form one arithmetic sequence.")
        divisor = math.gcd(*heights)
        factors = tuple(height // divisor for height in heights)
        represented_heights = tuple(divisor * factor for factor in factors)
        consecutive_factors = tuple(range(factors[0], factors[-1] + 1))
        if represented_heights != heights or factors != consecutive_factors:
            raise ValueError(
                "Dynamic heights cannot be represented as one derived Torch Dim."
            )
        factor = torch.export.Dim(
            "height_factor",
            min=factors[0],
            max=factors[-1],
        )
        dimensions[2] = divisor * factor

    if not dimensions:
        if batch_aligned_inputs:
            raise ValueError("Batch-aligned inputs require a dynamic batch dimension.")
        return None
    shapes = [dimensions]
    shapes.extend({0: batch_dimension} for _ in range(int(batch_aligned_inputs)))
    return tuple(shapes)


def prepare_qat_model(
    model,
    example_inputs,
    quantizer,
    dynamic_shapes=None,
    freeze_kept_bn_stats=False,
):
    exported = torch.export.export_for_training(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    ).module()
    float_node_count = len(list(exported.graph.nodes))
    prepared = prepare_qat_pt2e(exported, quantizer)
    if freeze_kept_bn_stats:
        freeze_kept_bn_running_stats(prepared)
    move_exported_model_to_train(prepared)
    if freeze_kept_bn_stats:
        # move_exported_model_to_train may rebuild the graph module; re-apply
        # the momentum pin afterwards so the frozen nodes persist.
        freeze_kept_bn_running_stats(prepared)
    return prepared, float_node_count


def freeze_kept_bn_running_stats(prepared):
    """Pin kept-BN momentum to 0.0 in the prepared PT2E graph.

    The eager BatchNorm momentum does not survive export: the prepared graph
    carries batch_norm as a call_function node with a constant momentum
    argument. Kept BNs (the ones inserted after rep-fused convs) are
    identified by their running_mean attribute name containing ``_blocks``.
    momentum=0.0 keeps running stats fixed during QAT training (locking the
    folding denominator gamma/sqrt(var+eps)) while training still normalizes
    with batch stats and gamma/beta stay trainable.
    """
    frozen = 0
    for node in prepared.graph.nodes:
        if node.op != "call_function" or "batch_norm" not in str(node.target):
            continue
        if len(node.args) < 7:
            continue
        running_mean_target = str(node.args[3])
        if "_blocks" not in running_mean_target:
            continue
        updated_args = list(node.args)
        updated_args[6] = 0.0
        node.args = tuple(updated_args)
        frozen += 1
    prepared.graph.lint()
    prepared.recompile()
    return frozen


def initialize_weight_observers(prepared):
    """Initialize weight and scalar-parameter observers without a model run."""
    initialized = []
    visited = set()
    static_cache = {}
    for node in prepared.graph.nodes:
        if node.op != "call_module":
            continue
        module = prepared.get_submodule(str(node.target))
        source = node.args[0] if node.args else None
        if not isinstance(source, Node):
            continue
        is_per_channel = getattr(module, "qscheme", None) in _PER_CHANNEL_QSCHEMES
        is_static_scalar = False
        if source.op == "get_attr":
            try:
                value = _get_exported_attribute(prepared, str(source.target))
            except AttributeError:
                pass
            else:
                is_static_scalar = (
                    isinstance(value, torch.Tensor)
                    and value.dtype.is_floating_point
                    and value.numel() == 1
                )
        if not is_per_channel and not is_static_scalar:
            continue
        if id(module) in visited:
            continue
        weight = _resolve_static_fx_value(prepared, source, static_cache)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(
                f"Per-channel observer {node.target} input is not a tensor."
            )
        with torch.no_grad():
            module(weight.detach())
        initialized.append(
            {
                "observer": str(node.target),
                "weight": (
                    str(source.target) if source.op == "get_attr" else source.name
                ),
                "channels": (
                    int(weight.shape[getattr(module, "ch_axis", 0)])
                    if is_per_channel
                    else 1
                ),
            }
        )
        visited.add(id(module))
    return initialized


def run_smoke_step(prepared, images, learning_rate=1.0e-6):
    move_exported_model_to_train(prepared)
    optimizer = torch.optim.SGD(prepared.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    outputs = prepared(images)
    output_tensors = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    loss = sum(output.square().mean() for output in output_tensors)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in prepared.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("QAT smoke step produced missing or non-finite gradients.")
    optimizer.step()
    detached_outputs = tuple(output.detach() for output in output_tensors)
    if not isinstance(outputs, (tuple, list)):
        detached_outputs = detached_outputs[0]
    return detached_outputs, float(loss.detach()), len(gradients)


def convert_prepared_model(prepared):
    model = copy.deepcopy(prepared)
    converted = convert_pt2e(model)
    move_exported_model_to_eval(converted)
    # convert_pt2e rebuilds the graph module; preserve graph identity
    # attributes so trainers/evaluators can detect full-recognition graphs.
    for attribute in ("graph_role", "model_type", "output_names"):
        if hasattr(prepared, attribute):
            setattr(converted, attribute, getattr(prepared, attribute))
    return converted


def initialize_kept_bn_statistics(
    model,
    loader,
    steps,
    rec_multi_head=False,
    momentum=0.9,
    device=None,
    training_momentum=None,
    freeze_bn_stats=False,
):
    """Measure fused-conv output statistics for kept identity BNs.

    QARepVGG insert_bn style: run ``steps`` batches through the eager
    (reparameterized) model, accumulate the per-channel mean/variance of every
    kept BN's input (the fused conv output), then set

        running_mean = measured mean
        running_var  = measured variance
        gamma        = sqrt(running_var + eps)
        beta         = measured mean

    which makes ``BN(conv_fused(x)) == conv_fused(x)`` an identity mapping, so
    the reparameterized conv+bn matches the original multi-branch output
    exactly and QAT training/validation statistics agree.

    ``training_momentum`` optionally overrides the BatchNorm momentum used
    during QAT training: a larger value makes running_var track the trained
    gamma faster. Tiny measured variances (e.g. 1e-11) keep gamma/sqrt(var+eps)
    near 1 only while running_var stays fresh; with the default 0.1 momentum
    the running_var lags behind the learned gamma and eval (running-stats)
    inference amplifies those channels (up to NaN).

    ``freeze_bn_stats`` pins the BatchNorm momentum to 0.0 so the measured
    running_mean/running_var never update during QAT training. This locks the
    folding denominator gamma/sqrt(var+eps) to its initialization value and
    prevents the "BN folding explosion" caused by gamma drifting away from a
    stale (tiny) running_var; gamma and beta remain trainable per-channel
    linear scales.

    Returns the number of kept BNs that were initialized.
    """
    from pytorchocr.modeling.backbones.rec_lcnetv3 import LearnableRepLayer

    layers = [
        module
        for module in model.modules()
        if isinstance(module, LearnableRepLayer) and module.keep_bn
    ]
    if not layers:
        return 0
    for layer in layers:
        layer.keep_bn = False
    accumulated = {}
    handles = []
    for index, layer in enumerate(layers):
        key = f"bn_{index}"
        accumulated[key] = {
            "mean": torch.zeros(
                layer.out_channels,
                device=layer.reparam_conv.weight.device,
            ),
            "var": torch.zeros(
                layer.out_channels,
                device=layer.reparam_conv.weight.device,
            ),
            "count": 0,
        }

        def make_hook(key):
            def hook(module, inputs, outputs):
                value = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                value = value.detach().float()
                mean = value.mean(dim=(0, 2, 3))
                var = ((value - mean.view(1, -1, 1, 1)) ** 2).mean(dim=(0, 2, 3))
                state = accumulated[key]
                if state["count"] == 0:
                    state["mean"].copy_(mean)
                    state["var"].copy_(var)
                else:
                    state["mean"].mul_(momentum).add_(mean * (1.0 - momentum))
                    state["var"].mul_(momentum).add_(var * (1.0 - momentum))
                state["count"] += 1

            return hook

        handles.append(layer.reparam_conv.register_forward_hook(make_hook(key)))
    try:
        for _ in range(int(steps)):
            images, batch_targets = next(iter(loader))
            if device is not None:
                images = images.to(device, non_blocking=True)
                batch_targets = {
                    key: (
                        value.to(device, non_blocking=True)
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in batch_targets.items()
                }
            with torch.no_grad():
                if rec_multi_head:
                    model(images, batch_targets["gtc_targets"])
                else:
                    model(images)
    finally:
        for handle in handles:
            handle.remove()
    for index, layer in enumerate(layers):
        key = f"bn_{index}"
        state = accumulated[key]
        if state["count"] == 0:
            raise RuntimeError("Kept BN statistics pass consumed no batches.")
        bn = layer.bn
        # Variance floor: tiny measured variances (e.g. 1e-11 from dead
        # channels) make gamma / sqrt(var+eps) blow up once gamma drifts
        # during QAT training (BN folding explosion, ratio up to ~2000).
        # Clamping the variance bounds the folding coefficient; gamma is
        # initialized from the floored variance so the identity mapping
        # BN(conv_fused(x)) == conv_fused(x) is preserved.
        safe_var = torch.clamp(state["var"], min=1e-4)
        bn.running_mean.copy_(state["mean"])
        bn.running_var.copy_(safe_var)
        bn.weight.data.copy_((safe_var + bn.eps).sqrt())
        bn.bias.data.copy_(state["mean"])
        if freeze_bn_stats:
            # momentum=0.0 keeps running stats fixed while training still
            # normalizes with batch stats and gamma/beta stay trainable.
            bn.momentum = 0.0
        elif training_momentum is not None:
            bn.momentum = float(training_momentum)
        layer.keep_bn = True
    return len(layers)

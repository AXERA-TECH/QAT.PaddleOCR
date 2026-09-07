"""PT2E QAT integration used by PyTorchOCR training and diagnostics."""

from .bridge import (
    AxeraQuantizerAdapter,
    DetInferenceWrapper,
    DetTrainingWrapper,
    FullDetTrainingWrapper,
    FullRecTrainingWrapper,
    RecCTCWrapper,
    build_qat_dynamic_shapes,
    convert_prepared_model,
    disable_learn,
    enable_learn,
    freeze_kept_bn_running_stats,
    initialize_kept_bn_statistics,
    initialize_weight_observers,
    is_lsq_config,
    load_axera_quantizer,
    prepare_qat_model,
    reparameterize_for_deploy,
    run_smoke_step,
)

from .folding import (
    apply_folded_state,
    build_folded_state,
    checkpoint_qat_config,
    collect_quantized_weight_map,
    folded_eager_model,
)
from .qparams import (
    activation_qparam_sites,
    transfer_activation_qparams_by_site,
)


def numpy_error_stats(*args, **kwargs):
    from .validation import numpy_error_stats as implementation

    return implementation(*args, **kwargs)


def qdq_stats(*args, **kwargs):
    from .validation import qdq_stats as implementation

    return implementation(*args, **kwargs)


def sequence_error_stats(*args, **kwargs):
    from .validation import sequence_error_stats as implementation

    return implementation(*args, **kwargs)


def run_ort(*args, **kwargs):
    from .validation import run_ort as implementation

    return implementation(*args, **kwargs)


def run_onnx_reference(*args, **kwargs):
    from .validation import run_onnx_reference as implementation

    return implementation(*args, **kwargs)


def validate_qdq_graph(*args, **kwargs):
    from .validation import validate_qdq_graph as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "AxeraQuantizerAdapter",
    "DetInferenceWrapper",
    "DetTrainingWrapper",
    "FullDetTrainingWrapper",
    "FullRecTrainingWrapper",
    "RecCTCWrapper",
    "build_qat_dynamic_shapes",
    "convert_prepared_model",
    "freeze_kept_bn_running_stats",
    "initialize_kept_bn_statistics",
    "disable_learn",
    "enable_learn",
    "initialize_weight_observers",
    "is_lsq_config",
    "load_axera_quantizer",
    "prepare_qat_model",
    "reparameterize_for_deploy",
    "run_smoke_step",
    "apply_folded_state",
    "build_folded_state",
    "checkpoint_qat_config",
    "collect_quantized_weight_map",
    "folded_eager_model",
    "activation_qparam_sites",
    "transfer_activation_qparams_by_site",
    "numpy_error_stats",
    "qdq_stats",
    "sequence_error_stats",
    "run_ort",
    "run_onnx_reference",
    "validate_qdq_graph",
]

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
    initialize_weight_observers,
    load_axera_quantizer,
    prepare_qat_model,
    reparameterize_for_deploy,
    run_smoke_step,
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
    "initialize_weight_observers",
    "load_axera_quantizer",
    "prepare_qat_model",
    "reparameterize_for_deploy",
    "run_smoke_step",
    "numpy_error_stats",
    "qdq_stats",
    "sequence_error_stats",
    "run_ort",
    "run_onnx_reference",
    "validate_qdq_graph",
]

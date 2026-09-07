from .checkpoint import build_prepared_qat_checkpoint, load_qat_checkpoint
from .contract import (
    audit_detection_label,
    audit_recognition_label,
    json_hash,
    load_characters,
    model_contract,
    stable_debug_selection,
    validate_audit,
)
from .data import build_diagnostic_dataset, sample_ids
from .errors import (
    ErrorAccumulator,
    compare_output_sets,
    compute_recognition_pair,
    ctc_collapse,
    new_recognition_pair,
    outputs_as_tuple,
    update_recognition_pair,
)
from .qat import (
    fake_quant_state,
    observer_qparams,
    operator_delta,
    prepared_random_stage_outputs,
    random_stage_comparison,
    state_preservation,
    tensor_record,
)
from .recognition import evaluate_recognition_qat
from .onnx import resolve_onnx_input_shape, static_onnx_input_shape
from .pretrained import compare_gradient_maps, tensor_difference

__all__ = [
    "ErrorAccumulator",
    "audit_detection_label",
    "audit_recognition_label",
    "build_diagnostic_dataset",
    "build_prepared_qat_checkpoint",
    "compute_recognition_pair",
    "compare_output_sets",
    "compare_gradient_maps",
    "ctc_collapse",
    "evaluate_recognition_qat",
    "fake_quant_state",
    "json_hash",
    "load_characters",
    "load_qat_checkpoint",
    "new_recognition_pair",
    "observer_qparams",
    "operator_delta",
    "prepared_random_stage_outputs",
    "random_stage_comparison",
    "outputs_as_tuple",
    "sample_ids",
    "model_contract",
    "resolve_onnx_input_shape",
    "stable_debug_selection",
    "state_preservation",
    "static_onnx_input_shape",
    "tensor_record",
    "tensor_difference",
    "update_recognition_pair",
    "validate_audit",
]

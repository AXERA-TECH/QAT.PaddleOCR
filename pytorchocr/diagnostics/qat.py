import copy
from collections import Counter

import torch
from torch.ao.quantization import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    move_exported_model_to_eval,
)

from .errors import compare_output_sets, outputs_as_tuple


def prepared_random_stage_outputs(prepared, example_inputs):
    """Run frozen prepared fake-off/on stages without mutating the source graph."""
    results = {}
    for stage, fake_quant_enabled in (
        ("prepared_fake_off", False),
        ("prepared_fake_on", True),
    ):
        model = copy.deepcopy(prepared)
        move_exported_model_to_eval(model)
        model.apply(disable_observer)
        model.apply(enable_fake_quant if fake_quant_enabled else disable_fake_quant)
        with torch.no_grad():
            outputs = outputs_as_tuple(model(*example_inputs))
        results[stage] = tuple(
            output.detach().cpu().clone() for output in outputs
        )
        del model
    return results


def random_stage_comparison(
    prepared_outputs,
    converted_outputs,
    output_names,
    quantonnx_outputs=None,
):
    result = {
        "input": "deterministic_random",
        "observer_state": "frozen_at_export",
        "comparisons": {
            "prepared_fake_off_to_fake_on": compare_output_sets(
                prepared_outputs["prepared_fake_off"],
                prepared_outputs["prepared_fake_on"],
                output_names,
            ),
            "prepared_fake_on_to_converted_pt2e": compare_output_sets(
                prepared_outputs["prepared_fake_on"],
                converted_outputs,
                output_names,
            ),
            "prepared_fake_off_to_converted_pt2e": compare_output_sets(
                prepared_outputs["prepared_fake_off"],
                converted_outputs,
                output_names,
            ),
        },
    }
    if quantonnx_outputs is not None:
        result["comparisons"]["converted_pt2e_to_quantonnx"] = (
            compare_output_sets(
                converted_outputs,
                quantonnx_outputs,
                output_names,
            )
        )
        result["comparisons"]["prepared_fake_on_to_quantonnx"] = (
            compare_output_sets(
                prepared_outputs["prepared_fake_on"],
                quantonnx_outputs,
                output_names,
            )
        )
    return result


def state_preservation(exported, prepared):
    exported_state = exported.state_dict()
    prepared_state = prepared.state_dict()
    common = sorted(set(exported_state).intersection(prepared_state))
    changed = []
    for name in common:
        reference = exported_state[name]
        actual = prepared_state[name]
        if torch.equal(reference, actual):
            continue
        difference = (reference.to(torch.float64) - actual.to(torch.float64)).abs()
        changed.append({"name": name, "max_abs": float(difference.max())})
    return {
        "exported_keys": len(exported_state),
        "prepared_keys": len(prepared_state),
        "common_keys": len(common),
        "changed_common_keys": changed,
    }


def operator_delta(exported, prepared):
    def counts(model):
        return Counter(
            str(node.target)
            for node in model.graph.nodes
            if node.op == "call_function"
        )

    exported_counts = counts(exported)
    prepared_counts = counts(prepared)
    return {
        name: prepared_counts[name] - exported_counts[name]
        for name in sorted(set(exported_counts).union(prepared_counts))
        if prepared_counts[name] != exported_counts[name]
    }


def fake_quant_state(model):
    modules = []
    for name, module in model.named_modules():
        if not hasattr(module, "fake_quant_enabled"):
            continue
        fake_quant_enabled = bool(module.fake_quant_enabled.detach().cpu().item())
        observer_enabled = bool(module.observer_enabled.detach().cpu().item())
        modules.append(
            {
                "name": name,
                "type": type(module).__name__,
                "fake_quant_enabled": fake_quant_enabled,
                "observer_enabled": observer_enabled,
            }
        )
    return {
        "modules": len(modules),
        "fake_quant_enabled": sum(item["fake_quant_enabled"] for item in modules),
        "observer_enabled": sum(item["observer_enabled"] for item in modules),
        "module_states": modules,
    }


def tensor_record(value):
    if value is None or not torch.is_tensor(value):
        return None
    tensor = value.detach().cpu()
    floating = tensor.to(torch.float64)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(floating).all()),
        "values": tensor.reshape(-1).tolist(),
    }


def observer_qparams(model):
    records = []
    for name, module in model.named_modules():
        if not hasattr(module, "fake_quant_enabled"):
            continue
        observer = getattr(module, "activation_post_process", None)
        records.append(
            {
                "name": name,
                "fake_quant_type": type(module).__name__,
                "observer_type": (
                    type(observer).__name__ if observer is not None else None
                ),
                "dtype": str(getattr(module, "dtype", None)),
                "qscheme": str(getattr(module, "qscheme", None)),
                "quant_min": getattr(module, "quant_min", None),
                "quant_max": getattr(module, "quant_max", None),
                "scale": tensor_record(getattr(module, "scale", None)),
                "zero_point": tensor_record(getattr(module, "zero_point", None)),
                "min_val": tensor_record(getattr(observer, "min_val", None)),
                "max_val": tensor_record(getattr(observer, "max_val", None)),
            }
        )
    invalid = []
    for record in records:
        for field in ("scale", "zero_point", "min_val", "max_val"):
            value = record[field]
            if value is not None and not value["finite"]:
                invalid.append({"name": record["name"], "field": field})
    return {"modules": len(records), "nonfinite": invalid, "records": records}

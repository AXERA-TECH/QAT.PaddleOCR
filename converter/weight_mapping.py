from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WeightMappingReport:
    source_count: int
    target_count: int
    copied_count: int
    allowed_missing_targets: tuple
    ignored_source_count: int


def paddle_to_torch_name(name):
    return name.replace("._mean", ".running_mean").replace(
        "._variance", ".running_var"
    )


def copy_paddle_state_dict_strict(
    module,
    paddle_state_dict,
    *,
    ignored_source_keys=(),
    allowed_missing_target_suffixes=("num_batches_tracked",),
    allowed_missing_target_prefixes=(),
    source_transpose_suffixes=(),
):
    """Validate a one-to-one Paddle-to-PyTorch mapping before copying."""
    target_state = module.state_dict()
    ignored_source_keys = set(ignored_source_keys)
    pending = {}
    unknown_sources = []
    duplicate_targets = []
    shape_mismatches = []

    ignored_source_count = 0
    for source_name, source_value in paddle_state_dict.items():
        if source_name in ignored_source_keys:
            ignored_source_count += 1
            continue
        target_name = paddle_to_torch_name(source_name)
        if target_name not in target_state:
            unknown_sources.append(source_name)
            continue
        if target_name in pending:
            duplicate_targets.append(target_name)
            continue
        source_array = source_value.numpy()
        if source_name.endswith(tuple(source_transpose_suffixes)):
            source_array = source_array.T
        source_tensor = torch.as_tensor(source_array)
        target_tensor = target_state[target_name]
        if source_tensor.shape != target_tensor.shape:
            shape_mismatches.append(
                f"{source_name} -> {target_name}: "
                f"source={tuple(source_tensor.shape)}, "
                f"target={tuple(target_tensor.shape)}"
            )
            continue
        pending[target_name] = source_tensor.to(dtype=target_tensor.dtype)

    missing_targets = sorted(set(target_state) - set(pending))
    allowed_missing = sorted(
        name
        for name in missing_targets
        if name.endswith(tuple(allowed_missing_target_suffixes))
        or name.startswith(tuple(allowed_missing_target_prefixes))
    )
    invalid_missing = sorted(set(missing_targets) - set(allowed_missing))
    if unknown_sources or duplicate_targets or shape_mismatches or invalid_missing:
        raise RuntimeError(
            "Paddle/PyTorch weight mapping is not one-to-one: "
            f"unknown_sources={sorted(unknown_sources)}, "
            f"duplicate_targets={sorted(duplicate_targets)}, "
            f"shape_mismatches={sorted(shape_mismatches)}, "
            f"missing_targets={invalid_missing}"
        )

    with torch.no_grad():
        for target_name, source_tensor in pending.items():
            target_state[target_name].copy_(source_tensor)

    return WeightMappingReport(
        source_count=len(paddle_state_dict) - ignored_source_count,
        target_count=len(target_state),
        copied_count=len(pending),
        allowed_missing_targets=tuple(allowed_missing),
        ignored_source_count=ignored_source_count,
    )

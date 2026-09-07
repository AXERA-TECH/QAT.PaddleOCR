import torch


class ErrorAccumulator:
    def __init__(self):
        self.absolute_sum = 0.0
        self.count = 0
        self.max_abs = 0.0

    def update(self, reference, actual):
        if reference.shape != actual.shape:
            raise ValueError(
                f"Output shape mismatch: {tuple(reference.shape)} != "
                f"{tuple(actual.shape)}"
            )
        difference = (reference.float() - actual.float()).abs()
        self.absolute_sum += float(difference.sum())
        self.count += difference.numel()
        self.max_abs = max(self.max_abs, float(difference.max()))

    def compute(self):
        if self.count == 0:
            raise ValueError("Error accumulator has no values.")
        return {
            "mae": self.absolute_sum / self.count,
            "max_abs": self.max_abs,
        }


def outputs_as_tuple(outputs):
    """Normalize model outputs to a tuple of deployment task tensors.

    Dict outputs come from the KD intermediate-exposure wrappers
    (FullRecTrainingWrapper / DetTrainingWrapper with expose_intermediates):
    recognition picks the CTC logits, detection picks the shrink map. The
    bare tensor/tuple paths are unchanged.
    """
    if isinstance(outputs, dict):
        if "ctc" in outputs:
            return (outputs["ctc"],)
        head = outputs.get("head_out")
        if isinstance(head, dict) and "ctc" in head:
            return (head["ctc"],)
        if "maps" in outputs:
            maps = outputs["maps"]
            shrink = maps[0] if isinstance(maps, (tuple, list)) else maps[:, :1]
            return (shrink,)
        raise KeyError(f"Outputs dict has no task output: {sorted(outputs)}")
    return tuple(outputs) if isinstance(outputs, (tuple, list)) else (outputs,)


def compare_output_sets(reference, actual, output_names=None):
    reference_outputs = outputs_as_tuple(reference)
    actual_outputs = outputs_as_tuple(actual)
    if len(reference_outputs) != len(actual_outputs):
        raise ValueError(
            "Output count mismatch: "
            f"{len(reference_outputs)} != {len(actual_outputs)}"
        )
    names = tuple(output_names or ())
    if names and len(names) != len(reference_outputs):
        raise ValueError(
            f"Output name count mismatch: {len(names)} != {len(reference_outputs)}"
        )
    records = []
    for index, (expected, observed) in enumerate(
        zip(reference_outputs, actual_outputs)
    ):
        expected = torch.as_tensor(expected).detach().cpu()
        observed = torch.as_tensor(observed).detach().cpu()
        if expected.shape != observed.shape:
            raise ValueError(
                f"Output {index} shape mismatch: {tuple(expected.shape)} != "
                f"{tuple(observed.shape)}"
            )
        difference = (expected.float() - observed.float()).abs()
        record = {
            "name": names[index] if names else str(index),
            "shape": list(expected.shape),
            "mae": float(difference.mean()),
            "max_abs": float(difference.max()),
            "finite": bool(torch.isfinite(observed).all()),
        }
        if expected.ndim >= 2 and expected.shape[-1] > 1:
            record["argmax_agreement"] = float(
                (expected.argmax(dim=-1) == observed.argmax(dim=-1))
                .float()
                .mean()
            )
        records.append(record)
    return records


def ctc_collapse(indices):
    sequence = []
    previous = None
    for raw_index in indices:
        index = int(raw_index)
        duplicate = previous == index
        previous = index
        if index != 0 and not duplicate:
            sequence.append(index)
    return tuple(sequence)


def new_recognition_pair():
    return {
        "logits": ErrorAccumulator(),
        "centered_logits": ErrorAccumulator(),
        "probabilities": ErrorAccumulator(),
        "argmax_matches": 0,
        "argmax_values": 0,
        "sequence_matches": 0,
        "sequence_values": 0,
    }


def update_recognition_pair(stats, reference, actual):
    reference = reference.float()
    actual = actual.float()
    stats["logits"].update(reference, actual)
    stats["centered_logits"].update(
        reference - reference.mean(dim=-1, keepdim=True),
        actual - actual.mean(dim=-1, keepdim=True),
    )
    stats["probabilities"].update(
        reference.softmax(dim=-1),
        actual.softmax(dim=-1),
    )
    reference_indices = reference.argmax(dim=-1)
    actual_indices = actual.argmax(dim=-1)
    stats["argmax_matches"] += int((reference_indices == actual_indices).sum())
    stats["argmax_values"] += reference_indices.numel()
    for reference_sequence, actual_sequence in zip(reference_indices, actual_indices):
        stats["sequence_matches"] += int(
            ctc_collapse(reference_sequence) == ctc_collapse(actual_sequence)
        )
        stats["sequence_values"] += 1


def compute_recognition_pair(stats):
    return {
        "logits": stats["logits"].compute(),
        "centered_logits": stats["centered_logits"].compute(),
        "probabilities": stats["probabilities"].compute(),
        "argmax_agreement": stats["argmax_matches"] / stats["argmax_values"],
        "ctc_sequence_agreement": (
            stats["sequence_matches"] / stats["sequence_values"]
        ),
    }

import math

import numpy as np


class GradientDifferenceAccumulator:
    def __init__(self):
        self.tensor_count = 0
        self.element_count = 0
        self.sum_abs = 0.0
        self.max_abs = 0.0
        self.reference_square_sum = 0.0
        self.candidate_square_sum = 0.0
        self.dot_sum = 0.0
        self.direction_disagreement_count = 0
        self.meaningful_direction_count = 0
        self.meaningful_direction_disagreement_count = 0

    def update(self, reference, candidate):
        reference = np.asarray(reference, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        if reference.shape != candidate.shape:
            raise ValueError(
                f"Gradient shape mismatch: {reference.shape} != {candidate.shape}"
            )
        difference = np.abs(reference - candidate)
        self.tensor_count += 1
        self.element_count += difference.size
        self.sum_abs += float(difference.sum())
        self.max_abs = max(self.max_abs, float(difference.max(initial=0.0)))
        self.reference_square_sum += float(np.square(reference).sum())
        self.candidate_square_sum += float(np.square(candidate).sum())
        self.dot_sum += float((reference * candidate).sum())
        direction_disagreement = reference * candidate < 0
        self.direction_disagreement_count += int(direction_disagreement.sum())
        meaningful_direction = np.maximum(np.abs(reference), np.abs(candidate)) >= 1e-8
        self.meaningful_direction_count += int(meaningful_direction.sum())
        self.meaningful_direction_disagreement_count += int(
            np.logical_and(direction_disagreement, meaningful_direction).sum()
        )

    def result(self):
        reference_norm = math.sqrt(self.reference_square_sum)
        candidate_norm = math.sqrt(self.candidate_square_sum)
        denominator = reference_norm * candidate_norm
        difference_l2 = math.sqrt(
            max(
                self.reference_square_sum
                + self.candidate_square_sum
                - 2.0 * self.dot_sum,
                0.0,
            )
        )
        return {
            "tensor_count": self.tensor_count,
            "element_count": self.element_count,
            "mae": (
                self.sum_abs / self.element_count if self.element_count else 0.0
            ),
            "max_abs": self.max_abs,
            "reference_l2": reference_norm,
            "candidate_l2": candidate_norm,
            "relative_l2": difference_l2 / max(reference_norm, 1.0e-30),
            "cosine_similarity": self.dot_sum / denominator if denominator else None,
            "direction_disagreement_count": self.direction_disagreement_count,
            "direction_disagreement_fraction": (
                self.direction_disagreement_count / self.element_count
                if self.element_count
                else 0.0
            ),
            "meaningful_direction_count": self.meaningful_direction_count,
            "meaningful_direction_disagreement_count": (
                self.meaningful_direction_disagreement_count
            ),
            "meaningful_direction_disagreement_fraction": (
                self.meaningful_direction_disagreement_count
                / self.meaningful_direction_count
                if self.meaningful_direction_count
                else 0.0
            ),
        }


def tensor_difference(reference, candidate):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Tensor shape mismatch: {reference.shape} != {candidate.shape}"
        )
    difference = np.abs(
        reference.astype(np.float64) - candidate.astype(np.float64)
    )
    reference_abs_mean = float(np.abs(reference.astype(np.float64)).mean())
    return {
        "shape": list(reference.shape),
        "finite": bool(
            np.isfinite(reference).all() and np.isfinite(candidate).all()
        ),
        "mae": float(difference.mean()),
        "relative_mae": float(difference.mean())
        / max(reference_abs_mean, 1.0e-30),
        "p99": float(np.quantile(difference, 0.99)),
        "max_abs": float(difference.max(initial=0.0)),
        "reference_abs_mean": reference_abs_mean,
    }


def compare_gradient_maps(
    reference,
    candidate,
    *,
    owners,
    transpose_reference=(),
):
    transpose_reference = set(transpose_reference)
    owner_stats = {owner: GradientDifferenceAccumulator() for owner in owners}
    overall = GradientDifferenceAccumulator()
    reference_names = set(reference)
    candidate_names = set(candidate)
    missing_candidate = sorted(reference_names - candidate_names)
    missing_reference = sorted(candidate_names - reference_names)
    shape_mismatches = []
    worst_tensors = []

    for name in sorted(reference_names & candidate_names):
        reference_value = np.asarray(reference[name])
        if name in transpose_reference:
            reference_value = reference_value.T
        candidate_value = np.asarray(candidate[name])
        if reference_value.shape != candidate_value.shape:
            shape_mismatches.append(
                {
                    "name": name,
                    "reference_shape": list(reference_value.shape),
                    "candidate_shape": list(candidate_value.shape),
                }
            )
            continue
        matching_owners = [owner for owner in owners if name.startswith(owner)]
        if len(matching_owners) != 1:
            raise ValueError(
                f"Gradient {name!r} matched owners {matching_owners}; expected one."
            )
        owner_stats[matching_owners[0]].update(reference_value, candidate_value)
        overall.update(reference_value, candidate_value)
        difference = np.abs(reference_value.astype(np.float64) - candidate_value)
        worst_tensors.append(
            {
                "name": name,
                "shape": list(reference_value.shape),
                "mae": float(difference.mean()),
                "max_abs": float(difference.max(initial=0.0)),
                "reference_l2": float(np.linalg.norm(reference_value)),
                "candidate_l2": float(np.linalg.norm(candidate_value)),
            }
        )

    return {
        "overall": overall.result(),
        "owners": {
            owner.rstrip("."): stats.result()
            for owner, stats in owner_stats.items()
        },
        "missing_candidate": missing_candidate,
        "missing_reference": missing_reference,
        "shape_mismatches": shape_mismatches,
        "worst_tensors": sorted(
            worst_tensors,
            key=lambda item: (item["max_abs"], item["mae"]),
            reverse=True,
        )[:20],
    }

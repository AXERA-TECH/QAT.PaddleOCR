import unittest

import numpy as np
import torch
from torch import nn

from converter.weight_mapping import copy_paddle_state_dict_strict


class _PaddleValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


class WeightMappingTest(unittest.TestCase):
    def test_maps_batch_norm_statistics_and_allows_counter(self):
        module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        source = {
            "0.weight": _PaddleValue(np.ones((4, 3, 1, 1), dtype=np.float32)),
            "0.bias": _PaddleValue(np.ones(4, dtype=np.float32)),
            "1.weight": _PaddleValue(np.ones(4, dtype=np.float32)),
            "1.bias": _PaddleValue(np.zeros(4, dtype=np.float32)),
            "1._mean": _PaddleValue(np.full(4, 2.0, dtype=np.float32)),
            "1._variance": _PaddleValue(np.full(4, 3.0, dtype=np.float32)),
        }

        report = copy_paddle_state_dict_strict(module, source)

        self.assertEqual(report.copied_count, 6)
        self.assertEqual(report.allowed_missing_targets, ("1.num_batches_tracked",))
        torch.testing.assert_close(module[1].running_mean, torch.full((4,), 2.0))

    def test_rejects_unknown_missing_and_shape_mismatch_before_copy(self):
        module = nn.Linear(2, 3)
        original_weight = module.weight.detach().clone()
        source = {
            "weight": _PaddleValue(np.ones((2, 3), dtype=np.float32)),
            "unknown": _PaddleValue(np.ones(1, dtype=np.float32)),
        }

        with self.assertRaisesRegex(RuntimeError, "unknown_sources=.*unknown"):
            copy_paddle_state_dict_strict(module, source)

        torch.testing.assert_close(module.weight, original_weight)

    def test_transposes_selected_matrix_weights_and_allows_prefixes(self):
        module = nn.Sequential(nn.Linear(2, 3, bias=False))
        source = {
            "0.weight": _PaddleValue(
                np.arange(6, dtype=np.float32).reshape(2, 3)
            ),
        }

        report = copy_paddle_state_dict_strict(
            module,
            source,
            source_transpose_suffixes=("fc.weight", "0.weight"),
            allowed_missing_target_prefixes=("head.gtc_head.",),
        )

        self.assertEqual(report.copied_count, 1)
        torch.testing.assert_close(
            module[0].weight,
            torch.arange(6, dtype=torch.float32).reshape(2, 3).T,
        )


if __name__ == "__main__":
    unittest.main()

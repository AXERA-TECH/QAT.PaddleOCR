import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class ReparameterizationCompareTest(unittest.TestCase):
    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 1, 2, 2), dtype=np.float32)

    def test_output_extracts_detection_maps(self):
        fake_paddle = types.ModuleType("paddle")
        fake_compare = types.ModuleType("compare_float_frameworks")
        fake_compare.DifferenceAccumulator = mock.Mock()
        fake_compare.build_dataloader = mock.Mock()
        fake_compare.build_route2 = mock.Mock()
        fake_compare.configure_eval = mock.Mock()
        fake_compare.load_yaml = mock.Mock()
        fake_torch = types.ModuleType("torch")
        path = ROOT_DIR / "tools/compare_reparameterization.py"
        spec = importlib.util.spec_from_file_location("_reparam_test", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {"paddle": fake_paddle, "torch": fake_torch, "compare_float_frameworks": fake_compare},
        ):
            spec.loader.exec_module(module)

        result = module.output(lambda _: {"maps": self.Tensor()}, None, "det")
        self.assertEqual(result.shape, (1, 1, 2, 2))

    def test_output_accepts_tensor_detection_wrapper(self):
        fake_paddle = types.ModuleType("paddle")
        fake_compare = types.ModuleType("compare_float_frameworks")
        fake_compare.DifferenceAccumulator = mock.Mock()
        fake_compare.build_dataloader = mock.Mock()
        fake_compare.build_route2 = mock.Mock()
        fake_compare.configure_eval = mock.Mock()
        fake_compare.load_yaml = mock.Mock()
        fake_torch = types.ModuleType("torch")
        path = ROOT_DIR / "tools/compare_reparameterization.py"
        spec = importlib.util.spec_from_file_location("_reparam_test_tensor", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {"paddle": fake_paddle, "torch": fake_torch, "compare_float_frameworks": fake_compare},
        ):
            spec.loader.exec_module(module)

        result = module.output(lambda _: self.Tensor(), None, "det")
        self.assertEqual(result.shape, (1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()

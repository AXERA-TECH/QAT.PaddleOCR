import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]


class RecFloatLayerDiagnosticTest(unittest.TestCase):
    def test_stats(self):
        fake_compare = types.ModuleType("compare_float_frameworks")
        for name in (
            "build_dataloader",
            "build_paddle",
            "build_route2",
            "configure_eval",
            "load_yaml",
        ):
            setattr(fake_compare, name, mock.Mock())
        fake_paddle = types.ModuleType("paddle")
        fake_torch = types.ModuleType("torch")
        path = ROOT_DIR / "tools/diagnose_rec_float_layers.py"
        spec = importlib.util.spec_from_file_location("_rec_layer_test", path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {
                "compare_float_frameworks": fake_compare,
                "paddle": fake_paddle,
                "torch": fake_torch,
            },
        ):
            spec.loader.exec_module(module)
        result = module.stats(np.array([0.0, 2.0]), np.array([1.0, 4.0]))
        self.assertEqual(result["shape"], [2])
        self.assertEqual(result["mae"], 1.5)
        self.assertEqual(result["max_abs"], 2.0)


if __name__ == "__main__":
    unittest.main()

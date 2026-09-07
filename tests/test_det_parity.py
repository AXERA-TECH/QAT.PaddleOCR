import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[1]


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.train()


class DetParityModeTest(unittest.TestCase):
    def test_build_torch_restores_inference_mode(self):
        fake_paddle = types.ModuleType("paddle")
        fake_training = types.ModuleType("pytorchocr.training")
        fake_architectures = types.ModuleType("ppocr.modeling.architectures")
        fake_architectures.build_model = mock.Mock()
        wrapper = _Wrapper()
        fake_training.build_det_model = mock.Mock(return_value=wrapper)
        fake_training.load_ocr_config = mock.Mock()

        module_name = "_compare_det_parity_test"
        module_path = ROOT_DIR / "tools" / "compare_det_parity.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {
                "paddle": fake_paddle,
                "ppocr.modeling.architectures": fake_architectures,
                "pytorchocr.training": fake_training,
            },
        ):
            spec.loader.exec_module(module)

        result = module.build_torch("model.yml", "weights.pth")

        self.assertIs(result, wrapper)
        self.assertFalse(result.training)
        self.assertFalse(result.bn.training)
        fake_training.build_det_model.assert_called_once_with(
            "model.yml",
            weights_path="weights.pth",
            reparameterize=False,
            graph_mode="inference",
        )


if __name__ == "__main__":
    unittest.main()

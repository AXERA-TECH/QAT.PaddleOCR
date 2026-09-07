import unittest

import torch

from pytorchocr.modeling.backbones.rec_lcnetv4 import ConvBNAct
from pytorchocr.modeling.backbones.rec_lcnetv3 import PPLCNetV3
from pytorchocr.modeling.backbones.rec_hgnet import PPHGNet_small
from pytorchocr.modeling.backbones.rec_pphgnetv2 import PPHGNetV2_B4


class ReparameterizationTest(unittest.TestCase):
    def test_pplcnetv3_top_level_rep_preserves_detection_features(self):
        torch.manual_seed(5)
        model = PPLCNetV3(scale=0.75, det=True).eval()
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = [output.clone() for output in model(inputs)]
            model.rep()
            actual = model(inputs)

        self.assertTrue(model.is_repped)
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(
                actual_output,
                expected_output,
                rtol=1.0e-4,
                atol=1.0e-5,
            )

    def test_even_kernel_same_padding_is_preserved(self):
        torch.manual_seed(7)
        module = ConvBNAct(
            3,
            5,
            kernel_size=2,
            padding="same",
            use_act=False,
        ).eval()
        inputs = torch.randn(2, 3, 12, 14)
        with torch.no_grad():
            expected = module(inputs)
            module.rep()
            actual = module(inputs)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

    def test_pphgnetv2_rep_preserves_detection_features_and_removes_bn(self):
        torch.manual_seed(11)
        model = PPHGNetV2_B4(det=True).eval()
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = [output.clone() for output in model(inputs)]
            model.rep()
            actual = model(inputs)

        self.assertEqual(
            sum(
                isinstance(module, torch.nn.BatchNorm2d)
                for module in model.modules()
            ),
            0,
        )
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(
                actual_output,
                expected_output,
                rtol=2.0e-3,
                atol=3.0e-5,
            )

    def test_pphgnet_rep_preserves_detection_features_and_removes_bn(self):
        torch.manual_seed(13)
        model = PPHGNet_small(det=True).eval()
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = [output.clone() for output in model(inputs)]
            model.rep()
            actual = model(inputs)

        self.assertEqual(
            sum(
                isinstance(module, torch.nn.BatchNorm2d)
                for module in model.modules()
            ),
            0,
        )
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(
                actual_output,
                expected_output,
                rtol=2.0e-3,
                atol=3.0e-5,
            )


if __name__ == "__main__":
    unittest.main()

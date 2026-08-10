import unittest

import torch

from pytorchocr.modeling.backbones.rec_lcnetv3 import (
    ConvBNLayer,
    LearnableRepLayer,
)


class RecognitionReparameterizationTest(unittest.TestCase):
    def test_conv_bn_rep_preserves_cpu_eval_output(self):
        torch.manual_seed(7)
        layer = ConvBNLayer(4, 4, 3, 1).eval()
        inputs = torch.randn(2, 4, 11, 13)
        with torch.no_grad():
            expected = layer(inputs)
            layer.rep()
            actual = layer(inputs)

        self.assertTrue(layer.is_repped)
        self.assertLessEqual(
            torch.max(torch.abs(expected - actual)).item(), 1e-5
        )

    def test_rep_layer_fuses_each_branch_once(self):
        torch.manual_seed(11)
        layer = LearnableRepLayer(
            4,
            4,
            kernel_size=3,
            stride=1,
            groups=1,
            num_conv_branches=2,
        ).eval()
        inputs = torch.randn(2, 4, 9, 9)
        with torch.no_grad():
            expected = layer(inputs)
            kernel, bias = layer._get_kernel_bias()
            layer.rep()
            actual = layer(inputs)

        self.assertEqual(layer.reparam_conv.weight.shape, kernel.shape)
        self.assertTrue(torch.equal(layer.reparam_conv.weight, kernel))
        self.assertTrue(torch.equal(layer.reparam_conv.bias, bias))
        self.assertLessEqual(
            torch.max(torch.abs(expected - actual)).item(), 1e-5
        )

        weight = layer.reparam_conv.weight.detach().clone()
        bias = layer.reparam_conv.bias.detach().clone()
        layer.rep()
        self.assertTrue(torch.equal(layer.reparam_conv.weight, weight))
        self.assertTrue(torch.equal(layer.reparam_conv.bias, bias))

    def test_identity_kernel_follows_branch_dtype_and_device(self):
        layer = LearnableRepLayer(
            4,
            4,
            kernel_size=3,
            stride=1,
            groups=4,
            num_conv_branches=1,
        ).to(dtype=torch.float64)
        kernel, bias = layer._fuse_bn_tensor(layer.identity)

        self.assertEqual(kernel.dtype, torch.float64)
        self.assertEqual(bias.dtype, torch.float64)
        self.assertEqual(kernel.device, layer.identity.weight.device)
        self.assertEqual(kernel.shape, (4, 1, 3, 3))


if __name__ == "__main__":
    unittest.main()

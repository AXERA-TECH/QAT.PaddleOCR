import unittest

import torch

from pytorchocr.modeling.backbones.rec_lcnetv4 import PPLCNetV4, RepDWConv
from pytorchocr.training.model_builder import build_task_model



class RepDWConvKeepBnTest(unittest.TestCase):
    """v6 RepDWConv rep(keep_bn) behavior: keep native post-sum BN or fold it."""

    def _module(self, channels=8, kernel_size=3):
        torch.manual_seed(3)
        module = RepDWConv(channels, kernel_size).eval()
        return module

    def test_keep_bn_true_preserves_post_bn_and_is_equivalent(self):
        module = self._module()
        inputs = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            expected = module(inputs).clone()
            module.rep(keep_bn=True)
            actual = module(inputs)
        self.assertTrue(module.is_repped)
        self.assertTrue(module.keep_bn)
        self.assertIsNotNone(module.bn)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)

    def test_keep_bn_false_folds_post_bn_away_and_is_equivalent(self):
        module = self._module()
        inputs = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            expected = module(inputs).clone()
            module.rep(keep_bn=False)
            actual = module(inputs)
        self.assertTrue(module.is_repped)
        self.assertFalse(module.keep_bn)
        self.assertFalse(hasattr(module, "bn"))
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


class V6RecKeepBnModelTest(unittest.TestCase):
    """Full v6 small rec deploy graph: keep_bn at reparameterization time."""

    CONFIG = "configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml"

    def _build(self, keep_bn):
        torch.manual_seed(0)
        return build_task_model(
            "rec",
            self.CONFIG,
            weights_path=None,
            reparameterize=True,
            rec_graph="deploy",
            keep_bn=keep_bn,
        )

    @staticmethod
    def _bn_count(model):
        return sum(
            isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()
        )

    def test_keep_bn_true_keeps_11_post_bn_plus_neck_bn(self):
        model = self._build(keep_bn=True)
        # 11 RepDWConv post-BN kept + 3 LightSVTR neck BN (conv_reduce/local/skip)
        self.assertEqual(self._bn_count(model), 14)
        rep_dw = [
            module
            for module in model.modules()
            if isinstance(module, RepDWConv)
        ]
        self.assertEqual(len(rep_dw), 11)
        self.assertTrue(all(module.keep_bn for module in rep_dw))

    def test_keep_bn_false_folds_post_bn(self):
        model = self._build(keep_bn=False)
        # only the 3 LightSVTR neck BNs remain (v5 rec would be 0)
        self.assertEqual(self._bn_count(model), 3)

    def test_keep_bn_forward_equivalent_to_unrepped(self):
        torch.manual_seed(1)
        model = self._build(keep_bn=True)
        model.eval()
        inputs = torch.randn(1, 3, 48, 64)
        unrepped = build_task_model(
            "rec",
            self.CONFIG,
            weights_path=None,
            reparameterize=False,
            rec_graph="deploy",
        )
        unrepped.eval()
        with torch.no_grad():
            expected = unrepped(inputs)
            actual = model(inputs)
        # Folded conv sums the three branch kernels, so tiny float reordering
        # noise is expected on logits (max ~4e-5); structure equivalence holds.
        torch.testing.assert_close(
            actual, expected, rtol=1.0e-3, atol=1.0e-4
        )


class V5RepFallbackTest(unittest.TestCase):
    """v5 rec rep(insert_identity_bn) path stays unaffected by the new keep_bn kwarg."""

    CONFIG = "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"

    def test_v5_deploy_rep_still_folds_forward_path(self):
        torch.manual_seed(0)
        model = build_task_model(
            "rec",
            self.CONFIG,
            weights_path=None,
            reparameterize=True,
            rec_graph="deploy",
            keep_bn=False,
        )
        # v5 rep keeps the training branches as dormant modules (dead BNs are
        # still present in eager module tree, matching pre-keep_bn behavior);
        # the forward path uses reparam_conv and QuantONNX ends up BN=0 after
        # PT2E fusion. Assert rep ran and the repped forward is finite.
        self.assertTrue(getattr(model.model.backbone, "is_repped", False))
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 48, 64))
        self.assertTrue(bool(torch.isfinite(out).all()))
        # T follows input width (width 64 -> T 8; contract width 320 -> T 40).
        self.assertEqual(tuple(out.shape), (1, 8, 18385))


if __name__ == "__main__":
    unittest.main()

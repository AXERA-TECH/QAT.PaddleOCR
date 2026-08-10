import unittest

import torch
from torch import nn

from pytorchocr.diagnostics import outputs_as_tuple


class ModelCompatibilityTest(unittest.TestCase):
    def test_swish_exports_as_native_silu(self):
        from pytorchocr.modeling.common import Swish

        inputs = torch.randn(1, 3, 4, 4)
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                module = Swish(inplace=inplace)
                expected = torch.nn.functional.silu(inputs)
                actual = module(inputs.clone())
                self.assertTrue(torch.equal(actual, expected))

                exported = torch.export.export_for_training(
                    module,
                    (inputs.clone(),),
                ).module()
                targets = {
                    node.target
                    for node in exported.graph.nodes
                    if node.op == "call_function"
                }
                self.assertTrue(
                    targets
                    & {
                        torch.ops.aten.silu.default,
                        torch.ops.aten.silu_.default,
                    }
                )
                self.assertNotIn(torch.ops.aten.sigmoid.default, targets)
                self.assertNotIn(torch.ops.aten.mul.Tensor, targets)

    def test_output_tensors_normalizes_single_and_tuple_outputs(self):
        tensor = torch.ones(1)

        self.assertEqual(outputs_as_tuple(tensor), (tensor,))
        self.assertEqual(outputs_as_tuple((tensor, tensor)), (tensor, tensor))

    def test_v5_v4_configs_build_expected_shapes(self):
        from pytorchocr.training import build_task_model

        cases = [
            ("det", "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml", [1, 3, 64, 64], 3),
            ("det", "configs/det/PP-OCRv4/PP-OCRv4_mobile_det.yml", [1, 3, 64, 64], 3),
            ("rec", "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml", [1, 3, 48, 64], 1),
            ("rec", "configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml", [1, 3, 48, 64], 1),
            ("det", "configs/det/PP-OCRv5/PP-OCRv5_server_det.yml", [1, 3, 64, 64], 3),
            ("det", "configs/det/PP-OCRv4/PP-OCRv4_server_det.yml", [1, 3, 64, 64], 3),
            ("rec", "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml", [1, 3, 48, 64], 1),
            ("rec", "configs/rec/PP-OCRv4/PP-OCRv4_server_rec.yml", [1, 3, 48, 64], 1),
        ]
        for task, config, shape, output_count in cases:
            with self.subTest(config=config):
                model = build_task_model(task, config, reparameterize=True)
                model.eval()
                with torch.no_grad():
                    outputs = outputs_as_tuple(model(torch.randn(*shape)))
                self.assertEqual(len(outputs), output_count)
                self.assertTrue(all(torch.isfinite(output).all() for output in outputs))

    def test_mobile_rec_svtr_uses_configured_conv4_kernel(self):
        from pytorchocr.training import build_rec_model

        for config in (
            "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
            "configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml",
        ):
            with self.subTest(config=config):
                wrapper = build_rec_model(config, reparameterize=False)
                encoder = wrapper.model.head.ctc_encoder.encoder
                self.assertEqual(encoder.conv1.conv.kernel_size, (1, 3))
                self.assertEqual(encoder.conv4.conv.kernel_size, (1, 3))

    def test_mobile_rec_can_enable_ctc_backbone_gradients(self):
        from pytorchocr.training import build_rec_model

        wrapper = build_rec_model(
            "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
            reparameterize=False,
            ctc_backbone_grad=True,
        )

        self.assertFalse(wrapper.model.head.ctc_encoder.encoder.use_guide)


if __name__ == "__main__":
    unittest.main()

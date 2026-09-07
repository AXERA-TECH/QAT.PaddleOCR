import tempfile
import unittest
from pathlib import Path

import torch

from pytorchocr.quantization import FullDetTrainingWrapper, FullRecTrainingWrapper
from pytorchocr.training import build_det_model, build_rec_model
from pytorchocr.modeling.heads.rec_nrtr_head import Transformer


CONFIG = "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
DET_CONFIG = "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml"
V6_DET_CONFIG = "configs/det/PP-OCRv6/PP-OCRv6_small_det.yml"


class ModelRoleTest(unittest.TestCase):
    def test_nrtr_training_mask_export_follows_reference_device(self):
        class MaskWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = Transformer(
                    d_model=8,
                    nhead=2,
                    num_encoder_layers=0,
                    num_decoder_layers=1,
                    dim_feedforward=16,
                    in_channels=8,
                    out_channels=16,
                )

            def forward(self, reference):
                return self.transformer.generate_square_subsequent_mask(
                    reference.shape[1], reference
                )

        exported = torch.export.export_for_training(
            MaskWrapper(),
            (torch.randn(2, 5, 8),),
        ).module()

        self.assertNotIn("device(type='cpu')", exported.code)
        self.assertEqual(tuple(exported(torch.randn(2, 5, 8)).shape), (1, 1, 5, 5))

    def test_full_recognition_export_wrapper_has_fixed_training_outputs(self):
        model = build_rec_model(
            CONFIG,
            reparameterize=False,
            graph_role="pretrained_train",
        )
        wrapper = FullRecTrainingWrapper(model, max_text_length=25)
        wrapper.set_validation_mode()
        targets = torch.zeros(1, 25, dtype=torch.int64)
        targets[:, 0:3] = torch.tensor([2, 4, 3])

        outputs = wrapper(torch.randn(1, 3, 48, 64), targets)

        self.assertEqual(wrapper.output_names, ("ctc", "ctc_neck", "gtc"))
        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[2].shape[:2]), (1, 24))

    def test_full_detection_export_wrapper_preserves_auxiliary_order(self):
        model = build_det_model(
            V6_DET_CONFIG,
            reparameterize=False,
            graph_mode="pretrained_train",
        )
        wrapper = FullDetTrainingWrapper(model)
        wrapper.set_validation_mode()

        outputs = wrapper(torch.randn(1, 3, 64, 64))

        self.assertEqual(
            wrapper.output_names,
            ("maps", "aux_maps_p4", "aux_maps_p3", "aux_maps_p2"),
        )
        self.assertEqual(len(outputs), 4)
        self.assertTrue(all(tuple(value.shape) == (1, 3, 64, 64) for value in outputs))

    def test_detection_pretrained_training_role_keeps_native_db_output(self):
        model = build_det_model(
            DET_CONFIG,
            reparameterize=False,
            graph_mode="pretrained_train",
        )
        output = model(torch.randn(1, 3, 64, 64))

        self.assertEqual(model.graph_role, "pretrained_train")
        self.assertEqual(set(output), {"maps"})
        self.assertEqual(tuple(output["maps"].shape), (1, 3, 64, 64))

    def test_detection_validation_keeps_three_maps_with_bn_in_eval(self):
        model = build_det_model(
            DET_CONFIG,
            reparameterize=False,
            graph_mode="pretrained_train",
        )

        model.set_validation_mode()
        output = model(torch.randn(1, 3, 64, 64))

        self.assertFalse(model.training)
        self.assertTrue(model.head.training)
        self.assertFalse(model.head.binarize.conv_bn1.training)
        self.assertEqual(tuple(output["maps"].shape), (1, 3, 64, 64))

    def test_detection_training_shrink_matches_deploy_projection(self):
        source = build_det_model(
            DET_CONFIG,
            reparameterize=False,
            graph_mode="pretrained_train",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "det.pt"
            torch.save(source.state_dict(), path)
            full = build_det_model(
                DET_CONFIG,
                weights_path=path,
                reparameterize=False,
                graph_mode="pretrained_train",
            )
            deploy = build_det_model(
                DET_CONFIG,
                weights_path=path,
                reparameterize=False,
                graph_mode="inference",
            )

        images = torch.randn(1, 3, 64, 64)
        full.set_validation_mode()
        deploy.eval()
        with torch.no_grad():
            full_shrink = full(images)["maps"][:, :1]
            deploy_shrink = deploy(images)

        torch.testing.assert_close(full_shrink, deploy_shrink, rtol=0, atol=0)

    def test_v6_detection_pretrained_training_keeps_auxiliary_maps(self):
        model = build_det_model(
            V6_DET_CONFIG,
            reparameterize=False,
            graph_mode="pretrained_train",
        )

        output = model(torch.randn(1, 3, 64, 64))

        self.assertEqual(
            set(output),
            {"maps", "aux_maps_p4", "aux_maps_p3", "aux_maps_p2"},
        )
        for value in output.values():
            self.assertEqual(tuple(value.shape), (1, 3, 64, 64))

    def test_v6_detection_validation_keeps_auxiliary_maps_with_bn_eval(self):
        model = build_det_model(
            V6_DET_CONFIG,
            reparameterize=False,
            graph_mode="pretrained_train",
        )

        model.set_validation_mode()
        output = model(torch.randn(1, 3, 64, 64))

        self.assertEqual(
            set(output),
            {"maps", "aux_maps_p4", "aux_maps_p3", "aux_maps_p2"},
        )
        self.assertTrue(model.neck.training)
        self.assertFalse(model.neck.ins_conv[0].in_conv.training)

    def test_pretrained_training_role_keeps_multi_head(self):
        model = build_rec_model(
            CONFIG,
            reparameterize=False,
            graph_role="pretrained_train",
        )

        self.assertEqual(model.graph_role, "pretrained_train")
        self.assertTrue(hasattr(model.head, "ctc_head"))
        self.assertTrue(hasattr(model.head, "gtc_head"))
        self.assertTrue(model.head.ctc_encoder.encoder.use_guide)

    def test_pretrained_training_validation_returns_raw_ctc_path(self):
        model = build_rec_model(
            CONFIG,
            reparameterize=False,
            graph_role="pretrained_train",
        )

        model.set_validation_mode()

        self.assertFalse(model.training)
        self.assertFalse(model.head.training)
        self.assertTrue(model.head.ctc_head.training)

    def test_full_state_strictly_reloads_for_training(self):
        source = build_rec_model(
            CONFIG,
            reparameterize=False,
            graph_role="pretrained_train",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.pt"
            torch.save(source.state_dict(), path)
            restored = build_rec_model(
                CONFIG,
                weights_path=path,
                reparameterize=False,
                graph_role="pretrained_train",
            )

        self.assertEqual(set(source.state_dict()), set(restored.state_dict()))

    def test_training_role_rejects_ctc_gradient_workaround(self):
        with self.assertRaisesRegex(ValueError, "configured guide detach"):
            build_rec_model(
                CONFIG,
                reparameterize=False,
                ctc_backbone_grad=True,
                graph_role="pretrained_train",
            )


if __name__ == "__main__":
    unittest.main()

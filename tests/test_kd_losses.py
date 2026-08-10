import unittest

import torch
import torch.nn.functional as F

from pytorchocr.training.losses import (
    KDCompositeLoss,
    KDFeatureLoss,
    KDLogitsLoss,
    KDMapsLoss,
    build_kd_criterion,
    default_kd_layers,
    normalize_outputs,
)


class KDLogitsLossTest(unittest.TestCase):
    def test_kl_zero_for_identical_logits(self):
        criterion = KDLogitsLoss(mode="kl", temperature=4.0)
        logits = torch.randn(2, 5, 8)
        loss = criterion(logits, logits.clone())
        self.assertLess(float(loss), 1.0e-6)

    def test_kl_positive_and_finite(self):
        criterion = KDLogitsLoss(mode="kl", temperature=2.0)
        student = torch.randn(2, 5, 8)
        teacher = torch.randn(2, 5, 8)
        loss = criterion(student, teacher)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_mse_equals_mean_squared_error(self):
        criterion = KDLogitsLoss(mode="mse")
        student = torch.randn(3, 4, 6)
        teacher = torch.randn(3, 4, 6)
        loss = criterion(student, teacher)
        expected = torch.mean((student - teacher.detach()) ** 2)
        torch.testing.assert_close(loss, expected)

    def test_shape_mismatch_rejected(self):
        criterion = KDLogitsLoss()
        with self.assertRaises(ValueError):
            criterion(torch.randn(2, 5, 8), torch.randn(2, 5, 7))

    def test_gradient_flows_to_student_only(self):
        criterion = KDLogitsLoss(mode="kl", temperature=4.0)
        student = torch.randn(2, 5, 8, requires_grad=True)
        teacher = torch.randn(2, 5, 8, requires_grad=True)
        loss = criterion(student, teacher)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)


class KDMapsLossTest(unittest.TestCase):
    def test_mse_over_all_maps(self):
        criterion = KDMapsLoss()
        student = tuple(torch.randn(2, 1, 16, 16) for _ in range(3))
        teacher = tuple(torch.randn(2, 1, 16, 16) for _ in range(3))
        loss = criterion(student, teacher)
        expected = sum(
            torch.mean((s - t.detach()) ** 2) for s, t in zip(student, teacher)
        )
        torch.testing.assert_close(loss, expected)

    def test_single_tensor_input(self):
        criterion = KDMapsLoss()
        student = torch.randn(2, 1, 16, 16)
        teacher = torch.randn(2, 1, 16, 16)
        loss = criterion(student, teacher)
        self.assertTrue(torch.isfinite(loss))

    def test_count_mismatch_rejected(self):
        criterion = KDMapsLoss()
        with self.assertRaises(ValueError):
            criterion(
                (torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)),
                (torch.randn(2, 1, 4, 4),),
            )


class KDFeatureLossTest(unittest.TestCase):
    def test_mse(self):
        criterion = KDFeatureLoss(mode="mse")
        student = torch.randn(2, 64, 8, 8)
        teacher = torch.randn(2, 64, 8, 8)
        loss = criterion(student, teacher)
        expected = torch.mean((student - teacher.detach()) ** 2)
        torch.testing.assert_close(loss, expected)

    def test_l1(self):
        criterion = KDFeatureLoss(mode="l1")
        student = torch.randn(2, 64, 8)
        teacher = torch.randn(2, 64, 8)
        loss = criterion(student, teacher)
        expected = torch.mean(torch.abs(student - teacher.detach()))
        torch.testing.assert_close(loss, expected)

    def test_multi_level_list_input(self):
        criterion = KDFeatureLoss(mode="mse")
        student = [torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8)]
        teacher = [torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8)]
        loss = criterion(student, teacher)
        expected = sum(
            torch.mean((s - t.detach()) ** 2) for s, t in zip(student, teacher)
        )
        torch.testing.assert_close(loss, expected)

    def test_multi_level_count_mismatch_rejected(self):
        criterion = KDFeatureLoss()
        with self.assertRaises(ValueError):
            criterion(
                [torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8)],
                [torch.randn(2, 32, 16, 16)],
            )

    def test_multi_level_gradient_flows_to_all_levels(self):
        criterion = KDFeatureLoss()
        student = [
            torch.randn(2, 32, 16, 16, requires_grad=True),
            torch.randn(2, 64, 8, 8, requires_grad=True),
        ]
        teacher = [torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8)]
        loss = criterion(student, teacher)
        loss.backward()
        self.assertIsNotNone(student[0].grad)
        self.assertIsNotNone(student[1].grad)


class KDLogitsTemperatureTest(unittest.TestCase):
    def test_kl_loss_scales_with_temperature_squared(self):
        """KL mode loss must equal T^2 * KL(softmax(s/T) || softmax(t/T))."""
        student = torch.randn(2, 8, 16)
        teacher = torch.randn(2, 8, 16)
        temperature = 3.0
        criterion = KDLogitsLoss(mode="kl", temperature=temperature)
        loss = criterion(student, teacher)

        teacher_prob = F.softmax(teacher.detach() / temperature, dim=-1)
        student_log_prob = F.log_softmax(student / temperature, dim=-1)
        # batchmean = sum over all elements / batch size (Paddle DML semantics).
        expected = (
            temperature**2
            * (teacher_prob * (teacher_prob.log() - student_log_prob)).sum()
            / student.shape[0]
        )
        torch.testing.assert_close(loss, expected, atol=1.0e-6, rtol=1.0e-6)

    def test_temperature_one_matches_plain_kl(self):
        student = torch.randn(2, 8, 16)
        teacher = torch.randn(2, 8, 16)
        criterion = KDLogitsLoss(mode="kl", temperature=1.0)
        loss = criterion(student, teacher)
        teacher_prob = F.softmax(teacher.detach(), dim=-1)
        student_log_prob = F.log_softmax(student, dim=-1)
        expected = (teacher_prob * (teacher_prob.log() - student_log_prob)).sum() / (
            student.shape[0]
        )
        torch.testing.assert_close(loss, expected, atol=1.0e-6, rtol=1.0e-6)

    def test_zero_temperature_rejected(self):
        with self.assertRaises(ValueError):
            KDLogitsLoss(mode="kl", temperature=0.0)
        with self.assertRaises(ValueError):
            KDLogitsLoss(mode="kl", temperature=-1.0)


class NormalizeOutputsTest(unittest.TestCase):
    def test_rec_tensor(self):
        logits = torch.randn(2, 40, 10)
        normalized = normalize_outputs(logits, "rec")
        self.assertIs(normalized["ctc"], logits)

    def test_rec_tuple(self):
        logits = torch.randn(2, 40, 10)
        neck = torch.randn(2, 40, 64)
        normalized = normalize_outputs((logits, neck), "rec")
        self.assertIs(normalized["ctc"], logits)
        self.assertIs(normalized["ctc_neck"], neck)

    def test_rec_full_wrapper_dict(self):
        outputs = {
            "ctc": torch.randn(2, 40, 10),
            "ctc_neck": torch.randn(2, 40, 64),
            "gtc": torch.randn(2, 25, 10),
            "backbone_out": torch.randn(2, 64, 12, 32),
            "neck_out": torch.randn(2, 64, 3, 32),
        }
        normalized = normalize_outputs(outputs, "rec", rec_graph="pretrained_train")
        self.assertEqual(set(normalized), set(outputs))

    def test_rec_base_model_dict(self):
        outputs = {
            "backbone_out": torch.randn(2, 64, 12, 32),
            "neck_out": torch.randn(2, 64, 3, 32),
            "head_out": {
                "ctc": torch.randn(2, 40, 10),
                "gtc": torch.randn(2, 25, 10),
            },
        }
        normalized = normalize_outputs(outputs, "rec", rec_graph="pretrained_train")
        self.assertIn("ctc", normalized)
        self.assertIn("ctc_neck", normalized)

    def test_det_tuple(self):
        maps = tuple(torch.randn(2, 1, 32, 32) for _ in range(3))
        normalized = normalize_outputs(maps, "det")
        self.assertEqual(normalized["maps"], maps)

    def test_det_expose_dict(self):
        outputs = {
            "maps": (torch.randn(2, 1, 32, 32),) * 3,
            "backbone_out": [torch.randn(2, 32, 32, 32)],
            "neck_out": torch.randn(2, 96, 8, 8),
        }
        normalized = normalize_outputs(outputs, "det")
        self.assertEqual(set(normalized), set(outputs))

    def test_missing_head_rejected(self):
        with self.assertRaises(KeyError):
            normalize_outputs({"ctc_neck": torch.randn(2, 40, 64)}, "rec")


class KDCompositeLossTest(unittest.TestCase):
    def test_composite_weighted_sum(self):
        layers = {"ctc": ("kl", 1.0), "ctc_neck": ("mse", 0.5)}
        criterion = KDCompositeLoss(layers, temperature=4.0, task="rec")
        student = {
            "ctc": torch.randn(2, 40, 10),
            "ctc_neck": torch.randn(2, 40, 64),
        }
        teacher = {
            "ctc": torch.randn(2, 40, 10),
            "ctc_neck": torch.randn(2, 40, 64),
        }
        losses = criterion(student, teacher)
        self.assertIn("kd_loss", losses)
        self.assertIn("kd_ctc", losses)
        self.assertIn("kd_ctc_neck", losses)
        expected = losses["kd_ctc"] + 0.5 * losses["kd_ctc_neck"]
        torch.testing.assert_close(losses["kd_loss"], expected)

    def test_zero_weight_layer_still_reported(self):
        layers = {"ctc": ("kl", 1.0), "backbone_out": ("mse", 0.0)}
        criterion = KDCompositeLoss(layers, temperature=4.0, task="rec")
        student = {
            "ctc": torch.randn(2, 40, 10),
            "backbone_out": torch.randn(2, 64, 6, 16),
        }
        teacher = {
            "ctc": torch.randn(2, 40, 10),
            "backbone_out": torch.randn(2, 64, 6, 16),
        }
        losses = criterion(student, teacher)
        self.assertIn("kd_backbone_out", losses)
        torch.testing.assert_close(losses["kd_loss"], losses["kd_ctc"])

    def test_missing_layer_rejected(self):
        criterion = KDCompositeLoss(
            {"ctc": ("kl", 1.0), "backbone_out": ("mse", 1.0)},
            temperature=4.0,
            task="rec",
        )
        with self.assertRaises(KeyError):
            criterion(
                {"ctc": torch.randn(2, 40, 10)},
                {"ctc": torch.randn(2, 40, 10)},
            )

    def test_gradient_flows_through_composite(self):
        criterion = KDCompositeLoss(
            {"ctc": ("kl", 1.0), "ctc_neck": ("mse", 0.5)},
            temperature=4.0,
            task="rec",
        )
        student = {
            "ctc": torch.randn(2, 40, 10, requires_grad=True),
            "ctc_neck": torch.randn(2, 40, 64, requires_grad=True),
        }
        teacher = {
            "ctc": torch.randn(2, 40, 10),
            "ctc_neck": torch.randn(2, 40, 64),
        }
        losses = criterion(student, teacher)
        losses["kd_loss"].backward()
        self.assertIsNotNone(student["ctc"].grad)
        self.assertIsNotNone(student["ctc_neck"].grad)


    def test_det_composite_with_maps_and_neck(self):
        criterion = KDCompositeLoss(
            {"maps": ("maps", 1.0), "neck_out": ("mse", 0.5)},
            task="det",
        )
        student = {
            "maps": tuple(torch.randn(2, 1, 16, 16) for _ in range(3)),
            "neck_out": torch.randn(2, 96, 8, 8),
        }
        teacher = {
            "maps": tuple(torch.randn(2, 1, 16, 16) for _ in range(3)),
            "neck_out": torch.randn(2, 96, 8, 8),
        }
        losses = criterion(student, teacher)
        self.assertIn("kd_loss", losses)
        self.assertIn("kd_maps", losses)
        self.assertIn("kd_neck_out", losses)
        expected = losses["kd_maps"] + 0.5 * losses["kd_neck_out"]
        torch.testing.assert_close(losses["kd_loss"], expected)

    def test_det_composite_multi_level_backbone(self):
        criterion = KDCompositeLoss(
            {"maps": ("maps", 1.0), "backbone_out": ("mse", 0.25)},
            task="det",
        )
        student = {
            "maps": tuple(torch.randn(2, 1, 16, 16) for _ in range(3)),
            "backbone_out": [torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8)],
        }
        teacher = {
            "maps": tuple(torch.randn(2, 1, 16, 16) for _ in range(3)),
            "backbone_out": [torch.randn(2, 32, 16, 16), torch.randn(2, 64, 8, 8)],
        }
        losses = criterion(student, teacher)
        self.assertIn("kd_backbone_out", losses)
        expected = losses["kd_maps"] + 0.25 * losses["kd_backbone_out"]
        torch.testing.assert_close(losses["kd_loss"], expected)


class DefaultKDLayersTest(unittest.TestCase):
    def test_rec_default_head_enabled(self):
        layers = default_kd_layers("rec")
        self.assertEqual(layers["ctc"][0], "kl")
        self.assertEqual(layers["ctc"][1], 1.0)
        self.assertEqual(layers["ctc_neck"][1], 0.0)
        self.assertEqual(layers["backbone_out"][1], 0.0)

    def test_det_default_head_enabled(self):
        layers = default_kd_layers("det")
        self.assertEqual(layers["maps"][0], "maps")
        self.assertEqual(layers["maps"][1], 1.0)
        self.assertEqual(layers["neck_out"][1], 0.0)

    def test_build_with_custom_head_weight(self):
        layers = default_kd_layers("rec", kd_mode="logits_mse", head_weight=2.0)
        self.assertEqual(layers["ctc"], ("logits_mse", 2.0))


class BuildKDCriterionTest(unittest.TestCase):
    def test_rec_default(self):
        criterion = build_kd_criterion("rec")
        self.assertIsInstance(criterion, KDCompositeLoss)

    def test_det_default(self):
        criterion = build_kd_criterion("det")
        self.assertIsInstance(criterion, KDCompositeLoss)

    def test_custom_layers(self):
        criterion = build_kd_criterion(
            "rec",
            kd_layers={"ctc": ("kl", 1.0)},
            kd_temperature=8.0,
        )
        losses = criterion(
            {"ctc": torch.randn(2, 40, 10)},
            {"ctc": torch.randn(2, 40, 10)},
        )
        self.assertIn("kd_loss", losses)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            build_kd_criterion("rec", kd_layers={"ctc": ("bogus", 1.0)})


if __name__ == "__main__":
    unittest.main()

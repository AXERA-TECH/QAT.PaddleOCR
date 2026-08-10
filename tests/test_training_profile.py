import tempfile
import unittest
from pathlib import Path

import torch

from pytorchocr.training import (
    build_optimizer,
    build_scheduler,
    epoch2_accuracy_guard,
    load_training_profile,
    profile_value,
    relocate_checkpoint_metadata,
    relocate_project_path,
    validate_resume_contract,
)
from pytorchocr.training.model_builder import resolve_config_path


class TrainingProfileTest(unittest.TestCase):
    def test_resolves_resource_from_parent_paddleocr_checkout(self):
        path = resolve_config_path("ppocr/utils/dict/ppocrv5_dict.txt")

        self.assertTrue(path.is_file())
        self.assertEqual(path.name, "ppocrv5_dict.txt")

    def test_checked_in_detection_profile(self):
        profile = load_training_profile(
            "configs/qat/training/ppocrv6_small_det_baseline.yml"
        )
        profile.validate_contract("det", "PP-OCRv6_small_det")

        self.assertTrue(profile.qat)
        self.assertTrue(profile.qat_config.is_file())
        self.assertEqual(profile.training["epochs"], 50)
        self.assertEqual(profile.training["image_shape"], [3, 640, 640])
        self.assertIsNone(profile.training["observer_freeze_epoch"])
        self.assertEqual(profile_value(4, profile, "batch_size"), 4)
        self.assertEqual(profile_value(None, profile, "batch_size"), 8)

    def test_checked_in_recognition_profile(self):
        profile = load_training_profile(
            "configs/qat/training/ppocrv6_small_rec_baseline.yml"
        )
        profile.validate_contract("rec", "PP-OCRv6_small_rec")

        self.assertTrue(profile.qat_config.is_file())
        self.assertEqual(profile.training["batch_size"], 64)
        self.assertEqual(profile.training["image_shape"], [3, 48, 320])
        self.assertTrue(profile.training["reparameterize"])

    def test_ppocrv5_mobile_detection_profile(self):
        profile = load_training_profile(
            "configs/qat/training/ppocrv5_mobile_det_baseline.yml"
        )
        profile.validate_contract("det", "PP-OCRv5_mobile_det")

        self.assertTrue(profile.qat)
        self.assertEqual(profile.qat_config.name, "ppocrv5_mobile_det_u8s8.json")
        self.assertEqual(profile.training["image_shape"], [3, 640, 640])
        self.assertIsNone(profile.training["observer_freeze_epoch"])

    def test_ppocrv5_mobile_rec_u16s16_sgd_dynamic_height_profile(self):
        profile = load_training_profile(
            "configs/qat/training/"
            "ppocrv5_mobile_rec_u16s16_sgd_dynamic_height.yml"
        )
        profile.validate_contract("rec", "PP-OCRv5_mobile_rec")

        self.assertEqual(
            profile.qat_config.name,
            "ppocrv5_mobile_rec_u16s16_attn_s16.json",
        )
        self.assertEqual(profile.training["optimizer"], "SGD")
        self.assertEqual(profile.training["momentum"], 0.9)
        self.assertEqual(profile.training["dynamic_heights"], [32, 48, 64])
        self.assertEqual(profile.training["image_shape"], [3, 48, 320])
        self.assertEqual(profile.training["augmentation"], "none")
        self.assertTrue(profile.training["multi_scale_training"])
        self.assertEqual(profile.training["rec_graph"], "pretrained_train")
        self.assertTrue(profile.training["reparameterize"])
        self.assertFalse(profile.training["rec_ctc_backbone_grad"])
        self.assertAlmostEqual(
            profile.training["float_accuracy_baseline"],
            0.5936446798266731,
        )
        self.assertEqual(profile.training["epoch2_max_accuracy_drop"], 0.10)

    def test_epoch2_accuracy_guard_triggers_at_ten_point_drop(self):
        guard = epoch2_accuracy_guard(
            2,
            {"acc": 0.4936446798266731},
            float_accuracy_baseline=0.5936446798266731,
            max_accuracy_drop=0.10,
        )

        self.assertTrue(guard["triggered"])
        self.assertAlmostEqual(guard["accuracy_drop"], 0.10)

    def test_epoch2_accuracy_guard_only_checks_second_epoch(self):
        guard = epoch2_accuracy_guard(
            1,
            {"acc": 0.0},
            float_accuracy_baseline=0.5936446798266731,
            max_accuracy_drop=0.10,
        )

        self.assertIsNone(guard)

    def test_ppocrv5_mobile_rec_full_training_profile(self):
        profile = load_training_profile(
            "configs/qat/training/ppocrv5_mobile_rec_full_pretrained_float_smoke.yml"
        )
        profile.validate_contract("rec", "PP-OCRv5_mobile_rec")

        self.assertFalse(profile.qat)
        self.assertEqual(profile.training["rec_graph"], "pretrained_train")
        self.assertFalse(profile.training["reparameterize"])

    def test_optimizer_applies_recognition_parameter_groups(self):
        class LAB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.default = torch.nn.Linear(2, 2)
                self.default.weight._paddle_weight_decay = 0.0
                self.lab = LAB()
                self.ctc_head = torch.nn.Module()
                self.ctc_head.fc = torch.nn.Linear(2, 3)

        optimizer = build_optimizer(
            Model(),
            {"Optimizer": {"name": "Adam"}},
            learning_rate=2.0e-5,
            weight_decay=3.0e-5,
            lab_lr_multiplier=0.1,
            ctc_fc_weight_decay=1.0e-5,
        )
        groups = {group["group_name"]: group for group in optimizer.param_groups}

        self.assertEqual(
            set(groups),
            {"default", "paddle_no_decay", "lab", "ctc_fc"},
        )
        self.assertEqual(groups["default"]["lr"], 2.0e-5)
        self.assertEqual(groups["paddle_no_decay"]["weight_decay"], 0.0)
        self.assertAlmostEqual(groups["lab"]["lr"], 2.0e-6, places=15)
        self.assertEqual(groups["ctc_fc"]["weight_decay"], 1.0e-5)

    def test_ppocrv5_optimizer_preserves_paddle_parameter_regularization(self):
        from pytorchocr.training import build_rec_model

        model = build_rec_model(
            "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
            reparameterize=False,
            graph_role="pretrained_train",
        )
        optimizer = build_optimizer(
            model,
            {"Optimizer": {"name": "Adam", "regularizer": {"factor": 3.0e-5}}},
            learning_rate=5.0e-4,
            lab_lr_multiplier=0.1,
        )
        groups = {group["group_name"]: group for group in optimizer.param_groups}

        self.assertEqual(groups["paddle_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(groups["ctc_fc"]["weight_decay"], 1.0e-5)
        self.assertEqual(groups["lab"]["lr"], 5.0e-5)

    def test_ppocrv5_optimizer_uses_model_lab_learning_rate_by_default(self):
        from pytorchocr.training import build_det_model

        model = build_det_model(
            "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml",
            reparameterize=False,
            graph_mode="pretrained_train",
        )
        optimizer = build_optimizer(
            model,
            {"Optimizer": {"name": "Adam", "regularizer": {"factor": 5.0e-5}}},
            learning_rate=1.0e-3,
        )
        groups = {group["group_name"]: group for group in optimizer.param_groups}

        self.assertEqual(groups["lab"]["lr"], 1.0e-4)
        self.assertEqual(len(groups["lab"]["params"]), 112)

    def test_ppocrv6_optimizer_preserves_stem_bn_no_decay(self):
        from pytorchocr.training import build_rec_model

        model = build_rec_model(
            "configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml",
            reparameterize=False,
            graph_role="pretrained_train",
        )
        optimizer = build_optimizer(
            model,
            {"Optimizer": {"name": "Adam", "regularizer": {"factor": 3.0e-5}}},
            learning_rate=5.0e-4,
        )
        groups = {group["group_name"]: group for group in optimizer.param_groups}

        self.assertEqual(groups["paddle_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(len(groups["paddle_no_decay"]["params"]), 10)

    def test_optimizer_override_selects_sgd(self):
        model = torch.nn.Linear(4, 3)

        optimizer = build_optimizer(
            model,
            {"Optimizer": {"name": "Adam"}},
            learning_rate=2.0e-5,
            optimizer_name="SGD",
            momentum=0.9,
        )

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.defaults["lr"], 2.0e-5)
        self.assertEqual(optimizer.defaults["momentum"], 0.9)

    def test_profile_rejects_unknown_training_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "invalid.yml"
            profile_path.write_text(
                "name: invalid\ntraining:\n  typo_learning_rate: 0.1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown training settings"):
                load_training_profile(profile_path)

    def test_profile_rejects_unknown_augmentation_preset(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "invalid.yml"
            profile_path.write_text(
                "name: invalid\ntraining:\n  augmentation: random_magic\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "augmentation"):
                load_training_profile(profile_path)

    def test_cosine_scheduler_preserves_final_factor(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=2.0e-5)
        scheduler = build_scheduler(
            optimizer,
            {"Optimizer": {"lr": {"name": "Cosine"}}},
            epochs=10,
            warmup_epochs=0,
            final_factor=0.1,
        )

        for _ in range(10):
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2.0e-6, places=12)

    def test_resume_contract_rejects_quantization_change(self):
        metadata = {
            "task": "det",
            "model_config": "/model.yml",
            "training_profile_sha256": "profile-a",
            "qat": True,
            "qat_config_sha256": "quant-a",
            "torch_version": "2.6.0",
            "reparameterized": True,
            "image_shape": [3, 640, 640],
        }
        changed = dict(metadata, qat_config_sha256="quant-b")

        with self.assertRaisesRegex(ValueError, "qat_config_sha256"):
            validate_resume_contract(metadata, changed)

    def test_resume_contract_rejects_recognition_graph_change(self):
        metadata = {
            "task": "rec",
            "model_config": "/model.yml",
            "training_profile_sha256": "profile-a",
            "qat": False,
            "qat_config_sha256": None,
            "torch_version": "2.6.0",
            "reparameterized": False,
            "image_shape": [3, 48, 320],
            "rec_graph": "pretrained_train",
        }
        changed = dict(metadata, rec_graph="deploy")

        with self.assertRaisesRegex(ValueError, "rec_graph"):
            validate_resume_contract(metadata, changed)

    def test_resume_contract_rejects_dynamic_height_change(self):
        metadata = {
            "task": "rec",
            "model_config": "/model.yml",
            "training_profile_sha256": "profile-a",
            "qat": True,
            "qat_config_sha256": "quant-a",
            "torch_version": "2.6.0",
            "reparameterized": True,
            "image_shape": [3, 48, 320],
            "dynamic_heights": [32, 48, 64],
        }
        changed = dict(metadata, dynamic_heights=[48])

        with self.assertRaisesRegex(ValueError, "dynamic_heights"):
            validate_resume_contract(metadata, changed)

    def test_resume_contract_rejects_augmentation_change(self):
        metadata = {
            "task": "rec",
            "model_config": "/model.yml",
            "training_profile_sha256": "profile-a",
            "qat": True,
            "qat_config_sha256": "quant-a",
            "torch_version": "2.6.0",
            "reparameterized": True,
            "image_shape": [3, 48, 320],
            "dynamic_heights": [32, 48, 64],
            "augmentation": "none",
            "multi_scale_training": True,
        }

        with self.assertRaisesRegex(ValueError, "augmentation"):
            validate_resume_contract(
                metadata,
                dict(metadata, augmentation="paddle"),
            )
        with self.assertRaisesRegex(ValueError, "multi_scale_training"):
            validate_resume_contract(
                metadata,
                dict(metadata, multi_scale_training=False),
            )

    def test_resume_contract_defaults_legacy_recognition_graph_to_deploy(self):
        metadata = {
            "task": "rec",
            "model_config": "/model.yml",
            "training_profile_sha256": "profile-a",
            "qat": False,
            "qat_config_sha256": None,
            "torch_version": "2.6.0",
            "reparameterized": False,
            "image_shape": [3, 48, 320],
        }

        validate_resume_contract(metadata, dict(metadata, rec_graph="deploy"))

    def test_resume_contract_accepts_relocated_model_config(self):
        metadata = {
            "task": "rec",
            "model_config": (
                "/home/heqi/project/PaddleOCR/route2/"
                "PaddleOCR2Pytorch-QAT/configs/rec/model.yml"
            ),
            "training_profile_sha256": "profile-a",
            "qat": True,
            "qat_config_sha256": "quant-a",
            "torch_version": "2.6.0",
            "reparameterized": True,
            "image_shape": [3, 48, 320],
        }
        relocated = dict(
            metadata,
            model_config="/home/heqi/project/PaddleOCR/configs/rec/model.yml",
        )

        validate_resume_contract(metadata, relocated)

    def test_resume_contract_rejects_different_relocated_model_config(self):
        metadata = {
            "task": "rec",
            "model_config": (
                "/home/heqi/project/PaddleOCR/route2/"
                "PaddleOCR2Pytorch-QAT/configs/rec/model.yml"
            ),
            "training_profile_sha256": "profile-a",
            "qat": True,
            "qat_config_sha256": "quant-a",
            "torch_version": "2.6.0",
            "reparameterized": True,
            "image_shape": [3, 48, 320],
        }
        changed = dict(
            metadata,
            model_config="/home/heqi/project/PaddleOCR/configs/rec/other.yml",
        )

        with self.assertRaisesRegex(ValueError, "model_config"):
            validate_resume_contract(metadata, changed)

    def test_resume_contract_relocation_keeps_hashes_strict(self):
        metadata = {
            "task": "rec",
            "model_config": (
                "/home/heqi/project/PaddleOCR/route2/"
                "PaddleOCR2Pytorch-QAT/configs/rec/model.yml"
            ),
            "training_profile_sha256": "profile-a",
            "qat": True,
            "qat_config_sha256": "quant-a",
            "torch_version": "2.6.0",
            "reparameterized": True,
            "image_shape": [3, 48, 320],
        }
        changed = dict(
            metadata,
            model_config="/home/heqi/project/PaddleOCR/configs/rec/model.yml",
            training_profile_sha256="profile-b",
        )

        with self.assertRaisesRegex(ValueError, "training_profile_sha256"):
            validate_resume_contract(metadata, changed)

    def test_relocates_known_legacy_checkpoint_paths(self):
        legacy_root = (
            "/home/heqi/project/PaddleOCR/route2/PaddleOCR2Pytorch-QAT"
        )
        metadata = {
            "model_config": f"{legacy_root}/configs/rec/model.yml",
            "qat_config": f"{legacy_root}/configs/qat/model.json",
            "weights": f"{legacy_root}/ptocr_model.pth",
            "label_file": "/home/heqi/dataset/labels.txt",
        }

        relocated = relocate_checkpoint_metadata(metadata)

        self.assertEqual(
            relocated["model_config"],
            "/home/heqi/project/PaddleOCR/configs/rec/model.yml",
        )
        self.assertEqual(
            relocated["qat_config"],
            "/home/heqi/project/PaddleOCR/configs/qat/model.json",
        )
        self.assertEqual(
            relocated["weights"],
            "/home/heqi/project/PaddleOCR/weights/ptocr_model.pth",
        )
        self.assertEqual(relocated["label_file"], metadata["label_file"])
        self.assertEqual(
            metadata["model_config"],
            f"{legacy_root}/configs/rec/model.yml",
        )

    def test_does_not_relocate_unrelated_absolute_path(self):
        path = "/home/heqi/dataset/model.yml"

        self.assertEqual(relocate_project_path(path), path)

if __name__ == "__main__":
    unittest.main()

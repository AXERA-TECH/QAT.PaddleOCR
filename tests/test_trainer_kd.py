import unittest

import torch
from torch import nn

from pytorchocr.training import Trainer, build_kd_criterion
from pytorchocr.training.losses import KDCompositeLoss


class SquaredLoss(nn.Module):
    def forward(self, predictions, targets):
        if isinstance(predictions, dict):
            predictions = predictions["ctc"]
        if isinstance(targets, dict):
            targets = targets["targets"].float()
        loss = torch.mean((predictions - targets) ** 2)
        return {"loss": loss}


class LinearModel(nn.Module):
    def __init__(self, out_features=8):
        super().__init__()
        self.fc = nn.Linear(4, out_features)

    def forward(self, images):
        return self.fc(images)


class TeacherWrapper(nn.Module):
    """Teacher exposing CTC-like logits and a neck feature for multi-layer KD."""

    def __init__(self, out_features=4):
        super().__init__()
        self.backbone = LinearModel(out_features)
        self.neck = nn.Identity()

    def forward(self, images):
        features = self.backbone(images)
        logits = self.neck(features)
        return logits


class KDRecModel(nn.Module):
    """Student that emits a dict {ctc, ctc_neck} for the full-rec KD contract."""

    def __init__(self, out_features=4):
        super().__init__()
        self.backbone = LinearModel(out_features)
        self.graph_role = "pretrained_train"
        self.model_type = "rec"

    def forward(self, images, data=None):
        features = self.backbone(images)
        return {"ctc": features, "ctc_neck": features, "gtc": features}


class KDTrainerTest(unittest.TestCase):
    def test_kd_requires_teacher_and_criterion_together(self):
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        with self.assertRaises(ValueError):
            Trainer(
                model,
                SquaredLoss(),
                optimizer,
                teacher=TeacherWrapper(),
                kd_criterion=None,
            )
        with self.assertRaises(ValueError):
            Trainer(
                model,
                SquaredLoss(),
                optimizer,
                teacher=None,
                kd_criterion=KDCompositeLoss({"ctc": ("mse", 1.0)}, task="rec"),
            )

    def test_teacher_frozen_after_construction(self):
        model = LinearModel()
        teacher = TeacherWrapper()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        trainer = Trainer(
            model,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=build_kd_criterion("rec", kd_layers={"ctc": ("mse", 1.0)}),
        )
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.parameters()))

    def test_teacher_batch_norm_stays_in_eval_mode(self):
        class BNDetTeacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 4, 3, padding=1),
                    nn.BatchNorm2d(4),
                )
                self.head = nn.Module()
                self.head.binarize = nn.Identity()
                self.head.thresh = nn.Identity()

            def step_function(self, shrink, threshold):
                return shrink * threshold

            def forward(self, images):
                return self.backbone(images)

        model = BNDetTeacher()
        teacher = BNDetTeacher()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        Trainer(
            model,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=build_kd_criterion("det", kd_layers={"maps": ("maps", 1.0)}),
        )
        # Teacher BN must use running statistics, not batch statistics.
        self.assertFalse(teacher.backbone[1].training)
        # The student keeps its training mode for task losses.
        self.assertTrue(model.backbone[1].training)

    def test_train_step_reports_kd_loss(self):
        model = LinearModel(out_features=8)
        teacher = TeacherWrapper(out_features=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        criterion = build_kd_criterion("rec", kd_layers={"ctc": ("mse", 1.0)})
        trainer = Trainer(
            model,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=criterion,
            kd_weight=0.5,
        )
        images = torch.randn(4, 4)
        targets = torch.randn(4, 8)
        expected_task = float(SquaredLoss()(model(images), targets)["loss"])
        losses = trainer.train_step(images, targets)
        self.assertIn("loss", losses)
        self.assertIn("kd_ctc", losses)
        # kd_loss = task_loss_before + 0.5 * kd_ctc (both from the same step).
        torch.testing.assert_close(
            torch.tensor(losses["loss"]),
            torch.tensor(expected_task + 0.5 * losses["kd_ctc"]),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_full_rec_dict_kd(self):
        model = KDRecModel()
        teacher = TeacherWrapper()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        criterion = build_kd_criterion(
            "rec",
            kd_layers={"ctc": ("mse", 1.0)},
            rec_graph="pretrained_train",
        )
        trainer = Trainer(
            model,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=criterion,
        )
        images = torch.randn(4, 4)
        targets = {
            "targets": torch.zeros(4, 4, dtype=torch.long),
            "gtc_targets": torch.zeros(4, 4, dtype=torch.long),
            "target_lengths": torch.ones(4, dtype=torch.long),
            "valid_ratio": torch.ones(4),
        }
        losses = trainer.train_step(images, targets)
        self.assertIn("kd_ctc", losses)

    def test_checkpoint_excludes_teacher(self):
        model = LinearModel()
        teacher = TeacherWrapper()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        criterion = build_kd_criterion("rec", kd_layers={"ctc": ("mse", 1.0)})
        trainer = Trainer(
            model,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=criterion,
        )
        state = trainer.state_dict()
        self.assertNotIn("teacher", state)
        self.assertIn("model", state)
        # Teacher state keys must not leak into the model state.
        model_keys = set(state["model"].keys())
        self.assertTrue(all("backbone.fc" not in key for key in model_keys))

    def test_no_teacher_behavior_unchanged(self):
        model = LinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        trainer = Trainer(model, SquaredLoss(), optimizer)
        images = torch.randn(4, 4)
        targets = torch.randn(4, 8)
        losses = trainer.train_step(images, targets)
        self.assertEqual(set(losses), {"loss"})

    def test_negative_kd_weight_rejected(self):
        model = LinearModel()
        teacher = TeacherWrapper()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        with self.assertRaises(ValueError):
            Trainer(
                model,
                SquaredLoss(),
                optimizer,
                teacher=teacher,
                kd_criterion=build_kd_criterion(
                    "rec",
                    kd_layers={"ctc": ("mse", 1.0)},
                ),
                kd_weight=-1.0,
            )

    def test_evaluate_reports_kd_loss(self):
        model = LinearModel(out_features=4)
        teacher = TeacherWrapper(out_features=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        criterion = build_kd_criterion("rec", kd_layers={"ctc": ("mse", 1.0)})
        trainer = Trainer(
            model,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=criterion,
            kd_weight=0.5,
        )
        images = torch.randn(2, 4)
        targets = {"targets": torch.randn(2, 4)}

        class OneBatchLoader:
            def __iter__(self):
                yield images, targets

        results = trainer.evaluate(OneBatchLoader())
        self.assertIn("loss", results)
        self.assertIn("kd_ctc", results)
        self.assertIn("kd_loss", results)

    def test_checkpoint_round_trip_preserves_kd_behavior(self):
        model = LinearModel(out_features=4)
        teacher = TeacherWrapper(out_features=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
        criterion = build_kd_criterion("rec", kd_layers={"ctc": ("mse", 1.0)})
        trainer = Trainer(
            model,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=criterion,
            kd_weight=1.0,
        )
        state = trainer.state_dict()
        self.assertNotIn("teacher", state)

        restored_model = LinearModel(out_features=4)
        restored_teacher = TeacherWrapper(out_features=4)
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=1.0e-3)
        restored = Trainer(
            restored_model,
            SquaredLoss(),
            restored_optimizer,
            teacher=restored_teacher,
            kd_criterion=build_kd_criterion(
                "rec",
                kd_layers={"ctc": ("mse", 1.0)},
            ),
        )
        restored.load_state_dict(state)
        images = torch.randn(2, 4)
        targets = {"targets": torch.randn(2, 4)}
        losses = restored.train_step(images, targets)
        self.assertIn("kd_ctc", losses)

    def test_teacher_full_recognition_data_path(self):
        """Bare pretrained_train teachers forward with (images, data)."""

        class BareRecTeacher(nn.Module):
            graph_role = "pretrained_train"
            model_type = "rec"
            return_all_feats = True

            def __init__(self, out_features=4):
                super().__init__()
                self.backbone = LinearModel(out_features)

            def forward(self, images, data=None):
                features = self.backbone(images)
                return {
                    "backbone_out": features,
                    "neck_out": features,
                    "head_out": {
                        "ctc": features,
                        "gtc": features,
                        "ctc_neck": features,
                    },
                }

        student = KDRecModel(out_features=4)
        teacher = BareRecTeacher(out_features=4)
        optimizer = torch.optim.SGD(student.parameters(), lr=1.0e-3)
        criterion = build_kd_criterion(
            "rec",
            kd_layers={"ctc": ("mse", 1.0)},
            rec_graph="pretrained_train",
        )
        trainer = Trainer(
            student,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=criterion,
        )
        images = torch.randn(4, 4)
        targets = {
            "targets": torch.zeros(4, 4, dtype=torch.long),
            "gtc_targets": torch.zeros(4, 4, dtype=torch.long),
            "target_lengths": torch.ones(4, dtype=torch.long),
            "valid_ratio": torch.ones(4),
        }
        losses = trainer.train_step(images, targets)
        self.assertIn("kd_ctc", losses)
        # return_all_feats teacher keeps full dict outputs while BN stays eval.
        self.assertTrue(teacher.training)
        self.assertTrue(
            all(
                not module.training
                for module in teacher.modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            )
        )

    def test_prepared_student_dict_outputs_with_teacher(self):
        """Exported (PT2E-like) student emitting a dict works with KD."""

        class DictRec(nn.Module):
            def __init__(self, out_features=4):
                super().__init__()
                self.projection = nn.Linear(4, out_features)

            def forward(self, images, gtc_targets=None):
                output = self.projection(images)
                return {"ctc": output, "ctc_neck": output, "gtc": output}

        gtc_targets = torch.zeros(2, 4, dtype=torch.long)
        exported = torch.export.export_for_training(
            DictRec(),
            (torch.randn(2, 4), gtc_targets),
        ).module()
        exported.graph_role = "pretrained_train"
        exported.model_type = "rec"
        teacher = TeacherWrapper(out_features=4)
        optimizer = torch.optim.SGD(exported.parameters(), lr=1.0e-3)
        criterion = build_kd_criterion(
            "rec",
            kd_layers={"ctc": ("mse", 1.0)},
            rec_graph="pretrained_train",
        )
        trainer = Trainer(
            exported,
            SquaredLoss(),
            optimizer,
            teacher=teacher,
            kd_criterion=criterion,
        )
        images = torch.randn(2, 4)
        targets = {
            "targets": torch.zeros(2, 4, dtype=torch.long),
            "gtc_targets": torch.zeros(2, 4, dtype=torch.long),
            "target_lengths": torch.ones(2, dtype=torch.long),
            "valid_ratio": torch.ones(2),
        }
        losses = trainer.train_step(images, targets)
        self.assertIn("kd_ctc", losses)


if __name__ == "__main__":
    unittest.main()

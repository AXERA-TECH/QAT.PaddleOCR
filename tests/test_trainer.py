import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from pytorchocr.training import Trainer, update_best_validation
from pytorchocr.quantization import DetTrainingWrapper, RecCTCWrapper


class SquaredLoss(nn.Module):
    def forward(self, predictions, targets):
        loss = torch.mean((predictions - targets) ** 2)
        return {"loss": loss}


class NonFiniteLoss(nn.Module):
    def forward(self, predictions, targets):
        return {"loss": predictions.sum() * torch.tensor(float("nan"))}


class ConvReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, images):
        return self.relu(self.conv(images))


class SampleCountMetric:
    main_indicator = "samples"

    def reset(self):
        self.samples = 0

    def update(self, outputs, targets):
        self.samples += outputs.shape[0]

    def compute(self):
        return {"samples": float(self.samples)}


class EvalSoftmaxCTCHead(nn.Module):
    def forward(self, inputs):
        return inputs if self.training else inputs.softmax(dim=-1)


class DummyRecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_transform = False
        self.use_neck = False
        self.backbone = nn.Identity()
        self.head = nn.Module()
        self.head.ctc_encoder = nn.Identity()
        self.head.ctc_head = EvalSoftmaxCTCHead()


class BranchingNeck(nn.Module):
    def forward(self, inputs):
        return {"fuse": inputs} if self.training else inputs


class TensorNeck(nn.Module):
    def forward(self, inputs):
        return inputs


class DummyDBHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.binarize = nn.Identity()
        self.thresh = nn.Identity()

    @staticmethod
    def step_function(shrink, threshold):
        return shrink - threshold


class DummyDetModel(nn.Module):
    def __init__(self, neck=None):
        super().__init__()
        self.backbone = nn.Identity()
        self.neck = neck if neck is not None else BranchingNeck()
        self.head = DummyDBHead()


class FullRecognitionModel(nn.Module):
    graph_role = "pretrained_train"
    model_type = "rec"

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)
        self.last_data = None

    def forward(self, images, data=None):
        self.last_data = data
        return self.projection(images)


class FullRecognitionLoss(nn.Module):
    def forward(self, predictions, targets):
        return {"loss": predictions.square().mean()}


class PreparedFullRecognitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_names = None

    def forward(self, predictions, targets):
        self.output_names = tuple(predictions)
        return {
            "loss": sum(value.square().mean() for value in predictions.values())
        }


class FullDetectionModel(nn.Module):
    graph_role = "pretrained_train"
    model_type = "det"

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, images):
        return self.projection(images)


class TrainerTest(unittest.TestCase):
    def build_trainer(self):
        model = nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return Trainer(model, SquaredLoss(), optimizer, scheduler=scheduler)

    def test_checkpoint_round_trip(self):
        torch.manual_seed(11)
        trainer = self.build_trainer()
        trainer.train_step(torch.randn(4, 3), torch.randn(4, 2))
        trainer.epoch = 3
        trainer.best_validation_loss = 0.25
        trainer.best_validation_metric_name = "score"
        trainer.best_validation_metric_value = 0.75
        trainer.step_scheduler()
        expected = {
            name: value.detach().clone()
            for name, value in trainer.model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(checkpoint, {"model": "smoke"})
            restored = self.build_trainer()
            metadata = restored.load_checkpoint(checkpoint)
        self.assertEqual(metadata, {"model": "smoke"})
        self.assertEqual(restored.epoch, 3)
        self.assertEqual(restored.global_step, 1)
        self.assertEqual(restored.best_validation_loss, 0.25)
        self.assertEqual(restored.best_validation_metric_name, "score")
        self.assertEqual(restored.best_validation_metric_value, 0.75)
        for name, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, expected[name])

    def test_evaluate_does_not_update_model(self):
        torch.manual_seed(13)
        trainer = self.build_trainer()
        images = torch.randn(5, 3)
        targets = torch.randn(5, 2)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)
        expected = {
            name: value.detach().clone()
            for name, value in trainer.model.state_dict().items()
        }
        losses = trainer.evaluate(loader)
        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected[name])

    def test_train_step_rejects_nonfinite_loss(self):
        model = nn.Linear(3, 2)
        trainer = Trainer(
            model,
            NonFiniteLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
        )

        with self.assertRaisesRegex(FloatingPointError, "Non-finite training loss"):
            trainer.train_step(torch.randn(2, 3), torch.randn(2, 2))

    def test_full_recognition_forwards_ctc_and_gtc_targets(self):
        model = FullRecognitionModel()
        trainer = Trainer(
            model,
            FullRecognitionLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
        )
        targets = {
            "targets": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "gtc_targets": torch.tensor(
                [[2, 4, 5, 3, 0], [2, 6, 3, 0, 0]]
            ),
            "target_lengths": torch.tensor([2, 1]),
            "valid_ratio": torch.tensor([1.0, 0.75]),
        }

        trainer.train_step(torch.randn(2, 3), targets)

        self.assertEqual(len(model.last_data), 4)
        for actual, name in zip(
            model.last_data,
            ("targets", "gtc_targets", "target_lengths", "valid_ratio"),
        ):
            torch.testing.assert_close(actual, targets[name])

    def test_full_recognition_rejects_missing_gtc_targets(self):
        model = FullRecognitionModel()
        trainer = Trainer(
            model,
            FullRecognitionLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
        )
        targets = {
            "targets": torch.tensor([[1, 2, 0]]),
            "target_lengths": torch.tensor([2]),
            "valid_ratio": torch.tensor([1.0]),
        }

        with self.assertRaisesRegex(ValueError, "gtc_targets"):
            trainer.train_step(torch.randn(1, 3), targets)

    def test_prepared_full_recognition_maps_tuple_outputs_for_multiloss(self):
        class PreparedFullRec(nn.Module):
            graph_role = "pretrained_train"
            model_type = "rec"
            output_names = ("ctc", "ctc_neck", "gtc")

            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(3, 2)

            def forward(self, images, gtc_targets):
                output = self.projection(images)
                return output, output + 1.0, output + 2.0

        targets = {
            "targets": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "gtc_targets": torch.tensor([[2, 4, 3], [2, 3, 0]]),
            "target_lengths": torch.tensor([2, 1]),
            "valid_ratio": torch.tensor([1.0, 0.75]),
        }
        exported = torch.export.export_for_training(
            PreparedFullRec(),
            (torch.randn(2, 3), targets["gtc_targets"]),
        ).module()
        exported.graph_role = "pretrained_train"
        exported.model_type = "rec"
        exported.output_names = PreparedFullRec.output_names
        criterion = PreparedFullRecognitionLoss()
        trainer = Trainer(
            exported,
            criterion,
            torch.optim.SGD(exported.parameters(), lr=0.1),
        )

        losses = trainer.train_step(torch.randn(2, 3), targets)

        self.assertIn("loss", losses)
        self.assertEqual(criterion.output_names, ("ctc", "ctc_neck", "gtc"))

    def test_full_detection_does_not_require_recognition_targets(self):
        model = FullDetectionModel()
        trainer = Trainer(
            model,
            SquaredLoss(),
            torch.optim.SGD(model.parameters(), lr=0.1),
        )

        losses = trainer.train_step(torch.randn(2, 3), torch.randn(2, 2))

        self.assertIn("loss", losses)

    def test_evaluate_merges_task_metric(self):
        trainer = self.build_trainer()
        images = torch.randn(5, 3)
        targets = torch.randn(5, 2)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)

        result = trainer.evaluate(loader, metric=SampleCountMetric())

        self.assertEqual(result["samples"], 5.0)
        self.assertIn("loss", result)

    def test_evaluate_restores_exported_batch_counters(self):
        class BatchNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(3)

            def forward(self, inputs):
                return self.bn(inputs)

        exported = torch.export.export_for_training(
            BatchNormModel().train(),
            (torch.randn(2, 3),),
        ).module()
        optimizer = torch.optim.SGD(exported.parameters(), lr=0.01)
        trainer = Trainer(exported, SquaredLoss(), optimizer)
        loader = DataLoader(
            TensorDataset(torch.randn(4, 3), torch.zeros(4, 3)),
            batch_size=2,
        )
        before = {
            name: value.clone()
            for name, value in exported.named_buffers()
            if name.endswith("num_batches_tracked")
        }

        trainer.evaluate(loader)

        after = {
            name: value
            for name, value in exported.named_buffers()
            if name.endswith("num_batches_tracked")
        }
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            torch.testing.assert_close(after[name], before[name])

    def test_exported_mode_transitions_are_tracked(self):
        exported = torch.export.export_for_training(
            nn.Linear(3, 2).train(),
            (torch.randn(2, 3),),
        ).module()
        trainer = Trainer(
            exported,
            SquaredLoss(),
            torch.optim.SGD(exported.parameters(), lr=0.01),
        )
        loader = DataLoader(
            TensorDataset(torch.randn(2, 3), torch.randn(2, 2)),
            batch_size=2,
        )

        self.assertEqual(trainer._exported_mode, "train")
        trainer.evaluate(loader)
        self.assertEqual(trainer._exported_mode, "eval")
        trainer.train_step(torch.randn(2, 3), torch.randn(2, 2))
        self.assertEqual(trainer._exported_mode, "train")

    def test_rec_validation_mode_preserves_raw_ctc_logits(self):
        wrapper = RecCTCWrapper(DummyRecModel())
        logits = torch.tensor([[[2.0, -1.0]]])
        wrapper.eval()
        self.assertNotEqual(float(wrapper(logits)[0, 0, 0]), 2.0)

        wrapper.set_validation_mode()

        torch.testing.assert_close(wrapper(logits), logits)
        self.assertFalse(wrapper.training)

    def test_det_validation_mode_preserves_training_graph_outputs(self):
        wrapper = DetTrainingWrapper(DummyDetModel())
        inputs = torch.ones(1, 1, 2, 2)
        wrapper.set_validation_mode()

        outputs = wrapper(inputs)

        self.assertEqual(len(outputs), 3)
        torch.testing.assert_close(outputs[0], inputs)
        self.assertFalse(wrapper.training)
        self.assertTrue(wrapper.model.neck.training)

    def test_det_wrapper_accepts_tensor_neck_output(self):
        wrapper = DetTrainingWrapper(DummyDetModel(neck=TensorNeck()))
        inputs = torch.ones(1, 1, 2, 2)

        outputs = wrapper(inputs)

        self.assertEqual(len(outputs), 3)
        torch.testing.assert_close(outputs[0], inputs)

    def test_best_checkpoint_selection_prefers_main_metric(self):
        trainer = self.build_trainer()
        metric = SampleCountMetric()

        self.assertTrue(
            update_best_validation(
                trainer, {"loss": 2.0, "samples": 4.0}, metric=metric
            )
        )
        self.assertFalse(
            update_best_validation(
                trainer, {"loss": 1.0, "samples": 3.0}, metric=metric
            )
        )
        self.assertEqual(trainer.best_validation_loss, 1.0)
        self.assertEqual(trainer.best_validation_metric_value, 4.0)
        self.assertTrue(
            update_best_validation(
                trainer, {"loss": 1.5, "samples": 5.0}, metric=metric
            )
        )

    def test_qat_evaluate_restores_observers(self):
        torch.manual_seed(17)
        images = torch.randn(2, 3, 8, 8)
        targets = torch.randn(2, 4, 8, 8)
        trainer = self.build_pt2e_trainer(images)
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)
        trainer.evaluate(loader)
        observers = [
            module
            for module in trainer.model.modules()
            if hasattr(module, "observer_enabled")
        ]
        self.assertTrue(observers)
        self.assertTrue(
            all(int(module.observer_enabled[0]) == 1 for module in observers)
        )
        trainer.freeze_observers()
        trainer.evaluate(loader)
        self.assertTrue(
            all(int(module.observer_enabled[0]) == 0 for module in observers)
        )

    def build_pt2e_trainer(self, images):
        exported = torch.export.export_for_training(
            ConvReluModel().train(),
            (images,),
        ).module()
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_qat=True)
        )
        model = prepare_qat_pt2e(exported, quantizer)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return Trainer(model, SquaredLoss(), optimizer)

    def test_pt2e_qat_observer_checkpoint_round_trip(self):
        torch.manual_seed(19)
        images = torch.randn(2, 3, 8, 8)
        targets = torch.randn(2, 4, 8, 8)
        trainer = self.build_pt2e_trainer(images)
        losses = trainer.train_step(images, targets)
        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        trainer.freeze_observers()
        observers = [
            module
            for module in trainer.model.modules()
            if hasattr(module, "observer_enabled")
        ]
        self.assertTrue(observers)
        self.assertTrue(
            all(int(module.observer_enabled[0]) == 0 for module in observers)
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pt2e_qat.pt"
            trainer.save_checkpoint(checkpoint)
            restored = self.build_pt2e_trainer(images)
            restored.load_checkpoint(checkpoint)

        self.assertEqual(restored.global_step, 1)
        self.assertTrue(restored.observers_frozen)
        restored_observers = [
            module
            for module in restored.model.modules()
            if hasattr(module, "observer_enabled")
        ]
        self.assertTrue(restored_observers)
        self.assertTrue(
            all(int(module.observer_enabled[0]) == 0 for module in restored_observers)
        )


if __name__ == "__main__":
    unittest.main()

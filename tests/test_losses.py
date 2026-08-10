import unittest

import torch

from pytorchocr.training import build_criterion, load_ocr_config
from pytorchocr.training.losses import CTCLoss, DBLoss, MultiLoss, NRTRLoss
from pytorchocr.training.losses.db import BalanceLoss


class LossSmokeTest(unittest.TestCase):
    def test_v5_det_factory_uses_paddle_dice_balance_loss(self):
        config = load_ocr_config(
            "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml"
        )

        criterion = build_criterion("det", config)

        self.assertIsInstance(criterion.segmentation, BalanceLoss)

    def test_dice_balance_loss_matches_paddle_scalar_ohem_semantics(self):
        prediction = torch.tensor([[[0.8, 0.2], [0.4, 0.1]]])
        target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        mask = torch.ones_like(target)
        criterion = BalanceLoss(balance_loss=True, negative_ratio=3.0)

        actual = criterion(prediction, target, mask)
        origin = criterion.loss(prediction, target, mask)
        expected = origin * 4.0 / (4.0 + criterion.eps)

        torch.testing.assert_close(actual, expected)

    def test_dice_balance_loss_returns_zero_without_positive_pixels(self):
        prediction = torch.full((1, 2, 2), 0.25)
        target = torch.zeros_like(prediction)
        mask = torch.ones_like(prediction)

        loss = BalanceLoss()(prediction, target, mask)

        torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_db_loss_backward(self):
        logits = torch.randn(2, 3, 16, 16, requires_grad=True)
        predictions = torch.sigmoid(logits)
        labels = {
            "threshold_map": torch.rand(2, 16, 16),
            "threshold_mask": torch.ones(2, 16, 16),
            "shrink_map": torch.randint(0, 2, (2, 16, 16)).float(),
            "shrink_mask": torch.ones(2, 16, 16),
        }
        losses = DBLoss()(predictions, labels)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_db_loss_accepts_expose_intermediates_dict_maps(self):
        """KD wrapper dicts carry maps as a tuple; DBLoss must consume them."""
        maps = tuple(
            torch.sigmoid(torch.randn(2, 1, 16, 16, requires_grad=True))
            for _ in range(3)
        )
        for tensor in maps:
            tensor.retain_grad()
        predictions = {"maps": maps}
        labels = {
            "threshold_map": torch.rand(2, 16, 16),
            "threshold_mask": torch.ones(2, 16, 16),
            "shrink_map": torch.randint(0, 2, (2, 16, 16)).float(),
            "shrink_mask": torch.ones(2, 16, 16),
        }
        losses = DBLoss()(predictions, labels)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        for tensor in maps:
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_db_loss_accepts_bare_maps_tuple(self):
        maps = tuple(
            torch.sigmoid(torch.randn(2, 1, 16, 16)) for _ in range(3)
        )
        labels = {
            "threshold_map": torch.rand(2, 16, 16),
            "threshold_mask": torch.ones(2, 16, 16),
            "shrink_map": torch.randint(0, 2, (2, 16, 16)).float(),
            "shrink_mask": torch.ones(2, 16, 16),
        }
        losses = DBLoss()(maps, labels)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_ctc_loss_backward(self):
        logits = torch.randn(2, 8, 6, requires_grad=True)
        targets = torch.tensor([[1, 2, 3], [2, 4, 0]])
        target_lengths = torch.tensor([3, 2])
        losses = CTCLoss()(logits, targets, target_lengths)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ctc_loss_matches_paddle_none_then_batch_mean(self):
        logits = torch.tensor(
            [
                [[2.0, 0.1, -0.2, 0.5], [0.4, 1.2, -0.7, 0.3]],
                [[1.4, 0.2, 0.5, -0.1], [0.1, 0.7, 1.1, -0.4]],
                [[0.2, 1.1, 0.1, -0.3], [0.8, -0.2, 0.3, 0.5]],
                [[1.0, 0.6, -0.1, 0.2], [0.3, 0.1, 1.3, -0.5]],
                [[0.4, -0.2, 1.2, 0.3], [0.9, 0.2, -0.3, 0.7]],
            ]
        ).transpose(0, 1)
        targets = torch.tensor([[1, 2, 0], [1, 2, 1]])
        target_lengths = torch.tensor([2, 3])

        loss = CTCLoss()(logits, targets, target_lengths)["loss"]

        torch.testing.assert_close(loss, torch.tensor(2.473982095718384))

    def test_nrtr_loss_ignores_padding_and_backpropagates(self):
        logits = torch.randn(2, 3, 9, requires_grad=True)
        targets = {
            "gtc_targets": torch.tensor(
                [
                    [2, 4, 5, 3, 0, 0],
                    [2, 6, 3, 0, 0, 0],
                ]
            ),
            "target_lengths": torch.tensor([2, 1]),
        }

        loss = NRTRLoss()(logits, targets)["loss"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_nrtr_loss_crops_fixed_graph_decoder_to_batch_target_length(self):
        logits = torch.randn(2, 8, 9, requires_grad=True)
        targets = {
            "gtc_targets": torch.tensor(
                [
                    [2, 4, 5, 3, 0, 0, 0, 0],
                    [2, 6, 3, 0, 0, 0, 0, 0],
                ]
            ),
            "target_lengths": torch.tensor([2, 1]),
        }

        loss = NRTRLoss()(logits, targets)["loss"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[:, 3:, :], torch.zeros_like(logits.grad[:, 3:, :])))

    def test_multi_loss_combines_ctc_and_gtc(self):
        ctc_logits = torch.randn(2, 6, 8, requires_grad=True)
        gtc_logits = torch.randn(2, 3, 11, requires_grad=True)
        targets = {
            "targets": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "gtc_targets": torch.tensor(
                [[2, 4, 5, 3, 0, 0], [2, 6, 3, 0, 0, 0]]
            ),
            "target_lengths": torch.tensor([2, 1]),
        }

        losses = MultiLoss()(
            {"ctc": ctc_logits, "gtc": gtc_logits},
            targets,
        )
        losses["loss"].backward()

        torch.testing.assert_close(
            losses["loss"],
            losses["CTCLoss"] + losses["NRTRLoss"],
        )
        self.assertTrue(torch.isfinite(ctc_logits.grad).all())
        self.assertTrue(torch.isfinite(gtc_logits.grad).all())

    def test_multi_loss_accepts_ctc_only_validation_output(self):
        ctc_logits = torch.randn(2, 6, 8, requires_grad=True)
        targets = {
            "targets": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "gtc_targets": torch.tensor(
                [[2, 4, 5, 3, 0, 0], [2, 6, 3, 0, 0, 0]]
            ),
            "target_lengths": torch.tensor([2, 1]),
        }

        losses = MultiLoss()(ctc_logits, targets)

        self.assertEqual(set(losses), {"CTCLoss", "loss"})
        torch.testing.assert_close(losses["loss"], losses["CTCLoss"])


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from torch import nn

from pytorchocr.quantization import DetTrainingWrapper, FullRecTrainingWrapper


class IdentityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.binarize = nn.Identity()
        self.thresh = nn.Identity()

    def step_function(self, shrink, threshold):
        return shrink * threshold


class FakeDetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.neck = nn.Identity()
        self.head = IdentityHead()


class FakeFullRecModel(nn.Module):
    use_transform = False
    use_neck = True

    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.neck = nn.Identity()
        self.head = nn.Module()
        self.head.ctc_encoder = nn.Identity()
        self.head.ctc_head = nn.Identity()
        self.head.before_gtc = nn.Identity()
        self.head.gtc_head = nn.Module()
        self.head.gtc_head.forward_train = (
            lambda features, targets: features[:, :1, :, :]
        )


class DetTrainingWrapperKDTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.randn(2, 3, 16, 16)

    def test_default_returns_maps_tuple(self):
        wrapper = DetTrainingWrapper(FakeDetModel())
        outputs = wrapper(self.features)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 3)
        # shrink is Identity -> same tensor; threshold same; binary = shrink*thr
        self.assertEqual(outputs[0].shape, (2, 3, 16, 16))

    def test_expose_intermediates_returns_dict(self):
        wrapper = DetTrainingWrapper(FakeDetModel(), expose_intermediates=True)
        outputs = wrapper(self.features)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(
            set(outputs),
            {"maps", "backbone_out", "neck_out"},
        )
        self.assertIsInstance(outputs["maps"], tuple)
        self.assertEqual(len(outputs["maps"]), 3)
        self.assertTrue(torch.is_tensor(outputs["backbone_out"]))
        self.assertTrue(torch.is_tensor(outputs["neck_out"]))
        self.assertEqual(outputs["neck_out"].shape, (2, 3, 16, 16))

    def test_expose_flag_defaults_false(self):
        wrapper = DetTrainingWrapper(FakeDetModel())
        self.assertFalse(wrapper.expose_intermediates)


class FullRecTrainingWrapperKDTest(unittest.TestCase):
    def setUp(self):
        self.features = torch.randn(2, 3, 12, 32)
        self.targets = torch.zeros(2, 10, dtype=torch.long)

    def test_default_returns_ctc_tuple(self):
        wrapper = FullRecTrainingWrapper(FakeFullRecModel(), max_text_length=25)
        outputs = wrapper(self.features, self.targets)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 3)

    def test_expose_intermediates_returns_dict(self):
        wrapper = FullRecTrainingWrapper(
            FakeFullRecModel(),
            max_text_length=25,
            expose_intermediates=True,
        )
        outputs = wrapper(self.features, self.targets)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(
            set(outputs),
            {"ctc", "ctc_neck", "gtc", "backbone_out", "neck_out"},
        )
        for key in ("ctc", "ctc_neck", "gtc", "backbone_out", "neck_out"):
            self.assertTrue(torch.is_tensor(outputs[key]), key)

    def test_expose_flag_defaults_false(self):
        wrapper = FullRecTrainingWrapper(FakeFullRecModel(), max_text_length=25)
        self.assertFalse(wrapper.expose_intermediates)


if __name__ == "__main__":
    unittest.main()

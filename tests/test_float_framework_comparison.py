import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch  # Import once before temporary module patches used by load_module().


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_module():
    fake_paddle = types.ModuleType("paddle")
    fake_paddle.Tensor = type("Tensor", (), {})
    fake_data = types.ModuleType("ppocr.data")
    fake_data.build_dataloader = mock.Mock()
    fake_metrics = types.ModuleType("ppocr.metrics")
    fake_metrics.build_metric = mock.Mock()
    fake_arch = types.ModuleType("ppocr.modeling.architectures")
    fake_arch.build_model = mock.Mock()
    fake_post = types.ModuleType("ppocr.postprocess")
    fake_post.build_post_process = mock.Mock()
    fake_training = types.ModuleType("pytorchocr.training")
    fake_training.build_det_model = mock.Mock()
    fake_training.build_rec_model = mock.Mock()
    fake_training.load_ocr_config = mock.Mock()
    fake_training.rec_out_channels = mock.Mock()
    modules = {
        "paddle": fake_paddle,
        "ppocr.data": fake_data,
        "ppocr.metrics": fake_metrics,
        "ppocr.modeling.architectures": fake_arch,
        "ppocr.postprocess": fake_post,
        "pytorchocr.training": fake_training,
    }
    path = ROOT_DIR / "tools/compare_float_frameworks.py"
    spec = importlib.util.spec_from_file_location("_float_comparison_test", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class DifferenceAccumulatorTest(unittest.TestCase):
    def test_aggregates_without_batch_weighting(self):
        module = load_module()
        stats = module.DifferenceAccumulator()
        stats.update(np.array([0.0]), np.array([1.0]))
        stats.update(
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 3.0, 3.0]),
            sample_offset=1,
        )
        result = stats.result()
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["mae"], 2.5)
        self.assertEqual(result["max_abs"], 3.0)
        self.assertEqual(result["p99_sample_count"], 4)
        self.assertEqual(result["worst_sample_index"], 1)

    def test_p99_sampling_is_bounded(self):
        module = load_module()
        stats = module.DifferenceAccumulator(samples_per_update=4)
        stats.update(np.zeros(100), np.arange(100))
        result = stats.result()
        self.assertEqual(result["count"], 100)
        self.assertEqual(result["p99_sample_count"], 4)

    def test_rejects_shape_mismatch(self):
        module = load_module()
        stats = module.DifferenceAccumulator()
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            stats.update(np.zeros((1, 2)), np.zeros((2, 1)))

    def test_centered_logits_remove_constant_offset(self):
        module = load_module()
        logits = np.array([[[1.0, 2.0, 4.0]]])
        np.testing.assert_allclose(
            module.centered(logits), module.centered(logits + 8.0), atol=1e-12
        )
        np.testing.assert_allclose(
            module.softmax(logits), module.softmax(logits + 8.0), atol=1e-12
        )

    def test_rec_behavior_tracks_first_text_mismatch(self):
        module = load_module()
        stats = module.BehaviorAccumulator("rec")
        stats.update(
            [("same", 0.9), ("expected", 0.8)],
            [("same", 0.7), ("actual", 0.6)],
            sample_offset=10,
        )
        result = stats.result()
        self.assertEqual(result["agreement"], 0.5)
        self.assertEqual(result["first_mismatch_index"], 11)

    def test_det_behavior_uses_one_pixel_tolerance(self):
        module = load_module()
        stats = module.BehaviorAccumulator("det")
        stats.update(
            [{"points": np.array([[[0, 0], [2, 2]]])}],
            [{"points": np.array([[[0, 1], [2, 2]]])}],
            sample_offset=0,
        )
        result = stats.result()
        self.assertEqual(result["agreement"], 1.0)
        self.assertEqual(result["box_count_agreement"], 1.0)

    def test_route_builder_forwards_reparameterize(self):
        module = load_module()
        wrapper = mock.Mock()
        wrapper.eval.return_value = wrapper
        wrapper.to.return_value = wrapper
        module.build_det_model.return_value = wrapper
        result = module.build_route2(
            "model.yml", "weights.pth", "det", "cpu", reparameterize=True
        )
        self.assertIs(result, wrapper)
        module.build_det_model.assert_called_once_with(
            "model.yml",
            weights_path="weights.pth",
            reparameterize=True,
            graph_mode="inference",
        )


if __name__ == "__main__":
    unittest.main()

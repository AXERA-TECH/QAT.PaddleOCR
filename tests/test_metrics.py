import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pytorchocr.training import (
    CTCRecognitionMetric,
    DetectionIoUEvaluator,
    DetectionMetric,
    build_validation_metric,
    load_ocr_config,
)


class StaticPostProcess:
    def __init__(self, points):
        self.points = points

    def __call__(self, outputs, shape_list):
        return [{"points": points} for points in self.points]


class MetricTest(unittest.TestCase):
    def test_detection_iou_handles_care_and_ignored_regions(self):
        care = np.asarray([[0, 0], [10, 0], [10, 10], [0, 10]])
        ignored = np.asarray([[20, 0], [30, 0], [30, 10], [20, 10]])
        evaluator = DetectionIoUEvaluator()
        result = evaluator.evaluate_image(
            [
                {"points": care, "ignore": False},
                {"points": ignored, "ignore": True},
            ],
            [{"points": care}, {"points": ignored}],
        )

        self.assertEqual(result, {"gt_care": 1, "det_care": 1, "matched": 1})
        self.assertEqual(
            evaluator.combine_results([result]),
            {"precision": 1.0, "recall": 1.0, "hmean": 1.0},
        )

    def test_detection_metric_uses_training_graph_shrink_output(self):
        polygon = np.asarray([[2, 2], [12, 2], [12, 12], [2, 12]])
        metric = DetectionMetric(StaticPostProcess([[polygon]]))
        outputs = (
            torch.ones(1, 1, 16, 16),
            torch.zeros(1, 1, 16, 16),
            torch.zeros(1, 1, 16, 16),
        )
        targets = {
            "polygons": [[polygon]],
            "ignore_tags": [np.asarray([False])],
            "shape": [np.asarray([16, 16, 1, 1], dtype=np.float32)],
        }

        metric.update(outputs, targets)

        self.assertEqual(
            metric.compute(),
            {"precision": 1.0, "recall": 1.0, "hmean": 1.0},
        )

    def test_ctc_recognition_accuracy_and_normalized_edit_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            dictionary = Path(directory) / "dict.txt"
            dictionary.write_text("a\nb\nc\n", encoding="utf-8")
            metric = CTCRecognitionMetric(dictionary)

            logits = torch.full((2, 4, 4), -10.0)
            for batch_index, indices in enumerate(([1, 1, 0, 2], [1, 0, 0, 0])):
                for time_index, class_index in enumerate(indices):
                    logits[batch_index, time_index, class_index] = 10.0
            targets = {
                "targets": torch.tensor([[1, 2, 0], [1, 3, 0]]),
            }
            metric.update(logits, targets)

        result = metric.compute()
        self.assertEqual(result["acc"], 0.5)
        self.assertEqual(result["norm_edit_dis"], 0.75)

    def test_checked_in_model_configs_build_task_metrics(self):
        det_path = "configs/det/PP-OCRv6/PP-OCRv6_small_det.yml"
        rec_path = "configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml"
        det_metric = build_validation_metric(
            "det", load_ocr_config(det_path), config_path=det_path
        )
        rec_metric = build_validation_metric(
            "rec", load_ocr_config(rec_path), config_path=rec_path
        )

        self.assertEqual(det_metric.main_indicator, "hmean")
        self.assertEqual(rec_metric.main_indicator, "acc")


if __name__ == "__main__":
    unittest.main()

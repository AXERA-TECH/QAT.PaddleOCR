import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from axera.eval_board_rec import make_parts, recognition_metrics


class BoardRecognitionMetricTests(unittest.TestCase):
    def test_matches_training_levenshtein_contract(self):
        metrics = recognition_metrics(["ab", "a"], ["ab", "ac"])

        self.assertEqual(metrics["acc"], 0.5)
        self.assertEqual(metrics["norm_edit_dis"], 0.75)

    def test_ignores_spaces_by_default(self):
        metrics = recognition_metrics(["a b"], ["ab"])

        self.assertEqual(metrics, {"acc": 1.0, "norm_edit_dis": 1.0})

    def test_make_parts_writes_matching_text_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                make_parts=directory,
                data_dir=directory,
                parts_size=1,
            )
            tensor = np.zeros((3, 48, 320), dtype=np.float32)
            with (
                mock.patch("axera.eval_board_rec.resolve_image", return_value="image.jpg"),
                mock.patch("axera.eval_board_rec.load_image", return_value=np.zeros((8, 8, 3))),
                mock.patch("axera.eval_board_rec.preprocess_image", return_value=tensor),
            ):
                make_parts(args, (3, 48, 320), ["a.jpg", "b.jpg"], ["A", "B"])

            self.assertEqual(
                json.loads((Path(directory) / "texts.json").read_text(encoding="utf-8")),
                ["A", "B"],
            )
            self.assertEqual(len(list(Path(directory).glob("part_*.npz"))), 2)


if __name__ == "__main__":
    unittest.main()

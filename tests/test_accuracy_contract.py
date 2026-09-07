import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pytorchocr.diagnostics import (
    audit_detection_label,
    audit_recognition_label,
    json_hash,
    stable_debug_selection,
)


class AccuracyContractTest(unittest.TestCase):
    @staticmethod
    def write_image(directory, name, shape=(20, 40, 3)):
        path = Path(directory) / name
        image = np.full(shape, 127, dtype=np.uint8)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")

    def test_json_hash_is_key_order_independent(self):
        self.assertEqual(json_hash({"a": 1, "b": 2}), json_hash({"b": 2, "a": 1}))

    def test_detection_audit_counts_polygons_and_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "det.jpg")
            annotations = [
                {"transcription": "text", "points": [[1, 1], [9, 1], [9, 9], [1, 9]]},
                {"transcription": "###", "points": [[10, 1], [18, 1], [18, 9], [10, 9]]},
            ]
            labels = Path(directory) / "det.txt"
            labels.write_text(
                f"det.jpg\t{json.dumps(annotations)}\n",
                encoding="utf-8",
            )
            first = audit_detection_label(labels, directory, 1)
            second = audit_detection_label(labels, directory, 1)

        self.assertEqual(first, second)
        self.assertEqual(first["care_polygons"], 1)
        self.assertEqual(first["ignore_polygons"], 1)
        self.assertEqual(first["missing_images"], 0)
        self.assertEqual(len(first["debug_samples"]), 1)

    def test_recognition_audit_reports_unknown_and_overlength(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "first.jpg")
            self.write_image(directory, "second.jpg", shape=(20, 80, 3))
            labels = Path(directory) / "rec.txt"
            labels.write_text("first.jpg\tabba\nsecond.jpg\tacccc\n", encoding="utf-8")
            result = audit_recognition_label(
                labels,
                directory,
                ["a", "b"],
                max_text_length=4,
                debug_count=2,
            )

        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["loader_accepted_samples"], 1)
        self.assertEqual(result["strictly_encodable_samples"], 1)
        self.assertEqual(result["overlength_texts"], 1)
        self.assertEqual(result["unknown_characters"], {"c": 4})

    def test_debug_selection_is_stable_and_unique(self):
        records = [
            {"line": index, "score": float(index), "image": f"{index}.jpg"}
            for index in range(1, 11)
        ]
        first = stable_debug_selection(records, 6, ("score",))
        second = stable_debug_selection(records, 6, ("score",))

        self.assertEqual(first, second)
        self.assertEqual(len({item["line"] for item in first}), 6)
        self.assertIn(1, {item["line"] for item in first})
        self.assertIn(10, {item["line"] for item in first})


if __name__ == "__main__":
    unittest.main()

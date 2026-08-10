import json
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from pytorchocr.training import CTCLoss, DBLoss
from pytorchocr.training.data import (
    DetectionDataset,
    RecognitionMultiScaleBatchSampler,
    RecognitionDataset,
    detection_collate,
)
from pytorchocr.training.data.augmentation import (
    DetectionSample,
    PaddleDetectionAugmentation,
    PaddleRecognitionAugmentation,
)


class DatasetTest(unittest.TestCase):
    @staticmethod
    def write_image(directory, name, shape=(32, 64, 3)):
        image = np.full(shape, 127, dtype=np.uint8)
        path = Path(directory) / name
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write test image: {path}")
        return path

    def test_detection_dataset_and_db_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "det.jpg")
            annotation = [
                {
                    "transcription": "text",
                    "points": [[8, 8], [48, 8], [48, 24], [8, 24]],
                }
            ]
            labels = Path(directory) / "det.txt"
            labels.write_text(
                f"det.jpg\t{json.dumps(annotation)}\n",
                encoding="utf-8",
            )
            dataset = DetectionDataset(
                labels,
                data_dir=directory,
                image_shape=(3, 32, 64),
            )
            image, targets = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 32, 64))
        self.assertGreater(float(targets["shrink_map"].sum()), 0)
        predictions = tuple(
            torch.sigmoid(torch.randn(1, 1, 32, 64, requires_grad=True))
            for _ in range(3)
        )
        batched_targets = {
            name: value.unsqueeze(0) for name, value in targets.items()
        }
        losses = DBLoss()(predictions, batched_targets)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_detection_dataset_uses_centered_letterbox(self):
        class CaptureMapGenerator:
            def __call__(self, image_shape, polygons, ignore_tags):
                self.image_shape = image_shape
                self.polygons = [polygon.copy() for polygon in polygons]
                height, width = image_shape
                return {"map": np.zeros((height, width), dtype=np.float32)}

        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "wide.jpg", shape=(20, 40, 3))
            annotation = [
                {
                    "transcription": "text",
                    "points": [[4, 2], [36, 2], [36, 18], [4, 18]],
                }
            ]
            labels = Path(directory) / "det.txt"
            labels.write_text(
                f"wide.jpg\t{json.dumps(annotation)}\n",
                encoding="utf-8",
            )
            map_generator = CaptureMapGenerator()
            dataset = DetectionDataset(
                labels,
                data_dir=directory,
                image_shape=(3, 40, 40),
                map_generator=map_generator,
            )
            image, _ = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 40, 40))
        self.assertEqual(map_generator.image_shape, (40, 40))
        np.testing.assert_allclose(
            map_generator.polygons[0],
            np.asarray([[4, 12], [36, 12], [36, 28], [4, 28]], dtype=np.float32),
        )
        expected_padding = (114 / 255.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(image[0, 0, 0]), expected_padding, places=5)

    def test_paddle_detection_augmentation_transforms_polygons_with_image(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        polygon = np.asarray(
            [[4, 2], [36, 2], [36, 18], [4, 18]],
            dtype=np.float32,
        )
        augmenter = PaddleDetectionAugmentation(
            (3, 20, 40),
            flip_probability=1.0,
            rotate_range=(0, 0),
            scale_range=(1, 1),
            crop_tries=0,
        )

        augmented = augmenter(
            DetectionSample(image, [polygon], [False], ["text"])
        )

        self.assertEqual(augmented.image.shape, (20, 40, 3))
        np.testing.assert_allclose(
            augmented.polygons[0],
            [[35, 2], [3, 2], [3, 18], [35, 18]],
            atol=1.0e-5,
        )

    def test_detection_dataset_paddle_preset_produces_valid_db_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "first.jpg", shape=(32, 64, 3))
            self.write_image(directory, "second.jpg", shape=(32, 64, 3))
            annotation = [
                {
                    "transcription": "text",
                    "points": [[8, 8], [48, 8], [48, 24], [8, 24]],
                }
            ]
            labels = Path(directory) / "det.txt"
            labels.write_text(
                "first.jpg\t" + json.dumps(annotation) + "\n"
                "second.jpg\t" + json.dumps(annotation) + "\n",
                encoding="utf-8",
            )
            random_state = np.random.get_state()
            np.random.seed(5)
            try:
                dataset = DetectionDataset(
                    labels,
                    data_dir=directory,
                    image_shape=(3, 32, 64),
                    augmentation="paddle",
                )
                outputs = [dataset[index] for index in range(2)]
            finally:
                np.random.set_state(random_state)

        for image, targets in outputs:
            self.assertEqual(tuple(image.shape), (3, 32, 64))
            self.assertTrue(torch.isfinite(image).all())
            self.assertEqual(tuple(targets["shrink_map"].shape), (32, 64))

    def test_detection_metric_targets_preserve_variable_polygons(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "first.jpg")
            self.write_image(directory, "second.jpg")
            first = [
                {
                    "transcription": "text",
                    "points": [[8, 8], [48, 8], [48, 24], [8, 24]],
                }
            ]
            second = first + [
                {
                    "transcription": "###",
                    "points": [[2, 2], [6, 2], [6, 6], [2, 6]],
                }
            ]
            labels = Path(directory) / "det.txt"
            labels.write_text(
                "first.jpg\t" + json.dumps(first) + "\n"
                "second.jpg\t" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            dataset = DetectionDataset(
                labels,
                data_dir=directory,
                image_shape=(3, 32, 64),
                return_polygons=True,
            )
            images, targets = next(
                iter(DataLoader(dataset, batch_size=2, collate_fn=detection_collate))
            )

        self.assertEqual(tuple(images.shape), (2, 3, 32, 64))
        self.assertEqual(tuple(targets["shrink_map"].shape), (2, 32, 64))
        self.assertEqual([len(item) for item in targets["polygons"]], [1, 2])
        self.assertEqual(targets["ignore_tags"][1].tolist(), [False, True])
        np.testing.assert_allclose(targets["shape"][0], [32, 64, 1, 1])

    def test_recognition_dataset_and_ctc_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "rec.jpg", shape=(20, 60, 3))
            dictionary = Path(directory) / "dict.txt"
            dictionary.write_text("a\nb\nc\n", encoding="utf-8")
            labels = Path(directory) / "rec.txt"
            labels.write_text("rec.jpg\tabc\n", encoding="utf-8")
            dataset = RecognitionDataset(
                labels,
                dictionary,
                data_dir=directory,
                image_shape=(3, 16, 64),
                max_text_length=8,
            )
            image, targets = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 16, 64))
        self.assertEqual(int(targets["target_lengths"]), 3)
        logits = torch.randn(1, 8, 4, requires_grad=True)
        batched_targets = {
            "targets": targets["targets"].unsqueeze(0),
            "target_lengths": targets["target_lengths"].unsqueeze(0),
        }
        losses = CTCLoss()(logits, batched_targets)
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_recognition_multi_head_targets_match_nrtr_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "rec.jpg", shape=(20, 60, 3))
            dictionary = Path(directory) / "dict.txt"
            dictionary.write_text("a\nb\nc\n", encoding="utf-8")
            labels = Path(directory) / "rec.txt"
            labels.write_text("rec.jpg\tabc\n", encoding="utf-8")
            dataset = RecognitionDataset(
                labels,
                dictionary,
                data_dir=directory,
                image_shape=(3, 16, 64),
                max_text_length=8,
                multi_head=True,
            )
            _, targets = dataset[0]

        self.assertEqual(targets["targets"].tolist(), [1, 2, 3, 0, 0, 0, 0, 0])
        self.assertEqual(
            targets["gtc_targets"].tolist(),
            [2, 4, 5, 6, 3, 0, 0, 0],
        )
        self.assertEqual(int(targets["target_lengths"]), 3)

    def test_paddle_recognition_concat_and_augmentation(self):
        augmenter = PaddleRecognitionAugmentation(
            (3, 16, 64),
            max_text_length=8,
            concat_probability=1.0,
            tia_probability=0.0,
            crop_probability=0.0,
            reverse_probability=0.0,
            noise_probability=0.0,
            jitter_probability=0.0,
            blur_probability=0.0,
            hsv_probability=0.0,
        )
        image = np.full((20, 20, 3), 32, dtype=np.uint8)
        external = np.full((10, 10, 3), 224, dtype=np.uint8)

        concatenated, text = augmenter.concatenate(image, "a", [(external, "b")])
        augmented = augmenter(concatenated)

        self.assertEqual(text, "ab")
        self.assertEqual(augmented.shape, (16, 32, 3))
        self.assertLess(float(augmented[:, :16].mean()), 64)
        self.assertGreater(float(augmented[:, 16:].mean()), 192)

    def test_recognition_dataset_paddle_preset_reencodes_augmented_label(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "first.jpg", shape=(24, 30, 3))
            self.write_image(directory, "second.jpg", shape=(24, 30, 3))
            dictionary = Path(directory) / "dict.txt"
            dictionary.write_text("a\nb\n", encoding="utf-8")
            labels = Path(directory) / "rec.txt"
            labels.write_text("first.jpg\ta\nsecond.jpg\tb\n", encoding="utf-8")
            dataset = RecognitionDataset(
                labels,
                dictionary,
                data_dir=directory,
                image_shape=(3, 16, 64),
                max_text_length=4,
                augmentation="paddle",
            )
            outputs = [dataset[index] for index in range(2)]

        for image, targets in outputs:
            self.assertEqual(tuple(image.shape), (3, 16, 64))
            self.assertTrue(torch.isfinite(image).all())
            length = int(targets["target_lengths"])
            self.assertGreaterEqual(length, 1)
            self.assertLessEqual(length, 4)

    def test_recognition_multi_scale_sampler_emits_real_height_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_image(directory, "rec.jpg", shape=(20, 40, 3))
            dictionary = Path(directory) / "dict.txt"
            dictionary.write_text("a\n", encoding="utf-8")
            labels = Path(directory) / "rec.txt"
            labels.write_text("rec.jpg\ta\n" * 12, encoding="utf-8")
            dataset = RecognitionDataset(
                labels,
                dictionary,
                data_dir=directory,
                image_shape=(3, 32, 64),
                max_text_length=4,
            )
            sampler = RecognitionMultiScaleBatchSampler(
                dataset,
                width=64,
                heights=[16, 32, 48],
                base_height=32,
                base_batch_size=2,
                fix_batch_size=False,
                drop_last=True,
                seed=7,
            )
            batches = list(DataLoader(dataset, batch_sampler=sampler))

        observed = {(images.shape[0], images.shape[2]) for images, _ in batches}
        self.assertTrue({(4, 16), (2, 32), (1, 48)}.issubset(observed))
        self.assertTrue(all(images.shape[3] == 64 for images, _ in batches))


if __name__ == "__main__":
    unittest.main()

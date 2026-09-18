import unittest

import cv2
import numpy as np

from axera.eval_board_det import (
    DEFAULT_TARGET,
    det_geometry,
    detect_boxes,
    draw_boxes,
    map_boxes_to_source,
    preprocess,
    unclip_box,
)


class EvalBoardDetPreprocessTest(unittest.TestCase):
    def test_defaults_to_letterbox_at_default_deployment_shape(self):
        image = np.full((20, 40, 3), 127, dtype=np.uint8)

        output = preprocess(image)

        self.assertEqual(output.shape, (1, 3, DEFAULT_TARGET, DEFAULT_TARGET))
        padding = (114.0 / 255.0 - 0.485) / 0.229
        resized = (127.0 / 255.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(output[0, 0, 0, 0]), padding, places=5)
        self.assertAlmostEqual(float(output[0, 0, 184, 0]), resized, places=5)

    def test_640_letterbox_remains_compatible(self):
        image = np.full((20, 40, 3), 127, dtype=np.uint8)

        output = preprocess(image, "letterbox", 640, 640)

        padding = (114.0 / 255.0 - 0.485) / 0.229
        resized = (127.0 / 255.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(output[0, 0, 0, 0]), padding, places=5)
        self.assertAlmostEqual(float(output[0, 0, 320, 0]), resized, places=5)

    def test_rejects_unknown_preprocessing(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "must be 'paddle' or 'letterbox'"):
            preprocess(image, "unknown")


class EvalBoardDetGeometryTest(unittest.TestCase):
    def test_paddle_resize_scales_axes_independently(self):
        geometry = det_geometry(320, 640, "paddle", 640, 640)

        self.assertEqual((geometry.resize_w, geometry.resize_h), (640, 640))
        self.assertEqual((geometry.left, geometry.top), (0, 0))
        self.assertAlmostEqual(geometry.scale_x, 640 / 640)
        self.assertAlmostEqual(geometry.scale_y, 320 / 640)

    def test_letterbox_offsets_centered_padding(self):
        geometry = det_geometry(320, 640, "letterbox", 640, 640)

        self.assertEqual((geometry.resize_w, geometry.resize_h), (640, 320))
        self.assertEqual((geometry.left, geometry.top), (0, 160))
        self.assertAlmostEqual(geometry.scale_x, 1.0)
        self.assertAlmostEqual(geometry.scale_y, 1.0)

    def test_geometry_inverts_the_preprocessing_padding(self):
        """A source pixel must survive preprocess -> box -> map back."""
        image = np.full((240, 400, 3), 127, dtype=np.uint8)
        geometry = det_geometry(240, 400, "letterbox", 640, 640)
        self.assertEqual(preprocess(image, "letterbox", 640, 640).shape, (1, 3, 640, 640))

        # The letterboxed content of source (0, 0) sits at (left, top); a source
        # span of 10 px is 10 / scale in target coordinates.
        self.assertEqual((geometry.left, geometry.top), (0, 128))
        span_x = 10 / geometry.scale_x
        span_y = 10 / geometry.scale_y
        box = np.array(
            [
                [geometry.left, geometry.top],
                [geometry.left + span_x, geometry.top],
                [geometry.left + span_x, geometry.top + span_y],
                [geometry.left, geometry.top + span_y],
            ],
            dtype=np.float32,
        )
        mapped = map_boxes_to_source([box], geometry, 240, 400)[0]

        self.assertTrue(
            np.array_equal(mapped, np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
        )

    def test_paddle_mapping_respects_non_square_source(self):
        geometry = det_geometry(320, 640, "paddle", 640, 640)
        box = np.array([[0, 0], [640, 0], [640, 640], [0, 640]], dtype=np.float32)

        mapped = map_boxes_to_source([box], geometry, 320, 640)[0]

        self.assertTrue(np.array_equal(mapped, np.array([[0, 0], [640, 0], [640, 320], [0, 320]])))


class EvalBoardDetBoxesTest(unittest.TestCase):
    def shrink_map_with_two_boxes(self):
        shrink = np.zeros((DEFAULT_TARGET, DEFAULT_TARGET), dtype=np.float32)
        shrink[100:140, 100:300] = 0.9
        shrink[400:430, 200:260] = 0.9
        return shrink

    def test_detects_two_boxes_with_scores(self):
        boxes, scores = detect_boxes(self.shrink_map_with_two_boxes())

        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(scores), 2)
        for box in boxes:
            self.assertEqual(box.shape, (4, 2))
        self.assertTrue(all(score > 0.8 for score in scores))

    def test_unclip_expands_every_side(self):
        """pyclipper-free unclip == rectangle expanded by area*ratio/perimeter."""
        box = np.array(
            [[100, 100], [300, 100], [300, 140], [100, 140]], dtype=np.float32
        )

        (center_x, center_y), (width, height), _ = cv2.minAreaRect(unclip_box(box, 1.4))

        distance = 200 * 40 * 1.4 / (2.0 * (200 + 40))
        self.assertAlmostEqual(center_x, 200, delta=0.5)
        self.assertAlmostEqual(center_y, 120, delta=0.5)
        self.assertAlmostEqual(width, 200 + 2 * distance, delta=0.5)
        self.assertAlmostEqual(height, 40 + 2 * distance, delta=0.5)

    def test_unclip_grows_detected_boxes_beyond_the_blob(self):
        boxes, _ = detect_boxes(self.shrink_map_with_two_boxes())

        areas = sorted(float(cv2.contourArea(box.astype(np.float32))) for box in boxes)
        self.assertGreater(areas[0], 60 * 30)  # small blob 60x30
        self.assertGreater(areas[1], 200 * 40)  # large blob 200x40

    def test_box_threshold_filters_low_scores(self):
        boxes, scores = detect_boxes(self.shrink_map_with_two_boxes(), thresh=0.5,
                                     box_thresh=0.95)

        self.assertEqual(boxes, [])
        self.assertEqual(scores, [])

    def test_threshold_above_map_returns_nothing(self):
        boxes, scores = detect_boxes(self.shrink_map_with_two_boxes(), thresh=0.99)

        self.assertEqual((boxes, scores), ([], []))

    def test_small_blob_is_dropped_by_min_side(self):
        shrink = np.zeros((DEFAULT_TARGET, DEFAULT_TARGET), dtype=np.float32)
        shrink[10:12, 10:12] = 0.9

        boxes, _ = detect_boxes(shrink)

        self.assertEqual(boxes, [])

    def test_boxes_map_back_to_source_coordinates(self):
        image_shape = (320, 640)
        geometry = det_geometry(*image_shape, "paddle", DEFAULT_TARGET, DEFAULT_TARGET)

        boxes, _ = detect_boxes(self.shrink_map_with_two_boxes())
        mapped = map_boxes_to_source(boxes, geometry, *image_shape)

        lefts = sorted(int(box[:, 0].min()) for box in mapped)
        expected_lefts = sorted(
            int(round(value * image_shape[1] / DEFAULT_TARGET))
            for value in (100, 200)
        )
        self.assertAlmostEqual(lefts[0], expected_lefts[0], delta=30)
        self.assertAlmostEqual(lefts[1], expected_lefts[1], delta=30)
        for box in mapped:
            self.assertGreaterEqual(int(box.min()), 0)
            self.assertLessEqual(int(box[:, 0].max()), image_shape[1])
            self.assertLessEqual(int(box[:, 1].max()), image_shape[0])

    def test_draw_boxes_marks_the_image_without_touching_the_input(self):
        shrink = self.shrink_map_with_two_boxes()
        boxes, scores = detect_boxes(shrink)
        image = np.full((240, 400, 3), 255, dtype=np.uint8)

        overlay = draw_boxes(image, boxes, scores, thickness=2, show_score=True)

        self.assertEqual(overlay.shape, image.shape)
        self.assertTrue(np.array_equal(image, np.full((240, 400, 3), 255, dtype=np.uint8)))
        self.assertFalse(np.array_equal(overlay, image))
        self.assertTrue(np.any(overlay[:, :, 2] != 255))  # red channel


if __name__ == "__main__":
    unittest.main()

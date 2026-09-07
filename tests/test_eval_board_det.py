import unittest

import cv2
import numpy as np

from axera.eval_board_det import preprocess


class EvalBoardDetPreprocessTest(unittest.TestCase):
    def test_defaults_to_paddle_fixed_shape_resize(self):
        image = np.full((20, 40, 3), 127, dtype=np.uint8)

        output = preprocess(image)

        self.assertEqual(output.shape, (1, 3, 640, 640))
        expected = (127.0 / 255.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(output[0, 0, 0, 0]), expected, places=5)
        self.assertAlmostEqual(float(output[0, 0, -1, -1]), expected, places=5)

    def test_letterbox_remains_explicit_compatibility_mode(self):
        image = np.full((20, 40, 3), 127, dtype=np.uint8)

        output = preprocess(image, "letterbox")

        padding = (114.0 / 255.0 - 0.485) / 0.229
        resized = (127.0 / 255.0 - 0.485) / 0.229
        self.assertAlmostEqual(float(output[0, 0, 0, 0]), padding, places=5)
        self.assertAlmostEqual(float(output[0, 0, 320, 0]), resized, places=5)

    def test_rejects_unknown_preprocessing(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "must be 'paddle' or 'letterbox'"):
            preprocess(image, "unknown")


if __name__ == "__main__":
    unittest.main()

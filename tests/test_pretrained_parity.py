import unittest

import numpy as np

from pytorchocr.diagnostics import compare_gradient_maps, tensor_difference


class PretrainedParityTest(unittest.TestCase):
    def test_tensor_difference_reports_shape_and_error(self):
        result = tensor_difference(
            np.array([[1.0, 2.0]], dtype=np.float32),
            np.array([[1.5, 1.0]], dtype=np.float32),
        )

        self.assertEqual(result["shape"], [1, 2])
        self.assertEqual(result["mae"], 0.75)
        self.assertEqual(result["max_abs"], 1.0)
        self.assertTrue(result["finite"])

    def test_gradient_comparison_supports_linear_transpose(self):
        reference = {
            "backbone.linear.weight": np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
            ),
            "head.ctc_head.bias": np.array([1.0, 2.0], dtype=np.float32),
        }
        candidate = {
            "backbone.linear.weight": reference["backbone.linear.weight"].T,
            "head.ctc_head.bias": np.array([1.5, 2.0], dtype=np.float32),
        }

        result = compare_gradient_maps(
            reference,
            candidate,
            owners=("backbone.", "head.ctc_head."),
            transpose_reference=("backbone.linear.weight",),
        )

        self.assertEqual(result["overall"]["tensor_count"], 2)
        self.assertEqual(result["owners"]["backbone"]["max_abs"], 0.0)
        self.assertEqual(result["owners"]["head.ctc_head"]["max_abs"], 0.5)
        self.assertEqual(result["missing_candidate"], [])
        self.assertEqual(result["shape_mismatches"], [])
        self.assertEqual(
            result["worst_tensors"][0]["name"],
            "head.ctc_head.bias",
        )


if __name__ == "__main__":
    unittest.main()

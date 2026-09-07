import unittest

from pytorchocr.diagnostics import resolve_onnx_input_shape, static_onnx_input_shape


class FakeInput:
    def __init__(self, shape):
        self.shape = shape


class FakeSession:
    def __init__(self, shape):
        self.input = FakeInput(shape)

    def get_inputs(self):
        return [self.input]


class OnnxMetricEvaluationTest(unittest.TestCase):
    def test_accepts_static_ocr_input_shape(self):
        self.assertEqual(
            static_onnx_input_shape(FakeSession([1, 3, 48, 320])),
            [1, 3, 48, 320],
        )

    def test_rejects_dynamic_spatial_shape(self):
        with self.assertRaisesRegex(ValueError, "static channel/height/width"):
            static_onnx_input_shape(FakeSession([1, 3, "height", 320]))

    def test_resolves_dynamic_height_with_explicit_shape(self):
        self.assertEqual(
            resolve_onnx_input_shape(
                FakeSession(["batch", 3, "16*s1", 320]),
                [3, 32, 320],
            ),
            ["batch", 3, 32, 320],
        )

    def test_rejects_explicit_shape_against_static_axis(self):
        with self.assertRaisesRegex(ValueError, "axis 2"):
            resolve_onnx_input_shape(
                FakeSession([1, 3, 48, 320]),
                [3, 32, 320],
            )


if __name__ == "__main__":
    unittest.main()

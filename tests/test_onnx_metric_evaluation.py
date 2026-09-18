import argparse
import sys
import unittest
from unittest.mock import patch

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
    def test_quantonnx_parser_defaults_to_letterbox(self):
        from tools import evaluate_onnx

        argv = [
            "evaluate_onnx.py",
            "--task", "det",
            "--model-config", "model.yml",
            "--onnx", "model.onnx",
            "--label-file", "val.txt",
        ]
        with patch.object(sys, "argv", argv):
            args = evaluate_onnx.parse_args()
        self.assertEqual(args.det_preprocess, "letterbox")

    def test_standalone_onnx_evaluation_selects_letterbox(self):
        from tools import eval as eval_tool

        args = argparse.Namespace(
            task="det", model_config="model.yml", label_file="val.txt",
            data_dir=".", batch_size=1, workers=0, image_shape=None,
            det_preprocess=None, onnx="model.onnx", pt=None, weights=None,
            stage="converted", ort_optimize=False, samples=None,
        )
        with patch.object(eval_tool, "load_ocr_config", return_value={"Global": {}}), \
             patch.object(eval_tool, "run_onnx_session", return_value=object()), \
             patch.object(eval_tool, "resolve_onnx_input_shape", return_value=[1, 3, 736, 736]), \
             patch.object(eval_tool, "evaluate_onnx", return_value={"hmean": 0.0}):
            eval_tool.main(args)
        self.assertEqual(args.det_preprocess, "letterbox")

    def test_standalone_pt_evaluation_selects_official_resize(self):
        from tools import eval as eval_tool

        args = argparse.Namespace(
            task="det", model_config="model.yml", label_file="val.txt",
            data_dir=".", batch_size=1, workers=0, image_shape=None,
            det_preprocess=None, onnx=None, pt="model.pt", weights=None,
            stage="converted", ort_optimize=False, samples=None,
        )
        with patch.object(eval_tool, "load_ocr_config", return_value={"Global": {}}), \
             patch.object(eval_tool, "load_pt2e", return_value=(object(), {})), \
             patch.object(eval_tool, "evaluate_pt2e", return_value={"hmean": 0.0}):
            eval_tool.main(args)
        self.assertEqual(args.det_preprocess, "official")

    def test_combined_detection_alignment_requires_explicit_preprocess(self):
        from tools import eval as eval_tool

        args = argparse.Namespace(
            task="det", model_config="model.yml", label_file="val.txt",
            data_dir=".", batch_size=1, workers=0, image_shape=None,
            det_preprocess=None, onnx="model.onnx", pt="model.pt", weights=None,
            stage="converted", ort_optimize=False, samples=None,
        )
        with patch.object(eval_tool, "load_ocr_config", return_value={"Global": {}}), \
             patch.object(eval_tool, "run_onnx_session", return_value=object()), \
             patch.object(eval_tool, "resolve_onnx_input_shape", return_value=[1, 3, 736, 736]), \
             self.assertRaisesRegex(ValueError, "requires explicit"):
            eval_tool.main(args)

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

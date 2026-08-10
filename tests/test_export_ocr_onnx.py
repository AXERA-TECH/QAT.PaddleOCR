import unittest
from unittest import mock

import torch

from pytorchocr.quantization.onnx_export import RecTrainingDeploymentProjection

from tools.export_ocr_onnx import (
    ModelSpec,
    build_float_export_model,
    build_parser,
    build_quant_export_model,
    make_example_inputs,
)


class ExportOcrOnnxTest(unittest.TestCase):
    def test_float_matrix_defaults_to_all_graph_and_reparameterization_variants(self):
        args = build_parser().parse_args(["float-matrix"])

        self.assertEqual(args.graphs, ("training", "inference"))
        self.assertEqual(
            args.reparameterizations,
            ("reparameterized", "non_reparameterized"),
        )

    def test_training_command_accepts_inference_quantonnx_graph(self):
        args = build_parser().parse_args(
            [
                "training",
                "--models",
                "ppocrv5_mobile_rec",
                "--format",
                "quantonnx",
                "--quant-graph",
                "inference",
            ]
        )

        self.assertEqual(args.quant_graph, "inference")

    def test_training_quantonnx_graph_remains_the_default(self):
        args = build_parser().parse_args(["training"])

        self.assertEqual(args.quant_graph, "training")

    def test_inference_graph_builds_deployment_only_task_model(self):
        for task, output_name in (("det", "maps"), ("rec", "logits")):
            spec = ModelSpec(
                task=task,
                model_config=f"{task}.yml",
                weights=f"{task}.pth",
                qat_config=f"{task}.json",
                image_shape=(3, 48, 320),
            )
            deployment_model = object()
            with self.subTest(task=task), mock.patch(
                "tools.export_ocr_onnx.build_task_model",
                return_value=deployment_model,
            ) as build_model:
                model, output_names, max_text_length = build_quant_export_model(
                    task,
                    spec,
                    reparameterize=True,
                    graph_mode="inference",
                )

            self.assertIs(model, deployment_model)
            self.assertEqual(output_names, (output_name,))
            self.assertIsNone(max_text_length)
            build_model.assert_called_once_with(
                task,
                spec.model_config,
                weights_path=spec.weights,
                reparameterize=True,
                det_graph="inference",
                rec_graph="deploy",
            )

    def test_float_inference_graph_builds_deployment_only_task_model(self):
        class DeploymentModel:
            def eval(self):
                return self

        for task, output_name in (("det", "maps"), ("rec", "logits")):
            spec = ModelSpec(
                task=task,
                model_config=f"{task}.yml",
                weights=f"{task}.pth",
                qat_config=f"{task}.json",
                image_shape=(3, 48, 320),
            )
            deployment_model = DeploymentModel()
            with self.subTest(task=task), mock.patch(
                "tools.export_ocr_onnx.build_task_model",
                return_value=deployment_model,
            ) as build_model:
                model, output_names, max_text_length = build_float_export_model(
                    task,
                    spec,
                    reparameterize=False,
                    graph_mode="inference",
                )

            self.assertIs(model, deployment_model)
            self.assertEqual(output_names, (output_name,))
            self.assertIsNone(max_text_length)
            build_model.assert_called_once_with(
                task,
                spec.model_config,
                weights_path=spec.weights,
                reparameterize=False,
                det_graph="inference",
                rec_graph="deploy",
            )

    def test_recognition_inference_graph_has_no_auxiliary_target_input(self):
        spec = ModelSpec(
            task="rec",
            model_config="rec.yml",
            weights="rec.pth",
            qat_config="rec.json",
            image_shape=(3, 48, 320),
        )

        inputs, input_names = make_example_inputs(
            spec,
            batch_size=2,
            include_rec_targets=False,
        )

        self.assertEqual(len(inputs), 1)
        self.assertEqual(input_names, ("images",))
        self.assertEqual(tuple(inputs[0].shape), (2, 3, 48, 320))

    def test_training_checkpoint_projection_returns_only_ctc(self):
        class FullTrainingModel(torch.nn.Module):
            def forward(self, images, gtc_targets):
                return images + 1, gtc_targets.float(), images - 1

        projection = RecTrainingDeploymentProjection(
            FullTrainingModel(),
            max_text_length=25,
        )
        images = torch.randn(2, 3, 4, 5)

        output = projection(images)

        torch.testing.assert_close(output, images + 1)


if __name__ == "__main__":
    unittest.main()

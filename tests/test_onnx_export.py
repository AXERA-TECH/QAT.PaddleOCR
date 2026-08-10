import unittest
import tempfile

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from pytorchocr.quantization import build_qat_dynamic_shapes
from pytorchocr.quantization.onnx_export import (
    export_onnx,
    fold_constant_zero_point_casts,
    insert_identity_for_requantize_dq_q,
    name_dynamic_input_axes,
    remove_exact_redundant_dq_q,
    set_static_input_axes,
)
from pytorchocr.quantization.validation import validate_qdq_graph


def build_dq_q_model(second_scale):
    scale_1 = numpy_helper.from_array(
        np.asarray(0.25, dtype=np.float32),
        "scale_1",
    )
    scale_2 = numpy_helper.from_array(
        np.asarray(second_scale, dtype=np.float32),
        "scale_2",
    )
    zero_point_1 = numpy_helper.from_array(
        np.asarray(3, dtype=np.uint8),
        "zero_point_1",
    )
    zero_point_2 = numpy_helper.from_array(
        np.asarray(3, dtype=np.uint8),
        "zero_point_2",
    )
    nodes = [
        helper.make_node(
            "DequantizeLinear",
            ["quantized", "scale_1", "zero_point_1"],
            ["dequantized_1"],
            name="dq_1",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["dequantized_1", "scale_2", "zero_point_2"],
            ["requantized"],
            name="q_2",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["requantized", "scale_2", "zero_point_2"],
            ["output"],
            name="dq_2",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "dq_q_test",
        [helper.make_tensor_value_info("quantized", TensorProto.UINT8, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [scale_1, scale_2, zero_point_1, zero_point_2],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 21)],
        ir_version=10,
    )


class OnnxExportTest(unittest.TestCase):
    def test_exports_dynamic_recognition_height(self):
        class ScaleModel(torch.nn.Module):
            def forward(self, images):
                return images * 2.0

        images = torch.randn(1, 3, 48, 64)
        dynamic_shapes = build_qat_dynamic_shapes(
            images,
            dynamic_heights=[32, 48, 64],
        )
        with tempfile.TemporaryDirectory() as directory:
            model = export_onnx(
                ScaleModel(),
                (images,),
                f"{directory}/dynamic_height.onnx",
                ["output"],
                optimize=False,
                dynamic_shapes=dynamic_shapes,
                dynamic_axis_names={0: {2: "height"}},
            )

        input_dimensions = model.graph.input[0].type.tensor_type.shape.dim
        self.assertEqual(input_dimensions[0].dim_value, 1)
        self.assertEqual(input_dimensions[1].dim_value, 3)
        self.assertEqual(input_dimensions[2].dim_param, "height")
        self.assertEqual(input_dimensions[3].dim_value, 64)

    def test_specializes_dynamic_height_for_quantonnx(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["images"], ["output"])],
                "dynamic_input",
                [
                    helper.make_tensor_value_info(
                        "images",
                        TensorProto.FLOAT,
                        [1, 3, "height", 320],
                    )
                ],
                [
                    helper.make_tensor_value_info(
                        "output",
                        TensorProto.FLOAT,
                        [1, 3, "height", 320],
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )

        self.assertEqual(set_static_input_axes(model, {0: {2: 48}}), 1)
        dimension = model.graph.input[0].type.tensor_type.shape.dim[2]
        self.assertEqual(dimension.dim_value, 48)
        self.assertFalse(dimension.dim_param)

    def test_accepts_silu_with_boundary_qdq(self):
        initializers = [
            numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.asarray(3, dtype=np.uint8), "zero_point"),
        ]
        nodes = [
            helper.make_node(
                "DequantizeLinear",
                ["input", "scale", "zero_point"],
                ["dequantized"],
            ),
            helper.make_node("Sigmoid", ["dequantized"], ["gate"]),
            helper.make_node("Mul", ["dequantized", "gate"], ["silu"]),
            helper.make_node(
                "QuantizeLinear",
                ["silu", "scale", "zero_point"],
                ["output"],
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "silu_boundary_qdq",
            [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 4])],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )

        stats = validate_qdq_graph(model)
        self.assertEqual(
            stats["silu_domains"],
            {"total": 1, "quantized": 1, "internal_qdq": 0},
        )

    def test_rejects_silu_with_internal_qdq(self):
        initializers = [
            numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.asarray(3, dtype=np.uint8), "zero_point"),
        ]
        nodes = [
            helper.make_node(
                "DequantizeLinear",
                ["input", "scale", "zero_point"],
                ["dequantized"],
            ),
            helper.make_node("Sigmoid", ["dequantized"], ["gate"]),
            helper.make_node(
                "QuantizeLinear",
                ["gate", "scale", "zero_point"],
                ["quantized_gate"],
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized_gate", "scale", "zero_point"],
                ["dequantized_gate"],
            ),
            helper.make_node(
                "Mul",
                ["dequantized", "dequantized_gate"],
                ["silu"],
            ),
            helper.make_node(
                "QuantizeLinear",
                ["silu", "scale", "zero_point"],
                ["output"],
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "silu_internal_qdq",
            [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 4])],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )

        with self.assertRaisesRegex(RuntimeError, "SiLU patterns"):
            validate_qdq_graph(model)

    def test_folds_all_exact_constant_zero_point_casts(self):
        initializers = [
            numpy_helper.from_array(
                np.zeros(4, dtype=np.int64), "source_weight_zero_point"
            ),
            numpy_helper.from_array(
                np.asarray(3, dtype=np.int64), "source_activation_zero_point"
            ),
            numpy_helper.from_array(
                np.ones(4, dtype=np.float32), "weight_scale"
            ),
            numpy_helper.from_array(
                np.asarray(0.25, dtype=np.float32), "activation_scale"
            ),
            numpy_helper.from_array(
                np.ones(4, dtype=np.int8), "quantized_weight"
            ),
        ]
        nodes = [
            helper.make_node(
                "Cast",
                ["source_weight_zero_point"],
                ["weight_zero_point"],
                name="weight_zero_point_cast",
                to=TensorProto.INT8,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized_weight", "weight_scale", "weight_zero_point"],
                ["dequantized_weight"],
                name="weight_dq",
                axis=0,
            ),
            helper.make_node(
                "Cast",
                ["source_activation_zero_point"],
                ["activation_zero_point"],
                name="activation_zero_point_cast",
                to=TensorProto.UINT8,
            ),
            helper.make_node(
                "QuantizeLinear",
                ["input", "activation_scale", "activation_zero_point"],
                ["output"],
                name="activation_q",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "zero_point_casts",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [
                helper.make_tensor_value_info("output", TensorProto.UINT8, [1]),
                helper.make_tensor_value_info(
                    "dequantized_weight", TensorProto.FLOAT, [4]
                ),
            ],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )

        self.assertEqual(fold_constant_zero_point_casts(model), 2)
        self.assertFalse(any(node.op_type == "Cast" for node in model.graph.node))
        values = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
        }
        self.assertEqual(values["weight_zero_point"].dtype, np.int8)
        self.assertEqual(values["activation_zero_point"].dtype, np.uint8)
        self.assertNotIn("source_weight_zero_point", values)
        self.assertNotIn("source_activation_zero_point", values)
        onnx.checker.check_model(model, full_check=True)

    def test_preserves_non_zero_point_and_overflowing_casts(self):
        initializers = [
            numpy_helper.from_array(
                np.asarray(1, dtype=np.int64), "ordinary_constant"
            ),
            numpy_helper.from_array(
                np.asarray(128, dtype=np.int64), "overflow_zero_point"
            ),
            numpy_helper.from_array(
                np.asarray(0.25, dtype=np.float32), "scale"
            ),
            numpy_helper.from_array(
                np.asarray(1, dtype=np.int8), "quantized"
            ),
        ]
        nodes = [
            helper.make_node(
                "Cast",
                ["ordinary_constant"],
                ["ordinary_float"],
                name="ordinary_cast",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Add", ["input", "ordinary_float"], ["added"], name="add"
            ),
            helper.make_node(
                "Cast",
                ["overflow_zero_point"],
                ["zero_point"],
                name="overflow_cast",
                to=TensorProto.INT8,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized", "scale", "zero_point"],
                ["dequantized"],
                name="dq",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "preserved_casts",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [])],
            [
                helper.make_tensor_value_info("added", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("dequantized", TensorProto.FLOAT, []),
            ],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )

        self.assertEqual(fold_constant_zero_point_casts(model), 0)
        self.assertEqual(
            [node.name for node in model.graph.node if node.op_type == "Cast"],
            ["ordinary_cast", "overflow_cast"],
        )

    def test_removes_exact_redundant_dq_q(self):
        model = build_dq_q_model(second_scale=0.25)
        self.assertEqual(remove_exact_redundant_dq_q(model), 1)
        self.assertEqual([node.name for node in model.graph.node], ["dq_2"])
        self.assertEqual(model.graph.node[0].input[0], "quantized")
        onnx.checker.check_model(model, full_check=True)

    def test_preserves_different_scale(self):
        model = build_dq_q_model(second_scale=0.2501)
        self.assertEqual(remove_exact_redundant_dq_q(model), 0)
        self.assertEqual(len(model.graph.node), 3)
        onnx.checker.check_model(model, full_check=True)

    def test_inserts_identity_for_different_scale(self):
        model = build_dq_q_model(second_scale=0.2501)
        self.assertEqual(insert_identity_for_requantize_dq_q(model), 1)
        self.assertEqual(
            [node.op_type for node in model.graph.node],
            ["DequantizeLinear", "Identity", "QuantizeLinear", "DequantizeLinear"],
        )
        self.assertEqual(model.graph.node[1].input[0], "dequantized_1")
        self.assertEqual(model.graph.node[2].input[0], "dequantized_1_identity")
        onnx.checker.check_model(model, full_check=True)

    def test_rejects_conv_output_without_qdq(self):
        weight = numpy_helper.from_array(
            np.ones((1, 1, 1, 1), dtype=np.float32),
            "weight",
        )
        nodes = [
            helper.make_node("Conv", ["input", "weight"], ["conv"], name="conv"),
            helper.make_node("Relu", ["conv"], ["output"], name="relu"),
        ]
        graph = helper.make_graph(
            nodes,
            "missing_conv_qdq",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
            [weight],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        with self.assertRaisesRegex(RuntimeError, "Conv outputs are missing QDQ: conv"):
            validate_qdq_graph(model)

    def test_rejects_batch_normalization(self):
        initializers = [
            numpy_helper.from_array(np.ones(1, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.zeros(1, dtype=np.float32), "bias"),
            numpy_helper.from_array(np.zeros(1, dtype=np.float32), "mean"),
            numpy_helper.from_array(np.ones(1, dtype=np.float32), "variance"),
        ]
        nodes = [
            helper.make_node(
                "BatchNormalization",
                ["input", "scale", "bias", "mean", "variance"],
                ["output"],
                name="batch_norm",
            )
        ]
        graph = helper.make_graph(
            nodes,
            "batch_norm_test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        with self.assertRaisesRegex(RuntimeError, "contains BatchNormalization"):
            validate_qdq_graph(model)

    def test_rejects_hard_activation_without_qdq(self):
        nodes = [
            helper.make_node(
                "HardSigmoid",
                ["input"],
                ["output"],
                name="hard_sigmoid",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "missing_hard_activation_qdq",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "HardSigmoid patterns are not fully quantized: 0 / 1",
        ):
            validate_qdq_graph(model)

    def test_rejects_paddle_hsigmoid_qdq_before_slope_mul(self):
        initializers = [
            numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "scale"),
            numpy_helper.from_array(np.asarray(0, dtype=np.uint8), "zero_point"),
            numpy_helper.from_array(np.asarray(1.2, dtype=np.float32), "slope"),
            numpy_helper.from_array(np.asarray(3.0, dtype=np.float32), "offset"),
            numpy_helper.from_array(np.asarray(0.0, dtype=np.float32), "clip_min"),
            numpy_helper.from_array(np.asarray(6.0, dtype=np.float32), "clip_max"),
            numpy_helper.from_array(np.asarray(6.0, dtype=np.float32), "divisor"),
        ]
        nodes = [
            helper.make_node(
                "DequantizeLinear",
                ["input", "scale", "zero_point"],
                ["dequantized"],
            ),
            helper.make_node("Mul", ["dequantized", "slope"], ["scaled"]),
            helper.make_node("Add", ["scaled", "offset"], ["shifted"]),
            helper.make_node(
                "Clip",
                ["shifted", "clip_min", "clip_max"],
                ["clipped"],
            ),
            helper.make_node("Div", ["clipped", "divisor"], ["gate"]),
            helper.make_node(
                "QuantizeLinear",
                ["gate", "scale", "zero_point"],
                ["quantized_gate"],
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized_gate", "scale", "zero_point"],
                ["output"],
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "paddle_hsigmoid_wrong_boundary",
            [helper.make_tensor_value_info("input", TensorProto.UINT8, [1, 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "HardSigmoid patterns are not fully quantized: 0 / 1",
        ):
            validate_qdq_graph(model)


if __name__ == "__main__":
    unittest.main()

import unittest
import json
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import (
    disable_observer,
    enable_fake_quant,
    move_exported_model_to_eval,
)
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.nn import functional as F

from pytorchocr.quantization import (
    build_qat_dynamic_shapes,
    convert_prepared_model,
    initialize_weight_observers,
)
from pytorchocr.quantization import load_axera_quantizer
from pytorchocr.quantization import prepare_qat_model
from pytorchocr.quantization.onnx_export import prepared_activation_qparams


ROOT_DIR = Path(__file__).resolve().parents[1]
DET_CONFIG = ROOT_DIR / "configs/qat/ppocrv6_small_det_u8s8.json"
REC_U16S16_CONFIG = (
    ROOT_DIR / "configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json"
)


class AxeraVendorTest(unittest.TestCase):
    def test_dynamic_batch_is_shared_with_recognition_targets(self):
        images = torch.randn(64, 3, 48, 64)

        dynamic_shapes = build_qat_dynamic_shapes(
            images,
            dynamic_batch=True,
            dynamic_heights=[32, 48, 64],
            batch_aligned_inputs=1,
            max_batch=96,
        )

        self.assertEqual(len(dynamic_shapes), 2)
        self.assertIs(dynamic_shapes[0][0], dynamic_shapes[1][0])
        self.assertIn(2, dynamic_shapes[0])
        self.assertNotIn(2, dynamic_shapes[1])

    def test_prepared_graph_supports_discrete_dynamic_heights(self):
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, images):
                return self.conv(images)

        images = torch.randn(1, 3, 48, 64)
        dynamic_shapes = build_qat_dynamic_shapes(
            images,
            dynamic_heights=[32, 48, 64],
        )
        prepared, _ = prepare_qat_model(
            ConvModel().train(),
            (images,),
            load_axera_quantizer(DET_CONFIG),
            dynamic_shapes=dynamic_shapes,
        )

        for height in (32, 48, 64):
            output = prepared(torch.randn(1, 3, height, 64))
            self.assertEqual(tuple(output.shape), (1, 4, height, 64))

    def test_keeps_integer_embedding_indices_outside_qat_domains(self):
        class TokenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(16, 8)
                self.projection = nn.Linear(8, 4)

            def forward(self, token_ids):
                return self.projection(self.embedding(token_ids))

        token_ids = torch.tensor([[2, 4, 3]], dtype=torch.int64)
        prepared, _ = prepare_qat_model(
            TokenModel().train(),
            (token_ids,),
            load_axera_quantizer(DET_CONFIG),
        )
        converted = convert_prepared_model(prepared)

        output = converted(token_ids)
        self.assertEqual(tuple(output.shape), (1, 3, 4))
        self.assertTrue(torch.isfinite(output).all())
        for node in converted.graph.nodes:
            if node.target != torch.ops.quantized_decomposed.quantize_per_tensor.default:
                continue
            source = node.args[0]
            value = source.meta.get("val")
            if isinstance(value, torch.Tensor):
                self.assertTrue(value.dtype.is_floating_point)

    def test_initializes_only_per_channel_weight_observers(self):
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 3, padding=1)

            def forward(self, images):
                return self.conv(images)

        prepared, _ = prepare_qat_model(
            ConvModel().train(),
            (torch.empty(1, 3, 8, 8),),
            load_axera_quantizer(DET_CONFIG),
        )

        initialized = initialize_weight_observers(prepared)
        self.assertEqual(
            {item["weight"] for item in initialized},
            {"conv.weight", "conv.bias"},
        )
        self.assertTrue(all(item["channels"] == 4 for item in initialized))

        per_tensor_modules = {
            id(module): module
            for module in prepared.modules()
            if getattr(module, "qscheme", None)
            in (torch.per_tensor_affine, torch.per_tensor_symmetric)
            and hasattr(module, "scale")
        }.values()
        self.assertTrue(per_tensor_modules)
        for module in per_tensor_modules:
            self.assertTrue(torch.equal(module.scale, torch.ones_like(module.scale)))
            self.assertTrue(
                torch.equal(module.zero_point, torch.zeros_like(module.zero_point))
            )

    def test_dyadic_static_scalars_use_weight_qspec(self):
        class AffineModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor([0.618031]))
                self.bias = nn.Parameter(torch.tensor([0.187708]))
                self.register_buffer("fixed", torch.tensor([1.2]))

            def forward(self, inputs):
                return (inputs * self.scale + self.bias) * self.fixed

        inputs = torch.randn(1, 3, 4, 4)
        exported = torch.export.export_for_training(
            AffineModel(),
            (inputs,),
        ).module()
        quantizer = load_axera_quantizer(REC_U16S16_CONFIG)
        exported = quantizer.transform_for_annotation(exported)
        exported = quantizer.annotate(exported)

        parameter_qspecs = {}
        buffer_qspecs = {}
        for node in exported.graph.nodes:
            annotation = node.meta.get("quantization_annotation")
            if annotation is None:
                continue
            for input_node, qspec in annotation.input_qspec_map.items():
                if input_node.op != "get_attr":
                    continue
                target = str(input_node.target)
                if target in ("scale", "bias"):
                    parameter_qspecs[target] = qspec
                elif target == "fixed":
                    buffer_qspecs[target] = qspec

        self.assertEqual(set(parameter_qspecs), {"scale", "bias"})
        for qspec in parameter_qspecs.values():
            self.assertEqual(qspec.dtype, torch.int16)
            self.assertEqual(qspec.qscheme, torch.per_tensor_symmetric)
            self.assertIsNone(qspec.ch_axis)
        self.assertEqual(buffer_qspecs["fixed"].dtype, torch.int16)
        self.assertEqual(
            buffer_qspecs["fixed"].qscheme,
            torch.per_tensor_symmetric,
        )

        prepared, _ = prepare_qat_model(
            AffineModel(),
            (inputs,),
            load_axera_quantizer(REC_U16S16_CONFIG),
        )
        initialized = initialize_weight_observers(prepared)
        self.assertEqual(
            {item["weight"] for item in initialized},
            {"scale", "bias", "fixed"},
        )
        for item in initialized:
            observer = prepared.get_submodule(item["observer"])
            self.assertEqual(observer.dtype, torch.int16)
            self.assertEqual(observer.qscheme, torch.per_tensor_symmetric)
            self.assertFalse(
                torch.equal(observer.scale, torch.ones_like(observer.scale))
            )
            self.assertTrue(
                torch.equal(
                    observer.zero_point,
                    torch.zeros_like(observer.zero_point),
                )
            )
        activation_qparams = prepared_activation_qparams(prepared)
        self.assertTrue(activation_qparams)
        self.assertTrue(all(record["scale_is_one"] for record in activation_qparams))
        self.assertTrue(
            all(record["zero_point_is_zero"] for record in activation_qparams)
        )
        uint16_fake_quant = [
            module
            for module in prepared.modules()
            if getattr(module, "dtype", None) == torch.uint16
            and hasattr(module, "fake_quant_enabled")
        ]
        self.assertTrue(uint16_fake_quant)
        self.assertTrue(
            all(type(module) is FakeQuantize for module in uint16_fake_quant)
        )

        move_exported_model_to_eval(prepared)
        prepared.apply(disable_observer)
        prepared.apply(enable_fake_quant)
        converted = convert_prepared_model(prepared)
        with torch.no_grad():
            prepared_output = prepared(inputs)
            converted_output = converted(inputs)
        torch.testing.assert_close(
            prepared_output,
            converted_output,
            rtol=0,
            atol=0,
        )

    def test_s16_domain_dyadic_scalars_use_s16_qspec(self):
        """Scalars on S16-domain Mul/Add nodes must be int16, not global S8.

        The AX650 TENG pixel-repeat packing requires (p.n * p.ksize) % 256 == 0;
        for the W*C=240 downsampling layout an int8 scalar (1920) fails while
        an int16 scalar (3840) passes.
        """

        class AffineModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor([1.0]))
                self.bias = nn.Parameter(torch.tensor([0.1]))

            def forward(self, inputs):
                return inputs * self.scale + self.bias

        inputs = torch.randn(1, 3, 4, 4)
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads((ROOT_DIR / "configs/qat/base_u8s8.json").read_text())
            config["regional_configs"] = [
                {
                    "module_names": ["mul"],
                    "module_type": "mul",
                    "module_config": {
                        "is_symmetric": True,
                        "output_is_symmetric": True,
                        "input": {
                            "dtype": "S16",
                            "qmin": -32767,
                            "qmax": 32767,
                        },
                        "output": {
                            "dtype": "S16",
                            "qmin": -32767,
                            "qmax": 32767,
                        },
                    },
                }
            ]
            config_path = Path(directory) / "u8s8_s16_mul.json"
            config_path.write_text(json.dumps(config))
            quantizer = load_axera_quantizer(str(config_path))
            prepared, _ = prepare_qat_model(
                AffineModel(),
                (inputs,),
                quantizer,
            )
        mul_node = next(
            node
            for node in prepared.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor
        )
        annotation = mul_node.meta["quantization_annotation"]
        scalar_qspecs = [
            qspec
            for input_node, qspec in annotation.input_qspec_map.items()
            if input_node.op == "get_attr"
        ]
        self.assertEqual(len(scalar_qspecs), 1)
        self.assertEqual(scalar_qspecs[0].dtype, torch.int16)
        self.assertEqual(scalar_qspecs[0].quant_min, -32767)
        self.assertEqual(scalar_qspecs[0].quant_max, 32767)

    def test_nonfinite_causal_mask_stays_float_until_softmax(self):
        class MaskedSoftmaxModel(nn.Module):
            def forward(self, scores):
                size = scores.shape[-1]
                mask = scores.new_zeros((size, size), dtype=torch.float32)
                mask_inf = torch.triu(
                    scores.new_full(
                        (size, size),
                        float("-inf"),
                        dtype=torch.float32,
                    ),
                    diagonal=1,
                )
                return torch.softmax(scores + mask + mask_inf, dim=-1)

        scores = torch.randn(2, 4, 4)
        exported = torch.export.export_for_training(
            MaskedSoftmaxModel(),
            (scores,),
        ).module()
        quantizer = load_axera_quantizer(REC_U16S16_CONFIG)
        annotated = quantizer.annotate(quantizer.transform_for_annotation(exported))
        masked_add = next(
            node
            for node in annotated.graph.nodes
            if node.target == torch.ops.aten.add.Tensor
            and any(
                input_node.target == torch.ops.aten.triu.default
                for input_node in node.all_input_nodes
            )
        )
        softmax = next(
            node
            for node in annotated.graph.nodes
            if node.target == torch.ops.aten.softmax.int
        )
        masked_add_annotation = masked_add.meta["quantization_annotation"]
        softmax_annotation = softmax.meta["quantization_annotation"]

        self.assertIsNone(masked_add_annotation.output_qspec)
        self.assertNotIn(masked_add, softmax_annotation.input_qspec_map)
        self.assertIsNotNone(softmax_annotation.output_qspec)

        prepared, _ = prepare_qat_model(
            MaskedSoftmaxModel(),
            (scores,),
            load_axera_quantizer(REC_U16S16_CONFIG),
        )
        output = prepared(scores)
        self.assertTrue(torch.isfinite(output).all())

    def test_regional_matmul_without_output_preserves_global_output_dtype(self):
        class MatMulModel(nn.Module):
            def forward(self, left, right):
                return torch.matmul(left, right)

        exported = torch.export.export_for_training(
            MatMulModel(),
            (torch.randn(1, 2, 3), torch.randn(1, 3, 4)),
        ).module()
        quantizer = load_axera_quantizer(DET_CONFIG)
        exported = quantizer.transform_for_annotation(exported)
        exported = quantizer.annotate(exported)
        matmul = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.matmul.default
        )
        annotation = matmul.meta["quantization_annotation"]
        self.assertEqual(
            {qspec.dtype for qspec in annotation.input_qspec_map.values()},
            {torch.int16},
        )
        self.assertEqual(annotation.output_qspec.dtype, torch.uint8)

    def test_regional_matmul_can_override_input_and_output_dtype(self):
        class MatMulModel(nn.Module):
            def forward(self, left, right):
                return torch.matmul(left, right)

        config = {
            "global_config": {
                "is_symmetric": False,
                "input": {"dtype": "U8", "qmin": 0, "qmax": 255},
                "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
            },
            "regional_configs": [
                {
                    "module_names": ["matmul"],
                    "module_type": "matmul",
                    "module_config": {
                        "is_symmetric": True,
                        "output_is_symmetric": False,
                        "input": {"dtype": "S8", "qmin": -127, "qmax": 127},
                        "output": {"dtype": "U8", "qmin": 0, "qmax": 255},
                    },
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            quantizer = load_axera_quantizer(config_file.name)
            exported = torch.export.export_for_training(
                MatMulModel(),
                (torch.randn(1, 2, 3), torch.randn(1, 3, 4)),
            ).module()
            exported = quantizer.transform_for_annotation(exported)
            exported = quantizer.annotate(exported)

        matmul = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.matmul.default
        )
        annotation = matmul.meta["quantization_annotation"]
        self.assertEqual(
            {qspec.dtype for qspec in annotation.input_qspec_map.values()},
            {torch.int8},
        )
        self.assertEqual(annotation.output_qspec.dtype, torch.uint8)
        self.assertEqual(annotation.output_qspec.qscheme, torch.per_tensor_affine)

    def test_prepares_concat_with_batchnorm_input(self):
        class BatchNormConcatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(4)
                self.conv2 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(4)

            def forward(self, images):
                direct_bn = self.bn1(self.conv1(images))
                activated = F.relu(self.bn2(self.conv2(images)))
                return torch.cat([direct_bn, activated], dim=1)

        images = torch.randn(1, 3, 8, 8)
        quantizer = load_axera_quantizer(DET_CONFIG)

        prepared, _ = prepare_qat_model(
            BatchNormConcatModel().train(),
            (images,),
            quantizer,
        )

        self.assertEqual(tuple(prepared(images).shape), (1, 8, 8, 8))

    def test_loads_vendored_quantizer(self):
        quantizer = load_axera_quantizer(DET_CONFIG)
        self.assertEqual(
            quantizer.quantizer.__class__.__module__,
            "pytorchocr.quantization.ax_quantizer",
        )

    def test_legacy_two_argument_loader_is_compatible(self):
        quantizer = load_axera_quantizer("/unused/QAT.axera", DET_CONFIG)
        self.assertEqual(
            quantizer.quantizer.__class__.__module__,
            "pytorchocr.quantization.ax_quantizer",
        )

    def test_annotates_same_padding_convolutions(self):
        class SamePaddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, 2, padding="same")
                self.relu1 = nn.ReLU(inplace=True)
                self.conv2 = nn.Conv2d(4, 4, 2, padding="same")
                self.relu2 = nn.ReLU(inplace=True)

            def forward(self, images):
                return self.relu2(self.conv2(self.relu1(self.conv1(images))))

        exported = torch.export.export_for_training(
            SamePaddingModel(),
            (torch.randn(1, 3, 8, 8),),
        ).module()
        quantizer = load_axera_quantizer(DET_CONFIG)
        exported = quantizer.transform_for_annotation(exported)
        exported = quantizer.annotate(exported)
        same_padding_convs = [
            node
            for node in exported.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.conv2d.padding
        ]
        self.assertEqual(len(same_padding_convs), 2)
        for conv in same_padding_convs:
            annotation = conv.meta.get("quantization_annotation")
            self.assertIsNotNone(annotation)
            self.assertTrue(annotation._annotated)
            relu = next(iter(conv.users))
            output_annotation = relu.meta.get("quantization_annotation")
            self.assertIsNotNone(output_annotation)
            self.assertIsNotNone(output_annotation.output_qspec)

    def test_annotates_scaled_hardsigmoid(self):
        class HardActivationModel(nn.Module):
            def forward(self, inputs):
                return F.hardsigmoid(1.2 * inputs)

        exported = torch.export.export_for_training(
            HardActivationModel(),
            (torch.randn(1, 3, 4, 4),),
        ).module()
        quantizer = load_axera_quantizer(DET_CONFIG)
        exported = quantizer.transform_for_annotation(exported)
        exported = quantizer.annotate(exported)
        scale_node = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.mul.Tensor
        )
        hard_sigmoid_node = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.hardsigmoid.default
        )

        scale_annotation = scale_node.meta.get("quantization_annotation")
        self.assertIsNotNone(scale_annotation)
        self.assertTrue(scale_annotation._annotated)
        self.assertIsNotNone(scale_annotation.input_qspec_map)
        scale_constant = scale_node.args[1]
        self.assertEqual(scale_constant.op, "get_attr")
        self.assertAlmostEqual(float(getattr(exported, scale_constant.target)), 1.2)
        self.assertIn(scale_constant, scale_annotation.input_qspec_map)
        self.assertIsNotNone(scale_annotation.output_qspec)

        hard_sigmoid_annotation = hard_sigmoid_node.meta.get(
            "quantization_annotation"
        )
        self.assertIsNotNone(hard_sigmoid_annotation)
        self.assertTrue(hard_sigmoid_annotation._annotated)
        self.assertIn(scale_node, hard_sigmoid_annotation.input_qspec_map)
        self.assertIsNotNone(hard_sigmoid_annotation.output_qspec)


if __name__ == "__main__":
    unittest.main()

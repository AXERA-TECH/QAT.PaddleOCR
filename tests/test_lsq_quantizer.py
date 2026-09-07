import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize
from torch.ao.quantization.quantize_pt2e import convert_pt2e

from pytorchocr.quantization import (
    disable_learn,
    enable_learn,
    initialize_weight_observers,
    is_lsq_config,
    load_axera_quantizer,
    prepare_qat_model,
)


class SmallConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x))


def _lsq_config(tmpdir, lsq=True):
    config = json.load(
        open("configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json")
    )
    if lsq:
        config["lsq"] = True
    else:
        config.pop("lsq", None)
    path = Path(tmpdir) / ("lsq.json" if lsq else "plain.json")
    path.write_text(json.dumps(config))
    return str(path)


class LSQQuantizerSelectionTest(unittest.TestCase):
    def test_default_config_uses_statistical_quantizer(self):
        quantizer = load_axera_quantizer(
            "configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json"
        )
        self.assertEqual(
            type(quantizer.quantizer).__module__.split(".")[-1],
            "ax_quantizer",
        )

    def test_lsq_config_uses_lsq_quantizer(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = _lsq_config(directory, lsq=True)
            self.assertTrue(is_lsq_config(config_path))
            quantizer = load_axera_quantizer(config_path)
            self.assertEqual(
                type(quantizer.quantizer).__module__.split(".")[-1],
                "ax_quantizer_lsq",
            )

    def test_lsq_flag_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(
                is_lsq_config(
                    "configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json"
                )
            )
            self.assertTrue(is_lsq_config(_lsq_config(directory, True)))
            self.assertFalse(is_lsq_config(_lsq_config(directory, False)))
            self.assertFalse(is_lsq_config("/nonexistent/path.json"))


class LSQFakeQuantizeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            self.quantizer = load_axera_quantizer(
                _lsq_config(directory, lsq=True)
            )
            self.model = SmallConv().eval()
            self.images = torch.randn(2, 3, 8, 8)
            self.prepared, _ = prepare_qat_model(
                self.model,
                (self.images,),
                self.quantizer,
            )
        self.modules = [
            self.prepared.get_submodule(node.target)
            for node in self.prepared.graph.nodes
            if node.op == "call_module"
        ]
        self.learnable = [
            m for m in self.modules if isinstance(m, _LearnableFakeQuantize)
        ]

    def test_learnable_modules_present_with_expected_shapes(self):
        self.assertTrue(self.learnable, "no _LearnableFakeQuantize modules")
        shapes = sorted(tuple(m.scale.shape) for m in self.learnable)
        # one per-channel weight scale [4] + two per-tensor activation scales [1]
        self.assertEqual(shapes, [(1,), (1,), (4,)])

    def test_enable_learn_toggles_learning(self):
        self.prepared.apply(enable_learn)
        for module in self.learnable:
            self.assertEqual(int(module.learning_enabled), 1)
            self.assertEqual(int(module.static_enabled), 0)
            self.assertTrue(module.scale.requires_grad)

    def test_disable_learn_turns_off_scale_gradients(self):
        self.prepared.apply(enable_learn)
        self.prepared.apply(disable_learn)
        for module in self.learnable:
            self.assertEqual(int(module.learning_enabled), 0)
            self.assertFalse(module.scale.requires_grad)

    def test_initialize_weight_observers_fills_weight_scale(self):
        # static pass must give the per-channel weight module a nonzero scale
        # derived from actual weight statistics instead of the default 1.0.
        initialized = initialize_weight_observers(self.prepared)
        weight_module = next(
            m for m in self.learnable if m.scale.numel() == 4
        )
        self.assertTrue(initialized, "no weight observers initialized")
        self.assertTrue(torch.isfinite(weight_module.scale).all())
        self.assertLess(float(weight_module.scale.max()), 1.0)

    def test_lsq_statistics_pass_enables_gradients(self):
        """After the YOLOv5-style two-phase init, scale gradients must flow."""
        self.prepared.apply(enable_learn)
        initialize_weight_observers(self.prepared)
        # activation statistics pass
        self.prepared.apply(torch.ao.quantization.disable_fake_quant)
        self.prepared.apply(torch.ao.quantization.enable_observer)
        with torch.no_grad():
            self.prepared(self.images)
        self.prepared.apply(torch.ao.quantization.enable_fake_quant)
        self.prepared.apply(torch.ao.quantization.disable_observer)

        output = self.prepared(self.images)
        self.assertGreater(int((output != 0).sum()), 0, "quantized output collapsed to zero")
        output.sum().backward()
        for module in self.learnable:
            self.assertIsNotNone(module.scale.grad)
            self.assertGreater(float(module.scale.grad.abs().sum()), 0.0)

    def test_convert_produces_standard_qdq(self):
        self.prepared.apply(enable_learn)
        initialize_weight_observers(self.prepared)
        converted = convert_pt2e(self.prepared)
        with torch.no_grad():
            output = converted(self.images)
        self.assertTrue(torch.isfinite(output).all())
        ops = [
            str(node.target)
            for node in converted.graph.nodes
            if node.op == "call_function"
        ]
        quant_ops = [op for op in ops if "quantize_per_tensor" in op]
        self.assertTrue(quant_ops, "converted graph has no quantize nodes")


class SmallPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)

    def forward(self, x):
        return F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1)


class AvgPoolRegionalAnnotationTest(unittest.TestCase):
    def test_avgpool_regional_input_qspec_applies(self):
        """Regional avgpool2d configs must not crash and must apply the input qspec.

        Regression test for the upstream bug where the regional branch passed a
        SourcePartition to _is_annotated; also guards the S16-downsample qspec
        contract (avg_pool input S16, producer output rewritten).
        """
        torch.manual_seed(0)
        model = SmallPool().eval()
        images = torch.randn(2, 3, 8, 8)
        exported = torch.export.export_for_training(model, (images,)).module()
        pool_name = next(
            node.name
            for node in exported.graph.nodes
            if node.op == "call_function"
            and node.target
            in (
                torch.ops.aten.avg_pool2d.default,
                torch.ops.aten.adaptive_avg_pool2d.default,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            config = json.load(open("configs/qat/base_u8s8.json"))
            config["lsq"] = True
            config["regional_configs"] = [
                {
                    "module_names": [pool_name],
                    "module_type": "avgpool2d",
                    "module_config": {
                        "is_symmetric": True,
                        "input": {
                            "dtype": "S16",
                            "qmin": -32767,
                            "qmax": 32767,
                        },
                    },
                }
            ]
            path = Path(directory) / "avgpool_regional.json"
            path.write_text(json.dumps(config))
            prepared, _ = prepare_qat_model(
                model,
                (images,),
                load_axera_quantizer(str(path)),
            )
        pool_node = next(
            node
            for node in prepared.graph.nodes
            if node.op == "call_function"
            and node.target
            in (
                torch.ops.aten.avg_pool2d.default,
                torch.ops.aten.adaptive_avg_pool2d.default,
            )
        )
        annotation = pool_node.meta["quantization_annotation"]
        # In the prepared graph the pool's args[0] is the inserted fake-quant
        # node; the input_qspec_map is keyed by the original producer nodes.
        input_spec = list(annotation.input_qspec_map.values())[0]
        self.assertEqual(input_spec.dtype, torch.int16)
        self.assertEqual(input_spec.quant_min, -32767)
        self.assertEqual(input_spec.quant_max, 32767)


if __name__ == "__main__":
    unittest.main()

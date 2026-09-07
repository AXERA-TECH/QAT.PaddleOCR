"""Contract tests for pytorchocr.quantization.folding (quantized-domain fold)."""

import json
from pathlib import Path

import pytest
import torch

from pytorchocr.quantization import (
    apply_folded_state,
    build_folded_state,
    checkpoint_qat_config,
    collect_quantized_weight_map,
    folded_eager_model,
)
from pytorchocr.quantization.folding import strip_state_prefix


class TestStripStatePrefix:
    def test_strips_prefix_and_drops_aux(self):
        state = {
            "model.backbone.conv.weight": torch.ones(1),
            "model.backbone.conv.bias": torch.zeros(1),
            "model.activation_post_process_1.scale": torch.ones(1),
            "_tensor_constant_3": torch.ones(1),
        }
        out = strip_state_prefix(state)
        assert set(out) == {"backbone.conv.weight", "backbone.conv.bias"}

    def test_keeps_unprefixed_keys(self):
        state = {"backbone.conv.weight": torch.ones(1)}
        assert strip_state_prefix(state) == state


class TestCheckpointQatConfig:
    def test_returns_metadata_path_when_no_copy(self):
        metadata = {"qat_config": "/tmp/whatever.json"}
        assert checkpoint_qat_config(metadata) == "/tmp/whatever.json"

    def test_prefers_archived_copy(self, tmp_path):
        archived = tmp_path / "ppocr_qat_s16.json"
        archived.write_text("{}")
        metadata = {
            "qat_config": "/tmp/live.json",
            "copied_configs": {
                "ppocrv5_mobile_rec.yml": str(tmp_path / "m.yml"),
                "ppocr_qat_s16.json": str(archived),
            },
        }
        assert checkpoint_qat_config(metadata) == str(archived)

    def test_ignores_non_json_copies(self, tmp_path):
        metadata = {
            "qat_config": "/tmp/live.json",
            "copied_configs": {"ppocrv5_mobile_rec.yml": str(tmp_path / "m.yml")},
        }
        assert checkpoint_qat_config(metadata) == "/tmp/live.json"


class TestCollectWeightMap:
    def test_empty_graph(self):
        class Dummy(torch.nn.Module):
            def forward(self, x):
                return x + 1

        gm = torch.fx.symbolic_trace(Dummy())
        assert collect_quantized_weight_map(gm) == {}


class TestApplyFoldedState:
    def test_applies_matching_keys(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 2, 1),
            torch.nn.Conv2d(2, 2, 1),
        )
        folded = {
            "0.weight": torch.randn_like(model[0].weight),
            "0.bias": torch.randn_like(model[0].bias),
            "1.weight": torch.randn_like(model[1].weight),
            "1.bias": torch.randn_like(model[1].bias),
            "0.bn.weight": torch.ones(2),  # not in model -> skipped
        }
        applied, skipped = apply_folded_state(model, folded)
        assert applied == 4
        assert skipped == 1
        assert torch.equal(model[0].weight, folded["0.weight"])
        assert torch.equal(model[1].bias, folded["1.bias"])


class TestFoldedEagerModel:
    @pytest.mark.slow
    def test_builds_folded_eager_model(self):
        checkpoint = (
            "runs/exp12a_ppocrv5_mobile_rec_v2qspec_noreparam/best.pt"
        )
        if not Path(checkpoint).exists():
            pytest.skip("exp12a checkpoint not present")
        folded = build_folded_state(
            torch.load(checkpoint, map_location="cpu", weights_only=False),
            "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
            "weights/ptocr_v5_mobile_rec_full.pth",
        )
        assert "backbone.conv1.conv.weight" in folded
        assert any("reparam_conv.weight" in k for k in folded)
        model, applied, skipped = folded_eager_model(
            checkpoint,
            "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
            "weights/ptocr_v5_mobile_rec_full.pth",
            folded,
        )
        assert applied > 0
        assert skipped == 126
        # conv1 folded weight matches
        assert torch.equal(
            model.backbone.conv1.conv.weight,
            folded["backbone.conv1.conv.weight"],
        )

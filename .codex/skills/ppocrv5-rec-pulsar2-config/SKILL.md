---
name: ppocrv5-rec-pulsar2-config
description: Generate and validate an AXERA Pulsar2 QuantONNX conversion config for PP-OCRv5 recognition models. Use when a PP-OCRv5 mobile-rec QAT smoke or trained QuantONNX must be compiled for AX650/NPU3, when exported ONNX node names changed, or when the two SVTR attention S16 layer overrides and retained requant boundary must be audited.
---

# PP-OCRv5 Rec Pulsar2 Config

Work from the repository root. Generate the config from the exact QuantONNX that will be
compiled; never copy ONNX node names from an older export.

## Generate

```bash
python .codex/skills/ppocrv5-rec-pulsar2-config/scripts/generate_ppocrv5_rec_pulsar2_config.py \
  --onnx /path/to/ppocrv5_mobile_rec_qdq.onnx \
  --output artifacts/pulsar2/ppocrv5_mobile_rec.json \
  --output-dir artifacts/pulsar2/ppocrv5_mobile_rec \
  --target-hardware AX650 \
  --npu-mode NPU3
```

The generator discovers both SVTR attention regions by topology and semantic QKV bias names. It
requires the following QuantONNX contract before writing anything:

- one FP32 NCHW input with default shape `[1, 3, 48, 320]` (deployment export defaults to
  `batchSize=1`; use `--expected-input-shape` when the ONNX batch differs for a specific task);
- one FP32 rank-3 CTC logits output with default class count `18385`;
- exactly two `MatMul -> Softmax -> MatMul` attention regions;
- QKV Linear output, scale path, first MatMul and Softmax remain S16;
- both attention MatMul inputs are S16/S16;
- second MatMul output returns to global U16;
- exactly seven SiLU patterns use boundary-only `DQ -> Sigmoid + Mul -> Q`, with no QDQ between
  Sigmoid and Mul;
- exactly one necessary DQ/Identity/Q requant remains by default.

Use `--expected-attention`, `--expected-classes`, `--expected-silu`, or `--expected-requant` only when
adapting another verified PP-OCRv5 rec variant. Do not weaken an expectation merely to make generation
pass.

The output has three `layer_configs` groups: QKV Add output S16, attention core input/output S16, and
second MatMul input S16 with its output left to QuantONNX U16. A sidecar `<config>.report.json` records
the ONNX/config SHA256, graph I/O, discovered node names, actual Q/DQ dtypes, and requant boundary.

## Validate

```bash
python .codex/skills/ppocrv5-rec-pulsar2-config/scripts/validate_ppocrv5_rec_pulsar2_config.py \
  --onnx /path/to/ppocrv5_mobile_rec_qdq.onnx \
  --config artifacts/pulsar2/ppocrv5_mobile_rec.json \
  --report artifacts/pulsar2/ppocrv5_mobile_rec.json.report.json
```

Run validation again after every ONNX export, optimization, qspec change, or Pulsar2 config edit.
Validation must fail on stale layer names, wrong S16/U16 boundaries, changed hashes, extra layer
overrides, changed requant count, or a decomposed SiLU with internal QDQ.

## Compile Contract

QuantONNX already carries Q/DQ parameters. Do not add PTQ rules or calibration-derived global
quantization to override them. The generated input processor is intentionally FP32/NCHW identity:
host code must perform PP-OCR equal-height resize, right zero padding, channel ordering, and `[-1, 1]`
normalization before inference.

Compile only after validation:

```bash
pulsar2 build \
  --input /path/to/ppocrv5_mobile_rec_qdq.onnx \
  --config artifacts/pulsar2/ppocrv5_mobile_rec.json \
  --output_dir artifacts/pulsar2/ppocrv5_mobile_rec
```

Keep the one Pool/Conv U16-to-U16 qparam boundary visible in the report. This skill does not remove it, edit the
ONNX, or change PT2E observer sharing. Any later export-stage removal requires a separate exact-qparam
proof and graph/accuracy validation.

Keep PyTorch `nn.Linear` as `aten.linear`. Pulsar2 frontend may lower rank-3 `MatMul + Add` patterns to
target-specific `FullyConnected`; do not replace `nn.Linear` with custom FC/Gemm before PT2E because
that would bypass the Linear QAT annotator.

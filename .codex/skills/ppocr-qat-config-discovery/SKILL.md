---
name: ppocr-qat-config-discovery
description: Discover Axera PT2E QAT regional nodes from the current PP-OCR detection or recognition export_for_training FX graph, then generate or check 8-bit, 16-bit, or mixed-region model QAT JSON. Use for any supported PP-OCR det/rec model when model code, weights, shape, Torch version, or qspec changes, and before QAT smoke or training.
---

# PP-OCR QAT Config Discovery

Work from the repository root. Use the exact model YAML, converted float weights, input shape,
reparameterization setting, and Torch environment intended for QAT. Never copy FX node names from an
older graph.

## Generate

The script reads `Architecture.model_type` to select det or rec automatically:

```bash
python .codex/skills/ppocr-qat-config-discovery/scripts/discover_ppocr_qat_config.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --base-config configs/qat/base_u16s16.json \
  --output configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json \
  --report artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_global_u16s16_qat_discovery_20260806.json \
  --image-shape 3 48 320 \
  --attention-dtype auto \
  --expected-attention 2
```

For det models, use the model's deployment shape and set `--expected-attention 0`. Models without a
supported Attention region retain the global config from the base JSON. Recognition models with
Attention are discovered from `nn_module_stack` and validated by QKV Linear, scale Mul, first MatMul,
Softmax, and second MatMul topology.

`--attention-dtype auto` is the safe default and follows the global activation width:

| Global activation/weight | Auto Attention | Attention output |
| --- | --- | --- |
| U8 / S8 | S8 | U8 |
| U16 / S16 | S16 | U16 |

The full 16-bit Attention contract is:

```text
QKV Linear: U16 -> S16, weight S16
scale Mul:  S16 -> S16
MatMul 1:   S16 -> S16
Softmax:    S16 -> S16
MatMul 2:   S16 -> U16
```

Explicit `--attention-dtype S8` or `S16` overrides auto inference. This preserves mixed contracts such
as global U8/S8 with local Attention S16. An explicit override creates intentional dtype boundaries;
inspect the resulting requants. Never call a U16/S16 model complete unless its Attention regions are
also explicitly or automatically resolved to S16. The generator refuses to overwrite its base config
or an existing output.

## Check

Run check against the checked-in generated config before every smoke or training run:

```bash
python .codex/skills/ppocr-qat-config-discovery/scripts/discover_ppocr_qat_config.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --base-config configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json \
  --image-shape 3 48 320 \
  --attention-dtype S16 \
  --expected-attention 2 \
  --check
```

After a check passes, start from floating-point weights and run QAT smoke. Do not restore a prepared
checkpoint across graph or qspec changes. Inspect the exported Q/DQ dtype and boundaries, then
generate Pulsar2 configuration from that exact QuantONNX.

## Acceptance

- Every requested Attention owner is complete and topologically validated.
- Every generated regional node name exists in the current exported FX graph.
- Global U8/S8 with auto Attention lowers as
  `MatMul(S8/S8 -> S8) -> Softmax(S8) -> MatMul(S8/S8 -> U8)`.
- Global U16/S16 with auto or explicit S16 Attention lowers as
  `MatMul(S16/S16 -> S16) -> Softmax(S16) -> MatMul(S16/S16 -> U16)`.
- Det/no-Attention generation reports zero regions and preserves a valid global U8/S8 or U16/S16 config.
- QuantONNX passes checker, project QDQ validation, ORT semantic comparison, and user structure review.

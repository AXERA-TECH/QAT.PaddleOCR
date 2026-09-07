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

### softmax·V MatMul entry: output follows the global activation

The second MatMul (softmax · V) entry intentionally **omits** `output`/`output_is_symmetric` so its
result domain follows the global activation (U8 for the U8/S8 contract, U16 for U16/S16). v6 rec
(2026-08-19) ships this form — the QuantONNX shows no requantize on `matmul_1/matmul_3` and their
outputs quantize to U8 by the global rule. Older configs (v5-rec
`configs/qat/ppocrv5_mobile_rec_u8s8_attn_s8_lsq.json`) spell the same choice out explicitly as
`output: <global activation>` with `output_is_symmetric: false`. `--check` normalizes both forms and
accepts them as equivalent (`_module_config_equivalent`).

### v6 keep-bn

v6 PPLCNetV4 reparameterization may keep the native post-sum BN
(`--keep-bn`, QAT-only, matches `train.py --keep-bn` and the keep-bn training record §1.2). Pass
`--keep-bn` to discovery so the prepared graph (and therefore the QAT JSON module names) matches the
training contract. For v5 / other models `--keep-bn` is a no-op.

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

Checked-in QAT JSON may be a union over several graph forms (training graph + folded graph + smoke
graph; see training record §39). `--check` is union-aware: each entry regenerated from the current
graph must be subsumed by a template entry of the same `module_type` (equal `module_config` — with
the softmax·V omission normalization above — and its names contained in the template's name list).
Template entries with no generated counterpart (e.g. S16 downsampling entries, or names of other
graph forms) are allowed — verify those by annotation inspection of the target prepared graph (e.g.
the `dump_blocks6_nodes.py` pattern in
`docs/references/development/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md`
§43.1).

For U8/S8 configs the output-projection entry was removed after validation (proj is FC-like; see
training record §34.4), so pass `--no-proj-entry` when checking them — otherwise the regenerated
config contains a proj entry the template lacks.

v6 rec training contract (2026-08-19) — check with `--strict-names` under the exact training
contract before training:

```bash
python .codex/skills/ppocr-qat-config-discovery/scripts/discover_ppocr_qat_config.py \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights weights/ptocr_v6_small_rec_full.pth \
  --base-config configs/qat/ppocrv6_small_rec_u8s8_attn_s8.json \
  --image-shape 3 48 320 \
  --attention-dtype S8 \
  --expected-attention 2 \
  --no-proj-entry \
  --rec-graph pretrained_train \
  --batch-size 64 \
  --keep-bn \
  --check --strict-names
```

The deploy contract is checked the same way with `--rec-graph deploy --batch-size 1 --keep-bn`
(normal mode, no `--strict-names`: the deploy graph has no gtc branch, and the ctc union names
are subsumed by the template).

Add `--strict-names` to also require every template entry (including hand-written entries such
as the v6 gtc branch) to hit the current prepared graph — per entry, at least one of its
`module_names` must exist. Union entries that mix names of several graph forms (e.g. the v6 ctc
scale Mul entry listing both the deploy-graph `mul_5/mul_6` and the keep-bn dynamic-batch-64
training-graph `mul_30/mul_31`) stay valid: each graph contract hits its own names. A template
entry whose names come from a completely different contract (e.g. static batch-1 gtc names
checked against a dynamic training graph) is reported and blocks the run.

## Node numbering pitfall (2026-08-13, updated 2026-08-19)

QAT regional `module_names` are matched against the graph produced by
`prepare_qat_pt2e`, whose node numbering depends on the training contract:

- dynamic batch / batch-aligned gtc targets (rec `pretrained_train`, batch > 1);
- the model wrapper passed to prepare (export the ORIGINAL model, not an
  already-exported GraphModule — nested export shifts numbers);
- batch size and `max_batch`.

Discovery therefore prepares the model with the same dynamic-shape contract as
train.py and remaps discovered roles onto the prepared graph. Symptom when this
is missed: a regional entry silently does not hit (e.g. scale Mul falls back to
global U16 input) and QuantONNX gains requantize Identity nodes
(`select -> Mul`, `mul_58` vs `mul_60/61`). Verify with
`--batch-size`/`--dynamic-batch` matching training, then re-check the exported
Identity count (expected: only the SE avg_pool u16->u16 boundary).

**Static batch 1 vs dynamic batch 64 is a graph-structure difference, not just
shapes (2026-08-19)**: enabling dynamic shapes inserts nodes, so Mul numbering
differs between the two contracts even for the same model code — e.g. v6 rec
ctc attention scale Mul is `mul_5/mul_6` in the static batch-1 training graph
but `mul_8/mul_9` under the dynamic batch-64 training contract (gtc branch
`mul_10..18`), and with `--keep-bn` it shifts again (`mul_30/31`). A config
checked against a static batch-1 QuantONNX therefore silently mis-targets the
dynamic training graph: the Mul entries hit the wrong nodes. Always check with
`--strict-names` under the exact training contract before training.

## Acceptance

- Every requested Attention owner is complete and topologically validated.
- Every generated regional node name exists in the current exported FX graph.
- Global U8/S8 with auto Attention lowers as
  `MatMul(S8/S8 -> S8) -> Softmax(S8) -> MatMul(S8/S8 -> U8)`.
- Global U16/S16 with auto or explicit S16 Attention lowers as
  `MatMul(S16/S16 -> S16) -> Softmax(S16) -> MatMul(S16/S16 -> U16)`.
- Det/no-Attention generation reports zero regions and preserves a valid global U8/S8 or U16/S16 config.
- QuantONNX passes checker, project QDQ validation, ORT semantic comparison, and user structure review.

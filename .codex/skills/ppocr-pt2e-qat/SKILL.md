---
name: ppocr-pt2e-qat
description: Train, export, debug, and validate PP-OCR detection or recognition models with PyTorch 2.6 PT2E QAT and Axera QuantONNX. Use when working in the PaddleOCR QAT project on QAT profiles, observer lifecycle, export_for_training or prepare_qat_pt2e parity, shared quantization domains, strict checkpoint reload, QuantONNX export, ONNX Runtime comparison, or Axera conversion acceptance.
---

# PP-OCR PT2E QAT

Work from the repository root. Keep Paddle model YAML, Axera quantization JSON, training
profile, checkpoint, and exported graph as one versioned contract.

## Establish The Contract

1. Identify `det` or `rec`, the exact model YAML and converted floating-point weights.
 2. Read the input shape from `Global.d2s_train_image_shape`; keep H/W fixed unless the training
    profile declares a tested discrete shape contract. PP-OCRv5 rec supports heights 32/48/64 as
    `16 * height_factor` with width 320; run prepared at every declared height. QuantONNX remains
    static at the profile `image_shape` and must not inherit the training height symbol.
 3. Export ONNX/QuantONNX with `batchSize=1` (static batch 1, deployment inference shape) unless
    the task or experiment explicitly requires another batch (e.g. training-graph parity check with
    batch 64, detection multi-batch validation). Recognition training graphs keep a dynamic-batch
    contract in `export_for_training` (gtc targets are batch-aligned), but the exported deployment
    graph is fixed at batch 1.
 4. Start QAT from floating-point pretrained weights whenever the model graph, quantization JSON, or
    PyTorch export environment changes. Do not resume a prepared checkpoint across graph changes.
 5. Use a QAT training profile. Let explicit CLI values override the profile, and let the profile
    override floating-point Paddle YAML training defaults.
 6. Keep random data augmentation disabled for the baseline. For detection, use PaddleOCR fixed-shape
    resize preprocessing and transform polygons with the same horizontal/vertical scales. The centered
    letterbox path is an explicit experimental option and must transform polygons with its scale and offset.

## Train

1. Run one real-data epoch through prepare, forward/backward, validation, checkpoint save, and strict
   reload before a full run.
2. Use low-learning-rate fine-tuning. Start with the checked-in profile; compare learning rates only
   after the end-to-end baseline passes. QAT is a quantized-domain fine-tune of pretrained weights:
   the Paddle floating-point training learning rate (e.g. 5e-4) is a reference only and must not be
   used directly for QAT (current baseline: lr 3e-5, warmup 5, Cosine).
3. Use **AdamW or SGD only** (`profile.training.optimizer: AdamW|SGD`). Never use `Adam`: its L2
   weight decay combines with LSQ gradient scaling (`use_grad_scaling`) and wrongly decays the
   learnable scale/zero_point quant parameters, which diverges early in QAT.
4. Keep observers enabled throughout training. During validation, disable observers temporarily and
   restore them afterward. Do not set `--observer-freeze-epoch` for the baseline.
5. Keep QAT EMA and DDP out of the baseline. Add them only through controlled metric comparisons.
6. Reject missing/non-finite gradients and non-finite loss immediately.

## Localize Accuracy Loss

Compare the same preprocessed samples in this order:

1. eager floating-point model;
2. `export_for_training` floating-point graph;
3. prepared graph with observer and fake quant disabled;
4. prepared graph with fake quant enabled;
5. `convert_pt2e` graph;
6. QuantONNX in ONNX Runtime;
7. Axera model with identical preprocessing and postprocessing.

Use `ORT_DISABLE_ALL` for the QuantONNX QDQ semantic reference. Run ORT graph optimization only as a
separate diagnostic: an optimized ORT session can pass checker/session creation while changing PP-OCR
task metrics materially. Record the execution backend for prepared and converted PT2E because CPU and
CUDA quantized-decomposed kernels may not produce identical task metrics.

Record `max_abs`, MAE, and task metrics at every boundary. For DB detection, compare shrink,
threshold, and binary maps before DB postprocess, then precision/recall/hmean. For recognition,
compare CTC logits before decoding, then sequence accuracy and normalized edit distance.

Inspect BatchNorm training state, momentum, and eps in exported and prepared graphs. If fake-quant-off
prepared output differs from exported float output, fix PT2E preparation before tuning quantization.

## Validate Quantization Structure

1. Discover regional targets from `source_fn_stack` plus topology. Never reuse stale FX node numbers.
2. Require every regional rule to match before training, then inspect actual ONNX Q/DQ dtype, scale,
   zero-point, and axis after export.
3. Preserve shared qparams across data-movement operations such as Concat, Split, and Reshape when
   required by Axera.
4. Use independent input/output qspec or local U16 only after a parity report identifies a sensitive
   boundary.
5. Run `onnx_program.optimize()` and remove only mathematically identical redundant DQ/Q pairs. Do not
   use percentage-tolerance qparam merging to hide a QAT graph error.

## Deliver

Follow [references/acceptance.md](references/acceptance.md). Use `tools/export_ocr_onnx.py checkpoint`, preserve
strict checkpoint loading, and treat successful export as insufficient without checker, ORT, QDQ
structure, and Axera conversion checks.

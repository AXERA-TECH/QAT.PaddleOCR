# PP-OCRv5 PyTorch / Axera QAT

本文档集中说明 PP-OCRv5 mobile recognition 和 detection 的 PyTorch 2.6 PT2E QAT 路线。
公共环境安装、数据格式、checkpoint 恢复、精度定位、QDQ 结构和 Axera 验收合同见
[主 README](README.md)。所有命令都从仓库根目录执行。

## 1. 支持范围

| 模型 | 训练图 | 部署输出 | 当前状态 |
| --- | --- | --- | --- |
| PP-OCRv5 mobile rec | CTC+NRTR/GTC MultiHead | 原始 CTC logits | exp16 已完成 QuantONNX、ORT 和 AX650 对齐 |
| PP-OCRv5 mobile det | DBHead 主分支和辅助监督 | shrink map | 浮点/QAT/QuantONNX 可执行，尚无同等级公开板端验收 |

先按主 README 第 2、3 节安装环境并准备数据，然后设置：

```bash
export PYTHON=/path/to/envs/ppocr-qat/bin/python
export CONVERT_PYTHON=/path/to/envs/ppocr-convert/bin/python
export PYTHONPATH="$PWD"
export DATA_ROOT=/path/to/ppocr_data
export REC_TRAIN_LABEL="$DATA_ROOT/rec_train.txt"
export REC_VAL_LABEL="$DATA_ROOT/rec_val.txt"
export DET_TRAIN_LABEL="$DATA_ROOT/det_train.txt"
export DET_VAL_LABEL="$DATA_ROOT/det_val.txt"
```

模型代码、输入 shape、PyTorch 版本、重参数化方式或 QAT JSON 变化后，必须从浮点权重重新 prepare，
不能跨图合同恢复 prepared checkpoint。QAT 只使用 AdamW 或 SGD，observer 在 baseline 训练期间保持
开启，QuantONNX 固定导出为静态 batch 1。

## 2. PP-OCRv5 Mobile Rec

### 2.1 当前合同

已完成板端对齐的 exp16 使用全局 U8/S8、Attention S8 和 CNN 下采样链 S16 混合位宽 LSQ：

```text
model:       configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml
float:       weights/ptocr_v5_mobile_rec_full.pth
qat config:  configs/qat/ppocrv5_mobile_rec_u8s8_attn_s8_downsample_s16_lsq.json
profile:     configs/qat/training/ppocrv5_mobile_rec_u8s8_sgd_dynamic_height_exp16_downsample_s16_reparam.yml
train graph: CTC+NRTR/GTC pretrained_train, dynamic batch
train shape: 3x48x320
deploy:      reparameterized CTC graph, static batch 1
```

profile 文件名保留了早期 `dynamic_height` 命名，但 exp16 profile 的
`multi_scale_training: false`，实际训练 shape 为固定 `3x48x320`。框架支持 32/48/64 三档离散高度，
但启用后属于新的训练图合同，必须重新发现 qspec、执行 smoke 并从浮点权重重训。

### 2.2 转换 Paddle 权重

```bash
env PYTHONPATH="$PWD" "$CONVERT_PYTHON" converter/ppocr_v5_rec_converter.py \
  --yaml_path configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --src_model_path weights/PP-OCRv5_mobile_rec_pretrained.pdparams \
  --output weights/ptocr_v5_mobile_rec_full.pth
```

转换必须保留完整 CTC+NRTR/GTC 训练结构。出现未知 Paddle 参数、普通参数缺失、shape 不一致或
ignored source 时，不要继续 QAT。随后检查完整训练图和部署图：

```bash
env PYTHONPATH="$PWD" "$CONVERT_PYTHON" tools/compare_rec_parity.py --help
env PYTHONPATH="$PWD" "$CONVERT_PYTHON" tools/compare_pretrained_training.py --help
```

### 2.3 核对 QAT 图并执行 Smoke

节点名必须从本次 `export_for_training` 图重新发现，不能复制历史 FX 编号：

```bash
$PYTHON .codex/skills/ppocr-qat-config-discovery/scripts/discover_ppocr_qat_config.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --base-config configs/qat/ppocrv5_mobile_rec_u8s8_attn_s8_downsample_s16_lsq.json \
  --image-shape 3 48 320 \
  --batch-size 64 \
  --expected-attention 2 \
  --attention-dtype S8 \
  --reparameterize \
  --rec-graph pretrained_train \
  --dynamic-batch \
  --no-proj-entry \
  --check \
  --strict-names
```

部署图 smoke：

```bash
$PYTHON tools/qat_smoke.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --qat-config configs/qat/ppocrv5_mobile_rec_u8s8_attn_s8_downsample_s16_lsq.json \
  --output /tmp/ppocrv5_mobile_rec_qat_smoke.onnx \
  --image-shape 3 48 320 \
  --reparameterize \
  --onnx-optimize \
  --ort-optimizer-check
```

smoke 后确认 Attention 连续 S8、四个下采样 Conv 及其相邻链路为 S16、其余激活为全局 U8，且不存在
未量化 Conv/Linear/MatMul、异常 requant 或未解释的 float island。

### 2.4 真实数据门禁和正式 QAT

首次使用当前图合同时先跑一个 epoch：

```bash
$PYTHON tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --label-file "$REC_TRAIN_LABEL" \
  --data-dir "$DATA_ROOT" \
  --val-label-file "$REC_VAL_LABEL" \
  --val-data-dir "$DATA_ROOT" \
  --output-dir runs/ppocrv5_mobile_rec_qat_gate \
  --training-profile configs/qat/training/ppocrv5_mobile_rec_u8s8_sgd_dynamic_height_exp16_downsample_s16_reparam.yml \
  --epochs 1 \
  --workers 0 \
  --device cuda
```

门禁必须覆盖完整训练图 backward、validation、checkpoint save、strict reload、convert 和 QuantONNX。
通过后从同一浮点权重重新启动正式训练：

```bash
$PYTHON tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --label-file "$REC_TRAIN_LABEL" \
  --data-dir "$DATA_ROOT" \
  --val-label-file "$REC_VAL_LABEL" \
  --val-data-dir "$DATA_ROOT" \
  --output-dir runs/ppocrv5_mobile_rec_qat \
  --training-profile configs/qat/training/ppocrv5_mobile_rec_u8s8_sgd_dynamic_height_exp16_downsample_s16_reparam.yml \
  --device cuda
```

### 2.5 QuantONNX 和 ORT

```bash
$PYTHON tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/ppocrv5_mobile_rec_qat/best.pt \
  --output exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --batch-size 1 \
  --ort-optimizer-check

$PYTHON tools/verify_qat_onnx.py \
  --model exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --check-value
```

全量评估必须同时记录两种 ORT 模式：

```bash
$PYTHON tools/evaluate_onnx.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --onnx exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --label-file "$REC_VAL_LABEL" \
  --data-dir "$DATA_ROOT" \
  --batch-size 1 \
  --no-ort-optimize

$PYTHON tools/evaluate_onnx.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --onnx exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --label-file "$REC_VAL_LABEL" \
  --data-dir "$DATA_ROOT" \
  --batch-size 1 \
  --ort-optimize
```

`ORT_DISABLE_ALL` 是 QDQ 语义基线，开启图优化只作为独立诊断。

### 2.6 Pulsar2 配置

配置必须从最终 QuantONNX 的真实节点生成，不能复用旧 ONNX 的节点名：

```bash
$PYTHON .codex/skills/ppocr-pulsar2-config/scripts/generate_pulsar2_config.py \
  --profile ppocrv5-rec \
  --onnx exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --output artifacts/pulsar2/ppocrv5_mobile_rec_best.json \
  --output-dir artifacts/pulsar2/ppocrv5_mobile_rec_best \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --attention-dtype S8 \
  --calibration-dataset /path/to/calibration.zip

$PYTHON .codex/skills/ppocr-pulsar2-config/scripts/validate_pulsar2_config.py \
  --profile ppocrv5-rec \
  --onnx exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --config artifacts/pulsar2/ppocrv5_mobile_rec_best.json \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --attention-dtype S8
```

Pulsar2 frontend 会重命名或融合节点，生成后仍需检查 `frontend/optimized.onnx` 中的 Attention S8、
下采样链 S16 和保留 requant，再构建正式 AXModel。

### 2.7 已验收结果

exp16 的训练内 best acc 为 0.5927。最终 QuantONNX 和 AX650 使用同一份 2077 样本验证集：

| 阶段 | acc | norm edit |
| --- | ---: | ---: |
| QuantONNX ORT `ORT_DISABLE_ALL` | 0.58546 | 0.8188 |
| AX650 | 0.58449 | 未记录 |

板端与 ORT acc 相差 0.001。50 样本逐层对比的 mean argmax agreement 为 0.9900，mean MAE 为
1.156。完整 provenance 见
v5 的完整训练过程和历史实验属于内部开发资料；本文件只保留面向用户的复现与部署命令。

## 3. PP-OCRv5 Mobile Det

检测训练图保留 DBHead 的 shrink、threshold、binary 和辅助监督，部署图只输出 shrink map。

### 3.1 转换和浮点对齐

```bash
env PYTHONPATH="$PWD" "$CONVERT_PYTHON" converter/ppocr_v5_det_converter.py \
  --yaml_path configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --src_model_path weights/PP-OCRv5_mobile_det_pretrained.pdparams

mv ptocr_v5_mobile_det.pth weights/ptocr_v5_mobile_det.pth
```

转换后使用 `tools/compare_det_parity.py` 和 `tools/compare_det_training.py` 检查 Paddle/PyTorch 的
shrink、threshold、binary、完整 loss 和梯度。

### 3.2 QAT 配置和 Smoke

```bash
$PYTHON .codex/skills/ppocr-qat-config-discovery/scripts/discover_ppocr_qat_config.py \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --weights weights/ptocr_v5_mobile_det.pth \
  --base-config configs/qat/ppocrv5_mobile_det_u8s8.json \
  --image-shape 3 640 640 \
  --expected-attention 0 \
  --reparameterize \
  --check \
  --strict-names

$PYTHON tools/qat_smoke.py \
  --task det \
  --det-graph training \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --weights weights/ptocr_v5_mobile_det.pth \
  --qat-config configs/qat/ppocrv5_mobile_det_u8s8.json \
  --output /tmp/ppocrv5_mobile_det_training_qat_smoke.onnx \
  --image-shape 3 640 640 \
  --reparameterize \
  --onnx-optimize \
  --ort-optimizer-check

$PYTHON tools/qat_smoke.py \
  --task det \
  --det-graph inference \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --weights weights/ptocr_v5_mobile_det.pth \
  --qat-config configs/qat/ppocrv5_mobile_det_u8s8.json \
  --output /tmp/ppocrv5_mobile_det_inference_qat_smoke.onnx \
  --image-shape 3 640 640 \
  --reparameterize \
  --onnx-optimize \
  --ort-optimizer-check
```

training smoke 必须覆盖完整 DBHead backward，inference smoke 必须确认部署投影只保留 shrink map。

### 3.3 正式 QAT、导出和评估

先增加 `--epochs 1 --workers 0` 完成真实数据门禁，再从浮点权重启动正式 run：

```bash
$PYTHON tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --weights weights/ptocr_v5_mobile_det.pth \
  --label-file "$DET_TRAIN_LABEL" \
  --data-dir "$DATA_ROOT" \
  --val-label-file "$DET_VAL_LABEL" \
  --val-data-dir "$DATA_ROOT" \
  --output-dir runs/ppocrv5_mobile_det_qat \
  --training-profile configs/qat/training/ppocrv5_mobile_det_baseline.yml \
  --optimizer AdamW \
  --device cuda

$PYTHON tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/ppocrv5_mobile_det_qat/best.pt \
  --output exports/quantonnx/ppocrv5_mobile_det_best.onnx \
  --batch-size 1 \
  --ort-optimizer-check

$PYTHON tools/verify_qat_onnx.py \
  --model exports/quantonnx/ppocrv5_mobile_det_best.onnx \
  --check-value
```

ORT 评估使用主 README 检测章节中的命令，将模型 YAML 和 ONNX 路径替换为本节 v5 路径，并分别运行
`--no-ort-optimize` 与 `--ort-optimize`。主指标为 hmean，同时记录 precision、recall 以及
shrink/threshold/binary map 的阶段误差。

## 4. 非重参化 QAT 和量化域折叠

非重参化多分支 QAT 到单分支部署模型属于高级实验路线，不是 exp16 默认流程。入口为：

```bash
$PYTHON tools/finetune_folded.py --help
```

该流程先从非重参化 prepared checkpoint 折叠权重，再针对新的单分支 FX 图重新 prepare 和微调。
内部已消失分支的 activation qparams 没有一一对应节点，默认重新观测；
`--activation-qparam-transfer semantic` 只尝试迁移能够按 FX producer 语义匹配的边界 scale/zero-point。
训练图、折叠图或 qspec 改变后不得直接恢复旧 prepared checkpoint。详细流程见
`.codex/skills/ppocr-quantized-domain-fold/` 和上述 v5 训练记录。

## 5. 验收要求

每个 v5 QAT run 仍需遵守主 README 的 checkpoint、精度定位、QDQ 和板端验收合同，至少记录：

- 模型 YAML、浮点权重、QAT JSON、training profile、输入 shape 和实际命令；
- eager、exported、prepared fake-off、fake-on、converted、QuantONNX 和 AXModel 的相邻阶段误差；
- ORT 未优化与开启优化两组全量任务指标；
- rec 的 CTC logits、argmax agreement、accuracy 和 normalized edit distance；
- det 的 shrink/threshold/binary map、precision、recall 和 hmean；
- checkpoint、QuantONNX、Pulsar2 配置、AXModel 和报告路径。

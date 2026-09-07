# PaddleOCR PyTorch / Axera QAT

本仓库将 PaddleOCR 的 PP-OCR 检测、识别模型和预训练权重严格转换到 PyTorch，使用 PyTorch 2.6
PT2E 完成浮点训练、QAT、checkpoint 恢复和 QuantONNX 导出，并面向 Axera/Pulsar2 进行结构与精度
验收。

## PP-OCRv6 精度概览

检测统一报告 `hmean`，识别统一报告 sequence accuracy，均为越高越好。QAT 指标优先采用 AX650
板端全量结果；QuantONNX 在 `ORT_DISABLE_ALL` 下的结果作为语义基线和板端对照。

| 模型 | 验证集 | 主指标 | 浮点指标 | 量化方式 | 配置 | 量化指标（板端优先） |
| --- | --- | --- | ---: | --- | --- | ---: |
| PP-OCRv6 small rec | TextOCR/COCO-Text 混合 `rec_val`（149,695）<br>ICDR（2,077） | accuracy | `0.7326`<br>`0.7400` | PTQ | [U8/S8 + Attention S8](configs/qat/ppocrv6_small_rec_u8s8_attn_s8.json)，512 个同验证集样本校准¹ | converted `0.6987`<br>converted `0.6755` |
| PP-OCRv6 small rec | TextOCR/COCO-Text 混合 `rec_val`（149,695）<br>ICDR（2,077） | accuracy | `0.7326`<br>`0.7400` | QAT | [U8/S8 + Attention S8](configs/qat/ppocrv6_small_rec_u8s8_attn_s8.json)，已验证配置 | AX650 `0.7070`（ORT `0.7072`）<br>AX650 `0.7304`（ORT `0.7232`） |
| PP-OCRv6 small det | 混合 `det_val`（3,624；TextOCR 3,124 + ICDR 500）<br>ICDAR2015 test（500） | hmean | `0.5272`<br>`0.6003` | PTQ | [U8/S8](configs/qat/ppocrv6_small_det_u8s8.json)，浮点微调权重、128 张训练图校准³ | prepared `0.5144` / converted `0.5148`<br>prepared `0.6138` / converted `0.6159` |
| PP-OCRv6 small det | 混合 `det_val`（3,624；TextOCR 3,124 + ICDR 500）<br>ICDAR2015 test（500） | hmean | `0.5272`<br>`0.6003` | QAT | [U8/S8](configs/qat/ppocrv6_small_det_u8s8.json)，浮点微调权重重新 prepare 后训练 | AX650 `0.5476`（ORT `0.5470`）<br>AX650 `0.6922`（ORT `0.6956`） |

注：
- v6-rec 的 PTQ 报告默认取各验证集前 512 个样本校准，校准集与评估集不独立，因此这里只归档
已有数值，不能据此发布 QAT/PTQ 收益结论；公平 PTQ 对照仍待补跑。

- v6-det 本次 PTQ、浮点和 QAT 均以浮点微调权重为起点；PTQ 使用训练集前 128 张样本校准，并在完整
det_val 上评估。因此 PTQ 与浮点的差异可用于观察 8bit 量化损失，PTQ 与 QAT 的差异仍同时包含 QAT 微调收益。

- 表中 v6-det 指标采用居中 `letterbox` 预处理；当前仓库支持与 PaddleOCR 对齐的固定尺寸 `resize` 预处理，
并默认使用该方式。不同预处理条件下的指标不可直接混用。

所有命令都从仓库根目录执行。命令的完整参数以对应工具的 `--help` 为准。

## 1. 环境安装

> **重要：Paddle 和 PyTorch 建议使用不同的 CUDA 大版本。** 本文转换环境使用 Paddle CUDA 12.6
> 和 PyTorch CUDA 11.8；两个框架会携带或依赖各自的 CUDA runtime、cuDNN 和相关动态库。

### 1.1 Paddle 权重转换环境

转换器需要同时导入 Paddle 和 PyTorch，使用 Python 3.10。建议使用独立环境，并按已验证版本安装：

```bash
conda create -n ppocr-convert python=3.10 -y
conda activate ppocr-convert
python -m pip install --upgrade pip
# 官方 Paddle CUDA 12.6 wheel；Torch 使用 CUDA 11.8 wheel
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4
python -m pip install opencv-python==4.6.0.66
python -m pip install PyYAML==6.0.3
```

上述依赖覆盖 Paddle 权重转换和 Paddle/PyTorch 浮点对齐。这里的 Paddle `cu126` 对应 CUDA 12.6，
Torch `cu118` 对应 CUDA 11.8；系统 NVIDIA 驱动需要同时满足两套 CUDA 构建的最低要求。若只做 CPU
权重转换，可将 Paddle GPU 包替换为 `paddlepaddle==3.0.0`，并将 Torch 替换为 CPU wheel。模型和权重
来源必须匹配，不能混用不同版本的 YAML 与 `.pdparams`。

### 1.2 QAT、训练和导出环境

使用 Python 3.10，完整依赖和已验证版本见
[`requirements-qat.txt`](requirements-qat.txt)。参考安装：

```bash
conda create -n ppocr-qat python=3.10 -y
conda activate ppocr-qat
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-qat.txt
```

## 2. 数据准备要求

默认用户已经完成数据下载、解压和标注转换。训练和验证数据需整理为 PaddleOCR 标注格式，并将路径
传给后续命令：

```bash
DATA_ROOT=/path/to/ppocr_data
REC_TRAIN_LABEL="$DATA_ROOT/rec_train.txt"
REC_VAL_LABEL="$DATA_ROOT/rec_val.txt"
DET_TRAIN_LABEL="$DATA_ROOT/det_train.txt"
DET_VAL_LABEL="$DATA_ROOT/det_val.txt"
```

识别标注每行使用 `图片路径<TAB>文本`，例如：

```text
rec/images/word_001.jpg<TAB>PaddleOCR
```

检测标注每行使用 `图片路径<TAB>polygon JSON`，例如：

```text
det/images/img_001.jpg<TAB>[{"transcription":"text","points":[[12,16],[80,16],[80,48],[12,48]]}]
```

识别字典、`max_text_length` 和 `use_space_char` 从模型 YAML 读取；检测忽略文本使用 `"###"` 或
`"*"`。train/val 应按原图划分，不能按 polygon 随机拆分。QAT baseline 关闭随机增强，验证集始终禁用增强。

v6-rec 固定输入为 `3x48x320`，识别预处理为等比缩放、右侧 zero padding 和 `[-1, 1]` 归一化。
v6-det 默认使用 PaddleOCR 固定输入尺寸 resize、polygon 同步缩放和 ImageNet mean/std 归一化；如需
复现本项目的居中 letterbox 实验，可在训练/评估命令中增加 `--det-preprocess letterbox`。
QuantONNX 部署输入固定为 batch 1。

## 3. PP-OCRv6 Small Rec 端到端复现

这是当前推荐主线。下面的命令依次完成 Paddle 权重转换、浮点微调、QAT、QuantONNX 和 ORT 评估。

### 3.1 转换 Paddle 预训练权重

将官方权重放到 `weights/PP-OCRv6_small_rec_pretrained.pdparams`：

```bash
python3 converter/ppocr_v6_rec_converter.py \
  --yaml_path configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --src_model_path weights/PP-OCRv6_small_rec_pretrained.pdparams \
  --save_mode full \
  --output weights/ptocr_v6_small_rec_full.pth
```

`--save_mode full` 必须保留 CTC+NRTR/GTC 训练权重。转换出现未知 Paddle 参数、普通模型参数缺失、
shape 不一致或 ignored source 时，不要继续训练。

转换后先运行完整训练图和部署图对齐工具：

```bash
python3 tools/compare_rec_parity.py --help
python3 tools/compare_pretrained_training.py --help
```

### 3.2 浮点微调（可选）

如果已有可用的纯 PyTorch 浮点 `state_dict`，可以直接跳过本节，进入 3.3 QAT 复现。只有需要
适配本地数据分布、或希望从官方转换权重重新获得浮点起点时，才执行浮点微调。

当前复现 profile 保持 Paddle 原始多分支结构和完整训练头：

```bash
python3 tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights weights/ptocr_v6_small_rec_full.pth \
  --label-file "$REC_TRAIN_LABEL" \
  --data-dir "$DATA_ROOT" \
  --val-label-file "$REC_VAL_LABEL" \
  --val-data-dir "$DATA_ROOT" \
  --output-dir runs/ppocrv6_small_rec_float \
  --training-profile configs/qat/training/ppocrv6_small_rec_ft_exp22.yml \
  --device cuda
```

浮点训练使用 `reparameterize: false` 和 `rec_graph: pretrained_train`。不要把 CTC-only deployment
graph 当作 Paddle pretrained training graph。

`best.pt` 是包含 optimizer、scheduler 和 metadata 的 trainer checkpoint，而 QAT 的 `--weights`
要求纯模型 `state_dict`。浮点训练结束后显式提取：

```bash
python3 -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); torch.save(checkpoint["model"], sys.argv[2])' \
  runs/ppocrv6_small_rec_float/best.pt \
  weights/ptocr_v6_small_rec_float_best.pth
```

### 3.3 QAT 复现

从同一浮点权重重新启动正式 run，不从门禁 checkpoint 续训：

```bash
python3 tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights weights/ptocr_v6_small_rec_float_best.pth \
  --label-file "$REC_TRAIN_LABEL" \
  --data-dir "$DATA_ROOT" \
  --val-label-file "$REC_VAL_LABEL" \
  --val-data-dir "$DATA_ROOT" \
  --output-dir runs/ppocrv6_small_rec_qat \
  --training-profile configs/qat/training/ppocrv6_small_rec_qat_from_exp22.yml \
  --qat-config configs/qat/ppocrv6_small_rec_u8s8_attn_s8.json \
  --optimizer AdamW \
  --keep-bn \
  --device cuda
```

QAT 优化器建议 AdamW 或 SGD。baseline 使用 FP32、关闭 AMP 和随机增强，observer 全程开启；验证时 trainer
临时关闭 observer 并在结束后恢复。除独立实验外，不设置 `--observer-freeze-epoch`。

QAT 配置、重参数化方式和训练参数已经在仓库中验证，当前仓库已验证的 QAT 配置可直接按本节命令复现。
修改模型代码、输入 shape、qspec、重参数化方式或预处理时，应使用
`.codex/skills/ppocr-qat-config-discovery` 从当前 `export_for_training` FX 图重新发现节点并生成/检查
QAT JSON；随后按 `.codex/skills/ppocr-pt2e-qat` 执行门禁、导出和验收。

训练目录：

```text
train.log                 终端训练输出和异常日志（恢复训练时追加）
best.pt                   主指标 acc 最优 checkpoint
last.pt                   最后一个 epoch checkpoint
epoch_NNNN.pt             周期 checkpoint
epoch_NNNN_step_*.pt      save_every_steps 中间 checkpoint
step_validation.jsonl     step 全量验证记录
```

恢复训练时，在上述命令中增加 `--resume runs/ppocrv6_small_rec_qat/last.pt`。恢复合同要求模型 YAML、
QAT JSON、Torch 版本、输入 shape、重参数化、keep-BN、optimizer、dynamic shape 和 batch policy 保持
一致；`--epochs` 表示恢复后的总 epoch 上限。

### 3.4 导出 QuantONNX

保留 checkpoint observer 统计的基线导出：

```bash
python3 tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/ppocrv6_small_rec_qat/best.pt \
  --output exports/quantonnx/ppocrv6_small_rec_best.onnx \
  --batch-size 1 \
  --ort-optimizer-check
```

训练后重校准是可选实验，默认关闭。它只使用代表性数据更新激活 scale/zero-point，不修改权重：

```bash
python3 tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/ppocrv6_small_rec_qat/best.pt \
  --output exports/quantonnx/ppocrv6_small_rec_best_recalibrated.onnx \
  --batch-size 1 \
  --recalibrate \
  --calibration-label-file "$REC_TRAIN_LABEL" \
  --calibration-data-dir "$DATA_ROOT" \
  --calibration-samples 512 \
  --calibration-batch-size 8 \
  --ort-optimizer-check
```

导出器按 checkpoint metadata 重建图并 strict load，随后执行 `convert_pt2e`、
`onnx_program.optimize()`、精确冗余 DQ/Q 清理、ONNX checker、QDQ 检查和未优化 ORT 数值比较。
QuantONNX 固定为静态 batch 1、输入 `images[1,3,48,320]`，输出原始 CTC logits，不含 Softmax 和
GTC 辅助分支。

### 3.5 QDQ 和 ORT 双指标

```bash
python3 tools/verify_qat_onnx.py \
  --model exports/quantonnx/ppocrv6_small_rec_best.onnx \
  --check-value
```

所有 QuantONNX 必须同时记录 ORT 未优化和开启优化两组全量指标：

```bash
python3 tools/evaluate_onnx.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --onnx exports/quantonnx/ppocrv6_small_rec_best.onnx \
  --label-file "$REC_VAL_LABEL" \
  --data-dir "$DATA_ROOT" \
  --batch-size 1 \
  --no-ort-optimize

python3 tools/evaluate_onnx.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --onnx exports/quantonnx/ppocrv6_small_rec_best.onnx \
  --label-file "$REC_VAL_LABEL" \
  --data-dir "$DATA_ROOT" \
  --batch-size 1 \
  --ort-optimize
```

`ORT_DISABLE_ALL` 是 QDQ 语义基线。ORT graph optimization 只作独立诊断，不能替代未优化指标。
`onnx_program.optimize()` 是导出阶段的 ONNXProgram 优化，与 ORT graph optimization 不是同一过程。

### 3.6 公平 PTQ 基线（可选）

PTQ 校准集必须独立于评估集。以下命令不训练参数，只比较浮点权重的 prepared fake-off/fake-on 和
converted：

```bash
python3 tools/compare_pt2e_quant_stages.py \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights weights/ptocr_v6_small_rec_float_best.pth \
  --qat-config configs/qat/ppocrv6_small_rec_u8s8_attn_s8.json \
  --label-file "$REC_VAL_LABEL" \
  --data-dir "$DATA_ROOT" \
  --calibration-label-file "$REC_TRAIN_LABEL" \
  --calibration-data-dir "$DATA_ROOT" \
  --calibration-samples 512 \
  --calibration-batch-size 8 \
  --samples 0 \
  --batch-size 32 \
  --keep-bn \
  --device cuda \
  --output /tmp/ppocrv6_small_rec_ptq.json
```

不要使用评估集前 N 个样本同时做校准和评估，也不要把不同数据集的 float/QAT 数值相减。

## 4. PP-OCRv6 Small Det QAT

检测训练图保留 DBHead 的 shrink、threshold、binary 和 Paddle 模型中的辅助监督；QuantONNX 部署图
只输出 shrink map。当前 v6-det 已完成“Float 微调 → 重新 prepare QAT → QuantONNX → Pulsar2/AX650”
双口径验收，推荐使用已验证的浮点微调后 QAT 路线。
PP-OCRv5 mobile det 见 [PP-OCRv5 QAT 使用指南](README-v5.md)。

### 4.1 转换检测权重

PP-OCRv6 small det：

```bash
python3 converter/ppocr_v6_det_converter.py \
  --yaml_path configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --src_model_path weights/PP-OCRv6_small_det_pretrained.pdparams \
  --output weights/ptocr_v6_small_det_full.pth
```

转换后使用 `tools/compare_det_parity.py` 和 `tools/compare_det_training.py` 检查 Paddle/PyTorch 的
shrink、threshold、binary、完整 loss 和梯度。

### 4.2 检测浮点训练

```bash
python3 tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights weights/ptocr_v6_small_det_full.pth \
  --label-file "$DET_TRAIN_LABEL" \
  --data-dir "$DATA_ROOT" \
  --val-label-file "$DET_VAL_LABEL" \
  --val-data-dir "$DATA_ROOT" \
  --output-dir runs/exp20b_ppocrv6_small_det_float_finetune_bs64 \
  --training-profile configs/qat/training/ppocrv6_small_det_float_finetune_bs64.yml \
  --device cuda
```

该 profile 使用官方等效 batch size 64、Adam、lr 1e-3、Paddle 风格增强和 FP32 训练；单卡显存不足时，
可改用 `configs/qat/training/ppocrv6_small_det_float_finetune.yml`（batch size 8、lr 1e-4）作为回退。
浮点训练必须使用完整 pretrained training graph，不在训练前部署重参数化。训练结束后从 best checkpoint
提取纯模型权重，供 QAT 重新 prepare：

```bash
python3 -c 'import sys, torch; checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False); torch.save(checkpoint["model"], sys.argv[2])' \
  runs/exp20b_ppocrv6_small_det_float_finetune_bs64/best.pt \
  weights/ptocr_v6_small_det_float_best.pth
```

### 4.3 检测 QAT 复现

使用仓库已验证的 U8/S8 QAT 配置，从浮点微调权重重新启动 QAT。训练过程中每个 epoch 执行验证，
并根据 hmean 更新 `best.pt`：

```bash
python3 tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights weights/ptocr_v6_small_det_float_best.pth \
  --label-file "$DET_TRAIN_LABEL" \
  --data-dir "$DATA_ROOT" \
  --val-label-file "$DET_VAL_LABEL" \
  --val-data-dir "$DATA_ROOT" \
  --output-dir runs/exp20c_ppocrv6_small_det_u8s8_keep_bn_from_exp20b \
  --training-profile configs/qat/training/ppocrv6_small_det_baseline.yml \
  --optimizer AdamW \
  --keep-bn \
  --device cuda
```

检测主指标为 hmean，同时记录 precision、recall、shrink/threshold/binary map 误差。

恢复训练时，在上述命令中增加 `--resume runs/exp20c_ppocrv6_small_det_u8s8_keep_bn_from_exp20b/last.pt`。
恢复合同要求模型 YAML、QAT JSON、Torch 版本、输入 shape、重参数化、keep-BN、optimizer 和 batch policy
保持一致；`--epochs` 表示恢复后的总 epoch 上限。

### 4.4 导出 QuantONNX

```bash
python3 tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/exp20c_ppocrv6_small_det_u8s8_keep_bn_from_exp20b/best.pt \
  --output exports/quantonnx/ppocrv6_small_det_exp20c.onnx \
  --batch-size 1 \
  --ort-optimizer-check

python3 tools/verify_qat_onnx.py \
  --model exports/quantonnx/ppocrv6_small_det_exp20c.onnx \
  --check-value
```

导出器按 checkpoint metadata 重建图并 strict load，随后执行 `convert_pt2e`、
`onnx_program.optimize()`、QDQ 检查和 ONNX checker。QuantONNX 固定为静态 batch 1，输入
`inputs_0[1,3,640,640]`，输出 shrink map。

### 4.5 QDQ 和 ORT 双指标

所有 QuantONNX 必须同时记录 ORT 未优化和开启优化两组全量指标：

```bash
python3 tools/evaluate_onnx.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --onnx exports/quantonnx/ppocrv6_small_det_exp20c.onnx \
  --label-file "$DET_VAL_LABEL" \
  --data-dir "$DATA_ROOT" \
  --batch-size 1 \
  --no-ort-optimize

python3 tools/evaluate_onnx.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --onnx exports/quantonnx/ppocrv6_small_det_exp20c.onnx \
  --label-file "$DET_VAL_LABEL" \
  --data-dir "$DATA_ROOT" \
  --batch-size 1 \
  --ort-optimize
```

两组指标必须分别记录。Pulsar2/AXModel 部署与板端验收见
[Axera 部署与板端精度验证指南](axera/README.md)。

上述评估命令默认使用 checkpoint/训练合同中的预处理；没有 checkpoint metadata 时，检测使用 PaddleOCR
固定尺寸 resize。旧 checkpoint 若没有 `det_preprocess` 字段，则保留历史 letterbox 回退以保证可复现。
若需复现本项目的居中 letterbox 实验，在 `evaluate_onnx.py` 和 `recompute_det_bins.py`
中同时增加 `--det-preprocess letterbox`，保证 polygon 坐标和模型输入使用同一预处理：

```bash
python3 tools/evaluate_onnx.py ... --task det --det-preprocess letterbox
python3 tools/recompute_det_bins.py ... --det-preprocess letterbox
```

## 5. Pulsar2 和 AXModel 验收

完整的 QuantONNX → Pulsar2 → AXModel → 板端精度验证流程见
[Axera 部署与板端精度验证指南](axera/README.md)。

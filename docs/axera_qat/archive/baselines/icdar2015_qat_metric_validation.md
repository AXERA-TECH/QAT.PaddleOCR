# ICDAR2015 QAT Metric 验证记录

## 1. 目的

2026-07-30 为路线 2 补齐正式任务指标。此前 `Trainer.evaluate()` 只计算 DB/CTC loss，不能判断
检测 hmean 或识别序列精度，也使 `best.pt` 只能按最低 validation loss 选择。

本次实现范围：

- det：复用现有 `DBPostProcess`，增加 ICDAR polygon IoU evaluator，输出 precision、recall、hmean；
- rec：按 CTC blank/重复规则解码，输出 sequence accuracy 和 normalized edit similarity；
- validation 暂时关闭 observer，结束后按原状态恢复；
- 新训练按模型 YAML 的 `Metric.main_indicator` 选择 `best.pt`，同时继续记录最低 validation loss；
- `--eval-only` 严格恢复 checkpoint 后只运行验证，不更新模型、optimizer 或 observer。

## 2. 检测坐标和 batch 契约

训练 dataset 的返回值保持不变。仅检测 validation dataset 增加 letterbox 后的可变长 `polygons`、
`ignore_tags` 和 `[height, width, 1, 1]` shape 信息，并使用专用 collate：固定尺寸 image/DB maps
执行 stack，polygon 列表按图片保留。

DB 后处理和 GT 都在 `640x640` 居中 letterbox 坐标系中比较。这样不需要用只支持 resize ratio 的
旧 `DBPostProcess` 反推带 top/left padding 的原图坐标，也不会把训练预处理与 metric 预处理分开。

## 3. 识别指标契约

识别 metric 使用与训练相同的字典和 `use_space_char` 配置。预测序列先执行 CTC 相邻重复折叠并删除
blank，标签只删除 padding blank。默认与 Paddle `RecMetric` 一致忽略空格：

```text
acc:            完整字符串严格相等的比例
norm_edit_dis:  1 - mean(Levenshtein / max(pred_length, label_length))
```

不引入 `rapidfuzz` 新依赖，Levenshtein 使用等价的动态规划实现。

## 4. 真实 Checkpoint 验证

### 4.1 PP-OCRv6 small det

严格恢复 50-epoch 训练产生的历史 `best.pt`。该 checkpoint 是旧 Trainer 按最低 DB loss 选出的
epoch 11，不是按 hmean 选出的 checkpoint。

```text
checkpoint:  output/icdar2015_ppocrv6_small_det_qat/best.pt
dataset:     ICDAR2015 test，500 images
input:       [N, 3, 640, 640]
epoch/step:  11 / 1375
```

| Graph / backend | Precision | Recall | Hmean |
|---|---:|---:|---:|
| prepared QAT / CUDA | 0.576404 | 0.390467 | 0.465557 |
| converted PT2E / CUDA | 0.573677 | 0.386134 | 0.461583 |
| converted PT2E / CPU | 0.580043 | 0.389023 | 0.465706 |
| QuantONNX / CPU ORT optimize off | 0.580576 | 0.388541 | 0.465532 |
| QuantONNX / CPU ORT optimize on | 0.575940 | 0.368801 | 0.449662 |

该结果证明 metric 能从 prepared QAT 训练图的 shrink 输出完成真实 DB 后处理和 polygon 匹配。它只
代表 epoch 11；由于历史 epoch checkpoint 没有逐轮 hmean 记录，不能据此断言 epoch 11 也是全程
hmean 最优。下一次完整训练将直接按 hmean 保存 `best.pt`。

### 4.2 PP-OCRv6 small rec CTC smoke（历史对照）

严格恢复 ICDAR 1-epoch QAT smoke 的 `best.pt`：

```text
checkpoint:  output/icdar2015_ppocrv6_small_rec_qat_smoke/best.pt
dataset:     ICDAR2015 rec test，2077 images
input:       [N, 3, 48, 320]
epoch/step:  1 / 69
```

| Graph / backend | Accuracy | Norm edit similarity |
|---|---:|---:|
| prepared QAT / CUDA | 0.718344 | 0.882055 |
| prepared QAT / CPU | 0.716899 | 0.881361 |
| converted PT2E / CUDA | 0.708233 | 0.876186 |
| converted PT2E / CPU | 0.712085 | 0.878123 |
| QuantONNX / CPU ORT optimize off | 0.717381 | 0.879247 |
| QuantONNX / CPU ORT optimize on | 0.655272 | 0.847278 |

这是 1 epoch smoke checkpoint 的历史序列指标，不代表正式 50-epoch recognition baseline。

### 4.3 PP-OCRv6 small rec CTC 正式 50 epoch

严格恢复正式训练产生的 `best.pt`。主指标为 YAML 中的 `acc`，因此 checkpoint 为 epoch 4，而不是
最低 validation loss 或最高 edit similarity 对应的轮次：

```text
checkpoint:  output/icdar2015_ppocrv6_small_rec_qat/best.pt
dataset:     ICDAR2015 rec test，2077 images
input:       [N, 3, 48, 320]
epoch/step:  4 / 276
```

| Graph / backend | Accuracy | Norm edit similarity | Val loss |
|---|---:|---:|---:|
| prepared QAT / CUDA | 0.733751 | 0.889306 | 0.601407 |
| converted PT2E / CPU | 0.725566 | 0.886938 | 0.594975 |
| QuantONNX / CPU ORT optimize off | 0.719788 | 0.885338 | - |
| QuantONNX / CPU ORT optimize on | 0.684641 | 0.862039 | - |

正式 QuantONNX 的 optimize-off reference 相比 prepared CUDA 低 0.013963 accuracy；由于两者执行
后端不同，还需要 CUDA converted 和 Axera 板端结果来进一步拆分误差来源。optimize-on 相比 optimize-off
额外下降 0.035147 accuracy，继续证明 ORT 默认 QDQ graph optimization 不能作为当前 Axera QDQ
图的语义参考。

### 4.4 结论

1. graph stage 与执行 backend 必须同时记录。converted PT2E 在 CPU/CUDA 上存在可测指标差异，不能用
   一个 backend 的结果代表所有执行环境。
2. 当前 QuantONNX 在 ORT 禁用图优化时基本保持 prepared/converted CPU 的任务指标，说明导出的 QDQ
   图本身可以继续用于 Axera 编译验证。
3. ORT 默认图优化会破坏当前 QDQ 图语义：det hmean 下降约 0.0159，rec accuracy 下降约 0.0631。
   `tools/evaluate_onnx.py` 因此默认使用 `ORT_DISABLE_ALL`；`--ort-optimize` 仅作为问题复现和诊断。
4. 分别导出启用/关闭 `onnx_program.optimize()` 的 rec ONNX 后，在相同 ORT 设置下得到完全相同的
   全集 metric，排除 ONNXProgram optimize 为本次精度下降根因。

## 5. 验证命令

在原训练命令中保留相同模型、权重、profile、数据和输出目录，增加 checkpoint 与 `--eval-only`：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth \
  --resume output/icdar2015_ppocrv6_small_det_qat/best.pt \
  --label-file /home/heqi/dataset/icdr/train_icdar2015_label.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/test_icdar2015_label.txt \
  --output-dir output/icdar2015_ppocrv6_small_det_qat \
  --training-profile configs/qat/training/ppocrv6_small_det_baseline.yml \
  --device cuda:3 \
  --eval-only
```

识别模型将 `task/model-config/weights/checkpoint/label/profile/output` 替换为对应 rec 路径。

增加 `--eval-stage converted` 可在同一命令下验证 converted PT2E。固定真实样本的逐 tensor 对比：

```bash
$PYTHON tools/compare_qat_stages.py \
  --checkpoint output/icdar2015_ppocrv6_small_rec_qat/best.pt \
  --onnx output/icdar2015_ppocrv6_small_rec_qat/best_qdq.onnx \
  --samples 8 --batch-size 1 --device cpu --no-ort-optimize
```

QuantONNX 全验证集 metric：

```bash
$PYTHON tools/evaluate_onnx.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --onnx output/icdar2015_ppocrv6_small_rec_qat/best_qdq.onnx \
  --label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --data-dir /home/heqi/dataset/icdr \
  --batch-size 1 --workers 0 --no-ort-optimize
```

`--no-ort-optimize` 是当前默认值，命令中显式写出用于强调验收契约。

## 6. 测试和剩余门槛

```text
pytest: 38 passed
```

覆盖可变 polygon collate、ignore region、IoU/hmean、CTC 重复折叠、sequence accuracy/edit distance、
Trainer metric 合并、observer 恢复、checkpoint metric 状态、主指标选优、stage error accumulator 和
QuantONNX 静态 shape 检查。

本次没有完成 Pulsar2/Axera 编译和板端指标。下一项精度工作是补齐浮点 PyTorch 与 Axera 两端的同集
指标；ORT optimizer 的 QDQ 语义问题应独立处理，不能当作 Axera 转换结果。

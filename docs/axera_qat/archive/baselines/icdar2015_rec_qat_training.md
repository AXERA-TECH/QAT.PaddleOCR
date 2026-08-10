# ICDAR2015 识别 QAT 正式训练记录

## 1. 运行状态

```text
启动日期:   2026-07-30
完成日期:   2026-07-30
状态:       训练和 QuantONNX 导出完成
tmux:       ppocrv6-rec-qat-icdar
device:     cuda:3
训练轮数:   50 epochs
退出状态:   TRAIN_EXIT_CODE=0
输出目录:   output/icdar2015_ppocrv6_small_rec_qat
```

查看终端和日志：

```bash
tmux attach -t ppocrv6-rec-qat-icdar
tail -f output/icdar2015_ppocrv6_small_rec_qat/train.log
```

本次是正式多 epoch 训练，不覆盖先前的 1-epoch smoke：

```text
正式训练: output/icdar2015_ppocrv6_small_rec_qat/
旧 smoke: output/icdar2015_ppocrv6_small_rec_qat_smoke/
```

## 2. 数据、模型和配置

```text
dataset root: /home/heqi/dataset/icdr
train labels: rec_gt_train.txt，4468 images
val labels:   rec_gt_test.txt，2077 images

model:        PP-OCRv6_small_rec CTC-only
model YAML:   configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml
float weight: ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth
profile:      configs/qat/training/ppocrv6_small_rec_baseline.yml
QAT config:   configs/qat/ppocrv6_small_rec_u8s8.json
```

`rec_gt_train.txt` 第 2917 行原标签含 Tab，当前 encoder 会跳过字典外的 Tab，因此目标文本按
`relles` 编码。该已知数据问题保留原始标签文件，不在本轮静默修改数据。

## 3. 有效训练参数

```text
epochs:                 50
batch_size:             64
steps_per_epoch:        69（QAT drop_last）
workers:                8
image_shape:            [3, 48, 320]
optimizer:              Adam
initial_learning_rate:  2e-5
warmup_epochs:          2
lr_final_factor:        0.1
AMP:                    false
reparameterize:         true
observer_freeze_epoch:  None
QAT EMA:                false
validation indicator:   acc
```

> 本文件记录的是 2026-07-30 的历史训练运行。后续 P3 浮点保持性验证发现识别模型在 GPU 上
> 的 Conv-BN 重参数化存在数值差异；当前仍使用 `reparameterize: true`，优先保证 PT2E 和
> QuantONNX 结构符合 Axera 要求，再以重新训练后的完整指标判断 QAT 能否补偿。由于本历史运行
> 不符合当前 P0 数据和评估合同，其指标仍不纳入当前 baseline、delta 或门禁。

训练期间 observer 持续更新；validation 临时关闭 observer，结束后恢复。验证同时输出 CTC loss、
sequence accuracy 和 normalized edit similarity。`best.pt` 按最高 `acc` 选择，最低 validation loss
仍在 checkpoint 中独立记录。

## 4. 启动命令

创建会话：

```bash
tmux new-session -d -s ppocrv6-rec-qat-icdar \
  -c /home/heqi/project/PaddleOCR/route2/PaddleOCR2Pytorch-QAT
```

会话内执行：

```bash
mkdir -p output/icdar2015_ppocrv6_small_rec_qat
set -o pipefail

env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth \
  --label-file /home/heqi/dataset/icdr/rec_gt_train.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --output-dir output/icdar2015_ppocrv6_small_rec_qat \
  --training-profile configs/qat/training/ppocrv6_small_rec_baseline.yml \
  --device cuda:3 \
  2>&1 | tee output/icdar2015_ppocrv6_small_rec_qat/train.log

status=${PIPESTATUS[0]}
echo "TRAIN_EXIT_CODE=$status"
```

## 5. Checkpoint 和验收项目

每轮保存 `epoch_NNNN.pt`，并更新：

```text
last.pt  最新完整训练状态，用于严格恢复
best.pt  当前最高 validation accuracy
```

训练完成后必须依次执行：

1. 从 `best.pt` 严格恢复 prepared PT2E 并复测完整验证集；
2. 比较 prepared/converted CPU/CUDA 的 accuracy 和 edit similarity；
3. 导出 `best_qdq.onnx`，执行 checker、QDQ 结构和 fixed-sample 数值检查；
4. 使用 `tools/evaluate_onnx.py` 且 ORT graph optimization disabled 计算完整验证集指标；
5. 在 Pulsar2/Axera 环境完成编译和板端指标。

当前机器仍缺少 Pulsar2 和 PP-OCR 专用编译配置，因此第 5 项待工具链环境补齐。

## 6. 正式训练结果

50 个 epoch 均完成 69 个训练 step 和完整 2077 图 validation，无 NaN 或提前关闭 observer。关键轮次：

| Epoch | Learning rate | Train loss | Val loss | Val acc | Norm edit similarity |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0000e-5 | 0.64736 | 0.61994 | 0.72460 | 0.88304 |
| 2 | 2.0000e-5 | 0.54826 | 0.61271 | 0.72605 | 0.88894 |
| 3 | 2.0000e-5 | 0.50470 | **0.59571** | 0.72605 | 0.88799 |
| 4 | 1.9981e-5 | 0.46101 | 0.60141 | **0.73375** | 0.88931 |
| 5 | 1.9923e-5 | 0.44022 | 0.62196 | 0.72316 | 0.88467 |
| 6 | 1.9827e-5 | 0.41565 | 0.61890 | 0.72316 | 0.88351 |
| 7 | 1.9693e-5 | 0.40712 | 0.62804 | 0.71112 | 0.88135 |
| 10 | 1.9072e-5 | 0.35473 | **0.58734** | 0.72364 | 0.88785 |
| 15 | 1.7364e-5 | 0.29116 | 0.61022 | 0.72557 | **0.89021** |
| 25 | 1.2175e-5 | 0.20947 | 0.65538 | 0.72508 | 0.88854 |
| 40 | 4.2334e-6 | 0.16010 | 0.69628 | 0.71642 | 0.88569 |
| 50 | 2.0193e-6 | 0.14755 | 0.71990 | 0.71016 | 0.88415 |

`best.pt` 严格检查结果：

```text
epoch/global_step:             4 / 276
observers_frozen:              false
best_validation_metric_name:   acc
best_validation_metric_value:  0.7337506018295619
best_validation_loss:          0.5957072580966986（保存 epoch 4 时的历史最小值，来自 epoch 3）
```

这证明 metric 与 loss 的最优 epoch 已被独立跟踪：checkpoint 文件由最高 accuracy 的 epoch 4 产生。
训练结束后的完整结果为：

```text
best accuracy:             epoch 4, 0.7337506018295619
best accuracy val loss:    0.6014064636758782
minimum val loss:          epoch 10, 0.5873402116650884
maximum edit similarity:   epoch 15, 0.890205122005796
last epoch val accuracy:   0.7101588830043332
last epoch val loss:       0.7198966481920445
last epoch train loss:     0.14754967753222023
total steps:               3450
observers_frozen:          false
```

`best.pt` 的 metadata 为 epoch 4/global step 276，主指标为 `acc`，因此正式交付使用 epoch 4，而不是
最低 loss 或最高 edit similarity 对应的 checkpoint。

## 7. 正式 checkpoint 跨阶段指标

从同一个 `best.pt` 严格恢复并在完整 ICDAR2015 识别验证集（2077 张）上测试：

| Graph / backend | Accuracy | Norm edit similarity | Val loss |
|---|---:|---:|---:|
| prepared QAT / CUDA | 0.733751 | 0.889306 | 0.601407 |
| converted PT2E / CPU | 0.725566 | 0.886938 | 0.594975 |
| QuantONNX / CPU ORT optimize off | 0.719788 | 0.885338 | - |
| QuantONNX / CPU ORT optimize on | 0.684641 | 0.862039 | - |

prepared CUDA 是训练期间 validation 的参考结果；converted CPU 和 QuantONNX 使用独立执行后端，
不能把后端差异误认为训练 loss 的变化。QuantONNX 的 QDQ 语义基准使用 `ORT_DISABLE_ALL`，ORT
默认优化结果仅作为诊断项。

## 8. QuantONNX 导出结构

正式导出文件：

```text
output/icdar2015_ppocrv6_small_rec_qat/best_qdq.onnx
input shape:       [1, 3, 48, 320]
ONNX nodes:         682
QuantizeLinear:     161
DequantizeLinear:   280
BatchNormalization: 0
Concat shared:      1 / 1
HardSigmoid:        5 / 5
unquantized Conv:   0
exact redundant DQ/Q removed: 8
```

导出过程严格恢复 `best.pt`，执行 `convert_pt2e`、`onnx_program.optimize()`、精确等价的冗余 Q/DQ
删除、ONNX checker 和 ORT session 检查。没有在 ONNX 文件上手工补写 QDQ；识别模型的量化域来自
prepared PT2E 图。

在 CPU 上对固定前 8 张验证图片执行 prepared、converted 和 QuantONNX 对比：

```text
prepared -> converted CTC sequence agreement: 1.000000
prepared -> converted argmax agreement:       0.996875
converted -> ONNX CTC sequence agreement:     1.000000
converted -> ONNX argmax agreement:           0.996875
converted -> ONNX logits MAE:                 1.003618
converted -> ONNX probability MAE:            6.60916e-7
all outputs finite:                            true
```

当前受限执行环境没有可用 CUDA，因此 2026-08-03 的 fixed-sample 复核使用 CPU；正式训练和 prepared
全集指标来自原 `cuda:3` 训练环境。

## 9. 当前结论和待办

| 检查项 | 状态 | 结论 |
|---|:---:|---|
| ICDAR2015 50-epoch QAT | 通过 | 3450 steps，无 NaN，observer 未提前关闭 |
| checkpoint 严格重载 | 通过 | `best.pt` 可还原 prepared PT2E 图和量化状态 |
| 任务指标选优 | 通过 | 按 `acc` 选择 epoch 4，loss/edit 独立记录 |
| QuantONNX 导出 | 通过 | checker、QDQ 结构和 ORT session 通过 |
| QuantONNX metric | 通过 | ORT optimize off acc 0.719788；优化 on 0.684641 |
| Pulsar2/Axera 编译 | 待完成 | 当前机器缺少工具和 PP-OCR 编译配置 |

当前结果证明 PP-OCRv6 small rec CTC 的训练、checkpoint、prepared/converted 评估和 QuantONNX 主链路
可用。仍需在具备 Pulsar2 的环境完成 Axera 编译、板端推理及板端指标回归。

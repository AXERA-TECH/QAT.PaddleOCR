# PP-OCRv5 Mobile Rec ICDAR2015 QAT 训练记录

> 2026-08-05 路线更新：本文记录的是历史 CTC-only QAT 图。关闭 `use_guide` 的 corrected 实验用于
> 修复当时的 backbone 无梯度问题，但不等价于 Paddle pretrained 的 `CTC + NRTR` 完整训练结构。
> 后续实现和训练以
> [PP-OCR Pretrained 训练结构复现计划](../plans/pretrained_training_structure_plan.md) 为准；本文指标不得
> 作为 full MultiHead 训练 baseline，也不得用于恢复新图 checkpoint。

状态：50 epoch 已完成；训练链路通过，精度恢复未通过。

## 1. 训练合同

2026-08-05 在原生 SiLU QuantONNX 结构通过用户检查、完整浮点精度和 PT2E fake-off 验证完成后，
从转换后的浮点权重启动正式 QAT。没有复用任何旧 prepared/QAT checkpoint。

```text
model:     configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml
weights:   ptocr_v5_mobile_rec.pth
qat:       configs/qat/ppocrv5_mobile_rec_u8s8_attn_s8_native_silu.json
profile:   configs/qat/training/ppocrv5_mobile_rec_native_silu_baseline.yml
train:     /home/heqi/dataset/icdr/rec_gt_train.txt，4468 张
val:       /home/heqi/dataset/icdr/rec_gt_test.txt，2077 张
input:     [3,48,320]
device:    cuda:1
```

## 2. 参数

```text
epochs:                  50
batch size:              64
steps/epoch:             69
learning rate:           2e-5
warmup:                  2 epochs
final LR factor:         0.1
AMP:                     false
reparameterize:          true
observer freeze epoch:   null
QAT EMA:                 false
random augmentation:     disabled
checkpoint interval:     1 epoch
```

observer 在训练期间保持开启；validation 临时关闭 observer，结束后恢复。没有提前 freeze observer。

## 3. 启动命令

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights ptocr_v5_mobile_rec.pth \
  --label-file /home/heqi/dataset/icdr/rec_gt_train.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --val-data-dir /home/heqi/dataset/icdr \
  --output-dir output/icdar2015_ppocrv5_mobile_rec_native_silu_qat \
  --training-profile configs/qat/training/ppocrv5_mobile_rec_native_silu_baseline.yml \
  --device cuda:1
```

```text
tmux:   ppocrv5-mobile-rec-native-silu-qat
log:    /tmp/ppocrv5_mobile_rec_native_silu_qat.log
output: output/icdar2015_ppocrv5_mobile_rec_native_silu_qat/
```

## 4. 训练前基线

```text
普通 float accuracy:               0.5936446769684897
重参数化 eager/exported accuracy:  0.5941261434761675
prepared fake-off accuracy:        0.5936446798266731
32 图 observer fake-on accuracy:   0.34569090033702454
32 图 observer converted accuracy: 0.2932113625421281
```

fake-on/converted 是未训练 QAT 图的起点，不作为最终精度。对应分层报告：

```text
artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_pt2e_fake_off_full_gpu2_20260805.json
artifacts/accuracy_baseline/p5_quant/ppocrv5_mobile_rec_quant_stages_full_gpu2_20260805.json
```

## 5. 50-Epoch 训练结果

50 个 epoch 全部完成，每个 epoch 为 69 个训练 step 和完整 2077 图 validation，共 3450 step。
训练过程无 NaN/Inf，observer 全程保持开启；validation 临时关闭 observer、保持 fake quant 开启。
关键轮次如下：

| Epoch | LR | Train loss | Val loss | Accuracy | Norm edit similarity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0e-5 | 7.02745 | 7.10706 | 0.00048 | 0.09788 |
| 5 | 1.9923e-5 | 5.13176 | 5.00048 | 0.01589 | 0.26032 |
| 9 | 1.9315e-5 | 3.96204 | 3.90818 | 0.08859 | 0.35902 |
| 15 | 1.7364e-5 | 3.54626 | 3.10193 | 0.07752 | 0.35363 |
| 30 | 9.2442e-6 | 3.53496 | 3.06554 | 0.06066 | 0.31069 |
| 41 | 3.8598e-6 | 3.49051 | 3.03131 | 0.05007 | 0.30262 |
| 47 | 2.3067e-6 | 3.76222 | 2.98711 | 0.04815 | 0.27674 |
| 48 | 2.1729e-6 | 3.52044 | 3.03370 | 0.06307 | 0.28765 |
| 50 | 2.0193e-6 | 4.31028 | 4.90273 | 0.00000 | 0.04322 |

汇总：

```text
best validation accuracy:       0.08858931150698122  (epoch 9)
best normalized edit similarity: 0.3590218867421564  (epoch 9)
minimum validation loss:        2.9871110121409097   (epoch 47)
final validation accuracy:      0.0                  (epoch 50)
final normalized edit similarity: 0.043219757025726335
final validation loss:          4.90273143305923
```

`best.pt` 按 validation 主指标 accuracy 保存，与日志最大值一致：

```text
checkpoint epoch:               9
checkpoint global step:         621
best metric name:               acc
best metric value:              0.08858931150698122
observers frozen:               false
```

`last.pt` 对应 epoch 50/global step 3450，并记录全程最低 val loss `2.9871110121409097`。最低 loss
与最高 sequence accuracy 不在同一轮，不能使用 epoch 47 替代按 accuracy 选择的 `best.pt`。

## 6. 与浮点基线比较

| 阶段 | Accuracy | Norm edit similarity |
| --- | ---: | ---: |
| Float 原生 SiLU | 0.59364468 | 0.81712050 |
| QAT best / epoch 9 | 0.08858931 | 0.35902189 |
| 绝对变化 | -0.50505537 | -0.45809861 |

本轮 QAT 工程链路完成，但精度没有恢复到浮点基线，且 epoch 间波动明显。该结果不能作为可部署
QAT 模型验收通过的证据。当前只记录训练期间 prepared/fake-quant-on validation 指标；尚未对
`best.pt` 执行后训练 strict reload、converted PT2E、QuantONNX 或 Axera 指标验证。

## 7. 训练产物

目录：

```text
output/icdar2015_ppocrv5_mobile_rec_native_silu_qat/
```

主要文件：

```text
best.pt             epoch 9，按最高 validation accuracy 保存
last.pt             epoch 50，最终训练状态
epoch_0001.pt ... epoch_0050.pt
train.log           50 行逐 epoch JSON 训练日志
```

下一阶段固定从 `best.pt` 开始：strict reload 后复测 prepared 指标，再评估 converted PT2E；只有
这些结果明确后才导出训练后 QuantONNX。本节不混入初始化 observer QuantONNX 的结构检查结果。

## 8. 训练代码审计（2026-08-05）

### 8.1 数据与标签

route2 的 `RecognitionDataset` 与 PaddleOCR `CTCLabelEncode + RecResizeImg` 已逐项对照：

```text
输入颜色:            BGR，一致
resize:              ceil(48 * width / height)，最大宽 320，一致
padding:             右侧补归一化空间的 0，一致
归一化:              [0,255] -> [-1,1]，一致
CTC blank:           index 0，一致
unknown character:   跳过该字符，一致
valid_ratio:         resized_width / 320，一致
```

route2 与 PaddleOCR 的 `ppocrv5_dict.txt` 内容对齐，共 18383 行。ICDAR2015 当前标签统计：

```text
train: 4468，文本长度 2..19，均值 5.335
val:   2077，文本长度 2..21，均值 5.326
CTC 所需最大 time steps（含连续重复字符）: train 19，val 23
模型 CTC time steps: 40
```

不存在目标长度超过 CTC 输出长度的样本，`zero_infinity=True` 没有在该数据集上静默屏蔽样本。
当前 route2 有意关闭 PaddleOCR 原始 `RecConAug/RecAug` 和多尺度 sampler，固定使用 `[3,48,320]`；
这是既定 QAT baseline 约束，不是本轮精度异常。

### 8.2 CTC loss reduction 修正

PaddleOCR 使用：

```python
nn.CTCLoss(reduction="none")
loss = loss.mean()
```

原 route2 使用 PyTorch `reduction="mean"`。PyTorch 的 `mean` 会先将每条样本 loss 除以
`target_length`，再执行 batch mean，语义不等价。固定张量跨框架结果：

```text
Paddle none:       [2.09566879, 2.85229540]
Torch none:        [2.09566879, 2.85229540]
Paddle/Torch none 后 batch mean: 2.47398210
Paddle/Torch reduction=mean:     0.99929976
```

`pytorchocr/training/losses/ctc.py` 已改为 `none -> batch mean`，并增加固定张量单测。
该问题改变不同文本长度样本的梯度权重，但 observer-only 无 backward 实验同样崩溃，因此它不是
observer 漂移的单独根因。

### 8.3 CTC-only 图冻结了 backbone

PaddleOCR v5 原始训练图使用 `CTC + NRTR` MultiLoss。CTC neck 配置 `use_guide=True`，会截断
CTC loss 到 backbone 的梯度；原始训练依赖 NRTR 辅助分支更新 backbone。route2 为保持部署图只捕获
CTC 分支，转换权重也不包含 NRTR 参数，因此当前 QAT 中 backbone 没有任何训练梯度：

```text
总参数张量:                    221
use_guide=True 有梯度参数:      43
use_guide=True backbone 梯度:    0
use_guide=False 有梯度参数:     221
use_guide=False backbone 梯度:   178
```

固定相同 dropout seed 时，`use_guide=True/False` 的 forward 输出逐元素完全相等，差异只在反向传播。
关闭 guide detach 后，现有 Axera regional QAT 配置仍能全部匹配：float/prepared 节点从
`500/984` 变为 `499/982`，observer 从 382 变为 381，仅删除 `detach` 及其 observer。

因此当前训练只能让 SVTR neck 和 CTC FC 适应量化，无法让 backbone 权重恢复全局 U8 量化误差。
后续 corrected smoke 应在训练捕获合同中显式关闭 CTC guide detach，不应把部署无关且缺失预训练权重的
NRTR 分支加入 QuantONNX。

### 8.4 Observer 与 BN 因果拆分

checkpoint 拆分报告：

```text
artifacts/accuracy_baseline/p5_quant/
ppocrv5_mobile_rec_qat_checkpoint_audit_256_gpu2_20260805.json
```

固定前 256 张验证图：

| Checkpoint | fake-off acc | fake-on acc | 初始权重 + checkpoint observer acc |
| --- | ---: | ---: | ---: |
| epoch 1 | 0.359375 | 0.003906 | 0.000000 |
| best / epoch 9 | 0.484375 | 0.074219 | 0.015625 |
| last / epoch 50 | 0.183594 | 0.000000 | 0.000000 |

epoch 1 在没有参数更新的 observer-only 组合上已经崩溃，证明 qparam 是首轮主要退化来源；epoch 9
的 fake-off 参数图反而恢复到 `0.484375`，说明 neck/head 参数确实在尝试补偿量化误差。

新增 69-batch train/eval、fake-on/off observer-only 对照：

```text
artifacts/accuracy_baseline/p5_quant/
ppocrv5_mobile_rec_observer_modes_69b_256_gpu2_20260805.json
```

| Observer 更新方式 | 更新时 backward | 更新后 fake-on acc | Blank argmax |
| --- | --- | ---: | ---: |
| 32 图 eval / fake-on 初始化 | 无 | 0.414062 | 0.892871 |
| 69 batch train / fake-on | 无 | 0.000000 | 0.911133 |
| 69 batch eval / fake-on | 无 | 0.000000 | 0.999902 |
| 69 batch train / fake-off | 无 | 0.019531 | 0.952539 |
| 69 batch eval / fake-off | 无 | 0.019531 | 0.972852 |

结论：train-mode BN/Dropout 不是唯一原因；即使 eval mode 且无 backward/optimizer，observer 覆盖完整
训练集范围后也发生 blank collapse。train mode 还会显著改写五个 SVTR neck BN 的 running stats，
例如 `conv1.norm.running_var max_abs=11364.64`。漂移最大的 activation observer 位于第二个 SVTR
block 的 `mlp.fc2/dropout/add`、CTC FC 以及 neck BN/SiLU 边界。较宽训练集范围造成 scale 扩张，而
backbone 又因 guide detach 无法训练，是当前不能恢复精度的组合原因。

validation 还会使 PT2E training IR 中独立的 `num_batches_tracked += 1` 执行。running mean/var 不在
validation 更新，但计数器会污染 checkpoint。`Trainer.evaluate()` 已保存并恢复这些计数器，新增测试通过。

### 8.5 Optimizer 与可复现性差异

当前 profile 与 PaddleOCR 原始训练配置仍有差异：

```text
当前 QAT:      Adam, lr 2e-5, warmup 2, weight_decay 0
Paddle 原始:   Adam, lr 5e-4, warmup 5, L2 3e-5
CTC FC:        Paddle 另设 fc_decay 1e-5
LAB 参数:      Paddle PPLCNetV3 使用 0.1 learning-rate multiplier
```

在 guide detach 未修正前 LAB/backbone 参数没有梯度，因此 LAB LR multiplier 不是本轮直接原因；打开
backbone 梯度后需要按 parameter group 恢复。训练入口当前也没有固定 Python/NumPy/Torch/DataLoader
随机种子，首个 shuffled batch 会主导未初始化 moving-average observer，导致同配置结果明显波动。

## 9. 同配置后台复现

按用户要求，在排查期间使用原 QAT JSON/profile 和旧 CTC reduction 语义重新完成 50 epoch。该进程在
CTC 修正前已加载 Python 模块，因此它是原始 baseline 的独立复现，不与 corrected 实验混用。

```text
output: output/icdar2015_ppocrv5_mobile_rec_native_silu_qat_repro_20260805/
log:    /tmp/ppocrv5_mobile_rec_native_silu_qat_repro_20260805.log
best:   epoch 45 / global step 3105
best accuracy:               0.16225324987963408
best normalized edit:        0.46710134539937165
best validation loss:        2.3526831323450264
epoch 50 accuracy:           0.0182956186807896
epoch 50 normalized edit:    0.2198784898833036
epoch 50 validation loss:    3.5926749417276094
```

上一轮同配置最佳 accuracy 为 `0.08858931`，本轮为 `0.16225325`，且最佳 epoch 从 9 变为 45。
两轮都远低于 float `0.59364468` 且后期回落，不能验收；差异进一步证明后续所有 smoke/正式训练必须
记录并固定 seed。

## 10. Corrected Smoke 前置项

下一次正式训练前按以下顺序执行，不直接复用以上 checkpoint：

1. 在训练合同中固定 seed，并记录到 checkpoint metadata；
2. 使用已修正的 Paddle-compatible CTC reduction；
3. 仅对训练捕获关闭 CTC guide detach，使 backbone 获得梯度；
4. 恢复全局 L2、CTC FC decay 和 LAB LR multiplier parameter groups；
5. 先跑一轮 real-data smoke，复测 fake-off/fake-on、strict reload 和 QuantONNX 结构；
6. 用户检查 smoke QuantONNX 后，再决定 observer 初始化策略和正式 epoch 数。

## 11. Corrected 1-Epoch Smoke

完成第 10 节前五项的代码实现，并使用独立 profile 执行一轮 real-data smoke：

```text
profile: configs/qat/training/ppocrv5_mobile_rec_native_silu_corrected_smoke.yml
output:  output/icdar2015_ppocrv5_mobile_rec_native_silu_qat_corrected_smoke/
device:  cuda:2
seed:    20260805
```

该 smoke 继续使用同一个 Axera QAT JSON，observer 训练期保持开启，不做 eager Conv-BN 固定融合。
训练合同变化仅包括：Paddle-compatible CTC reduction、CTC backbone gradient、固定 seed 和 Paddle
optimizer parameter groups。实际 optimizer groups：

| Group | 参数张量 | Epoch 1 LR | Weight decay |
| --- | ---: | ---: | ---: |
| default | 107 | 2e-5 | 3e-5 |
| LAB | 112 | 2e-6 | 3e-5 |
| CTC FC | 2 | 2e-5 | 1e-5 |

首轮训练与完整验证：

```text
steps:                   69
train CTC loss:          19.349967514259227
prepared val CTC loss:   14.164839108784994
prepared val accuracy:   0.24891670678863745
prepared val norm edit:  0.573328392839707
converted val CTC loss:  12.826198736826578
converted val accuracy:  0.23976889744824265
converted val norm edit: 0.5472116806854409
```

strict reload `best.pt` 后的 2077 图 prepared 指标与训练结束逐项一致。旧合同 epoch 1 accuracy 约为
`0.00048`，corrected smoke 为 `0.24892`，说明恢复 backbone 梯度、loss 语义和 optimizer groups 的方向
有效；但一轮 smoke 仍不是正式精度结论。

首次 smoke 日志中的单值 `learning_rate=2e-6` 取到了第一个 LAB group；checkpoint 内三组 LR 正确。
日志代码已改为同时输出 default `learning_rate` 和完整 `learning_rates` 映射，后续运行不会再混淆。

### 11.1 Smoke QuantONNX

导出文件：

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/
ppocrv5_mobile_rec_native_silu_backbone_grad_corrected_smoke_20260805.onnx
```

```text
size:                   5.5 MiB
input:                  [1,3,48,320]
output:                 [1,40,18385]
float/prepared/converted nodes: 499 / 982 / 1340
ONNX nodes:             943
QuantizeLinear:         244
DequantizeLinear:       429
Q zero-point dtype:     U8 224 / S8 20
Concat shared domain:   1 / 1
BatchNormalization:     0
direct/redundant DQ-Q:  0 / 0
requantize DQ-Q:        0
SiLU internal QDQ:      0
unquantized Conv output: 0
```

结构门禁通过，但数值门禁未通过，不能标记为最终可部署模型：

```text
prepared -> converted（GPU2, 8 图）:
  logits MAE/max:       8.43317 / 90.95215
  argmax agreement:     0.953125
  CTC sequence agree:   0.25

converted -> ORT_DISABLE_ALL（8 图）:
  logits MAE/max:       1.95642 / 37.17175
  argmax agreement:     0.96875
  CTC sequence agree:   0.625
```

该问题不是 corrected guide 图新引入：训练前历史报告中 fake-on -> converted 已有 logits
`MAE=1.13814`、argmax `0.96577`、完整 accuracy `0.34569 -> 0.29321`。corrected smoke 的完整
prepared/converted accuracy delta 为 `-0.00915`，较 raw logits 差异温和，但仍需独立定位 PT2E
Conv-BN convert 和 ONNX quantized-decomposed lowering；正式训练不能替代这项结构/数值修复。

补充转换顺序 A/B：同一个 prepared checkpoint 分别在 training IR 状态直接 `convert_pt2e`，以及先
`move_exported_model_to_eval` 再 `convert_pt2e`，两种 converted 输出逐元素完全相等。因此当前 parity
差异不是 `move_exported_model_to_eval` 调用顺序造成，公共转换函数暂不修改。

## 12. Corrected 正式训练（关闭 warmup）

2026-08-05 按要求使用修复后的训练代码重新训练。该实验从原始浮点权重开始，不恢复 corrected smoke
或任何历史 prepared/QAT checkpoint；observer 在训练期保持开启，validation 时临时关闭。

```text
status:          completed（TRAIN_EXIT_CODE=0）
profile:         configs/qat/training/ppocrv5_mobile_rec_native_silu_corrected_no_warmup.yml
output:          output/icdar2015_ppocrv5_mobile_rec_native_silu_qat_corrected_no_warmup_20260805/
log:             /tmp/ppocrv5_mobile_rec_native_silu_qat_corrected_no_warmup_20260805.log
tmux:            ppocrv5-rec-corrected-nowarmup
device:          cuda:2
epochs:          50
batch size:      64
initial LR:      2e-5（default/CTC FC），2e-6（LAB）
warmup epochs:   0
final LR factor: 0.1
seed:            20260805
backbone grad:   enabled
CTC reduction:   none 后 batch mean
observer freeze: null
```

启动命令：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights ptocr_v5_mobile_rec.pth \
  --label-file /home/heqi/dataset/icdr/rec_gt_train.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --val-data-dir /home/heqi/dataset/icdr \
  --output-dir output/icdar2015_ppocrv5_mobile_rec_native_silu_qat_corrected_no_warmup_20260805 \
  --training-profile configs/qat/training/ppocrv5_mobile_rec_native_silu_corrected_no_warmup.yml \
  --device cuda:2
```

50 个 epoch 共 3450 step，训练期间没有 NaN/Inf，observer 始终保持开启。关键指标：

| Epoch | Default LR | Train loss | Val loss | Accuracy | Norm edit similarity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.0000e-5 | 19.34997 | 14.16484 | 0.24892 | 0.57333 |
| 13 | 1.7561e-5 | 6.34456 | 6.04797 | 0.46461 | 0.74046 |
| 23 | 1.2686e-5 | 5.55515 | 5.45211 | 0.48387 | 0.76081 |
| 36（best acc） | 5.7099e-6 | 5.40799 | 5.45612 | **0.49350** | 0.75980 |
| 38（best loss/edit） | 4.8391e-6 | 5.23453 | **5.44881** | 0.48917 | **0.76527** |
| 50（last） | 2.0178e-6 | 5.71670 | 6.07476 | 0.43187 | 0.72480 |

`best.pt` 在独立进程中从相同浮点权重和 QAT 图严格恢复后，使用 `cuda:2`、`workers=0` 完整验证
2077 张图，结果与 epoch 36 训练内验证逐项一致：

```text
epoch/global step:  36 / 2484
prepared val loss:  5.456118515043547
prepared accuracy:  0.49350024073182475
prepared norm edit: 0.759802848945844
```

训练首次验证的 default LR 已是 `2e-5`，说明没有执行 warmup；LAB 参数组从 `2e-6` 开始。与旧异常
50-epoch 复现的 best accuracy `0.16225` 相比，本轮提高到 `0.49350`，但仍低于浮点模型
`0.59364`，绝对差约 `0.10014`，因此当前结果不能标记为精度验收通过。

产物：

```text
9ecd25535b6e5b9f6c81e382e6e3fe9aad47a7170867a855e7e904ed2d4b3e9a  best.pt
81ed7672b3bc67eaae9e611a939e19bc528e97243bcf730d3bfc77d58bd353b0  last.pt
720e5d23cbd53038b9f33388702e54030be726fa62db14befde8e7804180923b  corrected no-warmup profile
```

严格恢复验证日志：

```text
/tmp/ppocrv5_mobile_rec_native_silu_qat_corrected_no_warmup_best_reload_eval_20260805.log
```

### 12.1 Checkpoint 清理

2026-08-05 项目结构整理前完成空间回收。该 run 仅保留 epoch 36、accuracy `0.49350` 的
`best.pt`；`epoch_0001.pt` 至 `epoch_0050.pt` 和 `last.pt` 已删除。

训练指标、profile、QAT JSON、严格恢复日志和文档均保留。旧异常 baseline/repro 的 checkpoint 已
全部删除，只保留历史指标，后续不得从这些旧合同恢复训练。

## 13. SGD 与动态高度新合同

2026-08-06 开始下一轮 PP-OCRv5 mobile rec QAT。该轮不修改历史 corrected no-warmup profile，新增：

```text
configs/qat/training/ppocrv5_mobile_rec_native_silu_sgd_dynamic_height.yml
```

新合同相对第 12 节的变化：

- optimizer 从 Adam 改为 SGD，momentum 为 `0.9`；
- 保持初始 default/CTC FC LR `2e-5`、LAB LR `2e-6`、cosine 和 no-warmup；
- 训练数据预处理仍固定为 `3x48x320`；
- `export_for_training` 同时设置 dynamic batch 和 derived height
  `16 * height_factor`，其中 `height_factor` 范围为 `[2, 4]`，因此只接受 H=32/48/64；
- optimizer、momentum 和 `dynamic_heights` 写入 checkpoint metadata/resume contract；
- checkpoint strict reload 从 metadata 重建同一个 prepared 动态训练图；
- smoke 对 H=32/48/64 执行 prepared forward；convert 和 QuantONNX 只使用静态 `3x48x320`；
- ONNX 导出后显式检查并固化静态 CHW，动态高度不能传播到 QuantONNX 输入。

已完成 PT2E prepared 探针，同一图的结果为：

```text
float nodes: 499
prepared state entries: 3129
batch=1, H=32 -> [1, 40, 18385]
batch=2, H=48 -> [2, 40, 18385]
batch=3, H=64 -> [3, 40, 18385]
```

这次图合同变化后不得恢复第 12 节的 Adam/static-height `best.pt`。正式训练仍遵循 smoke
QuantONNX 先交付结构检查的门禁。

2026-08-06 smoke 已完成。训练 prepared 图分别执行 H=32/48/64，三档输出均为
`[1,40,18385]`；QuantONNX 只使用 profile 基准尺寸导出，输入为静态 `[1,3,48,320]`，不存在
动态高度符号。

```text
artifact: /tmp/ppocrv5_mobile_rec_sgd_dynamic_train_static_h48_qat_smoke_20260806.onnx
size:     5.4 MiB

float/prepared/converted nodes: 500 / 982 / 1339
gradient tensors:               43
ONNX input:                     [1, 3, 48, 320]
ONNX output:                    [1, 40, 18385]
PyTorch -> ORT argmax agreement: 1.0
PyTorch -> ORT probability MAE:  1.1897581e-5
```

QuantONNX 结构验证结果：`QuantizeLinear=243`、`DequantizeLinear=429`，Concat 唯一量化域共享
qparams，7 个 SiLU 与 2 个标准 hard activation 均已量化，未发现未量化 Conv 输出，也没有残留
direct/requant DQ -> Q。ORT 验证使用 `ORT_DISABLE_ALL`，输出有限。本产物仅为初始化 observer 的
smoke 图，正式训练仍需等待结构人工检查通过后再启动。

## 14. 全局 U16/S16 与 Attention 连续 S16 合同

2026-08-06 在 SGD/dynamic-height 训练合同上更新量化位宽。全局 activation 为 U16，所有量化权重为
S16；两个 SVTR Attention 不能只继承全局配置，必须显式保持以下连续 S16 域：

```text
QKV Linear: U16 -> S16, weight S16
scale Mul:  S16 -> S16
MatMul 1:   S16/S16 -> S16
Softmax:    S16 -> S16
MatMul 2:   S16/S16 -> U16
```

当前合同文件：

```text
configs/qat/base_u16s16.json
configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json
configs/qat/training/ppocrv5_mobile_rec_u16s16_sgd_dynamic_height.yml
artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_global_u16s16_qat_discovery_20260806.json
```

旧 U8/S8 配置和历史 native-SiLU 训练 profile 已删除，只在历史章节中保留路径作为问题来源记录；
不得从第 12、13 节旧 checkpoint 恢复到当前图。当前 smoke 从浮点权重重新 prepare，QuantONNX 为：

```text
/tmp/ppocrv5_mobile_rec_global_u16s16_attn_s16_qat_smoke_20260806.onnx
input/output: [1,3,48,320] -> [1,40,18385]
Q/DQ: 243 / 429
activation Q zero-point dtype: U16 223 / S16 20
quantized rank>=2 weight dtype: S16 47
direct/requant DQ->Q: 0 / 0
argmax agreement: 1.0
probability MAE/max: 3.3771138e-7 / 0.006369978
```

唯一 Identity-isolated qparam 边界位于 `AveragePool -> Conv`，在 Attention 外部。Pulsar2 静态配置
生成与校验通过，识别到两个 Attention、7 个 SiLU 和一个必要 Pool/Conv qparam 边界。当前阶段只完成
初始化 observer smoke；继续正式训练前仍需用户审核 QuantONNX 结构。

配置发现 skill 随后补齐 8/16-bit 兼容，但没有改变上述 v5-rec 配置。`--attention-dtype auto` 根据
全局 activation 将 U8 解析为 S8、U16 解析为 S16；显式 S8/S16 仍用于混合位宽对照。真实图验证：

```text
PP-OCRv5 mobile rec + U16/S16 base: auto -> S16，--check PASS
PP-OCRv6 small rec + U8/S8 base:    auto -> S8，发现 2 个 Attention
/tmp/ppocrv6_small_rec_u8s8_auto_skill_20260806_v2.json
/tmp/ppocrv6_small_rec_u8s8_auto_skill_20260806_v2.report.json
```

16-bit QuantONNX 同时通过更新后的 `tools/verify_qat_onnx.py -c`：0 error、0 warning；验证器现在识别
S16 per-channel Conv/Linear 权重及 `DQ(weight) -> Transpose -> MatMul -> Add -> Q` Linear lowering，
并保留 S8 回归覆盖。

本轮最终验证：QAT discovery skill 与 Pulsar2 skill 均通过 `quick_validate.py`；项目测试为
`144 passed, 14 subtests passed`，`git diff --check` 通过。4 条 warning 分别来自 NVML 不可用、
same-padding 内部 padding，以及初始化 observer 的默认 qparams，不是测试失败。

## 15. 数据增强迁移与真实多尺度 DataLoader

2026-08-06 从 `references/PytorchOCR/torchocr/data/` 迁移 det/rec 增强和 MultiScaleSampler。当前
v5-rec QAT 合同显式保持 `augmentation: none`，因此 RecAug/RecConAug 不参与 baseline；同时设置
`multi_scale_training: true`，修复此前 `dynamic_heights` 只用于 PT2E 图捕获、DataLoader 始终 H=48
的问题。以 H=48、batch 64 为基准，H=32/48/64 对应 batch 96/64/48。

`augmentation` 和 `multi_scale_training` 已加入 checkpoint resume contract。该变化改变训练数据和
checkpoint 合同，不能恢复第 14 节之前的 prepared checkpoint。QuantONNX 仍固定 H=48。实现和差异
见 `docs/architecture/data_augmentation.md`；定向 dataset/profile 测试为 `34 passed`。尚未启动正式
训练，也未声明启用增强后的任务精度。

真实 `/home/heqi/dataset/icdr/rec_gt_train.txt` DataLoader 检查得到
`[96,3,32,320]`、`[64,3,48,320]`、`[48,3,64,320]` 三档 batch。最终项目测试为
`151 passed, 14 subtests passed`，4 条 warning 与前述既有环境/observer warning 相同。

## 16. Exp1 全局 U16/S16 正式训练

2026-08-06 使用用户调低后的初始学习率 `1.5e-5` 启动正式训练。权重文件是完整 MultiHead 浮点权重，
但当前 QAT 合同的 `rec_graph` 为 `deploy`，实际 prepare 的是 CTC-only 推理/部署图，NRTR 辅助头没有
进入 PT2E 图。该实验不恢复第 1 至 15 节的任何 prepared/QAT checkpoint。训练目录按新约定使用
`exp[数字]` 前缀：

```text
run:     runs/exp1_ppocrv5_mobile_rec_u16s16_qat
tmux:    ppocrv5-rec-u16s16-exp1
log:     runs/exp1_ppocrv5_mobile_rec_u16s16_qat/train.log
device:  physical GPU 2; CUDA_VISIBLE_DEVICES=2; process device cuda:0
status:  completed (50/50 epochs)
```

启动命令：

```bash
env PYTHONPATH=/home/heqi/project/PaddleOCR \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=2 \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --label-file /home/heqi/dataset/icdr/rec_gt_train.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --val-data-dir /home/heqi/dataset/icdr \
  --output-dir runs/exp1_ppocrv5_mobile_rec_u16s16_qat \
  --training-profile configs/qat/training/ppocrv5_mobile_rec_u16s16_sgd_dynamic_height.yml \
  --device cuda:0
```

训练合同：SGD、momentum `0.9`、50 epoch、no-warmup、observer 全程开启、增强关闭。训练启用
H=32/48/64 的真实多尺度 batch，分别为 96/64/48；QuantONNX 仍固定 H=48。数据集为 4468 张训练图
和 2077 张验证图。

阶段指标：

| Epoch | Default LR | Train loss | Val loss | Accuracy | Norm edit similarity | Observer frozen |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 1.5000e-5 | 38.71489 | 82.82874 | 0.00000 | 0.00000 | false |
| 2 | 1.4987e-5 | 28.19450 | 34.13306 | 0.00000 | 0.02477 | false |
| 3 | 1.4947e-5 | 26.81435 | 35.56046 | 0.00000 | 0.00827 | false |
| 4 | 1.4880e-5 | 24.63260 | 28.51519 | 0.00000 | 0.00253 | false |
| 5 | 1.4788e-5 | 24.20464 | 29.40712 | 0.00000 | 0.00403 | false |
| 6 | 1.4670e-5 | 23.87847 | 26.82940 | 0.00000 | 0.00430 | false |
| 7 | 1.4526e-5 | 25.13287 | 34.62297 | 0.00000 | 0.05487 | false |
| 8 | 1.4358e-5 | 26.29036 | 36.36643 | 0.00000 | 0.00239 | false |
| 9 | 1.4165e-5 | 25.15114 | 69.33011 | 0.00000 | 0.04600 | false |
| 10 | 1.3949e-5 | 23.67914 | 55.20395 | 0.00000 | 0.04250 | false |

| 25 | 8.6738e-6 | 22.65520 | 78.17298 | 0.00000 | **0.05496** | false |
| 50 | 1.5133e-6 | 23.53904 | 29.61370 | 0.00000 | 0.00000 | false |

50 个 epoch 全部完成，共 3250 个训练 step；无 NaN/Inf，observer 全程保持开启。全程 validation
accuracy 为 `0.0`，最高 normalized edit similarity 仅为 `0.05496118`（epoch 25），因此本轮训练
不通过精度验收，也不导出部署 QuantONNX。

该轮 metadata 明确记录 `rec_graph=deploy`、`head_schema=[ctc]`、`loss_schema=[CTCLoss]`。后续如需
验证完整 Paddle pretrained 训练结构，必须另建 `pretrained_train` 浮点训练合同，不能把本轮 checkpoint
当作 CTC+NRTR 完整训练产物。

## 17. 浮点权重 PT2E 图保持与零校准转换

2026-08-06 使用同一份浮点权重、同一 U16/S16 配置和静态 H=48，在 GPU 3 上完成全量 2077 张验证图。
该测试不执行 activation observer 校准；prepare 完成后立即关闭 observer 和 fake-quant。模型角色为
`rec_graph=deploy` 的 CTC-only 推理图。

### 17.1 Prepared fake-off

```text
artifact: artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_u16s16_prepared_fake_off_full_gpu3_20260806.json
eager accuracy:           0.5936446798266731
exported accuracy:        0.5936446798266731
prepared fake-off acc:    0.5936446798266731
prepared fake-off edit:   0.8169468283815902
eager -> prepared logits MAE/max: 4.9817430e-6 / 0.001947403
argmax agreement:         1.0
CTC sequence agreement:   1.0
```

因此 `export_for_training` + `prepare_qat_pt2e` 的浮点保持路径正常，prepared 本身没有造成精度损失。

### 17.2 Converted zero-activation-calibration

为使 `convert_pt2e` 能处理 S16 per-channel 权重，只初始化了 80 个静态权重 observer；没有执行任何
输入激活 observer 更新。这不是 PTQ 校准。结果：

```text
artifact: artifacts/accuracy_baseline/p5_quant/ppocrv5_mobile_rec_u16s16_no_activation_calibration_converted_gpu3_20260806.json
prepared fake-off accuracy: 0.5936446798266731
converted accuracy:         0.0
converted norm edit:        0.0
prepared -> converted logits MAE/max: 11.14148 / 103.20934
prepared -> converted CTC sequence agreement: 0.03948002
```

`convert_pt2e` 后 fake-quant/observer 模块数为 `0`，它们已被固化为 Q/DQ 算子，不能再通过
`disable_fake_quant` 或 `disable_observer` 恢复浮点执行。由于 activation qparams 没有初始化，转换图
使用默认 qparams（warning: `must run observer before calling calculate_qparams`），因此 converted=0
是“零激活校准 converted”的预期结果，不能用来判断 PT2E 图结构损坏，也不能作为 PTQ 结果。真正的
训练后 converted 精度必须从带有训练 observer state 的 QAT checkpoint 转换后再测。

## 18. 完整 pretrained_train 图 prepared fake-off

2026-08-06 按用户要求重新使用完整训练图进行 prepared 精度验证，不使用第 17 节的 deploy CTC-only
wrapper。模型由 `build_rec_model(..., graph_role="pretrained_train")` 构建，并通过
`FullRecTrainingWrapper` 捕获完整训练分支：

```text
input:   images, gtc_targets
output:  ctc, ctc_neck, gtc
role:    pretrained_train
wrapper: pytorchocr.quantization.FullRecTrainingWrapper
```

测试过程不执行 activation observer 校准，prepare 后立即关闭 observer 和 fake-quant。由于完整 NRTR
teacher-forcing 图的 CPU prepare/前向成本较高，本轮采用固定 batch=64、256 张验证样本完成 prepared
smoke，不能与全量 2077 张 deploy 指标混用：

```text
artifact: artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_full_training_graph_prepared_fake_off_full_cpu_20260806.json
exported nodes: 2129
prepared nodes: 4136
finite:        true
accuracy:      0.6640625
norm edit:     0.8566886780753968
```

该结果证明完整训练图可以被 `export_for_training` 和 `prepare_qat_pt2e` 捕获，且 prepared fake-off
前向有限。它只代表 256 张样本的训练图 smoke，不代表当前 `exp1` 的 QAT 训练精度，也不代表可部署
QuantONNX 精度。当前完整 `pretrained_train` PT2E/QAT 训练仍需单独解决动态 batch、NRTR 辅助分支的
设备常量和完整图 QuantONNX 输出合同。

## 19. 识别 QAT 默认训练图合同

2026-08-06 将识别 QAT 的默认图从历史 `deploy` 投影修正为 `pretrained_train`。该默认值由
`tools/train.py` 在 `qat=true` 且 `task=rec` 时解析；当前 v5-rec profile 也显式记录同一合同：

```text
rec_graph:                 pretrained_train
reparameterize:            false
rec_ctc_backbone_grad:     false
input:                     images, gtc_targets
head_schema:               [ctc, ctc_neck, gtc]
loss_schema:               [CTCLoss, NRTRLoss]
```

`--rec-graph pretrained_train` 保留 Paddle pretrained 的 CTC+NRTR 完整训练结构。CTC guide detach
继续生效，backbone 由 NRTR 辅助 loss 更新。`--rec-graph deploy` 是显式 CTC-only 部署投影，只允许
用于部署诊断或导出，不再作为识别 QAT 默认训练图。第 16、17 节的 deploy/CTC-only 指标仍仅作为
历史失败与图保持诊断记录，不能用于本轮训练基线，也不能恢复到新 prepared 图。

checkpoint 严格重建同步恢复 `FullRecTrainingWrapper` 双输入图，prepared 动态 batch 维同时绑定
`images` 和 `gtc_targets`，上限取训练多尺度 sampler 的最大 batch。新正式训练必须从完整浮点权重
重新开始，并使用新的 `runs/expN_...` 目录。

本轮增加第 2 epoch 精度异常门禁：浮点基线固定为 `0.5936446798266731`，最大允许下降为 `0.10`。
若第 2 epoch `val_acc <= 0.4936446798266731`，训练先保存 `epoch2_accuracy_guard.json` 和
`debug_epoch_0002.pt`，随后立即终止并开始定位问题，不继续执行第 3 epoch。该门禁参数已进入
training profile 和 checkpoint metadata，修改阈值后不得直接恢复旧实验。

## 20. 重参数化完整训练图的 prepare 前参数与 fake-off 对齐

2026-08-07 对用户检查的“完整 `pretrained_train` 图 + `reparameterize=true` + 全局 U16/S16”合同
补充正式训练前验证。首先在 `prepare_qat_pt2e` 前审计 LAB 参数：

```text
artifact: artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_reparameterized_prepared_input_lab_audit_20260807.json
LAB parameters:                         112
scale / bias:                           56 / 56
checkpoint -> reparameterized eager:    max_abs=0, missing=0, unexpected=0
reparameterized eager -> exported:      max_abs=0, missing=0, unexpected=0
scale range:                            [0.0069139893, 3.2344076633]
bias range:                             [-1.4993383884, 0.6253563166]
scale == 1 count:                       0
bias == 0 count:                        0
```

因此送入 prepare 的重参数化网络保留了正式浮点 LAB 参数。初始化 QuantONNX 中的 `Mul * 1`、
`Add + 0` 不是重参数化结果，而是 `training_steps=0` 时 activation observer 保持默认
`scale=1, zero_point=0`，标量 LAB 输入在立即 convert 时被取整后的结构检查值。

完整图 GPU 对齐前发现 NRTR causal mask 在 CPU capture 中把 `.to(tgt.device)` 固化成
`device(type='cpu')`。已改为通过 `tgt.new_zeros/new_full` 创建 mask，使 exported 图运行设备随输入
变化，并增加回归测试。随后在物理 GPU 0、TF32 关闭、ICDAR2015 全量 2077 张验证集完成对齐：

```text
artifact: artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_full_training_reparameterized_u16s16_prepared_fake_off_full_gpu0_20260807.json
graph:       pretrained_train
inputs:      images, gtc_targets
outputs:     ctc, ctc_neck, gtc
batch size:  64
exported / prepared nodes: 795 / 1527
```

任务指标：

| 阶段 | accuracy | norm_edit_dis |
|---|---:|---:|
| eager | `0.5936446798266731` | `0.8169468283815902` |
| exported | `0.5936446798266731` | `0.8169468283815902` |
| prepared fake-off | `0.5936446798266731` | `0.8169468283815902` |

输出误差：

| 边界 | 输出 | MAE | max_abs |
|---|---|---:|---:|
| eager -> exported | ctc | `0` | `0` |
| eager -> exported | ctc_neck | `0` | `0` |
| eager -> exported | gtc | `0` | `0` |
| exported -> prepared fake-off | ctc | `4.9817430e-6` | `0.001947403` |
| exported -> prepared fake-off | ctc_neck | `2.1623145e-6` | `0.000375271` |
| exported -> prepared fake-off | gtc | `5.3511944e-7` | `5.7220459e-5` |

三个阶段的三个输出均为有限值；prepared CTC argmax agreement 和 CTC sequence agreement 均为
`1.0`。该结果通过正式训练前 fake-off 浮点保持门禁，但不改变初始化 QuantONNX 仅用于结构检查的
定位；fake-on、observer 更新和训练后 converted 精度仍需由后续 QAT smoke/checkpoint 单独验证。

## 21. 重参数化训练 QuantONNX 交付与跨模型结构审计

2026-08-07 在 NRTR causal mask 设备修正后，重新导出完整 `pretrained_train` 初始化 QuantONNX：

```text
ONNX:   exports/training_reparameterize_review_after_mask_fix/quantonnx/ppocrv5_mobile_rec_training_init_qat.onnx
report: exports/training_reparameterize_review_after_mask_fix/training_onnx_report.json
graph:  images + gtc_targets -> ctc + ctc_neck + gtc
shape:  images=[1,3,48,320], gtc_targets=[1,25]
role:   full training graph, reparameterize=true, training_steps=0
qspec:  configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json
```

结构与语义检查结果：

```text
file size:                  72,350,634 bytes
ONNX nodes:                 1443
QuantizeLinear:             412
DequantizeLinear:           531
BatchNormalization:         0
unquantized Conv outputs:   0
requantize DQ/Q:            0
regional targets:           10 / 10
ORT_DISABLE_ALL:            3 outputs MAE=0, max_abs=0, all finite
```

该文件包含 GTC/NRTR 辅助训练头，不是 CTC-only 部署图。其 activation observer 仍处于初始化状态；
因此 LAB scalar 输入经过立即 convert 后仍显示为 `Mul * 1`、`Add + 0`。这不否定第 20 节已确认的
prepare 前正式 LAB 参数，但也意味着该文件当前只用于结构检查，不能作为训练后数值结果。

随后对四个当前目标模型使用正式权重、`reparameterize=true` 和完整训练 wrapper 执行实际
`export_for_training` 审计：

```text
artifact: artifacts/accuracy_baseline/p4_structure/ppocrv5_v6_training_export_structure_audit_20260807.json
```

| 模型 | exported nodes | scalar parameters | 直接进入 Mul/Add 的 scalar | 硬编码 CPU device |
|---|---:|---:|---:|---:|
| PP-OCRv5 mobile rec | 794 | 112 | 112（56 scale + 56 bias） | 0 |
| PP-OCRv5 mobile det | 529 | 114 | 104（52 scale + 52 bias） | 0 |
| PP-OCRv6 small rec | 629 | 0 | 0 | 0 |
| PP-OCRv6 small det | 567 | 8 | 0 | 0 |

因此初始化 observer 将非单位 LAB 参数量化成 `1/0` 的问题不只影响 v5-rec，也已确认影响使用同一
PPLCNetV3 实现的 v5-det。v6 small rec/det 使用 PPLCNetV4，没有直接作为 `Mul/Add` 操作数的
可训练 scalar；v6-det 的 8 个 scalar 是 DB 辅助头单输出通道卷积 bias，不属于 LAB，也不会走该
scalar `Mul/Add` 路径。

为避免只依赖源码扫描，再将 CPU capture 后的四个 exported 完整训练图整体移动到物理 GPU 0 实跑：

```text
artifact: artifacts/accuracy_baseline/p4_structure/ppocrv5_v6_training_export_cuda_audit_20260807.json
```

| 模型 | 实跑输出 | CUDA device | finite |
|---|---|---|---|
| PP-OCRv5 mobile rec | ctc, ctc_neck, gtc | 全部 cuda:0 | 全部 true |
| PP-OCRv5 mobile det | maps | cuda:0 | true |
| PP-OCRv6 small rec | ctc, ctc_neck, gtc | 全部 cuda:0 | 全部 true |
| PP-OCRv6 small det | maps, aux_maps_p4, aux_maps_p3, aux_maps_p2 | 全部 cuda:0 | 全部 true |

当前四模型的辅助训练头未再出现 CPU 算子或 CPU tensor；v5/v6 recognition 的 GTC 输出和 v6-det
三路辅助监督均已覆盖。该结论针对 exported float 完整训练图；prepared fake-on 的 observer 设备
生命周期仍属于后续独立 QAT smoke 门禁，不能由本次结果替代。

仓库内其他结构的源码审计结论：

- PP-OCRv4 mobile rec/det 同样使用 `rec_lcnetv3.PPLCNetV3`，因此后续支持时必须按同一 LAB scalar
  observer 问题处理，不能直接复用初始化 `1/0` QuantONNX。
- `rec_pphgnetv2` 的 B0-B3 定义启用 LAB，若未来开放这些变体也有同类风险；当前 PP-OCRv5 server
  rec/det 使用 B4，构造时明确 `use_lab=false`，不属于该 LAB 路径。
- TPS 与 CANHead 仍存在先创建 CPU tensor、再调用 `.to(input.device)` 的写法。它们不在本次四模型
  图中，尚未做完整 FX/CUDA 验收；在纳入 QAT 前应改为 `input.new_*` 或显式输入设备创建并执行同一
  CPU-capture-to-CUDA 门禁。
- PP-OCRv4/v5/v6 MultiHead 的 NRTR causal mask 共用已修正实现，因此该辅助头的修复不是 v5-rec
  特例；本次 v5/v6 recognition 实跑已验证两条现行配置。

## 22. LAB 参数错误使用 activation qspec 的根因

2026-08-07 进一步检查第 21 节初始化 QuantONNX 后，确认之前“LAB scalar observer 保持默认值”的
描述只说明了直接现象，根因是 Axera dyadic annotator 的 qspec 分类错误。

`ax_quantizer_utils._do_annotate_dyadic()` 对 `Mul/Add` 两个 tensor 输入无条件使用
`get_input_act_qspec()`，没有区分普通 activation 与通过 `get_attr` 进入图的可训练参数。因此 112 个
LAB `scale/bias` 使用了全局 U16 activation fake-quant：

```text
dtype:       torch.uint16
qscheme:     torch.per_tensor_affine
quant range: [0, 65535]
initial:     scale=1, zero_point=0
```

它们没有使用全局权重合同要求的 S16 per-channel symmetric qspec。当前
`initialize_weight_observers()` 又只主动初始化 per-channel Conv/Linear 权重 observer，因此 110 个正常
权重 observer 被初始化，而 112 个 LAB activation observer 仍保持默认值。

实际节点验证：

| ONNX 节点 | 正式浮点参数 | 错误 qspec | 折叠后的整数 | DQ 后值 |
|---|---:|---|---:|---:|
| `node_Mul_449` | LAB scale `0.6180310845` | U16, scale=1, zp=0 | `1` | `1.0` |
| `node_Add_454` | LAB bias `0.1877079457` | U16, scale=1, zp=0 | `0` | `0.0` |
| `node_Mul_464` | LAB scale `0.0095746592` | U16, scale=1, zp=0 | `0` | `0.0` |
| `node_Add_469` | LAB bias `-0.0004117570` | U16, scale=1, zp=0 | `0` | `0.0` |

`onnx_program.optimize()` 已将这些常量的 `QuantizeLinear` 折叠为 U16 initializer，所以文件中表现为
`integer initializer -> DQ -> Mul/Add`；这不是 ONNX 后处理引入的问题，而是 prepared annotation 已经
选择了错误 qspec。

因此第 21 节 QuantONNX 只能用于观察现有错误，不能作为合格的 smoke 结构产物。其
`ORT_DISABLE_ALL MAE=0` 只证明 converted PT2E 与 ONNX 一致，两侧使用的是同一组错误 LAB 值；它
不能证明与浮点模型一致。prepared fake-off 对齐也不会暴露该问题，因为 fake quant 被关闭。

正确修复应在 PT2E annotation 阶段完成，而不是导出后补写常量：

1. 识别 `Mul/Add` 输入中可静态解析到 `nn.Parameter` 的 operand；
2. 对该 operand 使用全局 S16 weight qspec，普通 feature 输入继续使用 U16 activation qspec；
3. 让 LAB 参数 observer 与 Conv/Linear 权重 observer 一样在零训练步导出前按真实参数初始化；
4. 增加 prepared qspec、初始化 QuantONNX 参数值、fake-on/converted 浮点误差和 v5-det 同源路径回归；
5. 修复后重新导出 smoke QuantONNX，再提交结构检查。

## 23. LAB/静态 scalar S16 修复与随机分阶段对比

2026-08-07 完成第 22 节问题修复。实现不在 ONNX 后处理阶段改常量，而是在
`AxeraQuantizerAdapter` 完成基础 annotation 后检查 `Mul/Add` operand：凡是可静态解析为浮点
scalar `get_attr` 的输入，包括 `nn.Parameter` 和 `_convert_scalars_to_attrs` 生成的 buffer，均改用
S16 per-tensor symmetric static-weight qspec。scalar 只有一个元素，不使用 Conv 专用 per-channel
qspec；observer 固定 `zero_point=0`，并使用：

```text
scale = max(max_abs / 32767, 2^-12)
```

`initialize_weight_observers()` 同步初始化这些 static scalar observer。prepared/ONNX activation qparam
审计会排除 static scalar，仍要求真正 activation observer 在零训练步导出时保持默认值。

修复期间随机阶段对比还发现 PyTorch 2.6 的 `FusedMovingAvgObsFakeQuantize` 对 U16/S16 范围不能保持
prepared 与 converted 语义一致：冻结 observer 后 prepared 仍会修改 qparam，而 converted 严格使用
记录值。量化器现对 U16/S16 activation 和 weight 使用标准 `FakeQuantize`，U8/S8 继续使用 fused
实现。最小 scalar 模型的 prepared fake-on -> converted PT2E 已达到逐元素完全一致。

两个中间产物仅保留为失败 provenance，不得用于后续结构验收：

```text
exports/training_reparameterize_lab_weight_fix_review/
  LAB 已改为 S16，但 16-bit fused fake-quant 导致 prepared/converted 不一致。

exports/training_reparameterize_lab_weight_fakequant_fix_review/
  LAB 和 16-bit fake-quant 已修复，但 attention 等固定 scalar buffer 仍保持默认 qparam。
```

最终提交用户检查的完整训练 QuantONNX：

```text
ONNX:   exports/training_reparameterize_static_scalar_fakequant_fix_review/quantonnx/ppocrv5_mobile_rec_training_init_qat.onnx
report: exports/training_reparameterize_static_scalar_fakequant_fix_review/training_onnx_report.json
size:   72,437,015 bytes
```

结构检查：

```text
graph role:                     complete pretrained_train
inputs / outputs:               images+gtc_targets / ctc+ctc_neck+gtc
weight/static observers init:   233
activation observers default:   316 / 316
ONNX nodes / Q / DQ:             1544 / 412 / 632
BatchNormalization:             0
unquantized Conv outputs:       0
regional targets:               10 / 10
Mul/Add S16 scalar DQ operands: 123
uninitialized scalar operands:  0
```

LAB 首组参数在最终 ONNX 中为：

| ONNX 节点 | 正式浮点值 | S16 integer | scale | zero point | DQ 后值 |
|---|---:|---:|---:|---:|---:|
| `node_Mul_450` | `0.6180310845` | `2531` | `0.000244140625` | `0` | `0.617919921875` |
| `node_Add_455` | `0.1877079457` | `769` | `0.000244140625` | `0` | `0.187744140625` |
| `node_Mul_465` | `0.0095746592` | `39` | `0.000244140625` | `0` | `0.009521484375` |
| `node_Add_470` | `-0.0004117570` | `-2` | `0.000244140625` | `0` | `-0.00048828125` |

导出工具现默认使用固定 seed 随机输入，冻结当前 observer 状态后记录 prepared fake-off、prepared
fake-on、converted PT2E 和 `ORT_DISABLE_ALL` QuantONNX 的逐输出对比。本次最终产物结果：

| 对比边界 | ctc MAE/max_abs | ctc_neck MAE/max_abs | gtc MAE/max_abs | 三输出 argmax |
|---|---|---|---|---|
| prepared fake-on -> converted | `0 / 0` | `0 / 0` | `0 / 0` | `1.0` |
| converted -> QuantONNX | `0 / 0` | `0 / 0` | `0 / 0` | `1.0` |
| prepared fake-on -> QuantONNX | `0 / 0` | `0 / 0` | `0 / 0` | `1.0` |

零训练步模型的 activation observer 故意保持 `scale=1, zero_point=0`，因此 fake-off -> fake-on 的
差异较大，仅表示初始化 activation 量化尚未经过 QAT observer 更新，不作为训练精度指标。关键结构
门禁是相同 observer 状态下 prepared fake-on、converted 和 QuantONNX 完全一致。该随机对比已接入
training、initialized 和 checkpoint 导出路径，不再要求为每次图语义检查运行全量数据集。

## 24. Exp2 正式 QAT 启动与 causal-mask 修复

2026-08-07 清理旧的无效导出审查目录，仅保留第 23 节最终静态 scalar 修复产物：

```text
deleted:
  exports/training_reparameterize_review/
  exports/training_reparameterize_review_after_mask_fix/
  exports/training_reparameterize_lab_weight_fix_review/
  exports/training_reparameterize_lab_weight_fakequant_fix_review/
kept:
  exports/training_reparameterize_static_scalar_fakequant_fix_review/
```

正式训练 profile 改为 `reparameterize: true`，并从完整浮点权重重新 prepare：

```text
run:       runs/exp2_ppocrv5_mobile_rec_u16s16_full_qat/
tmux:      ppocrv5-rec-u16s16-full-exp2
device:    physical GPU 3; CUDA_VISIBLE_DEVICES=3; process device cuda:0
weights:   weights/ptocr_v5_mobile_rec_full.pth
train/val: ICDAR 4468 / 2077
graph:     pretrained_train; CTC + CTC neck + GTC/NRTR
qat:       U16 activation + S16 weight, reparameterize=true
optimizer: SGD, momentum=0.9, lr=1.5e-5, warmup=0
observer:  enabled throughout training; validation temporarily disables and restores
shape:     prepared heights 32/48/64, width 320; QuantONNX remains static H=48
```

首个真实 batch 发现原 annotation 将 NRTR causal mask 的
`new_full(-inf) -> triu -> Add -> Softmax` 路径量化，导致 U16 observer 收到 276 个 `-inf`，
qparam 随后变为 NaN。`pytorchocr/quantization/bridge.py` 现让非有限 mask 及 masked-score Add 保持
float，Softmax 的有限输出继续使用原 S16 qspec。最小回归为 `13 passed`，真实完整图由 795 个
exported nodes prepare 为 1519 个 nodes。

完整首批 forward、loss、backward 门禁通过：

```text
batch:       [64, 3, 48, 320], gtc_targets=[64, 25]
CTCLoss:     11.591854095458984
NRTRLoss:    5.101428508758545
total loss:  16.693283081054688
gradients:   304 groups, all finite
```

固定 PT2E decoder 输出为 24 个时间位置，而当前 batch 的有效 NRTR target 为 12 个位置。Paddle
官方模型在 forward 内按 batch `max_len` 动态截断 decoder；当前 `NRTRLoss` 对固定图 logits 执行
等价的时间维裁剪，若 logits 不足仍失败。相关定向测试最终为 `49 passed`，loss/profile 测试为
`36 passed`。

## 25. Exp2 epoch-2 accuracy guard 终止

修复启动问题后，从同一图合同的 `epoch_0001.pt` 严格恢复继续训练。该 checkpoint 的 metadata 为
`rec_graph=pretrained_train`、`head_schema=[ctc, ctc_neck, gtc]`、`reparameterized=true`、
`global_step=65`，不是旧 `exp1` 的 deploy/CTC-only checkpoint。

epoch2 完成 65 steps 和验证后触发异常门禁并停止：

```text
float_accuracy_baseline: 0.5936446798266731
qat epoch1 val_acc:      0.0014443909484833895
qat epoch2 val_acc:      0.0019258545979778526
epoch2 norm_edit_dis:    0.13383079232862416
accuracy_drop:           0.5917188252286952
allowed_drop:            0.10
status:                  stopped for debug; no epoch3/full training
```

这些指标是异常 run 的 provenance，不作为后续 baseline。2026-08-10 清理重复 checkpoint 后，仅保留
epoch-2 debug checkpoint、训练日志和门禁报告：

```text
runs/exp2_ppocrv5_mobile_rec_u16s16_full_qat/debug_epoch_0002.pt
runs/exp2_ppocrv5_mobile_rec_u16s16_full_qat/epoch2_accuracy_guard.json
runs/exp2_ppocrv5_mobile_rec_u16s16_full_qat/train.log
```

`epoch_0001.pt`、`epoch_0002.pt`、`best.pt` 和 `last.pt` 与 debug checkpoint 重复且不能恢复正式训练，
连同三个已解决启动失败日志一起删除。

### 25.1 Exp2 debug QuantONNX

已从 `debug_epoch_0002.pt` 投影并导出 CTC-only 推理 QuantONNX，仅供结构和转换边界检查：

```text
model:   exports/quantonnx/exp2_ppocrv5_mobile_rec_u16s16_epoch2_debug_qat.onnx
input:   images [1, 3, 48, 320]
output:  logits [1, 40, 18385]
nodes:   951
Q / DQ:  245 / 435
BN:      0
```

Q zero-point 包含 `int16=20`、`uint16=225`，weight QDQ 为 `int16=47`；没有未量化 Conv 或直接冗余
DQ/Q。随机输入转换边界为：

```text
converted -> QuantONNX:          MAE=0.08953574299812317, max_abs=1.3739471435546875, argmax=1.0
prepared fake-on -> converted:   MAE=0.04732336476445198, max_abs=0.3931007385253906
prepared fake-off -> fake-on:    MAE=5543.93896484375, max_abs=13654.58984375
```

导出使用 `onnx_program.optimize()`，并通过 ONNX checker、QDQ 基础结构检查和
`ORT_DISABLE_ALL` session 创建。图只保留 `images -> logits`，没有 NRTR/GTC 辅助分支。真实验证集
精度门禁失败，因此该模型不是精度合格交付，也不得进入 Pulsar2 最终验收。

### 25.2 Exp2 epoch-2 参数域审计

审计报告：

```text
artifacts/accuracy_baseline/p4_structure/exp2_debug_epoch2_qat_audit_20260807.json
```

64 张验证样本、GPU 3 的参数和 buffer 交叉实验：

```text
checkpoint weights + fake-off:        acc=0.0, logits_abs_max=46422.36
checkpoint weights + fake-on:         acc=0.0, logits_abs_max=193.28
initial weights + checkpoint qparams: acc=0.6875, norm_edit_dis=0.89893
checkpoint params + initial buffers:  acc=0.0
initial params + checkpoint buffers:  acc=0.5
checkpoint params without LAB:       acc=0.671875, norm_edit_dis=0.90050
LAB parameters only:                 acc=0.0
CTC parameters only:                 acc=0.671875
GTC parameters only:                 acc=0.6875
```

observer/qparam 不是 Exp2 崩溃的首要来源；epoch-2 的 LAB 参数更新足以破坏 CTC 精度，BN running
statistics 漂移是次要放大因素，典型 `ctc_encoder.conv1.norm.running_var` 最大变化为 `12742.09`。
Exp2 的任何 checkpoint 均不得继续正式训练。后续实验必须从浮点权重重新 prepare，每次只改变一个
训练或图变量，并继续使用 epoch-2 accuracy guard。

## 26. Exp3 关闭重参数化对照实验

2026-08-10 建立 Exp3。该实验从同一完整浮点权重重新 prepare，不恢复 Exp2 checkpoint；相对 Exp2
只关闭部署重参数化：

```text
run:       runs/exp3_ppocrv5_mobile_rec_u16s16_no_reparameterize/
tmux:      ppocrv5-rec-u16s16-exp3-no-reparam
device:    物理 GPU 3；CUDA_VISIBLE_DEVICES=3；进程内 device cuda:0
weights:   weights/ptocr_v5_mobile_rec_full.pth
profile:   configs/qat/training/ppocrv5_mobile_rec_u16s16_sgd_dynamic_height_exp3_no_reparameterize.yml
graph:     pretrained_train; CTC + CTC neck + GTC/NRTR
qat:       U16 activation + S16 weight
reparam:   false
optimizer: SGD, momentum=0.9, lr=1.5e-5, warmup=0
observer:  enabled throughout training; validation temporarily disables and restores
shape:     prepared heights 32/48/64, width 320; QuantONNX remains static H=48
guard:     epoch 2 accuracy drop must be <= 0.10 from float baseline 0.5936446798
```

除 profile 名称和 `reparameterize` 外，Exp3 与 Exp2 的训练字段保持一致，包括 seed。当前 quantizer
仍使用 activation observer `eps=2**-12`；本轮不修复该问题，以免引入第二个实验变量。该限制会使
U16 输入 `[-1,1]` 的 scale 被截断为 `2^-12`，因此 Exp3 仅用于判断关闭重参数化是否改变训练崩溃，
不作为最终 16-bit 精度方案。

Exp2/Exp3 profile 自动对比确认，除 profile `name` 外，`training` 中唯一差异为：

```text
reparameterize: true -> false
```

`tests/test_training_profile.py` 定向测试为 `26 passed`。Exp3 通过 tmux
`ppocrv5-rec-u16s16-exp3-no-reparam` 在物理 GPU 3 上运行，完成两个 epoch 后由精度门禁终止。
epoch 1 完成 65 steps：

```text
CTCLoss:          18.97657910860502
NRTRLoss:          4.322487864127526
total loss:       23.299066983736477
val CTCLoss:      16.79372530272513
val NRTRLoss:      2.895066983772047
val loss:         19.68879243099328
val accuracy:      0.038517091959557055
val norm_edit_dis: 0.43992978214210765
observer frozen:   false
```

Exp3 epoch-1 accuracy 高于 Exp2 的 `0.0014443909`，但仍比浮点基线低约 `0.5551`，不能依据单个 epoch
宣称关闭重参数化已恢复精度。

epoch 2 完成 65 steps 和验证后触发同一异常门禁并自动停止：

```text
CTCLoss:                 10.12136159309974
NRTRLoss:                 3.8299038043388953
total loss:              13.951265364426833
val CTCLoss:              9.526182738217441
val NRTRLoss:             2.419817819739833
val loss:                11.946000518220844
val accuracy:             0.38998555609051516
val norm_edit_dis:        0.7045760263381834
float accuracy baseline:  0.5936446798266731
accuracy drop:            0.20365912373615797
allowed drop:             0.10
observer frozen:          false
status:                   stopped by epoch-2 accuracy guard
```

关闭重参数化使 epoch-2 accuracy 从 Exp2 的 `0.0019258546` 提升至 `0.3899855561`，说明重参数化是
Exp2 精度崩溃的重要影响因素；但 Exp3 仍未通过浮点保持门禁，不能继续 epoch 3 或作为正式 16-bit
baseline。当前已知的 U16 observer `eps=2**-12` 问题仍存在，本实验没有修改它，以保持单变量对照。

2026-08-10 清理重复 checkpoint 后保留的诊断产物：

```text
runs/exp3_ppocrv5_mobile_rec_u16s16_no_reparameterize/debug_epoch_0002.pt
runs/exp3_ppocrv5_mobile_rec_u16s16_no_reparameterize/epoch2_accuracy_guard.json
runs/exp3_ppocrv5_mobile_rec_u16s16_no_reparameterize/train.log
```

### 26.1 Exp3 epoch-2 QuantONNX 导出

从 `debug_epoch_0002.pt` 按 checkpoint metadata 严格重建 `reparameterized=false` 的完整训练图，转换
PT2E 后投影为 CTC-only 部署图。导出保留 `onnx_program.optimize()`：

```text
model:   exports/quantonnx/exp3_ppocrv5_mobile_rec_u16s16_epoch2_debug_no_reparameterize_qat.onnx
input:   images [1, 3, 48, 320]
output:  logits [1, 40, 18385]
nodes:   1802
Q / DQ:  507 / 762
BN:      19
```

Q zero-point 包含 `uint16=487`、`int16=20`，量化 weight QDQ 为 `int16=145`。图中没有未量化 Conv
输出、冗余 DQ/Q 或内部 SiLU QDQ，Concat 的量化输入共享 qparams。ONNX checker 和
`ORT_DISABLE_ALL` session 创建通过。

该文件是诊断 QuantONNX，不是 Axera 合格交付。统一 QDQ 门禁因 19 个 `BatchNormalization` 和 1 个
`Conv -> QDQ -> BatchNormalization` 路径拒绝该图。关闭重参数化后，这些 identity BN 保留在 prepared
训练图中，`convert_pt2e` 和 ONNX optimize 不会全部将其融合进 Conv。本次没有通过 ONNX 后处理折叠
BN，因为那会改变 Exp3 的无重参数化图合同。文件保留供结构检查，不能直接进入 Pulsar2 最终验收。

## 27. runs 实验清理

2026-08-10 清理当前训练图合同已不再使用的 runs 产物。删除范围包括：

- PP-OCRv6 det 的 CTW1500 pilot、same-pad、hard-activation 和 add-boundary 前置实验；
- PP-OCRv5 det、PP-OCRv5 rec、PP-OCRv6 rec 的可再生 smoke 目录；
- PP-OCRv5 rec 旧 native-SiLU baseline/repro 日志目录；
- `exp1_ppocrv5_mobile_rec_u16s16_qat` 的 deploy/CTC-only 50-epoch 失败实验；
- Exp2/Exp3 中与 epoch-2 debug checkpoint 重复的 epoch、best 和 last checkpoint；
- Exp2 已解决的三份启动失败日志和 exit-code 文件。

清理后 `runs/` 保留：

```text
icdar2015_ppocrv5_mobile_det_qat/                         v5 det 历史 QuantONNX 交付
icdar2015_ppocrv6_small_det_qat/                          v6 det 历史 QuantONNX 交付
icdar2015_ppocrv6_small_rec_qat/                          v6 rec 历史 QuantONNX 交付
icdar2015_ppocrv5_mobile_rec_native_silu_qat_corrected_no_warmup_20260805/
                                                          accuracy 0.49350 的 best.pt
exp2_ppocrv5_mobile_rec_u16s16_full_qat/                  epoch-2 debug、guard、train.log
exp3_ppocrv5_mobile_rec_u16s16_no_reparameterize/         epoch-2 debug、guard、train.log
```

Exp1、Exp2 和 Exp3 均不是后续正式训练的恢复起点。新的训练实验必须从完整浮点权重重新 prepare，并使用
新的 `expN_...` 目录。

## 28. v5/v6 Float ONNX 重参数化矩阵

2026-08-10 使用完整浮点 pretrained 权重导出 PP-OCRv5 mobile、PP-OCRv6 small 的 det/rec 训练图和
推理图，并分别覆盖重参数化、非重参数化结构，共 16 份 Float ONNX：

```text
root:   exports/float_onnx/v5_v6_reparameterization_matrix/
report: exports/float_onnx/v5_v6_reparameterization_matrix/float_onnx_report.json
```

图合同：

```text
det training:  images -> maps(shrink/threshold/binary)，v6 额外输出 aux_maps_p4/p3/p2
det inference: images -> maps(shrink)
rec training:  images + gtc_targets -> ctc + ctc_neck + gtc
rec inference: images -> logits(CTC)
```

结构统计：

| 模型 | training reparam nodes/BN | training non-reparam nodes/BN | inference reparam nodes/BN | inference non-reparam nodes/BN |
| --- | ---: | ---: | ---: | ---: |
| v5 mobile det | `271 / 0` | `505 / 19` | `261 / 0` | `495 / 19` |
| v5 mobile rec | `461 / 0` | `695 / 19` | `259 / 0` | `493 / 19` |
| v6 small det | `309 / 0` | `373 / 10` | `249 / 0` | `313 / 10` |
| v6 small rec | `433 / 0` | `477 / 11` | `231 / 0` | `275 / 11` |

16 份 ONNX 均通过 checker 和 `ORT_DISABLE_ALL` 随机输入对齐，所有输出有限。逐输出 MAE/max_abs、
shape、文件大小和路径见 JSON 报告。非重参数化图保留 BN 是预期结构差异，不能直接作为 Axera
QuantONNX；重参数化版本 BN 均为 0。导出命令和可复现参数见
`docs/axera_qat/guides/training_onnx_export.md`。

# ICDAR2015 检测 QAT 训练记录

## 1. 运行状态

```text
启动日期:   2026-07-29
完成日期:   2026-07-30
状态:       训练和 QuantONNX 导出完成
tmux:       ppocrv6-det-qat-icdar（训练进程已退出，会话仍保留）
device:     cuda:3
训练轮数:   50 epochs
退出状态:   TRAIN_EXIT_CODE=0
```

进入保留的会话并查看最终终端输出：

```bash
tmux attach -t ppocrv6-det-qat-icdar
```

查看日志：

```bash
tail -f output/icdar2015_ppocrv6_small_det_qat/train.log
```

## 2. 数据集

使用解压后的 ICDAR2015 检测数据：

```text
root:  /home/heqi/dataset/icdr
train: train_icdar2015_label.txt
       icdar_c4_train_imgs/，1000 images，11886 polygons
val:   test_icdar2015_label.txt
       ch4_test_images/，500 images，5230 polygons
```

训练集有效文本 polygon 为 4468，忽略区域为 7418；验证集分别为 2077 和 3153。图片均为
`1280x720`，标签与图片一一对应，无非法、退化或越界四边形，train/val 无内容重复。

当前 QAT baseline 不启用随机增强。图片以单一比例缩放为 `640x360`，再上下各居中 padding 140
像素到 `640x640`；polygon 使用相同缩放和平移。

## 3. 模型和量化配置

```text
model:       PP-OCRv6_small_det
model YAML:  configs/det/PP-OCRv6/PP-OCRv6_small_det.yml
float weight: ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth
profile:     configs/qat/training/ppocrv6_small_det_baseline.yml
QAT config:  configs/qat/ppocrv6_small_det_u8s8.json
```

有效参数：

```text
epochs:                 50
batch_size:             8
steps_per_epoch:        125
workers:                8
image_shape:            [3, 640, 640]
optimizer:              Adam
initial_learning_rate:  2e-5
warmup_epochs:          2
lr_final_factor:        0.1
AMP:                    false
reparameterize:         true
observer_freeze_epoch:  None
QAT EMA:                false
```

observer 在训练期间持续更新；validation 临时关闭 observer，结束后恢复。`best.pt` 依据最低
validation DB loss 选择，不提前冻结 observer。

## 4. 启动命令

创建 tmux 会话：

```bash
tmux new-session -s ppocrv6-det-qat-icdar
```

tmux 内执行以下命令。先创建输出目录，确保 `tee` 在首次运行时可以打开日志文件；通过
`pipefail` 和 `PIPESTATUS` 保留 Python 训练进程的真实退出状态。

```bash
cd /home/heqi/project/PaddleOCR/route2/PaddleOCR2Pytorch-QAT
mkdir -p output/icdar2015_ppocrv6_small_det_qat
set -o pipefail

env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth \
  --label-file /home/heqi/dataset/icdr/train_icdar2015_label.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/test_icdar2015_label.txt \
  --output-dir output/icdar2015_ppocrv6_small_det_qat \
  --training-profile configs/qat/training/ppocrv6_small_det_baseline.yml \
  --device cuda:3 \
  2>&1 | tee output/icdar2015_ppocrv6_small_det_qat/train.log

status=${PIPESTATUS[0]}
echo "TRAIN_EXIT_CODE=$status"
```

标准输出和错误输出同时写入：

```text
output/icdar2015_ppocrv6_small_det_qat/train.log
```

## 5. 中间结果

训练期间 GPU 3 显存约 5.3 GiB，训练利用率约 70% 到 85%。前 12 epochs 如下：

| Epoch | Learning rate | Train loss | Val loss | Observer frozen |
|---:|---:|---:|---:|:---:|
| 1 | 1.0000e-5 | 3.23082 | 2.80676 | false |
| 2 | 2.0000e-5 | 2.69242 | 2.67062 | false |
| 3 | 2.0000e-5 | 2.53921 | 2.62145 | false |
| 4 | 1.9981e-5 | 2.46341 | 2.61487 | false |
| 5 | 1.9923e-5 | 2.38378 | 2.59092 | false |
| 6 | 1.9827e-5 | 2.32433 | 2.58860 | false |
| 7 | 1.9693e-5 | 2.28097 | 2.60546 | false |
| 8 | 1.9522e-5 | 2.25868 | 2.58577 | false |
| 9 | 1.9315e-5 | 2.20653 | 2.57466 | false |
| 10 | 1.9072e-5 | 2.16257 | 2.56381 | false |
| 11 | 1.8794e-5 | 2.14632 | **2.53858** | false |
| 12 | 1.8483e-5 | 2.11877 | 2.55411 | false |

当前最佳验证 loss 来自 epoch 11。训练 loss 总体下降，validation loss 存在正常波动，尚未观察到
NaN、梯度异常或 observer 被提前关闭。

最终完成 50 epochs、6250 个训练 step，tmux 返回 `TRAIN_EXIT_CODE=0`：

```text
epoch 50 learning_rate: 2.01927e-6
epoch 50 train loss:    1.62210
epoch 50 val loss:      2.59889
best epoch:             11
best val loss:          2.5385760883
```

`best.pt` metadata 确认 epoch 11/global step 1375，`last.pt` 为 epoch 50/global step 6250；二者
observer 状态均为未冻结。后期 train loss 继续下降而 validation loss 平台化，因此交付使用
`best.pt`，不使用最后一轮权重。

## 6. Checkpoint

输出目录：

```text
output/icdar2015_ppocrv6_small_det_qat/
```

本次运行产生的文件：

```text
train.log                 50 epochs 的逐轮 JSON 日志
epoch_0001.pt ...
epoch_0050.pt             每轮完整训练状态
last.pt                   epoch 50，恢复训练入口
best.pt                   epoch 11，最低 validation DB loss
best_qdq.onnx             从 best.pt 严格导出的 QuantONNX
```

checkpoint 包含 prepared PT2E model、observer/fake-quant、optimizer、scheduler、epoch/global step、
profile/量化配置 SHA256、Torch 版本、输入 shape 和 observer 策略。

2026-08-03 复核的关键交付物指纹如下。复制、归档或交付产物后应重新计算并比对：

```text
69b8ffc47405f26927588a0cf1b62dd41a1463c80af8debce00f8dfd30d38e09  output/icdar2015_ppocrv6_small_det_qat/best.pt
1672cc164a435b4851bcb9e1dc07fb7b25a98073cb7b4c734479b70dc5a5ba13  output/icdar2015_ppocrv6_small_det_qat/best_qdq.onnx
f96e4494d8a8dc656d130dd02790dd9a5650eb9ad951b98e37c8430ce6ae5a36  output/icdar2015_ppocrv6_small_det_qat/train.log
3f558757b47104fc072b2f7fa962d1162f07fe9e829cccb1bbb9d0113785bf12  configs/qat/ppocrv6_small_det_u8s8.json
244479fe5d1bf5990a102a898fe586457911deed851089fc1a3ac48637ce51e3  configs/qat/training/ppocrv6_small_det_baseline.yml
```

## 7. 恢复训练

训练异常中断时使用同一模型、profile、数据和输出目录恢复：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth \
  --resume output/icdar2015_ppocrv6_small_det_qat/last.pt \
  --label-file /home/heqi/dataset/icdr/train_icdar2015_label.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/test_icdar2015_label.txt \
  --output-dir output/icdar2015_ppocrv6_small_det_qat \
  --training-profile configs/qat/training/ppocrv6_small_det_baseline.yml \
  --device cuda:3
```

恢复时会严格检查 task、模型 YAML、profile/量化配置 SHA256、Torch 版本、shape 和重参数化状态。

## 8. QuantONNX 验收

完成 50 epochs 后已执行：

```bash
env PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python \
  tools/export_ocr_onnx.py checkpoint \
  --checkpoint output/icdar2015_ppocrv6_small_det_qat/best.pt \
  --output output/icdar2015_ppocrv6_small_det_qat/best_qdq.onnx \
  --batch-size 1
```

严格 checkpoint reload、`convert_pt2e`、`onnx_program.optimize()`、ONNX checker 和 ORT 均通过：

```text
checkpoint: best.pt（epoch 11）
output: best_qdq.onnx
input:  [1, 3, 640, 640]
output: [1, 1, 640, 640]
float/prepared/converted nodes: 402/798/1283
ONNX nodes: 761
Q/DQ: 177/335
zero-point dtype: uint8=177
BatchNormalization: 0
Conv -> QDQ -> BatchNorm: 0
Concat quantized/shared: 2/2
HardSigmoid quantized: 13/13
unquantized Conv outputs: 0
direct/redundant/requant DQ->Q: 0/0/0
output finite: true
PyTorch/ORT MAE: 0.06930594
PyTorch/ORT max_abs: 0.85487026
```

本机没有 `pulsar2` 可执行文件和 PP-OCR 专用编译配置，因此 Axera 编译/板端运行仍待在对应工具链
环境执行。不得通过手工修改当前 QuantONNX 绕过转换问题。

2026-07-30 补齐 DBPostProcess 和 DetMetric 后，从同一 `best.pt` 严格恢复并验证 500 张测试图：

```text
precision: 0.5764036958
recall:    0.3904670197
hmean:     0.4655568312
```

该历史 `best.pt` 仍是旧 Trainer 按最低 validation loss 选出的 epoch 11，不能证明它也是 50 epochs
中的 hmean 最优。新训练已改为按 YAML 的 `Metric.main_indicator=hmean` 选择最佳 checkpoint。

## 9. 当前结论和待办

| 检查项 | 状态 | 结论 |
|---|:---:|---|
| ICDAR2015 50-epoch QAT | 通过 | 6250 steps，无 NaN，observer 未提前关闭 |
| checkpoint 严格重载 | 通过 | `best.pt` 可还原 prepared PT2E 图和量化状态 |
| QuantONNX 导出 | 通过 | `onnx_program.optimize()`、checker 和 ORT 均通过 |
| Axera 量化域结构检查 | 通过 | Conv、Concat、HardSigmoid 和 DQ/Q 边界符合当前规则 |
| 检测精度评估 | 通过 | precision 0.57640、recall 0.39047、hmean 0.46556 |
| QuantONNX metric | 通过 | ORT optimize off hmean 0.46553；optimize on 仅 0.44966 |
| Pulsar2/Axera 编译 | 待完成 | 当前机器缺少工具和 PP-OCR 编译配置 |

因此，当前结果证明检测 QAT 的训练、checkpoint、prepared QAT metric 和 QuantONNX 主链路可用。
下一阶段需比较 float/prepared/converted/ORT/Axera 的同集指标，并在具备 Pulsar2 的环境完成编译和
板端回归。完整 metric 契约见 [ICDAR2015 QAT Metric 验证记录](icdar2015_qat_metric_validation.md)。

## 10. 2026-08-03 完成状态复核

再次检查保留的 tmux 会话、训练日志和输出目录，确认本次检测 QAT 已结束，而不是仍在后台训练：

```text
tmux session:       ppocrv6-det-qat-icdar（仅保留终端历史）
最后训练 epoch:     50
总训练 step:        6250
训练退出状态:       TRAIN_EXIT_CODE=0
最后 observer 状态: observers_frozen=false
best checkpoint:    output/icdar2015_ppocrv6_small_det_qat/best.pt
last checkpoint:    output/icdar2015_ppocrv6_small_det_qat/last.pt
QuantONNX:           output/icdar2015_ppocrv6_small_det_qat/best_qdq.onnx
完整日志:            output/icdar2015_ppocrv6_small_det_qat/train.log
```

复核时 tmux pane 停留在 shell prompt，末尾为 epoch 50 的 JSON 日志及 `TRAIN_EXIT_CODE=0`。输出目录
包含 `epoch_0001.pt` 到 `epoch_0050.pt`、`best.pt`、`last.pt`、`best_qdq.onnx` 和 `train.log`。

本记录中的 `best.pt` 是旧 Trainer 按最低 validation loss 选出的 epoch 11。后续代码虽已支持按
`Metric.main_indicator=hmean` 选优，但不能据此改写这次历史训练的 checkpoint 语义。若需要获得按
hmean 选优的模型，必须从浮点预训练权重重新训练并逐 epoch 计算检测指标。

当前可确认的验收边界为：训练、strict checkpoint reload、PT2E convert、QuantONNX 导出、ONNX
checker、QDQ 结构检查和 ORT 禁用优化时的验证集指标均已完成；Pulsar2 编译、Axera 板端推理和板端
hmean 仍未完成。

本次复核没有重新启动或覆盖训练任务；文档记录的是 tmux 中已完成的原始运行。保留该会话仅用于
查看终端历史，后续新实验应使用新的输出目录和 tmux 会话名，避免覆盖上述可追溯产物。

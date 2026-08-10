# PP-OCRv5 Mobile Det ICDAR2015 QAT 训练记录

## 1. 任务状态

```text
启动日期:   2026-08-03
状态:       50-epoch 正式 QAT 训练和 QuantONNX 验收完成
tmux:       ppocrv5-mobile-det-qat-icdar50
device:     cuda:0
输出目录:   output/icdar2015_ppocrv5_mobile_det_qat
```

查看实时终端和日志：

```bash
tmux attach -t ppocrv5-mobile-det-qat-icdar50
tail -f output/icdar2015_ppocrv5_mobile_det_qat/train.log
```

训练共完成 50 个 epoch、6250 个 step；Python 训练流程正常结束。tmux socket 在当前受限复核环境中
不可访问，因此没有重新读取 pane 中的退出码；50 个 epoch checkpoint、可加载 metadata、strict
checkpoint reload 和 QuantONNX 验收均已通过。

## 2. 模型、权重和数据

```text
model:         PP-OCRv5_mobile_det
model YAML:    configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml
Paddle weight: PP-OCRv5_mobile_det_pretrained.pdparams
Torch weight:  ptocr_v5_mobile_det.pth
train:         /home/heqi/dataset/icdr/train_icdar2015_label.txt
val:           /home/heqi/dataset/icdr/test_icdar2015_label.txt
data root:     /home/heqi/dataset/icdr
```

官方 Paddle 权重 SHA256：

```text
7e2e3b0bd5bbdcb0b842cb92aaacc2852f80299a4858b8767a45bd0c6e955648
```

转换器执行严格一对一映射：905 个 Paddle 参数全部复制，PyTorch 额外的 150 个 key 全部是
BatchNorm `num_batches_tracked`。相同随机输入上的 Paddle/PyTorch shrink-map MAE 为
`1.0518916467e-12`，因此本次 QAT 从已验证的浮点基线开始。

## 3. 训练配置

```text
profile:                configs/qat/training/ppocrv5_mobile_det_baseline.yml
QAT config:             configs/qat/ppocrv5_mobile_det_u8s8.json
epochs:                 50
batch size:             8
steps per epoch:        125
workers:                8
image shape:            [3, 640, 640]
initial learning rate:  2e-5
warmup epochs:          2
lr final factor:        0.1
AMP:                    false
reparameterize:         true
observer freeze epoch:  None
QAT EMA:                false
best indicator:         hmean
```

训练使用固定 `640x640` 确定性居中 letterbox，padding value 为 114，不启用随机数据增强。
observer 在训练期持续更新，validation 时临时关闭，结束后恢复。

## 4. 启动命令

```bash
cd /home/heqi/project/PaddleOCR/route2/PaddleOCR2Pytorch-QAT
mkdir -p output/icdar2015_ppocrv5_mobile_det_qat
set -o pipefail

env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --weights ptocr_v5_mobile_det.pth \
  --label-file /home/heqi/dataset/icdr/train_icdar2015_label.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/test_icdar2015_label.txt \
  --output-dir output/icdar2015_ppocrv5_mobile_det_qat \
  --training-profile configs/qat/training/ppocrv5_mobile_det_baseline.yml \
  --device cuda:0 \
  2>&1 | tee output/icdar2015_ppocrv5_mobile_det_qat/train.log

status=${PIPESTATUS[0]}
echo "TRAIN_EXIT_CODE=$status"
```

## 5. 训练进展

训练过程关键结果：

| Epoch | Learning rate | Train loss | Val loss | Precision | Recall | Hmean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0e-5 | 8.16613 | 6.81939 | 0.00509 | 0.02937 | 0.00867 |
| 2 | 2.0e-5 | 4.99343 | 4.18437 | 0.03390 | 0.02504 | 0.02880 |
| 3 | 2.0e-5 | 4.06671 | 3.98213 | 0.04671 | 0.03418 | 0.03948 |
| 28 | 1.041e-5 | 3.19083 | 3.21627 | 0.30960 | 0.09629 | 0.14690 |
| 37 | 5.521e-6 | 3.06583 | 3.25454 | 0.34347 | 0.10881 | **0.16527** |
| 50 | 2.019e-6 | 3.51719 | 3.52597 | 0.28619 | 0.08281 | 0.12845 |

训练未出现 NaN，50 个 epoch 的 `observers_frozen` 均为 `false`。训练过程按 hmean 选择 best，
不是按最后一轮或 validation loss 选择。

最终 `best.pt` metadata：

```text
best epoch:       37
global step:      4625
best indicator:   hmean
best hmean:       0.1652650823
observer frozen:  false
last epoch:       50
last global step: 6250
```

## 6. 产物和指纹

```text
output/icdar2015_ppocrv5_mobile_det_qat/best.pt
output/icdar2015_ppocrv5_mobile_det_qat/last.pt
output/icdar2015_ppocrv5_mobile_det_qat/best_qdq.onnx
output/icdar2015_ppocrv5_mobile_det_qat/train.log
```

2026-08-03 SHA256：

```text
ec7566884cf2f4aa951c47e9a52f1fc7b8683531b18dfc7107d09f38c9a10485  output/icdar2015_ppocrv5_mobile_det_qat/best.pt
6df22c798bba4d559b296fdc2ca9ae425df30959d689b61fe0c64d8e90b117ec  output/icdar2015_ppocrv5_mobile_det_qat/last.pt
aff5348fcfa6dd7dd79480dfdceae662aea4aae0f75394fb0354fdccbbed7847  output/icdar2015_ppocrv5_mobile_det_qat/best_qdq.onnx
d23d39283ebeb70e2468a532df507e1d06aa02217cdea02529665a6f3dc4ef07  output/icdar2015_ppocrv5_mobile_det_qat/train.log
5b39ed086fb1071e81456da659dc518e882b4551eb60889d09e3c4eb1573a571  configs/qat/training/ppocrv5_mobile_det_baseline.yml
3f558757b47104fc072b2f7fa962d1162f07fe9e829cccb1bbb9d0113785bf12  configs/qat/ppocrv5_mobile_det_u8s8.json
d5013fadb085acfac1ed29dcd3106ccc4beca24e31e92258f0b6ee2b793904d3  ptocr_v5_mobile_det.pth
```

## 7. QuantONNX 验收

从 `best.pt` strict reload、`convert_pt2e` 和 `onnx_program.optimize()` 后导出：

```text
input/output:           [1,3,640,640] / [1,1,640,640]
float/prepared/converted nodes: 528 / 1061 / 1525
ONNX nodes:             980
Quantize/Dequantize:    250 / 469
zero-point dtype:       uint8=250
BatchNormalization:     0
Concat shared domains:  1 / 1
HardSigmoid quantized:  10 / 10
unquantized Conv:       0
direct DQ -> Q:         0
redundant DQ -> Q:      0
requant DQ -> Q:         0
PyTorch/ORT MAE:        2.8693705e-5
PyTorch/ORT max abs:    0.3058824
ORT optimized A/B MAE:  7.4352793e-5
```

ORT optimize-off 的 500 张 ICDAR2015 test 评估：

```text
precision:       0.3599257885
recall:          0.0934039480
hmean:           0.1483180428
sample_count:    500
provider:        CPUExecutionProvider
finite:          true
```

同一 `best.pt` 的独立 CPU prepared 评估为 precision `0.3499197`、recall `0.1049591`、hmean
`0.1614815`。训练时 CUDA prepared hmean 为 `0.1652651`；CPU/CUDA PT2E backend 的轻微差异已记录，
不能将三者视为 bit-exact。

## 8. 后续验收

当前机器仍缺少 Pulsar2 和 Axera 板端环境，因此训练和 QuantONNX 通过不能替代芯片验收。

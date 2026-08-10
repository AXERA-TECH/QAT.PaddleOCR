# PP-OCRv5 Mobile Rec QAT Smoke 记录

## 1. 范围

本记录只验证 PP-OCRv5 mobile rec 的真实预训练权重转换、Paddle/PyTorch CTC 浮点对齐，以及
Axera PT2E QAT 到 QuantONNX 的单步 smoke。按当前检查节点停止，不启动真实数据训练，不生成 QAT
checkpoint，也不声明任务精度。

## 2. 权重和字典

```text
model YAML: configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml
Paddle:     PP-OCRv5_mobile_rec_pretrained.pdparams
PyTorch:    ptocr_v5_mobile_rec.pth
dictionary: pytorchocr/utils/dict/ppocrv5_dict.txt
classes:    18385（18383 字典字符 + space + CTC blank）
```

SHA256：

```text
04745475b97a1faf029c7442a4c4421b156249b9395814e509bf4a9804e37750  PP-OCRv5_mobile_rec_pretrained.pdparams
7667656321365b7c438564e3ebe4abe130a4e9c57740c40364f81df7a127dd80  ptocr_v5_mobile_rec.pth
d1979e9f794c464c0d2e0b70a7fe14dd978e9dc644c0e71f14158cdf8342af1b  pytorchocr/utils/dict/ppocrv5_dict.txt
```

原 Paddle state dict 有 968 个参数。本路线仅支持 CTC，明确排除 84 个 `head.gtc_head` /
`head.before_gtc` source 参数；剩余 884 个 backbone、SVTR 和 CTC 参数全部严格映射。

PyTorch 完整 MultiHead 有 1143 个 state key。未由 Paddle CTC source 覆盖的 259 个 key 组成：

```text
BatchNorm num_batches_tracked: 151
head.before_gtc:                 1
head.gtc_head (NRTR):          107
other:                           0
```

保存的 `.pth` 包含完整 PyTorch state dict，但未转换的 NRTR/GTC target 仍是随机初始化值；该权重只允许
通过 route2 `RecCTCWrapper` 使用，不能作为 NRTR/GTC 模型权重。

## 3. 构图修复

严格转换首先发现：

```text
Paddle head.ctc_encoder.encoder.conv4.conv.weight: [60, 960, 1, 3]
PyTorch target:                                      [60, 960, 3, 3]
```

根因是 route2 `EncoderWithSVTR.conv4` 未传入 YAML 的 `kernel_size=[1,3]`，退回了 `ConvBNLayer`
默认 `3x3`。现已与 Paddle 实现对齐，`conv1` 和 `conv4` 都使用配置 kernel 和对应 padding；新增测试
锁定 v5/v4 mobile rec 的两个卷积均为 `1x3`。没有 reshape 权重或放宽 strict 校验。

## 4. 浮点对齐

命令：

```bash
env HOME=/tmp/ppocr_route2_home FLAGS_use_mkldnn=0 \
  /home/heqi/miniforge3/envs/ocr_moderation/bin/python \
  tools/compare_rec_parity.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --paddle-weights PP-OCRv5_mobile_rec_pretrained.pdparams \
  --torch-weights ptocr_v5_mobile_rec.pth \
  --image-shape 3 48 320
```

结果：

```text
output shape:          [1, 40, 18385]
logits MAE:            5.8733417e-6
logits p99:            1.9550323e-5
logits max abs:        4.1007996e-5
relative MAE:          6.5300385e-7
probability MAE:       1.9486809e-10
probability max abs:   1.9073486e-6
argmax agreement:      1.0
finite:                true / true
```

该结果在 `Swish.forward -> F.silu` 后重新测得。`export_for_training` 图中为单个 `aten.silu`，
没有独立 `aten.sigmoid`/`aten.mul`，说明 QAT annotator 面向完整激活边界。

## 5. 历史全局 U8/S16 Smoke（已被替代）

```bash
env PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python \
  tools/qat_smoke.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights ptocr_v5_mobile_rec.pth \
  --qat-config configs/qat/ppocrv5_mobile_rec_u8s8.json \
  --output /tmp/ppocrv5_mobile_rec_pretrained_qat_smoke.onnx \
  --image-shape 3 48 320
```

smoke 覆盖 `export_for_training -> prepare_qat_pt2e -> backward -> convert_pt2e ->
onnx_program.optimize() -> ONNX checker -> QDQ validator -> ORT`，结果：

```text
float/prepared/converted nodes: 507 / 998 / 1371
gradient tensors:               43
output shape:                   [1, 40, 18385]
ONNX nodes:                     959
Quantize/Dequantize:            252 / 438
U8/S16 activation domains:      234 / 18
BatchNormalization:             0
Concat shared domains:          1 / 1
HardSigmoid quantized:          2 / 2
unquantized Conv:               0
direct DQ -> Q:                 3
necessary requant DQ -> Q:      3
redundant DQ -> Q:              0
ORT argmax agreement:           1.0
ORT optimized argmax agreement: 1.0
```

导出器优化期间移除了 8 组 qparam 完全一致的冗余 DQ/Q；最终保留的 3 组均为必要 requant。
QuantONNX 已归档到：

```text
output/ppocrv5_mobile_rec_qat_smoke/pretrained_qdq.onnx
SHA256: df9e43d4e168121c8ea8ea92970a511b3a63e41b1b018dda5007f527c7785ac9
```

上述结果仅作为修改 Attention qspec 前的问题基线，不作为后续训练或 Pulsar2 转换输入。其 3 个
requant 中有 2 个位于 Attention 内部，已由局部 S8 QAT 配置消除。

## 6. 历史局部 S8 Attention Smoke（已被 16-bit 合同替代）

以下配置和产物仅用于历史结构对照，配置文件已经删除，不能用于恢复训练或生成新产物：

```text
configs/qat/ppocrv5_mobile_rec_u8s8_attn_s8_native_silu.json
global activation: U8
weight:            S8
QKV Linear output: S8
scale Mul:         S8 -> S8
first MatMul:      S8/S8 -> S8
Softmax:           S8 -> S8
second MatMul:     S8/S8 -> U8
```

FX 节点由项目 skill 根据 `nn_module_stack` 和拓扑重新发现：

```text
.codex/skills/ppocr-qat-config-discovery/
```

当前配置在真实权重、`1x3x48x320` 的 `export_for_training` 图上检查通过，严格发现两个 owner：

```text
model.head.ctc_encoder.encoder.svtr_block.0.mixer
model.head.ctc_encoder.encoder.svtr_block.1.mixer
```

最新 QuantONNX：

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv5_mobile_rec_attn_s8_native_silu_smoke_20260804.onnx
SHA256: 8070c4071e1b332e196de15c6dc144a46127d86708f25ed4ff228f87ea4d17f5
```

结构结果：

```text
ONNX nodes:               941
QuantizeLinear:           243
DequantizeLinear:         429
zero-point dtype:         S8 20 / U8 223
BatchNormalization:       0
Pad:                      0
unquantized Conv:         0
Concat shared domains:    1 / 1
HardSigmoid QDQ:          2 / 2
SiLU boundary QDQ:        7 / 7
SiLU internal QDQ:        0
Identity:                 1
necessary requant:        1
```

两个 Attention 都满足：

```text
first MatMul:  S8/S8 -> S8
second MatMul: S8/S8 -> U8
```

唯一保留 requant 在 Attention 外部：

```text
node_DequantizeLinear_1506
  -> node_DequantizeLinear_1506_to_node_QuantizeLinear_1512_identity
  -> node_QuantizeLinear_1512

U8(scale=0.19532202184200287, zero_point=8)
  -> U8(scale=0.1938261240720749, zero_point=8)
```

该边两侧 qparam 不同，不是冗余 QDQ。按当前约束保留在 PT2E/QAT 图中；已撤销 Pool/Conv observer
共享方案，只允许后续 export 阶段在有严格数值证明时单独处理。

相对旧图恰好减少 7 组 Q/DQ。ONNX 仍以 `Sigmoid + Mul` 表示 SiLU，但每一处结构均为
`DQ -> Sigmoid + Mul -> Q`，Sigmoid 与 Mul 之间不再存在内部 QDQ。

## 7. Pulsar2 配置生成

项目 skill：

```text
.codex/skills/ppocrv5-rec-pulsar2-config/
```

当前 smoke 已生成：

```text
artifacts/pulsar2/ppocrv5_mobile_rec_attn_s8_native_silu_smoke_20260804.json
artifacts/pulsar2/ppocrv5_mobile_rec_attn_s8_native_silu_smoke_20260804.json.report.json
```

生成器从精确 QuantONNX 自动读取 I/O、SHA256、Attention 节点和 Q/DQ dtype，再生成 Pulsar2
`layer_configs`；不使用训练 FX 名称或旧 ONNX 编号。配置使用 `QuantONNX`、`AX650`、`NPU3`，
输入处理保持 FP32/NCHW identity，host 侧执行 BGR、等高缩放、右 padding 和 `[-1,1]` 归一化。

当前环境没有 Pulsar2 可执行程序，因此这里只完成配置生成和静态校验，不声明 AXModel 转换通过。

对旧 QuantONNX 和 `tmp/frontend/optimized.onnx` 的逐节点比较表明，frontend 将 9 组
`Transpose + MatMul + Add` 替换为 9 个 `FullyConnected`，对应 2 个 QKV Linear、2 个 Attention
projection、4 个 MLP Linear 和 1 个 CTC Linear。PyTorch 已正确使用 `nn.Linear -> aten.linear`；
`FullyConnected` 是 Pulsar2 目标算子，不迁入 Paddle→PyTorch 或 PT2E 图。除 Q/DQ domain 改为
`com.microsoft`、主 opset 21 降到 18 和属性规范化外，没有发现其他应前移的实质融合。
完整文件 hash 和算子差异记录在
`artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_frontend_optimization_20260805.json`。

## 8. 当前结论

PP-OCRv5 mobile rec 的 CTC 浮点转换与真实权重 QAT/QuantONNX smoke 已通过，结构已由用户确认。
原生 SiLU 50 epoch 正式 QAT 已启动；训练合同和指标单独记录在
`docs/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md`。Pulsar2/AXModel 板端结果仍待完成。

## 9. 原生 SiLU 浮点精度重测

2026-08-05 使用当前 `F.silu` 实现、官方 Paddle Eval 输入和完整 2077 张 ICDAR2015 rec 验证集，
在 GPU 2、batch 128 上重新执行三框架比较。普通未重参数化 route2 浮点模型结果：

```text
accuracy:                    0.5936446769684897
normalized edit similarity:  0.8171205000785132
Paddle/route2 argmax:         0.9998916706788638
Paddle/route2 probability MAE: 1.2945054054636001e-8
```

accuracy 与原生 SiLU 修改前完全一致，argmax 继续通过 `>= 0.9998` 门禁。报告：

```text
artifacts/accuracy_baseline/p2_float/rec_frameworks_native_silu_gpu2_20260805.json
```

## 10. PT2E observer-off / fake-quant-off

新增 `tools/compare_pt2e_float_preservation.py`，从同一浮点权重独立构造 QAT 捕获使用的重参数化
eager、`export_for_training` 和 prepared PT2E 图。prepared 显式调用 `disable_observer` 与
`disable_fake_quant`，完整 2077 张 GPU 2 结果：

| Stage | Accuracy | Normalized edit similarity |
| --- | ---: | ---: |
| eager reparameterized | 0.5941261434761675 | 0.8172744911430516 |
| exported float | 0.5941261434761675 | 0.8172744911430516 |
| prepared observer/fake-quant off | 0.5936446798266731 | 0.8172057106216953 |

```text
eager -> exported logits MAE/max:          0 / 0
exported -> prepared-off logits MAE/max:   0.0034760113 / 0.4623374939
centered logits MAE/max:                   0.0012416810 / 0.2682209015
probability MAE/max:                       7.8889902e-9 / 0.0617193282
argmax agreement:                          0.9999759268
CTC sequence agreement:                    0.9990370727
accuracy delta:                            -0.0004814636
normalized edit similarity delta:          -0.0000687805
```

prepared 图包含 382 个 fake-quant 模块；关闭前 observer/fake quant 为 `382/382`，关闭后为 `0/0`。
exported 的 236 个原始 state key 在 prepare 后逐项完全一致。差异来自 PT2E 对 CTC encoder 中 5 个
Conv-BN 的标准 QAT 重写，新增 `sqrt/div/mul/add/reshape` 后改变 GPU 浮点运算顺序；不是 observer 或
fake quant 仍在生效。

任务指标 delta 均小于 `0.001`，通过任务精度门禁；raw logits 的 `MAE <= 1e-6、max <= 1e-5`
严格 tensor 门禁未通过。按当前约束保留正常 PT2E 图，不增加 eager Conv-BN 固定融合。完整报告：

```text
artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_pt2e_fake_off_full_gpu2_20260805.json
```

## 11. 全局 U16/S16 与 Attention 连续 S16 Smoke

2026-08-06 将当前 QAT 合同切换为全局 U16 activation、S16 weight，并对两个 SVTR Attention 显式
配置连续 S16 域。仅修改全局 qspec 不足以满足合同；缺少 Attention 区域规则会在内部产生 U16/S16
requant。当前配置固定为：

```text
config: configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json
global activation: U16 [0, 65535]
global weight:     S16 [-32767, 32767]
QKV Linear:       U16 -> S16, weight S16
scale Mul:        S16 -> S16
MatMul 1:         S16/S16 -> S16
Softmax:          S16 -> S16
MatMul 2:         S16/S16 -> U16
```

节点由当前真实权重的 `export_for_training` FX 图重新发现，两个 Attention owner 均完整匹配。发现报告：

```text
artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_global_u16s16_qat_discovery_20260806.json
config SHA256: a9c31a796e44e6f4c061f281244337703233cb8b657d4bde3ba73d133c9b1f39
```

初始化 observer smoke 产物：

```text
/tmp/ppocrv5_mobile_rec_global_u16s16_attn_s16_qat_smoke_20260806.onnx
SHA256: fc0265b51f3e65c49df171016bcdceb10ac2c7dd0c9c3121c43bad3c43e72942
input:  [1, 3, 48, 320]
output: [1, 40, 18385]
QuantizeLinear / DequantizeLinear: 243 / 429
activation Q zero-point dtype: U16 223 / S16 20
quantized rank>=2 weight dtype: S16 47
direct DQ->Q / requant DQ->Q: 0 / 0
PyTorch/ORT argmax agreement: 1.0
probability MAE / max: 3.3771138e-7 / 0.006369978
tools/verify_qat_onnx.py -c: PASS, 0 error / 0 warning
recognized lowering: Conv 38 / Linear 9 / SiLU 7
```

导出器在一个 qparam 不同的 DQ/Q 边界间插入 Identity，位置为
`AveragePool -> Q -> DQ -> Identity -> Q -> DQ -> Conv`。该边位于 Attention 外，不是 Attention
dtype 往返；Attention 内没有 direct/requant DQ->Q。对应 Pulsar2 配置静态检查发现 2 个 Attention、
7 个 SiLU 和 1 个必要 Pool/Conv qparam 边界，检查通过。该 smoke 仍需用户完成人工结构检查，未启动
正式多 epoch QAT。

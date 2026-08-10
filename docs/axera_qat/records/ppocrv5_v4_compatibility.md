# PP-OCRv5/v4 QAT 兼容性记录

## 1. 验证范围

2026-08-03 在 route2 独立 PyTorch/QAT 工程中，对 PP-OCRv5/v4 的 mobile/server det/rec
配置执行随机权重结构 smoke。输入使用固定静态 shape，量化规则暂复用 PP-OCRv6 全局 U8 激活、S8
权重配置；结果用于验证模型构图、PT2E `prepare_qat_pt2e`、backward、`convert_pt2e`、QuantONNX
导出和 QDQ 结构，不代表对应模型已经完成预训练权重转换或真实数据训练。

统一命令模板：

```bash
env PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python \
  tools/check_model_compatibility.py \
  --task det|rec \
  --model-config configs/{det,rec}/PP-OCRv{4,5}/<model>.yml \
  --qat-config configs/qat/ppocrv6_small_{det,rec}_u8s8.json \
  --image-shape 3 128 128 \
  --output /tmp/<model>_compatibility.onnx
```

工具按 `build -> forward -> prepare -> backward -> convert -> onnx -> qdq` 分阶段报告；任一阶段
失败都返回非零状态，并输出 JSON，避免 shell pipeline 隐藏 Python 失败码。

## 2. 初始失败

初始 server smoke 能够完成构图、PT2E prepare、反向传播、convert 和 ONNX 导出，但 QDQ validator
报告：

```text
PP-OCRv5 server det: node_Conv_215, node_Conv_231 missing QDQ
PP-OCRv5 server rec: node_Conv_41, node_Conv_57 missing QDQ
PP-OCRv4 server det: Concat inputs do not share one quantization domain
PP-OCRv4 server rec: Concat inputs do not share one quantization domain
```

这不是 ONNX 导出后漏插 QDQ。反查 ONNX producer/user 拓扑可见，v5 server 的两个 Conv 都是
`Conv -> BatchNormalization -> ReLU -> QDQ`。v4 server 的失败 Concat 则是第一路使用独立 qparam，
其他支路共享另一个 qparam。

## 3. 根因

两类 server backbone 的 `ConvBNAct` 都保留了独立 BatchNorm：

```text
PP-OCRv5 server: PPHGNetV2_B4
PP-OCRv4 server: PPHGNet_small
```

PT2E 会在 annotation/prepare 过程中尝试做 Conv-BN 处理。Concat annotation 中如果使用了即将被
融合或删除的 BN 边作为共享域根，prepare 后该 edge-root 会失效或只保留部分 qspec，最终形成：

```text
Concat input 0: Q(scale_a, zero_a) -> DQ
Concat input 1..N: Q(scale_b, zero_b) -> DQ
```

因此仅修改 QuantONNX 或放宽 validator 都不能解决问题；需要在 `export_for_training` 前固定部署
形态，使 PT2E 从一开始就看到无 BN 的 Conv 图。

## 4. 修复

为两个 backbone 增加部署重参数化：

```text
pytorchocr/modeling/backbones/rec_pphgnetv2.py
  ConvBNAct.rep()
  LightConvBNAct.rep()
  StemBlock.rep()
  HGV2_Block.rep()
  HGV2_Stage.rep()
  PPHGNetV2.rep()

pytorchocr/modeling/backbones/rec_hgnet.py
  ConvBNAct.rep()
  PPHGNet.rep()
```

融合公式使用 BatchNorm 的 running statistics：

```text
W_fused = W * (gamma / sqrt(running_var + eps))
b_fused = beta - running_mean * gamma / sqrt(running_var + eps)
```

训练/QAT builder 已经在加载浮点权重后调用 `reparameterize_for_deploy()`，所以新增 `rep()` 会在
PT2E 捕获前生效。没有修改 `tmp/QAT.axera`，也没有手工修改任何 ONNX 文件。

同时，QDQ validator 现在将残留 `BatchNormalization` 视为硬失败，避免“BN 存在但 Conv 输出有
QDQ”被误报为通过。

## 5. 修复后结果

8 个配置均通过完整 staged smoke：

| 模型 | build/forward | prepare/backward | convert/ONNX | QDQ validator |
|---|:---:|:---:|:---:|:---:|
| PP-OCRv5 mobile det | 通过 | 通过 | 通过 | 通过 |
| PP-OCRv5 server det | 通过 | 通过 | 通过 | 通过 |
| PP-OCRv4 mobile det | 通过 | 通过 | 通过 | 通过 |
| PP-OCRv4 server det | 通过 | 通过 | 通过 | 通过 |
| PP-OCRv5 mobile rec CTC | 通过 | 通过 | 通过 | 通过 |
| PP-OCRv5 server rec CTC | 通过 | 通过 | 通过 | 通过 |
| PP-OCRv4 mobile rec CTC | 通过 | 通过 | 通过 | 通过 |
| PP-OCRv4 server rec CTC | 通过 | 通过 | 通过 | 通过 |

修复后的 server 关键结构统计：

```text
v5 server det: Conv outputs unquantized=0, BN=0, Concat shared=8/8
v5 server rec: Conv outputs unquantized=0, BN=0, Concat shared=8/8
v4 server det: Conv outputs unquantized=0, BN=0, Concat shared=6/6
v4 server rec: Conv outputs unquantized=0, BN=0, Concat shared=6/6
```

backbone 浮点融合数值等价测试也通过；PPHGNetV2/PPHGNet 的检测 feature 输出仅存在 FP32
融合计算的微小舍入差异，测试阈值为 `rtol=2e-3, atol=3e-5`。

## 6. PP-OCRv5 mobile det 真实权重验证

2026-08-03 下载官方预训练权重并完成转换：

```text
Paddle: PP-OCRv5_mobile_det_pretrained.pdparams
SHA256: 7e2e3b0bd5bbdcb0b842cb92aaacc2852f80299a4858b8767a45bd0c6e955648
Torch:  ptocr_v5_mobile_det.pth
```

`converter/weight_mapping.py` 对转换执行先校验、后复制的严格映射：未知 source key、缺失 target
key、重复映射或 shape 不一致都会在写入权重前失败。真实权重映射结果为：

```text
Paddle source parameters: 905
PyTorch target state keys: 1055
copied:                    905
allowed missing:           150
```

150 个允许缺失项全部是 PyTorch BatchNorm 的 `num_batches_tracked`，不存在未转换的可训练参数或
running statistics。

使用 `tools/compare_det_parity.py` 在相同随机输入和 eval running statistics 下比较 Paddle 与
PyTorch shrink map：

```text
input:          [1, 3, 128, 128]
output:         [1, 1, 128, 128]
MAE:            1.0518916467e-12
p99:            8.2991391537e-12
max_abs:        3.7744030124e-11
relative MAE:   2.2330784698e-6
finite:         true / true
```

此前工具曾报告 `MAE=0.0985`，根因不是 converter 或模型结构，而是 inference wrapper 构建后仍处于
QAT capture 状态，PyTorch backbone BN 使用当前随机 batch statistics，Paddle 使用 eval running
statistics。工具现已显式调用 `wrapper.eval()`，并有回归测试检查 wrapper 和所有 BN 的状态。

随后使用 ICDAR2015 真实数据完成 1 epoch QAT 工程 smoke；该 smoke 后又使用同一已验证浮点权重
完成独立的 50 epoch 正式 QAT：

```text
train/val:       1000 / 500 images
batch size:      4
training steps:  250
observer frozen: false
train loss:      7.103099
val loss:        4.824806
val precision:   0.0248417
val recall:      0.0245546
val hmean:       0.0246973
checkpoint:      output/icdar2015_ppocrv5_mobile_det_qat_smoke/best.pt
```

从该 checkpoint strict reload 并重新导出的 QuantONNX 结构为：

```text
input/output:           [1,3,640,640] / [1,1,640,640]
ONNX nodes:             976
Quantize/Dequantize:    250 / 465
BatchNormalization:     0
Concat shared domains:  1 / 1
HardSigmoid quantized:  10 / 10
unquantized Conv:       0
redundant DQ -> Q:      0
QuantONNX SHA256:       c3bf9d4b5fe848df41efee3fc93f5c5f850900389c5f65f50bacbfd23ef05b9a
```

ORT 禁用 graph optimization 后在 500 张 ICDAR2015 test 图上的复核结果：

```text
precision: 0.0248067010
recall:    0.0370727010
hmean:     0.0297239915
finite:    true
```

该 1-epoch 结果仅作为早期 smoke。50 epoch 正式任务的结果为：

```text
best epoch/global step: 37 / 4625
prepared CUDA hmean:   0.1652651
prepared CPU hmean:    0.1614815
QuantONNX ORT hmean:   0.1483180
```

这证明 v5 mobile det 的真实权重转换、浮点对齐、长时间 QAT、checkpoint、convert、QuantONNX 和
全验证集 ORT 链路可用，但仍不等同于 Pulsar2 编译和 AXModel 板端验收。

## 7. 当前边界

除 v5 mobile det 已完成上述真实权重 50 epoch 工程验证外，其余 v4/v5 组合目前只证明兼容结构和随机权重
QuantONNX QDQ 规则可用。正式支持仍需要：

1. v5 server/rec 和 v4 各组合的 Paddle 预训练权重转换及 Paddle/PyTorch 输出对齐；
2. 每个模型独立的 QAT profile 与 Axera 量化 JSON；
3. 其余 v5/v4 模型在对应数据集上的真实多 epoch QAT 训练和收敛后 task metric；
4. ORT optimize-off、Pulsar2 编译、AXModel 和板端前后处理回归。

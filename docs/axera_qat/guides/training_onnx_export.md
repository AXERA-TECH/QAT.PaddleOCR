# 完整训练结构 ONNX 导出

## 1. 目标与合同

2026-08-06 为以下四个完整 Paddle pretrained-training 模型导出 Float ONNX 和 PT2E
QuantONNX：

```text
PP-OCRv5 mobile rec
PP-OCRv5 mobile det
PP-OCRv6 small rec
PP-OCRv6 small det
```

这里的“训练结构”指保留训练阶段辅助输出，不包含 loss 和 optimizer：

```text
v5-rec: images + gtc_targets -> ctc, ctc_neck, gtc
v5-det: images               -> maps(shrink/threshold/binary)
v6-rec: images + gtc_targets -> ctc, ctc_neck, gtc
v6-det: images               -> maps, aux_maps_p4, aux_maps_p3, aux_maps_p2
```

识别模型固定 `max_text_length=25`，`gtc_targets` shape 为 `[B,25]`，NRTR 输出 24 个 decoder
step。原 Paddle eager graph 按 batch 内最长 label 动态裁剪；导出图计算完整 padding 序列，NRTRLoss
仍按 `ignore_index=0` 忽略 padding，因此有效 token 合同不变。

Float ONNX 默认不重参数化，用于检查 Paddle 原始训练 backbone 和全部辅助头。QuantONNX 在 PT2E
capture 前默认重参数化，这是现有 Axera QAT 合同；否则 PPLCNetV3 identity BN 等独立训练分支无法
被 PT2E Conv-BN rewrite 消除。重参数化只改变等价 backbone 实现，训练输出 schema 保持不变。

## 2. 导出工具

统一入口：

```bash
env PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python \
  tools/export_ocr_onnx.py training \
  --format both \
  --output-dir exports/training
```

支持 `--models` 选择子集，`--format float|quantonnx|both` 分阶段执行。`--quant-graph` 控制
QuantONNX 保留完整训练结构还是只保留部署推理路径。报告采用增量合并，单模型失败或只重跑一种
格式不会覆盖其他已通过记录。默认策略：

```text
--no-float-reparameterize
--quant-reparameterize
--quant-graph training
--onnx-optimize
--ort-check
```

### 2.1 推理 QuantONNX

推理图参考 Ultralytics `model.export(format="onnx")` 的部署语义，在 PT2E capture 前切换为部署
wrapper，而不是导出完整图后按节点名删除：

```bash
env PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python \
  tools/export_ocr_onnx.py training \
  --models ppocrv5_mobile_rec \
  --format quantonnx \
  --quant-graph inference \
  --output-dir exports/inference
```

推理合同为：

```text
det: images -> maps（仅 DB shrink，移除 threshold/binary 和 aux maps）
rec: images -> logits（仅 CTC，移除 ctc_neck、NRTR/gtc 和 gtc_targets）
```

因此被移除分支的参数、observer 和 QDQ 节点不会进入 QuantONNX。`training` 仍是默认值，已有命令
行为不变。

已有产物可用 `tools/export_ocr_onnx.py audit --output-dir exports/training` 重跑 ONNX full checker；该
子命令不会重新构建或导出模型。

### 2.2 Float 重参数化矩阵

需要同时检查原始 Conv-BN 结构和部署重参数化结构时，使用独立的 `float-matrix` 子命令。默认对
PP-OCRv5 mobile det/rec 和 PP-OCRv6 small det/rec 导出训练图、推理图，以及两种重参数化状态，
共 16 份 Float ONNX：

```bash
env PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python \
  tools/export_ocr_onnx.py float-matrix \
  --models ppocrv5_mobile_det ppocrv5_mobile_rec \
           ppocrv6_small_det ppocrv6_small_rec \
  --graphs training inference \
  --reparameterizations reparameterized non_reparameterized \
  --output-dir exports/float_onnx/v5_v6_reparameterization_matrix \
  --onnx-optimize --ort-check
```

输出命名规则：

```text
<model>/<model>_<training|inference>_<reparameterized|non_reparameterized>_float.onnx
```

训练图保留原始辅助输出：det 保留 shrink/threshold/binary 及 v6 auxiliary maps，rec 保留 CTC、CTC
neck 和 GTC/NRTR；推理图只保留 det shrink map 或 rec CTC logits。输入 shape 固定为 det
`[1,3,640,640]`、rec `[1,3,48,320]`，训练 rec 额外输入 `gtc_targets [1,25]`。

非重参数化 Float 图会保留模型定义中的 BatchNormalization，用于检查原始训练结构，不能直接作为
Axera QuantONNX 或 Pulsar2 输入。重参数化 Float 图中的 Conv-BN 已折叠，BN 应为 0；如果后续要
进入 PT2E QAT，必须基于对应结构重新 `export_for_training -> prepare`，不能混用两种状态的
checkpoint。

共同导出流程包含 `onnx_program.optimize()`、zero-point Cast 折叠、完全相同冗余 DQ/Q 删除、不同
qparam 的直接 DQ/Q Identity 隔离和 ONNX full checker。QuantONNX 额外要求：

- QAT JSON 中所有 regional node 名在本次 prepared 图中命中；
- activation observer 保持 `scale=1.0, zero_point=0`；
- 非浮点 token/index/mask 边不得进入 QAT 激活域；
- BN、Conv-QDQ-BN、冗余 DQ/Q 和未量化 Conv 输出均为 0；
- Add/Concat 输入共享 Axera 量化域；
- ORT 使用 `ORT_DISABLE_ALL` 与 converted PT2E 比较全部输出。

## 3. 产物

```text
exports/training/float/ppocrv5_mobile_rec_training_float.onnx
exports/training/float/ppocrv5_mobile_det_training_float.onnx
exports/training/float/ppocrv6_small_rec_training_float.onnx
exports/training/float/ppocrv6_small_det_training_float.onnx

exports/training/quantonnx/ppocrv5_mobile_rec_training_init_qat.onnx
exports/training/quantonnx/ppocrv5_mobile_det_training_init_qat.onnx
exports/training/quantonnx/ppocrv6_small_rec_training_init_qat.onnx
exports/training/quantonnx/ppocrv6_small_det_training_init_qat.onnx

exports/inference/quantonnx/ppocrv5_mobile_rec_inference_init_qat.onnx

exports/training/training_onnx_report.json
```

`training_init_qat` 明确表示本批 QuantONNX 是完整训练结构的初始化-observer 图，`training_steps=0`；
它不是多 epoch QAT checkpoint 的导出结果。权重 observer 仅用静态权重初始化，未运行推理、校准或训练。
`inference_init_qat` 同样表示初始化 observer，但其报告键为 `quantonnx_inference`，不会覆盖完整训练图
的 `quantonnx` 记录。

## 4. 验证结果

Float ONNX 的 PyTorch -> ORT optimize-off 最大绝对误差：

| 模型 | 输出数 | 最大 max_abs |
|---|---:|---:|
| PP-OCRv5 mobile rec | 3 | `6.58035e-5` |
| PP-OCRv5 mobile det | 1 | `8.34465e-7` |
| PP-OCRv6 small rec | 3 | `3.96371e-5` |
| PP-OCRv6 small det | 4 | `1.10418e-5` |

QuantONNX 的 converted PT2E -> ORT optimize-off 全部输出 MAE/max_abs 均为 `0/0`。结构摘要：

| 模型 | Q / DQ | BN | 未量化 Conv | Add 域 | Concat 域 |
|---|---:|---:|---:|---:|---:|
| PP-OCRv5 mobile rec | 412 / 531 | 0 | 0 | 78/78 | 1/1 |
| PP-OCRv5 mobile det | 258 / 384 | 0 | 0 | 63/63 | 2/2 |
| PP-OCRv6 small rec | 328 / 479 | 0 | 0 | 35/35 | 1/1 |
| PP-OCRv6 small det | 224 / 404 | 0 | 0 | 21/21 | 6/6 |

详细 input/output shape、observer 数、节点数、QDQ dtype 和逐输出误差以
`exports/training/training_onnx_report.json` 为准。

## 5. 推理 QuantONNX Smoke

2026-08-06 使用初始化 observer 验证 `--quant-graph inference`：

| 模型 | 输入 | 输出 | 辅助分支名称 | ORT_DISABLE_ALL |
| --- | --- | --- | ---: | --- |
| PP-OCRv5 mobile rec | `images [1,3,48,320]` | `logits [1,40,18385]` | 0 | converted 对比 MAE/max `0/0` |
| PP-OCRv6 small det | `images [1,3,640,640]` | `maps [1,1,640,640]` | 0 | 输出有限 |

产物：

```text
/tmp/ppocrv5_mobile_rec_inference_quantonnx_smoke_20260806/quantonnx/
  ppocrv5_mobile_rec_inference_init_qat.onnx

/tmp/ppocrv6_small_det_inference_quantonnx_smoke_20260806/quantonnx/
  ppocrv6_small_det_inference_init_qat.onnx
```

> 注意：`exports/inference/` 目录当前不存在，上述推理 smoke 产物仅在 `/tmp/` 下保留；推理导出
> 示例应重新执行生成，或按 `--output-dir` 指定新的保留目录。

rec 图没有 `gtc_targets` 输入，也没有 `gtc_head/before_gtc/NRTR` 节点或 initializer；det 图没有
`threshold/aux_maps/aux_binarize/aux_thresh` 节点或 initializer。两张图均通过 ONNX checker 和项目
QDQ 结构验证，未发现 BN、冗余 DQ/Q 或未量化 Conv 输出。

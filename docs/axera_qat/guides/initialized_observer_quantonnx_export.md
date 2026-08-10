# 初始化 Observer QuantONNX 导出记录

## 1. 目标

2026-08-05 为以下模型导出初始化量化参数的 QuantONNX：

- PP-OCRv5 mobile det；
- PP-OCRv6 small det；
- PP-OCRv6 small rec CTC。

本次不执行 prepared/converted 模型推理、不运行 ORT、不校准、不训练。激活 observer 不接触输入，
保持初始化状态：

```text
activation scale:      1.0
activation zero-point: 0
inference runs:        0
calibration runs:      0
training steps:        0
```

PT2E 的 per-channel S8 权重和偏置 observer 不能保持标量初始化状态，否则 convert 后的 qparam
维度与输出通道数不匹配。因此导出器只求值不依赖输入 placeholder 的静态参数子图，用模型权重初始化
per-channel observer。该过程不会执行激活路径。`torch.export` 和 `torch.onnx.export` 仍会进行必要的
图捕获，但不会更新已经移除 observer 的 converted 图。

## 2. 导出工具

入口：

```text
tools/export_ocr_onnx.py initialized
```

处理顺序：

1. 从浮点预训练权重构建并 reparameterize 模型；
2. `export_for_training -> prepare_qat_pt2e`，不调用 prepared forward；
3. 只初始化 per-channel 静态参数 observer；
4. 检查所有 per-tensor 激活 observer 均为 `scale=1/zero_point=0`；
5. `convert_pt2e -> torch.onnx.export -> onnx_program.optimize()`；
6. 执行 ONNX checker、QDQ 结构门禁和 QuantizeLinear qparam 静态审计；
7. 不创建 ORT session，不比较模型输出。

示例：

```bash
$PYTHON tools/export_ocr_onnx.py initialized \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth \
  --qat-config configs/qat/ppocrv6_small_rec_u8s8.json \
  --image-shape 3 48 320 \
  --output /home/heqi/project/PaddleOCR/tmp/route2_qat_exports/initialized_observers/ppocrv6_small_rec_init_observer_qat.onnx
```

默认报告路径为 `<output>.json`。

## 3. 导出结果

输出目录：

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/initialized_observers/
```

| 模型 | 输入 | QuantizeLinear | 静态参数 observer | scale 唯一值 | zero-point 唯一值 |
| --- | --- | ---: | ---: | --- | --- |
| PP-OCRv5 mobile det | 1x3x640x640 | 250 | 120 | `{1.0}` | `{0}` |
| PP-OCRv6 small det | 1x3x640x640 | 178 | 160 | `{1.0}` | `{0}` |
| PP-OCRv6 small rec | 1x3x48x320 | 158 | 118 | `{1.0}` | `{0}` |

三个模型均通过 ONNX full checker 和项目 QDQ 结构门禁。识别模型保留配置中的局部 S16 激活域，
其 zero-point dtype 为 `int16`，但数值仍为 `0`；全局激活域为 U8。

SHA256：

```text
b84ab91e1bcbd82670e6a4dc9a8089573a390a645e2215ba18545de920418fda  ppocrv5_mobile_det_init_observer_qat.onnx
55c798a37b0c3685b96d9b6d470f2ef38f34ecd9d9addd9ae36713199e29dc9f  ppocrv6_small_det_init_observer_qat.onnx
402361065e39d5c7943c8b61b830c96abd74dd6ffa3c74b0c74d58f07012886b  ppocrv6_small_rec_init_observer_qat.onnx
```

这些模型用于量化图结构和工具链规则分析，不代表校准后或 QAT 后的可用精度，不能用于精度验收。

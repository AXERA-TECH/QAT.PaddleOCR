---
name: ppocr-pulsar2-config
description: Generate and validate a generic Axera Pulsar2 QuantONNX conversion config from the exact ONNX graph, with optional explicit model-specific layer rules and frontend-node validation.
---

# PP-OCR Pulsar2 配置生成

用于 PP-OCR det/rec 或其他已经完成 PT2E QAT、QDQ 检查的 QuantONNX。该 skill 负责生成 Pulsar2
通用配置骨架、保留用户显式提供的 `layer_configs`，并验证配置与实际 ONNX/frontend 图的一致性。

使用 `--profile ppocrv5-rec` 可启用原 PP-OCRv5 rec 专项规则：自动发现两组 SVTR Attention，校验
S8/S16 边界、SiLU 边界 QDQ 和必要 requant。该 profile 仍只适用于满足对应结构合同的 v5-rec，不能
用于 v6 det/rec。

## 工作边界

- QuantONNX 导出、QDQ 结构和精度检查先按 `ppocr-pt2e-qat` 及仓库根目录 README 完成；本 skill 不修改 ONNX。
- 不从模型名猜测 Attention、DetHead、SVTR 或其他区域规则；模型专属 `layer_configs` 必须由用户提供，
  或使用对应的专项 skill 自动发现。
- QuantONNX 已经包含 Q/DQ。Pulsar2 的 `calibration_dataset` 仅满足工具链字段要求，不能覆盖 QAT qparams
  或把编译过程当作重新 PTQ。
- 输入处理默认生成 FP32/NCHW identity processor；如果模型使用其他 layout、dtype 或预处理，必须显式调整
  参数，并保证主机/板端输入与之匹配。

## 生成

```bash
python3 .codex/skills/ppocr-pulsar2-config/scripts/generate_pulsar2_config.py \
  --profile ppocrv5-rec \
  --onnx /path/to/ppocrv5_mobile_rec_qdq.onnx \
  --output /path/to/config/ppocrv5_mobile_rec.json \
  --output-dir /path/to/output/ppocrv5_mobile_rec \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --attention-dtype S8 \
  --calibration-dataset /path/to/calibration.zip
```

通用模式示例：

```bash
python3 .codex/skills/ppocr-pulsar2-config/scripts/generate_pulsar2_config.py \
  --onnx /path/to/model_qdq.onnx \
  --output /path/to/config/model.json \
  --output-dir /path/to/output/model \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --calibration-dataset /path/to/calibration.zip
```

通用模式如果需要模型专属区域覆盖，使用 JSON 文件显式提供 `layer_configs`：

```bash
python3 .codex/skills/ppocr-pulsar2-config/scripts/generate_pulsar2_config.py \
  --onnx /path/to/model_qdq.onnx \
  --output /path/to/config/model.json \
  --layer-configs /path/to/layer_configs.json \
  --frontend-onnx /path/to/frontend/optimized.onnx
```

`layer_configs.json` 必须是 Pulsar2 `quant.layer_configs` 数组；每个覆盖项至少包含
`layer_names`。传入 `--frontend-onnx` 时，名称按 frontend 优化图校验；否则按原始 QuantONNX 校验。

## 校验

```bash
python3 .codex/skills/ppocr-pulsar2-config/scripts/validate_pulsar2_config.py \
  --onnx /path/to/model_qdq.onnx \
  --config /path/to/config/model.json \
  --frontend-onnx /path/to/frontend/optimized.onnx
```

v5-rec profile 校验示例：

```bash
python3 .codex/skills/ppocr-pulsar2-config/scripts/validate_pulsar2_config.py \
  --profile ppocrv5-rec \
  --onnx /path/to/ppocrv5_mobile_rec_qdq.onnx \
  --config /path/to/config/ppocrv5_mobile_rec.json \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --attention-dtype S8
```

QuantONNX、frontend 图、目标硬件、输入 processor 或 `layer_configs` 发生变化后重新校验。校验通过后，
在已安装 Pulsar2 的环境中执行：

```bash
pulsar2 build \
  --input /path/to/model_qdq.onnx \
  --config /path/to/config/model.json \
  --output_dir /path/to/output/model
```

编译完成后检查 `compiled.axmodel` 和 `frontend/optimized.onnx`；frontend 若重命名、融合或拆分节点，
应使用新 frontend 图重新校验或生成配置。

# PP-OCR QuantONNX / Pulsar2 交接清单

最后更新：2026-08-10

## 1. 交付状态

当前已保留 PP-OCRv6 small det/rec 和 PP-OCRv5 mobile det 的 QAT QuantONNX，均完成 ONNX checker、
QDQ 结构检查和 ORT optimize-off 评估记录。PP-OCRv5 mobile rec 当前只有初始化训练图和 Exp2 epoch-2
debug 结构图，训练因精度门禁失败而暂停，不能作为正式精度交付。

本机没有 `pulsar2` 可执行程序、Axera 编译环境或板端运行环境，因此本文只定义可复现的编译和验收
契约，不把 ONNX/ORT 通过写成 AXModel 验收通过。当前没有已编译的 `.axmodel`。

Pulsar2 必须为 3.4 或更高版本；`tmp/QAT.axera/pulsar2` 只包含上游说明、示例 `config.json` 和
4-bit 转换脚本，不包含编译器。

## 2. 正式输入模型

### 2.1 PP-OCRv6 small det

```text
QuantONNX: runs/icdar2015_ppocrv6_small_det_qat/best_qdq.onnx
input:     inputs_0 [1, 3, 640, 640] FP32
output:    maps     [1, 1, 640, 640] FP32
nodes:     761
Q / DQ:    177 / 335
domains:   U8 activation 177
BN:        0
Concat:    2 / 2 shared qparams
HardSigmoid: 13 / 13 quantized
unquantized Conv: 0
redundant / requant DQ->Q: 0 / 0
```

ORT optimize-off 验证集结果：precision `0.580576`、recall `0.388541`、hmean `0.465532`。

### 2.2 PP-OCRv6 small rec CTC

```text
QuantONNX: runs/icdar2015_ppocrv6_small_rec_qat/best_qdq.onnx
input:     images [1, 3, 48, 320] FP32
output:    logits [1, 40, 18710] FP32
nodes:     682
Q / DQ:    161 / 280
domains:   U8 143 / S16 18
BN:        0
Concat:    1 / 1 shared qparams
HardSigmoid: 5 / 5 quantized
unquantized Conv: 0
redundant / requant DQ->Q: 0 / 2
```

两个 requant 均为 LightSVTR Attention 的 S16 MatMul 输出切换到 U8 域，不是可删除的冗余边。
ORT optimize-off 验证集结果：accuracy `0.719788`、normalized edit similarity `0.885338`。

### 2.3 PP-OCRv5 mobile det

```text
QuantONNX: runs/icdar2015_ppocrv5_mobile_det_qat/best_qdq.onnx
input:     inputs_0 [1, 3, 640, 640] FP32
output:    maps     [1, 1, 640, 640] FP32
nodes:     980
Q / DQ:    250 / 469
BN:        0
unquantized Conv: 0
```

该文件来自修复后的真实权重 QAT 训练链路，仍需在目标 Pulsar2 版本和板端完成编译与精度验收。
Pad/QDQ 规则曾导致 Axera 转换精度异常，编译前必须重新执行 QDQ 审计，不能直接沿用历史转换配置。

### 2.4 PP-OCRv5 mobile rec

正式 QAT 训练目录：

```text
runs/exp2_ppocrv5_mobile_rec_u16s16_full_qat/
```

当前只允许交付以下两个文件做结构检查：

```text
training graph:
exports/training_reparameterize_static_scalar_fakequant_fix_review/quantonnx/ppocrv5_mobile_rec_training_init_qat.onnx

deployment/debug graph:
exports/quantonnx/exp2_ppocrv5_mobile_rec_u16s16_epoch2_debug_qat.onnx
```

training graph 保留 `images + gtc_targets` 双输入和 CTC/CTC neck/GTC 三个训练输出；deployment/debug
graph 只保留 `images -> logits`。Exp2 在 epoch 2 因 accuracy 从浮点基线 `0.5936447` 降至
`0.0019259` 被立即停止。该模型未通过精度交付门禁，禁止进入 Pulsar2 最终编译验收。

## 3. Pulsar2 编译

先确认工具链：

```bash
pulsar2 version
```

为 det 和 rec 分别创建编译目录及配置。可从 `tmp/QAT.axera/pulsar2/config.json` 起步，但必须确认：

- `model_type` 为 `QuantONNX`；
- target hardware 和 `npu_mode` 与实际芯片一致；
- `compiler.check` 保持 `2`；
- 不对已有 QDQ 模型重新执行 PTQ；
- input tensor 名称分别为 `inputs_0` 和 `images`；
- 输入 layout、RGB 顺序、归一化和 padding 不由编译配置重复执行。

编译命令模板：

```bash
pulsar2 build \
  --config det_config.json \
  --input runs/icdar2015_ppocrv6_small_det_qat/best_qdq.onnx \
  --output_dir artifacts/pulsar2/ppocrv6_small_det

pulsar2 build \
  --config rec_config.json \
  --input runs/icdar2015_ppocrv6_small_rec_qat/best_qdq.onnx \
  --output_dir artifacts/pulsar2/ppocrv6_small_rec
```

编译日志必须与 `compiled.axmodel` 一起保留，并记录 Pulsar2 版本、target hardware、输入输出、配置
路径和所有 warning。后续实验不生成或记录文件、配置、数据、checkpoint、ONNX 或 AXModel 的哈希值。
若编译器报告 unsupported op 或非法量化域，应回到 PT2E QAT
图或量化配置修复，不能手工改写最终 ONNX 的 QDQ。

## 4. 板端精度契约

det 前处理必须复用训练时的确定性居中 letterbox：原 ICDAR2015 `1280x720` 图片保持比例缩放到
`640x360`，上下各 padding 140，padding 值为 114；polygon 使用同一缩放和平移。后处理使用相同
DBPostProcess 参数和 polygon metric，输出 precision/recall/hmean。

rec 输入固定为 `3x48x320`，保持比例 resize 后右侧 padding；字典、`use_space_char`、CTC blank、
相邻重复折叠和空格处理必须与 `RecMetric` 一致。模型输出是 raw logits，板端不可在 CTC 解码前
额外改变时间维或类别顺序。

验收顺序：

1. 固定 8 张样本比较 ORT optimize-off 与 AXEngine 输出，记录 MAE、max_abs 和任务序列一致率；
2. det 跑完整 500 张验证集，rec 跑完整 2077 张验证集；
3. 同时记录 AXEngine 版本、芯片、频率、batch 和前后处理代码 commit；
4. 若指标下降，先比较模型原始输出，再定位前处理、编译或后处理边界。

ORT 默认 graph optimization 会改变当前 QDQ 图的任务指标，不能作为板端 reference。所有 QDQ 语义
基准使用 `ORT_DISABLE_ALL`。

## 5. PP-OCRv5 mobile rec 候选配置

当前 v5-rec smoke 使用全局 U16 activation、S16 weight，并为两个 SVTR Attention 显式建立连续
S16 域。最新 QuantONNX 为：

```text
/tmp/ppocrv5_mobile_rec_global_u16s16_attn_s16_qat_smoke_20260806.onnx
input:  images [1,3,48,320] FP32
output: logits [1,40,18385] FP32
Pad / BN: 0 / 0
activation Q zero-point: U16 223 / S16 20
quantized rank>=2 weight: S16 47
Attention: 2 x (QKV U16->S16, MatMul S16/S16->S16, Softmax S16, MatMul S16/S16->U16)
SiLU: 7 x boundary-only QDQ，内部 QDQ 0
direct/requant DQ->Q: 0 / 0
Identity-isolated boundary: 1 x U16->U16，位于 Attention 外 Pool/Conv qparam 边界
```

该合同必须显式包含 Attention S16 区域。只有全局 U16/S16 而没有 QKV、scale Mul、两次 MatMul 和
Softmax 的区域规则，会在 Attention 内重新落回 U16 域并产生 requant，不能视为完整 16-bit 配置。

Pulsar2 配置不得复制旧 ONNX 节点名，使用项目 skill 从该精确文件生成和校验。当前静态校验产物为：

```text
.codex/skills/ppocrv5-rec-pulsar2-config/
/tmp/ppocrv5_mobile_rec_global_u16s16_attn_s16_pulsar2_20260806.json
/tmp/ppocrv5_mobile_rec_global_u16s16_attn_s16_pulsar2_20260806.report.json
```

配置固定 `model_type=QuantONNX`、`target_hardware=AX650`、`npu_mode=NPU3`，并为 QKV output、
Attention S16 core 和第二 MatMul S16 input 生成三组 `layer_configs`。第二 MatMul output 由 ONNX
QDQ 回到全局 U16。静态校验发现两个 Attention、7 个 SiLU 和一个 Attention 外 Pool/Conv 必要
qparam 边界；该边保持可见，不由配置伪装消除。

Pulsar2 frontend 会把该模型已有 9 个 `nn.Linear` 对应的 `Transpose + MatMul + Add` 降低为
`FullyConnected`。这是 rank-3 Linear 的目标端 lowering，不要求 Paddle→PyTorch 输出自定义
FC/Gemm；保持 `nn.Linear -> aten.linear` 才能继续使用现有 PT2E Linear annotator。

当前文件是初始化 observer 的预训练 smoke，不是正式 QAT 权重。Exp2 的 debug QuantONNX 也只用于
结构检查，不能作为正式编译输入。用户确认新的 smoke 结构且真实数据阶段精度门禁通过后，才可以从
best checkpoint 导出正式 QuantONNX，并必须针对最终文件重新生成和校验配置；不能复用历史节点名、
qparams 或配置报告。

## 6. PP-OCRv5/v4 扩展契约

训练和导出入口已经按 `task + Paddle model YAML + training profile + QAT JSON` 配置驱动，但这不等于
任意 PaddleOCR YAML 自动受到支持。每个 v5/v4 模型至少需要：

1. 将对应 Paddle YAML 和字典带入当前仓库，并确认所有 backbone/neck/head 在 PyTorch 仓库中实现；
2. 转换浮点权重，执行 Paddle/PyTorch 同输入输出对齐；
3. 明确 det 单输出部署分支或 rec CTC 分支，不把训练辅助头带入部署图；
4. 新建模型专属 training profile 和 QAT JSON，不能默认复用 v6 observer 区域编号；
5. 检查是否需要 `rep()`、固定 H/W、Attention 局部 S8/S16、Concat shared domain 和 hard activation 规则；
6. 从浮点权重重新 `export_for_training -> prepare_qat_pt2e`，完成真实数据 backward 和 strict reload；
7. 重复 checker、QDQ 结构、ORT optimize-off、Pulsar2 和板端全量指标验收。

当前仓库已有 v5/v4 mobile/server det/rec 的兼容 YAML，并已通过随机权重 PT2E/QuantONNX
结构 smoke。尚无对应的正式 PyTorch 转换权重、专属 profile 或模型专属 QAT JSON，因此不能将
结构 smoke 视为真实模型训练或板端支持。

## 7. 当前交接入口

完整项目流程见仓库根目录 `README.md`；项目约束见 `AGENTS.md`。PP-OCRv5 rec Exp2/Exp3 的训练
门禁、保留 checkpoint、QuantONNX 和参数域审计结果见：

```text
docs/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md
```

接手者应先阅读上述训练记录，再按本文件第 3、4 节对目标 QuantONNX 执行 Pulsar2 编译和板端验收；
不要把历史 `output/` 路径下的模型当作当前交付产物。当前模型产物统一位于 `runs/` 或 `exports/`。

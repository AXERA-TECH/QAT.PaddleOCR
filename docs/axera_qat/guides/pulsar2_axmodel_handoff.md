# PP-OCR QuantONNX / Pulsar2 交接清单

最后更新：2026-08-10

## 1. 交付状态

当前已保留 PP-OCRv6 small det/rec 和 PP-OCRv5 mobile det 的 QAT QuantONNX，均完成 ONNX checker
和 ORT optimize-off 评估记录。注意：v6 rec/det 与 v5 det 的 `best_qdq.onnx` 生成于 2026-07-30，
早于 SiLU 边界 QDQ 门禁；用当前 `validate_qdq_graph` 复核时 v6 rec 图因 4 个内部 QDQ 的 SiLU
被标记 blocked（v6 det 与 v5 det 通过），编译前必须重新执行 QDQ 审计。
PP-OCRv5 mobile rec 的 Exp1-Exp4 均因 epoch-2 精度门禁停止（acc 0.0019 / 0.3899 / 0.3717），
只有初始化训练图和 Exp2/Exp3/Exp4 epoch-2 debug 结构图，不能作为正式精度交付。

本机没有 `pulsar2` 可执行程序、Axera 编译环境或板端运行环境，因此本文只定义可复现的编译和验收
契约，不把 ONNX/ORT 通过写成 AXModel 验收通过。当前没有已编译的 `.axmodel`。

Pulsar2 必须为 3.4 或更高版本；本机 `tmp/QAT.axera/pulsar2` 目录已不存在（旧的上游说明与
示例 `config.json` 已随重构清理），配置从第 3 节的模板或既有 `artifacts/pulsar2/` 配置起步。

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

当前只允许交付以下文件做结构检查：

```text
training graph:
exports/training_reparameterize_static_scalar_fakequant_fix_review/quantonnx/ppocrv5_mobile_rec_training_init_qat.onnx

deployment/debug graphs（exp2/exp3/exp4 各 epoch-2 debug 图）:
exports/quantonnx/exp2_ppocrv5_mobile_rec_u16s16_epoch2_debug_qat.onnx
exports/quantonnx/exp3_ppocrv5_mobile_rec_u16s16_epoch2_debug_no_reparameterize_qat.onnx
/tmp/exp4_kd_debug_qat.onnx（KD 图，见记录 §29.3）
```

training graph 保留 `images + gtc_targets` 双输入和 CTC/CTC neck/GTC 三个训练输出；deployment/debug
graph 只保留 `images -> logits`。Exp2 在 epoch 2 因 accuracy 从浮点基线 `0.5936447` 降至
`0.0019259` 被立即停止。该模型未通过精度交付门禁，禁止进入 Pulsar2 最终编译验收。

## 3. Pulsar2 编译

先确认工具链：

```bash
pulsar2 version
```

为 det 和 rec 分别创建编译目录及配置。可从已有 `artifacts/pulsar2/` 配置或下方模板起步，但必须确认：

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

**当前合同（2026-08-18）**:U8/S8 全局 + Attention S8 + 下采样链 S16（exp15/16）。
正式 QuantONNX 与 Pulsar2 配置以 exp16 best checkpoint 产物为准:

```text
QuantONNX: exports/quantonnx/exp16_u8s8_downsample_s16_reparam/
           ppocrv5_mobile_rec_exp16_u8s8_downsample_s16_reparam_qdq.onnx
Pulsar2:   artifacts/pulsar2/exp16_u8s8_downsample_s16/
           ppocrv5_mobile_rec_exp16_u8s8.json(.report.json)
结构:      941 节点、Q 243 / DQ 429、激活 zp dtype S8 20 / U8 198(另 S16 25 于下采样链)、
           Attention 2 x (QKV S8, MatMul S8/S8->S8, Softmax S8, MatMul S8/S8->U8)、
           Identity 1(avg_pool 输出 U8->U8 尺度边界)
```

- 板端验收（2026-08-18）:exp16 上板 acc **0.58449** vs ORT 全量 **0.58546**
  （差 0.001）;exp13 的 0.044 上板差距已随下采样链 S16 消除;
- 编译注意:远端 frontend 会把 QKV 的 `MatMul + Add` 融合为 `op_N:onnx.FullyConnected`,
  Pulsar2 配置的 QKV `layer_names` 必须按 `output_dir/frontend/optimized.onnx` 实际节点
  remap（exp16 为 `op_8`/`op_12`）,本地生成配置与远端 remap 版本可不同;
- 下采样链 S16 域包含 conv2d_29-32、lab mul/add 链与 avg_pool 输入,注意 S16 域 dyadic
  标量 Mul 的 Axera 约束（操作数序与 int16 标量）已在导出端处理,重训/重导后需复核。

**历史合同（已替代,仅追溯）**:2026-08-06 全局 U16 activation、S16 weight + Attention
连续 S16 域（`/tmp/ppocrv5_mobile_rec_global_u16s16_attn_s16_qat_smoke_20260806.onnx`,
U16 223 / S16 20）。该合同要求显式包含 QKV、scale Mul、两次 MatMul 和 Softmax 的区域规则,
否则 Attention 内落回 U16 域产生 requant;现已被 U8/S8 部署位宽路线替代（U16/S16 最优
exp12a 0.6172,仅作精度上限参考,不用于部署）。

Pulsar2 配置不得复制旧 ONNX 节点名，使用项目 skill 从最终 QuantONNX 精确文件生成和校验:

```text
.codex/skills/ppocrv5-rec-pulsar2-config/
```

配置固定 `model_type=QuantONNX`、`target_hardware=AX650`、`npu_mode=NPU3`。Pulsar2 frontend
会把 `nn.Linear` 对应的 `Transpose + MatMul + Add` 降低为 `FullyConnected`,这是 rank-3
Linear 的目标端 lowering,不要求 Paddle→PyTorch 输出自定义 FC/Gemm;保持
`nn.Linear -> aten.linear` 才能继续使用现有 PT2E Linear annotator。

每次从新 checkpoint 导出 QuantONNX 后,必须针对最终文件重新生成和校验配置;不能复用历史
节点名、qparams 或配置报告。

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

完整项目流程见仓库根目录 `README.md`；项目约束见 `AGENTS.md`。PP-OCRv5 rec Exp2/Exp3/Exp4 的
训练门禁、保留 checkpoint、QuantONNX 和参数域审计结果见：

```text
docs/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md
```

接手者应先阅读上述训练记录，再按本文件第 3、4 节对目标 QuantONNX 执行 Pulsar2 编译和板端验收；
不要把历史 `output/` 路径下的模型当作当前交付产物。当前模型产物统一位于 `runs/` 或 `exports/`。

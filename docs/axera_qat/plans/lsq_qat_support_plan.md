# LSQ(可学习步长量化)QAT 支持计划

状态：S0-S3 已完成（2026-08-10）。`ax_quantizer_lsq.py` 与 utils 支持、`"lsq": true` 桥接开关、
训练入口 `--lsq`/profile `lsq` + epoch-0 两阶段统计已落地；全量回归 242 passed + 16 subtests；
train.py 端到端 smoke（真实权重 + lsq config，1 epoch）通过，LSQ scale 参数入 checkpoint。

## 1. 目标与动机

当前 v5/v6 QAT 使用 Axera vendored 的统计 observer：activation 走
`FakeQuantize.with_args(eps=2**-12)`（U16/S16 分支）或 `FusedMovingAvgObsFakeQuantize`，
scale 由输入统计估计。已知问题：U16 observer `eps=2**-12` 会把 `[-1,1]` 输入的 scale 截断到
`2**-12`（Exp2-Exp4 精度崩溃主因），且统计 observer 无法让 scale 随训练自适应。

本计划引入 **LSQ（Learned Step Size Quantization, Esser et al., CVPR 2020）**：
量化步长 `s` 作为**可学习参数**参与训练（STE + LSQ 梯度），使每层激活/权重 scale 由任务损失
直接驱动，替代/增强统计 observer。

目标：

```text
1. 激活（U16/U8）与权重（S16/S8）支持可学习 scale，替代统计 observer 或与之并行
2. 保持 PT2E prepare/convert、checkpoint resume、QuantONNX 导出链路不变
3. 通过 QAT JSON 开关选择 observer 类型（默认仍为统计 observer，不影响现有合同）
4. 以 Exp5（修复 eps 基线）为对照，验证 LSQ 是否能恢复/超越 v5 rec QAT 精度
```

边界（本计划不做）：

```text
1. 不改 Axera vendored quantizer 的 dtype/量化域规则；LSQ 只替换 scale 的来源，
   U16 激活域、S16 权重域、Attention S16、Concat 共享域等 QAT JSON 合同不变
2. 不做 per-channel 激活 LSQ（Axera 激活域为 per-tensor）
3. 不改 Pulsar2 配置/编译规则（QDQ 输出格式不变）
4. 不改 references/ 仓库
```

## 2. 当前实现基线（已核实）

- `pytorchocr/quantization/ax_quantizer.py:95-160`：activation qspec 按 dtype 分支，
  U16/S16 用 `FakeQuantize`（`extra_args={"eps": 2**-12}`），其余用
  `FusedMovingAvgObsFakeQuantize`；权重 S16 用 `FakeQuantize`（L166-205）；
- `pytorchocr/quantization/bridge.py:92-130`：`AxeraQuantizerAdapter` 在 annotate 后做
  scalar 权重 qspec、attention mask float、Concat 共享域等图级修正；
- `pytorchocr/quantization/bridge.py:564` `prepare_qat_model`：
  `export_for_training -> prepare_qat_pt2e -> move_exported_model_to_train`；
- `initialize_weight_observers`（bridge.py）：prepare 后静态初始化权重 observer，
  不跑推理——LSQ 的 scale 初始化需兼容该路径；
- QAT JSON 示例：`configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json`。

## 3. 设计

### 3.1 量化模块：复用 torch 内置 `_LearnableFakeQuantize`

**调研结论（2026-08-10，已实测验证）**：

- torch 2.6 官方提供 `_LearnableFakeQuantize`（`torch.ao.quantization._learnable_fake_quantize`，
  内部模块，继承 `FakeQuantizeBase`，有 `with_args`）——LSQ 风格可学习 scale/zero-point：
  scale 为 Parameter，`enable_param_learning()` 开启学习，`use_grad_scaling` 支持
  LSQ 论文的 `1/√(N·Qp)` 归一化（Axera YOLOv5 未启用，保持 False）；
- **Axera 官方已在 `QAT.Ultralytics.YOLOv5`（references/ 拷贝）落地**：`utils/ax_quantizer_lsq.py`
  用 `_LearnableFakeQuantize.with_args(observer=HistogramObserver, eps=2**-12)`（act）与
  `_LearnableFakeQuantize.with_args(observer=PerChannelMinMaxObserver, channel_len=..., eps=2**-12)`
  （weight）；`utils/ax_quantizer_utils.py` 在 weight qspec 处从权重形状注入 `channel_len`；
- 本项目已验证（内存实验）：`_LearnableFakeQuantize.with_args(...)` 作为 qspec ctr →
  `prepare_qat_pt2e` ✓、scale 梯度 ✓、`convert_pt2e` 后转标准 QDQ（scale 固化为 frozen
  参数，`quantize/dequantize_per_tensor` 节点）→ 现有导出链路零改动。

因此**不自实现** `LSQFakeQuantize`。已落地到本仓库：

```text
pytorchocr/quantization/ax_quantizer_lsq.py  拷贝自 QAT.Ultralytics.YOLOv5（AGPL-3.0，见 UPSTREAM）
pytorchocr/quantization/ax_quantizer_utils.py 移植 get_weight_shape / _ctr_is_fakequat /
                                              get_weight_qspec(weight_node_shape) + channel_len 注入
pytorchocr/quantization/bridge.py            load_axera_quantizer 按 QAT JSON 顶层 "lsq": true
                                              选择 ax_quantizer_lsq.AXQuantizer
```

`_LearnableFakeQuantize` 关键行为（已实测）：

- `forward`：static 模式（static_enabled=1）用 observer 统计 + `copy_` 覆盖 scale；
  learning 模式（static_enabled=0）用 `_fake_quantize_learnable_*_affine`（内置可学习 op，
  对 scale 有梯度）；
- `enable_param_learning()`：`learning_enabled=1, static_enabled=0, scale.requires_grad=True`；
- **scale 初始值陷阱（已定位）**：默认 scale=1.0 对 conv 权重（~U(-0.3,0.3)）量化后全塌缩为 0
  （实测输出非零 0/512），梯度消失。必须先用 observer 统计初始化（见 3.3）。

### 3.2 quantizer 注入点

QAT JSON 顶层新增 `"lsq": true` 开关（默认缺省走现有统计 quantizer，合同不变）：

```text
bridge.load_axera_quantizer：读取 QAT JSON 顶层 "lsq" 标志
  false（默认）→ ax_quantizer.AXQuantizer（现状）
  true         → ax_quantizer_lsq.AXQuantizer（U16/S16 合同下 act/weight 均 _LearnableFakeQuantize）
```

`ax_quantizer_lsq.AXQuantizer` 的行为与 vendored 版一致（global/regional、OP_TO_ANNOTATOR、
`AxeraQuantizerAdapter` 图级修正），仅 ctr 换成 `_LearnableFakeQuantize` + weight 注入
`channel_len`。已实测：U16 act（0-65535）、S16 weight per-channel（scale 形状 [C]）、
regional（matmul/softmax）均正确构建。

### 3.3 scale 初始化（YOLOv5 epoch-0 两阶段统计）

LSQ 对初始 scale 极敏感（默认 1.0 会量化塌缩、梯度消失，已实测）。采用
`references/QAT.Ultralytics.YOLOv5/qat_base_ptq.py` 的流程：

```text
prepare 后（训练前）:
  model.apply(enable_learn)              # 全部 _LearnableFakeQuantize → enable_param_learning()
                                         # （learning_enabled=1，但此时 scale 仍是初始值）

epoch-0 统计阶段（LSQ 特有）:
  model.apply(disable_fake_quant)        # 关 fake quant：前向只统计不量化
  model.apply(enable_observer)           # observer 统计 → scale.data.copy_(统计值)
  ...（epoch-0 验证）...
  model.apply(enable_fake_quant)         # 重新量化
  model.apply(disable_observer)          # 关统计：scale 固定为统计值，进入可学习
```

之后训练：scale 由 `_fake_quantize_learnable_*_affine` 的梯度更新。

与本项目 `initialize_weight_observers`（静态初始化权重 observer，不跑推理）的关系：
LSQ 模式改用上述"首个统计 forward"流程；`initialize_weight_observers` 需对
`_LearnableFakeQuantize` 模块提供等价初始化或跳过逻辑（实现时确认）。

### 3.4 训练与验证合同

- 训练前：`model.apply(enable_learn)` 一次性开启 scale 学习（参照 YOLOv5，非每 epoch）；
- epoch-0 统计阶段（3.3）：统计 scale 后 `disable_observer` 保持学习；此后 scale 始终
  可训练直到训练结束（`disable_observer` 只关统计，不影响 scale 参数梯度）；
- scale 参数进入 optimizer 的 default 参数组（或单独 group，可配置 `lsq_scale_lr_mult`）；
- checkpoint 保存 scale（model state_dict 中），resume 严格恢复；
- validation / `--eval-only`：scale 参数不变（learning 状态可保持），fake-quant 输出确定；
  是否在验证时 `disable_learn`（YOLOv5 定义了但未使用）由实现时对照验证指标决定；
- epoch-2 精度门禁、observer 生命周期、dynamic heights、amp 规则保持现状。

### 3.5 导出链路（已实测验证）

- `convert_pt2e` 对 `_LearnableFakeQuantize` 转标准 QDQ：scale 固化为图内 frozen 参数，
  `quantize/dequantize_per_tensor` 节点（实测通过）；
- 因此现有 `export_ocr_onnx.py`/`onnx_export.py` 导出链路**零改动**；
- QuantONNX 的 Q/DQ 数量、dtype、共享域应与同配置统计 observer 一致（S3 确认）；
- `validate_qdq_graph`、ORT `ORT_DISABLE_ALL` 对比照常执行。

## 4. 执行顺序与门禁

1. **S0 可行性验证（已完成，2026-08-10）**：
   - `_LearnableFakeQuantize.with_args(...)` 作为 qspec ctr：prepare ✓、scale 梯度 ✓、
     convert_pt2e 转标准 QDQ ✓、state_dict 含 scale ✓；
   - weight per-channel `channel_len` 注入（从权重形状）✓（scale 形状 [C]）；
   - 根因定位：默认 scale=1.0 量化塌缩→梯度消失，须 3.3 统计初始化（YOLOv5 epoch-0 流程）；
   - Axera YOLOv5 参考已拷贝至 `references/QAT.Ultralytics.YOLOv5/`（AGPL-3.0，已记录）。
2. **S1 实现（已完成）**：`ax_quantizer_lsq.py` 拷贝 + utils 的 channel_len/辅助函数 +
   bridge 的 `"lsq": true` 开关 + train.py `--lsq`/profile `lsq` + `_prepare_lsq_training`
   （weight 静态初始化 + enable_learn + act 统计 pass，resume 时仅 re-enable）。
   门禁：`pytorchocr` import 通过，QAT JSON 新旧字段兼容，默认行为不变。
3. **S2 单元测试（已完成）**：`tests/test_lsq_quantizer.py`（9 个）覆盖 quantizer 开关、
   `_LearnableFakeQuantize` 形状/学习开关、weight 静态初始化、统计 pass 后梯度流动、
   convert 转标准 QDQ；profile `lsq` 字段与 resume 合同校验。全量回归不回归。
4. **S3 smoke（已完成）**：train.py 端到端 smoke（v5 rec U16/S16 + lsq config，1 epoch CPU）：
   prepare→统计→训练→val→checkpoint 全链路通过，1051 个 LSQ scale 参数入 checkpoint
   （per-tensor [1]/per-channel [C] 形状齐全）。注意：统计阶段 scale 仍被 `eps=2**-12`
   clamp（`_LearnableFakeQuantize.forward` 的 `clamp_(min=eps)`），学习阶段由梯度驱动；
   是否脱离截断需 S4 真实训练确认。
5. **S4 真实数据对照（已完成，2026-08-18 补记）**：
   - Exp5：修复 `eps=2**-12`（改小 eps 或按域配置）的无 LSQ 基线；
   - Exp6：QAT JSON 顶层 `"lsq": true`，其他合同与 Exp5 相同；
   - 对比 epoch-2 acc / 完整 50 epoch（若门禁通过）。
   门禁：Exp6 相对 Exp5 精度持平或提升；任何下降都需定位。
   结果：Exp5-12b 全部执行并记录于
   `docs/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md` §30-38；
   LSQ 为 QAT 必需（PTQ 对比见训练记录 §43.3：U8/S8 域 0.2427 → QAT 0.5927）。
6. **S5 文档（部分完成）**：更新本计划状态 + `docs/axera_qat/records/` 实验记录 +
   `docs/axera_qat/readme.md` 支持矩阵（记录已更新；支持矩阵 2026-08-18 补更新，
   见 axera readme §5 的 U8/S8 Exp13-16 行）。

## 5. 产物清单

```text
pytorchocr/quantization/ax_quantizer_lsq.py   LSQ quantizer（vendored from QAT.Ultralytics.YOLOv5）
pytorchocr/quantization/ax_quantizer_utils.py  get_weight_shape/_ctr_is_fakequat/channel_len 注入
pytorchocr/quantization/bridge.py              "lsq": true 开关
configs/qat/（新增带 "lsq": true 的 QAT JSON）
tests/test_lsq_quantizer.py
configs/qat/training/ppocrv5_mobile_rec_*_lsq.yml   Exp6 profile
docs/axera_qat/records/（Exp5/Exp6 记录）
```

## 6. 风险与停止条件

- PT2E convert 对自定义 FakeQuantize 支持不足：S0 先验证，不通过则改用
  torch.ao 内置 FakeQuantize + 可学习 observer 的等价实现（`observer=True` 参数化
  的 LSQ 变体），或评估在 `onnx_export.py` 侧注入学习 scale；
- scale 梯度不稳定（LSQ 常见问题）：S2 用解析式测试约束；训练中观测 scale 发散则停止；
- Axera/Pulsar2 对学习 scale 的数值范围无额外约束（QDQ FP32 scale），但需在 S3 确认
  导出 scale 无异常值；
- LSQ 结果不优于修复 eps 的统计 observer（Exp5 基线）：则 LSQ 作为备选方案记录，
  不进入正式训练；
- 不破坏现有 moving_average 合同的回归测试失败。

## 7. 待确认项

1. QAT JSON 顶层 `"lsq": true` 开关（已落地，默认缺省走统计 observer）；
2. scale 学习率倍率 `lsq_scale_lr_mult` 默认 1.0 是否合适，是否先做网格（1.0/0.1/0.01）；
3. 是否同时给权重开 LSQ（per-channel S16），还是先只做激活（U16）——计划先只做激活，
   权重保持统计 observer（单变量更清晰）；
4. S4 先做 Exp5（修复 eps）还是直接 Exp6（LSQ），计划先 Exp5 建立修复后的对照基线。

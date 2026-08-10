# QAT.Ultralytics 可复用经验评估

## 1. 调研范围

本次只读分析以下工作区，不修改其代码：

```text
/home/heqi/project-qat/ultralytics
branch: qat
HEAD: 553bfc242
```

该工作区包含未提交修改，因此本文结论对应当前工作树，不等同于上游 Ultralytics 8.4.21。
重点阅读了 `AGENTS.md`、`.codex/skills/`、`train_qat.py`、`export.py`、
`ultralytics/utils/qat_utils.py`、`ax_quantizer*.py`、PT2E/observer 测试以及 AXERA 部署文档。

## 2. 结论

可复用内容主要是流程和验收方法，不是直接复制 YOLO 代码：

1. 训练时不提前冻结 observer，验证时才临时关闭 observer；
2. 从浮点预训练权重使用低学习率、较长 QAT 日程，先 smoke 再全量训练；
3. 分阶段对齐 eager/exported/prepared/converted/QuantONNX/AXModel，避免把 PT2E 图错误误判为量化误差；
4. 网络、Torch 或导出环境变化后重新发现量化节点，并严格校验最终 Q/DQ dtype 和 qparam；
5. checkpoint、模型图、量化配置和导出 wrapper 必须严格对应，不通过宽松加载或手改 ONNX 掩盖问题；
6. 输入/输出 qspec 解耦、区域量化和可选 observer 是有价值的后续能力，但应由精度定位结果驱动；
7. LSQ、KD、QAT EMA、DDP 和近似 ONNX qparam 合并不属于当前 PP-OCR 基线方案。

## 3. Skill 评估

| Skill | 可复用程度 | PP-OCR 处理方式 |
|---|---|---|
| `pt2e-accuracy-check` | 高 | 工作流与模型无关，应直接采用其分层数值对齐和 BN 参数检查方法 |
| `yolo-qat-task-onboarding` | 中 | 采用“先浮点契约、再真实数据反向传播、再导出/部署”的顺序；YOLO head 脚本不可复用 |
| `yolo-qat-config-discovery` | 中 | 采用按 source stack/拓扑重新发现、禁止复制旧 FX 名称的原则；Attention 脚本不可直接用于 PP-OCR |
| `axera-quantonnx-config` | 中 | 采用从本次精确 QuantONNX 生成转换配置、校验实际 Q/DQ 的原则；YOLO Attention override 不适用 |
| `yolo26-qat-delivery` | 中 | 采用严格 checkpoint、真实 `convert_pt2e` 精度、ONNX/ORT/AXModel 分层验收清单 |
| `qat-cn-todos` | 低 | 仅借鉴任务记录流程；当前路线 2 已有独立文档，不复制其目录体系 |

已在项目根目录创建 `.codex/skills/ppocr-pt2e-qat`，复用 `pt2e-accuracy-check` 的方法，
并将输出契约替换为 DB 检测 maps、CTC logits 和对应 metric。迁移详情见
[QAT.YOLO 方法迁移记录](qat_yolo_migration.md)。

## 4. 可直接采用的训练技巧

### 4.1 Observer 全程更新

Ultralytics 正式训练路径没有调用 `disable_observer`；该调用只出现在离线 QAT 对齐脚本和测试中。
这与当前决定一致：训练命令不设置 observer freeze，validation 临时关闭 observer，结束后恢复。

短日程中提前冻结会让激活范围由训练早期少量 batch 决定；检测增强和权重更新会继续改变分布。
是否冻结应成为有对照实验的数据结论，而不是默认步骤。

### 4.2 低学习率、较长日程

Ultralytics 已交付 profile 使用：

```text
epochs=50
lr0=2e-5
lrf=0.1
qat_ema=False
```

其策略是从浮点权重做小步长 QAT 微调，不是使用浮点训练学习率重新训练。该数值不能直接复制到
Adam 的 PP-OCR；应在新数据集规模确认后比较至少两组，例如 `1e-4` 和 `2e-5`，其余条件一致。

### 4.3 先完成端到端 smoke

正式全量训练前，先用真实格式小数据完成：

```text
prepare -> forward/backward -> validation -> checkpoint strict reload
-> convert_pt2e -> QuantONNX -> ORT -> AXERA convert
```

当前 scaled HardSigmoid smoke 已完成上述大部分链路并通过 AXERA 转换。正式训练前仍缺检测 metric
和新图 checkpoint strict reload。

### 4.4 不默认使用 EMA

Ultralytics 虽实现 QAT EMA，但正式 profile 使用 noEMA。PT2E observer/fake-quant buffer 与普通参数
EMA 的语义不同，EMA 还会增加 checkpoint 和导出选择复杂度。PP-OCR 基线继续 noEMA；只有固定
验证协议证明 EMA 有增益时再引入。

### 4.5 固定空间尺寸，动态 batch

PP-OCR 当前训练和导出使用固定 `H/W`、仅动态 batch，适合 Torch 2.6 PT2E。训练和验证必须使用同一
空间尺寸，避免 validator 产生不同 shape 后触发 graph constraint 或 reshape 错误。

Ultralytics 当前实现存在不一致：调试文档和日志称固定 `H/W`，但
`prepare_pt2e_qat_model()` 实际对 `N/H/W` 都使用 `Dim.AUTO`。在完成约束测试前不复制这一实现。

## 5. 必须采用的分层精度检查

参考 `pt2e-accuracy-check`，每次变更 prepare、BN、quantizer、训练/验证模式或导出时记录：

| 阶段 | 检查目标 |
|---|---|
| eager float train/eval | 原始模型和任务输出契约正确 |
| exported float | `export_for_training` 未改变输出和 BN 超参数 |
| prepared，关闭 observer/fake-quant | 与 exported float 基本一致，定位 PT2E prepare 自身误差 |
| prepared，开启 fake-quant | 观察训练内模拟量化误差 |
| `convert_pt2e` | 以真实 Q/DQ 作为 QAT 精度交付依据 |
| QuantONNX | 对齐 converted PyTorch、ORT 和可选 ReferenceEvaluator |
| AXModel | 使用相同预处理、原始输出和后处理对齐 QuantONNX |

至少保存 `max_abs`、`MAE` 和任务 metric。DB 检测应优先比较 shrink map、threshold map 和最终
binary map，再比较 DBPostProcess 后的 precision/recall/hmean；不能只比较 validation loss。

BN 检查包括 training flag、momentum 和 eps。当前 PP-OCR QAT 默认先执行 `rep()`，最终 ONNX BN=0，
但 exported/prepared 阶段仍应增加一次明确检查，避免 Torch 版本变化引入回归。

## 6. 值得后续移植的 Quantizer 能力

### 6.1 输入和输出 qspec 解耦

Ultralytics 支持分别配置：

```text
input/output dtype
input/output symmetric or affine
input/output activation observer
regional input-only override
```

这可表达 `U16 -> U8`、`S8 -> S8` 或只修改算子输入而不强迫输出切域。对 PP-OCR 的潜在用途是：

- HardSigmoid、Sigmoid 和 DBHead 输出固定回到 U8；
- 只对误差敏感的 neck/head 局部使用 U16；
- 避免 regional 输入覆盖无意改变下游量化域。

采用前必须先有分层误差报告，不应直接复制 YOLO 的 SiLU/Attention/分类塔配置。

### 6.2 可配置 observer

Ultralytics 默认 quantizer 支持 moving-average、minmax 和 histogram，并允许输出 observer 独立选择。
正式交付 profile 仍统一使用 moving-average，因此该能力属于实验工具，不代表 histogram 已验证更好。

PP-OCR 可在固定 checkpoint 和数据顺序下做 observer A/B；必须同时比较 QAT metric、converted metric、
QDQ scale 稳定性和 AXModel 结果。

### 6.3 融合子图末端放置输出 qspec

其 Conv regional output 会放到 `Conv -> BN -> Activation` 的末端，而不是 Conv 与 BN 之间，并有
`convert_pt2e` 后 BN=0 的回归测试。该方法适用于未来不执行预先重参数化的模型；当前 PP-OCR 已默认
`rep()`，优先级低于 metric 和数据增强。

### 6.4 共享量化域

Ultralytics 通过 `SharedQuantizationSpec` 保持 Concat/Split 等数据搬运算子的 qparam 一致。路线 2 已在
`AxeraQuantizerAdapter` 实现 Concat 共享域，并验证 `2/2`，不需要重复移植。后续遇到 Split/Reshape
再按实际 PP-OCR 拓扑扩展，不能复制 YOLO 固定分支数量。

### 6.5 配置发现和严格命中

Regional 配置按 FX 节点名选择时，结构或 Torch 版本变化可能静默失效。后续若 PP-OCR 引入局部 U16，
应增加以下闭环：

1. 按 `source_fn_stack` 和拓扑发现节点角色；
2. 生成候选 JSON，不覆盖模板；
3. prepare 前严格检查全部 regional 节点命中；
4. 导出后按拓扑复核实际 Q/DQ dtype，不能只看“导出成功”。

## 7. 暂不采用的能力

### 7.1 LSQ

Ultralytics 的 LSQ quantizer 未进入正式交付 profile，且与默认 quantizer 的 output observer、regional
缺省输出语义仍不一致。当前不引入。

### 7.2 Knowledge Distillation

其检测 KD 使用 float teacher，对分类 raw score 做温度 KL、对 box raw output 做 MSE。这个实现绑定
YOLO 输出，不能直接用于 DB。PP-OCR 若量化损失在完整 baseline 后仍明显，可考虑对 float teacher 和
QAT student 的 shrink/threshold maps 做蒸馏，但必须单独设计 mask 和 loss 权重。

### 7.3 QAT EMA 和 DDP

QAT EMA 不是正式 profile 默认；DDP 虽已验证两卡，但 observer buffer 仍依赖 DDP broadcast，并要求
单卡 converted/ONNX 对齐。当前先完成单卡正确性和 metric，不增加这两个变量。

### 7.4 近似 ONNX qparam 合并

Ultralytics `export.py` 包含 scale 比小于 2%、zero-point 接近时合并 DQ/Q，以及 Split/Reshape 对齐等
YOLO/AXERA 后处理。这类操作必须有同一 observer 来源和数值等价证明。路线 2 继续优先修正 QAT 图，
只删除完全相同 qparam 的冗余 DQ/Q，不照搬近似合并阈值。

## 8. 风险与待验证项

### 8.1 ConvTranspose per-channel axis

Ultralytics 和路线 2 当前代码都存在以下字面不一致：

```text
MovingAveragePerChannelMinMaxObserver(ch_axis=1)
QuantizationSpec(ch_axis=0)
```

PP-OCR DBHead 使用两个 ConvTranspose。当前 scaled HardSigmoid smoke ONNX 的权重 DQ 均为正确的
`axis=1`，scale 数量为 24 和 1，因此没有现存导出错误。后续应增加 prepared/converted/ONNX 回归测试，
确认 Torch 升级后仍保持 axis=1，再决定是否修正 qspec 字段。

### 8.2 Metric 曾是正式训练短板

路线 2 的 `DetectionDataset` 已改为保持比例的居中 letterbox、归一化和 DB map。QAT baseline 明确
不执行随机缩放、旋转、RandomCrop、CopyPaste 等增强。本评估编写时 Trainer 尚未接入
DBPostProcess/DetMetric，因此当时 validation 只能比较 loss。2026-07-30 已补齐 det/rec metric、
converted/QuantONNX 同集验证和按主指标选择 checkpoint，见
[ICDAR2015 QAT Metric 验证记录](../baselines/icdar2015_qat_metric_validation.md)。

### 8.3 训练参数必须记录到 checkpoint

当前 checkpoint 已记录 training profile/量化配置路径和 SHA256、Torch 版本、输入 shape、observer
策略及 scheduler 参数。数据集版本和 git revision 仍待补充。图或配置变化后只允许从浮点权重重新
prepare/QAT。

## 9. 后续建议顺序

training profile 和项目 skill 已完成首轮迁移。后续建议按以下顺序执行：

1. 等数据集完成后检查格式、规模、train/val 划分和重复样本；
2. 补充 DetMetric，并保持当前无随机增强的确定性 baseline；
3. 实现 skill 规定的 exported/prepared(fake off)/prepared(fake on)/converted 分层数值对齐；
4. 使用 baseline profile 运行 1 epoch 全链路 smoke，严格恢复 checkpoint 并完成 QuantONNX/AXERA 转换；
5. 再比较学习率和训练日程，启动正式训练；
6. 只有定位到具体敏感张量后，才实验局部 U16、独立 output qspec/observer 或 KD。

# PaddleOCR PyTorch / Axera QAT 路线

本目录记录 det/rec 浮点转换、训练语义、PT2E QAT、QuantONNX 和 Pulsar2 适配。仓库全部文档的
分类说明见 [docs 总索引](../readme.md)。

## 文档导航

- 当前执行计划：[精度恢复总计划](plans/model_accuracy_validation_plan.md)、
  [pretrained 训练结构复现](plans/pretrained_training_structure_plan.md)、
  [检测专项](plans/detection_accuracy_recovery_plan.md)、
  [识别专项](plans/recognition_accuracy_recovery_plan.md)。
- 当前实验记录：[精度验证结果](records/model_accuracy_validation_results.md)、
  [v5 mobile det 训练](records/icdar2015_ppocrv5_mobile_det_qat_training.md)、
  [v5 mobile rec smoke](records/ppocrv5_mobile_rec_qat_smoke.md)、
  [v5 mobile rec 训练](records/icdar2015_ppocrv5_mobile_rec_qat_training.md)、
  [v5/v4 兼容性](records/ppocrv5_v4_compatibility.md)。
- 操作指南：[训练图 ONNX 导出](guides/training_onnx_export.md)、
  [初始化 observer 导出](guides/initialized_observer_quantonnx_export.md)、
  [Pulsar2/AXModel 交付](guides/pulsar2_axmodel_handoff.md)。
- 历史归档：[早期 v6/ICDAR 基线](archive/baselines/)、
  [Ultralytics 调研与 QAT.YOLO 迁移](archive/migration/)。归档指标只用于追溯，不能代替当前模型验收。

2026-08-05 起，四个优先模型必须先完成 pretrained 训练结构复现，不再用单输出部署图代替完整
训练图。v5-rec 的 QAT regional 配置发现与 Pulsar2 转换配置生成分别由项目内
`.codex/skills/ppocr-qat-config-discovery` 和 `.codex/skills/ppocrv5-rec-pulsar2-config` 执行。

## 1. 目标

本目录记录 PaddleOCR2Pytorch 的训练和 Axera QAT 支持状态。当前结构复现优先级为：

1. PP-OCRv5 mobile rec；
2. PP-OCRv5 mobile det；
3. PP-OCRv6 small rec；
4. PP-OCRv6 small det；
5. 上述模型通过后再扩展 v4、server 和 v6 tiny/medium。

早期版本基于 Paddle 预训练权重直接微调单 CTC/主 DBHead 部署图；该策略产生了 PP-OCRv5 rec
backbone 梯度合同错误，现只作为历史诊断。当前要求先复现完整 MultiHead/MultiLoss 或 DB 辅助训练
结构，再显式投影出单 CTC/单 shrink 部署图。

## 2. 代码基线

```text
PaddleOCR2Pytorch upstream:
  repository: https://github.com/frotms/PaddleOCR2Pytorch
  commit:     c05307fbd61e575d55f0fef0334026c7536aec76
  date:       2026-07-08

QAT.axera reference:
  repository: https://github.com/AXERA-TECH/QAT.axera
  commit:     4603b160bad6b212551721ec3cd3b75895b17de3
```

工作区使用 partial clone 和 sparse checkout，仅拉取模型、配置、转换器和必要工具，避免下载
上游仓库中的大量图片资源。开发分支为：

```text
feat/axera-qat-training
```

当前参考环境：

```text
Python:       3.10
PyTorch:      2.6.0+cu118
ONNX:         1.17.0
ONNX Runtime: 1.21.0
```

QAT.axera requirements 使用 PyTorch 2.6.0，版本主线一致。当前受限执行环境未暴露 CUDA，
CPU 可用于构图和导出 smoke test，正式训练和性能验证必须在可用 GPU 环境重新执行。

> 2026-07-29 更正：当日检查发现原 Conv annotator 未覆盖 `aten.conv2d.padding`。因此下方
> 2026-07-28 的 Q/DQ 数量保留为问题排查历史，不代表可部署结果；当前有效统计和模型以
> “CTW1500 多 epoch 检测 QAT”中的 SAME-padding 修复版为准。

## 3. 工程边界

新增代码按职责拆分：

```text
pytorchocr/training/             dataset、loss、metric、optimizer、trainer
pytorchocr/quantization/         PT2E wrapper、Axera quantizer、graph validation
tools/train.py                   浮点/QAT 共用训练入口
pytorchocr/quantization/onnx_export.py  ONNX 导出、QDQ rewrite 和 checker 公共后端
tools/export_ocr_onnx.py                checkpoint/initialized/training/audit 唯一用户入口
tools/validate_qat.py            QDQ、ORT 和数值检查
configs/qat/                     模型量化配置
tests/                           单元测试和模型 smoke test
```

第三方 QAT 规则以固定 commit 放入 `pytorchocr/quantization/`。复制的 quantizer 源文件仅将
`utils.ax_quantizer_utils` 改为包内相对导入，许可证和来源记录保存在同目录；项目侧 Concat 共享域
等行为仍放在 `bridge.py` 的 adapter 中，不混入上游源码。ONNX 导出后端位于
`pytorchocr/quantization/onnx_export.py`。

## 4. 阶段和验收门槛

### 阶段 0：独立工作区

- [x] 创建独立 Git 工作区；
- [x] sparse checkout 上游代码；
- [x] 创建开发分支；
- [x] 记录上游和 QAT.axera commit；
- [x] 验证 PyTorch 2.6、ONNX 和 ORT 环境；
- [x] vendor QAT.axera 最小依赖、许可证和固定 commit 记录。

### 阶段 1：PP-OCRv6 浮点基线

- [x] 构建 det small 模型并通过随机输入 forward；
- [x] 构建 rec small 模型并通过随机输入 forward；
- [x] 转换 Paddle det 预训练权重；
- [x] 获取并转换 Paddle rec 预训练权重；
- [x] 比较 Paddle 与 PyTorch det 最终输出；
- [x] 比较 Paddle 与 PyTorch rec CTC logits 和最终 argmax；
- [ ] 保存可复现的输入、输出和误差统计。

### 阶段 2：QAT.axera feasibility spike

- [x] 为 det 主分支建立静态 QAT wrapper；
- [x] 为 rec CTC 路径建立静态 QAT wrapper；
- [x] det 通过 `torch.export.export_for_training`；
- [x] det 通过 `prepare_qat_pt2e` 和单步 backward；
- [x] det 通过 `convert_pt2e` 和 ONNX 导出；
- [x] det 检查 U8 激活、S8 权重、Concat 共享域和直接 `DQ -> Q`；
- [x] det 通过 ONNX checker 和 ORT；
- [ ] 通过 Axera 编译 smoke test；
- [x] rec 随机权重图重复以上检查；
- [x] rec 预训练权重重复以上检查。

只有阶段 2 通过后，才继续大规模移植 dataset/loss，避免先建设训练框架后才发现模型无法被
PT2E/QAT.axera 稳定捕获。

### 阶段 3：训练基础设施

- [x] DB Dice/Focal、threshold L1 和 binary Dice loss；
- [x] CTC loss；
- [x] 通用 Trainer、optimizer、scheduler、AMP、梯度裁剪和 observer freeze；
- [x] checkpoint/resume，包含 optimizer、scheduler、AMP scaler 和 QAT observer 状态；
- [x] 统一配置解析、float/QAT 训练命令行和 pretrained 接口；
- [x] det PaddleOCR 标注读取、保持比例的居中 letterbox、shrink/border map；
- [x] rec PaddleOCR 标注读取、resize/padding、字典编码；
- [x] 使用 PaddleOCR 格式样本验证 det/rec float 与 QAT 完整训练循环；
- [x] 从 QAT checkpoint 恢复并导出 batch-1 ONNX QDQ；
- [x] QAT det/rec 使用固定 shape 和确定性预处理，不启用随机增强；
- [x] det DBPostProcess/precision/recall/hmean metric；
- [x] rec CTC sequence accuracy/normalized edit distance metric；
- [ ] 单卡训练先行，DDP 后补。

### 阶段 4：正式验证

每个模型必须通过：

1. Paddle/PyTorch 浮点输出对齐；
2. loss、backward、optimizer 和 checkpoint smoke test；
3. QAT observer 初始化、训练期持续更新和 validation 临时关闭/恢复；
4. PT2E convert 和标准 ONNX QDQ 导出；
5. ONNX checker 和 ORT；
6. QDQ dtype、qparams、fan-out 和 requantization 检查；
7. Axera 编译和板端运行；
8. 浮点/QAT 验证集精度回归。

## 5. 模型支持矩阵

| 模型 | 浮点构图 | Paddle 权重 | 训练 | QAT | ONNX | Axera |
| --- | --- | --- | --- | --- | --- | --- |
| PP-OCRv6 small det | 已对齐 | 已转换 | ICDAR 50 epoch，hmean 0.46556 | checkpoint 已导出 | checker/ORT 通过 | 待编译 |
| PP-OCRv6 small rec CTC | CTC 已对齐 | 已转换 | ICDAR 50 epoch，best acc 0.73375 | checkpoint 已导出 | checker/ORT 通过 | 待编译 |
| PP-OCRv6 tiny/medium | 未开始 | 未开始 | 未开始 | 未开始 | 未开始 | 未开始 |
| PP-OCRv5 mobile det | 已对齐 | 已严格转换 | ICDAR 50 epoch，best hmean 0.16527 | checkpoint 已导出 | checker/ORT 500 图通过，hmean 0.14832 | 待编译 |
| PP-OCRv5 mobile rec CTC | 已对齐，完整指标已重测 | 已严格转换 | 原生 SiLU 50 epoch QAT 进行中 | 局部 S8 smoke 与 fake-off 全验证集通过任务门禁 | checker/ORT 通过，Pulsar2 配置已生成 | 待编译 |
| PP-OCRv5 server det/rec | 随机权重构图通过 | 未转换 | 未开始 | 随机权重 smoke 通过 | QDQ 结构通过 | 未开始 |
| PP-OCRv4 mobile/server | 随机权重构图通过 | 未转换 | 未开始 | 随机权重 smoke 通过 | QDQ 结构通过 | 未开始 |

## 6. 已知风险

1. 上游项目主要面向推理，缺少完整训练基础设施；
2. v6 det 转换器会丢弃辅助检测头，rec 转换器会丢弃 NRTR/GTC 权重；
3. dict 输出、`self.training` 分支和 rec `data` 参数可能影响 PT2E 捕获；
4. 训练图与最终推理图必须使用同一组 observer/qspec，不能靠 ONNX 后处理掩盖差异；
5. Axera quantizer 的 `validate()` 当前为空，算子已 annotation 不代表一定能通过编译器；
6. rec 动态宽度先固定为 320，动态 shape 在静态链路稳定后单独验证。

## 7. 当前 QAT 图策略

PP-OCRv6 的 PPLCNetV4 和 RepLKFPN 都包含仅用于训练的多分支重参数化结构。首版采用：

```text
加载 Paddle 浮点权重
  -> eval running statistics
  -> backbone.rep() + neck.rep()
  -> 捕获部署形态的训练图
  -> Axera PT2E QAT 微调
  -> convert_pt2e
  -> ONNX QDQ
```

这意味着 QAT 微调直接优化最终部署结构，不继续更新已经折叠的 BN running statistics。优点是
训练图和部署图一致，避免在 QAT 结束后再做多分支合并而改变权重和量化边界；代价是不能复刻原始
多分支浮点训练过程。完整浮点训练若需要保留多分支，应在进入 QAT 阶段前单独完成。

当前项目 adapter 还补充一条 QAT.axera 规则：Concat 的所有输入和输出通过
`SharedQuantizationSpec` 共用一个 observer/fake-quant 域。Add 输入允许使用不同 qparam，这与
`exp61_yolo11n_siluInU16.onnx` 的结构一致。

检测训练和部署使用同一份 prepared/converted PT2E 图。`DetTrainingWrapper` 在训练时显式返回
`(shrink, threshold, binary)`，DB loss 同时约束三个输出；导出时只在 converted 图外包一层
`OutputSelector(index=0)`。Torch ONNX 导出阶段的 DCE 会移除 threshold/binary 分支，不需要重新
捕获推理图、迁移 observer，或在 ONNX 文件导出后修改 QDQ 节点。

## 8. 实验记录

### 2026-07-28：工作区初始化

```text
结果：partial clone 成功，工作树约 12 MB
分支：feat/axera-qat-training
环境：PyTorch 2.6.0+cu118 / ONNX 1.17.0 / ORT 1.21.0
限制：当前执行环境 CUDA 不可用
下一步：构建 PP-OCRv6 small det/rec 随机输入浮点基线
```

### 2026-07-28：浮点转换和框架对齐

PP-OCRv6 small det Paddle 权重已转换为 PyTorch。相同随机输入 `1x3x128x128` 的最终输出：

```text
shape: [1, 1, 128, 128]
MAE:   6.65449e-11
P99:   4.07454e-10
max:   1.49885e-9
```

PP-OCRv6 small rec 随机输入 forward 输出为 `[1, 40, 18710]`。完整 MultiHead 参数量为
29,269,478，首版 QAT 必须使用 CTC-only wrapper，不能把 NRTR 分支带入部署图。后续已下载并转换
官方预训练权重，详细结果见本节末的预训练识别实验。

### 2026-07-28：训练态多分支 QAT 对照

未调用 `rep()` 的检测图可完成 PT2E QAT、单步 backward、转换、ONNX checker 和 ORT：

```text
float / prepared / converted nodes: 908 / 1742 / 1508
gradient tensors:                  279
ONNX nodes:                        958
Q / DQ:                            248 / 381
BatchNormalization:                10
PyTorch vs ORT MAE / max:          0 / 0
```

但 10 个 PPLCNetV4 `RepDWConv` 均为 `Add -> QDQ -> BN -> QDQ`，BN 前后 qparam 不同。
这些节点来自 `(3x3 DW + 1x1 DW + identity) -> Add -> BN`，不是普通 Conv-BN pattern；直接保留
多分支图不符合部署重参数化结构，因此不作为默认 QAT 路线。

Torch 2.6 ONNX exporter 对 native BatchNorm 的 tuple 输出存在缺陷。项目侧提供了导出 lowering，
显式生成推理态单输出 `BatchNormalization`；没有修改 QAT.axera，也没有在导出后重写量化图。

### 2026-07-28：部署重参数化 QAT

修复 `ConvBNAct.rep()` 对 2x2 `padding="same"` 的错误处理后，重参数化前后浮点结果为：

```text
MAE / max:          3.33251e-10 / 7.07223e-9
parameters:         2,510,462 -> 2,476,682
BatchNorm modules:  74 -> 4
```

默认重参数化 QAT smoke 结果：

```text
float / prepared / converted nodes: 390 / 778 / 1279
gradient tensors:                  165, all finite
ONNX nodes:                        790
Q / DQ:                            184 / 341
activation zero-point dtype:       uint8, 184 / 184
BatchNormalization:                0
Conv -> QDQ -> BN:                 0
direct DQ -> Q:                    0
Concat shared qparams:             2 / 2
Add shared qparams:                0 / 29 (允许独立输入 scale)
PyTorch vs ORT MAE / max:          0 / 0
```

当前结论是 PP-OCRv6 small det 的 PyTorch/Axera QAT 导出链路可行。Axera 编译器和板端尚未在本环境
执行，不能把 checker/ORT 通过等同于芯片验收。

### 2026-07-28：识别 CTC-only QAT

识别 wrapper 只捕获 `PPLCNetV4 -> LightSVTR -> CTC Linear`，不执行 NRTR/GTC，也不在模型图中
增加 Softmax。固定输入 `1x3x48x320`、随机权重的 smoke 结果：

```text
output:                            [1, 40, 18710]
float / prepared / converted:      339 / 648 / 1039
gradient tensors:                  145, all finite
ONNX nodes:                        1416
Q / DQ:                            168 / 316
activation domains:                U8 148, S16 20
BatchNormalization:                0
Conv -> QDQ -> BN:                 0
direct DQ -> Q:                    2 necessary, 0 redundant
Concat shared qparams:             1 / 1
PyTorch vs ORT MAE / max:          0 / 0
```

两个直接 DQ->Q 都是 LightSVTR Attention 的 S16 MatMul 输出切回 U8 Mul 输入，qparam 和 dtype
发生变化，属于显式 requant，不是冗余边。额外调用 `onnx_program.optimize()` 会删除透明算子并留下
8 个同 qparam 的表面冗余 QDQ。这是初始实验结论；当前导出已改为默认执行 optimize，并在保存前
严格清理 dtype、scale、zero-point 和 axis 完全一致的冗余 DQ/Q，见后续实验记录。

该结果最初只证明随机权重 rec 图可行；后续已补充预训练权重转换、框架对齐、真实 CTCLoss QAT
batch 和 checkpoint 导出验证。

### 2026-07-28：训练基础设施和检测训练图导出

模型训练接口修复包括：`BaseModel` 收到 neck 字典输出时，将其中的 `fuse` tensor 传入 DBHead；
DB step function 改为数值稳定的 `torch.sigmoid(k * (x - y))`，避免原先 reciprocal/exp 写法在
随机输入 backward 中生成非有限梯度。该修改不改变 DB 二值化函数的数学定义。

当前训练模块已包含：

```text
det loss:     shrink Dice + Focal、threshold masked L1、binary Dice
rec loss:     torch CTCLoss，支持 [N,T,C] logits 和变长标签
Trainer:      AMP、gradient clipping、scheduler、observer freeze、checkpoint/resume
checkpoint:   model/observer、optimizer、scheduler、scaler、epoch/global_step
tests:        dataset/loss、PT2E checkpoint、ONNX QDQ、重参数化，共 9 项通过
```

完整检测训练图 `DetTrainingWrapper` 的 QAT 单步训练到单分支部署 ONNX 结果：

```text
input / exported output:            [1,3,128,128] / [1,1,128,128]
float / prepared / converted nodes: 418 / 835 / 1322
gradient tensors:                  174, all finite
ONNX Conv / ConvTranspose:          83 / 2
Q / DQ:                            184 / 392
BatchNormalization:                0
Conv -> QDQ -> BN:                 0
direct / redundant DQ -> Q:        0 / 0
Concat shared qparams:             2 / 2
activation zero-point dtype:       uint8, 184 / 184
PyTorch vs ORT MAE / max:          0 / 0
```

导出的 ONNX 只剩一个 Sigmoid，说明 threshold/binary 训练分支已经由导出器裁剪。当前单元测试的
Trainer 已使用 prepared PT2E GraphModule 验证训练、observer 冻结及 checkpoint 严格恢复。

### 2026-07-28：最小训练闭环和 checkpoint 导出

新增统一入口和可复用组件：

```text
pytorchocr/training/model_builder.py       det/rec 构图、权重加载、rep
pytorchocr/training/data/det.py            检测标注、居中 letterbox、DB maps
pytorchocr/training/data/rec.py            识别标注、resize/padding、CTC encode
tools/train.py                             float/Axera QAT 训练
tools/export_ocr_onnx.py checkpoint        prepared checkpoint -> ONNX QDQ
pytorchocr/quantization/validation.py       共享 QDQ/ORT 检查
```

当前数据路径兼容 PaddleOCR 的文本标注格式。det 使用保持宽高比的缩放和居中 padding，rec 使用等高缩放、
右侧 padding 和 `[-1, 1]` 归一化。尚未移植 det random crop/CopyPaste/IaaAugment、rec augmentation/
多尺度 sampler 和正式 metric，因此该路径已能用于训练链路验证，但还不能视为完整复刻 Paddle recipe。

当前 QAT 基线采用确定性预处理，不启用 random crop、CopyPaste、旋转、随机缩放等数据增强。检测输入
通过 `--image-shape C H W` 配置；未传该参数时读取模型 YAML 的
`Global.d2s_train_image_shape`。det 对原图和标注多边形使用同一缩放比例，再按四周居中 padding 到目标
尺寸，默认 padding 像素值为 `114`，避免直接拉伸改变文字几何比例。

PaddleOCR 格式临时样本上的单 batch 结果：

```text
det float: loss 6.13959，三分支 backward/optimizer/checkpoint 通过
det QAT:   loss 7.81917，184 U8 activation observers，checkpoint/resume 通过
rec float: loss_ctc 93.44492，随机权重，backward/optimizer/checkpoint 通过
rec QAT:   loss_ctc 93.68918，随机权重，U8/S16 observers，checkpoint 通过
```

observer 不能在 prepared 图第一次 forward 前冻结：per-channel 权重 observer 尚未初始化时只有标量
qparam，会造成 scale/zero-point 与通道维不匹配。训练入口即使配置
`--observer-freeze-epoch 0`，也会先执行一个校准/训练 batch，再冻结；从已有 checkpoint 恢复时可在
epoch 开始直接保持冻结状态。

训练 batch 大于 1 时，PT2E 捕获把 batch 维声明为动态，空间尺寸仍固定；示例 batch 为 1 时，
Torch 2.6 会将 batch 特化为常量，因此保持静态 batch 1。动态训练 checkpoint 可重新构建同一 prepared
图并导出静态 batch-1 部署 ONNX。旧 checkpoint 可通过 `--dynamic-batch` 显式覆盖缺失的元数据。

checkpoint 导出验证结果：

```text
det: input [1,3,64,64] -> [1,1,64,64]
     Q/DQ 184/392，U8 184，Concat 2/2，共享域正确
     BN 0，冗余 DQ->Q 0，PyTorch/ORT MAE/max 0/0

rec: input [1,3,48,64] -> [1,40,18710]
     Q/DQ 168/316，U8 148 + S16 20，Concat 1/1
     必要 requant 2，冗余 DQ->Q 0，PyTorch/ORT MAE/max 0/0
```

典型命令：

```bash
python tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth \
  --label-file /path/to/train.txt --data-dir /path/to/images \
  --output-dir output/ppocrv6_det_qat \
  --qat --qat-config configs/qat/ppocrv6_small_det_u8s8.json \
  --device cuda

python tools/export_ocr_onnx.py checkpoint \
  --checkpoint output/ppocrv6_det_qat/last.pt \
  --output output/ppocrv6_det_qat.onnx \
  --batch-size 1
```

### 2026-07-28：PP-OCRv6 small rec 预训练权重 QAT

官方权重：

```text
source: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/
        PP-OCRv6_small_rec_pretrained.pdparams
size:   119 MB
CTC classes: 18710
conversion:  skipped 84 Paddle GTC keys，PyTorch state_dict 移除 108 个 GTC keys
```

相同随机输入、BN eval、CTC softmax 关闭后的 Paddle/PyTorch logits：

```text
shape:                 [1,40,18710]
MAE / relative MAE:    3.40214e-5 / 3.41700e-5
P99 / max:             1.13487e-4 / 2.09808e-4
probability MAE / max: 1.00109e-9 / 8.16584e-6
argmax agreement:      100%
```

因此权重映射和 CTC 部署路径已对齐。logits 误差高于 det，但相对于平均绝对值约 0.996 很小，且
Softmax 与逐时间步 argmax 一致，无需为该误差修改模型实现。

使用预训练权重和 PaddleOCR 格式样本执行一次真实 CTCLoss QAT：

```text
input:              batch 2, [3,48,320]
loss_ctc:           5.48448
observer:           首 batch 后冻结
checkpoint export:  batch 1, [1,3,48,320] -> [1,40,18710]
Q / DQ:             168 / 316
activation domains: U8 148 + S16 20
Concat shared:      1 / 1
necessary/redundant DQ->Q: 2 / 0
```

预训练 rec 的 QDQ 数值不是跨后端 bit-exact。checkpoint 导出结果为：

```text
converted PyTorch vs ONNX Reference: logits MAE/max 0.26849 / 2.22592
converted PyTorch vs ORT no-opt:     logits MAE/max 0.25057 / 2.00333
ORT no-opt vs optimized:             logits MAE/max 0.69930 / 5.11962
```

三条路径的逐时间步 CTC argmax agreement 都是 100%；probability MAE 约 `1.0e-5` 到 `1.5e-5`，
但单点 probability max error 可达约 0.19。该现象在预训练深层 Attention 的 float-QDQ 图出现，
随机权重图不会充分暴露；ORT graph optimizer 还会进一步放大差异。因此工具默认报告 ORT 禁用
优化和 ORT 默认优化；ONNX Python ReferenceEvaluator 因优化后大图执行很慢，改为通过
`--onnx-reference` 显式启用。不再把 ORT bit-exact 当作 rec 图语义的充分条件。

这不是 Axera 板端数值验收结果。正式结论必须以 Axera 编译器生成的模型、板端输出和识别验证集
精度为准；在此之前，QDQ 结构通过只能说明导出域满足当前规则，不能说明量化精度已经验收。

### 2026-07-28：默认 ONNXProgram optimize 和最终模型重导出

QAT ONNX 导出默认执行 `onnx_program.optimize()`。优化后的 rec 图会生成 8 个 qparam 完全一致的
直接 `DQ -> Q`；导出器只在 dtype、scale、zero-point 和 axis bit-exact 时旁路该量化往返，不使用
Ultralytics 导出脚本中的 2% scale 或 zero-point 容差近似合并。不同 scale 的最小 ONNX 单元测试
确认不会被清理。

最终模型统计：

```text
det optimized:
  nodes / Q / DQ:          790 / 184 / 341
  activation domains:      U8 184
  Concat shared:           2 / 2
  necessary/redundant DQ-Q: 0 / 0

rec optimized + exact cleanup:
  nodes / Q / DQ:          678 / 160 / 277
  activation domains:      U8 142 + S16 18
  Concat shared:           1 / 1
  necessary/redundant DQ-Q: 2 / 0
```

det 和 rec 优化模型均通过 ONNX full checker。相同输入下，优化后的模型与此前未调用
`ONNXProgram.optimize()` 的模型在 ORT no-opt 上逐值一致，MAE/max 都是 0。最终文件位于：

```text
/home/heqi/project/PaddleOCR/exports/quantonnx/
  ppocrv6_small_det_qat_qdq.onnx
  ppocrv6_small_rec_qat_qdq.onnx
```

### 2026-07-29：QAT.axera 最小依赖内置

内置文件来自 QAT.axera commit `4603b160bad6b212551721ec3cd3b75895b17de3`：

```text
pytorchocr/quantization/
  ax_quantizer.py
  ax_quantizer_utils.py
  quantized_decomposed_dequantize_per_channel.py
  LICENSE
  UPSTREAM.md
```

删除运行时 `sys.path.insert()` 和顶层 `utils` 导入。`--axera-root` 保留为可选的旧命令兼容参数，
但训练、smoke 和导出不再使用外部目录。旧 API 的 `(axera_root, config_path)` 两参数形式也可继续
调用，实际使用第二个配置路径和项目内 vendor。

外部版本产生的旧 checkpoint 在内置 quantizer 图上 `strict=True` 加载成功：

```text
det float/prepared/converted: 418 / 835 / 1322
det ONNX Q/DQ:               184 / 392
det PyTorch/ORT MAE/max:     0 / 0

rec float/prepared/converted: 339 / 648 / 1039
rec ONNX Q/DQ:                168 / 316
rec CTC argmax agreement:     100%
```

使用内置 quantizer 新建的 det checkpoint 已完成 observer freeze、第二 epoch resume；预训练 rec
也完成 CTCLoss QAT 和 checkpoint 到 ONNX 导出。图节点、QDQ dtype 和 Concat 共享域与外部版本基线
一致。

### 2026-07-29：CTW1500 多 epoch 检测 QAT

数据使用 `/home/heqi/dataset/ctw1500/imgs`，包含 `training.txt` 对应 1000 张训练图片和 `test.txt`
对应 500 张测试图片。两份标注已是 PaddleOCR 的 `image_path<TAB>polygon JSON` 格式，无需转换。

训练入口增加可选的 `--val-label-file` 和 `--val-data-dir`。验证时临时关闭 observer，不更新模型
参数，验证完成后按原状态恢复 observer；checkpoint 记录 `best_validation_loss` 并保存 `best.pt`。
PT2E fused fake-quant observer 不支持 FP16 激活，真实 GPU 试跑暴露了
`expected scalar type Float but found Half`，因此 QAT 模式现自动关闭 AMP，浮点训练不受影响。

最初训练结果如下，但导出图随后发现 `stem2a -> stem2b` 中间缺少 QDQ，因此该 checkpoint 和 ONNX
仅保存在 `output/ctw1500_ppocrv6_small_det_qat_pre_samepad_fix` 作为问题复现，不可用于部署：

```text
epoch  train loss  val loss  observer
1      2.93491     2.23530   active
2      2.71974     2.19439   frozen
3      2.73745     2.27135   frozen
```

根因是 PyTorch 对 `Conv2d(padding="same")` 导出 `aten.conv2d.padding`，而原 Axera Conv annotator
只匹配普通 Conv overload，导致 `stem2a` 和 `stem2b` 都没有 annotation。修复 annotator 后从预训练
权重重新进行了 3 epoch QAT：

```text
epoch  train loss  val loss  observer
1      2.85320     2.24350   active
2      2.85567     2.24851   frozen
3      2.62378     2.28974   frozen
```

第 1 epoch 为 SAME-padding-only 中间版本的 `best.pt`。该版本后来发现 hard activation annotation
不完整，现保存在 `output/ctw1500_ppocrv6_small_det_qat_pre_hardact_fix`，统计：

```text
nodes / Q / DQ:           794 / 185 / 344
activation domains:       U8 185
Concat shared:            2 / 2
BatchNormalization:       0
Conv-QDQ-BN:              0
direct/redundant DQ-Q:    0 / 0
unquantized Conv outputs: 0
output:                   [1,1,640,640], finite
```

修复后的局部 ONNX 路径为：

```text
Conv(stem2a) -> Relu -> QuantizeLinear -> DequantizeLinear -> Conv(stem2b)
```

导出校验新增 `unquantized_conv_outputs`，任何 Conv/ConvTranspose 输出（允许融合 Relu/Clip）后缺少
QDQ 都会直接终止导出。相同问题也存在于旧 rec 图；修复后 rec smoke 的 Q/DQ 为 `161/280`，
U8/S16 域为 `143/18`，缺失 Conv 为 0，但旧 rec checkpoint 必须重新训练。

### 2026-07-29：HardSigmoid annotation

检查 SAME-padding 修复版 det ONNX 时发现 8 个 Clip。回溯 FX source stack 后确认：

```text
8 x Paddle Hsigmoid: relu6(1.2 * x + 3) / 6
5 x nn.Hardsigmoid:  backbone SE gate
0 x HardSwish:       当前 PP-OCRv6 small det
```

原 Axera quantizer 没有注册 HardSigmoid。5 个 `aten.hardsigmoid.default` 的 annotation
为 `None`；8 个 Paddle Hsigmoid 被 Add/Mul/Clamp 的零散规则分别量化，产生内部 QDQ：

```text
原始零散规则：Mul -> QDQ -> Add -> QDQ -> Clip -> Div -> QDQ
错误的整式匹配：QDQ -> Mul -> Add -> Clip -> Div -> QDQ
最终 Axera 边界：Mul -> QDQ -> Add -> Clip -> Div -> QDQ
```

#### 是否可以直接替换为 PyTorch HardSigmoid

不能写成 `HardSigmoid(x, slope=0.2, offset=0.5)`。Paddle 的
`F.hardsigmoid` 支持 `slope` 和 `offset`，而 PyTorch 2.6 的
`nn.Hardsigmoid`/`F.hardsigmoid` 只接受 `inplace`，计算公式固定为
`clip(x / 6 + 0.5, 0, 1)`。直接使用 `F.hardsigmoid(x)` 会把斜率从
Paddle 的 `0.2` 改成 `1/6`，模型语义不等价。

可以使用下面的等价替换：

```python
F.hardsigmoid(1.2 * x)
```

因为：

```text
F.hardsigmoid(1.2 * x)
= clip(1.2 * x + 3, 0, 6) / 6
= clip(0.2 * x + 0.5, 0, 1)
```

在当前 PyTorch 2.6 环境中，该表达式与
`F.relu6(1.2 * x + 3) / 6` 的数值对比 `max_abs=0`；PT2E 图为
`aten.mul -> aten.hardsigmoid`，ONNX 图为
`Mul -> HardSigmoid(alpha=1/6, beta=0.5)`。这不是单个
`HardSigmoid(alpha=0.2, beta=0.5)`，但整体计算与 Paddle 完全等价，
对应的 Axera 量化边界应为：

```text
DQ -> Mul(1.2) -> QDQ -> HardSigmoid -> QDQ
```

如果必须导出单个 ONNX `HardSigmoid(alpha=0.2, beta=0.5)`，标准 PyTorch
接口无法直接表达，需要自定义 Torch 算子和 ONNX lowering，并为 PT2E QAT
注册该自定义算子的 annotation。转换代码现已改为
`F.hardsigmoid(1.2 * x)`；Mul 和 HardSigmoid 分别 annotation，使 `1.2 * x`
保留独立输入/输出量化边界。prepared QAT 图和 observer 节点名称因此发生变化，现有
`best.pt` 不能作为完整 QAT checkpoint 直接复用，需要从浮点权重重新 prepare/QAT。

修改后的 PP-OCRv6 small det 预训练 smoke 结果：

```text
prepared nodes:             798
ONNX nodes / Q / DQ:        761 / 177 / 335
HardSigmoid domains:        13 / 13
  scaled Paddle semantics:  8
  standard:                 5
unquantized Conv outputs:   0
Concat shared:              2 / 2
```

13 个 HardSigmoid 来自两套不同的 SE 实现：

- 5 个标准 HardSigmoid 来自 `PPLCNetV4-small` backbone。四个 stage 配置中的
  `use_se=True` 数量分别为 `1/1/2/1`，共 5 个；Paddle 使用无参数
  `Hardsigmoid()`，PyTorch 使用 `nn.Hardsigmoid()`。ONNX 均为
  `HardSigmoid(alpha=1/6, beta=0.5)`，上游量化节点来自 Conv，不存在
  `Mul(1.2)`。
- 8 个 scaled HardSigmoid 来自 `RepLKFPN` neck 的 4 个 `RSELayer` SE 和
  4 个 `inp_conv_se`。原 Paddle MobileNetV3 `SEModule` 显式使用
  `slope=0.2, offset=0.5`，转换后由 `Mul(1.2) -> HardSigmoid` 保持语义。

逐节点检查 8 条 scaled HardSigmoid 路径均为：

```text
DQ(x) -------\
              Mul(1.2) -> QDQ -> HardSigmoid -> QDQ
DQ(const 1.2)/
```

因此常量 `1.2` 不是未量化的浮点旁路：scalar transform 将其变成 `get_attr`
Tensor，Mul annotator 为激活输入和常量输入都配置 U8 qspec，Mul 输出再单独量化后
进入 HardSigmoid。smoke ONNX 保存为
`/tmp/ppocrv6_small_det_scaled_hsigmoid_qat_smoke.onnx`。这只验证新 prepared 图、
反向传播、转换和 QuantONNX 导出；主目录中的旧 `best.pt`/`best_qdq.onnx` 仍对应
复合 Hsigmoid 图，完成新一轮 QAT 前不覆盖。

该 smoke ONNX 已通过现有 Axera 转换链，未出现算子或量化域转换错误。这确认
`DQ -> Mul(1.2) -> QDQ -> HardSigmoid -> QDQ` 可被当前工具链接受；结论仅覆盖
模型转换兼容性，板端输出一致性和检测精度仍需在重新 QAT 后验证。

修复内容：

1. 注册 `aten.hardsigmoid[_.default]`；
2. 在 Add/Mul annotator 之前匹配 Paddle Hsigmoid，但 partition 只包含 `Add -> ReLU6 -> Div`；
3. scalar transform 后的 `get_attr` tensor 常量也参与 `1.2/3/6` pattern 校验；
4. 导出器强制检查 standard/Paddle hard activation 是否均有完整输入输出量化域。

由于 prepared observer 图发生变化，SAME-padding-only checkpoint 不能复用，保存在
`output/ctw1500_ppocrv6_small_det_qat_pre_hardact_fix`。从预训练权重重新进行 3 epoch CTW1500 QAT：

```text
epoch  train loss  val loss  observer
1      2.85642     2.17435   active
2      2.94537     2.61810   frozen
3      2.77078     2.54949   frozen
```

上述 3 epoch 版本错误地把 slope Mul 合入 Hsigmoid，现保存在
`output/ctw1500_ppocrv6_small_det_qat_pre_addboundary_fix`。收窄 partition 后再次从预训练权重训练：

```text
epoch  train loss  val loss  observer
1      2.88058     2.15951   active
2      3.15191     2.43304   frozen
3      2.80440     2.63493   frozen
```

最终第 1 epoch `best.pt` 导出结果：

```text
nodes / Q / DQ:             777 / 177 / 335
activation domains:         U8 177
hard activation domains:    13 / 13
  standard Hardsigmoid:     5
  Paddle Hsigmoid:          8
unquantized Conv outputs:   0
Concat shared:              2 / 2
direct/redundant DQ-Q:      0 / 0
```

8 个 Paddle Hsigmoid 均逐路径验证为：

```text
Mul -> QuantizeLinear -> DequantizeLinear -> Add -> Clip -> Div -> QuantizeLinear
```

校验器不仅检查 Add 输入为 DQ，还反查 `Mul -> Q -> DQ -> Add`，错误整式匹配版本会报
`HardSigmoid patterns are not fully quantized: 5 / 13`。

主模型仍位于 `output/ctw1500_ppocrv6_small_det_qat_pilot/best_qdq.onnx`。识别 smoke 中 5 个标准
HardSigmoid 为 `5/5` 完整域。当前 det 和 PP-OCRv6 small rec CTC 图均未产生 ONNX HardSwish；
此前仅基于标准 ATen 合成图加入的 HardSwish annotator 已撤销，后续遇到实际模型时再按图结构处理。

该轮当时只完成训练链路工程验证，尚未接入 DBPostProcess 和 DetMetric，因此对应 CTW1500 记录
不能把验证 loss 解释为 precision、recall 或 hmean。后续 metric 实现见下一节；该旧 CTW1500
checkpoint 仍未补跑正式指标。

### 2026-07-30：det/rec 正式 metric

新增检测可变 polygon validation collate、`DBPostProcess` + ICDAR IoU evaluator、CTC sequence
accuracy/normalized edit similarity，并增加严格 checkpoint `--eval-only`。新训练依据 YAML 的
`Metric.main_indicator` 选择 `best.pt`，不再用 loss 代替任务精度；最低 validation loss 仍独立记录。

现有 checkpoint 的真实验证结果：

```text
det prepared/CUDA: precision=0.57640, recall=0.39047, hmean=0.46556
rec prepared/CUDA: acc=0.73375, norm_edit_dis=0.88931
tests:             38 passed
```

det checkpoint 是历史代码按最低 loss 选出的 epoch 11；rec 正式 50-epoch checkpoint 按 `acc`
选择 epoch 4。两者的完整跨阶段结果和复现命令见
[ICDAR2015 QAT Metric 验证记录](archive/baselines/icdar2015_qat_metric_validation.md)。

同一验证集完成 converted PT2E 和 QuantONNX A/B。正式 rec QuantONNX 在 ORT 禁用图优化时 acc
`0.71979`，默认优化后为 `0.68464`；det 的对应 hmean 为 `0.46553` 和 `0.44966`。历史 rec smoke
为 `0.71738` 和 `0.65527`。启用或关闭
`onnx_program.optimize()` 不改变该结论，问题位于 ORT 的 QDQ graph optimization。项目的 QDQ
reference 工具默认 `ORT_DISABLE_ALL`，优化模式只保留为诊断项。完整 backend/stage 表见上述文档。

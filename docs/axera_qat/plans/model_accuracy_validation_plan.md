# PP-OCR 浮点与 QAT 精度恢复执行计划

## 1. 计划目标

本计划统一覆盖 PP-OCR 检测和识别模型，目标不是继续增加模型数量，而是先建立可信、可定位、可重复
的精度验证链路：

```text
Paddle float
-> PyTorch eager float
-> export_for_training float
-> prepared(observer off, fake quant off)
-> prepared(observer off, fake quant on)
-> converted PT2E
-> QuantONNX / ORT
-> AXModel
```

每个边界同时验证张量、任务输出和完整验证集指标。首个异常边界未定位前，不进入后续边界，也不启动
新的完整 QAT 训练。

首批模型调整为：

1. PP-OCRv5 mobile rec；
2. PP-OCRv5 mobile det；
3. PP-OCRv6 small rec；
4. PP-OCRv6 small det。

v5 用于先校正成熟的 PytorchOCR/PaddleOCR 训练评估链路；v6 在同一验收体系稳定后回归。四个模型
在进入本计划的 export/PT2E/QAT 阶段前，必须先完成
[`pretrained_training_structure_plan.md`](pretrained_training_structure_plan.md)：完整复现 Paddle
pretrained 训练结构、辅助头、targets、loss 和梯度，再显式投影为单输出部署图。

PP-OCRv5 rec 现有 CTC-only 训练/QAT 结果属于 legacy 部署图诊断，不能作为 pretrained 训练结构
baseline；不得通过关闭 `use_guide` 代替缺失的 NRTR 训练分支。

## 2. 当前问题和路线决策

### 2.1 当前证据不足

- Paddle/PyTorch 已有固定随机输入输出对齐，但没有统一的完整验证集 float A/B；
- route2 检测改变了标准数据采样和 DB loss，现有低 hmean 不能直接归因于 QAT；
- 识别尚未统一验证 resize、`valid_ratio`、字典、blank、CTCLoss 和 decoder；
- 现有 stage comparison 关闭 observer，但没有显式关闭 fake quant；
- ORT graph optimization 已出现任务指标变化，导出成功不能代表精度通过。

### 2.2 工程路线

保留 route2 作为主工程，保留其中：

- PP-OCRv5/v6 模型和严格权重转换；
- PPLCNetV4、RepLKFPN 和部署重参数化；
- PT2E/Axera quantizer、共享量化域和 QuantONNX；
- checkpoint、导出和 QDQ 结构检查。

PaddleOCR 和 `tmp/PytorchOCR` 作为参考实现，用于校正：

- 数据处理和 target；
- DB/CTC loss；
- DB postprocess、CTC decoder 和 metric；
- float 训练及完整验证集结果。

不整体迁移到 PytorchOCR，也不直接复制整套代码。每项能力先对照测试，再按 route2 模块边界接入。

### 当前暂停范围

PP-OCRv6 small det 和 PP-OCRv6 small rec 的 QuantONNX 均包含 `Pad`。截至 2026-08-04，`Pad`
模型在 Axera 工具链上的转换或执行仍存在未解决问题，因此两个模型的 Axera 适配和正式 QAT 训练
全部暂停。现有 smoke QuantONNX 只作为结构诊断产物，不作为正式训练或部署验收依据。

PP-OCRv5 mobile rec 当前 smoke 不包含 `Pad`，作为下一候选模型；PP-OCRv5 mobile det 仍包含
`Pad`，其 Axera 适配和正式 QAT 暂不推进。Pad 暂停只约束对应模型的 Axera/正式 QAT 阶段，不阻止
按 pretrained 结构计划完成 full-training 模型、权重、loss、gradient 和 deployment projection
复现。v5 rec 也必须完成新结构的 smoke 确认门禁后，才能开始正式训练。

恢复任一模型前，必须先在 Axera 工具链上完成最小 `Pad` 模型验证，明确支持的输入/输出 dtype、
量化域共享要求、padding mode、pads/value 表达和转换结果；随后重新执行该模型的 PT2E smoke，
由用户确认结构后才能恢复后续计划。

## 3. 执行规则

每个工作包必须完成：

1. 固定输入、配置、权重、随机种子和 backend；
2. 保存改动前结果；
3. 一次只修改一个变量；
4. 运行局部单元测试；
5. 运行固定样本集成测试；
6. 运行完整验证集任务指标；
7. 保存命令、Git 状态、配置 hash、权重 hash、数据 hash 和结果 JSON；
8. 通过阶段门禁后再继续。

### Smoke QuantONNX 结构门禁

任何模型开始正式 float/QAT 训练前，必须先使用真实预训练权重完成一次当前配置的 PT2E smoke，
并将导出的 QuantONNX 交给用户进行结构分析。流程固定为：

1. 使用当前模型 YAML、当前 QAT JSON、当前 vendor quantizer 和当前导出代码；
2. 使用 Axera deploy 基线图（当前为 `reparameterized: true`），固定 H/W 和 batch；
3. 执行 `export_for_training`、`prepare_qat_pt2e`、至少一次 backward、`convert_pt2e`；
4. 导出 QuantONNX，执行 ONNX checker、QDQ 结构检查和 ORT optimize-off 可执行性检查；
5. 保存 smoke ONNX、命令、输入 shape、配置/权重/量化器 hash 和结构 JSON；
6. 将 smoke ONNX 交给用户检查 Conv/BN、Pad/Pool、激活、Q/DQ 共享域和 requantize 边界；
7. 未经用户确认，不开始正式数据集、多 epoch float/QAT 训练。

任一量化规则、qspec、Pad/Pool、Hard activation、共享量化域或模型图发生修改，都必须重新
执行 smoke。量化图发生变化后，旧 prepared/QAT checkpoint 不得跨图恢复；导出必须 strict load，
不得使用 `strict=False` 掩盖 observer 或 qspec 不匹配。

验证分为三层：

- L1 张量：shape、dtype、finite、MAE、max_abs、p99；
- L2 行为：target、框、token、文本、ignore/match；
- L3 指标：det precision/recall/hmean，rec accuracy/edit distance。

初始容差：

| 边界 | 通过标准 |
| --- | --- |
| 相同预处理输入 | max_abs `<= 1e-6` |
| eager/exported/fake-quant-off | MAE `<= 1e-6`，max_abs `<= 1e-5` |
| Paddle/PyTorch det float map | MAE `<= 1e-4`，无 NaN/Inf |
| Paddle/PyTorch rec probability | MAE `<= 1e-7`，argmax agreement `>= 0.9998` |
| det 框架间 metric | precision/recall/hmean 绝对差 `<= 0.005` |
| rec 框架间 metric | accuracy/edit distance 绝对差 `<= 0.001` |

若 backend 或库版本导致容差需要调整，必须先保存首个失败样本并在报告中说明，不能直接放宽。

## 4. 总体阶段

```text
P0 冻结合同
 -> P1 官方 Paddle 浮点基线
 -> P2 PyTorch 浮点转换验收
 -> P3 数据、target、loss、后处理对齐
 -> P4 export/PT2E 浮点保持性
 -> P5 fake quant 与 convert 精度
 -> P6 QuantONNX 与 Axera 精度
 -> P7 float/QAT 训练验证
 -> P8 v6 和其他模型扩展
```

pretrained 训练结构计划及 P0-P4 是阻塞性基础工作。完整训练结构和 P4 未通过前，P5-P8 不开始。

## 5. P0：冻结实验合同

### P0.1 公共合同

- [ ] 固定模型 YAML、Paddle/PyTorch 权重路径及 SHA256；
- [ ] 固定 train/val 标签、图片目录、样本数和数据 hash；
- [ ] 固定 Paddle、PyTorch、CUDA、OpenCV、Shapely、Pyclipper、ONNX 和 ORT 版本；
- [ ] 固定 CPU/CUDA backend；
- [ ] 历史 checkpoint、ONNX 和日志只读归档，不覆盖；
- [ ] 生成 det 16 张、rec 32 张 debug 样本清单。

### P0.2 检测合同

- [ ] 固定颜色顺序、normalize、动态 eval resize 和固定 640 部署 resize；
- [ ] 固定 DB `thresh`、`box_thresh`、`max_candidates`、`unclip_ratio`；
- [ ] 固定 IoU、don't-care overlap、`box_type` 和 `score_mode`；
- [ ] 统计 care/ignore polygon、空标注和文本框尺寸。

### P0.3 识别合同

- [ ] 固定输入 shape、resize、padding、`valid_ratio`；
- [ ] 固定字典及 SHA256、类别数、blank index、最大文本长度；
- [ ] 固定 CTC collapse、space、unknown 和 metric 文本标准化；
- [ ] 统计字符覆盖、未知字符、超长和空文本样本。

### P0 验证

- 连续运行两次审计，样本顺序和统计必须一致；
- 所有路径启动时打印并校验同一合同 hash；
- det 标注可视化、rec label encode/decode 抽查通过；
- strict load 不允许静默缺失部署分支参数。

### P0 产物和门禁

产物：

```text
artifacts/accuracy_contract/common.json
artifacts/accuracy_contract/det.json
artifacts/accuracy_contract/rec.json
artifacts/accuracy_contract/det_debug_samples.txt
artifacts/accuracy_contract/rec_debug_samples.txt
```

门禁：合同、hash 和 debug 样本齐全后进入 P1。

## 6. P1：建立 Paddle 官方浮点基线

### P1.1 检测

- [ ] 用官方 Paddle 权重和官方 eval transform 跑完整验证集；
- [ ] 保存 debug 样本 normalized input、shrink map、框、score 和匹配统计；
- [ ] 保存逐图 `gtCare/detCare/detMatched`；
- [ ] 输出 precision、recall、hmean。

### P1.2 识别

- [ ] 用官方 Paddle 权重和官方 eval transform 跑完整验证集；
- [ ] 保存 input、raw/centered logits、probability、argmax 和文本；
- [ ] 保存错误样本、置信度和 normalized edit distance；
- [ ] 输出 accuracy 和 edit distance。

### P1 验证

- 完整验证运行两次，指标一致；
- 样本数与 P0 一致，无 NaN/Inf；
- debug 中间张量和逐样本任务结果可复现；
- 任务绝对精度门槛以本阶段实测为准，不引用其他数据集指标。

### P1 产物和门禁

产物：Paddle det/rec baseline JSON、逐样本结果和 debug tensors。

门禁：det 和 rec 至少各获得一个官方浮点完整验证集基线。任一失败时只修复数据和官方评估入口。

## 7. P2：PyTorch 浮点转换验收

### P2.1 PytorchOCR 对照

- [ ] v5 det/rec strict load 对应权重；
- [ ] 直接使用 P1 保存的 normalized input，绕过预处理比较网络输出；
- [ ] 再使用各自完整预处理跑验证集；
- [ ] 分别记录原生后处理和共享 reference 后处理结果。

### P2.2 route2 eager 对照

- [ ] v5 det/rec strict load；
- [ ] 同一 normalized input 比较 Paddle/PytorchOCR/route2；
- [ ] det 比较 shrink map；
- [ ] rec 比较 raw/centered logits、probability、argmax 和文本；
- [ ] 使用共享 reference 后处理跑完整验证集。

### P2.3 重参数化对照

- [ ] 比较 reparameterize 前后 det/rec debug tensors；
- [ ] 对完整验证集比较任务指标；
- [ ] 检查 BN eval 状态、eps、running mean/var 和折叠参数；
- [ ] 单独报告 eager 与 reparameterized，不混合结果。
- [ ] Axera QAT 基线使用 reparameterized deploy graph，优先保证 PT2E/QuantONNX 无残留 BN 且
      Conv QDQ 完整；eager Conv-BN 仅作浮点诊断对照。
- [ ] 单独记录 eager/reparameterized 浮点差异，并通过 QAT 完整指标判断训练能否补偿，不修改
      Paddle baseline 或放宽其门禁。

### P2 验证

- 网络输入相同时，输出达到第 3 节容差；
- 共享后处理下，任务指标达到框架间 metric 容差；
- tensor 对齐但 metric 不对齐时，转 P3 后处理排查；
- tensor 不对齐时，停止并修复模型实现、权重映射或 BN 状态。

### P2 产物和门禁

产物：三框架 float tensor/metric 报告和重参数化报告。

门禁：route2 eager float 必须与 Paddle 对齐。随机输入 parity 不能单独通过本阶段。

## 8. P3：训练与评估语义对齐

P3 按最小单元先测后改。每个子项完成参考对比、route2 修改、单元测试和完整任务回归。

### P3.1 公共预处理

- [ ] 对齐 decode BGR/RGB、interpolation、除以 255、mean/std 和 CHW；
- [ ] 使用全黑、全白、单色、边界尺寸和真实图片；
- [ ] 相同 profile 的 normalized input max_abs `<= 1e-6`。

验证：新增 transform 单元测试，debug 样本逐张一致。

### P3.2 检测几何和 target

- [ ] 分离官方动态 eval、固定 640 部署和训练 crop 三种 profile；
- [ ] 对齐 resize、padding、polygon 正逆变换；
- [ ] 对齐 `EastRandomCropData`、越界过滤和空目标；
- [ ] 对齐 `MakeShrinkMap`、`MakeBorderMap`、ignore 和小文本；
- [ ] 检查 epoch-dependent shrink ratio。

验证：

- 人工 polygon 正逆变换误差 `<= 0.5` pixel；
- 二值 map/mask 完全一致；
- threshold map MAE `<= 1e-6`，max_abs `<= 1e-5`；
- 固定 seed 的 crop 可复现。

### P3.3 检测 DB loss

- [ ] v5 支持标准 `BalanceLoss`、`main_loss_type`、OHEM ratio、alpha/beta；
- [ ] 现有 Dice/Focal 改为显式实验选项；
- [ ] v6 独立核对 DiceFocal、gamma 和辅助分支权重；
- [ ] YAML 参数不得在 `tools/train.py` 中被静默忽略。

验证：全正、全负、混合、全 mask-out 输入下，子 loss、总 loss 和 gradient 与参考误差 `<= 1e-6`。

### P3.4 检测 postprocess 和 metric

- [ ] 对齐 contour、score、unclip、rounding、poly/quad、fast/slow；
- [ ] 对齐 invalid polygon、ignore、don't-care overlap 和一对一匹配；
- [ ] 对相同保存 shrink map 比较三路径框和逐图 matcher；
- [ ] 动态 eval 与固定 640 分别报告。

验证：框数量和过滤决定一致；坐标仅允许 OpenCV 引起的 `<= 1` pixel 差；聚合 metric 达到容差。

### P3.5 识别 resize 和 valid_ratio

- [ ] 对齐保持比例 resize、宽度 rounding、截断和右侧 padding；
- [ ] 对齐 `valid_ratio` 定义、batch collate 和模型使用方式；
- [ ] 分离固定部署 shape 和训练多尺度 profile；
- [ ] 覆盖窄图、超宽图和临界宽度。

验证：每个测试宽度的 resized width、valid_ratio 和 input tensor 一致。

### P3.6 识别字典和 label

- [ ] 对齐 blank、字符索引偏移、space、unknown 和超长处理；
- [ ] 对齐 label padding、label length 和 dtype；
- [ ] 对字典执行 encode/decode round trip；
- [ ] v5/v6 字典不得混用。

验证：固定文本集合的 token ids、length 和反解文本完全一致。

### P3.7 CTCLoss 和 decoder

- [ ] 对齐 logits/log_softmax 输入、N/T/C 维度、blank 和 zero_infinity；
- [ ] 覆盖重复字符、空 target、最大长度和无效 length；
- [ ] 对齐 argmax、重复折叠、blank 删除、confidence 和 metric normalize。

验证：loss 和 logits gradient 误差 `<= 1e-6`；人工 token、保存 logits 和真实输出文本完全一致。

### P3.8 数据 profile

显式提供：

- `float_reference`：复现 PaddleOCR/PytorchOCR 标准训练语义；
- `qat_deterministic`：固定 shape、受控增强；
- `eval_reference`：统一完整验证集评估；
- `deploy_fixed_shape`：固定 shape 部署评估。

验证：同一权重重复 eval 指标稳定；train profile 差异只在 P7 配对训练中评估。

### P3 门禁

- det target、loss、postprocess 和 metric 自动化测试通过；
- rec resize、label、CTCLoss 和 decoder 自动化测试通过；
- 使用修正后的 route2 eager 再跑 P2 完整验证集并达标；
- det/rec Axera QAT 使用通过结构验收的 reparameterized graph；融合差异作为独立风险项跟踪；
- P3 未通过前不做 PT2E 量化调参。

## 9. P4：export 和 prepared 浮点保持性

### P4.1 stage 工具改造

- [x] v5-rec 同时加载 eager 和 `export_for_training` float；
- [x] v5-rec 显式调用 `disable_observer` 和 `disable_fake_quant`；
- [x] v5-rec 报告所有 observer/fake-quant 模块状态；
- [x] v5-rec 每阶段从同一未污染 state 构造；
- [ ] 状态在 validation 和异常退出后恢复。

### P4.2 检测验证

比较：

```text
eager float
-> exported float
-> prepared(observer off, fake quant off)
```

记录 shrink、threshold、binary map、框和 hmean。

### P4.3 识别验证

比较相同三个阶段，记录 raw/centered logits、probability、argmax、文本、accuracy 和 edit distance。

v5-rec 已完成完整 2077 张验证：任务 accuracy/edit delta 为 `0.00048146/0.00006878`，通过
`<=0.001`；严格 logits tensor 容差因 5 个 Conv-BN 的 PT2E QAT 重写未通过。保留正常 PT2E 图，
不采用 eager Conv-BN 固定融合。结果见
`artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_pt2e_fake_off_full_gpu2_20260805.json`。

### P4 验证

- observer/fake-quant on/off/恢复状态测试通过；
- debug 样本满足 eager/exported/fake-quant-off 容差；
- det 完整 metric 差 `<= 0.005`；
- rec 完整 metric 差 `<= 0.001`；
- fake-quant-off 仍不一致时，只排查 capture、wrapper、BN 和 reparameterization。

### P4 产物和门禁

产物：float-preservation stage report 和状态清单。

门禁：det/rec fake-quant-off prepared 均保持 float 精度后，才进入 P5。

## 10. P5：fake quant 和 converted PT2E

### P5.1 fake quant on/off

- [ ] 比较 prepared fake quant off/on；
- [ ] 记录每个 observer 的 scale、zero-point、dtype、min/max；
- [ ] det 记录三张 map、框和 metric delta；
- [ ] rec 记录逐时间步 argmax disagreement、文本变化和 metric delta；
- [ ] 敏感区域通过可重复的区域开关 A/B 定位。

### P5.2 converted

- [ ] prepared fake quant on 与 converted 使用相同 qparams；
- [ ] 比较 CPU/CUDA converted；
- [ ] 验证 QDQ dtype、axis、共享量化域和必要 requant；
- [ ] 运行完整验证集。

### P5 验证

- 所有输出 finite；
- 每个边界分别报告 tensor、行为和任务指标 delta；
- 结构测试和现有 Axera QDQ 测试通过；
- 精度异常先定位到具体量化区域，再修改 qspec，不以 ONNX 后处理掩盖。

### P5 产物和门禁

产物：fake-quant 和 converted det/rec 报告、observer qparams 清单。

门禁：能够明确 fake quant 与 convert 各自损失后进入 P6。

## 11. P6：QuantONNX、ORT 和 Axera

### P6.1 QuantONNX

- [ ] converted 与 ONNX ReferenceEvaluator 比较；
- [ ] converted 与 ORT `ORT_DISABLE_ALL` 比较；
- [ ] ORT optimize-off/on 独立比较；
- [ ] ONNX checker、shape inference、QDQ validator；
- [ ] 保存 optimize 前后模型和 SHA256。

### P6.2 检测验收

- [ ] debug 样本比较 maps、框和 match；
- [ ] 完整验证集比较 precision/recall/hmean；
- [ ] 固定 shape 坐标还原使用 P3 合同。

### P6.3 识别验收

- [ ] debug 样本比较 probability、argmax、sequence 和文本；
- [ ] 完整验证集比较 accuracy/edit distance；
- [ ] 明确记录 ORT optimization 对任务指标的影响。

### P6.4 Axera

- [ ] Pulsar2 编译并保存配置；
- [ ] det 16 张、rec 32 张执行 ORT/AXModel A/B；
- [ ] 比较原始输出和任务输出；
- [ ] 条件允许时跑完整验证集；
- [ ] 保存 AXModel hash 和工具版本。

### P6 验证和门禁

- ORT optimize-off 是 QuantONNX 语义基准；
- optimize-on 只作诊断；
- 编译成功不等于精度通过；
- det/rec 均提交逐边界 metric delta 后才完成 P6。

## 12. P7：训练验证

### P7.1 小样本 float overfit

- [ ] det 选择 8-16 张，rec 选择 16-32 张；
- [ ] 关闭随机增强，使用 P3 已对齐的数据和 loss；
- [ ] 记录 loss、gradient、任务输出和训练集指标；
- [ ] strict reload 最佳 checkpoint。

验证：loss 明显下降，det hmean 建议达到 `0.8`，rec accuracy 建议达到 `0.95`；未达到时与参考框架
同配置 overfit 比较，不直接放宽标准。

### P7.2 float 短周期

- [ ] 1-3 epoch 完整 train/val；
- [ ] `float_reference` 与 `qat_deterministic` 配对；
- [ ] 每轮完整任务 metric；
- [ ] save/resume 后指标一致。

### P7.3 QAT 短周期

- [ ] 从与 float control 相同的 checkpoint 开始；
- [ ] observer 训练期保持开启；
- [ ] validation 关闭 observer、保持 fake quant 开启；
- [ ] 另行评估 fake-quant-off float-preservation；
- [ ] 导出 converted 和 QuantONNX。

### P7.4 完整 QAT

基线通过后，按单变量顺序评估：

1. deterministic 与受控增强；
2. observer 全程开启与冻结；
3. 学习率；
4. EMA；
5. 重参数化训练策略。

验证：正式报告必须同时包含训练起点 float、结束 fake-quant-off、prepared fake-quant-on、converted、
QuantONNX 和 AXModel 指标。

### P7 门禁

正式训练前必须同时满足：

- 当前量化规则对应的 smoke QuantONNX 已通过结构和 ORT optimize-off 检查；
- 用户已完成 smoke ONNX 结构分析并明确确认可以进入训练；
- 模型涉及的 `Pad` 已在 Axera 工具链完成转换和执行验收；
- 正式训练从当前浮点权重或与当前 observer/qspec 图严格匹配的 checkpoint 开始；
- float control、短周期配对、strict reload 和逐阶段导出全部通过。

满足以上条件后，才能开始正式数据集、多 epoch QAT 训练并声明模型 QAT 训练受支持。

## 13. P8：模型扩展

扩展顺序：

1. PP-OCRv5 mobile rec 完成 full MultiHead/MultiLoss 和 P0-P7；
2. PP-OCRv5 mobile det 完成完整 DB 训练结构和 P0-P7；
3. PP-OCRv6 small rec 完成 full MultiHead/MultiLoss 和 P0-P7；
4. PP-OCRv6 small det 完成 aux DBHead/DBLoss 和 P0-P7；
5. PP-OCRv4 mobile det/rec、PP-OCRv5/v6 server、v6 tiny/medium。

每个模型必须独立确认：

- 权重转换和完整验证集 float 指标；
- 数据、loss、后处理和 metric；
- 输入 shape、字典、head 和 reparameterization；
- 独立 QAT JSON 和量化结构；
- P4-P6 各阶段精度。

不得以 v5 mobile 或 v6 small 通过替代其他模型验收。

## 14. 实施批次

### 批次 1：先查清当前异常

1. P0 合同和数据审计；
2. P1 Paddle v5 det/rec 完整浮点指标；
3. P2 PytorchOCR/route2 v5 det/rec 指标；
4. 输出首个异常边界，不改 QAT 参数。

### 批次 2：修复训练评估基础

1. P3 det 数据、DB loss、postprocess；
2. P3 rec resize、字典、CTCLoss、decoder；
3. 重跑 P1-P2；
4. float 对齐后冻结 reference profile。

### 批次 3：精确评估 QAT

1. P4 fake-quant-off；
2. P5 fake quant on 和 converted；
3. P6 QuantONNX/ORT；
4. 输出每个边界的真实精度损失。

### 批次 4：训练和扩展

1. v5 det/rec float overfit 和短周期；
2. v5 det/rec QAT 配对和完整训练；
3. v6 历史 checkpoint 补齐阶段报告；
4. 必要时再训练 v6；
5. 扩展 v4/server/其他规模。

## 15. 测试和工具清单

计划新增或扩展：

```text
tools/audit_ocr_dataset.py
tools/compare_det_frameworks.py
tools/compare_rec_frameworks.py
tools/dump_det_pipeline.py
tools/dump_rec_pipeline.py
tools/compare_db_loss.py
tools/compare_ctc_loss.py
tools/compare_det_postprocess.py
tools/compare_qat_stages.py

tests/test_det_transforms.py
tests/test_det_targets.py
tests/test_rec_transforms.py
tests/test_rec_encoding.py
tests/test_losses.py
tests/test_metrics.py
tests/test_stage_comparison.py
```

所有报告至少包含：命令、Git commit/dirty status、配置/权重/数据 hash、backend、样本数、耗时、数值
误差、任务指标、通过/失败和首个失败样本。

## 16. 停止条件

出现以下任一情况，停止进入下一阶段：

- Paddle 官方完整验证集基线不可复现；
- 权重 strict load 或同输入 float 输出不一致；
- det postprocess 对同一 map 输出不同框；
- rec decoder 对同一 logits 输出不同文本；
- target、DBLoss、CTCLoss 或 gradient 尚未对齐；
- exported/fake-quant-off prepared 已有不可解释误差；
- 只有 validation loss，没有任务指标；
- 只有随机输入或 smoke，没有完整验证集；
- 只验证 ONNX checker/编译，没有任务精度。

## 17. 完成定义

一个 det 或 rec 模型只有同时满足以下条件，才标记为支持：

1. Paddle/PyTorch 完整验证集 float 对齐；
2. 数据、loss、后处理和 metric 参考测试通过；
3. eager/exported/fake-quant-off 精度保持；
4. fake quant 和 converted 的损失独立可解释；
5. QuantONNX checker、QDQ 结构、ORT 和任务指标通过；
6. checkpoint strict reload 和训练 smoke 通过；
7. 完整 QAT 有同起点 float control；
8. Axera 编译和板端任务输出通过或明确标记尚未验收。

检测和识别的技术细节保留在以下附录：

- [检测精度专项附录](detection_accuracy_recovery_plan.md)
- [识别精度专项附录](recognition_accuracy_recovery_plan.md)

后续状态和实施顺序以本主计划为准。

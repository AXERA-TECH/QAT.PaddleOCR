# PP-OCR 检测精度基线恢复计划

本文件是检测模型专项计划。统一入口见
[PP-OCR 浮点与 QAT 精度验证总计划](model_accuracy_validation_plan.md)，识别模型见
[PP-OCR 识别精度基线恢复计划](recognition_accuracy_recovery_plan.md)。

2026-08-05 更新：检测模型在进入本计划的 PT2E/QAT 阶段前，先执行
[PP-OCR Pretrained 训练结构复现计划](pretrained_training_structure_plan.md)。v5 需复现完整 DB 训练
输出和 loss，v6 还必须补齐 `aux_maps_p4/p3/p2` 与辅助 DBLoss；单 shrink 仅属于部署图。

## 1. 背景

当前 route2 已完成 PP-OCRv5/v6 检测模型的 Paddle 权重转换、PT2E QAT、QuantONNX 导出和
验证集评估，但现有结果不能准确衡量 QAT 精度损失：

- Paddle 转 PyTorch 只完成固定随机输入的输出对齐，没有完整验证集任务指标 A/B；
- 所谓浮点基线是 tensor-level parity，不是 Paddle/PyTorch 浮点 hmean 基线；
- route2 检测训练使用整图居中 letterbox 和自研 DB loss，与 PaddleOCR/PytorchOCR 标准训练链路
  存在实质差异；
- 现有 PT2E stage comparison 只关闭 observer，没有显式关闭 fake quant；
- 因此目前无法区分精度损失来自模型转换、数据处理、loss、PT2E prepare、fake quant、convert，
  还是 QuantONNX/ORT。

`tmp/PytorchOCR` 已包含较完整的 PaddleOCR 数据变换、DB loss、后处理和检测 metric，并支持
PP-OCRv5；route2 已具备 PP-OCRv6、严格权重转换、PT2E/Axera QAT 和 QuantONNX。后续采用组合
路线：保留 route2 主工程，以 PytorchOCR/PaddleOCR 作为浮点训练与评估参考，不整体迁移工程。

## 2. 目标

1. 建立同一权重、同一数据集、同一预处理和后处理下的 Paddle/PytorchOCR/route2 浮点指标基线；
2. 查明当前检测 hmean 不合理的首个误差边界；
3. 将 route2 检测数据、DB loss、后处理和 metric 恢复到可与 PaddleOCR 对照的行为；
4. 建立 float、PT2E prepare、fake-quant、converted 和 QuantONNX 的逐阶段精度报告；
5. 浮点链路未验收前，不用新的多 epoch QAT 结果判断量化策略优劣。

## 3. 工程选择

### 3.1 保留在 route2 的能力

- PP-OCRv5/v6 PyTorch 模型实现；
- Paddle 权重严格转换和 tensor parity 工具；
- PPLCNetV4、RepLKFPN 和部署重参数化；
- Axera quantizer、共享量化域和算子专项规则；
- PT2E checkpoint、convert、QuantONNX 导出及 QDQ 检查。

### 3.2 从 PytorchOCR/PaddleOCR 对照或迁移的能力

- PaddleOCR 配置驱动的数据 transform 顺序；
- `EastRandomCropData`/固定形状训练输入策略；
- `MakeShrinkMap` 和 `MakeBorderMap`；
- 标准 DB `BalanceLoss`、OHEM 和配置参数；
- `DBPostProcess` 和 ICDAR detection metric；
- float train/eval 的配置语义。

不得直接整目录复制。每项行为先建立小样本 A/B，再以 route2 当前模块边界接入。

## 4. 执行原则和验证层级

每个阶段都必须包含以下闭环：

1. **冻结输入**：固定配置、权重、样本、随机种子、backend 和依赖版本；
2. **改动前复现**：保存当前行为，不能只保留改动后的结果；
3. **单变量修改**：一次只修改数据、loss、postprocess、PT2E 状态或导出中的一项；
4. **单元验证**：验证局部张量、状态或结构；
5. **集成验证**：在固定小样本上验证完整前向和任务输出；
6. **全量验证**：在完整验证集上计算 precision、recall、hmean；
7. **结果归档**：保存 JSON、日志、样本清单、配置指纹和必要的中间张量；
8. **门禁判断**：通过后才能进入下一阶段，失败时回到首个不一致边界。

验证分为三层，不能互相替代：

- L1 数值一致性：tensor shape、finite、MAE、max_abs、p99；
- L2 行为一致性：target mask、预测框、ignore/match 数量、序列化结果；
- L3 任务精度：precision、recall、hmean。

建议初始容差如下，若因 backend 需要放宽，必须在报告中说明依据：

| 边界 | 初始通过标准 |
| --- | --- |
| 相同预处理输入 | shape/dtype 一致，max_abs `<= 1e-6` |
| 二值 map/mask | 逐元素完全一致 |
| 连续 threshold map | MAE `<= 1e-6`，max_abs `<= 1e-5` |
| Paddle/PyTorch 浮点 shrink map | MAE `<= 1e-4`，无非有限值 |
| eager/exported/fake-quant-off | MAE `<= 1e-6`，max_abs `<= 1e-5` |
| 同一 shrink map 的 postprocess | 框数量、坐标、匹配统计一致 |
| 框架间全验证集 metric | precision/recall/hmean 绝对差 `<= 0.005` |

任务指标一致性门槛只用于比较同权重、同输入和同后处理路径，不代表模型绝对精度合格。绝对精度
门槛应由 Paddle 官方模型在目标数据集上的实测结果确定，不能预先引用其他数据集或论文指标。

## 5. 详细执行阶段

### 阶段 A：冻结现状和统一实验契约

#### A1. 模型和权重清单

- [ ] 首个排查模型固定为 PP-OCRv5 mobile det；
- [ ] 记录 Paddle YAML、Paddle 权重和 PyTorch 权重的绝对路径及 SHA256；
- [ ] 记录转换器版本、strict mapping 统计和允许缺失 key；
- [ ] 记录 eager、reparameterized、QAT checkpoint 和 QuantONNX 产物；
- [ ] PP-OCRv6 small det 只在 v5 通过阶段 F 后开始。

验证：重新执行 strict load 和单张随机输入 parity，结果必须与历史记录一致。输出
`artifacts/det_accuracy/model_contract.json`。

#### A2. 数据集清单

- [ ] 固定 ICDAR2015 train/test 的 label file、data root、样本数量和文件哈希；
- [ ] 检查标签 JSON 可解析、图片存在、polygon 点数合法；
- [ ] 统计 care/ignore polygon 数、空标注图、图片尺寸分布和文本框尺寸分布；
- [ ] 固定 16 张 debug 样本，覆盖横图、竖图、小文本、多文本和 ignore region；
- [ ] 生成排序稳定的 `debug_samples.txt`，后续所有局部比较复用该文件。

验证：连续运行两次数据审计，统计和样本顺序必须一致；抽查 16 张图的标注可视化。

#### A3. 推理与 metric 合同

- [ ] 固定 BGR/RGB、normalize、resize/letterbox、padding value；
- [ ] 固定 `thresh`、`box_thresh`、`max_candidates`、`unclip_ratio`、`score_mode` 和 `box_type`；
- [ ] 固定 IoU 和 don't-care area precision 阈值；
- [ ] 明确动态原尺寸评估与固定 `640x640` 部署评估是两个实验，不混合结果；
- [ ] 记录 Paddle、PytorchOCR、route2、Torch、CUDA、OpenCV、Shapely 和 Pyclipper 版本。

验证：生成 `evaluation_contract.json`，三个框架入口启动时打印并校验同一合同哈希。

**阶段 A 门禁**：A1-A3 产物齐全，历史产物只读归档，三条路径引用同一数据和评估合同。

### 阶段 B：建立浮点验证集基线

#### B1. Paddle 官方浮点基线

- [ ] 使用官方 Paddle 模型和官方 eval transform 评估完整验证集；
- [ ] 保存 16 张 debug 样本的 normalized input、shrink map、预测框和逐图匹配统计；
- [ ] 保存完整 precision、recall、hmean 和每图结果 JSON；
- [ ] 再运行一次，确认结果可复现。

验证：两次 hmean 差为 0 或仅有可解释的浮点末位差；不存在 NaN/Inf；样本数与阶段 A 一致。

#### B2. PytorchOCR 浮点基线

- [ ] 加载与 Paddle 对应的转换权重，禁止静默跳过参数；
- [ ] 使用与 B1 相同的 debug 输入张量，先比较 shrink map；
- [ ] 使用同一评估合同跑完整验证集；
- [ ] 分别记录 PytorchOCR 自带 postprocess 和共享 reference postprocess 的结果。

验证：固定输入 shrink map 满足浮点容差；共享 postprocess 下 metric 与 Paddle 绝对差 `<= 0.005`。
若自带与共享 postprocess 不同，先进入阶段 E，不进入 QAT。

#### B3. route2 eager 浮点基线

- [ ] 使用 strict converted weights 和 `eval()`；
- [ ] 比较 Paddle、PytorchOCR、route2 的同输入 shrink map；
- [ ] 运行 route2 原生评估和共享 reference 评估；
- [ ] 保存当前 route2 结果作为修改前基线。

验证：同输入输出满足浮点容差；共享 postprocess 下 metric 与 B1/B2 差 `<= 0.005`。若 tensor 对齐但
metric 不对齐，问题属于预处理/坐标恢复/metric，不属于模型转换。

#### B4. route2 浮点图边界

依次比较：

1. eager eval；
2. `export_for_training` float；
3. reparameterized eager；
4. reparameterized `export_for_training`。

验证：16 张 debug 样本逐阶段记录 MAE/max_abs；每个阶段跑完整验证集。`export_for_training` 相对
对应 eager 满足 `1e-6/1e-5` 容差；reparameterization 使用独立容差和报告，不得只验证随机输入。

**阶段 B 门禁**：至少获得 Paddle 官方绝对基线，并确认 route2 eager 在共享输入、共享后处理下与
参考 metric 对齐。否则停止训练和 QAT，转到首个失败阶段。

### 阶段 C：数据和 target parity

#### C1. 解码与 normalize parity

- [ ] 使用同一图片字节，分别导出 decoded BGR、模型输入前 HWC 和 normalized CHW；
- [ ] 检查 resize interpolation、除以 255 的时机、mean/std 和 contiguous layout；
- [ ] 增加全黑、全白、单色和真实图片测试。

验证：shape/dtype 一致，normalized tensor max_abs `<= 1e-6`；增加自动化测试。

#### C2. eval 几何变换 parity

- [ ] 对比 Paddle `DetResizeForTest`、PytorchOCR 和 route2 letterbox；
- [ ] 保存 scale、padding、原图 shape、网络 shape 和逆变换参数；
- [ ] 用人工构造矩形验证正变换和逆变换；
- [ ] 分离“官方动态尺寸 eval”和“固定 640 部署 eval”两条 profile。

验证：人工 polygon 经过正逆变换后坐标误差 `<= 0.5` pixel；同一 shrink map 的框可恢复到同一原图坐标。

#### C3. 训练 crop/resize parity

- [ ] 固定随机种子，对齐 `EastRandomCropData` 的 crop 区域选择；
- [ ] 比较 top-left padding 与当前 centered letterbox；
- [ ] 比较 polygon 裁剪、越界过滤和空目标行为；
- [ ] 将标准训练链路和 deterministic QAT 链路做成显式 profile。

验证：固定 crop 参数时输出图像和 polygon 一致；随机模式在相同 seed 下可复现；空目标 batch 可正常处理。

#### C4. DB target parity

- [ ] 比较 polygon orientation 和合法性处理；
- [ ] 比较 min text size、ignore tag 和 shrink ratio；
- [ ] 比较 shrink map/mask 和 threshold map/mask；
- [ ] 检查 epoch-dependent shrink ratio 是否启用；
- [ ] 保存 target 可视化及像素统计。

验证：二值 map/mask 逐元素一致；threshold map 满足连续 map 容差；差异像素必须可定位到明确 polygon。

#### C5. 数据 profile 精度 A/B

- [ ] `float_reference` 复现参考 transform；
- [ ] `qat_deterministic` 使用固定 shape 和受控增强；
- [ ] 使用同一浮点权重分别评估两个 eval profile；
- [ ] 后续短训练中只改变 train profile，eval 始终使用固定 reference contract。

验证：先确认 eval profile 不改变模型语义；train profile 的影响必须通过阶段 G 的配对训练衡量，不能
由单张图主观判断。

**阶段 C 门禁**：C1-C4 自动化测试通过；训练和评估 profile 显式分离；所有几何变换可逆且有元数据。

### 阶段 D：DB loss parity

#### D1. PP-OCRv5 标准 DB loss

- [ ] 实现配置驱动的 `BalanceLoss`、`main_loss_type` 和 `ohem_ratio`；
- [ ] 对齐正负样本计数、hard-negative 排序、mask 和 eps；
- [ ] 对齐 `alpha * shrink + beta * threshold + binary`；
- [ ] 保留现有 Dice/Focal，命名为独立实验 loss，不作为 v5 默认值。

验证：构造全正、全负、混合、全 mask-out 和无负样本输入；与 PytorchOCR 逐项比较子 loss、总 loss
和 prediction gradient，误差 `<= 1e-6`。

#### D2. PP-OCRv6 loss 合同

- [ ] 核对 Paddle v6 `DiceFocalLoss` 的准确组成和 gamma；
- [ ] 核对 DBHead 辅助输出及 `aux_weight_p4/p3/p2`；
- [ ] 确认 route2 部署重参数化训练图是否保留训练所需辅助分支；
- [ ] 禁止将 v5 OHEM 配置隐式用于 v6。

验证：以 Paddle v6 固定 predictions/targets 比较所有主/辅助 loss 和梯度；如果 route2 首版不支持辅助
loss，必须量化其精度影响并在支持矩阵中标记，不得称为完整训练等价。

#### D3. loss 集成回归

- [ ] 单 batch 前向、反向和 optimizer step；
- [ ] 检查所有应训练参数存在 finite gradient；
- [ ] 记录各 loss 的数量级和 OHEM 正负样本数；
- [ ] 连续运行相同 seed，首步 loss 一致。

验证：新增 loss 单元测试和训练 smoke；原 rec/PT2E/QDQ 测试不得回归。

**阶段 D 门禁**：v5 loss/gradient 与参考一致；v6 的支持边界明确；不再由 `tools/train.py` 静默忽略
YAML 中的 `main_loss_type`、`balance_loss` 或 `ohem_ratio`。

### 阶段 E：后处理和 metric parity

#### E1. DBPostProcess 单元 parity

- [ ] 使用人工 shrink map 覆盖空图、单框、多框、边界框、小框和重叠框；
- [ ] 对齐 contour API、mini box、score、unclip 和坐标 rounding；
- [ ] 对齐 `poly/quad`、`fast/slow` 和 dilation 行为；
- [ ] 输出 score，不在 metric 前丢弃诊断信息。

验证：相同 map 下框数量和过滤决定完全一致，坐标允许 `<= 1` pixel 的 OpenCV 版本差异；任何差异都
保存中间 contour 和 score。

#### E2. DetectionIoUEvaluator parity

- [ ] 覆盖 care/ignore、invalid polygon、空 GT、空 prediction、重复 prediction 和 IoU 边界；
- [ ] 对齐 `>` 与 `>=`；
- [ ] 对齐 don't-care overlap 和一对一匹配顺序；
- [ ] 比较逐图 `gtCare/detCare/detMatched`，不能只比较最终 hmean。

验证：参考测试集逐图统计完全一致，聚合 precision/recall/hmean 完全一致。

#### E3. 全验证集 postprocess A/B

- [ ] 同一组保存的 shrink map 分别进入 Paddle/PytorchOCR/route2 postprocess；
- [ ] 比较每图预测框、score、care 和 matched；
- [ ] 输出首个不一致样本及可视化；
- [ ] 完成动态 eval 与固定 640 eval 的独立报告。

验证：共享 reference postprocess 的结果完全一致；原生 postprocess metric 差 `<= 0.005`，否则继续修复。

**阶段 E 门禁**：给定相同 shrink map，不再出现框架间不可解释的 metric 差异。

### 阶段 F：PT2E 分阶段精度审计

#### F1. stage 工具状态控制

- [ ] 增加 eager 和 `export_for_training` float 输入；
- [ ] 显式调用并验证 `disable_observer`、`disable_fake_quant`、`enable_fake_quant`；
- [ ] 读取每个 fake-quant/observer 模块状态并写入报告；
- [ ] 每个阶段从同一 checkpoint/state 构造，禁止前一阶段推理污染后一阶段 observer；
- [ ] prepared 与 converted 分别记录 CPU/CUDA backend。

验证：状态单元测试覆盖 on/off/恢复和异常退出；fake-quant-off 输出必须不含有效量化扰动。

#### F2. float 到 prepared 边界

依次比较 eager、exported float、prepared observer off + fake quant off。

验证：16 张 debug 样本满足 `1e-6/1e-5` 容差；完整验证集 metric 差 `<= 0.005`。未通过时只排查
PT2E capture、BN/reparameterization 和 wrapper，不调整量化参数。

#### F3. fake quant 精度损失

比较 prepared fake quant off/on，并记录每个 observer 的 scale、zero-point、dtype、min/max。

验证：输出 finite；完整验证集报告 map 误差和 metric delta；敏感层分析必须基于可重复的逐层或区域
A/B，不能直接用 ONNX 图猜测。

#### F4. convert PT2E 边界

- [ ] prepared fake quant on 与 converted 使用相同 observer qparams；
- [ ] 比较 shrink/threshold/binary map；
- [ ] 比较 CPU 与 CUDA converted；
- [ ] 验证 Q/DQ dtype、axis 和共享域结构。

验证：记录 prepared-to-converted 的 MAE、max_abs、框级和 metric delta；结构测试全部通过。

#### F5. QuantONNX/ORT 边界

- [ ] converted 与 ONNX ReferenceEvaluator 比较；
- [ ] converted 与 ORT `ORT_DISABLE_ALL` 比较；
- [ ] ORT optimize off/on 独立比较；
- [ ] 执行 ONNX checker、shape inference 和 QDQ validator；
- [ ] 保存 optimize 前后模型和哈希。

验证：以 ORT optimize-off 作为 QuantONNX 语义基准；optimize-on 只能作为诊断，不得覆盖交付基线。

#### F6. Axera 边界

- [ ] Pulsar2 编译；
- [ ] 固定 16 张样本执行 ORT/AXModel A/B；
- [ ] 比较板端原始 shrink map和最终框；
- [ ] 记录编译配置、工具版本和 AXModel 哈希。

验证：编译成功不等于精度通过；必须提交 map、框和完整验证集 metric 差异。

**阶段 F 门禁**：每个边界都有独立 delta，且能明确首个显著下降阶段。fake-quant-off 未通过前禁止
修改 observer、位宽或 Axera 算子规则。

### 阶段 G：短周期训练和正式 QAT

#### G1. 小样本 float overfit

- [ ] 从 debug 集选择 8 至 16 张 care 标注有效的图片；
- [ ] 关闭随机增强，使用已对齐 target 和标准 loss；
- [ ] 记录初始和每轮 loss、gradient、shrink map 及训练集 hmean；
- [ ] 保存最佳 checkpoint 并 strict reload。

验证：loss 明显下降、预测由空结果转为覆盖 GT、训练集 hmean 持续提升。建议目标 `hmean >= 0.8`；
若数据难度导致无法达到，必须用 Paddle/PytorchOCR 同配置 overfit 作为相对基线。

#### G2. 1 至 3 epoch float smoke

- [ ] 完整 train/val loader；
- [ ] `float_reference` 与 `qat_deterministic` 只改变 train transform；
- [ ] 每轮计算完整 validation metric；
- [ ] 验证 checkpoint save/resume 后指标一致。

验证：训练无 NaN/Inf，指标方向与参考实现一致，resume 前后同 checkpoint metric 差 `<= 0.005`。

#### G3. 1 至 3 epoch QAT 配对 smoke

- [ ] 从与 G2 相同的浮点 checkpoint 开始；
- [ ] observer 训练期保持开启，validation 临时关闭并恢复；
- [ ] 同时导出 fake-quant-off、prepared、converted 和 QuantONNX 结果；
- [ ] 与同轮 float control 比较，而不是只比较历史模型。

验证：训练、strict reload、阶段 F 和 QuantONNX 全链路通过；observer 状态恢复测试通过。

#### G4. 完整 QAT 和单变量实验

基线通过后，依次单独评估：

1. deterministic 与受控增强；
2. observer 全程开启与冻结；
3. 学习率；
4. EMA；
5. 重参数化前后训练策略。

每次只改变一个变量，固定 seed，并保留 float control。

验证：正式报告必须同时包含训练起点 float、结束时 fake-quant-off、prepared fake-quant-on、converted、
QuantONNX 和 Axera metric，不得只报告 best QAT hmean。

#### G5. PP-OCRv6 回归

- [ ] 复用 A-F 的工具和报告格式；
- [ ] 单独处理 PPLCNetV4、RepLKFPN、辅助分支和 v6 DiceFocal loss；
- [ ] 不复用 v5 的精度阈值或量化 JSON；
- [ ] 完成 v6 float control 后再开始 v6 QAT。

验证：v6 必须独立通过 A-F 门禁；v5 通过不代表 v6 自动通过。

**阶段 G 门禁**：float control 可信、短周期配对实验完成、完整 QAT 各边界精度可追溯。

## 6. 优先级和停止条件

优先级：

1. 完整验证集浮点 hmean；
2. eval 预处理、坐标还原和 postprocess parity；
3. 训练 target 与 DB loss parity；
4. observer/fake-quant-off stage comparison；
5. 新的 QAT 训练；
6. Axera 编译和板端验收。

出现以下情况时停止进入下一阶段：

- Paddle 与转换后 PyTorch 的同输入输出不一致；
- route2 eager float 与参考浮点任务指标明显不一致；
- `export_for_training` 或 fake-quant-off prepared 已产生不可解释误差；
- 后处理对同一 shrink map 给出不同框；
- 训练 target/loss 尚未完成参考实现对齐；
- 只比较 validation loss，没有任务 metric。

## 7. 验证矩阵

| 阶段 | 单元测试 | 固定样本 | 完整验证集 | 关键产物 |
| --- | --- | --- | --- | --- |
| A | 路径/hash/schema | 16 张清单 | 数据统计 | contract JSON |
| B | 权重 strict load | map/box parity | float hmean | float baseline report |
| C | transform/target | input/map 可视化 | eval profile A/B | data parity report |
| D | loss/gradient | 单 batch backward | float smoke | loss parity report |
| E | contour/IoU | box/match parity | metric parity | postprocess report |
| F | observer/fake-quant 状态 | 各 stage map/box | 各 stage hmean | stage report |
| G | train/resume | overfit/smoke | full QAT | training report |

每个报告至少包含：Git commit、dirty status、命令、配置 hash、权重 hash、数据合同 hash、backend、样本数、
耗时、数值误差、任务指标、通过/失败和首个失败样本。

## 8. 建议新增或扩展的工具

以下是计划中的职责划分，文件名可在实施时按现有结构调整：

- `tools/audit_det_dataset.py`：数据集、标注和尺寸统计；
- `tools/compare_det_frameworks.py`：Paddle/PytorchOCR/route2 浮点 map 和 metric；
- `tools/dump_det_pipeline.py`：预处理、polygon、target 和逆变换中间结果；
- `tools/compare_db_loss.py`：子 loss、OHEM 和 gradient parity；
- `tools/compare_det_postprocess.py`：相同 map 的框和 matcher parity；
- `tools/compare_qat_stages.py`：扩展 fake quant on/off 和 float stages；
- `tests/test_det_transforms.py`：decode/resize/crop/normalize；
- `tests/test_det_targets.py`：shrink/border map；
- `tests/test_losses.py`：标准 DB/OHEM 和 v6 loss；
- `tests/test_metrics.py`：postprocess 和 IoU evaluator；
- `tests/test_stage_comparison.py`：observer/fake-quant 状态和 stage parity。

## 9. 交付物

- Paddle/PytorchOCR/route2 浮点验证集对比工具和报告；
- 固定样本的数据、target、loss 和 postprocess parity 测试；
- 配置驱动的标准 DB loss 与确定性 QAT 数据 profile；
- 完整 PT2E stage comparison，包含 fake quant on/off；
- PP-OCRv5 mobile det 精度根因报告；
- PP-OCRv6 small det 回归报告；
- 更新后的训练、QuantONNX 和 Axera 验收文档。

## 10. 当前判断

当前检测精度异常不能直接归因于 Axera QAT。route2 在浮点验证集基线缺失的情况下，同时改变了
检测训练采样方式和 DB loss。应先修复实验基线，再评估 observer、fake quant 和 QuantONNX 的实际
精度损失。

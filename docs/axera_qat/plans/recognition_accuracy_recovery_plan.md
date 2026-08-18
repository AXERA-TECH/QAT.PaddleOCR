# PP-OCR 识别精度基线恢复计划

2026-08-05 更新：模型结构工作先执行
[PP-OCR Pretrained 训练结构复现计划](pretrained_training_structure_plan.md)。本计划中的 CTC-only
边界继续用于部署输出精度验证，但不再代表完整 pretrained/QAT 训练结构。

## 1. 背景

当前 route2 已完成 PP-OCRv6 small rec 的 Paddle 权重转换、50 epoch QAT、QuantONNX 和验证集评估，
并完成 PP-OCRv5 mobile rec 的真实权重浮点对齐及 QAT/QuantONNX smoke。但是现有证据仍有缺口：

- Paddle/PyTorch 浮点转换主要验证固定随机输入 logits、Softmax 和 argmax；
- 尚未统一完成 Paddle/PytorchOCR/route2 的完整验证集 accuracy/edit distance A/B；
- 当前 stage comparison 只关闭 observer，没有显式关闭 fake quant；
- rec resize/padding、`valid_ratio`、字典索引、CTC blank 和重复字符折叠都可能造成指标差异；
- PP-OCRv5 mobile rec 尚未开始真实数据正式训练；
- ORT graph optimization 曾明显改变 rec 任务指标，必须独立审计。

因此识别模型也必须建立独立于检测模型的浮点和 QAT 分阶段精度链路。

## 2. 目标

1. 建立同权重、同图片、同字典和同解码规则下的 Paddle/PytorchOCR/route2 浮点验证集基线；
2. 区分模型转换、预处理、CTC head、解码、PT2E prepare、fake quant、convert 和 ONNX 的误差；
3. 验证识别训练 label、length、CTCLoss 和 metric 与参考实现一致；
4. 建立 PP-OCRv5 mobile rec 的可信 float control，再决定是否启动完整 QAT；
5. 对 PP-OCRv6 small rec 的历史 QAT 结果重新补齐 float/fake-quant-off 边界。

## 3. 验证层级和初始门槛

每个阶段同时验证：

- L1 数值：logits、centered logits、Softmax probability；
- L2 行为：逐时间步 argmax、CTC collapse 序列、解码文本；
- L3 任务指标：accuracy、normalized edit distance；
- L4 结构：输出 shape、时间步、类别数、blank index 和 QDQ 结构。

建议初始门槛：

| 边界 | 初始通过标准 |
| --- | --- |
| 相同预处理输入 | shape/dtype 一致，max_abs `<= 1e-6` |
| Paddle/PyTorch probability | MAE `<= 1e-7`，argmax agreement `>= 0.9998` |
| eager/exported/fake-quant-off | logits MAE `<= 1e-6`，argmax agreement `1.0` |
| CTC decode parity | 固定样本序列和文本完全一致 |
| 框架间验证集 metric | accuracy/edit distance 绝对差 `<= 0.001` |

logits 可存在对 Softmax 不敏感的整体平移，因此必须同时报告 raw logits、centered logits 和 probability，
不能只用 raw logits MAE 判定识别错误。

## 4. 详细执行阶段

### 阶段 A：冻结识别实验合同

#### A1. 模型与权重

- [ ] 首个模型固定为 PP-OCRv5 mobile rec；
- [ ] 记录 Paddle YAML、Paddle/PyTorch 权重路径和 SHA256；
- [ ] 记录 full MultiHead、CTC deploy projection、SVTR kernel、输出时间步和类别数；
- [ ] full-training 模型 strict load CTC 和 GTC/NRTR 全部参数；
- [ ] 固定 deployment projection 明确删除的 GTC/NRTR 参数集合。

验证：重新执行固定输入 Paddle/PyTorch parity，结果与历史记录一致；输出
`artifacts/rec_accuracy/model_contract.json`。

#### A2. 数据与字典

- [ ] 固定 ICDAR2015 rec train/test 标签、图片根目录及文件哈希；
- [ ] 统计样本数、空文本、超长文本、字符覆盖率和缺失字符；
- [ ] 固定字符字典、`use_space_char`、最大文本长度和字典 SHA256；
- [ ] 固定 32 张 debug 样本，覆盖窄图、宽图、重复字符、空格、数字、大小写和未知字符；
- [ ] 明确 v5/v6 字典不能混用。

验证：连续两次审计结果一致；所有 debug label 可编码和反解码；未知字符处理有明确记录。

#### A3. 识别评估合同

- [ ] 固定输入 shape、颜色顺序、normalize、resize interpolation 和右侧 padding value；
- [ ] 固定 CTC blank index、重复折叠、blank 移除和字符映射；
- [ ] 固定是否移除空格、大小写规则和 metric 文本标准化；
- [ ] 固定 CPU/CUDA、Paddle/PyTorch/ORT 版本；
- [ ] 每个入口校验同一合同 hash。

验证：人工 token 序列在三条路径下得到完全相同的文本和置信度。

**阶段 A 门禁**：模型、数据、字典和解码合同齐全，三条框架路径不再各自隐式选择字典或解码规则。

### 阶段 B：完整验证集浮点基线

#### B1. Paddle 官方浮点基线

- [ ] 使用官方 Paddle 模型和 eval transform 跑完整验证集；
- [ ] 保存 debug 样本的 input、logits、probability、argmax、token 序列和文本；
- [ ] 输出 accuracy、normalized edit distance、错误样本和置信度；
- [ ] 重复运行验证可复现性。

验证：两次任务指标一致，无 NaN/Inf，处理样本数与数据合同一致。

#### B2. PytorchOCR 浮点基线

- [ ] strict load 对应转换权重；
- [ ] 复用 B1 保存的 normalized input 比较模型输出；
- [ ] 分别使用 PytorchOCR decoder 和共享 reference decoder；
- [ ] 跑完整验证集。

验证：probability、argmax 和共享 decoder 指标满足阶段 3 门槛；否则定位到模型输出或 decoder。

#### B3. route2 eager 浮点基线

- [ ] strict load route2 权重，关闭训练态 softmax 等分支；
- [ ] 同输入比较 Paddle/PytorchOCR/route2 logits；
- [ ] 比较 raw、centered、probability 和 argmax；
- [ ] 使用共享 decoder 跑完整验证集。

验证：固定输入 argmax agreement 和验证集 metric 达标。tensor 对齐但文本不一致时只排查 decoder/字典。

#### B4. route2 浮点图边界

依次比较：

1. eager eval；
2. `export_for_training` float；
3. reparameterized eager；
4. reparameterized exported float。

验证：32 张 debug 样本和完整验证集均比较；exported 相对 eager 的 argmax agreement 为 `1.0`，metric
差 `<= 0.001`。重参数化单独报告，不能只测随机输入。

**阶段 B 门禁**：route2 eager/exported float 在共享输入和 decoder 下与 Paddle 官方验证集指标一致。

### 阶段 C：识别预处理和 label parity

#### C1. 图像 decode/normalize

- [ ] 比较 OpenCV/Paddle decode 的 BGR/RGB；
- [ ] 比较高度缩放、宽度计算、最大宽度截断和 interpolation；
- [ ] 比较 normalize 顺序和 CHW layout；
- [ ] 覆盖全黑、全白、窄图、超宽图和真实图片。

验证：相同 profile 的 normalized input max_abs `<= 1e-6`。

#### C2. resize/padding 与 valid_ratio

- [ ] 对齐保持比例 resize 后宽度 rounding；
- [ ] 对齐右侧 padding value；
- [ ] 对齐 `valid_ratio` 的定义、范围和 batch collate；
- [ ] 检查模型是否实际使用 `valid_ratio` 或 sequence mask；
- [ ] 分离固定 `48x320` 部署 profile 和训练多尺度 profile。

验证：人工宽度集合逐项比较 resized width、valid_ratio 和 input tensor；边界宽度必须有自动化测试。

#### C3. 字典和 CTC label encode

- [ ] 对齐 blank 的插入位置；
- [ ] 对齐字符索引偏移、space、unknown 和超长文本过滤；
- [ ] 对齐 label padding value、label length 和 batch dtype；
- [ ] 对字符字典做 round-trip 测试。

验证：固定文本集合的 encoded ids、length 和 decoded text 完全一致。

#### C4. 训练与评估 profile

- [ ] `float_reference` 复现 PaddleOCR/PytorchOCR 训练预处理；
- [ ] `qat_deterministic` 使用固定 shape 和关闭随机增强；
- [ ] eval 始终使用固定 reference contract；
- [ ] 记录两种 train profile，不通过隐式 CLI 行为切换。

验证：eval profile 对同一权重给出稳定指标；train profile 差异在阶段 G 用配对训练验证。

**阶段 C 门禁**：input、valid_ratio、label ids 和 length 均有参考 parity 测试。

### 阶段 D：CTCLoss 和 decoder parity

#### D1. CTCLoss 数值和梯度

- [ ] 对齐 logits/log_softmax 的输入约定；
- [ ] 对齐 `[N,T,C]` 与 `[T,N,C]` 转换；
- [ ] 对齐 blank、input length、target length 和 zero_infinity；
- [ ] 覆盖重复字符、空 target、最大长度和无效 length；
- [ ] 比较 loss 和 logits gradient。

验证：相同 logits/labels 下 route2 与参考 CTCLoss 数值和 gradient 误差 `<= 1e-6`。

#### D2. CTC decoder parity

- [ ] 对齐 argmax、重复折叠和 blank 删除顺序；
- [ ] 对齐 token confidence 和文本 confidence；
- [ ] 对齐 metric 文本 normalize；
- [ ] 覆盖连续重复字符被 blank 分隔和未分隔两种情况。

验证：人工 token、保存 logits 和真实模型输出的文本及 confidence 一致。

#### D3. 单 batch 训练回归

- [ ] forward、loss、backward、optimizer step；
- [ ] 检查所有 CTC 部署分支参数有 finite gradient；
- [ ] 同 seed 首步 loss 可复现；
- [ ] checkpoint strict reload 后输出一致。

验证：loss/decoder 单元测试、训练 smoke 和现有 det/QDQ 测试全部通过。

**阶段 D 门禁**：CTCLoss、gradient 和 decoder 均与参考行为一致。

### 阶段 E：PT2E 分阶段精度审计

#### E1. observer/fake-quant 显式状态

- [x] stage 工具加入 eager/exported float（`tools/compare_pt2e_float_preservation.py`）；
- [x] 显式控制 observer off、fake quant off/on（`disable_observer`/`disable_fake_quant`）；
- [x] 报告所有模块状态和 observer qparams（含 trainer `_any_observer_enabled` 状态记忆）；
- [x] 每个阶段从未污染的同一 checkpoint 构造（同一浮点权重分别 prepare）；
- [x] 状态在异常退出后可恢复（trainer evaluate finally 对称恢复 + 单元测试）。

（2026-08-18 补记：E1 已由 `compare_pt2e_float_preservation.py` 与 trainer 验证恢复逻辑落地，
完整结果见训练记录 §17-20 与 `records/model_accuracy_validation_results.md` P4。）

验证：状态单元测试覆盖 off/on/恢复；fake-quant-off 不产生有效量化扰动。

#### E2. float 到 prepared

比较 eager、exported float、prepared observer off + fake quant off。

验证：logits、centered logits、probability、argmax、文本和完整验证集 metric 均达标。未通过时不调整
observer 或量化位宽。

#### E3. fake quant on/off

- [ ] 比较 prepared fake quant off/on；
- [ ] 记录每个 observer 的 scale、zero-point、dtype 和范围；
- [ ] 输出逐时间步 argmax disagreement；
- [ ] 输出错误文本变化和置信度变化。

验证：输出 finite，完整报告 accuracy/edit-distance delta，并定位首个敏感量化区域。

#### E4. converted PT2E

- [ ] prepared fake quant on 与 converted 使用相同 qparams；
- [ ] 比较 CPU/CUDA converted；
- [ ] 比较 logits/probability/sequence/metric；
- [ ] 验证 QDQ dtype、axis、Concat shared domain 和必要 requant。

验证：数值、任务指标和结构均形成独立报告。

#### E5. QuantONNX/ORT

- [ ] converted 与 ONNX ReferenceEvaluator 比较；
- [ ] converted 与 ORT optimize-off 比较；
- [ ] ORT optimize-off/on 独立比较；
- [ ] checker、shape inference 和 QDQ validator；
- [ ] 保存模型 hash 和逐样本输出。

验证：ORT optimize-off 为语义基准；若 optimize-on 改变 accuracy，不得作为默认精度结果。

#### E6. Axera

- [ ] Pulsar2 编译；
- [ ] 32 张 debug 样本执行 ORT/AXModel A/B；
- [ ] 比较 probability、argmax、CTC sequence 和文本；
- [ ] 跑完整验证集任务指标；
- [ ] 记录编译配置和 AXModel hash。

验证：编译成功之外，必须报告 sequence agreement、accuracy 和 edit-distance delta。

**阶段 E 门禁**：eager 到 Axera 每个边界均有独立数值和任务指标，可明确首个精度下降阶段。

### 阶段 F：短周期训练和正式 QAT

#### F1. 小样本 float overfit

- [ ] 选择 16 至 32 张字符均可编码的图片；
- [ ] 固定 shape、关闭随机增强；
- [ ] 记录 loss、argmax sequence、文本和训练集 accuracy；
- [ ] strict reload 最佳 checkpoint。

验证：loss 明显下降，训练集 accuracy 持续提升；建议目标 `>= 0.95`，否则与参考框架同配置 overfit 比较。

#### F2. 1 至 3 epoch float smoke

- [ ] 完整 train/val loader；
- [ ] float reference 和 deterministic profile 配对；
- [ ] 每轮计算 accuracy/edit distance；
- [ ] save/resume 指标回归。

验证：训练 finite，指标方向与参考一致，resume 前后 metric 差 `<= 0.001`。

#### F3. 1 至 3 epoch QAT smoke

- [ ] 从 F2 同一浮点 checkpoint 开始；
- [ ] observer 训练期保持开启；
- [ ] validation 只冻结 observer，fake quant 保持开启；
- [ ] 额外单独评估 fake-quant-off float-preservation；
- [ ] 导出 converted 和 QuantONNX。

验证：训练、strict reload、阶段 E、ONNX 和任务指标全链路通过。

#### F4. 完整 QAT

- [ ] PP-OCRv5 mobile rec 获得 float control 后再启动；
- [ ] observer freeze、增强、EMA 和学习率只做单变量实验；
- [ ] 同时保留 float control；
- [ ] PP-OCRv6 历史 QAT checkpoint 补跑阶段 B/E，不直接重训覆盖。

验证：报告训练起点 float、结束 fake-quant-off、prepared、converted、QuantONNX 和 Axera 指标。

#### F5. 版本扩展

- [ ] PP-OCRv5 mobile rec；
- [ ] PP-OCRv6 small rec；
- [ ] PP-OCRv4 mobile rec；
- [ ] v5/v6 server 和其他规模。

验证：每个模型独立通过 A-E；字典、输入 shape、head 和量化 JSON 均不能无验证复用。

**阶段 F 门禁**：可信 float control、短周期配对实验和完整逐阶段精度报告齐全。

## 5. 优先级和停止条件

执行优先级：

1. Paddle/PytorchOCR/route2 完整验证集浮点指标；
2. resize/padding、valid_ratio、字典和 decoder；
3. CTCLoss/gradient；
4. fake-quant-off prepared；
5. fake quant、converted 和 QuantONNX；
6. 新的正式 QAT；
7. Axera 板端验收。

以下任一条件成立时停止进入后续阶段：

- 字典 hash、类别数、blank index 或输出时间步不一致；
- 相同 normalized input 的浮点 probability/argmax 不一致；
- eager 与 exported/fake-quant-off prepared 不一致；
- decoder 对相同 logits 输出不同文本；
- CTCLoss 或 gradient 尚未对齐；
- 只有 logits MAE，没有 sequence 和任务指标；
- 只有 smoke，没有完整验证集评估。

## 6. 验证矩阵

| 阶段 | 单元测试 | 固定样本 | 完整验证集 | 关键产物 |
| --- | --- | --- | --- | --- |
| A | schema/hash/encode | 32 张清单 | 数据统计 | rec contract JSON |
| B | strict load | logits/prob/sequence | float acc/edit | float baseline report |
| C | resize/valid_ratio | input/label parity | eval profile | data parity report |
| D | CTC loss/gradient | batch train/decode | float smoke | CTC parity report |
| E | observer/fake-quant | stage logits/text | stage acc/edit | stage report |
| F | train/resume | overfit/QAT smoke | full QAT | training report |

## 7. 建议工具和测试

- `tools/audit_rec_dataset.py`：数据、字符覆盖和字典审计；
- `tools/compare_rec_frameworks.py`：三框架浮点输出和任务指标；
- `tools/dump_rec_pipeline.py`：resize、padding、valid_ratio、label ids；
- `tools/compare_ctc_loss.py`：loss 和 gradient parity；
- `tools/compare_qat_stages.py`：扩展 float/fake-quant on/off；
- `tests/test_rec_transforms.py`：resize/padding/valid_ratio；
- `tests/test_rec_encoding.py`：字典、blank 和 label encode；
- `tests/test_losses.py`：CTCLoss 数值和 gradient；
- `tests/test_metrics.py`：CTC collapse、accuracy 和 edit distance；
- `tests/test_stage_comparison.py`：rec 各 PT2E stage。

## 8. 交付物

- PP-OCRv5 mobile rec 三框架浮点验证集报告；
- PP-OCRv6 small rec 历史 checkpoint 的补充 stage 报告；
- resize/valid_ratio、字典、CTCLoss 和 decoder parity 测试；
- observer/fake-quant on/off 分阶段评估；
- QuantONNX optimize-off/on 识别精度报告；
- PP-OCRv5/v6 rec QAT 与 Axera 验收报告。

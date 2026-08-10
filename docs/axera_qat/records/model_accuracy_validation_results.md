# PP-OCR 浮点与 QAT 精度验证实施记录

本文件记录
[PP-OCR 浮点与 QAT 精度恢复执行计划](../plans/model_accuracy_validation_plan.md) 的实际命令、指标、
阶段结论和首个失败边界。所有 smoke、固定输入误差和完整验证集任务指标分开记录。

## 1. P0 实验合同

### 1.1 数据集

2026-08-03 对合同工具执行两次独立审计，合同 hash 和 det/rec JSON、debug sample 清单均字节一致。

| 任务 | Train | Val | 缺失/坏图 | 标签解析错误 |
| --- | ---: | ---: | ---: | ---: |
| ICDAR2015 det | 1000 | 500 | 0 | 0 |
| ICDAR2015 rec | 4468 | 2077 | 0 | 0 |

识别数据编码统计：

| Split | Loader 接受 | 严格无损编码 | 未知字符 |
| --- | ---: | ---: | --- |
| Train | 4468 | 4467 | 1 个 `TAB` |
| Val | 2077 | 2077 | 0 |

训练标签第 2917 行为 `train/word_2917.png\tre\tlles`。Paddle 和 route2 encoder 都会跳过字典外
`TAB`，loader 实际编码为 `relles`；因此该样本可训练但不是标签文本的无损编码。验证集不受影响。

### 1.2 合同指纹

合同通过 `artifacts/accuracy_contract/*.json` 追溯（哈希已按 AGENTS.md 规则移除，不再记录）。

产物：

```text
artifacts/accuracy_contract/common.json
artifacts/accuracy_contract/det.json
artifacts/accuracy_contract/rec.json
artifacts/accuracy_contract/det_debug_samples.txt
artifacts/accuracy_contract/rec_debug_samples.txt
```

P0 结论：通过。

## 2. 固定输入浮点转换复核

P0 后按本轮合同重新运行 Paddle/PyTorch 固定输入 parity；以下指标均来自本轮执行。

### 2.1 PP-OCRv5 mobile det

输入 `1x3x128x128`，BN eval：

```text
shape:          [1,1,128,128]
MAE:            1.0518916467e-12
p99:            8.2991391537e-12
max_abs:        3.7744030124e-11
relative MAE:   2.2330784698e-6
Paddle finite:  true
PyTorch finite: true
```

### 2.2 PP-OCRv5 mobile rec

输入 `1x3x48x320`，CTC pre-softmax logits：

```text
shape:                [1,40,18385]
logits MAE:           5.9062313085e-6
logits p99:           1.9073486328e-5
logits max_abs:       4.2915344238e-5
probability MAE:      1.8858725692e-10
probability max_abs:  2.5033950806e-6
argmax agreement:     1.0
Paddle finite:        true
PyTorch finite:       true
```

该结果只证明固定输入网络前向对齐，不替代完整验证集任务指标。

## 3. P1 Paddle 官方完整验证集浮点基线

使用 PaddleOCR 官方 v5 YAML、官方预训练权重和官方 eval transform，在 Paddle 3.0.0 / GPU 上执行。
det 使用动态 `DetResizeForTest`；rec 使用 `RecResizeImg [3,48,320]`。两项均独立运行两次。

### 3.1 PP-OCRv5 mobile det

| Run | Images | Precision | Recall | Hmean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 500 | 0.4664310954 | 0.3813192104 | 0.4196026490 |
| 2 | 500 | 0.4664310954 | 0.3813192104 | 0.4196026490 |

两次任务指标完全一致。速度只用于环境记录：`40.50` 和 `41.89` FPS，不作为精度门禁。

### 3.2 PP-OCRv5 mobile rec

| Run | Images | Accuracy | Norm edit distance |
| --- | ---: | ---: | ---: |
| 1 | 2077 | 0.5936446770 | 0.8168694512 |
| 2 | 2077 | 0.5936446770 | 0.8168694512 |

两次任务指标完全一致。速度为 `573.14` 和 `347.65` FPS，仅作环境记录。

P1 结论：通过。上述指标是当前 ICDAR2015 数据合同上的官方浮点绝对基线。

## 4. 指标隔离规则

P0 之前的 route2 训练、QAT 和 ONNX 指标仅保留在历史文档中，不进入本轮基线、delta、门禁或归因。
这些历史实验的数据 profile、DB loss 和评估 resize 与本轮合同不一致，因此不得与本轮 P1-P6 指标
直接比较。后续表格只记录按 P0 合同重新执行并能追溯到对应产物的结果。

## 5. P2 同输入 PyTorch 浮点转换验收

新增 `tools/compare_float_frameworks.py`。该工具只使用 PaddleOCR 官方 Eval dataloader 生成 normalized
input，三个模型依次接收同一 NumPy batch，并分别使用独立的 Paddle 官方 postprocess/metric 实例。
正式结果复用 P1 的 GPU、batch size 和 worker 数。JSON 包含命令、环境、Git 状态和输入标识（哈希按 AGENTS.md 规则不记录）。

### 5.1 PP-OCRv5 mobile det

GPU 0，batch 1，完整 500 张：

| 模型 | Map MAE | Map p99 | Map max | Precision | Recall | Hmean | Hmean delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paddle | 0 | 0 | 0 | 0.4664310954 | 0.3813192104 | 0.4196026490 | 0 |
| PytorchOCR | 1.3382878e-4 | 1.4053515e-7 | 0.9631467201 | 0.4584795322 | 0.3774675012 | 0.4140480591 | -0.0055545899 |
| route2 | 2.2533953e-6 | 2.6493603e-9 | 0.0402350724 | 0.4658823529 | 0.3813192104 | 0.4193804607 | -0.0002221883 |

route2 达到 map MAE `<= 1e-4` 和任务指标绝对差 `<= 0.005` 的门禁。行为层仍存在稀疏差异：

```text
box count agreement:             491 / 500 = 0.982
all boxes within 1 px:           481 / 500 = 0.962
first mismatch sample index:     37 (0-based)
worst tensor sample index:       479 (0-based)
worst sample map max_abs:        0.0402350724
```

索引 37 对应标签第 38 行 `ch4_test_images/img_376.jpg`；索引 479 对应第 480 行
`ch4_test_images/img_177.jpg`。PytorchOCR 的 hmean 差 `0.0055546`，未通过任务门禁；route2 的改动已显著
缩小差距，但框行为尚未完全一致。

### 5.2 PP-OCRv5 mobile rec

GPU 0，batch 128，完整 2077 张。PytorchOCR 与 route2 的 CTC 路径输出相同；PytorchOCR 旧版 GTC/NRTR
结构与 route2 checkpoint 不兼容，因此只对部署目标 CTC 路径执行完整参数检查，状态记录为
`strict_ctc_path_gtc_excluded`。

| 模型 | Probability MAE | Argmax agreement | Accuracy | Edit distance | Edit delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paddle | 0 | 1 | 0.5936446770 | 0.8168694512 | 0 |
| PytorchOCR | 1.2940247e-8 | 0.9998916707 | 0.5936446770 | 0.8171205001 | +0.0002510489 |
| route2 | 1.2940247e-8 | 0.9998916707 | 0.5936446770 | 0.8171205001 | +0.0002510489 |

概率 MAE `<= 1e-7`、accuracy/edit distance 差 `<= 0.001` 均通过；按 2026-08-04 确认的
argmax agreement `>= 0.9998` 门禁，当前 `0.9998916707` 通过。解码文本一致
`2073 / 2077 = 0.9980741454`：

```text
first text mismatch sample index: 828 (0-based), test/word_829.png, label FIRE
worst tensor sample index:        1955 (0-based), test/word_1956.png, label NOT
raw logits MAE / max_abs:         0.0040959616 / 1.4892654419
centered logits MAE / max_abs:    0.0016339023 / 0.9388599396
probability max_abs:              0.1795483753
```

### 5.3 P2 结论

- det route2 的 L1 MAE 和 L3 metric 通过，但 L2 框行为未完全一致；
- rec 的 L1 probability MAE、L2 argmax agreement 和 L3 metric 均通过；4 张文本差异保留为诊断样本；
- det 的 L2 框行为仍有稀疏差异，P3 继续定位，但不再由 rec eager float 门禁阻塞；
- reparameterized graph 与 eager/Paddle 的差异单独报告，不覆盖 P2 eager 结论。

正式产物：

```text
artifacts/accuracy_baseline/p2_float/det_frameworks_gpu0.json
artifacts/accuracy_baseline/p2_float/rec_frameworks_gpu0.json
```

`diagnostic_rec_batch32_gpu3.json` 明确为非合同 diagnostic，不进入本轮 baseline、delta 或门禁。

## 6. P3 首个浮点差异定位

P3 未修改模型、权重或 QAT 配置，只增加层级诊断工具并对 P2 的首个/最坏样本复核。

### 6.1 识别层级结果

GPU 0、batch 128，样本 `828` 和 `1955`：

| 边界 | sample 828 MAE | sample 1955 MAE | 结论 |
| --- | ---: | ---: | --- |
| `blocks5` 输出 | 8.0771e-5 | 6.6198e-5 | 上游已有小量差异 |
| `blocks6.0` 输出 | 2.0285e-4 | 1.2289e-4 | 渐进增加 |
| `blocks6.1` 输出 | 1.2976e-4 | 8.1947e-5 | 未发生突变 |
| `blocks6.2` 输出 | 1.1468e-4 | 9.1999e-5 | 未发生突变 |
| `blocks6.3.dw_conv` 输出 | 4.0758e-5 | 3.9036e-5 | depthwise 实现对齐 |
| `blocks6.3.pw_conv` 输出 | 6.8541e-3 | 5.6953e-3 | 首个明显放大点 |
| CTC encoder 输出 | 1.8174e-3 | 5.1515e-3 | 继续传播/放大 |
| CTC logits | 5.1815e-3 | 2.8422e-2 | 影响最终 logits |

`blocks6.3.pw_conv` 的 BN running variance 最小约 `4.8e-10`，同时存在 `LAB scale=2.0455` 和
激活后 `LAB scale=3.2344`，对上游误差高度敏感。该层包括 pointwise 的 4 个 Conv-BN 分支、identity
分支、LAB 和 Hardswish。

### 6.2 交叉输入验证

将 Paddle 的 `blocks6.3.dw_conv` 输出直接输入 PyTorch pointwise，将 PyTorch 的同一输出直接输入
Paddle pointwise，GPU 0 结果为：

```text
Paddle input -> PyTorch pointwise: MAE 1.1240572e-7, max_abs 3.0517578e-5
PyTorch input -> Paddle pointwise: MAE 1.1030002e-7, max_abs 3.0517578e-5
native depthwise input delta:      MAE 3.9035785e-5, max_abs 3.7183762e-3
```

因此 pointwise 层自身的权重、BN、LAB、激活公式已对齐；原生执行中的小量上游差异经过敏感
pointwise 分支放大，不能通过修改 pointwise 公式或导出图来掩盖。

CPU batch 128 的 sample 1955 也出现同一结构，但数值幅度不同：

```text
blocks6.3.dw_conv MAE: 7.4504164e-7
blocks6.3.pw_conv MAE: 8.1968088e-5
ctc_logits MAE:        1.6926078e-4
```

关闭 PyTorch TF32 后，sample 1955 的 GPU backbone MAE 反而由 `0.0035324` 增至 `0.1027498`，
所以当前不能将差异归因于 TF32；默认 GPU backend 作为本轮复现环境保留。

### 6.3 P3 结论与下一步

- route2 的检测/识别 float 结构与 Paddle 权重映射没有发现独立错误；
- 主要问题是 PP-OCRv5 recognition backbone 的 GPU 数值敏感性，首个显著放大点为
  `blocks6.3.pw_conv`；
- 不调整 observer、fake quant、qspec 或 QAT 训练参数；
- 下一步执行未重参数化 eager 与 deploy reparameterized A/B，验证是否能降低累积误差，并单独报告
  reparameterization 前后任务指标；
- P4 fake-quant-off 仍保持阻塞，直到 float A/B 的差异边界被记录并接受。

诊断产物：

```text
artifacts/accuracy_baseline/p3_diagnostics/rec_float_layers_gpu0_batch128.json
artifacts/accuracy_baseline/p3_diagnostics/rec_float_layers_gpu0_batch128_tf32off.json
artifacts/accuracy_baseline/p3_diagnostics/rec_float_stages_gpu0_batch128.json
artifacts/accuracy_baseline/p3_diagnostics/rec_float_blocks6_gpu0_batch128.json
artifacts/accuracy_baseline/p3_diagnostics/rec_float_block63_gpu0_batch128.json
artifacts/accuracy_baseline/p3_diagnostics/rec_float_pointwise_crossfeed_gpu0_batch128.json
artifacts/accuracy_baseline/p3_diagnostics/rec_float_pointwise_crossfeed_cpu_batch128.json
```

### 6.4 重参数化 A/B

使用相同 P2 合同、GPU 0、batch 128，仅对 route2 执行 `reparameterize_for_deploy`，Paddle 和
PytorchOCR reference 保持不变。结果如下：

```text
route2 eager:
  accuracy:          0.59364467697
  norm_edit_dis:     0.81712050008
  argmax agreement:  0.99989167068
  raw logits MAE:    0.00409596157

route2 reparameterized:
  accuracy:          0.59412614062
  norm_edit_dis:     0.81719424808
  argmax agreement:  0.99939817044
  raw logits MAE:    0.10335824664
```

重参数化后 tensor/argmax 明显恶化，不能作为当前修复；该结果单独保存在
`artifacts/accuracy_baseline/p3_reparameterized/rec_frameworks_gpu0.json`，不覆盖 eager P2
结果。下一步先做 route2 eager 与 reparameterized 自身的 float preservation，对融合实现建立独立门禁。

route2 自身保持性 smoke（rec，GPU 0，batch 2，2 张图）结果：

```text
eager vs reparameterized MAE: 0.0993857140
max_abs:                      1.8693885803
p99:                          1.0471965790
worst sample index:           1
```

该误差远超 float preservation 门禁，证明 reparameterized graph 不能替代 Paddle/route2 eager
浮点参考。2026-08-04 决策为优先保证 Axera PT2E/QuantONNX 结构，因此后续仍以该 deploy graph
作为 QAT 结构基线，并通过重新训练后的完整任务指标判断是否能够补偿融合误差。产物：

```text
artifacts/accuracy_baseline/p3_reparameterized/rec_eager_vs_deploy_gpu0.json
```

### 6.5 重参数化策略与检测对照

检查确认 `LearnableRepLayer._get_kernel_bias()` 当前每个 Conv 分支只累加一次；identity kernel
已经按 BN 参数的 device/dtype 创建。没有继续修改融合公式来掩盖 GPU 数值差异。

保留以下实现和验证：

- `configs/qat/training/ppocrv6_small_rec_baseline.yml` 保持 `training.reparameterize: true`；
- 新增 `tests/test_rec_reparameterization.py`，覆盖 Conv-BN CPU 等价性、分支单次累加、幂等调用和
  identity device/dtype；
- 修正 `tools/compare_reparameterization.py`，使其兼容 detection wrapper 的 tensor 输出和旧 dict 输出。

局部测试：`13 passed`。

检测使用本轮 PP-OCRv5 mobile det 浮点权重，在 ICDAR2015 合同的 500 张验证图上执行 CPU
eager/deploy 保持性对照。结果与识别单独存放，不改变 P2 框架对齐基线：

```text
artifact: artifacts/accuracy_baseline/p3_reparameterized/det_eager_vs_deploy_cpu.json
samples:  500
MAE:      8.4945070e-09
max_abs:  2.7680397e-04
p99:      8.2707174e-12
worst:    sample index 498
```

CPU 检测结果的 MAE 很小，但 `max_abs` 高于严格单样本阈值；后续仍补充 GPU maps/框/任务指标
对照。该风险不阻塞 deploy graph 的 PT2E/ONNX 结构验收，但不能把 CPU 结果作为部署端浮点等价证明。

### 6.6 eager Conv-BN 的 PT2E/ONNX 结构检查

PP-OCRv6 small rec 使用真实浮点权重执行 `--no-reparameterize` 全链路 smoke。`export_for_training`、
`prepare_qat_pt2e`、反向和 `convert_pt2e` 均可执行，但最终 QuantONNX 不满足 Axera 结构门禁：

```text
BatchNormalization:       13
unquantized Conv outputs:  2 (node_Conv_103, node_Conv_119)
validate_qdq_graph:        FAIL
```

残留节点包括 `backbone.conv1.stem2a/stem2b` 的 2 个 Conv-BN，以及 11 个
`token_mixer.rep_dw.bn`。因此不固定使用 eager Conv-BN；Axera 基线恢复 reparameterized graph，
优先要求 ONNX 无 BN、Conv QDQ 完整和共享域结构正确。QAT 只能作为融合误差的潜在补偿手段，
是否有效以重新训练后的 accuracy/edit distance 为准，不能预先视为已恢复。

### 6.7 reparameterized PT2E/QuantONNX 结构 smoke

同一 PP-OCRv6 small rec 权重使用 `reparameterize: true` 重新执行完整 smoke，结构验收通过：

```text
nodes:                     682
BatchNormalization:        0
QuantizeLinear:            161
DequantizeLinear:          280
redundant DQ -> Q:          0
requantize DQ -> Q:         2
Conv -> QDQ -> BN:          0
unquantized Conv outputs:   0
Concat quantized/shared:    1 / 1
HardSigmoid quantized:      5 / 5
validate_qdq_graph:         PASS
ORT output shape:           [1, 40, 18710]
ORT optimize-off finite:    true
ORT optimize-on finite:     true
ORT optimize on/off MAE:    0.4238455892
ORT optimize on/off max:    3.7355744839
```

ORT optimize-on 仍会显著改变输出，不能用 QAT 补偿该运行时图改写；QuantONNX 语义和后续任务指标
继续以 `ORT_DISABLE_ALL` 为基准。产物：

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv6_small_rec_reparameterized_qat_smoke_20260804.onnx
artifacts/accuracy_baseline/p4_structure/ppocrv6_small_rec_reparameterized_qat_smoke_20260804.json
```

### 6.8 zero-point Cast 批量折叠与 det/rec 重导出

`export_qat_onnx()` 增加通用的 constant zero-point Cast 折叠。处理条件为：

- Cast 输入必须是整数 initializer；
- 所有消费者必须是 Q/DQ，且只作为 zero-point（input index 2）；
- 目标 dtype 必须能精确表示全部常量值；
- 不处理激活 Cast、shape Cast、混合消费者或溢出转换。

单元测试覆盖一次折叠多个 INT8/U8 zero-point Cast，以及普通 Cast 和 INT8 溢出的保留路径。
相关回归 `22 passed`。

PP-OCRv6 small rec 重导出后，原 CTC FC 权重 zero-point 的唯一 Cast 已删除；新旧模型在 ORT
optimize-off 下逐值一致：MAE/max 均为 `0`。PP-OCRv6 small det 使用同一导出函数生成固定 640
结构 smoke。最终统计：

| Task | Nodes | Cast | BN | Q / DQ | Conv missing QDQ | Concat shared | Hard activation QDQ | Validator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rec | 681 | 0 | 0 | 161 / 280 | 0 | 1 / 1 | 5 / 5 | PASS |
| det | 761 | 0 | 0 | 177 / 335 | 0 | 2 / 2 | 13 / 13 | PASS |

ORT optimize-off 输出均 finite。optimize-on/off 差异继续只作诊断：rec MAE/max 为
`0.4238455892 / 3.7355744839`，det 为 `0.0003736290 / 0.0078426953`。

```text
rec: /home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv6_small_rec_reparameterized_qat_smoke_20260804_castfold.onnx
det: /home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv6_small_det_reparameterized_qat_smoke_20260804_castfold.onnx
report: artifacts/accuracy_baseline/p4_structure/ppocrv6_castfold_exports_20260804.json
```

## 7. 当前状态

| 阶段 | 状态 | 结论 |
| --- | --- | --- |
| P0 合同 | 通过 | 双跑 hash 和产物一致 |
| P1 Paddle float | 通过 | det/rec 完整验证集双跑一致 |
| P2 PyTorch float | 部分通过 | rec 按 argmax `>=0.9998` 通过；det 稀疏框行为差异继续跟踪 |
| P3 训练评估语义 | 已完成（2026-08-07） | 重参数化 A/B、eager Conv-BN 结构检查、reparam PT2E smoke 均已执行并给出结论；融合误差已记录并接受为 QAT 恢复风险，不采用 eager Conv-BN deploy 图 |
| P4 fake-quant-off | rec 已验证 | v5 rec 任务指标 delta 通过；严格 tensor 容差因 PT2E Conv-BN QAT 重写未通过，det 待验证 |

## 8. Pad 量化规则变更后的 det smoke

### 8.1 背景和 checkpoint 处置

检测模型此前在 Axera 转换阶段出现精度崩溃，定位到 `Pad` 量化规则。用户已修改：

```text
pytorchocr/quantization/ax_quantizer_utils.py
```

该修改会改变 PT2E observer/fake-quant 图。历史 det QAT checkpoint 的非 observer 参数虽然仍可
对齐，但 observer 状态数量和命名图已发生变化，严格恢复失败；因此旧
`runs/icdar2015_ppocrv6_small_det_qat/best.pt` 不再用于导出或继续训练。本轮 smoke 从真实
浮点预训练权重重新构建，不使用 `strict=False`。

### 8.2 Smoke 命令和产物

```bash
env PYTHONPATH="$PWD" \
/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/qat_smoke.py \
  --task det \
  --det-graph inference \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights weights/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth \
  --qat-config configs/qat/ppocrv6_small_det_u8s8.json \
  --output /home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv6_small_det_qat_smoke_after_utils_20260804.onnx \
  --image-shape 3 640 640 \
  --reparameterize \
  --onnx-optimize
```

产物：

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv6_small_det_qat_smoke_after_utils_20260804.onnx
artifacts/accuracy_baseline/p4_structure/ppocrv6_small_det_smoke_after_utils_20260804.json
```

该文件是结构 smoke，不是完整数据集精度模型，也不是正式 QAT 权重。

### 8.3 PT2E 和 QuantONNX 结果

```text
PT2E backward:             PASS
float nodes:               374
converted nodes:           1242
gradient tensors:          165
loss:                      0.0161454752
converted/pre-export MAE:  0

ONNX nodes:                763
QuantizeLinear:            178
DequantizeLinear:          336
Cast:                      0
BatchNormalization:        0
Conv missing QDQ:          0
direct DQ -> Q:            0
redundant DQ -> Q:         0
Concat shared domains:     2 / 2
Hard activation QDQ:       13 / 13
zero-point dtype:          uint8
output shape:              [1, 1, 640, 640]
ORT optimize-off output:   finite
PyTorch -> ORT MAE:        0.0001125386479543522
PyTorch -> ORT max_abs:    0.003921381197869778
```

本次 smoke 中 `Pad` 已形成完整共享量化域：

```text
DQ -> Pad -> Q
input scale / output scale: 0.1031050831079483 / 0.1031050831079483
input zero-point / output zero-point: U8(0) / U8(0)
```

因此新规则已进入导出图，且没有产生 Pad 前后量化域断裂。`tools/verify_qat_onnx.py -c` 当前仍
报告 143 条错误，全部来自 GELU/Erf 分解链及其常量输入的通用规则检查，不是 Pad 节点；该检查器
结果单独保留为待处理的规则兼容性问题，不能将本 smoke 宣称为 Axera 全规则验收通过。

### 8.4 阶段结论

- [x] 新量化规则下 det PT2E smoke 可反向、可 convert、可导出 QuantONNX；
- [x] Pad 输入输出共享 U8 量化域；
- [x] 当前结构统计满足本项目基础 ONNX 门禁；
- [ ] 用户结构分析确认；
- [ ] 重新启动 det 正式 QAT 训练；
- [ ] 完整数据集精度和 Axera 转换精度验证。

后续所有 det/rec 模型均遵循同一顺序：先用当前规则和真实浮点权重 smoke 导出 QuantONNX，交由
用户分析确认，再开始正式训练。任何量化规则修改都使此前 QAT checkpoint 失效，必须重新 smoke
并从匹配的浮点权重重新开始。

## 9. v6 det/rec Axera 适配暂停

2026-08-04 更新：PP-OCRv6 small det 在 Axera 工具链上仍存在 `Pad` 相关问题；随后确认
PP-OCRv6 small rec 的 QuantONNX 同样包含一个 `Pad`：

```text
rec Pad: node_Pad_263
pattern: DQ -> Pad -> Q
```

因此当前状态调整为：

- PP-OCRv6 small det：暂停 Axera 适配和正式 QAT 重训；
- PP-OCRv6 small rec：暂停 Axera 适配和后续训练；
- 不使用历史 det/rec QAT checkpoint 推进部署验收；
- 不因 ONNX checker、ORT 或项目基础 QDQ validator 通过而视为 Axera 已支持；
- 恢复前先构造最小 `Pad` QuantONNX，在 Axera 工具链验证转换和运行语义。

## 10. PP-OCRv5 mobile rec 候选 smoke

由于 v6 det/rec 和 v5 mobile det 均受 `Pad` 工具链问题阻塞，先检查 PP-OCRv5 mobile rec。其
当前真实预训练权重 smoke 不包含 `Pad`，因此登记为下一候选；后续该模型已在 U16/S16 合同下
执行 Exp1-Exp4（含 KD），结果见 icdar2015_ppocrv5_mobile_rec_qat_training.md。

### 10.1 历史全局配置产物（已被替代）

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv5_mobile_rec_qat_smoke_latest_20260804.onnx
```

使用当前模型、量化配置、vendor quantizer 和导出器，从
`ptocr_v5_mobile_rec.pth` 重新构建，没有复用旧 v5 rec smoke checkpoint。

### 10.2 结构结果

```text
input shape:              [1, 3, 48, 320]
output shape:             [1, 40, 18385]
ONNX nodes:               961
QuantizeLinear:           252
DequantizeLinear:         438
Cast:                     0
BatchNormalization:       0
Pad:                      0
unquantized Conv outputs: 0
Concat shared domains:    1 / 1
HardSigmoid QDQ:          2 / 2
HardSwish nodes:          28
requantize Identity:      3
ORT optimize-off:         finite
```

导出过程折叠 1 个 constant zero-point Cast，删除 8 组完全等价冗余 DQ/Q，并保留 3 个非同 qparam
的 `DQ -> Identity -> Q` requantize 边界。该 3 个 Identity 是导出图中的显式边界，不是冗余节点。

### 10.3 当前门禁

- [x] 真实权重 PT2E backward、convert 和 QuantONNX 导出；
- [x] ONNX 无 `Pad`、无 BN、Conv QDQ 完整；
- [x] ORT optimize-off 可执行；
- [ ] 用户结构分析确认；
- [ ] 创建/启动正式 QAT 训练；
- [ ] 完整验证集、Axera 转换和 AXModel 验证。

PP-OCRv5 mobile det 的已有 QuantONNX 仍包含 1 个 `Pad`，不作为当前候选；v5 server det/rec
当前没有对应的真实 PyTorch 权重，也不进入本轮。

### 10.4 局部 S8 Attention 最新产物

Attention 内部 2 个 requant 已通过 QAT qspec 修正，不通过导出图改写消除。当前产物：

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv5_mobile_rec_attn_s8_native_silu_smoke_20260804.onnx
```

```text
nodes / Q / DQ:          941 / 243 / 429
activation domains:      S8 20 / U8 223
Attention:               2 / 2
first MatMul:            S8/S8 -> S8，2 / 2
second MatMul:           S8/S8 -> U8，2 / 2
Pad / BN / unquant Conv: 0 / 0 / 0
SiLU boundary/internal:  7 / 0
necessary requant:       1（Attention 外 Pool/Conv U8 -> U8）
```

最后一个 requant 保留在 PT2E 图；此前 Pool/Conv observer 共享优化已撤销。项目新增两个 skill：

```text
.codex/skills/ppocr-qat-config-discovery/
.codex/skills/ppocrv5-rec-pulsar2-config/
```

前者从当前 FX 图生成/检查 QAT regional 配置；后者从精确 QuantONNX 生成/校验 Pulsar2 配置。
Pulsar2 smoke 配置和审计报告位于 `artifacts/pulsar2/`。当前仍等待用户结构确认，不启动正式训练。

Swish 已改为原生 `F.silu`，FX 图保持 `aten.silu`，ONNX 中 7 个 `Sigmoid + Mul` 均无内部 QDQ。
`tmp/frontend/optimized.onnx` 仅将 9 个现有 Linear 的 `Transpose + MatMul + Add` 降低为
Pulsar2 `FullyConnected`；该目标端优化不前移到 Paddle→PyTorch，以保留 PT2E Linear QAT 标注。
对比证据保存在
`artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_frontend_optimization_20260805.json`。

### 10.5 v5 rec 原生 SiLU 浮点和 fake-quant-off

当前原生 SiLU 普通未重参数化浮点模型已在完整 2077 张 ICDAR2015 rec 验证集重测：

```text
route2 accuracy:             0.5936446769684897
route2 norm_edit_dis:        0.8171205000785132
Paddle/route2 argmax:        0.9998916706788638
Paddle/route2 probability MAE: 1.2945054054636001e-8
artifact: artifacts/accuracy_baseline/p2_float/rec_frameworks_native_silu_gpu2_20260805.json
```

QAT 捕获使用的重参数化 eager/exported 图及 prepared observer/fake-quant-off 完整结果：

```text
eager/exported accuracy:     0.5941261434761675 / 0.5941261434761675
prepared-off accuracy:       0.5936446798266731
accuracy delta:              -0.0004814636494944
edit similarity delta:       -0.0000687805213563
argmax / CTC sequence:       0.9999759268175252 / 0.9990370727010110
exported/prepared logits:    MAE 0.003476011266185565, max 0.4623374938964844
exported/prepared probability: MAE 7.888990221661101e-9, max 0.06171932816505432
```

382 个 observer/fake-quant 模块均已从 enabled 切换到 disabled；236 个共同 state key 完全相等。
差异定位为 PT2E 对 5 个 Conv-BN 的标准 QAT arithmetic rewrite。任务指标 delta 通过 `<= 0.001`，
但严格 tensor 容差未通过；不采用 eager Conv-BN 固定融合，后续将该差异作为正常 PT2E 图的已知
数值风险继续跟踪。报告：

```text
artifacts/accuracy_baseline/p4_structure/ppocrv5_mobile_rec_pt2e_fake_off_full_gpu2_20260805.json
```

### 10.6 暂停的 v6 rec 诊断产物

最新 v6 rec smoke 诊断产物保留，不继续进入正式训练：

```text
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports/ppocrv6_small_rec_qat_smoke_latest_20260804.onnx
```

该文件包含当前导出器的 constant zero-point Cast 折叠，以及 2 个非同 qparam 的
`DQ -> Identity -> Q` requantize 边界，仅供模型结构和工具链问题分析。

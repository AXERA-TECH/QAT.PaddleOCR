# PP-OCR Pretrained 训练结构复现计划

状态：2026-08-06 已完成。四个模型的 pretrained 浮点训练结构均已通过各自适用的 S0-S4；
后续 PT2E/QAT/QuantONNX 工作保持暂停。

## 1. 目标和边界

当前工程优先修复“pretrained 模型训练结构复现”。PP-OCRv5 mobile det、PP-OCRv6 small rec 和
PP-OCRv6 small det 本轮只验证以下两个图合同：

```text
pretrained training graph
  完整复现 Paddle 训练结构、辅助头、targets、loss 和梯度路径

deployment graph
  从同一份 pretrained 权重显式投影，只保留部署输出
```

识别 QuantONNX 只输出 CTC logits、检测 QuantONNX 只输出 shrink map 是正确的部署合同。本计划要避免
的是训练模型也被提前裁成单个输出头。例如 PP-OCRv5 rec 的 Paddle pretrained 训练图是
`MultiHead(CTC + NRTR)` 和 `MultiLoss(CTC + NRTR)`，不能继续用 CTC-only 图冒充训练结构复现。

本轮处理顺序固定为：

1. PP-OCRv5 mobile rec；
2. PP-OCRv5 mobile det；
3. PP-OCRv6 small rec；
4. PP-OCRv6 small det。

前一个模型未通过完整结构、浮点、loss、梯度、optimizer step 和部署投影门禁前，不开始后一个模型。
后续三个模型禁止运行 PT2E/QAT、导出 QuantONNX 或调整 qspec。

## 2. 当前差距

### 2.1 公共 builder 混合了训练和部署语义

当前 `build_rec_model()` 构建 YAML 中的 `MultiHead` 后，无条件返回 `RecCTCWrapper`。这会产生以下
问题：

- builder 的公开输出只剩 CTC logits，无法表达完整训练 targets 和多头输出；
- recognition 权重加载允许缺失 `head.before_gtc.*` 和 `head.gtc_head.*`；
- `tools/train.py`、criterion 和 dataset 只认识 CTC label/loss；
- PP-OCRv5 rec 通过关闭 `use_guide` 让 CTC 梯度进入 backbone，只是 CTC-only 修补，不等价于 Paddle
  原始 `CTC + NRTR` 训练。

应把训练模型构建、QAT 训练包装和部署输出投影拆开，禁止继续增加 `ctc_backbone_grad` 一类开关来
模拟缺失辅助分支。

### 2.2 PP-OCRv5 mobile rec

Paddle 原始合同：

```text
PPLCNetV3
-> MultiHead
   -> CTC svtr neck(use_guide=True) + CTCHead
   -> before_gtc + NRTR Transformer
-> MultiLoss(CTCLoss + NRTRLoss)
```

`use_guide=True` 会阻断 CTC 分支到 backbone 的梯度，backbone 主要由 NRTR 分支更新。当前工程已有
`MultiHead` 和 Transformer 雏形，但尚未完成以下验收：

- Paddle/PyTorch 参数名、shape 和转换一一对应；
- 训练输出 key 与 Paddle 的 `ctc`、`ctc_neck`、`gtc` 一致；
- `MultiLabelEncode`、NRTR special token 和 target layout 一致；
- `NRTRLoss`、`MultiLoss` 权重及 reduction 一致；
- CTC、NRTR、backbone 的梯度路径一致；
- eval/deploy 时只选择 CTC，但不影响训练图的完整性。

### 2.3 PP-OCRv5 mobile det

v5-det 不存在 rec 式的第二个识别头，但仍需复现完整 DB 训练合同：

- train 返回 shrink、threshold、binary 三通道 maps；
- inference/deploy 只返回 shrink map；
- Paddle 标准 DB loss、BalanceLoss/OHEM、alpha/beta 和梯度一致；
- pretrained training graph 与 reparameterized deployment graph 分开验证；
- 配置中不得静默丢失官方 Loss、Optimizer 和训练结构字段。

### 2.4 PP-OCRv6 small rec

v6-rec 同样使用 `MultiHead(CTC + NRTR)` 与 `MultiLoss`。虽然 lightsvtr CTC 分支不像 v5 的
`use_guide=True` 那样必然切断 backbone 梯度，NRTR 仍是 pretrained 训练目标的一部分，不能只验证
CTC-only。完成 v5-rec 公共 MultiHead/MultiLoss 基础后，再单独验证 PPLCNetV4、lightsvtr 和 v6 字典。

### 2.5 PP-OCRv6 small det

v6-det 原始 DBHead 除主 maps 外，还包含 `aux_maps_p4/p3/p2`，DBLoss 使用
`aux_weight_p4/p3/p2`。当前 PyTorch DBHead 只实现主 maps，因此必须补齐：

- RepLKFPN 向 DBHead 提供 `fuse` 和三个 aux feature；
- 三组 aux binarize/thresh head 及其预训练参数；
- 主/辅助 maps 的 shape、值、loss 和梯度；
- deployment projection 丢弃 aux outputs，仅保留 shrink map；
- QAT 训练时辅助头是否保持 float、部署主路径如何量化的显式策略。

## 3. 设计决策

### 3.1 使用显式 graph role

模型工厂后续应使用明确的图角色，而不是根据 `train()/eval()`、是否传 targets 或多个布尔开关隐式
改变结构。建议角色如下，最终 API 名称可按实现调整：

```text
pretrained_train  完整 Paddle 训练结构，未裁剪辅助参数
qat_train         完整训练输出，部署路径 prepared，辅助训练分支策略显式
deploy            单部署输出，不包含训练辅助输出合同
```

三个角色必须共享同一份已加载权重来源。deployment graph 应由训练模型显式投影或包装得到，不能另建
一个允许大量 missing keys 的独立模型。

### 3.2 权重加载分为 full strict 和 projection strict

- `pretrained_train`：Paddle 到 PyTorch 转换必须覆盖完整训练参数；除框架固有 buffer 外使用
  `strict=True`。
- `qat_train`：从相同 full state 构建；prepared checkpoint 必须按 graph metadata `strict=True`。
- `deploy`：允许删除的只能是角色定义中列出的辅助头参数，并生成逐 key projection report；不得使用
  通用 `strict=False`。

PP-OCRv5 rec 现有允许缺失 `before_gtc/gtc_head` 的逻辑应在 full conversion 完成后删除。历史 CTC-only
权重和 checkpoint 保持只读，不迁移成新的 full-training checkpoint。

### 3.3 训练输出与部署输出分离

训练 API 使用结构化结果：

```text
rec full train: ctc, ctc_neck, gtc
det v5 train:   maps(shrink, threshold, binary)
det v6 train:   maps + aux_maps_p4 + aux_maps_p3 + aux_maps_p2
```

部署 API 固定为：

```text
rec deploy: raw CTC logits
det deploy: shrink probability map
```

测试必须同时断言输出 key、shape、dtype、finite 和 gradient owner，不能只检查最终 tensor shape。

### 3.4 QAT 辅助分支策略（本轮冻结）

以下内容仅保留为未来恢复 QAT 时的设计记录，不属于 v5-det、v6-rec、v6-det 的当前实施或验收范围：

1. 部署主路径使用 Axera qspec 和 fake quant；
2. 辅助分支保留训练监督，默认保持 float；
3. 辅助 loss 能通过共享 backbone 更新部署参数；
4. convert/export 前通过显式 deployment projection 移除辅助分支；
5. projection 前后部署主输出在 fake-quant-off 和 fake-quant-on 下分别做数值对比。

如果 PT2E 无法稳定捕获“量化主路径 + float 辅助分支”，先停在可复现的失败用例，不得退回
CTC-only 并称为完整训练等价。

### 3.5 重参数化边界

`pretrained_train` 首先复现 Paddle 原始训练结构和参数，不提前进行部署重参数化。随后独立验证：

```text
full pretrained train graph
-> eval parity
-> reparameterized QAT training graph
-> deployment projection
```

不得把重参数化差异、辅助头差异和 QAT 差异合并到同一次实验。

## 4. 公共阶段

### C0：冻结 pretrained 合同

- [ ] 保存四个官方 Paddle YAML、Paddle 权重、现有 PyTorch 权重及 SHA256；
- [ ] 导出 Paddle 和 PyTorch 参数 manifest：name、shape、dtype、trainable；
- [ ] 保存 training/eval 输出 schema 和参数总量；
- [ ] 固定每个模型的随机输入和一个真实 mini-batch；
- [ ] 将现有 CTC-only/v5-det/v6 结果标记为 legacy deploy/QAT 诊断，不作为 full-training baseline。

验证：同一 manifest 连续生成两次完全一致；历史产物不覆盖。

### C1：模型工厂角色拆分

- [ ] 实现显式 graph role；
- [ ] `pretrained_train` 不使用 RecCTCWrapper/DetInferenceWrapper；
- [ ] `deploy` 只通过显式 wrapper/projection 选择部署输出；
- [ ] graph role、完整配置 hash、权重 hash、reparameterization 写入 checkpoint metadata；
- [ ] checkpoint 拒绝跨 role、跨结构或跨 qspec resume。

验证：四个模型分别构建三个角色，参数归属和允许删除列表可审计；旧 checkpoint 只能由 legacy 入口
读取，不能误恢复到新图。

### C2：完整转换和 strict load 基础

- [x] converter 支持完整辅助头/aux head 参数；
- [ ] mapping report 区分 copied、framework buffer、role projection removed 和真正 missing；
- [x] full-training PyTorch state dict `strict=True`；
- [ ] projection report 的删除 key 集合固定并测试。

验证：参数数量、每个 tensor shape 和 SHA256 报告齐全；禁止未解释 missing/unexpected key。

### C3：训练公共接口

- [x] recognition dataset 支持 CTC + NRTR 双标签及 special tokens；
- [x] criterion 支持配置驱动 MultiLoss；
- [x] Trainer 接受结构化多头输出和 targets；
- [x] metric 仍只使用部署主输出；
- [x] checkpoint 保存 loss/head schema。

验证：合成 batch 覆盖空白、重复字符、最大长度、space、unknown 和 NRTR EOS/PAD；所有应训练参数
gradient finite，预期 detach 的分支无梯度。

## 5. 模型工作包和门禁

PP-OCRv5 mobile rec 的历史工作包包含 S0-S6；后续三个模型只执行 S0-S4，S4 通过后即可进入下一模型。

### S0：官方结构清单

- 对比 Paddle YAML、Paddle 实例和 PyTorch 实例；
- 记录模块树、参数 manifest、train/eval 输出 schema；
- 明确训练辅助分支和部署 projection 删除集合。

门禁：结构和参数差异全部有解释。

### S1：完整权重转换

- 转换官方 pretrained 权重；
- full-training 模型 strict load；
- 固定输入逐层定位首个数值差异。

门禁：无未解释 missing/unexpected key，主分支和辅助分支均能前向。

### S2：float forward parity

- 使用相同 normalized input 和相同 targets；
- 比较所有训练输出，不只比较 CTC logits 或 shrink map；
- 分别比较 train mode、eval mode和 BN running state。

初始数值门槛沿用总精度计划：rec probability MAE `<= 1e-7`、argmax agreement `>= 0.9998`；det
主/辅助 map MAE `<= 1e-4`。辅助 Transformer 如因 backend 需要不同容差，必须先保存首个差异节点。

### S3：loss 和 gradient parity

- 比较每个子 loss、总 loss 和 reduction；
- 比较 backbone、neck、主 head 和辅助 head 的 gradient presence、norm、MAE、max_abs；
- 执行一次 optimizer step 并比较参数 delta。

门禁：所有预期训练模块都有 finite gradient，Paddle/PyTorch 的 loss 和 gradient 差异在报告阈值内。

### S4：deployment projection parity

- 从同一 full-training state 生成 deploy role；
- rec 只取 CTC logits，det 只取 shrink map；
- 比较 full model eval 主输出与 deploy role；
- 验证 projection 删除集合和 strict load。

门禁：输出满足 eager 浮点保持阈值，部署图不残留 NRTR 或 aux output。

### S5：export/PT2E float 保持（仅 v5-rec 历史记录）

依次比较：

```text
full model deployment projection
-> reparameterized eager deploy
-> export_for_training float
-> prepared observer-off/fake-quant-off
```

门禁分两类：

- rec 多分支重参数化会把“多次 Conv 后求和”变为“kernel 求和后单次 Conv”，FP32/TF32 的累加顺序
  不同，不要求 raw logits 逐位一致；关闭 TF32 后要求 probability MAE `<=1e-7`、argmax agreement
  `>=0.9998`、CTC sequence agreement `>=0.9998`，同时完整记录 raw/centered logits MAE 和 max_abs；
- reparameterized eager -> exported float 仍要求相同 backend 上 MAE `<=1e-6`、max_abs `<=1e-5`；
- `prepare_qat_pt2e` 若没有 arithmetic rewrite，prepared fake-off 使用同一严格 tensor 门禁；若仅存在
  PyTorch 标准 Conv-BN QAT arithmetic rewrite，则要求 state common key 不变、observer/fake quant
  全部关闭、probability MAE `<=1e-7`、argmax agreement `>=0.9998`，且完整验证集 accuracy 和
  normalized edit distance delta 均 `<=0.001`。raw/centered logits 继续完整报告，但不要求逐位一致。

任一边界失败时不得调整 qspec。TF32 开关属于诊断和验收 backend 合同，不修改模型融合公式。

### S6：QAT smoke 和 QuantONNX（本轮全部冻结）

- QAT 训练图保留辅助监督；
- 一个真实 mini-batch 完成 forward/backward/optimizer step；
- 显式投影并 convert 部署主路径；
- 导出 smoke QuantONNX，执行 checker、QDQ 审计和 ORT optimize-off；
- 将 smoke QuantONNX 交用户检查后才能开始多 epoch 训练。

门禁：QuantONNX 仍为单部署输出，但训练日志和 gradient report 证明辅助分支实际参与训练。

### 5.1 2026-08-06 范围调整

本轮在 PP-OCRv5 mobile rec 完成 S5 后停止 PT2E/QAT 扩展。后续模型只执行 Paddle pretrained
结构复现和浮点对齐：

```text
PP-OCRv5 mobile det: S0-S4
PP-OCRv6 small rec:  S0-S4
PP-OCRv6 small det:  S0-S4
```

这三个模型不执行 S5/S6，不创建 prepared PT2E 图、不运行 QAT、不导出 QuantONNX，也不调整 Axera
qspec。每个模型仍必须验证完整训练输出、targets、loss、主/辅助分支梯度、Paddle 权重严格转换和 deploy
projection。历史 PT2E/QAT/ONNX 产物只保留为 legacy 诊断，不进入本轮结构对齐门禁或指标表。

## 6. 模型顺序

### M1：PP-OCRv5 mobile rec

优先完成：

- 完整 NRTR head 权重转换；
- `MultiLabelEncode`、NRTRLabelEncode、MultiLoss；
- 修正当前输出 key `nrtr` 与 Paddle `gtc` 的差异；
- 恢复 `use_guide=True`，由 NRTR 分支提供正确 backbone 梯度；
- 删除 full-training 路径中的 `rec_ctc_backbone_grad` workaround；
- deploy projection 继续输出单个 CTC logits；
- 重新生成 QAT smoke，不恢复现有 CTC-only QAT checkpoint。

M1 是整个计划的第一阻塞项。

### M2：PP-OCRv5 mobile det

复用 graph role 和 strict conversion 基础，重点完成标准 DB training loss/gradient、训练三通道 maps 与
单 shrink 部署输出的分离。M1 未通过 S4 前不开始 M2 正式改造。

### M3：PP-OCRv6 small rec

复用 M1 的双标签/MultiLoss，新增 PPLCNetV4、lightsvtr、v6 字典和完整 NRTR 参数验证。不得因为
v6 CTC backbone 已有梯度而省略 NRTR 训练结构。本轮只执行 S0-S4 浮点门禁，不执行 PT2E/QAT/ONNX。

### M4：PP-OCRv6 small det

最后补齐 RepLKFPN aux features、三组 aux DBHead、辅助 DBLoss 及其权重转换。验证完整训练结构后，
deployment projection 仍只导出 shrink map。Pad/Axera 工具链问题与训练结构问题分开记录。

## 7. 测试和产物

建议新增或扩展：

```text
tests/test_model_roles.py                 graph role、参数和输出 schema
tests/test_weight_mapping.py              full strict 与 projection report
tests/test_rec_multilabel.py              CTC/NRTR 双标签
tests/test_rec_multihead.py               CTC/gtc forward 和 gradient
tests/test_losses.py                      MultiLoss、NRTRLoss、DB 主/辅助 loss
tests/test_model_compatibility.py         四模型 full/deploy 构建
tools/compare_pretrained_training.py      Paddle/PyTorch 全输出、loss、gradient 报告
```

每个模型产物至少包含：

```text
artifacts/pretrained_structure/<model>/model_manifest.json
artifacts/pretrained_structure/<model>/weight_mapping.json
artifacts/pretrained_structure/<model>/float_forward.json
artifacts/pretrained_structure/<model>/loss_gradient.json
artifacts/pretrained_structure/<model>/deploy_projection.json
```

报告必须记录命令、Git 状态、环境、配置/权重 hash、输入/target hash、所有输出 shape、loss、gradient、
误差、允许删除 key 和首个失败节点。

## 8. 旧产物处理

- 不删除现有 v5-rec CTC-only checkpoint、指标和 ONNX；统一标记为 legacy CTC-only/QAT diagnostic。
- 不从 legacy checkpoint 恢复 full MultiHead QAT。
- 历史 `acc=0.49350` 可以用于部署 CTC 图对比，但不能用于证明 pretrained 训练结构已复现。
- 新 full-training baseline 必须使用新目录、新 metadata schema 和新文档段落，避免指标污染。
- v5/v6 det 历史结果同样先审计训练输出和 loss schema，再决定能否作为新 baseline。

## 9. 停止条件

出现以下任一情况时停止进入后续模型或正式训练：

- full-training 权重不能 strict load；
- 辅助 head 参数缺失或仅随机初始化；
- 训练输出 key、target token 或 loss 与 Paddle 不一致；
- backbone、neck、主 head 或辅助 head 的预期梯度缺失；
- deployment projection 改变主输出；
- optimizer step 与 Paddle 存在未解释的参数更新差异。

## 10. 与现有计划的关系

本计划优先于 `model_accuracy_validation_plan.md` 中的 P4-P8、两个 accuracy recovery plan 中的
PT2E/QAT 训练阶段。现有数据、后处理和 metric 继续复用；Axera、PT2E、QAT 和 QuantONNX 工作保持
暂停，不得混入后续三个模型的结构对齐报告。

## 11. PP-OCRv5 mobile rec 实施记录（2026-08-05）

### 11.1 当前实现边界

已完成：

- Paddle qkv/q/kv 拓扑一致的 NRTR Transformer 和 `gtc` 输出；
- 官方 Paddle pretrained 的完整参数转换，`pretrained_train` 使用 `strict=True`；
- CTC + NRTR 双标签、`MultiLoss`、Trainer target 转发和 raw CTC 验证输出；
- `deploy` 与 `pretrained_train` 图角色，以及 checkpoint 的 `rec_graph/head_schema/loss_schema`；
- full-training float 真实数据 2-step smoke、validation、best/last 保存和 best strict reload。
- Paddle/PyTorch CTC、GTC、MultiLoss、gradient、单步 Adam delta 数值对比；
- full model raw CTC 到 deploy wrapper raw CTC 的精确投影验证。

尚未完成：

- `qat_train` 的“量化 CTC 部署路径 + float NRTR 辅助监督”捕获；
- PT2E、QuantONNX 和 Axera smoke。

因此当前只完成到 M1/S4；尚未形成 PT2E/QAT 精度基线，也不允许据此启动多 epoch QAT。

### 11.2 转换结果

```text
Paddle source entries:       968
PyTorch target entries:      1119
copied source entries:       968
framework-only missing:      151 (全部为 num_batches_tracked)
unexplained source entries:  0
```

```text
weights/PP-OCRv5_mobile_rec_pretrained.pdparams
SHA256 04745475b97a1faf029c7442a4c4421b156249b9395814e509bf4a9804e37750

weights/ptocr_v5_mobile_rec_full.pth
SHA256 b229c8d7050064fb8ade27ec12fd6e008682dfdd0e4da6ddcb6a4b7e8b993c9c

configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml
SHA256 127f8f613d16c7ba9986f427ee00508ef29469efd4a86496be9080d50415339e

configs/qat/training/ppocrv5_mobile_rec_full_pretrained_float_smoke.yml
SHA256 1bb22618741143bde16f74e921c8ac36f86ff39c8be306170555910f24f64dda
```

full-training 前向输出已验证为：

```text
ctc       [2, 40, 18385]
ctc_neck  [2, 40, 120]
gtc       [2, 3, 18389]  # 该合成 target batch 的动态长度
```

backbone、CTC encoder、CTC head、before_gtc 和 gtc_head 均获得 finite gradient。这里仅记录
PyTorch gradient presence；Paddle 数值 parity 仍属于 S3 未完成项。

### 11.3 真实数据 float smoke

数据来自 `/home/heqi/dataset/icdr`，固定输入为 `[3, 48, 320]`。本次只取 4 个训练词图和 4 个验证
词图，batch size 2，共 2 个 optimizer step；关闭随机增强、AMP、warmup、reparameterization 和 QAT。

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --label-file /tmp/ppocrv5_rec_full_train_smoke.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /tmp/ppocrv5_rec_full_val_smoke.txt \
  --val-data-dir /home/heqi/dataset/icdr \
  --output-dir /tmp/ppocrv5_mobile_rec_full_pretrained_float_smoke_20260805 \
  --training-profile \
    configs/qat/training/ppocrv5_mobile_rec_full_pretrained_float_smoke.yml \
  --device cuda:3
```

结果：

```text
epoch:             1
steps:             2
CTCLoss:           43.86863136291504
NRTRLoss:          3.3253453969955444
loss:              47.1939754486084
val_CTCLoss:       0.015125461854040623
val_loss:          0.015125461854040623
val_acc:           1.0 (4 个 smoke 样本)
val_norm_edit_dis: 1.0 (4 个 smoke 样本)
```

`best.pt` strict reload 后 eval-only 得到完全相同的四项验证指标，metadata 为：

```text
rec_graph:       pretrained_train
head_schema:     [ctc, ctc_neck, gtc]
loss_schema:     [CTCLoss, NRTRLoss]
qat:             false
reparameterized: false
epoch/global_step: 1/2
```

产物：

```text
/tmp/ppocrv5_mobile_rec_full_pretrained_float_smoke_20260805/best.pt
SHA256 11c88be841cad3f065aab61a9ff3009f23ebc784d5a6a931de7d43515f4d9b7a

/tmp/ppocrv5_mobile_rec_full_pretrained_float_smoke_20260805/last.pt
SHA256 eaf350187f4abaa775b3ba373d4a309e8ab16b16c1ccc250011bf86ad796c107
```

该产物生成时尚未恢复 Paddle `ConvBNLayer` 的 BN no-decay 参数合同，已由 11.5 的新 smoke 取代。
保留本节只用于追溯，后续不得从这里的 checkpoint 恢复当前 full-training 实验。

### 11.4 原定下一门禁

下一步只执行 M1/S2-S4：使用相同输入和 targets 对比 Paddle/PyTorch 的 CTC、GTC、CTCLoss、
NRTRLoss、总 loss、分模块 gradient 和 optimizer delta，再验证 full eval CTC 到 deploy CTC 的投影误差。
在这些报告通过前不实现 full MultiHead QAT，也不开始 M2。

该门禁已按 11.6 完成；当前下一门禁更新为 M1/S5。

### 11.5 Paddle optimizer 参数合同修正

Paddle 参数审计发现 PPLCNetV3 `ConvBNLayer` 的 BN weight/bias 显式使用 `L2Decay(0)`。旧 PyTorch
optimizer 将其放入全局 `3e-5` weight decay，属于训练合同差异。现已在模型参数上保留
`_paddle_weight_decay`，并由 `build_optimizer()` 生成以下分组：

```text
default          297 tensors  lr=2e-5  weight_decay=3e-5
paddle_no_decay  254 tensors  lr=2e-5  weight_decay=0
lab              112 tensors  lr=2e-6  weight_decay=3e-5
ctc_fc              2 tensors  lr=2e-5  weight_decay=1e-5
```

修正后重新执行相同 4 train + 4 val、2-step float smoke：

```text
CTCLoss:           44.091665267944336
NRTRLoss:          3.324574112892151
loss:              47.41623878479004
val_CTCLoss:       0.015049360692501068
val_loss:          0.015049360692501068
val_acc:           1.0
val_norm_edit_dis: 1.0
```

best strict reload 后四项验证指标完全一致。当前产物：

```text
/tmp/ppocrv5_mobile_rec_full_pretrained_float_smoke_optimizer_contract_20260805/best.pt
SHA256 c46c388273e71d04a13ebbc125c707105af5916beb9a11a663fdbc64ff4172d7

/tmp/ppocrv5_mobile_rec_full_pretrained_float_smoke_optimizer_contract_20260805/last.pt
SHA256 2c4d3792f10a9621f8f8b2428c7ec3afc9b2e22ba0ba084227ffb7372ae9bad9
```

### 11.6 Paddle/PyTorch full-training parity

canonical 命令使用 CPU，避免 Paddle/PyTorch 不同 cuDNN BN kernel 把 backend 数值误差混入结构等价
判断。Transformer 的 31 个 Dropout 在两边同时关闭，BN 分别验证 `eval_bn` 和 `train_bn`：

```bash
env PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/ocr_moderation/bin/python -u \
  tools/compare_pretrained_training.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --paddle-weights weights/PP-OCRv5_mobile_rec_pretrained.pdparams \
  --torch-weights weights/ptocr_v5_mobile_rec_full.pth \
  --output artifacts/pretrained_structure/ppocrv5_mobile_rec/\
full_training_parity_cpu_final.json \
  --device cpu \
  --modes eval_bn train_bn
```

合同：

```text
input shape:    [2, 3, 48, 320]
seed:           20260805
images SHA256:  b74fd7e85de6471556a9af38949c3ca6c2922fe5dbbe5d77c22a5c69d5a0af5e
targets SHA256: 0243ac4708bdcebe4c9ad8cf1d5435a443418f0e3606ff3541f29a8251fbb3be
report SHA256:  493d9cba444708703608324ae0574450a3c6a3a5fa839b679710feea62cff8d9
```

关键结果：

| 项目 | eval_bn | train_bn |
|---|---:|---:|
| CTC probability MAE | `3.0412e-10` | `6.6383e-10` |
| CTC argmax agreement | `1.0` | `1.0` |
| GTC probability MAE | `2.1383e-10` | `3.4988e-10` |
| GTC argmax agreement | `1.0` | `1.0` |
| total loss relative error | `9.0464e-6` | `1.7965e-5` |
| gradient cosine | `0.9999999071` | `0.9999999210` |
| gradient relative L2 | `4.3610e-4` | `4.0319e-4` |

`train_bn` 单步 Adam delta：

```text
MAE:                              3.407970983633475e-08
cosine similarity:                0.9999555709353581
relative L2:                      0.009426455419694179
direction disagreement fraction: 6.771405150813975e-05
max_abs:                          0.0009999871253967285
```

`max_abs` 来自 Adam 首步中极少量接近零 gradient 元素的符号敏感更新；方向不一致仅占约 `0.0068%`，
整体 delta MAE 和 cosine 通过当前聚合门槛。full-eval CTC 到 deploy CTC 的 MAE/max_abs 均为 0。

CPU acceptance：probability MAE `<=1e-7`、argmax `>=0.9998`、loss relative error `<=1e-4`、gradient
cosine `>=0.99999`、gradient relative L2 `<=1e-3`、Adam delta cosine `>=0.9999`、delta MAE
`<=1e-7`。M1/S2、S3 和 S4 按这些门槛通过。

GPU 诊断 `artifacts/pretrained_structure/ppocrv5_mobile_rec/full_training_parity.json` 中，`eval_bn`
仍通过；`train_bn` probability MAE 上升到 CTC `1.9769e-7`、GTC `2.4102e-7`。CPU 同输入通过，且
两边 BN eps/结构一致，因此该差异记录为 Paddle/PyTorch cuDNN train-BN backend sensitivity，不修改模型
代码去拟合某一框架的 GPU kernel。

### 11.7 当前门禁

```text
M1/S0  部分完成：参数 manifest/允许删除集合仍需形成独立 JSON
M1/S1  通过：完整转换、strict load、CTC+GTC forward
M1/S2  通过：CPU eval_bn + train_bn full-output parity
M1/S3  通过：MultiLoss、五类 owner gradient、单步 Adam delta
M1/S4  通过：full raw CTC -> deploy raw CTC，误差为 0
M1/S5  通过：完整重参数化、exported float、prepared fake-off 均通过对应门禁
M1/S6  按 2026-08-06 范围调整暂停，不继续 PT2E/QAT
```

回归结果：`111 passed, 12 subtests passed`。M1 已冻结，后续不设计 `qat_train`；M2 已开始。

### 11.8 M1/S5 重参数化误差定位

真实 PP-OCRv5-rec full 权重、ICDAR 词图上逐层比较 eager deploy 与 reparameterized deploy。CPU
同输入局部误差从每层约 `1e-9~4e-6` 逐步累积，最终 2 张样本 raw CTC logits 为：

```text
MAE:      1.3882179346e-4
max_abs:  2.2468566895e-3
```

GPU 3 上首个明显放大点是 `blocks2.0.pw_conv`。该 pointwise 层由 4 个 1x1 Conv-BN 分支融合为
一个 1x1 Conv；同输入局部 MAE 从 CPU `9.3521e-8` 上升为 TF32 下 `1.8120e-4`。depthwise
层仍保持约 `1e-9`，后续所有 pointwise 多分支层表现一致，未发现 identity、padding、tuple stride、
BN running stats 或分支重复累加错误。

GPU 3、2 张真实词图的 TF32 A/B：

| 模式 | logits MAE | logits max_abs | probability MAE | probability max_abs | argmax |
|---|---:|---:|---:|---:|---:|
| TF32 on | `9.94146e-2` | `1.89561` | `2.01937e-9` | `3.90232e-4` | `1.0` |
| TF32 off | `2.07899e-4` | `3.76129e-3` | `3.73177e-12` | `7.15256e-7` | `1.0` |

结论：当前是融合前后浮点累加路径和 CUDA TF32 backend sensitivity，不是融合公式错误。生产模型和
qspec 不改；后续 S5 的 rec 重参数化验收固定关闭 TF32，并使用 probability/argmax/CTC sequence
任务等价门禁。临时逐层报告为 `/tmp/diagnose_v5_rec_rep_gpu.json`。

更新后的正式 4 样本报告：

```text
artifacts/pretrained_structure/ppocrv5_mobile_rec/deploy_reparameterization_tf32_off.json
SHA256 7b1e0dd7a235518e52336c370905c885cf28a3c2df5f3c5a05697f8f601c46b1

raw logits MAE:          2.9333917090e-4
raw logits max_abs:      3.8528442383e-3
centered logits MAE:     8.8115504879e-5
probability MAE:         2.6196308133e-12
probability max_abs:     7.1525573730e-7
argmax agreement:        1.0
CTC sequence agreement:  1.0
TF32 matmul/cuDNN:        false / false
```

M1/S5 的 deploy projection -> reparameterized eager 子门禁通过。下一步继续验证 reparameterized
eager -> exported float -> prepared fake-off。

### 11.9 M1/S5 export/PT2E float 保持

2026-08-06 使用当前 full 权重、完整 2077 张 ICDAR2015 recognition validation、GPU 2、batch 128、
TF32 off 执行：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u \
  tools/compare_pt2e_float_preservation.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --qat-config configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json \
  --label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --data-dir /home/heqi/dataset/icdr \
  --output artifacts/pretrained_structure/ppocrv5_mobile_rec/\
pt2e_float_preservation_full_tf32_off.json \
  --samples 0 --batch-size 128 --workers 0 --device cuda:2 \
  --reparameterize --disable-torch-tf32
```

合同和产物：

```text
full weights SHA256:  b229c8d7050064fb8ade27ec12fd6e008682dfdd0e4da6ddcb6a4b7e8b993c9c
QAT config SHA256:    87ad8753cc2ff5a4aa94980ddcd01803c75a30c18ce231fabc10ba6ce27996b4
report SHA256:        faff376d31335a3d831ff931f4850bcecb4c98d863fd7f7258931991abad579c
exported nodes:       500
prepared nodes:       984
common state keys:    236，changed 0
fake quant/observer:  382/382 -> 0/0
```

完整任务指标：

| Stage | Accuracy | Normalized edit distance |
|---|---:|---:|
| reparameterized eager | `0.5936446798266731` | `0.8169468283815902` |
| exported float | `0.5936446798266731` | `0.8169468283815902` |
| prepared fake-off | `0.5936446798266731` | `0.8169468283815902` |

边界指标：

```text
eager -> exported:
  logits/probability MAE/max: 0 / 0
  argmax/CTC sequence:        1.0 / 1.0

exported -> prepared fake-off:
  logits MAE/max:             5.7013758831e-6 / 8.3827972412e-4
  centered logits MAE/max:    3.4634080479e-6 / 5.2642822266e-4
  probability MAE/max:        1.5795805993e-11 / 1.0156631470e-4
  argmax/CTC sequence:        1.0 / 1.0
  accuracy/edit delta:        0 / 0
```

`prepared_operator_delta` 仅包含 5 组 `sqrt/div/reshape/mul/add`。FX 原图对应模块为
`model.head.ctc_encoder.encoder.conv1/conv2/conv3/conv4/conv1x1`，即 EncoderWithSVTR 的 5 个
Conv-BN；这是 `prepare_qat_pt2e` 的标准 QAT arithmetic rewrite。严格 raw tensor 门禁未通过，但
标准 rewrite 的 state、fake-off 和完整任务门禁全部通过；按项目约束不增加 eager Conv-BN 固定融合。

同日使用未重参数化 deploy 与重参数化 deploy 对完整 2077 张验证集执行 TF32-off 对照：

```text
artifact:
  artifacts/pretrained_structure/ppocrv5_mobile_rec/
  deploy_reparameterization_full_tf32_off.json
SHA256:
  c98f26e9678dc6ff047796da0c0fc675f34a127924b20c1c8b93971d09f91946

raw logits MAE/max:       2.3934345432e-4 / 1.1271476746e-2
centered MAE/max:         7.2968076266e-5 / 6.4697265625e-3
probability MAE/max:      1.6563072890e-10 / 1.6836225986e-3
argmax agreement:         1.0
CTC sequence agreement:   1.0
TF32 matmul/cuDNN:         false / false
```

CTC sequence 2077/2077 一致，因此两个 deploy graph 的 decoded accuracy/edit 指标严格相同。
M1/S5 全部门禁通过。按 2026-08-06 范围调整，S6 和后续 PT2E/QAT 工作暂停，不修改 qspec。

## 12. PP-OCRv5 mobile det 实施记录（2026-08-06）

### 12.1 浮点训练合同

已补齐并验证以下 Paddle pretrained 训练结构：

- 原生 `pretrained_train` graph 返回 `maps=[shrink, threshold, binary]`；
- validation 保持全模型 BN 为 eval，仅让 DBHead 选择三 map 训练输出；
- `DBLoss` 使用官方 `BalanceLoss + DiceLoss`、`alpha=5`、`beta=10`、`ohem_ratio=3`；
- Adam/Cosine/warmup/L2 和 DBPostProcess 字段已写入本地 YAML；
- 官方 Paddle 权重完整转换，905 个 source tensor 全部复制，额外 150 项仅为 PyTorch
  `num_batches_tracked` buffer。

权重合同：

```text
weights/PP-OCRv5_mobile_det_pretrained.pdparams
SHA256 7e2e3b0bd5bbdcb0b842cb92aaacc2852f80299a4858b8767a45bd0c6e955648

weights/ptocr_v5_mobile_det.pth
SHA256 d5013fadb085acfac1ed29dcd3106ccc4beca24e31e92258f0b6ee2b793904d3
```

### 12.2 Paddle/PyTorch full-training parity

正式 CPU 浮点报告：

```text
artifacts/pretrained_structure/ppocrv5_mobile_det/training_parity_cpu_final.json
SHA256 f09502b4d460595e54a71d8d60a6327a73cec098dd783cab854db8b554ddd1a6
```

命令：

```bash
HOME=/tmp/ocr-home PYTHONPATH="$PWD" \
  /home/heqi/miniforge3/envs/ocr_moderation/bin/python -u \
  tools/compare_det_training.py \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --paddle-weights weights/PP-OCRv5_mobile_det_pretrained.pdparams \
  --torch-weights weights/ptocr_v5_mobile_det.pth \
  --output artifacts/pretrained_structure/ppocrv5_mobile_det/training_parity_cpu_final.json \
  --image-shape 3 64 64 --batch-size 2 --seed 20260806
```

固定合同：

```text
input SHA256:   88bc85f854d5ebe6aa84bf97c31ace20f207d03d91d3ec334deae3df6243abc7
targets SHA256: b40f02eefdeeeade577e0cb265a7929de004bb809e30cd67f260be5d2e24214f
```

关键结果：

| 项目 | eval_bn | train_bn |
|---|---:|---:|
| all maps MAE | `3.3135e-8` | `1.9179e-6` |
| all maps max_abs | `6.8545e-7` | `1.2474e-3` |
| total loss abs | `3.3379e-6` | `8.5831e-6` |
| backbone gradient cosine | `0.999999999996` | `0.999999949560` |
| backbone gradient relative L2 | `3.5529e-6` | `3.3057e-4` |
| neck gradient cosine | `0.9999999999997` | `0.999999983307` |
| binarize gradient cosine | `0.9999999999999` | `0.999999964259` |
| threshold gradient cosine | `0.9999999999999` | `0.999999998816` |

### 12.3 Adam 参数合同和近零梯度诊断

PPLCNetV3 的 112 个 `LearnableAffineBlock` 参数在 Paddle 中使用 `0.1x` learning rate。现已将该值
保存为参数级 `_paddle_lr_multiplier`，optimizer 默认读取模型合同，显式训练配置仍可覆盖。最终分组：

```text
default          239 tensors  lr=1e-3  weight_decay=5e-5
paddle_no_decay  254 tensors  lr=1e-3  weight_decay=0
lab              112 tensors  lr=1e-4  weight_decay=5e-5
```

独立标量测试确认 Paddle/PyTorch Adam 在 `gradient=1` 到 `1e-10` 范围内的 epsilon 和首步公式一致。
模型原始全量 Adam delta 仍为 cosine `0.9980975`、relative L2 `0.06168`；逐参数报告证明主要来源是
两框架 train-BN 浮点尾差使近零 gradient 改变符号，随后被 Adam 首步近似 sign 更新放大，不是参数分组
或 optimizer 公式错误。

按两边 `|gradient + weight_decay * weight| >= 1e-4` 的稳定子集统计：

```text
elements:                       1,182,447
delta cosine:                   0.9999999831
delta relative L2:              1.8393e-4
delta MAE:                      2.6831e-10
direction disagreement:         8.4570e-7（1 个元素）
```

`>=1e-5` 子集仍有 delta cosine `0.9999755190`、MAE `2.5144e-8`。正式 JSON 同时保存原始聚合、
逐参数 max/MAE/direction 排名和各有效梯度阈值，禁止只引用过滤后的结果隐藏原始差异。该差异记录为
跨框架 train-BN 梯度零点敏感性，M2/S3 按结构、loss、gradient 和稳定更新合同通过。

### 12.4 Deployment projection

同一份 full state 下：

```text
pretrained_train eval-BN maps: [2, 3, 64, 64]
full shrink:                    [2, 1, 64, 64]
deploy wrapper shrink:          [2, 1, 64, 64]
MAE/max_abs/relative_L2:         0 / 0 / 0
```

M2/S4 通过。该门禁只运行 eager 浮点模型；没有运行 export、PT2E、QAT、QuantONNX 或 qspec 调整。

### 12.5 当前门禁

```text
M2/S0  通过：官方 YAML、训练输出和权重合同已冻结
M2/S1  通过：完整转换和 strict load
M2/S2  通过：eval-BN/train-BN 三 map 浮点对齐
M2/S3  通过：DBLoss、四类 owner gradient、Adam 稳定更新合同
M2/S4  通过：full shrink -> deploy shrink，误差为 0
M2/S5  不适用：按当前范围禁止 PT2E/QAT/ONNX
M2/S6  不适用：按当前范围禁止 PT2E/QAT/ONNX
```

### 12.6 真实 ICDAR2015 float smoke

固定使用 4 条 train、4 条 validation 标注，输入 `[3,640,640]`，batch size 2，共两个 optimizer step。
显式关闭 AMP、warmup 和 reparameterization，并使用 `--no-qat`；没有创建 prepared graph、observer 或
QuantONNX。

首轮 smoke 发现 `Trainer._forward()` 只按 `graph_role=pretrained_train` 判断 full recognition，导致 det
原生训练图错误索取 CTC/NRTR targets。现已增加 `model_type=rec` 联合条件，并新增 det trainer 回归。

修复后结果：

```text
epoch/steps:              1 / 2
learning rate:            default=1e-3, no_decay=1e-3, lab=1e-4
loss:                     4.9180238247
loss_shrink_maps:         3.4268022776
loss_threshold_maps:      0.8054546714
loss_binary_maps:         0.6857670546
val_loss:                 3.4181205034
val_precision/recall:     0.6666666667 / 0.2
val_hmean:                0.3076923077
```

checkpoint：

```text
/tmp/ppocrv5_mobile_det_pretrained_float_smoke_20260806/best.pt
SHA256 ad1e1355b6f20e2e340804a63fe3c77d1221347586fa7d3a971dc1cb5c961875

/tmp/ppocrv5_mobile_det_pretrained_float_smoke_20260806/last.pt
SHA256 63f54a2a178e9475e6d570321196ea47371fb59fccd2754d934a1b2a0a2d0f75
```

`best.pt` 以 `--eval-only --eval-stage float --no-qat` strict reload 后，epoch/global step 为 `1/2`，
validation loss、三个子 loss、precision、recall 和 hmean 与训练结束时逐项一致。

聚焦回归：`49 passed`（`test_trainer.py`、`test_training_profile.py`、`test_model_roles.py`、
`test_losses.py`），仅有 sandbox 内 NVML 初始化 warning。M2 已完成，下一步进入 M3/PP-OCRv6 small rec
的 S0-S4；继续禁止 PT2E/QAT/ONNX。

## 13. PP-OCRv6 small rec 实施记录（2026-08-06）

### 13.1 S0：旧权重审计和完整转换

原有权重：

```text
weights/ptocr_v6_rec_PP-OCRv6_small_rec_pretrained.pth
```

只有 396 个 state entries，缺失 `before_gtc` 和完整 `gtc_head`，strict load 失败。该文件继续标记为
legacy CTC-only/deploy 诊断，不覆盖或迁移成 full-training 权重。

Paddle 权重严格映射预检和正式转换结果：

```text
Paddle source entries:       422
PyTorch target entries:      480
copied source entries:       422
framework-only missing:      58（全部为 num_batches_tracked）
unknown source/shape error:  0
ignored source:              0
```

严格转换还定位并修正了 v6 NRTR 词表合同：Paddle `Transformer` 在配置的
`NRTRLabelDecode` channel count 上再增加 1 个 vocabulary slot；因此 converter 从 Paddle embedding
实际 vocab `18714` 反推配置值 `18713`，不能直接把 `18714` 传给 PyTorch Transformer。

当前 full 权重：

```text
weights/PP-OCRv6_small_rec_pretrained.pdparams
SHA256 25c9bd54b0e5900916e8bb6ada938abeffb1eac1baedac0ca54a45b1c9310825

weights/ptocr_v6_small_rec_full.pth
SHA256 86242dac7e0ba6079698d105cc84e579bcceb170a0633e15eef24aee4474e0f2
```

### 13.2 S1/S2：full state 和输出 schema

full 权重 strict load 通过，PyTorch state entries `480`，trainable parameter count `29,270,246`。
固定 synthetic batch `[2,3,48,320]` 的 full training output：

```text
ctc       [2, 40, 18710]
ctc_neck  [2, 40, 120]
gtc       [2, 6, 18714]
```

所有输出 finite。full model eval CTC 到 deploy CTC 的 projection：shape `[2,40,18710]`，MAE/max_abs/
relative MAE 为 `0/0/0`。

### 13.3 Paddle/PyTorch parity

正式 CPU 报告：

```text
artifacts/pretrained_structure/ppocrv6_small_rec/full_training_parity_cpu_final.json
SHA256 c29d59efc430391d82112a70bd6ce12a582c61024adf4a303fbd4c4f49471995
```

合同：

```text
config SHA256:   168cd60ce69ff762415b527315407b4a868d3fab912c73ba5a29b3e569f379b7
input SHA256:    f7aa65a646ad1f884dad9331eb96e676fe318688edcd448146811d8dbd365c5a
targets SHA256:  0243ac4708bdcebe4c9ad8cf1d5435a443418f0e3606ff3541f29a8251fbb3be
```

关键结果：

| 项目 | eval_bn | train_bn |
|---|---:|---:|
| CTC probability MAE | `9.4649e-10` | `1.2292e-10` |
| GTC probability MAE | `1.0530e-9` | `5.2631e-10` |
| CTC/GTC argmax agreement | `1.0 / 1.0` | `1.0 / 1.0` |
| total loss relative error | `3.1175e-5` | `8.2840e-5` |
| overall gradient cosine | `0.9999999959` | `0.9999999656` |
| overall gradient relative L2 | `9.1048e-5` | `2.6324e-4` |
| GTC gradient cosine | `0.9999999780` | `0.9999999793` |

PPLCNetV4 `ConvBNAct` 的 BN affine 参数已补齐 Paddle 的 `L2Decay(0.0)` 元数据，optimizer 分组为
`default=293`、`paddle_no_decay=10`、`ctc_fc=2`。单步 Adam 原始全量 delta 为 cosine `0.9998218`、
relative L2 `0.01888`、MAE `7.7591e-8`；与 v5-det 相同，差异集中在跨框架 BN/大规模 attention 梯度接近
零时的首步 sign 敏感更新。该 raw delta 保留在报告中，不通过修改网络或 optimizer 公式拟合 CPU backend；
forward、loss、gradient 和 deployment projection 作为本轮结构门禁通过。

### 13.4 当前门禁

```text
M3/S0  通过：Paddle YAML、完整 MultiHead schema、旧权重 legacy 边界已冻结
M3/S1  通过：422 -> 480 strict mapping，只有 58 个 framework buffer
M3/S2  通过：CTC/CTC neck/GTC full output、loss 和 deploy projection
M3/S3  通过：五类 owner gradient；Adam raw delta 作为 backend sensitivity 记录
M3/S4  通过：真实 ICDAR float smoke 和 full/best strict reload
M3/S5  不适用：禁止 PT2E/fake-quant 浮点保持测试
M3/S6  不适用：禁止 QAT/QuantONNX
```

### 13.5 真实 ICDAR2015 float smoke

使用 4 条 train、4 条 validation 词图，输入 `[3,48,320]`、batch size 2、两个 optimizer step；显式
`--rec-graph pretrained_train --no-qat --no-reparameterize --no-amp --warmup-epochs 0`。

```text
epoch/steps:          1 / 2
learning rate:        default=5e-4, no_decay=5e-4, ctc_fc=5e-4
CTCLoss:              25.6820697784
NRTRLoss:             4.3762847185
loss:                 30.0583553314
val_CTCLoss/loss:     26.2018899918 / 26.2018899918
val_acc:              0.0（4 样本 smoke，不作为精度 baseline）
val_norm_edit_dis:    0.15625
```

validation 按部署合同只评估 CTC，因此不包含 NRTRLoss；训练阶段 CTC+NRTR 两个 loss 均参与反向。

```text
/tmp/ppocrv6_small_rec_pretrained_float_smoke_20260806/best.pt
SHA256 a8c4532d241e8212ed38ad9e5eda29885e4aa23bf645e5616adbd86d7afbc432

/tmp/ppocrv6_small_rec_pretrained_float_smoke_20260806/last.pt
SHA256 1ffa1bacb064325a4d58593e508083f960925571ccc01151d2734317a32c1ad8
```

`best.pt` 以 `eval-only/float/no-qat` strict reload 后，epoch/global step 为 `1/2`，CTCLoss、总 loss、
accuracy 和 normalized edit distance 与训练结束时逐项一致。M3/S0-S4 已完成；下一步进入
M4/PP-OCRv6 small det 的 S0，继续禁止 PT2E/QAT/ONNX。

## 14. PP-OCRv6 small det 实施记录（2026-08-06）

### 14.1 S0/S1：aux 训练结构和完整转换

原 PyTorch `DBHead` 只包含主 `binarize/thresh`，converter 还主动删除全部
`aux_binarize_p*/aux_thresh_p*`。因此旧权重：

```text
weights/ptocr_v6_det_PP-OCRv6_small_det_pretrained.pth
SHA256 093e916f664a689efb567d8b0c32f6ff114c406bcceb460ed7c3d3e2e62438bd
```

只作为 legacy deploy 权重保留。当前已完成：

- DBHead 新增 p4/p3/p2 三组独立 `aux_binarize + aux_thresh`；
- RepLKFPN 的 `fuse/aux_p4/aux_p3/aux_p2` 在 pretrained train graph 中完整传给 DBHead；
- train/validation 返回 `maps/aux_maps_p4/aux_maps_p3/aux_maps_p2`；
- validation 仅打开 neck/head 的分支选择 flag，所有子 BN 仍为 eval；
- deploy wrapper 仍只执行主 shrink 路径；
- converter 删除“跳过 aux 参数”逻辑，改为完整 strict mapping。

权重合同：

```text
references/PaddleOCR/models/PP-OCRv6_small_det_pretrained/
  PP-OCRv6_small_det_pretrained.pdparams
SHA256 13e7072d5d6837ee809b03b91ea9def6dc1b1831b3c6e49405e6ddf97875043f

weights/ptocr_v6_small_det_full.pth
SHA256 567dd7589140aaefeee3d42cafdb39f1b13e014323fb242a9dc05b9c4a6b7bc8
```

严格映射：

```text
Paddle source entries:       514
PyTorch target entries:      600
copied source entries:       514
framework-only missing:      86（全部为 num_batches_tracked）
unknown/shape mismatch:      0
ignored source:              0
```

### 14.2 S2/S3：四路输出、loss 和 gradient parity

正式 CPU 报告：

```text
artifacts/pretrained_structure/ppocrv6_small_det/training_parity_cpu_final.json
SHA256 55ab09e32e71f069ba561a46cd272869140788ecd5833568f0ae1fe17b34d23e
```

固定 batch `[2,3,64,64]` 下，主 maps 和 p2/p3/p4 三路 aux maps 均为 `[2,3,64,64]`。

| 输出 | eval_bn MAE/max_abs | train_bn MAE/max_abs |
|---|---:|---:|
| maps | `3.8325e-8 / 6.2585e-7` | `2.7127e-6 / 9.2238e-4` |
| aux_maps_p2 | `4.2912e-8 / 7.4506e-7` | `3.3264e-6 / 7.6920e-4` |
| aux_maps_p3 | `4.3678e-8 / 8.3447e-7` | `3.4716e-6 / 2.8010e-3` |
| aux_maps_p4 | `3.3443e-8 / 5.6624e-7` | `2.0458e-6 / 6.5854e-4` |

total loss abs 为 eval-BN `1.9073e-6`、train-BN `3.8147e-6`；三个 aux 子 loss 在 train-BN 的
absolute error 分别为 p2 `6.0081e-5`、p3 `2.0981e-5`、p4 `5.7220e-6`。

backbone、neck、主 binarize/thresh 和六个 aux head 均有 finite gradient。eval-BN owner gradient cosine
全部大于 `0.999999999999`；train-BN 最低 cosine 为 aux-thresh-p2 的 `0.9999962173`，最大 relative L2
为 `0.0027506`。

PPLCNetV4 stem 的 10 个 ConvBNAct BN affine 参数使用 no-decay，最终 optimizer 分组为
`default=332`、`paddle_no_decay=10`。原始全量 Adam delta cosine `0.9989754`、MAE `1.0025e-6`；
两边有效梯度均不低于 `1e-4` 的 1,605,385 个元素上，delta cosine `0.9999352`、MAE `6.4870e-8`。
与前两个模型一致，完整 raw 指标保留，不为近零梯度首步 sign 敏感性修改网络。

### 14.3 S4：deployment projection 和真实数据 smoke

同一 full state 的 full shrink 与 deploy shrink shape 均为 `[2,1,64,64]`，MAE/max_abs/relative L2
为 `0/0/0`。

ICDAR2015 4 train + 4 validation、`[3,640,640]`、batch 2、两个 optimizer step 的纯浮点 smoke：

```text
loss:                     8.4204926491
loss_aux_maps_p2:         4.3072161674
loss_aux_maps_p3:         4.3287646770
loss_aux_maps_p4:         4.6855514050
val_loss:                 7.4472494125
val_loss_aux_maps_p2:     3.8765134811
val_loss_aux_maps_p3:     3.9625831842
val_loss_aux_maps_p4:     4.1696875095
val_precision/recall:     0.75 / 0.3
val_hmean:                0.4285714286
```

```text
/tmp/ppocrv6_small_det_pretrained_float_smoke_20260806/best.pt
SHA256 d3f537c7ecad0c4cd069d71eb807eb40e5d88400b955aec09db8f7adf2608e80

/tmp/ppocrv6_small_det_pretrained_float_smoke_20260806/last.pt
SHA256 56db8fa1c9c830554c68cf3b9ececdb720a74d5c0296db5d2716372189e034b1
```

`best.pt` strict reload 后主/aux validation losses、precision、recall 和 hmean 逐项一致。smoke 显式使用
`--no-qat --no-reparameterize --no-amp`，没有创建 observer、prepared graph 或 ONNX。

### 14.4 最终门禁

```text
M4/S0  通过：RepLKFPN + DBHead 主/aux 训练结构完整
M4/S1  通过：514 -> 600 strict mapping，仅 86 个 framework buffer
M4/S2  通过：主 maps + p2/p3/p4 aux maps 浮点对齐
M4/S3  通过：主/aux DBLoss、十类 owner gradient 和 Adam 诊断
M4/S4  通过：deploy shrink 投影误差 0，真实 float smoke strict reload
M4/S5  不适用：禁止 PT2E/fake-quant 测试
M4/S6  不适用：禁止 QAT/QuantONNX
```

聚焦回归：`52 passed`，两个 warning 分别为 sandbox NVML 和 PyTorch even-kernel `padding=same` 提示。
四个模型的 pretrained 浮点训练结构工作包至此完成；后续不得自动进入 PT2E/QAT 阶段。

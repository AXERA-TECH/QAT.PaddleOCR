# PP-OCRv5/v6 浮点 + QAT 蒸馏(KD)支持计划

状态：已确认范围（2026-08-10），等待实现。

## 1. 目标和边界

为本项目的 v5/v6 det/rec 训练链路增加知识蒸馏(KD)能力，覆盖**浮点训练**与 **Axera PT2E QAT
训练**两种模式。KD 采用标准"冻结浮点 Teacher + 可训练 Student"结构：Teacher 独立于 PT2E
QAT 图之外（不参与量化），Student 为现有单模型路径（浮点 eager 或 prepared QAT 图）。

目标：

```text
1. v5/v6 det：Teacher/Student 在 DB maps(shrink/threshold/binary)上做 MSE KD（输出头）
2. v5/v6 rec：Teacher/Student 在 CTC logits 上做温度 KL(或 MSE)KD（输出头）
3. 支持多层中间特征 KD：backbone_out / neck 特征(ctc_neck、fuse)按层独立配置权重，
   默认 0 关闭，实验确认后再逐步启用
4. 同时支持浮点训练(amp 可用)与 QAT 训练(FP32 激活)两种模式
5. 不破坏现有单模型训练、checkpoint resume、epoch-2 精度门禁合同
```

边界（本计划不做）：

```text
1. 不做 Teacher 量化（Teacher 永远保持 FP32 浮点，冻结、不进入 optimizer）
2. 不引入 DistillationModel dict 合同；沿用现有单模型输出结构，
   Teacher/Student 构建时开启 return_all_feats=True 暴露中间特征
   （backbone_out / ctc_neck / fuse），KD 按层配置消费
3. 不做 NRTR/GTC 输出的 KD（v5 rec 全训练图先只对 CTC logits 做 KD，GTC KD 另行实验）
4. 不做 v3/v4 蒸馏模型转换（本项目只覆盖 v5/v6，且 KD 只做训练期监督，不新增部署结构）
5. 不改动 references/ 仓库；借鉴 references/PytorchOCR 的 KD loss 公式与多层 key 机制，
   最小实现到主工程
```

### 1.1 GTC KD 延后的理由（已确认）

v5 rec 全训练图含 CTC + NRTR/GTC 双头，本计划第一轮只对 CTC logits 做 KD，GTC KD
作为后续独立实验。理由：

1. **单变量实验规则**：沿用 Exp2/Exp3 的纪律——每轮只改一个变量才能归因。CTC KD 是
   "加 KD vs 不加 KD"的单一变量；若同时叠加 CTC KD + GTC KD，无法区分各自贡献，也不利于
   定位崩溃（如 Exp2 的 epoch-2 精度断崖）。当前 v5 rec 基线是 reparam=false 下 epoch-2
   acc 0.39，CTC 是部署目标，先验证"KD 能否恢复 QAT 精度"这一命题最直接。
2. **GTC 输出对齐工程复杂度更高**：CTC KD 只需比较形状相同的 raw logits（teacher/student
   均为 `[B, T, C_cls]`）；NRTR decoder 输出依赖 `gtc_targets` 输入和 causal/attention
   mask，student 侧（FullRecTrainingWrapper）按 PT2E 静态图把 gtc_targets pad 到
   max_text_length=25，teacher 侧 eager NRTR 按 Paddle 动态 `max_len` 裁剪，长度维度
   不一致，需要额外的裁剪/掩码对齐；且 PT2E 图 NRTR 路径已有 causal mask 修复历史
   （non-finite causal mask），叠加 KD 会引入新的不确定性。
3. **GTC 是训练辅助，不是部署交付**：NRTR/GTC 头只在训练阶段帮助 backbone + CTC 学习，
   部署图不保留。KD 的最终约束对象仍是 CTC 精度，GTC KD 通过 backbone 间接影响 CTC，
   因果链更长，应在 CTC KD 效果确认后独立评估。
4. **增量设计，不推翻架构**：Trainer 的 `teacher` + `kd_criterion` 构件天然支持后续新增
   `KDGTCLoss`（train_step 中多一次 kd 计算），无需改动 Trainer 架构、profile 合同或
   CLI，GTC KD 是增量而非重构。

执行顺序：CTC KD 验证通过后再单独做 GTC KD 实验，结论按同一门禁体系记录。

## 2. 设计

### 2.1 组件

```text
pytorchocr/training/losses/kd.py      KD loss（KDLogitsLoss / KDMapsLoss / KDFeatureLoss /
                                      KDCompositeLoss / build_kd_criterion；
                                      按 key 提取层输出，逐层计算并加权聚合）
pytorchocr/quantization/bridge.py     FullRecTrainingWrapper / DetTrainingWrapper 扩展：
                                      训练输出额外暴露 backbone_out / fuse，
                                      供 QAT prepared 图的多层 KD 消费
pytorchocr/training/trainer.py        Trainer 增加可选 teacher + kd_criterion + kd_weight；
                                      _teacher_forward 支持 pretrained_train 数据路径；
                                      _kd_losses 逐层计算并加权
tools/train.py                        --kd / --kd-layers / --kd-*weight 等 CLI +
                                      teacher 构建（权重默认回退 --weights）+
                                      return_all_feats 注入 + metadata
pytorchocr/training/profile.py        training profile 增加 KD 字段与校验
```

> 多层 KD 依赖 `return_all_feats` 暴露中间特征：eager 全模型（rec pretrained_train）走
> `BaseModel.forward` 已支持；但 QAT 学生（`FullRecTrainingWrapper` / `DetTrainingWrapper`）
> 是自定义 forward，当前只返回任务输出，**必须扩展 wrapper 暴露 `backbone_out` / `fuse`**
> 才能做 backbone/neck 层 KD——这是 S2 的必改项，输出头 KD 不受影响。

### 2.2 Teacher 构建合同

- Teacher 与 Student 使用**相同 graph role**构建，保证输出结构对齐：
  - rec deploy（含 QAT 部署图）：`RecCTCWrapper` → CTC logits `[B, T, C]`
  - rec pretrained_train（v5 rec 全训练图，含 QAT）：MultiHead dict `{ctc, ctc_neck, gtc}`，KD 取 `ctc`
  - det training（含 QAT）：`DetTrainingWrapper` → `(shrink, threshold, binary)` 元组
- Teacher/Student 均以 `return_all_feats=True` 构建，暴露 `backbone_out`、neck 特征
  （rec `ctc_neck`、det `fuse`）等中间输出供多层 KD 消费；现有 BaseModel.forward
  已支持 `return_all_feats`（训练时返回完整 dict）；QAT wrapper 的中间层暴露见 2.1 说明；
- Teacher 权重默认取 Student 的浮点权重（`--weights`/profile `weights`），**无需显式指定**
  `--teacher-weights`；仅当 Teacher 需要不同权重（如专用大模型 teacher）时才显式提供。
  `reparameterize` 与 Student 一致；
- Teacher 冻结：`requires_grad=False`；前向在 `torch.no_grad()` 与 eval 模式（CTCHead 保持
  `training=True` 输出 raw logits，与现有 `set_validation_mode` 语义一致）；
- Teacher 不进入 optimizer、不进入 checkpoint state_dict、不参与 observer/fake-quant。

### 2.3 KD loss（多层）

参考 `references/PytorchOCR` 的 `key` + `model_name_pairs` 机制（每个 KD loss 指定从
`return_all_feats` 输出 dict 中消费哪个键），KD 按层独立配置：

| 层（key） | rec（pretrained_train） | det（training） | 默认权重 |
| --- | --- | --- | --- |
| `backbone_out` | backbone 特征 `[B, C, H, W]` | backbone 多级特征 | 0（关闭） |
| neck 特征 | `ctc_neck` `[B, T, C]` | `fuse` `[B, C, H, W]` | 0（关闭） |
| `head_out`（输出头） | CTC logits `[B, T, C_cls]` | shrink/threshold/binary maps | `kd_weight`（1.0） |

- **输出头 KD**（默认启用）：
  - rec `--kd-mode logits`：温度 KL
    `loss = T² · mean(KL(softmax(s/T) ‖ softmax(t/T)))`，`use_log` 取 log-softmax 数值稳定形式；
    另支持 `--kd-mode logits_mse`：`mean((s - t)²)`；
  - det `--kd-mode maps`：对 shrink/threshold/binary 三 map 做 MSE（逐 map 等权）；
- **中间层 KD**（默认关闭）：每层独立可配置权重（如 `kd_backbone_weight`、`kd_neck_weight`，
  默认 0），loss 类型 rec 用 L2/MSE（参考 v3 rec 的 `DistillationDistanceLoss key=backbone_out`），
  det 用 MSE 或 KL（参考 CML 的 `thrink_maps` DML + KL）；
- `--kd-weight`（默认 1.0）：
  `total = task_loss + Σ_layer kd_weight_layer · kd_loss_layer`；
- **实验顺序**：先输出头 KD 验证有效性，再按单变量规则逐层启用中间层 KD。

### 2.4 与现有合同兼容

- checkpoint resume：KD 合同键（`kd`、`kd_layers`、`kd_mode`、`kd_weight`、
  `kd_temperature`、`teacher_model_config`）加入 resume 校验，变化必须重新 prepare；
- epoch-2 精度门禁、observer 生命周期、dynamic heights、amp 规则全部保持；
- QAT + KD：Teacher 不参与 PT2E 捕获，prepared 图只含 Student，导出/验收合同不变；
- 中间层 KD 依赖 `return_all_feats`：仅训练期消费，部署投影（det 单 shrink、rec 单 CTC）
  不受影响。

## 3. 执行顺序与门禁

1. **S1 KD loss**：`kd.py`（输出头 + 中间层 + 复合聚合）+ 单元测试（固定输入数值检查、backward）；
2. **S2 中间层暴露 + Trainer 集成**：扩展 `FullRecTrainingWrapper` / `DetTrainingWrapper`
   暴露 `backbone_out` / `fuse`（回归测试确认输出头语义不变）；Trainer 增加 teacher 前向、
   kd 逐层合并、evaluate 报告各层 kd_loss；单测覆盖 train_step 含 kd_loss、Teacher 无梯度、
   checkpoint 不含 teacher；
3. **S3 训练入口**：`tools/train.py` CLI（`--kd-layers`、各层权重）+ teacher 构建 +
   `return_all_feats` 注入 + metadata；profile 合同扩展；
4. **S4 回归**：全量 `pytest -q tests`，单模型（无 KD）行为不变；
5. **S5 smoke**：v6 rec / v6 det 各跑 KD 随机输入 smoke（float + QAT），验证
   输出头 KD 与中间层 KD 均有限、QAT observer 不受 teacher 影响；
6. **S6 文档**：更新本文档状态 + `docs/axera_qat/guides/kd_training.md`（命令与结果记录）。

门禁：每阶段测试通过后才进入下一阶段；S4 前不跑真实数据训练。中间层 KD 仅在输出头 KD
smoke 通过后按单变量规则逐层启用。

## 4. 产物清单

```text
pytorchocr/training/losses/kd.py
pytorchocr/quantization/bridge.py（wrapper 中间层暴露扩展）
pytorchocr/training/trainer.py（teacher + kd 集成）
tools/train.py、pytorchocr/training/profile.py（KD 入口与合同）
tests/test_kd_losses.py
tests/test_trainer_kd.py
tests/test_wrapper_kd_outputs.py（wrapper 中间层暴露回归）
configs/qat/training/（后续真实实验新增 kd profile）
docs/axera_qat/guides/kd_training.md
```

## 5. 停止条件

- KD loss 数值与参考公式不一致且无法定位；
- Teacher 输出结构与 Student 不对齐（形状/语义错误）导致 kd_loss 恒非有限；
- QAT + KD 下 observer/qparam 行为被 teacher 干扰（prepared 图结构变化）；
- 无 KD 的单模型回归测试失败。

## 6. 待确认项（已确认）

1. KD 范围：v5/v6 float + QAT，Teacher 固定浮点 —— 已确认；
2. 输出头 KD：det 三 map MSE、rec CTC logits 温度 KL —— 已确认；
3. 多层中间特征 KD：`backbone_out` / neck 特征（`ctc_neck`、`fuse`）按层独立配置、
   默认关闭，先输出头后中间层的实验顺序 —— 已确认；
4. Teacher 权重默认取 Student 浮点权重（`--weights`），不显式指定 `--teacher-weights` —— 已确认；
5. 不做 v3/v4 蒸馏模型转换、不改 references —— 已确认。

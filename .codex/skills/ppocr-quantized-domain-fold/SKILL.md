---
name: ppocr-quantized-domain-fold
description: 将已训练的非重参化（多分支）PP-OCR QAT checkpoint 折叠为推理单分支模型并继续 finetune。使用场景：exp9 式非重参化 QAT 训练完成后需要部署单分支结构、QAT JSON 或 qspec 变更后需要从既有量化域权重重新开始、或需要对比折叠前后精度时。包含量化域权重提取、折叠后 finetune、QuantONNX 导出与验收。
---

# PP-OCR Quantized-Domain Folding

将"训练非重参化、推理重参化"方案固化为可复用工作流。核心结论：**expN 非重参化
QAT checkpoint 的权重是 LSQ 量化域耦合的**，裸权重（剥离 `activation_post_process`
后当浮点权重）评估 acc=0.0，必须做量化域折叠后才能作为折叠模型的浮点起点。

## 何时使用

- 非重参化 QAT 训练（exp9 式）完成，需要导出推理单分支模型并保持精度；
- qspec/QAT JSON 变更后需要重新训练（起点 = 量化域折叠权重，而不是裸权重）；
- 对比"折叠前后精度"、检查 QuantONNX 结构（identity/requant 归属）。

## 流程总览

```text
非重参化 QAT checkpoint (best.pt)
  ├─ [可跳过] 折叠前精度核验: eager 加载裸权重 acc 应≈0（量化域耦合证明）
  ├─ 量化域折叠: 对每分支取 fq(fold(w)) 求和 → folded_state.pt
  ├─ 量化域起点 eval（--qat-config, 随机数据 observer 初始化）: acc 相对训练 best 差值应 ≤ 0.10
  ├─ 折叠模型 prepare（新 qspec）→ smoke 门禁（loss 相对训练末值增量应 < 1.5 倍，且不发散）
  ├─ finetune（epoch1 val_acc 相对训练 best 差值应 ≤ 0.05）
  └─ convert → QuantONNX → eval.py 核验 + QDQ 结构审计
```

## 入口(2026-08-13 常态化迁移)

折叠 finetune 已提升为常态化流程,主入口与公共 API:

- **CLI**: `tools/finetune_folded.py`(两种模式:仅提取+eval、提取+finetune)
- **公共 API**: `pytorchocr/quantization/folding.py`
  (`build_folded_state` / `apply_folded_state` / `folded_eager_model` /
  `checkpoint_qat_config` / `collect_quantized_weight_map`)
- **测试**: `tests/test_folded_finetune.py`

本 skill 的 `scripts/` 目录现为上述 CLI 的薄封装,保持历史命令可用
(`fold_quantized_domain.py` 的 `--checkpoint/--output` 映射到
`--source-checkpoint/--fold-state-output`)。

## 1. 量化域折叠权重提取

脚本：`tools/finetune_folded.py`（从非重参化 checkpoint 的 prepared 图提取）。

```bash
env PYTHONPATH="$PWD" python tools/finetune_folded.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --source-checkpoint runs/expN_.../best.pt \
  --fold-state-output /tmp/folded_state.pt
```

要点：
- 每个 conv 分支权重取 `fq(mul(w, bn/norm_fold_scale))`（per-channel，含 clip）；
- 多分支按 `_get_kernel_bias` 规则在**量化域求和**（conv_kxk pad 到 kxk、1x1 pad、
  identity id_tensor）；
- 覆盖 ConvBNLayer（区分 `bn`/`norm` 命名）、Linear 权重、SE 裸 Conv2d（含 bias fq）；
- 输出 state 可直接 `strict=True` 加载到 `reparameterize_for_deploy` 后的模型；
- 重建 checkpoint 图时自动使用训练时归档的 qspec 副本（metadata `copied_configs`）。

量化域起点验证（finetune 前的有意义指标，推荐）：

```bash
env PYTHONPATH="$PWD" python tools/finetune_folded.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --source-checkpoint runs/expN_.../best.pt \
  --fold-state-output /tmp/folded_state.pt \
  --val-label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --val-data-dir /home/heqi/dataset/icdr \
  --training-profile configs/qat/training/....yml \
  --qat-config configs/qat/....json \
  --device cuda:N
```

带 `--qat-config` 时执行 prepare + convert + 验证集评估。observer 统计初始化
用**随机数据**（只解决 fake quant scale/zero_point 形状初始化，不依赖训练
数据；折叠权重与 checkpoint 的量化形状不同，直接加载会报 shape 错误）。
输出的是**折叠 finetune 的真实起点精度**。验收判据用**与源训练 best 的
差值**，不写死绝对值：

- 量化域起点 acc 与训练 best val_acc 的差值:随机数据 observer 初始化应
  ≤ 0.10,真实数据统计初始化应 ≤ 0.05(exp12a 实测:随机 0.558 vs 训练
  0.617 差值 0.06;真实数据 0.605 差值 0.01);
- 裸浮点前向（不带 `--qat-config`）acc 与训练差值可能很大（exp12a 折叠
  仅 0.26），属正常现象，不代表折叠失败，不能作为验收依据。

## 2. 折叠模型 finetune

脚本：`tools/finetune_folded.py`（源自 exp10a/exp11 的 fold_finetune 脚本）。

```bash
env PYTHONPATH="$PWD" python tools/finetune_folded.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --label-file /home/heqi/dataset/icdr/rec_gt_train.txt \
  --data-dir /home/heqi/dataset/icdr \
  --val-label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --val-data-dir /home/heqi/dataset/icdr \
  --output-dir runs/expN_folded_finetune \
  --training-profile configs/qat/training/....yml \
  --source-checkpoint runs/expN_.../best.pt \
  --folded-state /tmp/folded_state.pt \
  --device cuda:N
```

流程：source checkpoint 浮点权重 → eager pretrained_train → `reparameterize_for_deploy`（裸单分支）
→ 覆盖 folded_state → `FullRecTrainingWrapper` prepare（LSQ）→ smoke → finetune。

优化器：仅允许 **AdamW 或 SGD**（profile `optimizer: AdamW|SGD`）。禁止 `Adam`——
其 L2 weight decay 与 LSQ 梯度缩放（`use_grad_scaling`）叠加会错误衰减 learnable
scale/zero_point，早期发散。折叠 finetune 是量化域微调，lr 应低于浮点训练
（exp12b 用 lr 1e-5、warmup 2、AdamW）。

验收标准（用相对源训练 best 的差值，不写死绝对值）：
- **smoke 门禁**：10 步 loss 相对训练末值应无明显放大（增量 < 1.5 倍），且
  不出现发散趋势（连续步 loss 递增/非有限值）。若 loss 远高于训练末值（如
  数量级放大），说明起点仍是错误域（裸权重）；
- **epoch1 val_acc**：与源训练 best val_acc 的差值应 ≤ 0.05；若差 > 0.1，
  说明折叠起点或 qspec 未对齐（如 scale Mul 条目未命中、量化域不匹配）；
- checkpoint metadata 需记录 `folded_state` 路径与 source checkpoint。

## 3. QuantONNX 导出与验收

折叠 checkpoint 是 `pretrained_train` 图（含 gtc_head），导出用部署投影。
**默认导出 batchSize=1**（静态 batch 1，部署推理形态）；训练图核对等特殊场景才用
其他 batch。

```bash
env PYTHONPATH="$PWD" python tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/expN_folded_finetune/best.pt \
  --output /tmp/expN_folded_qdq.onnx \
  --batch-size 1
```

或复用 exp10 的独立导出脚本（`/tmp/opencode/exp10_export.py` 思路：prepare 参数与
训练一致 + `RecTrainingDeploymentProjection` 投影 + `export_onnx(optimize=True)`；
导出输入用 batch 1 的 `export_images`，ONNX 输入即静态 batch 1）。

验收：
- QDQ 审计：Identity 应只剩 SE avg_pool 类 u16→u16 边界（确认不处理），
  QK^T/softmax·V 的 MatMul 输入全 S16，matmul2 输出 U16；
- ORT 全量 eval（`tools/eval.py --onnx ...`）与训练 val_acc 差值应 ≤ 0.03
  （exp12a 实测 0.616 vs 训练 0.617）；
- 若出现多余 identity/requant，先查 qspec（见
  `ppocr-qat-config-discovery` skill 与 docs/axera_qat/records/reparameterization_qat_issues.md 问题 10）。

## 关键陷阱

1. **裸权重不可用**：非重参化 checkpoint 权重必须过 fake quant 才正确（量化域
   耦合）。剥离 quantization 参数直接折叠会导致 epoch1 acc 接近 0、smoke loss
   数量级放大——判据是相对源训练 best 的差值，而非绝对值。
2. **output 字段需 quantizer 支持**：`ax_quantizer_lsq.py get_config` 已修复支持
   `output`/`output_is_symmetric`；旧 checkpoint 与新 qspec 图不兼容是预期的，
   必须重新 prepare，不得跨图恢复。
3. **分支命名差异**：ConvBNLayer 的 norm 属性可能叫 `bn`（backbone）或 `norm`
   （ctc_encoder），折叠脚本必须两者都处理。
4. **proj/qkv/mlp/ctc_head.fc 是 FC 性质**：MatMul 输入维持 U16 域（Axera 约束只
   针对真正的 attention MatMul），不要为它们加 S16 条目（会引入 requant）。
5. **checkpoint 与 qspec 版本绑定**：折叠脚本用 `build_prepared_qat_checkpoint`
   strict 重建 checkpoint 图。若 checkpoint 由修改前的 qspec/quantizer 训练（如
   exp9、或 qspec 文件后续改动），当前配置重建会报 missing/unexpected。
   脚本已自动优先使用训练时归档的 qspec 副本（metadata `copied_configs`，
   run 目录内），避免此问题；若仍失败，须使用当时归档的 folded state
   （exp9 的已存于 `cache/exp10a_fold_finetune_scripts/` 关联路径）。
6. **纯浮点 eval 不代表 finetune 起点**：折叠权重携带分支级 fake-quant 的
   clip 语义，裸 deploy 浮点前向精度与训练可能差很多（exp12a 折叠实测
   0.26 vs 训练 0.617，差值 ~0.36）是正常现象，不代表折叠失败。有意义的
   起点指标是 `--qat-config` 下 prepare+convert 的量化域 eval，其与训练
   best 的差值应 ≤ 0.05。
7. **qspec 节点名随图形态变化**：非重参化训练图与折叠后训练图的 scale Mul
   节点名不同（exp12a: 训练图 `mul_352/353` vs 折叠图 `mul_60/61`）。若两者
   都要用同一 qspec，条目须同时包含两组名字（`module_names` 可多值），否则
   未命中的图退回 global 域并产生 requant identity。

## 相关文档

- `docs/axera_qat/plans/train_noreparam_infer_reparam.md`：方案与决策记录；
- `docs/axera_qat/records/reparameterization_qat_issues.md`：问题 9（量化域耦合）、
  问题 10（output 字段失效）、问题 11（qspec 节点名随图形态变化）；
- `docs/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md` §32-39：
  exp9/exp10/exp10a/exp11/exp12/exp12a/exp12b 实验记录。

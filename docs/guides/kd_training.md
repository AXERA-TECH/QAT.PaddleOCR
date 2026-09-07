# KD 蒸馏训练指南

当前训练框架支持 v5/v6 det/rec 的浮点与 QAT 知识蒸馏(KD)：冻结浮点 Teacher，可训练 Student
在输出头(CTC logits / DB maps)与可选中间层(backbone_out / neck 特征)上对齐。

KD 是可选训练能力，不属于当前 v6 浮点/QAT 默认复现路线。启用前应先完成不带 KD 的浮点或 QAT
基线，并单独记录 KD 配置和验证集指标。Teacher 与 Student 目前需要具有可对齐的输出或特征形状，
框架不会自动增加跨结构的特征投影层。

## 1. 启用方式

CLI 参数（优先级高于 profile）：

```text
--kd                         启用 KD（需要 --weights 提供 Teacher 浮点基线）
--teacher-weights            可选，默认取 --weights（Student 浮点权重）
--teacher-model-config       可选，默认取 --model-config
--kd-mode                    rec: logits（默认，温度 KL）/ logits_mse
                             det: maps（默认，三 map MSE）
--kd-weight                  输出头 KD 总权重（默认 1.0）
--kd-temperature             温度 KL 的 T（默认 4.0）
--kd-neck-weight             中间层 neck 特征 KD 权重（默认 0，关闭）
--kd-backbone-weight         中间层 backbone 特征 KD 权重（默认 0，关闭）
```

也可在 training profile 的 `training:` 下声明同名字段（`kd: true`、`kd_mode`、
`kd_weight`、`kd_temperature`、`kd_neck_weight`、`kd_backbone_weight`、
`teacher_weights`、`teacher_model_config`)。

示例(浮点 rec KD):

```bash
python3 tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights weights/ptocr_v6_small_rec_full.pth \
  --label-file /path/to/rec_train.txt \
  --val-label-file /path/to/rec_val.txt \
  --data-dir /path/to/dataset \
  --output-dir runs/kd_rec \
  --kd --kd-mode logits --kd-weight 1.0 --kd-temperature 4.0 \
  --epochs 20 --batch-size 32
```

示例(QAT det KD,带中间层):

```bash
python3 tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights weights/ptocr_v6_small_det_full.pth \
  --qat-config configs/qat/ppocrv6_small_det_u8s8.json \
  --label-file /path/to/det_train.txt \
  --val-label-file /path/to/det_val.txt \
  --data-dir /path/to/dataset \
  --output-dir runs/kd_det_qat \
  --kd --kd-weight 1.0 --kd-neck-weight 0.5 \
  --training-profile configs/qat/training/ppocrv6_small_det_baseline.yml
```

## 2. 训练合同

- Teacher 冻结(`requires_grad=False`)，使用 eval/BN running-stat 语义；CTCHead 保持 raw logits。
- Teacher 不进入 optimizer、checkpoint `state_dict`、PT2E 图或 observer/fake quant。
- 输出头 KD 默认启用；中间层 KD 默认关闭（`--kd-neck-weight`/`--kd-backbone-weight` 默认为 0）。
- 中间层 KD 需要暴露中间特征：QAT 图在 prepare 前设置 wrapper `expose_intermediates`，浮点图设置
  模型 `return_all_feats=True`。
- KD 开关或任意 KD 权重变化后，必须从浮点权重重新 prepare，不能跨训练合同恢复旧 checkpoint。resume
  合同包含 `kd`、`kd_mode`、`kd_weight`、`kd_temperature`、`kd_neck_weight`、
  `kd_backbone_weight` 和 `teacher_model_config`。
- 训练损失为 `total = task_loss + kd_weight · Σ_layer kd_weight_layer · kd_loss_layer`。
- 每层 KD loss 会单独上报（如 `kd_ctc`、`kd_maps`、`kd_neck_out`），`kd_loss` 为加权和。
- v5 rec 全训练图（GTC/NRTR）的 KD 只消费 CTC logits 与中间层，不对 GTC 输出做 KD。

## 3. 输出结构对照

| 训练模式 | Student 输出 | KD 可用层 |
| --- | --- | --- |
| rec deploy(浮点/QAT) | CTC logits tensor | `ctc` |
| rec pretrained_train(浮点) | BaseModel return_all_feats dict，结构为 `{backbone_out, neck_out(=ctc_neck), head_out{ctc, gtc, ctc_neck}}`，由 `normalize_outputs` 拆出 | `ctc`、`ctc_neck`、`backbone_out`、`gtc`(不 KD) |
| rec pretrained_train(QAT) | FullRecTrainingWrapper dict(expose) | `ctc`、`ctc_neck`、`backbone_out`、`neck_out` |
| det training(浮点) | return_all_feats dict | `maps`、`neck_out`、`backbone_out` |
| det training(QAT) | DetTrainingWrapper dict(expose) | `maps`、`neck_out`、`backbone_out` |

## 4. KD checkpoint 的 QuantONNX 导出

KD 训练改变了 prepared 图输出结构（wrapper `expose_intermediates` 的 dict 输出），导出链路
已适配：

- `build_prepared_qat_checkpoint` 从 checkpoint metadata 的 `kd` 字段决定
  `expose_intermediates`，严格重建与训练一致的图；
- `RecTrainingDeploymentProjection` / `OutputSelector` / `outputs_as_tuple` 支持
  dict 输出（rec 取 `ctc`、det 取 shrink）；
- 导出命令与普通 checkpoint 相同：

```bash
python3 tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/<kd_run>/debug_epoch_0002.pt \
  --output /tmp/<kd_run>_debug_qat.onnx
```

非重参数化模型可能因 BatchNorm 等算子不满足部署工具链约束而无法直接作为 Axera 交付件；应先按
主流程完成重参数化、QDQ 结构检查和 ORT 验证，再进行 Pulsar2/AXModel 验收。KD 训练不会改变
QuantONNX 的部署输出，Teacher 也不会被导出。

## 5. 验证要求

KD 训练完成后，应按照主流程同时评估 Student 与不带 KD 的基线，并记录：

- 训练和验证集上的 task loss、KD loss 及任务主指标；
- Student 在 exported float、prepared fake-off、prepared fake-on、converted PT2E 和 QuantONNX
  各阶段的输出及任务指标；
- 识别模型的 CTC logits、序列准确率和 normalized edit distance，或检测模型的 shrink map、
  precision、recall 和 hmean；
- QAT 模型的 QuantONNX 结构、ORT 语义基线和目标部署平台结果。

KD 是否带来收益必须以相同数据、预处理、Student 初始权重和 QAT 配置下的对照实验为准，不能用
不同模型或不同数据集的历史结果直接推断。

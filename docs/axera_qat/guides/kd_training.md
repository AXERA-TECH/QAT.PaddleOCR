# KD 蒸馏训练指南

v5/v6 det/rec 的浮点与 QAT 训练支持知识蒸馏(KD):冻结浮点 Teacher,可训练 Student 在
输出头(CTC logits / DB maps)与可选中间层(backbone_out / neck 特征)上对齐。实现见
`docs/axera_qat/plans/distillation_support_plan.md`。

## 1. 启用方式

CLI 参数(优先级高于 profile):

```text
--kd                         启用 KD(需 --weights 提供 teacher 浮点基线)
--teacher-weights            可选;默认取 --weights(student 浮点权重)
--teacher-model-config       可选;默认取 --model-config
--kd-mode                    rec: logits(默认,温度 KL) / logits_mse
                             det: maps(默认,三 map MSE)
--kd-weight                  输出头 KD 总权重(默认 1.0)
--kd-temperature             温度 KL 的 T(默认 4.0)
--kd-neck-weight             中间层 neck 特征 KD 权重(默认 0,关闭)
--kd-backbone-weight         中间层 backbone 特征 KD 权重(默认 0,关闭)
```

也可在 training profile 的 `training:` 下声明同名字段(`kd: true`、`kd_mode`、
`kd_weight`、`kd_temperature`、`kd_neck_weight`、`kd_backbone_weight`、
`teacher_weights`、`teacher_model_config`)。

示例(浮点 rec KD):

```bash
env PYTHONPATH="$PWD" "$PYTHON" tools/train.py \
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
env PYTHONPATH="$PWD" "$PYTHON" tools/train.py \
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

## 2. 合同

- Teacher 冻结(`requires_grad=False`)、eval/BN-running 语义、CTCHead 保持 raw logits;
- Teacher 不进入 optimizer、checkpoint state_dict、PT2E 图或 observer/fake-quant;
- 输出头 KD 默认启用;中间层 KD 默认关闭(`--kd-neck-weight`/`--kd-backbone-weight` 默认 0);
- 中间层 KD 需要中间特征暴露:QAT 图在 prepare 前设置 wrapper `expose_intermediates`,
  浮点图设置模型 `return_all_feats=True`;因此 **KD 开关/中间层权重变化后必须从浮点权重
  重新 prepare,不得跨合同恢复旧 checkpoint**(resume 合同含 `kd`/`kd_mode`/`kd_weight`/
  `kd_temperature`/`kd_neck_weight`/`kd_backbone_weight`/`teacher_model_config`);
- 训练 loss:`total = task_loss + kd_weight · Σ_layer kd_weight_layer · kd_loss_layer`;
- 每层 kd loss 单独上报(`kd_ctc`、`kd_maps`、`kd_neck_out` 等),`kd_loss` 为加权和;
- v5 rec 全训练图(GTC/NRTR)的 KD 只消费 CTC logits 与中间层,不做 GTC 输出 KD
  (理由见计划 §1.1)。

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
env PYTHONPATH="$PWD" "$PYTHON" tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/<kd_run>/debug_epoch_0002.pt \
  --output /tmp/<kd_run>_debug_qat.onnx
```

- 非重参数化 KD 训练图（如 v5 rec `reparameterize: false`）保留 BN，QDQ 门禁会记录
  `{"status": "blocked", "error": "ONNX graph contains BatchNormalization nodes."}`，
  ONNX 文件仍生成供结构检查（与 exp3/exp4 诊断一致），不作为 Axera 交付。

## 5. 结果记录

2026-08-10 S5 smoke(随机输入,train_step 单步):

```text
float rec deploy KD (v6 rec):  loss/kd_ctc 有限,teacher 冻结,KD 可反向
float det KD (v6 det):         kd_maps=0.1035,kd_neck_out/kd_backbone_out 按权重,
                               loss 有限
QAT rec pretrained_train (v6): CTCLoss+NRTRLoss+kd_ctc+kd_ctc_neck 均有限
```

2026-08-10 Exp4(v5 rec QAT + KD,exp3 基础上单变量):

```text
epoch 2: val_acc=0.3717(drop 0.222 > 0.10 门禁停止)
对比: exp3(无 KD)=0.3899, exp4(KD)=0.3717
结论: 输出头 KD(weight=1.0)未恢复精度,KD loss 量级(~70-99)主导了 CTC 梯度;
      主因仍是未修复的 U16 observer eps=2**-12 截断,KD 无法弥补量化噪声
```

正式实验指标、checkpoint、QuantONNX 与精度记录随实验写入 `docs/axera_qat/records/`。

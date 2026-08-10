# 数据增强与多尺度训练

## 1. 实现来源

检测和识别增强参考 `references/PytorchOCR/torchocr/data/` 迁移，生产代码不依赖只读
`references/`。对应来源：

```text
references/PytorchOCR/torchocr/data/imaug/iaa_augment.py
references/PytorchOCR/torchocr/data/imaug/copy_paste.py
references/PytorchOCR/torchocr/data/imaug/random_crop_data.py
references/PytorchOCR/torchocr/data/imaug/rec_img_aug.py
references/PytorchOCR/torchocr/data/imaug/text_image_aug/
references/PytorchOCR/torchocr/data/multi_scale_sampler.py
```

当前实现位于：

```text
pytorchocr/training/data/augmentation.py
pytorchocr/training/data/text_image_aug/
pytorchocr/training/data/sampler.py
```

参考实现的 det `IaaAugment` 依赖 `imgaug`，当前已验证环境没有该包。项目使用 OpenCV 实现同一组
PP-OCRv5 几何合同，不增加运行时依赖：随机水平翻转、`[-10,10]` 旋转、`[0.5,3]` 缩放、文本安全
随机裁剪。CopyPaste 保留外部样本和非重叠粘贴行为。rec 的 TIA MLS warp、RecAug 概率和 RecConAug
拼接约束按 PytorchOCR 迁移。

## 2. 训练合同

训练 profile 和 CLI 使用：

```yaml
training:
  augmentation: none       # none 或 paddle
  dynamic_heights: [32, 48, 64]
  multi_scale_training: true
```

```bash
tools/train.py \
  --augmentation paddle \
  --dynamic-heights 32 48 64 \
  --multi-scale-training
```

`augmentation=paddle` 对 det 启用 CopyPaste、几何变换和随机裁剪；对 rec 启用 RecConAug 和 RecAug。
验证集始终使用确定性预处理，不读取训练增强 preset。`augmentation` 和
`multi_scale_training` 均写入 checkpoint metadata，并参与 strict resume 合同检查。

QAT baseline 默认且显式使用 `augmentation: none`。PP-OCRv5 mobile rec 是一个单独合同：随机增强
关闭，但 `multi_scale_training: true`，DataLoader 实际产生 H=32/48/64 的 batch。以 H=48、batch
64 为基准且 `fix_batch_size=false` 时，三档 batch size 分别为 96、64、48，宽度固定 320。
QuantONNX 仍静态导出 `[1,3,48,320]`。

## 3. 数据顺序

det：

```text
decode + polygon
-> optional CopyPaste
-> optional flip/rotate/scale
-> optional text-safe crop to target shape
-> deterministic letterbox if still required
-> DB border/shrink maps
-> normalize
```

rec：

```text
decode + text
-> optional RecConAug and label concatenation
-> optional RecAug
-> re-encode CTC/NRTR labels
-> resize to sampled H and right-pad to W
-> normalize
```

多尺度 sampler 以 `(width, height, index)` 访问 RecognitionDataset，确保一个 batch 内 shape 一致。
增强后标签无法编码时立即抛错，不采用参考实现递归换样本的静默恢复策略。

## 4. 验证

定向测试覆盖：

- det 图像与 polygon 同步变换、默认 paddle preset 和 DB maps；
- rec 拼接、增强后标签重编码和默认 paddle preset；
- H=16/32/48 测试 batch 的真实 DataLoader shape；
- profile 值检查以及 augmentation/multi-scale resume mismatch。

实现阶段定向结果：`34 passed`。使用 `/home/heqi/dataset/icdr/rec_gt_train.txt` 的真实 DataLoader
检查得到：

```text
H=32: [96, 3, 32, 320]
H=48: [64, 3, 48, 320]
H=64: [48, 3, 64, 320]
```

项目全量测试为 `151 passed, 14 subtests passed`。正式浮点复现或 QAT 实验仍需分别记录启用 preset
后的任务指标，不得将“增强代码可运行”视为训练精度已对齐 PaddleOCR。

# QAT.YOLO 方法迁移记录

## 2026-07-29：首轮迁移

### 迁移目标

将 `/home/heqi/project-qat/ultralytics` 中已经验证的通用 QAT 方法迁移到 PP-OCR 路线，不复制
YOLO head、Attention 节点名、KD loss 或近似 ONNX qparam 合并逻辑。

### 项目 Skill

新增项目级 skill：

```text
/home/heqi/project/PaddleOCR/.codex/skills/ppocr-pt2e-qat/
  SKILL.md
  agents/openai.yaml
  references/acceptance.md
```

从 `pt2e-accuracy-check` 和 `yolo26-qat-delivery` 迁移以下流程：

1. eager float -> exported float -> prepared fake-off -> prepared fake-on -> converted ->
   QuantONNX -> Axera 分层对齐；
2. exported/prepared 图的 BatchNorm training、momentum、eps 检查；
3. checkpoint、模型图、量化 JSON、training profile 严格对应；
4. 区域配置按 `source_fn_stack` 和拓扑发现，禁止复用旧 FX 节点编号；
5. ONNX 只删除 qparam 完全一致的冗余 DQ/Q，不迁移 2% 近似合并。

PP-OCR 专用输出契约：

- det：先比较 shrink/threshold/binary maps，再比较 DBPostProcess 的 precision/recall/hmean；
- rec：先比较 CTC logits，再比较 sequence accuracy/normalized edit distance。

### Training Profile

新增：

```text
configs/qat/training/ppocrv6_small_det_baseline.yml
configs/qat/training/ppocrv6_small_rec_baseline.yml
pytorchocr/training/profile.py
```

`tools/train.py` 新增 `--training-profile`，优先级为：

```text
显式 CLI > training profile > Paddle 模型 YAML
```

baseline 从 QAT.YOLO 迁移 `50 epochs / lr0=2e-5 / lrf=0.1 / noEMA / observer 全程更新`。
PP-OCR 继续使用模型 YAML 中的 Adam、weight decay 和 2 epoch warmup。profile 固定 det
`3x640x640`、rec `3x48x320`，使用 FP32 QAT，不设置 observer freeze，不启用随机增强。

training profile 绑定对应 Axera 量化 JSON。checkpoint metadata 新增 profile/量化 JSON 的路径和
SHA256、Torch 版本、总 epoch、warmup、cosine 最终比例和 observer 策略。

resume 会主动比对 task、模型 YAML、training profile/量化配置 SHA256、Torch 版本、shape 和
reparameterize 状态。量化图契约不一致时直接拒绝恢复，不能依赖宽松 checkpoint 加载。

### Scheduler 适配

原训练代码 cosine 最终衰减到 0。新增 `--lr-final-factor` 后：

```text
lr(epoch_end) = initial_lr * lr_final_factor
```

baseline 设置为 `0.1`，即 `2e-5 -> 2e-6`。未使用 training profile 时默认值仍为 `0.0`，保持
旧浮点训练行为。

### 数据策略

本项目 baseline 不迁移 YOLO/Paddle 的随机增强。det 使用保持宽高比的确定性居中 letterbox，padding
像素值为 114；训练和 validation 使用相同 shape contract。此项与早期“补齐 Paddle 随机增强”的建议
不同，相关路线文档已同步修正。

### 验证

新增 profile 解析、未知字段拒绝、det/rec contract 和 cosine 最终比例测试。首轮执行全部测试：

```text
24 tests passed
```

迁移后仍未完成：真实数据 1 epoch strict reload/QuantONNX/Axera 闭环、DBPostProcess/DetMetric、
自动化 exported/prepared/converted 数值报告。

## 2026-07-29：ICDAR2015 rec QAT smoke

### 数据解压与验收

将 `/home/heqi/dataset/icdr` 下 5 个 ZIP 解压到标签约定目录：

```text
icdar_c4_train_imgs/  1000 ICDAR det train images
ch4_test_images/       500 ICDAR det test images
train/                4468 ICDAR rec train images
test/                 2077 ICDAR rec test images
ctw1500/imgs/         1000/500 CTW1500 train/test images
```

全部 ZIP 完整性检查通过。ICDAR det 的图片、标签、四边形和边界检查通过；rec 图片与标签一一
对应，长度不超过 25，train/test 无内容重复。`rec_gt_train.txt:2917` 存在一条历史转换异常：
原始 GT 的连续空格被转换为 Tab，当前 encoder 会跳过 Tab 并得到 `relles`。本次 smoke 保留原标签，
正式识别训练前需要单独清洗。

### Smoke 运行

使用 `ppocrv6_small_rec_baseline.yml`，仅通过 CLI 覆盖：

```text
epochs=1
workers=0
device=cuda:3
batch_size=64（沿用 profile）
observer_freeze_epoch=None
```

第一次运行使用 profile 的 `workers=8`，受限沙箱禁止 DataLoader worker 创建资源共享 socket；第二次
使用 `workers=0` 完成 PT2E prepare，但沙箱内 PyTorch CUDA 不可见。最终在沙箱外使用 GPU 3 成功
完成 69 个训练 step 和 33 个 validation batch：

```text
train loss_ctc: 0.6622695901
val loss_ctc:   0.6539707456
observers_frozen: false
```

产物：

```text
output/icdar2015_ppocrv6_small_rec_qat_smoke/best.pt
output/icdar2015_ppocrv6_small_rec_qat_smoke/last.pt
output/icdar2015_ppocrv6_small_rec_qat_smoke/epoch_0001.pt
```

### Strict Reload 与 QuantONNX

`tools/export_ocr_onnx.py checkpoint` 从 `best.pt` 严格重建 prepared graph 并导出：

```text
output/icdar2015_ppocrv6_small_rec_qat_smoke/best_qdq.onnx
size: 6,711,111 bytes
input:  [1, 3, 48, 320]
output: [1, 40, 18710]
nodes: 682
Q/DQ: 161/280
BatchNormalization: 0
Concat shared qparams: 1/1
unquantized Conv outputs: 0
```

导出优化删除了 8 组 qparam 完全一致的冗余 DQ/Q，最终 `redundant_dq_q=0`，保留 2 组真实
requantization。导出测试输入上 PyTorch/ORT 的 CTC argmax agreement 为 `1.0`，probability MAE 为
`3.01e-6`。后续正式 RecMetric 已补齐，该 checkpoint 在 2077 张 ICDAR test 图上的 sequence
accuracy 为 `0.7183437650`，normalized edit similarity 为 `0.8820551125`；仍只代表 1 epoch smoke。

## 2026-07-29：ICDAR2015 det 正式 QAT 启动

在 tmux 会话 `ppocrv6-det-qat-icdar` 中启动 PP-OCRv6 small det QAT：

```text
train: /home/heqi/dataset/icdr/train_icdar2015_label.txt（1000 images）
val:   /home/heqi/dataset/icdr/test_icdar2015_label.txt（500 images）
profile: configs/qat/training/ppocrv6_small_det_baseline.yml
device: cuda:3
epochs: 50
batch_size: 8
image_shape: [3, 640, 640]
learning_rate: 2e-5
lr_final_factor: 0.1
observer_freeze_epoch: None
```

输出和日志：

```text
output/icdar2015_ppocrv6_small_det_qat/
output/icdar2015_ppocrv6_small_det_qat/train.log
```

启动验收时已完成 4 epoch，每个 epoch 为 125 个训练 step，observer 均保持更新：

```text
epoch  train_loss  val_loss
1      3.23082     2.80676
2      2.69242     2.67062
3      2.53921     2.62145
4      2.46341     2.61487
```

GPU 3 显存占用约 5.3 GiB、利用率约 84%；`best.pt`、`last.pt` 和逐 epoch checkpoint 已正常
生成。训练启动时 metric 仍只有 DB loss；后续完成结果中的 DBPostProcess/DetMetric 为补充实现。

### 完成结果（2026-07-30 补记）

训练完成 50 epoch/6250 step，tmux 返回 `TRAIN_EXIT_CODE=0`。`best.pt` 为 epoch 11，validation
loss `2.5385760883`；`last.pt` 为 epoch 50。observer 全程未冻结。

从 `best.pt` 严格恢复并导出 `best_qdq.onnx`：761 nodes、Q/DQ `177/335`、BN=0、Concat 共享域
`2/2`、HardSigmoid 量化域 `13/13`、无未量化 Conv 输出或 DQ/Q requant。ONNX checker 和 ORT
通过。详细记录见 [ICDAR2015 检测 QAT 训练记录](../baselines/icdar2015_det_qat_training.md)。本机缺少
`pulsar2`，Axera 编译和板端 metric 尚待对应工具链环境完成。

同日补齐 DBPostProcess/DetMetric 后，epoch 11 `best.pt` 在 500 张 ICDAR test 图上得到 precision
`0.5764036958`、recall `0.3904670197`、hmean `0.4655568312`。该 checkpoint 是历史代码按最低 loss
选择；未来训练已改为按 hmean 选择。det/rec metric 细节见
[ICDAR2015 QAT Metric 验证记录](../baselines/icdar2015_qat_metric_validation.md)。

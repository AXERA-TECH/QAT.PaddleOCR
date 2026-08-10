# PaddleOCR PyTorch / Axera QAT 指南

本文档是本项目唯一的使用入口，覆盖环境安装、Paddle 权重转换、数据准备、浮点训练、QAT 配置与
训练、checkpoint 恢复、Float ONNX/QuantONNX 导出及 Axera/Pulsar2 验收。设计和实验记录见
[`docs/axera_qat/`](docs/axera_qat/readme.md)，各命令的完整参数以对应工具的 `--help` 为准。

## 1. 项目目标与流程

本项目将 PaddleOCR 的 PP-OCR 检测和识别模型转换为 PyTorch 模型，使用 PyTorch 2.6 PT2E 完成
Axera QAT，并导出带标准 ONNX Q/DQ 节点的 QuantONNX。

标准流程如下：

```text
Paddle YAML + .pdparams
-> 严格转换为完整 PyTorch training state dict
-> Paddle/PyTorch 浮点结构与精度检查
-> 从当前 export_for_training 图生成或检查 Axera QAT JSON
-> PT2E prepare/backward/convert/QuantONNX smoke
-> 人工检查 smoke QuantONNX 结构
-> 真实数据 QAT
-> 严格恢复 best.pt
-> convert_pt2e
-> QuantONNX
-> ONNX checker/QDQ/ORT/Axera-Pulsar2 验收
```

不要使用修改前图结构生成的 prepared checkpoint。模型代码、输入 shape、PyTorch 版本、部署重参数化
或 QAT qspec 变化后，必须重新从浮点权重开始 prepare 和训练。

## 2. 当前主要模型

| 模型 | Paddle YAML | PyTorch 权重 | QAT JSON | Training profile |
| --- | --- | --- | --- | --- |
| PP-OCRv5 mobile det | `configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml` | `weights/ptocr_v5_mobile_det.pth` | `configs/qat/ppocrv5_mobile_det_u8s8.json` | `configs/qat/training/ppocrv5_mobile_det_baseline.yml` |
| PP-OCRv5 mobile rec | `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml` | `weights/ptocr_v5_mobile_rec_full.pth` | `configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json` | `configs/qat/training/ppocrv5_mobile_rec_u16s16_sgd_dynamic_height.yml` |
| PP-OCRv6 small det | `configs/det/PP-OCRv6/PP-OCRv6_small_det.yml` | `weights/ptocr_v6_small_det_full.pth` | `configs/qat/ppocrv6_small_det_u8s8.json` | `configs/qat/training/ppocrv6_small_det_baseline.yml` |
| PP-OCRv6 small rec | `configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml` | `weights/ptocr_v6_small_rec_full.pth` | `configs/qat/ppocrv6_small_rec_u8s8.json` | `configs/qat/training/ppocrv6_small_rec_baseline.yml` |

检测训练图保留 DBHead 的 shrink、threshold、binary 和 Paddle 模型中的辅助监督，部署图只导出 shrink
map。识别完整训练图保留 CTC+NRTR/GTC MultiHead，当前 PT2E QAT 部署图使用 CTC 路径，导出原始 CTC
logits，不包含最终 Softmax。

## 3. 环境安装

所有命令都从仓库根目录执行：

```bash
cd /home/heqi/project/PaddleOCR
```

建议使用两个独立的 Python 3.10 环境，避免 Paddle 和 PT2E 导出依赖相互影响。

### 3.1 QAT、训练和导出环境

当前实际验证环境为：

```text
Python          3.10.19
PyTorch         2.6.0+cu118
ONNX            1.21.0
ONNX Runtime    1.21.0
ONNX Script     0.6.2
NumPy           1.26.4
OpenCV          4.8.1
PyYAML          6.0.3
Shapely         2.1.2
pyclipper       1.4.0
```

参考安装命令：

```bash
conda create -n ppocr-qat python=3.10 -y
conda activate ppocr-qat

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install \
  onnx==1.21.0 \
  onnxruntime==1.21.0 \
  onnxscript==0.6.2 \
  numpy==1.26.4 \
  opencv-python==4.8.1.78 \
  PyYAML==6.0.3 \
  shapely==2.1.2 \
  pyclipper==1.4.0 \
  pillow \
  rich \
  pytest
```

CUDA wheel 必须与机器驱动兼容。无 GPU 时可安装 PyTorch CPU wheel完成构图、smoke 和导出，但正式
QAT 训练应使用 CUDA 环境。

设置后续命令使用的解释器：

```bash
export PYTHON=/path/to/envs/ppocr-qat/bin/python
export PYTHONPATH="$PWD"
```

本机已有环境可直接使用：

```bash
export PYTHON=/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python
export PYTHONPATH="$PWD"
```

检查环境：

```bash
$PYTHON -c "import torch, onnx, onnxruntime, onnxscript; \
print(torch.__version__, onnx.__version__, onnxruntime.__version__, onnxscript.__version__)"
$PYTHON -m pytest -q tests
```

Axera quantizer 已放入 `pytorchocr/quantization`，不需要设置外部 `QAT.axera` 的 `PYTHONPATH`。

### 3.2 Paddle 权重转换环境

转换器需要同时导入 Paddle 和 PyTorch。当前验证组合为 Python 3.10、Paddle 3.0.0、PyTorch 2.6.0
和 NumPy 1.26.4：

```bash
conda create -n ppocr-convert python=3.10 -y
conda activate ppocr-convert

python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.0.0
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 opencv-python PyYAML
```

仅加载和转换权重时 Paddle CPU 包已经足够。以下命令用 `CONVERT_PYTHON` 表示转换环境：

```bash
export CONVERT_PYTHON=/path/to/envs/ppocr-convert/bin/python
```

本机已有环境为：

```bash
export CONVERT_PYTHON=/home/heqi/miniforge3/envs/ocr_moderation/bin/python
```

## 4. Paddle 到 PyTorch 模型转换

将官方 Paddle `.pdparams` 放入 `weights/`。转换必须使用与权重匹配的 YAML，并保留完整训练结构，
不要在转换阶段删除检测辅助头或识别 NRTR/GTC 分支。

### 4.1 PP-OCRv5 mobile recognition

```bash
env HOME=/tmp/ppocr_qat_home PYTHONPATH="$PWD" \
  "$CONVERT_PYTHON" converter/ppocr_v5_rec_converter.py \
  --yaml_path configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --src_model_path weights/PP-OCRv5_mobile_rec_pretrained.pdparams \
  --output weights/ptocr_v5_mobile_rec_full.pth
```

### 4.2 PP-OCRv5 mobile detection

当前 v5 det 转换器根据输入文件名生成 `ptocr_v5_mobile_det.pth`，输出到当前工作目录：

```bash
env HOME=/tmp/ppocr_qat_home PYTHONPATH="$PWD" \
  "$CONVERT_PYTHON" converter/ppocr_v5_det_converter.py \
  --yaml_path configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --src_model_path weights/PP-OCRv5_mobile_det_pretrained.pdparams

mv ptocr_v5_mobile_det.pth weights/ptocr_v5_mobile_det.pth
```

严格映射应只允许 PyTorch BatchNorm 的 `num_batches_tracked` 缺失。出现未知 Paddle 参数、普通模型
参数缺失、shape 不一致或 ignored source 时，不要继续 QAT。

### 4.3 PP-OCRv6 small recognition

`--save_mode full` 保留完整 CTC+NRTR/GTC 训练权重：

```bash
env HOME=/tmp/ppocr_qat_home PYTHONPATH="$PWD" \
  "$CONVERT_PYTHON" converter/ppocr_v6_rec_converter.py \
  --yaml_path configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --src_model_path weights/PP-OCRv6_small_rec_pretrained.pdparams \
  --save_mode full \
  --output weights/ptocr_v6_small_rec_full.pth
```

### 4.4 PP-OCRv6 small detection

```bash
env HOME=/tmp/ppocr_qat_home PYTHONPATH="$PWD" \
  "$CONVERT_PYTHON" converter/ppocr_v6_det_converter.py \
  --yaml_path configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --src_model_path weights/PP-OCRv6_small_det_pretrained.pdparams \
  --output weights/ptocr_v6_small_det_full.pth
```

### 4.5 转换后验证

转换成功只说明 state dict 可加载。至少继续执行 Paddle/PyTorch 浮点输出对齐：

```bash
$CONVERT_PYTHON tools/compare_rec_parity.py --help
$CONVERT_PYTHON tools/compare_det_parity.py --help
$CONVERT_PYTHON tools/compare_pretrained_training.py --help
$CONVERT_PYTHON tools/compare_det_training.py --help
```

部署浮点输出建议满足 cosine similarity 不低于 `0.9998`，同时记录 `max_abs` 和 MAE。完整训练图还要
检查全部输出 shape、loss 和梯度；不能仅验证单一 CTC 或 shrink 输出。

## 5. 数据格式

### 5.1 检测标注

检测标注每行由图片相对路径、Tab 和 polygon JSON 组成：

```text
images/img_001.jpg<TAB>[{"transcription":"text","points":[[12,16],[80,16],[80,48],[12,48]]}]
```

忽略文本使用 `"###"` 或 `"*"`。检测数据链路包含 OpenCV BGR 解码、居中 letterbox、polygon 同步
变换、DB shrink/threshold map 与 mask，以及 ImageNet mean/std 归一化。每个文字区域至少需要 3 个
polygon 点；train/val 应按原图划分，不能按 polygon 随机拆分。

### 5.2 识别标注

识别标注每行由图片相对路径、Tab 和文本组成：

```text
images/word_001.jpg<TAB>PaddleOCR
```

识别数据链路执行等高缩放、右侧 zero padding、`[-1, 1]` 归一化和 CTC 字典编码。字典、
`max_text_length` 和 `use_space_char` 从模型 YAML 读取。标签不能超过配置的最大长度，字符应存在于
对应字典中；全部字符未知的样本会被过滤。

`--data-dir` 与标注中的相对路径拼接。检测默认输入 `3x640x640`，识别默认输入 `3x48x320`。QAT
baseline 关闭随机增强；验证集始终禁用增强。训练入口支持 `--augmentation paddle`，识别可通过
`--multi-scale-training --dynamic-heights 32 48 64` 让 DataLoader 实际产生三档高度 batch。

### 5.3 数据目录与规模

推荐目录：

```text
/data/ppocr/
  det/
    images/train/
    images/val/
    train.txt
    val.txt
  rec/
    images/train/
    images/val/
    train.txt
    val.txt
```

工程验证至少需要足以覆盖一个完整 epoch 的独立 train/val 数据。检测可先使用 ICDAR2015 的 1000 张
训练图和 500 张验证图；识别建议至少准备 10,000 张训练裁剪图和 1,000 张验证裁剪图。正式微调应以
真实部署分布为主，检测通常需要 5,000 至 20,000 张带 polygon 的图片，识别通常需要 100,000 张以上
且覆盖实际字符、字体和退化分布。小型公开数据集只能验证工程链路，不能复现官方多语言精度。

## 6. 浮点训练

检测示例：

```bash
$PYTHON tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights weights/ptocr_v6_small_det_full.pth \
  --label-file /path/to/det_train.txt \
  --data-dir /path/to/det_dataset \
  --val-label-file /path/to/det_val.txt \
  --output-dir runs/ppocrv6_small_det_float \
  --device cuda \
  --epochs 20 \
  --batch-size 8
```

识别示例：

```bash
$PYTHON tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights weights/ptocr_v6_small_rec_full.pth \
  --label-file /path/to/rec_train.txt \
  --data-dir /path/to/rec_dataset \
  --val-label-file /path/to/rec_val.txt \
  --output-dir runs/ppocrv6_small_rec_float \
  --device cuda \
  --epochs 20 \
  --batch-size 64
```

浮点训练保留 Paddle 对应的完整训练结构。optimizer、基础学习率、weight decay、scheduler 和 warmup
默认从模型 YAML 读取，也可通过 CLI 覆盖。部署重参数化只在明确实验需要时使用
`--reparameterize`；不要把 deployment graph 当作 pretrained training graph。

## 7. QAT 配置生成与检查

QAT JSON 定义量化 dtype、qmin/qmax、observer 共享和区域规则；`configs/qat/training/*.yml` 定义 epochs、
batch size、学习率、输入 shape 和 observer 生命周期。两者不能混用。

当前多数旧模型规则为 U8 activation 和 S8 Conv/Linear weight。PP-OCRv5 mobile rec 当前改为
全局 U16 activation、S16 weight；两个 SVTR Attention 显式保持 S16，第二个 MatMul 输出恢复 U16。
通用配置发现器兼容两种全局合同：`U8/S8` 自动生成 Attention S8，`U16/S16` 自动生成 Attention
S16；也可显式指定局部 S16 形成混合位宽配置。当前 v5-rec 命令保留显式 `S16`，用于把该要求写入
可复现命令并防止误读。

### 7.1 使用已有配置

模型代码、Torch 版本和输入 shape 未变化时，先检查已有配置能否匹配当前 FX 图：

```bash
$PYTHON .codex/skills/ppocr-qat-config-discovery/scripts/discover_ppocr_qat_config.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --base-config configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json \
  --image-shape 3 48 320 \
  --attention-dtype S16 \
  --expected-attention 2 \
  --check
```

`--check` 必须输出 `QAT config FX node structure: PASS`。不要从历史 ONNX 或旧 FX 图复制
`linear_4`、`matmul_2` 等易变化的节点名。

### 7.2 重新生成 det/rec 配置

图结构变化后，以现有 dtype 规则作为模板，从当前 `export_for_training` 图重新发现节点：

```bash
$PYTHON .codex/skills/ppocr-qat-config-discovery/scripts/discover_ppocr_qat_config.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --base-config configs/qat/base_u16s16.json \
  --output /tmp/ppocrv5_mobile_rec_generated.json \
  --report /tmp/ppocrv5_mobile_rec_generated.report.json \
  --image-shape 3 48 320 \
  --attention-dtype S16 \
  --expected-attention 2
```

生成器不会覆盖 base config 或已有输出文件。检查 `/tmp` 产物后，再以新文件作为 `--base-config`
执行一次 `--check`。通用 skill 从 YAML 自动识别 det/rec；无 Attention 模型生成全局配置，含
Attention 模型根据 `nn_module_stack` 和实际拓扑重新定位区域，不能复用旧节点编号。

### 7.3 分阶段检查

```bash
$PYTHON tools/check_model_compatibility.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --qat-config /tmp/ppocrv5_mobile_rec_generated.json \
  --image-shape 3 48 320 \
  --output /tmp/ppocrv5_mobile_rec_compat.onnx
```

工具依次检查 `build -> forward -> prepare -> backward -> convert -> onnx -> qdq`。定位问题时可使用
`--stop-after` 停在指定阶段。

## 8. 训练前 QAT smoke

任何新模型、模型代码修改或 qspec 修改后，都必须先导出 smoke QuantONNX：

```bash
$PYTHON tools/qat_smoke.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --qat-config configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json \
  --output /tmp/ppocrv5_mobile_rec_qat_smoke.onnx \
  --image-shape 3 48 320 \
  --dynamic-heights 32 48 64 \
  --reparameterize \
  --onnx-optimize
```

检测模型将 `--task`、YAML、权重、QAT JSON 和 `--image-shape` 改为对应 det 配置。smoke 必须覆盖
prepare、backward、convert、ONNX checker、未优化 ORT 数值和 QDQ 结构检查。

将 smoke QuantONNX 交付结构检查并确认以下内容后，才能启动正式多 epoch 训练：

- Conv/BN、Concat、Split、Reshape 等共享量化域符合 Axera 规则；
- MatMul 输入 dtype、Attention 连续 S16 域和 S16 -> U16 出口符合 QAT JSON；
- Pad、Cast、SiLU/HardSigmoid 等实际 ONNX 展开没有异常 QDQ；
- 不存在未量化的 Conv、Linear 或 MatMul；
- ONNX checker 和 `tools/verify_qat_onnx.py` 通过。

## 9. QAT 训练

### 9.1 PP-OCRv5 mobile recognition

```bash
$PYTHON tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --label-file /path/to/rec_train.txt \
  --data-dir /path/to/rec_dataset \
  --val-label-file /path/to/rec_val.txt \
  --val-data-dir /path/to/rec_dataset \
  --output-dir runs/ppocrv5_mobile_rec_qat \
  --training-profile configs/qat/training/ppocrv5_mobile_rec_u16s16_sgd_dynamic_height.yml \
  --device cuda
```

该 profile 使用 SGD、momentum `0.9`，关闭 warmup，并设置 PP-OCRv5 rec 所需的 LAB
learning-rate multiplier 和 CTC FC weight decay。识别 QAT 默认 `--rec-graph pretrained_train`：保留
CTC、CTC neck 和 NRTR 辅助头，使用 `CTCLoss + NRTRLoss`，并保持 Paddle 配置的 guide detach；因此
profile 使用 `reparameterize: true`、`rec_ctc_backbone_grad: false`。重参数化在加载正式 LAB 参数后、
PT2E capture 前执行；`--rec-graph deploy` 是显式的
CTC-only 部署投影，只用于部署诊断或导出，不是默认 QAT 训练图。训练预处理高度为 48，prepared
训练图支持高度 32、48、64，宽度固定为 320；QuantONNX 静态导出为 `3x48x320`。

该 profile 同时设置浮点 accuracy 基线 `0.5936446798266731` 和第 2 epoch 最大允许下降 `0.10`。
第 2 epoch 验证 accuracy 下降达到或超过 10 个点时，训练会保存
`epoch2_accuracy_guard.json`、`debug_epoch_0002.pt` 并立即终止，后续应直接进入精度排查。
可分别用 `--float-accuracy-baseline` 和 `--epoch2-max-accuracy-drop` 覆盖；两个参数必须同时提供。

### 9.2 PP-OCRv5 mobile detection

```bash
$PYTHON tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --weights weights/ptocr_v5_mobile_det.pth \
  --label-file /path/to/det_train.txt \
  --data-dir /path/to/det_dataset \
  --val-label-file /path/to/det_val.txt \
  --val-data-dir /path/to/det_dataset \
  --output-dir runs/ppocrv5_mobile_det_qat \
  --training-profile configs/qat/training/ppocrv5_mobile_det_baseline.yml \
  --device cuda
```

### 9.3 PP-OCRv6 small recognition

```bash
$PYTHON tools/train.py \
  --task rec \
  --model-config configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml \
  --weights weights/ptocr_v6_small_rec_full.pth \
  --label-file /path/to/rec_train.txt \
  --data-dir /path/to/rec_dataset \
  --val-label-file /path/to/rec_val.txt \
  --output-dir runs/ppocrv6_small_rec_qat \
  --training-profile configs/qat/training/ppocrv6_small_rec_baseline.yml \
  --device cuda
```

### 9.4 PP-OCRv6 small detection

```bash
$PYTHON tools/train.py \
  --task det \
  --model-config configs/det/PP-OCRv6/PP-OCRv6_small_det.yml \
  --weights weights/ptocr_v6_small_det_full.pth \
  --label-file /path/to/det_train.txt \
  --data-dir /path/to/det_dataset \
  --val-label-file /path/to/det_val.txt \
  --output-dir runs/ppocrv6_small_det_qat \
  --training-profile configs/qat/training/ppocrv6_small_det_baseline.yml \
  --device cuda
```

显式 CLI 参数优先于 training profile，training profile 优先于 Paddle YAML。首次真实数据验证建议追加
`--epochs 1 --batch-size 2 --workers 0`，确认 train、validation、save、strict reload 和 export 全链路后
再运行完整 profile。

baseline 训练期间 observer 保持开启；验证阶段由 trainer 临时关闭 observer，验证后恢复。不要默认设置
`--observer-freeze-epoch`。QAT 使用 FP32 并关闭 AMP，随机增强保持关闭；PP-OCRv5 rec 当前仅启用
明确要求的 H=32/48/64 多尺度 sampler，不启用 RecAug/RecConAug。

训练目录包含：

```text
best.pt       主指标最优 checkpoint，det 使用 hmean，rec 使用 acc
last.pt       最后一个 epoch checkpoint
epoch_NNNN.pt 按 --save-every 保存的周期 checkpoint
```

## 10. 恢复训练

恢复训练需要保持模型 YAML、QAT JSON、Torch 版本、输入 shape、重参数化、optimizer 和 batch policy
一致。checkpoint 会恢复 prepared model、observer/fake-quant、optimizer、scheduler、epoch 和
global step；禁止跨图合同恢复：

```bash
$PYTHON tools/train.py \
  ...原训练参数... \
  --resume runs/ppocrv5_mobile_rec_qat/last.pt
```

`--epochs` 表示恢复后的总 epoch 上限，不是额外训练的 epoch 数。

只做严格恢复和验证时，在原训练命令中同时提供 `--resume`、`--val-label-file` 和 `--eval-only`。
默认验证 prepared QAT；使用 `--eval-stage converted` 可验证 converted PT2E。验证阶段会临时关闭
observer 并切换 eval，结束后恢复训练状态。

## 11. 模型导出

统一入口为 `tools/export_ocr_onnx.py`，包含五个子命令：

```text
checkpoint   从训练后的 QAT checkpoint 严格恢复并导出部署 QuantONNX
initialized  不推理、不训练，以激活 scale=1/zero-point=0 导出初始化 QuantONNX
training     导出四个模型的完整训练 Float ONNX/初始化 QuantONNX
float-matrix 导出 v5/v6 det/rec 的训练/推理 Float ONNX，分别覆盖重参数化和非重参数化
audit        重新检查已有 training ONNX
```

同时导出 PP-OCRv5 mobile、PP-OCRv6 small 的 det/rec 训练图和推理图，并分别保留重参数化、
非重参数化结构：

```bash
$PYTHON tools/export_ocr_onnx.py float-matrix \
  --models ppocrv5_mobile_det ppocrv5_mobile_rec \
           ppocrv6_small_det ppocrv6_small_rec \
  --graphs training inference \
  --reparameterizations reparameterized non_reparameterized \
  --output-dir exports/float_onnx/v5_v6_reparameterization_matrix
```

Float 矩阵导出默认执行 `onnx_program.optimize()`、ONNX checker 和 `ORT_DISABLE_ALL` 随机输入对齐。
非重参数化图保留原始 BN，仅用于结构、精度和后续 QAT 图合同分析。

### 11.1 导出训练后的 QuantONNX

```bash
$PYTHON tools/export_ocr_onnx.py checkpoint \
  --checkpoint runs/ppocrv5_mobile_rec_qat/best.pt \
  --output exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --batch-size 1
```

模型 YAML、QAT JSON、输入 shape 和重参数化状态默认从 checkpoint metadata 读取。导出器执行严格
`state_dict` 加载、`convert_pt2e`、`onnx_program.optimize()`、精确冗余 DQ/Q 清理、zero-point Cast
折叠、ONNX checker、QDQ 校验和未优化 ORT 对比。ORT graph-optimization A/B 需显式增加
`--ort-optimizer-check`。不要通过放宽 strict reload 导出旧图 checkpoint。

### 11.2 导出初始化 observer 的 QuantONNX

该模式不进行 calibration、推理或训练，激活 scale 保持 `1.0`、zero-point 保持 `0`：

```bash
$PYTHON tools/export_ocr_onnx.py initialized \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --qat-config configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16.json \
  --image-shape 3 48 320 \
  --output /tmp/ppocrv5_mobile_rec_initialized.onnx \
  --report /tmp/ppocrv5_mobile_rec_initialized.json
```

### 11.3 导出四个模型的训练或推理 ONNX

同时导出 Float ONNX 和 initialized QuantONNX：

```bash
$PYTHON tools/export_ocr_onnx.py training \
  --models \
    ppocrv5_mobile_rec ppocrv5_mobile_det \
    ppocrv6_small_rec ppocrv6_small_det \
  --format both \
  --output-dir exports/training \
  --batch-size 1 \
  --onnx-optimize \
  --ort-check
```

`training` 子命令使用内置模型表中的固定 YAML、权重和 QAT JSON。输出位于：

```text
exports/training/float/
exports/training/quantonnx/
exports/training/training_onnx_report.json
```

默认 `--quant-graph training` 保留 pretrained 训练辅助分支。导出部署推理 QuantONNX 时指定：

```bash
$PYTHON tools/export_ocr_onnx.py training \
  --models ppocrv5_mobile_rec \
  --format quantonnx \
  --quant-graph inference \
  --output-dir exports/inference \
  --batch-size 1 \
  --onnx-optimize \
  --ort-check
```

推理模式在 PT2E capture 前剔除辅助分支：检测模型只保留 DB shrink map，识别模型只保留 CTC logits，
并移除 `gtc_targets` 输入。产物命名为 `*_inference_init_qat.onnx`，报告键为
`quantonnx_inference`；不会覆盖同目录下完整训练图的 `quantonnx` 记录。

重新检查已有产物：

```bash
$PYTHON tools/export_ocr_onnx.py audit \
  --models ppocrv5_mobile_rec ppocrv5_mobile_det ppocrv6_small_rec ppocrv6_small_det \
  --output-dir exports/training
```

### 11.4 导出后结构与数据集验证

```bash
$PYTHON tools/verify_qat_onnx.py \
  --model exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --check-value

$PYTHON tools/evaluate_onnx.py \
  --task rec \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --onnx exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --label-file /path/to/rec_val.txt \
  --data-dir /path/to/rec_dataset \
  --batch-size 1 \
  --no-ort-optimize
```

QuantONNX 语义基线使用 ORT `ORT_DISABLE_ALL`。ORT graph optimization 只能作为独立 A/B 诊断，不能
代替未优化 ORT 指标。`onnx_program.optimize()` 是导出阶段的 ONNXProgram 优化，与 ORT graph
optimization 不是同一过程。

## 12. PP-OCRv5 rec Pulsar2 配置

Pulsar2 配置必须从最终要编译的精确 QuantONNX 生成，不能复用旧 ONNX 节点名：

```bash
$PYTHON .codex/skills/ppocrv5-rec-pulsar2-config/scripts/generate_ppocrv5_rec_pulsar2_config.py \
  --onnx exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --output artifacts/pulsar2/ppocrv5_mobile_rec_best.json \
  --output-dir artifacts/pulsar2/ppocrv5_mobile_rec_best \
  --target-hardware AX650 \
  --npu-mode NPU3

$PYTHON .codex/skills/ppocrv5-rec-pulsar2-config/scripts/validate_ppocrv5_rec_pulsar2_config.py \
  --onnx exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --config artifacts/pulsar2/ppocrv5_mobile_rec_best.json \
  --report artifacts/pulsar2/ppocrv5_mobile_rec_best.json.report.json
```

验证通过后才能执行：

```bash
pulsar2 build \
  --input exports/quantonnx/ppocrv5_mobile_rec_best.onnx \
  --config artifacts/pulsar2/ppocrv5_mobile_rec_best.json \
  --output_dir artifacts/pulsar2/ppocrv5_mobile_rec_best
```

QuantONNX 和 ORT 通过不等于 AXModel 验收通过。最终还需要记录 Pulsar2 版本、目标芯片、编译日志、
板端预处理、板端输出误差和任务指标。

## 13. QDQ 与阶段验收

`tools/verify_qat_onnx.py` 和导出器至少检查：

- ONNX checker 和 `ORT_DISABLE_ALL` session；
- 未量化 Conv、ConvTranspose、Linear 和 MatMul；
- 冗余或必要的直接 `DequantizeLinear -> QuantizeLinear` 边界；
- Conv/ConvTranspose 及其融合激活后的 QDQ；
- Conv-BN 是否已正确处理，QuantONNX 不应残留异常 BatchNormalization；
- Concat、Split、Reshape 等数据移动算子的共享 qparam；
- HardSigmoid、Paddle Hsigmoid 和 SiLU 展开后的完整量化边界；
- QuantizeLinear/DequantizeLinear 的 dtype、zero-point 和 axis；
- prepared fake-on、converted PT2E 与 QuantONNX 的随机输入数值差异。

SiLU 在 ONNX opset 21 中通常降低为 `Sigmoid + Mul`，两者之间不应插入 QDQ。Paddle
`slope=0.2, offset=0.5` 的 Hsigmoid 可表现为 `hardsigmoid(1.2 * x)`，常量乘法和激活边界都必须按
实际 PT2E 图检查。不要根据算子名称猜测 HardSwish，也不要在 ONNX 导出后任意补写 QDQ。

精度定位必须按同一批、同一预处理输入依次比较：

```text
Paddle float
-> eager PyTorch float
-> exported PyTorch float
-> prepared observer-off/fake-quant-off
-> prepared fake-quant-on
-> converted PT2E
-> QuantONNX with ORT_DISABLE_ALL
-> Axera/Pulsar2
```

每个边界记录 `max_abs`、MAE 和任务指标。识别额外记录 CTC probability、argmax agreement、accuracy
和 normalized edit similarity；检测额外记录 shrink/threshold/binary map 误差及
precision/recall/hmean。ORT graph optimization 只能作为独立诊断，不能代替未优化 ORT 基线。

## 14. 浮点对齐与测试

Paddle/PyTorch 部署输出对齐：

```bash
env HOME=/tmp/ppocr_qat_home FLAGS_use_mkldnn=0 PYTHONPATH="$PWD" \
  "$CONVERT_PYTHON" tools/compare_det_parity.py \
  --model-config configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml \
  --paddle-weights weights/PP-OCRv5_mobile_det_pretrained.pdparams \
  --torch-weights weights/ptocr_v5_mobile_det.pth \
  --image-shape 3 128 128

env HOME=/tmp/ppocr_qat_home FLAGS_use_mkldnn=0 PYTHONPATH="$PWD" \
  "$CONVERT_PYTHON" tools/compare_rec_parity.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --paddle-weights weights/PP-OCRv5_mobile_rec_pretrained.pdparams \
  --torch-weights weights/ptocr_v5_mobile_rec_full.pth \
  --image-shape 3 48 320
```

完整训练图、loss 和梯度对齐使用：

```bash
$CONVERT_PYTHON tools/compare_pretrained_training.py --help
$CONVERT_PYTHON tools/compare_det_training.py --help
```

PT2E 浮点保持和真实数据阶段比较使用：

```bash
$PYTHON tools/compare_pt2e_float_preservation.py --help
$PYTHON tools/compare_qat_stages.py --help
```

运行完整测试：

```bash
env PYTHONPATH="$PWD" "$PYTHON" -m pytest -q tests
```

测试应覆盖数据预处理与 loss backward、Paddle/PyTorch 完整训练输出、PT2E prepare/convert、observer
生命周期、checkpoint strict reload、部署投影、QuantONNX 导出、QDQ 审计和 ORT 对比。

## 15. 必须记录的结果

每次训练或导出至少记录：

- Git commit 和工作树状态；
- Python、Torch、ONNX、ORT 和 CUDA 版本；
- Paddle YAML、PyTorch 权重、QAT JSON 和 training profile 的明确路径；
- 输入 shape、batch policy、是否重参数化；
- eager float、exported float、prepared fake-off、prepared fake-on、converted、QuantONNX 的
  `max_abs` 和 MAE；
- det 的 shrink/threshold/binary map 误差以及 precision/recall/hmean；
- rec 的 CTC logits/probability/argmax agreement、accuracy 和 normalized edit similarity；
- checkpoint、smoke QuantONNX、best QuantONNX 和 Pulsar2 配置路径。

任何阶段失败时先定位该阶段，不要用后续 QAT 训练补偿模型转换、fake-quant-off、PT2E convert 或
QuantONNX 结构错误。

## 16. 当前状态与限制

- 当前只验证单卡训练，尚未接入 DDP；recognition 动态宽度尚未支持，部署宽度固定。
- QAT baseline 关闭随机增强；Paddle 的其他模型专属增强需要逐模型验收后才能启用。
- ONNX checker 和 ORT 通过不等于 Axera 芯片验收；仍需 Pulsar2 编译和板端结果。
- PP-OCRv5 mobile rec 的 Exp2 已在 epoch 2 精度门禁停止。该 checkpoint 和 QuantONNX 仅用于结构与
  根因分析，不得恢复为正式训练；当前结论、Exp3 对照和保留产物见
  [`v5 mobile rec QAT 训练记录`](docs/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md)。
- 新模型或图/qspec 变更后，必须先完成随机输入 smoke，并将 smoke QuantONNX 交付人工结构检查；确认
  前不得启动正式多 epoch QAT。

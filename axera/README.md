# Axera 部署与板端精度验证

本文说明如何将已有 QuantONNX 准备为 Pulsar2 转换配置，生成 Axera AXModel，并在 AX650 板端完成输入
准备、模型推理和精度验收。QuantONNX 导出和 QDQ 检查参见仓库根目录 [`README.md`](../README.md)。

```text
QuantONNX
-> Pulsar2 转换配置
-> Pulsar2 编译
-> compiled.axmodel
-> AX650 板端输入准备
-> AXModel 推理
-> AX650 板端推理与全量精度验证
```

本文面向 PP-OCRv6 small det、PP-OCRv6 small rec，也适用于已经通过相同 QAT 合同的 v5 模型。
完成 Pulsar2 编译后，再通过用户自己的文件传输方式将以下内容放到板端工作目录：

- `compiled.axmodel`；
- 识别或检测验证集、标签和字典；
- 本目录中的板端脚本；
- 与模型一致的输入 shape、预处理和后处理参数。

## 1. 生成 Pulsar2 转换配置

配置生成和 Pulsar2 编译必须在已安装 Pulsar2 的工具链环境中执行；以下命令均假定当前 shell 已经在
该环境中，不通过 SSH 远程编排。AX650 板端只负责加载生成的 `compiled.axmodel`，见第 3 节。

### 1.1 使用已验证配置

已验证配置位于 [`config/`](config/)：

| 模型 | 配置 |
| --- | --- |
| PP-OCRv6 small det U8/S8 + keep-BN | [`ppocrv6_small_det_u8s8_keep_bn.json`](config/ppocrv6_small_det_u8s8_keep_bn.json) |
| PP-OCRv6 small rec U8/S8 + Attention S8 | [`ppocrv6_small_rec_u8s8_attn_s8.json`](config/ppocrv6_small_rec_u8s8_attn_s8.json) |
| PP-OCRv6 small rec W8A16 + Attention S16 | [`ppocrv6_small_rec_w8a16_attn_s16.json`](config/ppocrv6_small_rec_w8a16_attn_s16.json) |
| PP-OCRv5 mobile rec U8/S8 + Attention S8 + downsample S16 | [`ppocrv5_mobile_rec_u8s8_attn_s8_downsample_s16.json`](config/ppocrv5_mobile_rec_u8s8_attn_s8_downsample_s16.json) |

从仓库根目录复制与模型对应的配置，再修改实际文件路径和目标设备参数：

```bash
cp axera/config/<verified-config>.json /path/to/pulsar2-work/config/<model>.json
```

至少确认以下字段：

```text
input                         = 本次最终 QuantONNX 路径
output_dir                    = 本次编译输出目录
model_type                    = QuantONNX
target_hardware               = 实际目标硬件
npu_mode                      = 实际部署 NPU mode
quant.input_configs           = 实际输入 tensor、校准数据路径
input_processors              = 与预处理一致，通常为 FP32/NCHW identity
```

QuantONNX 已经携带 Q/DQ，`calibration_dataset` 仅满足 Pulsar2 转换流程的输入要求，不得借此对模型重新
执行 PTQ 或覆盖 QAT qparams。输入输出名称、shape、dtype 和 batch 必须从本次 QuantONNX/AXModel 实际
接口确认，不能直接套用其他模型配置。

### 1.2 通用配置生成与校验

模型结构、QuantONNX、frontend 图或 qspec 发生变化时，不能沿用旧的 `layer_names`。使用通用 skill
从本次最终 QuantONNX 生成配置骨架；模型专属 `layer_configs` 必须由用户显式提供，不能由模型名称猜测。

```bash
cd /path/to/PaddleOCR
python3 .codex/skills/ppocr-pulsar2-config/scripts/generate_pulsar2_config.py \
  --onnx /path/to/model_qdq.onnx \
  --output /path/to/pulsar2-work/config/model.json \
  --output-dir /path/to/pulsar2-work/output/model \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --calibration-dataset /path/to/calibration.zip
```

如果已有模型专属的 `layer_configs`，通过 `--layer-configs` 传入；如果这些名称来自 Pulsar2 frontend
优化图，同时传入 `--frontend-onnx`：

```bash
python3 .codex/skills/ppocr-pulsar2-config/scripts/generate_pulsar2_config.py \
  --onnx /path/to/model_qdq.onnx \
  --output /path/to/pulsar2-work/config/model.json \
  --layer-configs /path/to/layer_configs.json \
  --frontend-onnx /path/to/frontend/optimized.onnx \
  --calibration-dataset /path/to/calibration.zip
```

生成后，按最终 QuantONNX 和 frontend 图校验：

```bash
python3 .codex/skills/ppocr-pulsar2-config/scripts/validate_pulsar2_config.py \
  --onnx /path/to/model_qdq.onnx \
  --config /path/to/pulsar2-work/config/model.json \
  --frontend-onnx /path/to/frontend/optimized.onnx
```

通用 skill 的详细说明见 `.codex/skills/ppocr-pulsar2-config/SKILL.md`。

### 1.3 PP-OCRv5 rec profile 配置生成

PP-OCRv5 rec 的 Attention 区域可通过通用 Pulsar2 skill 的 `ppocrv5-rec` profile 从本次实际
QuantONNX 自动发现；该 profile 不适用于 v6 det/rec：

当前仓库提供的自动生成器针对 PP-OCRv5 recognition：

```bash
cd /path/to/PaddleOCR
python3 .codex/skills/ppocr-pulsar2-config/scripts/generate_pulsar2_config.py \
  --profile ppocrv5-rec \
  --onnx /path/to/ppocrv5_mobile_rec_qdq.onnx \
  --output /path/to/pulsar2-work/config/ppocrv5_mobile_rec.json \
  --output-dir /path/to/pulsar2-work/output/ppocrv5_mobile_rec \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --attention-dtype S8 \
  --calibration-dataset /path/to/calibration.zip
```

生成后执行校验：

```bash
cd /path/to/PaddleOCR
python3 .codex/skills/ppocr-pulsar2-config/scripts/validate_pulsar2_config.py \
  --profile ppocrv5-rec \
  --onnx /path/to/ppocrv5_mobile_rec_qdq.onnx \
  --config /path/to/pulsar2-work/config/ppocrv5_mobile_rec.json \
  --target-hardware AX650 \
  --npu-mode NPU3 \
  --attention-dtype S8
```

该 profile 目前只覆盖 PP-OCRv5 rec 的已验证 Attention 合同。其他模型应使用对应的已验证配置或按实际
frontend 图生成专用配置，不得把 v5-rec 的节点规则复制到 v6 det/rec。

## 2. Pulsar2 转换

在安装 Pulsar2 的工具链环境中确认版本并执行转换：

```bash
pulsar2 version
pulsar2 build \
  --input /path/to/quantonnx/<model>.onnx \
  --config /path/to/pulsar2-work/config/<model>.json \
  --output_dir /path/to/pulsar2-work/output/<model>
```

如果配置文件已经填写 `input` 和 `output_dir`，也可以直接执行：

```bash
pulsar2 build --config /path/to/pulsar2-work/config/<model>.json
```

编译后检查并保留：

```text
/path/to/pulsar2-work/output/<model>/compiled.axmodel
/path/to/pulsar2-work/output/<model>/frontend/optimized.onnx
/path/to/pulsar2-work/output/<model>/quant/quant_axmodel.onnx（如工具链生成）
编译日志、Pulsar2 版本和本次使用的配置
```

若 frontend 对节点进行了重命名、融合或拆分，必须根据 `frontend/optimized.onnx` 实际节点重新调整
`layer_configs` 并再次编译。若出现 unsupported op、非法量化域或输入输出不匹配，应回到 QuantONNX、
QAT 配置或 Pulsar2 配置排查，不能直接手工改写最终 QDQ 图。

## 3. 部署到 AX650 板端

在 AX650 板端进入包含模型和脚本的工作目录，并确认 `axengine` 可用：

```bash
cd /path/to/qat-ppocr
python3 -c "import axengine; print('axengine available')"
```

AXModel 不能使用普通 CPU ONNX Runtime 直接加载；板端推理必须使用
`axengine`/`AxEngineExecutionProvider`。

部署前确保 `compiled.axmodel`、输入数据和评估脚本属于同一模型版本。输入输出名称、shape、dtype 和
batch 不在本文中写死，必须以实际 AXModel、对应 QuantONNX 以及 Pulsar2 配置为准。输入数据必须按照
实际模型的输入 shape、layout、dtype 和预处理合同生成；输出解析必须按照实际模型的输出 shape 和任务
语义配置，不能直接套用其他模型的名称或 shape。

运行前先查看脚本接口：

```bash
python3 axera/eval_board_det.py --help
python3 axera/eval_board_rec.py --help
```

脚本职责如下：

| 脚本 | 执行位置 | 用途 |
| --- | --- | --- |
| `eval_board_rec.py` | AX650 板端 | 识别输入预处理（自包含）、AXModel 推理、CTC 解码和 accuracy |
| `eval_board_det.py` | AX650 板端 | 检测 AXModel 推理并保存 shrink map bin；`--vis-dir` 输出叠加框可视化（`--vis-score`、`--no-maps`） |

### 3.1 识别板端输入和评估

识别输入按模型合同进行等比缩放、右侧 zero padding、`[-1,1]` 归一化。
**`eval_board_rec.py` 已内置该预处理**（无需独立预处理脚本）；两种模式：

- **在线模式（默认）**：直接从挂载数据集读图并逐批预处理——板端内存小也能跑
  （逐批释放），适合数千张量级的验证集（如 ICDAR2015 test 2077 张）；
- **分片模式（`--parts-dir`）**：读取预处理好的 npz 分片，在慢速 NFS 或
  十万张量级（如 rec_val 149,695）时更快。分片可由脚本自身生成：

```bash
# 可选：先生成分片和同目录的 texts.json
python3 axera/eval_board_rec.py \
  --label-file /path/to/dataset/rec/rec_gt_test.txt \
  --data-dir /path/to/dataset/rec \
  --dictionary /path/to/ppocrv6_dict.txt \
  --make-parts /path/to/rec/icdr_parts --parts-size 500

# 分片模式评估（README 约定接口）
python3 axera/eval_board_rec.py \
  --axmodel /path/to/compiled.axmodel \
  --texts /path/to/rec/icdr_parts/texts.json \
  --parts-dir /path/to/icdr_parts \
  --dictionary /path/to/ppocrv6_dict.txt \
  --batch-size 1 --use-space-char

# 在线模式评估（自包含预处理,无需 --texts/--parts-dir）
python3 axera/eval_board_rec.py \
  --axmodel /path/to/compiled.axmodel \
  --label-file /path/to/dataset/rec/rec_gt_test.txt \
  --data-dir /path/to/dataset/rec \
  --dictionary /path/to/ppocrv6_dict.txt \
  --batch-size 1 --use-space-char
```

`--make-parts` 会同时生成 `part_*.npz` 和 `<parts-dir>/texts.json`，两者样本顺序一致；不要
手工重排其中任一文件。分片模式只读取 npz 和 JSON，在板端未安装 OpenCV/Pillow 时也可执行。

v6-rec 使用 `use_space_char: true` 时解码字典的字符列表末尾必须追加空格
（脚本默认 `--use-space-char`）；不使用空格的 v5-rec 字典传
`--no-use-space-char`。CTC blank 仍为类别 0。
板端脚本必须与参考评估使用完全相同的字典、预处理、CTC collapse 和文本标准化。

实现细节：标签中的图片路径可为相对路径（相对 `--data-dir`）或绝对路径
（按 basename 在 `--data-dir` 的 `val/`、`test/`、根目录候选中解析）；
有 cv2 时用 cv2.resize，无 cv2 时回退 PIL。

### 3.2 检测板端输入和评估

检测 QuantONNX/AXModel 默认使用居中 `letterbox`，目标 H/W 自动从 AXModel 输入读取，padding
value 为 114，并使用相同的 scale 和 offset 变换 polygon。当前新 v6-det 部署输入为 736x736；
历史 640x640 AXModel 会自动按 640x640 兼容。若要与 PaddleOCR 浮点/PT2E 的官方动态 resize
对照，显式传 `--det-preprocess paddle`。板端输出为 shrink map，使用与 ORT 相同的 DB postprocess
和 polygon metric 计算 precision、recall、hmean。

板端执行检测模型：

```bash
python3 axera/eval_board_det.py \
  --axmodel /path/to/compiled.axmodel \
  --image-dir /path/to/det-images \
  --label-file /path/to/det_val.txt \
  --output-dir /path/to/outputBin_detval
```

**结果可视化（可选）**：`--vis-dir` 会在板端直接跑一次 DB 后处理，并把每个样本的检测框叠加图
写到 `<vis-dir>/<stem>.jpg`，便于快速核对模型行为，无需把 bin 拷回主机：

```bash
python3 axera/eval_board_det.py \
  --axmodel /path/to/compiled.axmodel \
  --image-dir /path/to/det-images \
  --label-file /path/to/det_val.txt \
  --output-dir /path/to/outputBin_detval \
  --vis-dir /path/to/vis_detval --vis-score
# 只要可视化、不落 maps.bin: 追加 --no-maps
```

AXModel 输入尺寸由脚本自动读取，不需要手工填写 736 或 640。若要使用 direct resize 对照，
在上述命令末尾增加 `--det-preprocess paddle`；这种模式必须与对应 ORT/QuantONNX 评估使用
相同的预处理参数。

- 阈值默认取 v6 det 配置（`configs/det/PP-OCRv6/PP-OCRv6_small_det.yml`）：
  `--vis-thresh 0.2`、`--vis-box-thresh 0.45`、`--vis-unclip-ratio 1.4`、
  `--vis-max-candidates 3000`，可按数据集覆盖；`--vis-score` 叠加分数文本，
  `--vis-thickness` 调整线宽。
- 框提取与 `pytorchocr/postprocess/db_postprocess.py` 一致（相同的 `get_mini_boxes` 排序、
  `box_score_fast` 打分、最小边 3 / 3+2 过滤）；仅 unclip 用等价的旋转矩形扩张
  （距离 `area * ratio / perimeter`）替代 pyclipper，因为板端没有 pyclipper/shapely。
- 框坐标按 `--det-preprocess` 的 scale/offset 严格反向映射回原图（与预处理互逆）。
- 可视化只用于人工核对；**正式 precision/recall/hmean 仍以导出的 `maps.bin` 在主机按同一
  DB 参数计算为准**，不要用叠加图反推指标。

板端全量评估需要记录输入样本顺序、输出 bin 对应关系、DB 参数和每个数据集的 precision、recall、
hmean。不同数据集的 hmean 不得直接相减。

## 4. 精度验收与问题定位

板端验收使用与模型合同一致的预处理、输入 shape、字典和后处理。识别记录 sequence accuracy 和
normalized edit similarity；检测记录 precision、recall 和 hmean。每个数据集单独记录，不跨数据集
直接比较或相减。

### 4.1 PP-OCRv6 small rec W8A16 验收结果

已验证配置为
[`ppocrv6_small_rec_w8a16_attn_s16.json`](config/ppocrv6_small_rec_w8a16_attn_s16.json)，
使用 Pulsar2 7.0、AX650/NPU3、静态输入 `1x3x48x320`。全量结果如下：

| 数据集 | ORT_DISABLE_ALL accuracy / NED | AX650 accuracy / NED | accuracy 差值（AX650 - ORT） |
| --- | ---: | ---: | ---: |
| TextOCR/COCO-Text 混合 `rec_val`（149,695） | `0.720191 / 0.814823` | `0.721093 / 0.815407` | `+0.000902` |
| ICDAR2015 test（2,077） | `0.738565 / 0.887125` | `0.737121 / 0.886952` | `-0.001444` |

50 张 ICDAR2015 仿真对比中，AXModel 与 QuantONNX 的逐帧 argmax agreement 和序列一致率均为
`1.0`，logits MAE 为 `0.07364`。两套全量板端 accuracy 与 ORT 的差值均不超过 `0.002`，通过
当前板端对齐判据。

### 4.2 per-layer dump dtype

板端逐层 dump 使用量化整型，读取 dtype 必须与 qparam 一致：

| 量化域 | 读取 dtype | 反量化 |
| --- | --- | --- |
| S8，zero-point 为 0 | `int8` | `q * scale` |
| U8，zero-point 非 0 | `uint8` | `(q - zero_point) * scale` |
| FP32 输出 | `float32` | 直接读取 |

用错 dtype 会把 S8 负值误判为大数，或把 U8 值域误判为饱和。出现板端与参考结果不一致时，先检查
输入预处理、模型输入输出和逐层整数域，再判断是否需要调整 qspec。

## 5. 交付物清单

每次部署至少保留以下内容：

- `compiled.axmodel` 及其对应的模型版本和输入 shape；
- 板端芯片、axengine 版本、NPU mode、输入 shape、batch 和预处理说明；
- 输入样本顺序、输出 bin 对应关系和后处理参数；
- det 的 precision/recall/hmean，或 rec 的 accuracy/normalized edit similarity；
- 已知限制和最终结论。

不记录文件哈希，不把局部样本结果当作全量板端精度。

当前已完成的 v6 验收实例和指标记录见根目录 README 及项目实验记录。

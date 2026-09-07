---
name: ppocr-accuracy-analysis
description: PP-OCR 精度定位与量化收益分析工作流：PTQ 全量 8/16bit 验证（LSQ 流程、不训练）、PT2E/QuantONNX/ORT 阶段对比、远端 axmodel vs ONNX 仿真对比、per-layer dump 逐层定位（含 dtype 读取陷阱）、NPU 板端全量评估，以及每个边界的指标合同与结论记录。使用场景：新 qspec/位宽方案评估、板端与 ORT 精度对不齐、PTQ 是否有收益等量化精度问题。
---

# PP-OCR 精度分析（Accuracy Analysis）

本 skill 固化 exp13/exp16 精度对齐与 PTQ 验证实验沉淀的工作流（实验记录见
`docs/references/development/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md`
§40.2/§43）。
精度定位的法定顺序与指标合同以 `AGENTS.md`「精度定位顺序」为准，本 skill 给出每个
边界的可执行命令、判据和已知陷阱。

## 何时使用

- 新位宽/qspec 方案（如全量 8bit vs 16bit、混合精度）需要量化收益评估；
- 板端（axmodel）精度与 ONNX/ORT 精度对不齐，需要定位差异来源；
- 回答"PTQ 是否有收益 / QAT 是否必需"；
- 新模型完成 QuantONNX 后需要完整的板端验收证据链。

## 精度定位顺序与指标合同

按同一批、同一预处理输入依次比较（AGENTS.md）：

```text
Paddle float -> eager PyTorch float -> exported float -> prepared fake-off
-> prepared fake-quant-on -> converted PT2E -> QuantONNX(ORT) -> Axera axmodel
```

每个边界至少记录 `max_abs`、MAE 和任务指标：
- 识别：CTC logits、softmax probability、逐帧 argmax agreement、序列准确率、
  normalized edit distance；
- 检测：shrink/threshold/binary map 误差、precision/recall/hmean。

**语义基线**：QuantONNX 的 ORT 评估必须 `ORT_DISABLE_ALL`；ORT graph optimization
只作独立诊断（本项目有优化后 argmax agreement 下降的案例，勿当基线）。

## 1. PTQ 精度验证（全量 8bit / 全量 16bit，不训练）

脚本：`scripts/ptq_eval_rec.py`。流程：浮点权重 → reparam deploy（CTC）→ LSQ prepare
→ 权重静态统计 + 训练集激活统计校准 → val 全量评估（prepared fake-quant-on 与
converted，不做任何训练 step）。

```bash
env PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=N \
  /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python \
  .codex/skills/ppocr-accuracy-analysis/scripts/ptq_eval_rec.py \
  --model-config configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  --weights weights/ptocr_v5_mobile_rec_full.pth \
  --train-label-file /home/heqi/dataset/icdr/rec_gt_train.txt \
  --val-label-file /home/heqi/dataset/icdr/rec_gt_test.txt \
  --data-dir /home/heqi/dataset/icdr \
  --bitwidths 8 16 \
  --calibration-batches 8 --calibration-batch-size 64 \
  --device cuda:0
```

**已知基线（v5 mobile rec，2026-08-17）**，作为新结果对照的参考系：

| 配置 | PTQ prepared fake-on | PTQ converted | 对比 |
| --- | ---: | ---: | --- |
| 全量 U8/S8 | 0.2364 | 0.2427 | 浮点 0.5936，8bit PTQ 严重退化 |
| 全量 U16/S16 | 0.5368 | 0.5392 | 低于浮点约 0.05，部分可用 |
| U8/S8+AttnS8 QAT（exp16） | - | - | 训练后 0.5927（QAT 相对 8bit PTQ +0.35） |

判据：
- 8bit PTQ 直接评估普遍严重退化（激活/attention 不做训练无法 8bit 化）；
- 16bit PTQ 可作为 QAT 起点或保底方案；LSQ 可学习 scale 训练（QAT）不可省略；
- 校准量（`--calibration-batches`）越大 HistogramObserver 越准，8→16 batch 已接近
  饱和；对比实验须固定校准量。

## 2. 本地阶段对比与 ONNX/ORT 基线

- 阶段对比（prepared vs converted，可选 --onnx 对比 QuantONNX）：

  ```bash
  env PYTHONPATH="$PWD" "$PYTHON" tools/compare_qat_stages.py \
    --checkpoint runs/<run>/best.pt \
    --label-file /path/to/rec_gt_test.txt --data-dir /path/to/dataset \
    --samples 512 --batch-size 64 --device cuda:0 \
    --onnx exports/quantonnx/<model>_qdq.onnx
  ```

- checkpoint / ONNX 全量任务指标：

  ```bash
  env PYTHONPATH="$PWD" "$PYTHON" tools/eval.py \
    --task rec --model-config configs/rec/<model>.yml \
    --pt runs/<run>/best.pt --stage converted \
    --label-file /path/to/rec_gt_test.txt --data-dir /path/to/dataset \
    --batch-size 64
  env PYTHONPATH="$PWD" "$PYTHON" tools/eval.py \
    --task rec --model-config configs/rec/<model>.yml \
    --onnx exports/quantonnx/<model>_qdq.onnx \
    --label-file /path/to/rec_gt_test.txt --data-dir /path/to/dataset \
    --image-shape 3 48 320   # 部署件为静态 batch 1
  ```

  也可用零 torch 依赖脚本（远端/板端）：`axera/eval_board_rec.py`（AXEngine，未安装
  `axengine` 时使用 CPU ORT）。

- 判据：prepared fake-on ≈ converted ≈ QuantONNX（训练内 val 与 ORT 全量的差值在
  v5 rec 上约 0.007，属正常；对齐模式的 argmax agreement 应为 1.0）。

## 3. 远端编译与 axmodel vs ONNX 仿真对比

远端 Pulsar2 服务器与板端访问方式见本地
`docs/references/development/remote_axera_access.md`（IP/账号/环境/工作目录，
`compare_onnx_ax.py` 等脚本路径）。编译：

```bash
ssh <user>@<pulsar2-server> 'cd /data/heqi/project/npu-codebase && source script/npu_dev \
  && cd /data/shared/heqi/self-developed/qat-ppocr \
  && pulsar2 build --input onnx/<model>_qdq.onnx \
       --config config/<model>.json --output_dir output/<run>'
```

50 样本仿真对比（本 skill 的默认协议，与 exp13/exp16 一致）：

```bash
# 远端工作目录 /data/shared/heqi/self-developed/qat-ppocr
python3 compare_rec_onnx_ax_batch.py \
  --onnx onnx/<run>/<model>_qdq.onnx \
  --axmodel output/<run>/compiled.axmodel \
  --dictionary-path ppocrv5_dict.txt \
  --label-file /data/shared/heqi/self-developed/dataset/icdr2015val/rec_gt_test.txt \
  --data-dir /data/shared/heqi/self-developed/dataset/icdr2015val \
  --samples 50 --tmp-dir /tmp/rec_cmp_<run>
```

记录指标：`mean_argmax_agree`、`ax_vs_onnx_seq_acc`、`mean_mae`、`mean_max_abs`、
`mean_rmse`、`ax_seq_acc`/`onnx_seq_acc`（50 样本子集不可当全量 acc，只做 ax 与
onnx 的相对比较）。

判据（v5 rec 参考）：exp13（全 8bit 下采样）0.9765/0.80/2.65/23.6/0.68-0.74；
exp16（下采样链 S16）0.9900/0.86/1.156/11.85/0.72-0.72；exp16 上板全量 0.58449
vs ORT 0.58546（差 0.001），exp13 为 0.5392 vs 0.5835（差 0.044）。

**编译陷阱（已固化为代码修复，重述以防复发）**：
- frontend 会重命名/融合节点：Pulsar2 配置的 `layer_names` 必须对
  `output_dir/frontend/optimized.onnx` 回填（attention QKV 常变成
  `op_N:onnx.FullyConnected`，需从 probe 量化的 `quant_axmodel.json`
  tensor_configs 确认 `op_N` 与 ONNX 名的对应）；
- S16 域的 lab scale Mul 必须张量在前、标量在后且标量 int16：导出 pass
  `swap_scalar_first_quantized_muls` + bridge 的 S16 dyadic 标量 qspec
  （`pytorchocr/quantization/onnx_export.py` / `bridge.py`，均有回归测试）。

## 4. per-layer dump 逐层定位（板端-ORT 对不齐时）

1. 插桩原始 QuantONNX（中间输出）在 ORT 跑同一输入；
2. 远端 `pulsar2 run --model output/<run>/quant/quant_axmodel.onnx
   --input_dir <in> --output_dir <out> --list <list> --enable_perlayer_output`
   得到中间层 bin；
3. 逐层对比，**dtype 读取陷阱**（exp13 踩坑记录 §40.2）：
   - S8 层（zp=0 有符号）用 `int8` 读，反量化 `q * scale`；
   - U8 层（zp≠0 无符号）用 `uint8` 读，反量化 `(q - zp) * scale`；
   - float32 输出层直接 float32 读；
   - scale/zp 取自原始 QuantONNX 的 consumer QuantizeLinear；
   - 用错 dtype 会把 S8 负值位移成大数误判"饱和"、把 U8 当 S8 误判值域错位。
4. 同时对比整数域（同 scale 下 NPU q 与 ORT q 的 int diff），定位注入误差的层。

判据：整数域 int diff 大于 ~2（8bit 下采样链典型值）即为分辨率不足层；此类层改
S16 域（`ppocr-qat-config-discovery` + qspec regional 条目），attention S8/S8→U8
在 NPU 上已验证正确、无饱和。

## 5. NPU 板端全量评估

板端访问方式见内部 `docs/references/development/remote_axera_access.md`；使用
`axera/eval_board_rec.py`（AXEngine + CPU ORT 双后端）评估。板端跑通后使用同一预处理数据和
CPU ORT 后端得到对比基线，报告：

```text
板端 axmodel acc / norm_edit  vs  CPU ORT acc / norm_edit  →  差值
```

验收判据（v5 rec）：差距 ≤ ~0.002 视为对齐（exp16 实测 0.001）；差距 ~0.04 量级
（exp13）必须回到 §4 逐层定位，不得直接推断为训练/量化配置问题。

## 6. 结论记录

把结果写入对应模型记录（`docs/references/development/axera_qat/records/`）：配置与命令全文、环境版本、
输入 shape/batch 策略、每个边界的 max_abs/MAE 与任务指标、checkpoint/QuantONNX/
Pulsar2 配置/axmodel 路径。不生成或记录哈希值。50 样本子集指标与全量指标分开写，
不得互相冒充。

## 关键陷阱速查

1. ORT 语义基线必须 `ORT_DISABLE_ALL`；graph optimization 只作诊断；
2. 部署 QuantONNX 是静态 batch 1；全量 eval 时 `--batch-size 1`；
3. Pulsar2 frontend 重命名：先看 `optimized.onnx` 再回填配置；
4. per-layer dump 的 S8/U8 读取 dtype 与反量化公式；
5. 8bit PTQ 会严重退化，属预期现象，不是流程 bug；
6. 标量操作数顺序/位宽对 AX650 TENG 打包敏感（已在导出链路修复，勿回退）；
7. `pulsar2 run` 每次调用约 17s 启动开销，批量跑（`--list` 单 session）。

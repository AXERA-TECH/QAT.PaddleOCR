# AGENTS.md

## 项目定位

本仓库是从 `frotms/PaddleOCR2Pytorch` 扩展出的 PP-OCR PyTorch 训练与 Axera QAT 工程。核心目标是：

1. 将 PaddleOCR 检测、识别模型和预训练权重严格转换到 PyTorch；
2. 使用 PyTorch 2.6 PT2E 完成 QAT prepare、训练、convert 和 checkpoint 恢复；
3. 导出符合 Axera/Pulsar2 约束的 ONNX QDQ（本文统一称 QuantONNX）；
4. 在 Paddle、eager PyTorch、exported float、prepared fake-off、prepared fake-on、converted PT2E、
   ONNX Runtime 和 Axera 工具链之间定位结构或精度差异。

检测模型训练图保留 DBHead 的 shrink、threshold、binary 以及模型定义中的辅助监督，部署图只保留
shrink map。识别模型训练图复现 CTC+NRTR/GTC MultiHead，部署图只保留原始 CTC logits，不包含
NRTR/GTC 辅助分支及最终 Softmax。

完整使用说明见 `README.md`，文档分类见 `docs/readme.md`，QAT 路线见 `docs/axera_qat/readme.md`。

## 已验证环境

```text
Python 3.10
PyTorch 2.6.0
ONNX 1.21.0
ONNX Runtime 1.21.0
```

本机默认解释器：

```bash
PYTHON=/home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python
```

运行命令时从仓库根目录开始，并显式设置：

```bash
env PYTHONPATH="$PWD" "$PYTHON" ...
```

不要依赖外部 `QAT.axera` Python 路径。当前 Axera quantizer 已 vendored 到 `pytorchocr/quantization/`（ax_quantizer.py、
ax_quantizer_utils.py、quantized_decomposed_dequantize_per_channel.py 及 LICENSE/UPSTREAM.md）；
来源和许可证见 `pytorchocr/quantization/` 内的说明文件。

## 目录职责

```text
configs/                 Paddle 模型 YAML、Axera QAT JSON、QAT training profile
converter/               Paddle 参数到 PyTorch state dict 的严格转换
pytorchocr/modeling/     PyTorch PP-OCR 模型实现（architectures/backbones/necks/heads）
pytorchocr/training/     dataset、loss、metric、模型工厂、trainer、profile 合同
pytorchocr/quantization/ PT2E/Axera adapter、vendored quantizer、QDQ 验证
pytorchocr/diagnostics/  checkpoint 重建、数据审计、误差和任务指标公共 API
pytorchocr/data/         PaddleOCR 官方数据管线兼容层（上游遗留，只读参考）
pytorchocr/postprocess/  后处理与 DB/CTC decode（上游遗留，部分由 training/metrics 替代）
pytorchocr/utils/        字典、公共工具（hashing 等）
tools/                   训练、导出、诊断和验证 CLI 编排
tests/                   稳定 package API 和工具契约回归
docs/                    设计、计划、实验过程和指标记录
artifacts/               小型 JSON 报告、accuracy contract、Pulsar2 配置
weights/                 Paddle/PyTorch 浮点权重，不提交大文件
runs/                    训练 checkpoint 和日志
exports/quantonnx/       需要保留的 QuantONNX
exports/frontend/        Axera frontend 优化模型参考
references/              PaddleOCR、PytorchOCR、QAT.axera、QAT.Ultralytics.YOLOv5 等只读参考 checkout
cache/                   可再生缓存和历史迁移文件，不作为生产代码依赖
.codex/skills/           本项目 QAT 和 Pulsar2 工作流 skill
```

`tests/` 与 `tools/` 不合并。CLI 只负责 argparse、流程编排和文件落盘；可复用逻辑应放入
`pytorchocr`。测试不得导入 tools 的私有 helper。ONNX 导出与图后处理公共 API 位于
`pytorchocr/quantization/onnx_export.py`；`tools/export_ocr_onnx.py` 只负责 CLI 编排。

默认不要修改 `references/` 内的仓库。需要借鉴 Paddle 或 Axera 行为时，先在参考实现中定位规则，
再把最小兼容改动实现到主工程，并记录来源和差异。

## QAT 图合同

标准 QAT 路径：

```text
转换后的浮点权重
-> model.eval()
-> 部署重参数化
-> torch.export.export_for_training
-> prepare_qat_pt2e
-> QAT 训练
-> convert_pt2e
-> ONNX QDQ
```

必须遵守：

1. 优先保证 PT2E prepared/converted 图和 QuantONNX 结构正确；不要强制使用 eager Conv-BN 图或新增
   eager Conv-BN 固定融合来掩盖 PT2E 问题。
2. 模型代码、输入 shape、PyTorch 版本、重参数化行为或 qspec 改变后，必须从浮点权重重新 prepare；
   不得跨图合同恢复 prepared QAT checkpoint。
3. QAT baseline 训练期间 observer 保持启用。验证时可临时关闭，验证结束必须恢复。除非独立实验明确
   要求，不设置 `--observer-freeze-epoch`。
4. QAT 使用 FP32 激活；当前训练入口会禁用 PT2E QAT 的 AMP。
5. QAT 配置必须按当前 `export_for_training` FX 图重新发现节点。禁止复制旧图中的 FX 节点编号或
   `linear_4`、`matmul_2` 等易变名称。
6. Concat、Split、Reshape 等数据移动算子是否共享 qparam 必须依据 Axera 规则和实际拓扑判断。
   不得通过宽松 scale 容差合并来隐藏 QAT 图错误。
7. 不在导出后任意补写 QDQ。导出阶段只允许有数学证明、结构测试和精度回归覆盖的图变换。
8. 保留 `onnx_program.optimize()`。冗余 DQ/Q 只在 qparam 完全一致时删除；不能合并的直接 DQ/Q
   边界按现有导出规则插入 Identity。
9. 激活/权重 dtype 与 MatMul/Attention 连续量化域属于模型 QAT 合同。需要调整时，同时更新 QAT JSON、发现
   skill、QuantONNX 结构测试和 Pulsar2 配置生成规则。
10. 导出 ONNX/QuantONNX 默认 `batchSize=1`（静态 batch 1，部署推理形态）；除非任务或实验特殊
    说明需要其他 batch（如训练图核对用 batch 64、检测多 batch 验证）。识别模型训练图合同为
    dynamic batch（gtc targets 需 batch-aligned），导出部署图时固定为 batch 1。

## 数据与训练合同

- QAT baseline 关闭随机增强。
- 检测输入使用固定 shape、padding value 114 的居中 letterbox，polygon 必须使用相同 scale 和
  offset；归一化采用 ImageNet mean/std。
- 识别输入按目标高度等比缩放、右侧 zero padding，并归一化到 `[-1, 1]`。
- PT2E 默认固定 H/W。PP-OCRv5 rec 可由 profile 显式声明
  `dynamic_heights: [32, 48, 64]`，使用 `16 * height_factor` 捕获三档离散高度；宽度仍固定为 320。
  batch size 大于 1 时训练入口同时记录 dynamic batch；batch size 为 1 时通常特化为静态 batch 1。
- 动态空间尺寸只属于 prepared QAT 训练图。QuantONNX 必须按 checkpoint 的 `image_shape` 静态导出；
  PP-OCRv5 rec 当前固定为 `3x48x320`，不得把动态高度传播到 QuantONNX 输入。
- optimizer 名称、SGD momentum 和 dynamic heights 都属于 checkpoint resume 合同；任一项变化后
  必须从浮点权重重新 prepare，不能恢复旧 Adam/static-shape checkpoint。
- **QAT 优化器只允许 AdamW 或 SGD**（profile `optimizer: AdamW|SGD`）。不要使用 `Adam`：
  Adam 的 L2 weight decay 与 LSQ 的梯度缩放（`use_grad_scaling`）叠加会错误衰减
  learnable scale/zero_point 量化参数，易导致 QAT 早期发散。QAT 是预训练权重的
  量化域微调，学习率应远低于浮点训练（当前基线 lr 3e-5、warmup 5、Cosine 调度；
  Paddle 浮点训练的 5e-4 仅作参考，不得直接用于 QAT）。
- 非有限 loss、缺失梯度和非有限梯度必须立即失败，不能跳过 batch。
- `--resume` 必须通过 metadata 合同检查并以 `strict=True` 加载 model state。
- checkpoint 需要保存训练 profile、QAT JSON 路径、模型配置、输入 shape、PyTorch 版本、
  reparameterization 和 batch policy。不要手工删除这些 metadata。
- 正式训练产物写入 `runs/`。临时 smoke 和诊断输出优先写 `/tmp`，不要覆盖已验收的 ONNX 或
  checkpoint。

当前训练入口按 `--save-every` 保存 `epoch_NNNN.pt`，并维护 `best.pt` 和 `last.pt`。计划中的“仅保留
N 个最优权重”策略尚未实现；在实现并测试前不要在文档中宣称已经支持。

## 强制执行顺序

任何新模型、模型代码修改或 qspec 修改都按以下顺序推进：

1. 复现 Paddle pretrained 的完整训练结构、辅助头、targets、loss 和梯度路径；
2. 验证完整 Paddle 权重到 PyTorch full-training 模型的严格转换和全部训练输出；
3. 从同一 full-training state 显式投影 deployment graph，识别只保留 CTC、检测只保留 shrink；
4. 验证 eager deploy float 与 `export_for_training` float；
5. 验证 prepared 图在 observer-off、fake-quant-off 时保持浮点精度；
6. 运行随机输入 QAT smoke，覆盖完整训练 backward、部署投影、convert、QuantONNX 和 QDQ 检查；
7. 将 smoke QuantONNX 交给用户检查模型结构；用户确认前不得启动正式多 epoch QAT；
8. 使用小规模真实数据完成一个 epoch 的 train、validation、save、strict reload 和 export；
9. 才能启动完整数据集训练；
10. 对 best checkpoint 导出 QuantONNX，执行 ORT、结构审计和 Axera/Pulsar2 验证；
11. 将配置、命令、checkpoint、QuantONNX 路径和全部指标记入对应文档。

当前模型结构复现优先级为 PP-OCRv5 rec、PP-OCRv5 det、PP-OCRv6 rec、PP-OCRv6 det。详细门禁见
`docs/axera_qat/plans/pretrained_training_structure_plan.md`。训练图必须保留 Paddle 原始辅助分支；单 CTC
或单 shrink 只属于 deployment graph，不能冒充 pretrained training graph。

如果检测或识别模型在 Axera 工具链因 Pad、QDQ 或算子约束被暂停，不要绕过 smoke 门禁继续正式
训练。先保留复现模型并记录暂停原因。

## 精度定位顺序

对完全相同的预处理样本，依次比较：

1. Paddle float；
2. eager PyTorch float；
3. exported PyTorch float；
4. prepared observer-off/fake-quant-off；
5. prepared fake-quant-on；
6. converted PT2E；
7. QuantONNX in ONNX Runtime；
8. Axera 编译模型。

每个边界至少记录 `max_abs`、MAE 和任务指标。检测需比较 shrink/threshold/binary map，再比较
precision、recall、hmean；识别需比较 CTC logits、softmax probability、argmax agreement、序列准确率
和 normalized edit distance。

QuantONNX 语义基线必须使用 `ORT_DISABLE_ALL`。ORT graph optimization 只能作为独立诊断；本项目
已有识别模型出现 checker/session 均通过但优化后 argmax agreement 下降的案例。

## 远端 Axera 编译产物 debug

上板/编译产物的精度核验与结构对比通过远端工具链进行，访问方式见仓库根目录 `tools.md`
（目标机 IP/账号、Pulsar2 工具链环境、工作目录、`compare_onnx_ax.py` 仿真对比脚本路径）。

关键注意点：

- **工具链会对 QuantONNX 做一次 frontend 优化**，生成的 `optimized.onnx` 节点可能被重命名/
  融合/拆分，导致基于原始 QuantONNX 生成的 Pulsar2 转换配置里 `layer_names`（attention S8 覆盖、
  requant 边界等）查找不到。排查时必须先查看 `output_dir/frontend/optimized.onnx` 的实际节点，
  再回填/重建配置。
- **编译产物与 QuantONNX 精度对齐**优先用 `compare_onnx_ax.py`（远端工作目录）逐样本对比
  axmodel 与 onnx 的 CTC logits（max_abs/MAE/argmax agreement、序列 acc/norm_edit_dis），
  以区分预处理差异、Pulsar2 frontend 优化差异与真实量化损失。
- 目标端/远端评估可用 `cache/eval_rec_ax.py`（axengine + CPU ORT 双后端，自带预处理与 CTC
  解码）与 `cache/eval_rec_onnx.py`（仅 CPU ORT，零 torch/pytorchocr 依赖）核对。
- 上板精度与 ONNX 精度对不齐时，先跑仿真对比（axmodel vs onnx），不要直接推断为训练/量化
  配置问题。
- **per-layer dump 的 dtype 陷阱**：`pulsar2 run --enable_perlayer_output` 的中间层 bin 是
  量化整型，float32 输出层是 float32。S8 层（zp=0，有符号）必须用 `int8` 读，反量化
  `q * scale`；U8 层（zp≠0，无符号）必须用 `uint8` 读，反量化 `(q - zp) * scale`。
  用错 dtype 会把 S8 负值位移成大数误判“饱和”、把 U8 当 S8 误判值域错位——本项目
  exp13 排查中两处都踩过，最终确认 attention S8/S8→U8 在 NPU 上正确、无饱和，
  真实差异源是 CNN 下采样路径的 U8 激活量化精度不足（8bit 固有代价）。
  排查记录见 `docs/axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md` §40.2（exp13 逐层对比，原独立记录已并入）。

## 常用命令

全量测试：

```bash
env PYTHONPATH="$PWD" "$PYTHON" -m pytest -q tests
```

分阶段结构定位：

```bash
env PYTHONPATH="$PWD" "$PYTHON" tools/check_model_compatibility.py --help
```

完整随机输入 QAT/QuantONNX/ORT smoke：

```bash
env PYTHONPATH="$PWD" "$PYTHON" tools/qat_smoke.py --help
```

训练：

```bash
env PYTHONPATH="$PWD" "$PYTHON" tools/train.py --help
```

从 QAT checkpoint 严格恢复并导出：

```bash
env PYTHONPATH="$PWD" "$PYTHON" tools/export_ocr_onnx.py checkpoint \
  --checkpoint /path/to/best.pt \
  --output /tmp/model_qdq.onnx
```

真实数据分阶段对比：

```bash
env PYTHONPATH="$PWD" "$PYTHON" tools/compare_qat_stages.py --help
```

PP-OCRv5 recognition 修改图或 qspec 后，使用：

- `.codex/skills/ppocr-qat-config-discovery` 为所有 det/rec 模型重新发现 Attention 区域并检查 QAT JSON；
- `.codex/skills/ppocrv5-rec-pulsar2-config` 从本次实际 QuantONNX 生成和验证 Pulsar2 配置；
- `.codex/skills/ppocr-pt2e-qat` 执行通用训练、导出和验收流程；
- `.codex/skills/ppocr-quantized-domain-fold` 把非重参化 QAT checkpoint 折叠为单分支并 finetune（量化域折叠；注意 qspec 节点名随图形态变化，训练图/折叠图 scale Mul 需同时命中）。常态化主入口为 `tools/finetune_folded.py`，公共逻辑在 `pytorchocr/quantization/folding.py`，skill 脚本为兼容封装。

## 修改与文档要求

1. 保持改动最小，不顺带重构无关 PaddleOCR 模块。
2. 不回滚工作树中来源不明的修改；本仓库经常包含用户正在进行的实验。
3. 手工修改量化规则前先阅读 `pytorchocr/quantization/` 内 vendored 的 ax_quantizer 实现与
   `references/QAT.axera/utils/` 的对应实现。
4. 公共逻辑放入 `pytorchocr`，CLI 保持薄，测试优先导入 package API。
5. 所有模型结构、qspec、预处理、loss、训练参数、observer 生命周期或 ONNX 图变换修改都必须添加
   定向回归测试。
6. 每次实验立即记录实际指标和产物路径，不把失败或历史异常指标冒充当前 baseline。历史记录应标注
   provenance，不静默覆盖。
7. 后续实验、导出、报告和交接文档不再生成或记录文件、配置、数据、checkpoint、ONNX 的哈希值；
   使用明确路径、配置内容、环境版本、Git 状态和实际指标追溯产物。
8. 不提交 `.pt`、`.pth`、`.pdparams`、大型 ONNX external data 或数据集。
9. 新增 QAT 文档按用途放入 `docs/axera_qat/plans/`、`guides/` 或 `records/`；被替代的记录移入
   `archive/`。架构和跨模块约束分别放入 `docs/architecture/` 或对应专题目录，文件名使用小写
   `snake_case`。

## 完成标准

一次 QAT 相关任务不能仅以“代码可运行”或“ONNX 可导出”结束。至少应满足：

- 定向测试和受影响的完整回归通过；
- checkpoint 能按 metadata 严格重建和加载；
- ONNX checker、QDQ 结构审计和未优化 ORT 通过；
- fake-quant-off 浮点保持和 fake-quant-on/converted 精度已量化记录；
- smoke QuantONNX 已先供结构检查；
- Axera/Pulsar2 要求的 dtype、共享量化域和保留 requant 与配置一致；
- 文档包含命令、环境、输入 shape、配置内容、指标和产物路径，不生成或记录哈希值。

# PP-OCR Harness 设计

## 1. 目的

本目录用于规划 PP-OCR 浮点训练、PT2E QAT、QuantONNX 导出和 Axera 部署的可执行验证框架。
Harness 面向开发、持续回归和实验复现，补充根目录 `README.md` 的用户操作说明，不替代用户文档。

当前仓库的公开使用路径应保持简洁：用户根据已验证的模型 profile 准备环境、权重和数据，执行训练、
导出、评估和部署命令。Harness 则负责把模型结构、量化配置、数据、环境和各阶段结果组织成可重复、
可中断、可诊断的执行链路。

## 2. 设计边界

Harness 分为三层，不能将三层混成一条面向所有人的默认命令：

| 层级 | 使用者 | 目标 | 默认行为 |
| --- | --- | --- | --- |
| 用户复现层 | 模型使用者 | 复现已验证的训练/导出/评估结果 | 只执行稳定 profile，不执行探索性门禁 |
| 开发验证层 | 模型和量化开发者 | 发现结构、预处理、qspec、权重和精度回归 | 执行完整阶段门禁，失败即停止 |
| 部署验收层 | 部署开发者 | 验证 Pulsar2、AXModel、仿真和板端精度 | 接入外部工具链和板端 adapter |

README 只保留用户复现层的稳定命令。模型 parity、QAT smoke、逐层 dump、失败分析和历史实验记录
属于 Harness 或内部开发资料，不重新塞回用户操作手册。

## 3. 标准执行链

Harness 的完整开发验证链路如下：

```text
读取模型 profile
    -> 检查环境、权重和数据
    -> Paddle 权重转换
    -> Paddle/PyTorch 浮点输出、loss、gradient 对齐
    -> 部署重参数化与浮点输出对齐
    -> export_for_training 浮点图对齐
    -> 根据当前 FX 图发现并检查 QAT JSON
    -> prepare QAT
    -> fake-off 浮点保持
    -> 随机输入 QAT smoke
    -> 一轮真实数据 train/validation/save/reload
    -> 正式 QAT 训练
    -> checkpoint strict reload
    -> convert PT2E
    -> 导出 QuantONNX
    -> ONNX checker/QDQ 结构检查
    -> ORT_DISABLE_ALL 和 ORT optimized 双评估
    -> Pulsar2 frontend/config/build
    -> AXModel 与 ONNX 仿真对齐
    -> 板端全量评估
```

用户复现层可以从“已验证 profile”直接进入训练或导出；开发验证层必须支持从任一阶段单独执行，
不能因为用户流程简化而删除这些阶段。

## 4. 当前工具与 Harness 阶段的映射

现有 `tools/` 目录保留单一职责 CLI，未来由 Harness 编排，不在 Harness 中复制其业务逻辑。

| Harness 阶段 | 当前工具或公共 API | 说明 |
| --- | --- | --- |
| 环境/数据检查 | `tools/audit_ocr_contract.py`、diagnostics API | 检查配置、标签、图片和输入合同 |
| 浮点结构/训练对齐 | `tools/compare_pretrained_training.py`、`tools/compare_det_training.py`、`tools/compare_rec_parity.py` | 输出、loss、梯度和参数更新对齐 |
| 模型兼容性 | `tools/check_model_compatibility.py` | 当前已有 build 到 QDQ 的分阶段雏形 |
| 重参数化对齐 | `tools/compare_reparameterization.py` | 折叠前后浮点行为检查 |
| prepared float 保持 | `tools/compare_pt2e_float_preservation.py` | exported float 与 fake-off 对比 |
| QAT smoke | `tools/qat_smoke.py` | prepare、backward、convert、QuantONNX、ORT smoke |
| 训练/恢复 | `tools/train.py` | profile、保存、验证和 strict resume |
| QAT 阶段对比 | `tools/compare_qat_stages.py`、`tools/compare_pt2e_quant_stages.py` | fake-on、converted 和 ONNX 对比 |
| 导出 | `tools/export_ocr_onnx.py` | checkpoint 重建、strict load 和 QuantONNX 导出 |
| QDQ/ORT | `tools/verify_qat_onnx.py`、`tools/evaluate_onnx.py` | 结构检查和双 ORT 指标 |
| Pulsar2 配置 | `.codex/skills/ppocr-pulsar2-config` | 通用配置及模型 profile |
| 板端部署 | `axera/` 下脚本和文档 | 输入准备、仿真对比和板端评估 |

第一版 Harness 应优先复用这些入口，新增逻辑放到 `pytorchocr/` 公共 API；Harness 只做参数解析、
阶段编排、状态管理和报告落盘。

## 5. Profile 与合同

每个模型 profile 应将以下内容绑定在一起：

- task：`det` 或 `rec`；
- Paddle 模型 YAML 和转换后的浮点权重；
- QAT JSON 和训练 profile；
- 输入 shape、batch policy、重参数化方式和 keep-BN；
- 预处理模式。检测默认与 PaddleOCR 固定尺寸 resize 对齐，居中 letterbox 只能作为显式实验选项；
- 训练图与部署图输出合同。检测训练保留 shrink、threshold、binary 和辅助监督，部署只保留 shrink；
  识别训练保留完整训练分支，部署只保留 CTC logits；
- optimizer、学习率、observer 生命周期和动态 shape 合同；
- 每个阶段的阈值、必需产物和允许的外部依赖。

模型代码、输入 shape、Torch 版本、重参数化方式或 qspec 变化后，Harness 必须强制从浮点权重重新
prepare，禁止跨图恢复 prepared QAT checkpoint。QAT baseline 使用 FP32 激活，observer 训练期间保持
启用，optimizer 只允许 profile 声明的 AdamW 或 SGD。

v6-rec 是当前主线；v6-det 有已验证的端到端路线；v5 模型属于独立的兼容/历史 profile。v5-rec 的
Pulsar2 Attention 规则通过通用 skill 的 `ppocrv5-rec` profile 使用，不应扩展为所有 det/rec 的默认
规则。

## 6. 阶段门禁

每个阶段都必须返回明确的 `passed` 或 `failed`，并记录首个失败边界。建议的最低门禁如下：

| 阶段 | 通过条件 |
| --- | --- |
| 环境 | Torch、ONNX、ORT 可导入，CUDA/GPU 状态符合 profile |
| 数据 | 标签可读，图片可解码，样本数和预处理合同符合预期 |
| 权重转换 | 无未解释的 missing/unexpected key |
| float parity | 输出、loss、gradient 在 profile 阈值内 |
| reparameterization | 折叠前后输出在阈值内 |
| prepare | 当前 `export_for_training` 图中的 QAT 节点和区域规则全部命中 |
| fake-off | prepared 图与 exported float 保持 |
| fake-on/backward | loss、输出、梯度均 finite，不跳过 batch |
| real-data gate | 至少一轮训练、验证、保存和 strict reload 成功 |
| convert | converted PT2E 输出 finite，量化结构符合合同 |
| QuantONNX | checker、QDQ dtype/scale/zero-point/axis 和输出合同通过 |
| ORT | `ORT_DISABLE_ALL` 为语义基线，optimized ORT 独立记录 |
| Axera | frontend 节点、Pulsar2 配置、AXModel 和输入输出合同匹配 |
| board | AXModel 与 ORT 的任务指标和数值差异在 profile 阈值内 |

检测应按 shrink、threshold、binary map，再到 precision、recall、hmean 定位；识别应按 CTC logits、
softmax、argmax agreement、sequence accuracy 和 normalized edit distance 定位。板端与 ORT 不一致时，
先做 axmodel/ONNX 仿真，再进入 per-layer dump，不能直接归因于训练或 qspec。

## 7. 产物与报告合同

每次运行使用独立目录，建议结构如下：

```text
runs/<run>/
├── manifest.json
├── environment.json
├── contract.json
├── stages/
│   ├── 01_convert.json
│   ├── 02_float_parity.json
│   ├── 03_prepare.json
│   ├── 04_qat_smoke.json
│   ├── 05_real_epoch.json
│   ├── 06_train.json
│   ├── 07_export.json
│   ├── 08_onnx_verify.json
│   ├── 09_ort_eval.json
│   └── 10_board_eval.json
├── checkpoints/
├── exports/
└── logs/
```

`manifest.json` 至少记录：模型和 QAT 配置路径、权重路径、输入 shape、batch policy、预处理、
重参数化/keep-BN、optimizer、设备、代码版本状态、阶段状态、指标和最终产物路径。报告必须区分：

- 训练内验证集与独立全量验证集；
- 50 样本仿真子集与全量板端指标；
- ORT 未优化与 ORT 图优化指标；
- 当前实验、历史实验和失败实验。

按照项目合同，不新增或记录文件、配置、数据、checkpoint、ONNX 的 hash；使用明确路径、配置内容、
环境版本、Git 状态和实际指标追溯产物。

失败报告建议采用以下形态：

```json
{
  "status": "failed",
  "stage": "float_parity",
  "error_type": "OutputMismatch",
  "first_failed_boundary": "backbone.stage2",
  "suggested_action": "检查 Paddle/PyTorch 预处理和权重映射"
}
```

## 8. 重跑与恢复

Harness 应支持：

1. `--stop-after <stage>`：只执行到指定阶段，用于快速确认环境、图或导出问题；
2. `--start-from <stage>`：复用已经通过合同的产物，从指定阶段继续；
3. 阶段级幂等：已存在的产物不能静默覆盖，除非显式指定新的 run 或覆盖策略；
4. 失败保留：失败阶段的日志、输入和报告保留，不能只返回一行错误；
5. strict resume：checkpoint 必须按 metadata 重建模型，模型 YAML、QAT JSON、输入 shape、optimizer、
   重参数化、keep-BN 和 batch policy 变化时拒绝恢复；
6. 外部 adapter 独立：Pulsar2 和板端失败不应破坏本地 QuantONNX、ORT 和阶段报告。

## 9. 计划中的目录

第一版实现可采用以下结构：

```text
harness/
├── README.md
├── run.py
├── contracts/
│   ├── ppocrv6_small_rec.yml
│   ├── ppocrv6_small_det.yml
│   └── ppocrv5_rec.yml
├── stages/
│   ├── environment.py
│   ├── data.py
│   ├── conversion.py
│   ├── parity.py
│   ├── qat.py
│   ├── export.py
│   ├── ort.py
│   └── axera.py
├── reporters/
│   ├── json_report.py
│   └── markdown_report.py
└── adapters/
    ├── local.py
    ├── pulsar2.py
    └── board.py
```

实现时应遵守：

- `tools/` 保留单功能 CLI，不把 CLI 私有 helper 作为 Harness API；
- 可复用模型、数据、量化和诊断逻辑放在 `pytorchocr/`；
- 阶段报告使用稳定 schema，便于 CI 和人工查看；
- 不把远端 IP、账号、绝对数据路径和工具链目录写入 profile；
- 不在导出完成后手工补写 QDQ，也不通过放宽 qparam 容差隐藏量化域错误；
- 随着 v6-rec 主线推进，先实现本地闭环，再接入 Pulsar2 和板端 adapter。

## 10. 与用户文档的关系

用户文档只描述已验证路线：

```text
环境安装
-> 数据路径准备
-> 已验证 profile 的训练/QAT 命令
-> QuantONNX 导出
-> ORT 评估
-> Axera 部署链接
```

Harness 的开发门禁、实验 provenance、失败记录和逐层诊断不应成为用户的必做步骤。若某个 profile
尚未通过 Harness 的完整门禁，则只能作为开发实验记录，不能在 README 中声明为已验证的公开复现路线。

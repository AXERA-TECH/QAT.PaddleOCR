# 项目文档索引

本目录保存 PP-OCR PyTorch 结构复现、PT2E QAT、QuantONNX 导出和 Axera/Pulsar2 适配过程中的
设计、计划、操作说明与实验指标。项目安装、完整操作流程和工具入口统一从仓库根目录的
[`README.md`](../README.md) 开始；命令的完整参数以对应工具的 `--help` 为准。

## 目录约定

| 目录 | 内容 | 使用原则 |
| --- | --- | --- |
| `architecture/` | PP-OCR 模型结构和实现差异 | 修改 converter 或训练图前阅读 |
| `axera_qat/plans/` | 当前有效的任务拆分、阶段门禁和验收标准 | 后续工作以这里的计划为准 |
| `axera_qat/guides/` | 可重复执行的导出、转换和交付流程 | 操作时优先使用，不从实验日志拼命令 |
| `axera_qat/records/` | 当前模型的实验过程、配置、指标和产物 | 新实验指标持续追加到对应记录 |
| `axera_qat/archive/` | 已结束或被新路线替代的基线与调研 | 只用于追溯，不作为当前精度结论 |
| `references/` | 外部工具链和量化规则摘要 | 作为 Axera 算子、dtype 和量化域约束来源 |
| `archive/project/` | 已完成的仓库结构整理记录 | 只在追查迁移决策时阅读 |

所有 Markdown 文件名使用小写 `snake_case`。新增 QAT 文档应按用途进入 `plans/`、`guides/` 或
`records/`，不再直接堆放到 `docs/axera_qat/` 根目录。

## 推荐阅读顺序

1. 阅读 [Axera QAT 路线总览](axera_qat/readme.md)，确认当前模型优先级和阶段门禁。
2. 阅读 [pretrained 训练结构复现计划](axera_qat/plans/pretrained_training_structure_plan.md)，确认
   Paddle 与 PyTorch 的训练图、部署图和输出头合同。
3. 按 [浮点与 QAT 精度恢复总计划](axera_qat/plans/model_accuracy_validation_plan.md) 完成逐阶段验证。
4. 从 `records/` 中读取对应模型的最新指标，不使用 `archive/` 的历史指标作为验收结果。
5. 导出和交付阶段使用 `guides/`，并从最终 QuantONNX 重新生成 Pulsar2 配置。

## 架构与参考

| 文档 | 内容 |
| --- | --- |
| [PP-OCRv5/v6 架构与优化](architecture/ppocrv5v6_architecture_optimizations.md) | 对照 Paddle 官方结构，说明 v5/v6 的 backbone、neck、head、训练分支和部署优化 |
| [数据增强与多尺度训练](architecture/data_augmentation.md) | PytorchOCR 增强迁移来源、det/rec pipeline、profile 开关和 QAT 默认合同 |
| [Paddle/Axera 量化信息](references/quant_info.md) | 汇总 Paddle QAT、ONNX QDQ、Axera dtype 和量化域约束 |

## QAT 计划

| 文档 | 内容 |
| --- | --- |
| [浮点与 QAT 精度恢复总计划](axera_qat/plans/model_accuracy_validation_plan.md) | det/rec 从 Paddle 到 QuantONNX 的阶段拆分、指标和门禁 |
| [pretrained 训练结构复现](axera_qat/plans/pretrained_training_structure_plan.md) | v5/v6 det/rec 完整训练图、权重转换和部署投影计划 |
| [检测精度恢复](axera_qat/plans/detection_accuracy_recovery_plan.md) | DB 检测模型的数据处理、输出图、损失和 metric 专项计划 |
| [识别精度恢复](axera_qat/plans/recognition_accuracy_recovery_plan.md) | CTC/MultiHead 识别模型的预处理、解码和精度专项计划 |

## 操作指南

| 文档 | 内容 |
| --- | --- |
| [训练与推理 ONNX 导出](axera_qat/guides/training_onnx_export.md) | 四类模型浮点训练 ONNX、完整训练 QuantONNX 和去辅助分支推理 QuantONNX 的导出合同 |
| [初始化 Observer QuantONNX](axera_qat/guides/initialized_observer_quantonnx_export.md) | 不训练、不校准并保持激活初始 qparams 的导出流程 |
| [Pulsar2/AXModel 交付](axera_qat/guides/pulsar2_axmodel_handoff.md) | QuantONNX 检查、Pulsar2 配置、编译和板端验收清单 |

## 当前实验记录

| 文档 | 内容 |
| --- | --- |
| [精度验证实施记录](axera_qat/records/model_accuracy_validation_results.md) | 总计划各阶段的真实命令、框架对齐指标和结论 |
| [v5 mobile det QAT 训练](axera_qat/records/icdar2015_ppocrv5_mobile_det_qat_training.md) | ICDAR2015 检测训练参数、checkpoint 和精度记录 |
| [v5 mobile rec QAT smoke](axera_qat/records/ppocrv5_mobile_rec_qat_smoke.md) | 识别模型浮点转换、PT2E 和 QuantONNX 结构 smoke 记录 |
| [v5 mobile rec QAT 训练](axera_qat/records/icdar2015_ppocrv5_mobile_rec_qat_training.md) | 识别 QAT 多轮实验、Exp2/Exp3 门禁、QuantONNX 审计、SGD 和动态训练高度合同 |
| [v5/v4 QAT 兼容性](axera_qat/records/ppocrv5_v4_compatibility.md) | mobile/server 与 v4/v5 模型结构、Conv-BN 和量化兼容性记录 |

## 历史归档

下列内容保留实验来源和早期决策背景，但已经被当前 v5/v6 结构复现与精度计划取代。

| 文档 | 归档原因 |
| --- | --- |
| [早期检测 QAT 训练](axera_qat/archive/baselines/icdar2015_det_qat_training.md) | PP-OCRv6 检测早期训练与工程验证基线 |
| [早期识别 QAT 训练](axera_qat/archive/baselines/icdar2015_rec_qat_training.md) | PP-OCRv6 识别早期训练基线 |
| [早期 QAT metric 验证](axera_qat/archive/baselines/icdar2015_qat_metric_validation.md) | 初期 det/rec metric 实现和验证记录 |
| [Ultralytics 复用评估](axera_qat/archive/migration/ultralytics_reuse_assessment.md) | 路线选择前的能力、技巧和风险调研 |
| [QAT.YOLO 方法迁移](axera_qat/archive/migration/qat_yolo_migration.md) | 已完成迁移过程和验证来源 |
| [项目结构迁移计划](archive/project/project_structure_reorganization_plan.md) | route2 合并到当前仓库的历史计划与决策 |
| [tools/tests 整理记录](archive/project/tools_tests_reorganization.md) | 工具职责拆分和目录重构记录 |

## 维护规则

1. 计划文档定义未完成工作和验收条件；实验数值只写入记录文档。
2. 每次训练记录 profile、命令、环境、checkpoint、QuantONNX 和阶段指标，不生成或记录哈希值。
3. smoke QuantONNX 必须先进行人工结构检查，再启动正式多 epoch QAT。
4. 旧路线被替代时移动到 `archive/`，保留原始内容并在本索引中说明原因。
5. 移动或重命名文档后必须检查仓库内 Markdown 相对链接。

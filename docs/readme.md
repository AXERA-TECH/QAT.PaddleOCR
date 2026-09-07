# 项目文档索引

`docs/` 保存公开文档索引、架构参考和可选训练指南。模型主流程、部署说明和内部设计/实验记录分别维护
在以下公开入口和本地开发目录中。

## 使用入口

| 文档 | 用途 |
| --- | --- |
| [根目录 README](../README.md) | PP-OCRv6 浮点/QAT 复现总入口 |
| [PP-OCRv5 QAT 使用指南](../README-v5.md) | PP-OCRv5 模型复现入口 |
| [Axera 部署与板端精度验证](../axera/README.md) | Pulsar2 转换、AXModel 部署和板端验证 |

## 可选训练指南

| 文档 | 用途 |
| --- | --- |
| [KD 蒸馏训练](guides/kd_training.md) | 浮点训练和 PT2E QAT 的 Teacher-Student 蒸馏配置 |

## 公开参考

| 文档 | 用途 |
| --- | --- |
| [PP-OCRv5/v6 架构与优化](architecture/ppocrv5v6_architecture_optimizations.md) | 模型结构、训练/部署图和架构差异 |
| [QARepVGG 量化参考](architecture/qarepvgg_quantization_reference.md) | 重参数化结构、量化域和 QAT 设计参考 |

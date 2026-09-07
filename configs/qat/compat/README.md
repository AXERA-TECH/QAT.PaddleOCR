# PP-OCRv5/v4 兼容配置

这些 YAML 只保留 `Architecture`、字典和输入 shape 等 route2 构图所需字段，不复制 Paddle 的训练
增强和优化器配置。QAT 训练仍必须为每个模型新建独立 profile 与量化 JSON，并从浮点权重开始。

当前已验证：

| 模型 | 随机权重浮点构图 | PT2E QAT | QuantONNX | 状态 |
|---|:---:|:---:|:---:|---|
| PP-OCRv5 mobile det | 通过 | 通过 | 通过 | 随机权重兼容 smoke 通过 |
| PP-OCRv4 mobile det | 通过 | 通过 | 通过 | 随机权重兼容 smoke 通过 |
| PP-OCRv5 mobile rec CTC | 通过 | 通过 | 通过 | 真实权重转换、浮点对齐和专属 QAT smoke 通过 |
| PP-OCRv4 mobile rec CTC | 通过 | 通过 | 通过 | 随机权重兼容 smoke 通过 |
| PP-OCRv5 server det | 通过 | 通过 | 通过 | PPHGNetV2 Conv-BN 融合后通过 |
| PP-OCRv4 server det | 通过 | 通过 | 通过 | PPHGNet Conv-BN 融合后通过 |
| PP-OCRv5 server rec CTC | 通过 | 通过 | 通过 | PPHGNetV2 Conv-BN 融合后通过 |
| PP-OCRv4 server rec CTC | 通过 | 通过 | 通过 | PPHGNet Conv-BN 融合后通过 |

除表中单独注明的 v5 mobile det/rec 外，PT2E/QuantONNX 结果使用随机权重和 v6 全局 U8/S8 JSON，
只用于算子覆盖，不表示已支持训练或部署。下一步必须完成其余模型的权重转换、Paddle/PyTorch 对齐、模型专属 QAT JSON、真实数据训练和 Axera
编译。此前 server 失败的根因是 PPHGNet/PPHGNetV2 的 Conv-BN 未在 PT2E 捕获前融合，已在
backbone 的 `rep()` 中修复；没有对 ONNX 手工补 QDQ。

# 已验证的 Pulsar2 配置

本目录保存已完成 Pulsar2 编译，并有仿真或 AX650 板端验证记录的 QuantONNX 转换配置。
板端与仿真脚本位于上级 `axera/` 目录，包括 `eval_board_det.py`、`eval_board_rec.py`、
`compare_det_onnx_ax.py` 和 `compare_rec_onnx_ax_batch.py`。

| 文件 | 模型 | 目标 | 验证记录 |
| --- | --- | --- | --- |
| [`ppocrv6_small_det_u8s8_keep_bn.json`](ppocrv6_small_det_u8s8_keep_bn.json) | PP-OCRv6 small det U8/S8 + keep-BN | AX650 / NPU1 | 已在 `axera/README.md` 的检测部署流程中验证 |
| [`ppocrv6_small_rec_u8s8_attn_s8.json`](ppocrv6_small_rec_u8s8_attn_s8.json) | PP-OCRv6 small rec U8/S8 + Attention S8 | AX650 / NPU3 | 已在 `axera/README.md` 的识别部署流程中验证 |
| [`ppocrv6_small_rec_w8a16_attn_s16.json`](ppocrv6_small_rec_w8a16_attn_s16.json) | PP-OCRv6 small rec W8A16 + Attention S16 | AX650 / NPU3 | Pulsar2 7.0、50 张仿真对比和两套全量板端评估通过 |
| [`ppocrv5_mobile_rec_u8s8_attn_s8_downsample_s16.json`](ppocrv5_mobile_rec_u8s8_attn_s8_downsample_s16.json) | PP-OCRv5 mobile rec U8/S8 + Attention S8 + downsample S16 | AX650 / NPU3 | 已验证的 v5 配置 |

这些配置是对应模型中实际使用并验证通过的版本，不是通用模板。部分历史配置仍指向原实验目录；
W8A16 配置的 `input` 与根 README 的标准导出路径一致。所有配置的 `calibration_dataset` 和
`output_dir` 都应在迁移到 Pulsar2 工作目录后按实际文件位置确认。识别模型若包含 frontend
优化后的 `layer_configs` 节点名，不能根据新的 QuantONNX 直接沿用；QuantONNX 或 frontend 图发生
变化后应重新生成或审计配置。

W8A16 配置显式声明了对应 frontend 图中的两个 QKV `FullyConnected` 和两组 Attention 数据流为
S16；其余普通路径保持 QuantONNX 图内的 U16 激活，权重为 S8。这里的 `layer_names` 是该次
Pulsar2 frontend 产物的节点名，不是 PT2E FX 名称。QuantONNX、Pulsar2 版本或 frontend 图变化后，
必须重新生成并校验这些映射，不能直接沿用。`calibration_dataset` 只满足工具链字段要求，不会覆盖
QAT qparams。配置使用可替换的模型、输出和校准集路径，复制到 Pulsar2 工作目录后按实际路径修改。

需要为其他 PP-OCR det/rec 或已完成 QAT 的 QuantONNX 生成配置时，使用通用 skill：

```text
.codex/skills/ppocr-pulsar2-config/
```

该 skill 从实际 QuantONNX 读取输入输出接口并生成通用配置骨架；模型专属的
`quant.layer_configs` 必须显式提供，并可使用 frontend 优化图校验节点名。PP-OCRv5 rec 的 Attention
区域使用同一 skill 的 profile：

```text
.codex/skills/ppocr-pulsar2-config --profile ppocrv5-rec
```

通用编译命令：

```bash
pulsar2 build \
  --input onnx/<model>.onnx \
  --config config/<model>.json \
  --output_dir output/<model>
```

编译前后的 QuantONNX、frontend `optimized.onnx`、编译日志和 `compiled.axmodel` 应一并保留。完整
部署和板端验证流程见 [`../README.md`](../README.md)。

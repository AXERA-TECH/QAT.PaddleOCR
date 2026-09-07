# 已验证的 Pulsar2 配置

本目录保存已完成 Pulsar2 编译，并有仿真或 AX650 板端验证记录的 QuantONNX 转换配置。
板端与仿真脚本位于上级 `axera/` 目录，包括 `eval_board_det.py`、`eval_board_rec.py`、
`compare_det_onnx_ax.py` 和 `compare_rec_onnx_ax_batch.py`。

| 文件 | 模型 | 目标 | 验证记录 |
| --- | --- | --- | --- |
| [`config-det-exp20c.json`](config-det-exp20c.json) | PP-OCRv6 small det exp20c | AX650 / NPU1 | 已在 `axera/README.md` 的检测部署流程中验证 |
| [`config-rec-exp22a.json`](config-rec-exp22a.json) | PP-OCRv6 small rec exp22a | AX650 / NPU3 | 已在 `axera/README.md` 的识别部署流程中验证 |
| [`ppocrv5_mobile_rec_exp16_u8s8.json`](ppocrv5_mobile_rec_exp16_u8s8.json) | PP-OCRv5 mobile rec exp16 | AX650 / NPU3 | 已验证的 v5 配置 |

这些配置是对应实验中实际使用并验证通过的版本，不是通用模板。配置中的 `input`、
`calibration_dataset` 和 `output_dir` 仍指向原实验目录，迁移到新的 Pulsar2 工作目录后必须按实际
文件位置调整。识别模型的 `layer_configs` 使用 frontend 优化后的节点名，不能根据新的 QuantONNX
直接沿用；如果 QuantONNX 或 frontend 图发生变化，应重新生成或审计配置。

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

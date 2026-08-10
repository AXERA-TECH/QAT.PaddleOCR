# Tools / Tests 结构整理记录

状态：2026-08-05 已完成主体整理和核心回归。

## 1. 问题边界

`tests/` 和 `tools/` 不应合并：前者验证稳定接口，后者提供训练、导出和诊断命令。当前问题是公共
实现落在 CLI 文件中，导致测试和其他工具把 CLI 当作库使用。

已确认的问题：

1. `tests/test_training_profile.py` 从 `tools/train.py` 导入 optimizer、scheduler 和 best metric 逻辑；
2. stage comparison 测试直接导入多个 tools 的 `_private` helper；
3. `audit_rec_qat_checkpoint.py`、`diagnose_rec_observer_modes.py`、三个 stage comparison 工具相互导入
   dataset、CTC collapse、误差累计和 checkpoint prepare 逻辑；
4. checkpoint metadata relocation、模型重建、PT2E prepare 和严格 state dict 加载在多个工具中重复；
5. 原 tools ONNX backend 与 `pytorchocr/quantization/validation.py` 重复解析 ONNX constant/qparam；
6. det/rec parity 工具仍按迁移前目录层级推导 PaddleOCR checkout；
7. `check_model_compatibility.py` 与 `qat_smoke.py` 都实现 build、prepare、backward、convert、ONNX、QDQ
   的随机输入链路。

## 2. 目标结构

```text
pytorchocr/training/
  factory.py              dataset、criterion、optimizer、scheduler、best metric
pytorchocr/diagnostics/
  data.py                 诊断数据集构建和 sample identity
  errors.py               tensor error、CTC collapse、recognition pair stats
  checkpoint.py           checkpoint metadata、PT2E prepared graph 严格重建
  contract.py             数据标签、字符覆盖和 accuracy contract 审计
pytorchocr/utils/
  hashing.py              跨模块复用的文件 SHA256
pytorchocr/quantization/
  validation.py           共享 ONNX constant/qparam 解析
tools/
  *.py                    argparse、流程编排和 JSON/ONNX 落盘
tests/
  *.py                    优先导入 pytorchocr 公共模块
```

ONNX 导出后端最终收敛到 `pytorchocr/quantization/onnx_export.py`，其低层 ONNX 图读取 helper 从
`pytorchocr.quantization.validation` 复用；tools 只保留用户 CLI。

## 3. 执行阶段

### P0：路径和职责修复

- 修复 parity 工具对 `references/PaddleOCR` 的定位；
- 公共训练 factory 移出 `tools/train.py`；
- tests 不再从 `tools/train.py` 导入训练实现。

### P1：诊断公共层

- 抽取 `ErrorAccumulator`、CTC collapse 和 recognition pair stats；
- 抽取 metadata 驱动的数据集构建；
- 抽取 QAT checkpoint prepared graph 严格重建；
- tools 之间不再导入 `_private` helper。

### P2：ONNX 和 CLI 收敛

- ONNX constant/qparam 读取只有一个实现；
- Float、checkpoint、初始化 observer 和完整训练结构 ONNX 导出统一为 `tools/export_ocr_onnx.py` 的
  `checkpoint/initialized/training/audit` 子命令；
- `check_model_compatibility.py` 保留 staged stop，用于定位失败阶段；
- `qat_smoke.py` 保留完整 ORT/ONNX 数值和结构门禁，不与 staged compatibility CLI 合并；
- det/rec random parity 保留现有命令，但共享 framework parity 实现；后续再决定是否删除兼容入口。

## 4. 兼容和验收

1. 不修改 QAT qspec、PT2E 图、observer 生命周期、数据预处理或 loss；
2. 不修改现有 CLI 参数和 JSON 字段；
3. 历史导出入口在统一 CLI 和全部文档迁移完成后删除；
4. tests 不通过 shell 调用生产逻辑，也不依赖 tools 的 `_private` 函数；ONNX export 测试直接导入
   `pytorchocr.quantization.onnx_export`；
5. 每阶段运行 training、stage comparison、Axera quantizer、ONNX export 和 model compatibility 测试；
6. 最终使用保留的 PP-OCRv5 rec checkpoint 做一次 metadata relocation + strict reload smoke。

## 5. 执行记录

### 2026-08-05 P0

1. 新增 `pytorchocr/training/factory.py`，接管 dataset、criterion、optimizer、scheduler 和 best
   validation 更新；`tools/train.py` 只保留训练流程和 CLI；
2. `tests/test_training_profile.py`、`tests/test_trainer.py` 改为验证 package API；
3. 修复三个 framework parity CLI 对 `references/PaddleOCR` 和 `references/PytorchOCR` 的路径定位；
4. training 定向回归：`32 passed`。

### 2026-08-05 P1

1. 新增 `pytorchocr/diagnostics/{checkpoint,data,errors,onnx,qat,recognition}.py`；
2. 七个 QAT/ONNX 诊断 CLI 改为复用公共 checkpoint 重建、数据集、CTC 和误差统计 API；
3. stage 和 model compatibility 测试不再导入 tools 私有 helper；
4. diagnostics 定向回归：`33 passed`。

### 2026-08-05 P2

1. `constant_tensors` 和包含 axis 的 `qparam_key` 统一到
   `pytorchocr/quantization/validation.py`；
2. 新增 `pytorchocr/diagnostics/contract.py`，`tools/audit_ocr_contract.py` 收敛为参数、环境、落盘
   编排；`tests/test_accuracy_contract.py` 改为 package API；
3. 文件 SHA256 统一到 `pytorchocr/utils/hashing.py`，training profile、accuracy contract 和 float
   framework comparison 共用同一实现；
4. 保留两个随机输入 CLI：`check_model_compatibility.py` 支持 `--stop-after`，`qat_smoke.py` 执行
   完整 prepared/converted/QuantONNX/ORT 检查；
5. 核心组合回归：`63 passed, 12 subtests passed`；contract/profile 补充回归：`25 passed`。
6. 使用保留 checkpoint
   `runs/icdar2015_ppocrv5_mobile_rec_native_silu_qat_corrected_no_warmup_20260805/best.pt`
   完成 metadata relocation、prepared 图重建、`strict=True` state dict 加载和 QuantONNX 导出；临时输出为
   `/tmp/ppocrv5_rec_tools_reorganization_strict_reload.onnx`。

该真实 checkpoint 导出图统计：`982` prepared nodes、`1340` converted nodes、`943` ONNX nodes、
`244` QuantizeLinear、`429` DequantizeLinear、`0` BatchNormalization、`0` direct/redundant/requantize
DQ-Q，且 Conv 输出均已量化。ORT graph optimization disabled 时 recognition argmax agreement 为
`1.0`；启用 ORT optimization 后为 `0.975`，因此优化执行结果仍只能作为独立诊断，不能替代未优化
ORT 的 QuantONNX 语义基线。

最终执行全部 `tests/`：`95 passed, 12 subtests passed`。`pytorchocr/diagnostics`、
`pytorchocr/training`、共享 hashing、全部 tools 和 tests 的 `compileall` 同时通过。

### 当前依赖边界

`tests/` 已不再导入 tools。`tools/export_ocr_onnx.py`、smoke/compatibility CLI 和
`tests/test_onnx_export.py` 共同复用 `pytorchocr.quantization.onnx_export`。

此次整理未修改 QAT qspec、PT2E prepare/convert、observer/fake-quant 状态、训练预处理、loss、CLI
参数或 JSON 报告字段。

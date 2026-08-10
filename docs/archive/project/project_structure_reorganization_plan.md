# 项目结构整理与迁移计划

状态：目录迁移和主包内部整理已执行；迁移后验证进行中。checkpoint 清理记录见 2.2 节。

日期：2026-08-05

## 1. 整理目标

迁移前真正持续开发的项目是：

```text
/home/heqi/project/PaddleOCR/route2/PaddleOCR2Pytorch-QAT
```

已将其提升为：

```text
/home/heqi/project/PaddleOCR
```

原外层 PaddleOCR 代码现只作为 Paddle 框架实现和模型配置参考，不再占据主工程根目录。迁移满足：

1. 保留 route2 的 Git 历史、当前修改、未跟踪文件和已确认保留的实验产物；
2. 参考源码有明确、稳定的位置，不与可删除缓存混淆；
3. QAT 代码、配置、测试、文档、指标、训练产物和 QuantONNX 分层存放；
4. 迁移后训练、严格恢复、PT2E prepare/convert、QuantONNX 导出命令可以从新根目录执行；
5. 目录迁移阶段不再清理文件、不重写历史指标、不进行代码重构。

## 2. 当前结构结论

### 2.1 Git 边界

```text
外层 /home/heqi/project/PaddleOCR/.git
  空目录，不是有效 Git 仓库；外层 git rev-parse 失败。

route2/PaddleOCR2Pytorch-QAT/.git
  有效 Git 仓库；origin 为 frotms/PaddleOCR2Pytorch。

tmp/PytorchOCR/.git
  独立参考仓库。

tmp/QAT.axera/.git
  独立参考仓库。
```

因此应提升 route2 的 `.git`，不能让外层空 `.git` 覆盖它。迁移前仍需保存 route2 的 status、tracked
diff、untracked manifest 和关键文件 SHA256，因为当前工作树包含大量未提交修改及未跟踪文件。

### 2.2 体积

2026-08-05 checkpoint 清理后的当前体积：

```text
route2/PaddleOCR2Pytorch-QAT:  841 MB
  output/:                     239 MB（du）/ 205 MB（文件字节统计）
  主代码、配置、文档、测试:   约 20 MB
  浮点/Paddle 权重:            约 581 MB

当前外层：
  tmp/:                        562 MB
  models/:                     762 MB
  output/:                     885 MB
  weights_pretrained/:         129 MB
```

清理前 `output/` 为 13,678,764,665 字节，清理后为 204,887,502 字节，共释放约 13.47 GB。
只删除 `.pt` checkpoint；ONNX、日志、JSON、profile、文档和源码均未删除。

保留的 4 个正式 best checkpoint：

```text
9ecd25535b6e5b9f6c81e382e6e3fe9aad47a7170867a855e7e904ed2d4b3e9a  PP-OCRv5 rec，acc 0.49350
ec7566884cf2f4aa951c47e9a52f1fc7b8683531b18dfc7107d09f38c9a10485  PP-OCRv5 det
69b8ffc47405f26927588a0cf1b62dd41a1463c80af8debce00f8dfd30d38e09  PP-OCRv6 det
fd6e107ea9841198bf7370cee6c54dd3a668a230438c400dc94da0571885e2cc  PP-OCRv6 rec
```

已删除：所有逐 epoch 和 `last.pt`；所有 smoke/pre-fix checkpoint；旧错误 CTC/backbone 合同的
PP-OCRv5 rec baseline 与 repro checkpoint。旧实验的指标记录仍保留在文档和 JSON 中。

### 2.3 路径耦合

主工程中约有 79 处旧绝对路径或旧 tmp 导出路径引用，主要包括：

```text
/home/heqi/project/PaddleOCR/route2/PaddleOCR2Pytorch-QAT
/home/heqi/project/PaddleOCR/tmp/route2_qat_exports
/home/heqi/project/PaddleOCR/tmp/frontend
```

引用分为三类，不能统一机械替换：

1. `USAGE.md`、skill、命令文档：必须改为新路径；
2. `artifacts/accuracy_contract/*.json`：仍会作为工具输入，必须改为新路径并重新验证；
3. 历史报告 JSON：保留原始路径作为实验 provenance，不修改数值记录，另加迁移映射说明。

## 3. 已落地结构

```text
/home/heqi/project/PaddleOCR/
├── .git/                    # 从 route2 提升的有效 Git 仓库
├── .codex/skills/           # 合并三个本项目 skill
├── configs/                 # 模型、QAT、训练 profile
├── converter/               # Paddle -> PyTorch 转换
├── pytorchocr/              # 主 Python 包
├── tools/                   # 训练、诊断、导出、验证 CLI
├── tests/                   # 回归测试
├── docs/                    # 设计、计划、实验记录
├── artifacts/               # 小型 JSON 指标和结构报告
├── weights/                 # 主工程使用的 Paddle/PyTorch 浮点权重
├── runs/                    # QAT checkpoint；由原 route2/output 迁入
├── exports/
│   ├── quantonnx/           # 原 tmp/route2_qat_exports
│   └── frontend/            # 原 tmp/frontend 优化参考图
├── references/
│   ├── PaddleOCR/           # 当前外层 PaddleOCR 参考源码与其资产
│   ├── PytorchOCR/          # 原 tmp/PytorchOCR，保留独立 .git
│   └── QAT.axera/           # 原 tmp/QAT.axera，保留独立 .git
└── cache/
    ├── legacy_paddle_qat/   # 原 tmp/ppocrv6_qat、ppocrv6_qat_u8
    ├── runtime/             # home、paddle_home、pytest、__pycache__ 等
    └── migration/           # manifest、路径映射、外层空 .git 归档
```

### 为什么参考源码不用 `cache/`

`PaddleOCR`、`PytorchOCR` 和 `QAT.axera` 是量化规则、转换语义和跨框架精度排查依据，不应被当作
可随时删除的数据。建议放在 `references/` 并在主仓库 `.gitignore` 中整体忽略；`cache/` 只放可以
重新生成或明确废弃的实验目录。

主包暂不改成 `src/` layout，也不调整 `pytorchocr` import。当前工作的主要风险来自 PT2E/QAT 图和
checkpoint 合同，目录整理不应顺带引入 Python 包结构重构。

## 4. 迁移映射

| 当前路径 | 目标路径 | 策略 |
| --- | --- | --- |
| `route2/PaddleOCR2Pytorch-QAT/*` | 根目录对应路径 | 主工程提升 |
| route2 有效 `.git` | 根目录 `.git` | 原样移动 |
| 外层 PaddleOCR 源码 | `references/PaddleOCR/` | 整体保留，不清理 |
| 外层空 `.git` | `cache/migration/outer-empty.git/` | 归档，不覆盖主仓库 |
| 外层 `.codex/skills/ppocr-pt2e-qat` | 根 `.codex/skills/` | 与 route2 两个 skill 合并 |
| `tmp/PytorchOCR` | `references/PytorchOCR` | 保留嵌套 Git |
| `tmp/QAT.axera` | `references/QAT.axera` | 保留嵌套 Git |
| `tmp/route2_qat_exports` | `exports/quantonnx` | 保留全部 ONNX |
| `tmp/frontend` | `exports/frontend` | 保留 ONNX external data |
| `tmp/ppocrv6_qat*` | `cache/legacy_paddle_qat/` | 历史 Paddle QAT 实验 |
| route2 根目录权重 | `weights/` | 迁移后移动并更新合同 |
| route2 `output/` | `runs/` | 建议提升时直接使用最终名称 |
| `info/quant_info.md` | `docs/references/quant_info.md` | 提取为主工程参考文档 |

`references/PaddleOCR/` 的精确边界应排除 `route2/`、拆分后的 `tmp/`、外层 `.codex/` 和空 `.git`，
避免形成主工程递归嵌套。

## 5. 分阶段执行

### 阶段 0：冻结与清单

1. 停止所有以旧目录为 cwd 的训练、导出和 tmux 任务；
2. 保存 route2 `git status --short`、`git diff --binary`、tracked/untracked 文件清单；
3. 保存主权重、QAT JSON、profile、保留的 best checkpoint、QuantONNX 的 SHA256；
4. 记录所有嵌套 Git 的 `rev-parse HEAD` 和 remote；
5. 生成 `cache/migration/path-map.tsv`，每个旧路径只对应一个新路径。

验证：清单文件可读，磁盘空间充足，所有待移动目录位于同一文件系统。

### 阶段 1：只改变目录归属

1. 在根目录内部建立临时 staging，先移出 route2 主工程；
2. 将当前外层参考内容移入 `references/PaddleOCR/`；
3. 分离 `tmp` 下参考仓库、导出模型和 legacy 实验；
4. 将 staging 中的 route2 内容和有效 `.git` 提升到根目录；
5. 合并三个 `.codex/skills`，不覆盖同名 skill；
6. 按第 7 节确认结果决定直接使用 `runs/`，或暂时保留 `output/`。

验证：

```text
git rev-parse --show-toplevel == /home/heqi/project/PaddleOCR
git status 与迁移前主工程 status 等价
三个嵌套/主 Git HEAD 不变
关键文件 SHA256 不变
pytest 核心回归通过
```

阶段 1 验收前不做任何删除。

### 阶段 2：更新活跃路径

1. 修改 `USAGE.md`、skill、运行命令中的旧 route2 根路径；
2. 修改 `artifacts/accuracy_contract/*.json` 的可执行路径；
3. 将 QuantONNX 活跃输出统一到 `exports/quantonnx/`；
4. 给历史报告增加旧路径到新路径的说明，不重写历史 JSON；
5. 扫描旧路径，要求活跃代码/配置引用为 0。

验证：float 模型加载、QAT prepare、严格 checkpoint reload、1-batch forward、QuantONNX smoke export
和 ONNX checker 均通过。此阶段不进行正式训练。

### 阶段 3：主工程内部归一化

1. 若阶段 1 未直接处理，则将 `output/` 改名为 `runs/`；
2. 根目录 `.pth/.pdparams` 移入 `weights/`；
3. 工具参数继续显式传入路径，不新增隐式全局搜索；
4. 更新 checkpoint contract 与文档；旧 checkpoint 中保存的绝对 metadata 只做兼容映射，不修改文件；
5. 新实验必须输出到 `runs/<model>/<experiment>/`。

验证：使用现有 `best.pt` 严格恢复并复现已记录指标；使用浮点权重完成新的 smoke，但不启动多 epoch。

### 阶段 4：未来 checkpoint 策略

本轮历史 checkpoint 空间回收已经完成。后续参考 PaddleOCR 的保存时机，按 optimizer
`global_step` 保存，不再只在 epoch 结束时保存。计划新增两个训练参数：

```text
save_step_interval:   每多少个 optimizer step 触发验证和候选 checkpoint 保存
max_keep_checkpoints: 候选 checkpoint 的 Top-K 保留数量
```

`max_keep_checkpoints` 语义：

```text
5:       保留验证主指标最优的 5 个 step checkpoint，另保留 best.pt 和 last.pt
0 或 -1: 保留全部 step checkpoint，另保留 best.pt 和 last.pt
>= 2:    合法的有限 Top-K
1:       非法；有限模式的最小值为 2
< -1:    非法
```

Top-K 按模型 `Metric.main_indicator` 排序；det 使用 hmean，rec 使用 accuracy。相同指标时优先保留
global step 较新的 checkpoint。`best.pt`、`last.pt` 是独立别名，不计入 Top-K 数量，所以 K=5 时
磁盘上最多有 7 份完整权重。候选 checkpoint 必须包含 model、optimizer、scheduler、scaler、observer、
epoch/global step 和训练合同，确保任意保留点都可严格恢复。

保存索引写入 `checkpoint_index.json`，采用临时文件加原子 rename 更新；删除旧 checkpoint 必须发生在
新 checkpoint 和索引成功落盘之后。恢复训练时从索引恢复 Top-K；索引缺失时从 checkpoint metadata
重建。训练结束仍保留完整 JSONL 指标、best 严格恢复报告及对应 QuantONNX 结构报告。

该功能在目录迁移完成后单独实现和测试；迁移本身不修改 Trainer 保存逻辑。

## 6. 回滚原则

1. 所有移动使用同文件系统 rename，不在迁移中复制后再删除大文件；
2. 每完成一个阶段立即验证，失败则按 `path-map.tsv` 逆序移动；
3. Git status 或关键 SHA256 不一致时停止，不继续下一阶段；
4. 不使用 `git reset --hard`、`git clean` 或跨仓库覆盖；
5. 不在结构迁移过程中修改 QAT 图、量化配置、训练参数或模型代码。

## 7. 已确认的决策

1. **参考源码归属**：当前外层 PaddleOCR 推荐放 `references/PaddleOCR/`，而不是可删除的 `cache/`；
2. **训练产物命名**：当前只剩 239 MB，推荐主工程提升时直接将 `output/` 放为 `runs/`；
3. **历史路径**：历史 JSON 保留旧绝对路径，仅修复活跃合同、skill 和使用说明，并增加 path map；
4. **未来 checkpoint 策略**：按 optimizer step 保存；有限模式最少保留 Top-2，K=5 时保留 Top-5
   加独立 `best.pt`/`last.pt`，`0/-1` 表示全部保留。

四项决策已确认并按上述映射执行。迁移前 manifest 和 SHA256 清单位于 `cache/migration/`。

## 8. 2026-08-05 执行补充：主包内部结构

根目录提升完成后继续整理 QAT Python 包。原结构为：

```text
pytorchocr/quantization/axera/
  bridge.py
  validation.py
  onnx_export.py
  onnx_export.py
  vendor/
    ax_quantizer.py
    ax_quantizer_utils.py
    quantized_decomposed_dequantize_per_channel.py
```

该工程目前只有一套 Axera PT2E quantizer，`axera/vendor` 两级包裹没有形成可替换后端边界，反而让
训练、测试和诊断脚本的导入路径过长。迁移后的目标结构为：

```text
pytorchocr/quantization/
  __init__.py
  bridge.py
  validation.py
  ax_quantizer.py
  ax_quantizer_utils.py
  quantized_decomposed_dequantize_per_channel.py
  LICENSE
  UPSTREAM.md
tools/
  export_ocr_onnx.py
```

边界约束：

1. `pytorchocr.quantization` 承载 PT2E 构图、QDQ 验证和 ONNX 导出公共 API；
2. ONNX 导出后端位于 `pytorchocr/quantization/onnx_export.py`，统一 CLI 位于
   `tools/export_ocr_onnx.py`，主包和测试不反向导入 `tools`；
3. ONNX/QDQ 验证暂留 `pytorchocr.quantization.validation`，因为训练诊断和多个工具共用；
4. QAT.axera 的 commit、许可证和本地修改记录随源码平移，源码内容不在此次目录整理中改写；
5. 更新仓库内全部活跃 import 和测试，不保留空的 `quantization.axera.vendor` 包；旧 checkpoint 只保存
   state dict 和合同 metadata，不依赖旧 Python class import 路径；
6. 迁移验证至少覆盖 quantizer 加载、PT2E prepare、ONNX rewrite 单测、训练核心单测和导出 CLI import。

实际迁移后 `git rev-parse --show-toplevel` 为 `/home/heqi/project/PaddleOCR`，HEAD 保持
`c05307fbd61e575d55f0fef0334026c7536aec76`。提升 `.git` 时曾因受保护挂载形成临时
`.git/.git`，已通过同文件系统 rename 展平；原外层空 Git 已保存到 `cache/migration/`。额外发现的
完整 PaddleOCR2Pytorch 参考副本保留在 `references/PaddleOCR2Pytorch/`，不作为主工程运行依赖。

2026-08-05 主包整理验证：`tests/test_axera_vendor.py` 和 `tests/test_onnx_export.py` 共 19 项通过。
旧 `pytorchocr/quantization/axera/vendor` 已移除，活跃代码统一从 `pytorchocr.quantization` 导入；
ONNX 导出实现由 `pytorchocr.quantization.onnx_export` 提供。

迁移最终验证结果：

```text
Git root:                  /home/heqi/project/PaddleOCR
Git HEAD:                  c05307fbd61e575d55f0fef0334026c7536aec76
关键文件 SHA256:           53 / 53 与迁移前一致
PytorchOCR reference HEAD: 77a4ece14540a553f9b9446da0eae98a3d76b969
QAT.axera reference HEAD:  4603b160bad6b212551721ec3cd3b75895b17de3
核心 pytest:               51 passed
导出 CLI import/help:      2 / 2 通过
既有 rec QuantONNX checker: 943 nodes，full_check 通过
```

保留的 PP-OCRv5 mobile rec `acc=0.49350` checkpoint 已在新路径重建相同 prepared PT2E 图并
`strict=True` 加载。最终验证命令未传 `--model-config`、`--qat-config` 或 `--weights`，旧 metadata
自动映射后直接导出到 `/tmp/ppocrv5_rec_migration_metadata_relocated.onnx`，不替换正式产物。导出结构
和语义检查结果：

```text
float / prepared / converted nodes: 499 / 982 / 1340
ONNX nodes:                         943
Q / DQ:                            244 / 429
BatchNormalization:                0
direct / redundant DQ -> Q:        0 / 0
Concat shared qparams:             1 / 1
SiLU total / quantized / internal: 7 / 7 / 0
Hard activation total / quantized: 2 / 2
rec argmax agreement:              1.0
output finite:                     true
```

5 个历史 QAT tmux pane 的 cwd 均已从 `/tmp` 切换到新根目录。活跃代码、skill、accuracy contract 和
`USAGE.md` 不再引用旧 route2 根目录；历史报告仍保留旧路径作为 provenance。恢复合同仅接受已知旧根
到新根且模型配置相对路径完全一致的 relocation；旧根目录下的根级权重同时映射到 `weights/`。
profile/QAT hash、Torch 版本和其余图合同继续严格校验。

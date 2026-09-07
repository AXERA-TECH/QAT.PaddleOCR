# PP-OCRv4 / PP-OCRv5 / PP-OCRv6 架构与优化

本文介绍 PP-OCRv4、PP-OCRv5 和 PP-OCRv6 的主要网络结构、训练分支、部署输出和重参数化方式，
帮助读者理解本仓库的 PyTorch 浮点训练与 PT2E QAT 实现。本文只记录稳定的模型设计和接口约定，
不记录具体机器、checkpoint、训练日志或单次实验结果。

官方资料：

- [PP-OCRv4](https://www.paddleocr.ai/latest/version2.x/ppocr/blog/PP-OCRv4_introduction.html)
- [PP-OCRv5](https://www.paddleocr.ai/latest/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- [PP-OCRv6](https://www.paddleocr.ai/latest/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)

本文所述本仓库实现主要位于 `pytorchocr/modeling/`；对应的配置位于 `configs/det/`、
`configs/rec/` 和 `configs/qat/`。

## 1. 版本演进

| 维度 | PP-OCRv4 | PP-OCRv5 | PP-OCRv6 |
| --- | --- | --- | --- |
| 检测骨干 | PP-LCNetV3 | PP-LCNetV3 | PPLCNetV4 |
| 检测 Neck | RSEFPN | RSEFPN | RepLKFPN |
| 检测 Head | DBHead / PFHeadLocal | DBHead | DBHead + P2/P3/P4 辅助监督 |
| 检测 Loss | Dice + OHEM | Dice + OHEM | Dice + Focal + 辅助损失 |
| 识别骨干 | PP-LCNetV3 | PP-LCNetV3 | PPLCNetV4 |
| 识别 Neck | SVTR Lite-Neck | SVTR Lite-Neck | LightSVTR |
| 识别 Head | CTC + NRTR | CTC + NRTR | CTC + NRTR |
| 重参数化 | LCNetV3 RepLayer | LCNetV3 RepLayer | RepDWConv、Stem、FPN 中的重参数化模块 |

三代模型都区分训练语义和部署语义：训练图可以包含辅助监督、多头输出和多分支结构，部署图只保留
推理所需的主输出并执行必要的结构折叠。

## 2. 共同结构

### 2.1 检测输出

DB 检测模型的训练 Head 通常输出三张图：`shrink`、`threshold` 和 `binary`。其中：

```text
binary = sigmoid(k * (shrink - threshold))
```

`shrink` 用于文本区域预测，`threshold` 用于可微分二值化，`binary` 用于训练监督或后处理。
部署时只需要 `shrink`，因此本仓库的检测 deployment graph 只导出 shrink map；训练期的完整 DBHead
输出不能被部署输出替代。

### 2.2 识别输出

识别模型使用 CTC 作为部署主路径，并使用 NRTR/GTC 作为训练辅助路径：

```text
图像 -> backbone -> sequence neck -> CTC head -> CTC logits
                              \-> NRTR/GTC head -> auxiliary loss
```

训练图保留 CTC、NRTR/GTC 及其 targets，推理图只保留原始 CTC logits，不包含最终 Softmax。Softmax、
CTC collapse 和文本解码属于推理后处理。

### 2.3 重参数化

MobileOne、RepVGG 和 UniRepLKNet 风格模块通常在训练时使用多个分支，在部署时将分支融合为一个卷积。
以卷积和 BN 分支为例，BN 融合公式为：

```text
s = gamma / sqrt(running_var + eps)
W' = W * s
b' = beta - running_mean * s
```

随后将各分支的 `W'` 和 `b'` 相加；1x1 卷积通过零填充对齐到目标 kernel，identity 分支通过单位
卷积核表示。融合后的卷积不再需要原分支的 BN，前提是融合发生在与 BN running statistics 一致的
推理语义下。

本仓库的实现包括：

| 结构 | 实现位置 | 说明 |
| --- | --- | --- |
| v5 `LearnableRepLayer` | `pytorchocr/modeling/backbones/rec_lcnetv3.py` | 多个空间卷积分支、identity 分支和 LAB |
| v6 `RepDWConv` / `ConvBNAct` | `pytorchocr/modeling/backbones/rec_lcnetv4.py` | depthwise 重参数化和残差结构 |
| v6 `DilatedReparamBlock` | `pytorchocr/modeling/necks/db_fpn.py` | 多个膨胀卷积分支融合为大核卷积 |

## 3. PP-OCRv4

### 3.1 检测

v4 检测主要由 PP-LCNetV3、RSEFPN 和 DBHead 组成。部分配置使用 PFHeadLocal，在 DB 主分支之外
增加局部 refinement 分支，再融合基础概率图和局部概率图。

DSR 通过训练阶段逐步改变 shrink ratio；CML 通过 Student/Teacher 或互学习分支约束 response map。
这些是训练策略，不改变部署时的 DB shrink 输出合同。

### 3.2 识别

v4 识别使用 PP-LCNetV3 和 SVTR Lite-Neck。`use_guide` 时，主干特征会通过 stop-gradient 进入指导
路径，由 NRTR 提供上下文监督；CTC 仍是部署主输出。DF、Multi-Scale 和 DKD 属于数据或训练策略，
不会增加部署输出。

## 4. PP-OCRv5

### 4.1 检测结构

v5 检测版 PP-LCNetV3 使用任务专用的下采样配置，输出多级特征供 RSEFPN 融合。RSEFPN 通过自顶向下
路径和横向连接构造高分辨率 `fuse` 特征，DBHead 再从 `fuse` 生成 shrink、threshold 和 binary。

### 4.2 识别结构

v5 识别骨干使用非对称 stride，使高度逐步压缩而保留适合文本序列的宽度。SVTR neck 的主要路径为：

```text
输入特征 -> 降维 -> SVTR block -> LayerNorm -> reshape
       \-> 跳跃路径 ------------------------------/
                    -> concat -> CTC head
```

`MultiHead` 同时构建 CTC 和 NRTR。训练时返回 CTC logits、CTC neck 特征和 NRTR/GTC 输出；推理时只
返回 CTC logits。CTC 与 NRTR 的损失共同构成完整 pretrained training graph。

### 4.3 重参数化卷积与 LAB

`LearnableRepLayer` 的训练结构由 identity、多个 kernel 分支和 1x1 分支组成，分支求和后经过
`LearnableAffineBlock`（LAB）和激活函数。部署时先吸收各分支 BN，再融合卷积核，最后删除训练分支。

## 5. PP-OCRv6

### 5.1 PPLCNetV4

PPLCNetV4 通过 `det` 和 `model_size` 选择检测/识别任务及 tiny、small、medium 规模。核心 block
采用 MetaFormer 风格：

```text
x1 = SE(DWConv(x)) + x
x2 = PWConv2(GELU(PWConv1(x1))) + x1
```

可重参数化的 RepDWConv 在满足 stride 和通道条件时使用 3x3、1x1 和 identity 分支；部署时融合为
单个 depthwise 卷积。部分 channel mixer 的 BN 使用零初始化，使残差结构在训练初期接近 identity。

检测和识别共用 PPLCNetV4 的基本 block，但采用不同的 stride 配置：

- 检测输出 stride 4、8、16、32 的多级特征；
- 识别使用非对称下采样，保留文本序列方向的信息，并将特征整理为 CTC/NRTR 输入序列。

### 5.2 检测：RepLKFPN 与辅助监督

RepLKFPN 是 RSEFPN 的大核重参数化版本。其 `DilatedReparamBlock` 将多个不同 dilation 的卷积分支
组合成等效的大感受野卷积：

```text
多分支 DWConv -> PWConv -> BN/SE -> fuse
```

训练时 Neck 还返回 `aux_p4`、`aux_p3` 和 `aux_p2`。这些特征经过辅助 DB Head 生成辅助 maps，并以
独立权重参与训练损失；部署图只保留 `fuse` 和最终 shrink map。

v6 的 DB 损失使用 DiceFocal，并将辅助层损失与主损失加权求和。辅助输出、辅助 loss 和辅助 targets
属于训练合同，不能在浮点训练结构或 QAT training graph 中提前删除。

### 5.3 识别：EncoderWithLightSVTR

v6 使用 `EncoderWithLightSVTR` 替代 v5 的拼接式 SVTR neck：

```text
skip = 1x1 Conv-BN-SiLU(x)
z = 1x1 Conv(x)
z = z + 1xk depthwise Conv(z)
z = SVTR blocks -> LayerNorm -> reshape
out = z + skip
```

这种加性跳连减少了 concat 带来的通道开销，同时保留局部卷积和全局 SVTR 建模能力。识别仍使用
CTC + NRTR/GTC MultiHead，CTC logits 是部署输出。

### 5.4 多语言字典

v6 使用扩展字典以覆盖更多语言。模型类别数由当前字典文件和 CTC blank 约定决定，不能跨 v5/v6
直接复用字典或分类层权重。当前仓库字典位于 `pytorchocr/utils/dict/`，模型配置和转换命令应使用
同一版本的字典。

## 6. 训练图、部署图与 QAT

### 6.1 图角色

本仓库明确区分三个阶段：

| 图角色 | 主要内容 | 用途 |
| --- | --- | --- |
| pretrained training graph | 完整训练 head、辅助输出、targets 和 loss 路径 | 浮点训练/微调、训练结构对齐 |
| deployment float graph | 重参数化后的单分支结构和主输出 | 浮点推理、QAT 起点 |
| PT2E QAT graph | 当前 deployment 或指定训练图上的 fake quant/observer | QAT、convert 和 QuantONNX |

检测 deployment graph 只输出 shrink，识别 deployment graph 只输出 CTC logits。训练辅助分支不能通过
简单截断输出的方式模拟。

### 6.2 QAT 量化域

本仓库 PT2E QAT 使用 `export_for_training` 和 `prepare_qat_pt2e` 构建量化图，再通过 `convert_pt2e`
和 QuantONNX 导出部署图。常见量化约定为：

- 权重使用 per-channel 量化；
- 激活使用 per-tensor 量化；
- Attention、下采样链和数据移动算子的量化域由当前模型拓扑及 QAT JSON 共同决定；
- Concat、Split、Reshape 等边界需要按实际 producer/consumer 关系判断是否共享 qparams；
- qparams 必须从当前 `export_for_training` 图发现，不能复制其他输入 shape 或旧图的节点编号。

量化域共享是拓扑合同，不是通过放宽 scale 容差合并不同 qparams。导出后只允许删除 qparams 完全
一致的冗余 Q/DQ，不能手工补写 QDQ 以掩盖 QAT 图问题。

### 6.3 非重参数化 QAT 与折叠

多分支 QAT 训练和单分支部署之间不是天然的参数一一对应关系：

- 分支权重可以按 BN 融合公式合并，但每个分支的激活 qparams 作用于不同节点；
- 重参数化后内部节点消失，分支级 activation qparams 无法逐一继承；
- 输入边界和输出边界可以作为新图初始化的候选，但必须在新图重新 prepare 后重新观测；
- 若需要从非重参数化 QAT checkpoint 继续到单分支模型，应使用仓库提供的量化域折叠流程，并进行
  折叠后的重新 prepare、短校准和 finetune。

该高级流程及其命令见 [PP-OCRv5 QAT 使用指南](../../README-v5.md) 的“非重参化 QAT 和量化域折叠”
章节；它不改变标准 v6 复现流程。

### 6.4 Teacher-Student KD

训练框架支持冻结浮点 Teacher 与可训练 Student 的输出或中间特征蒸馏。KD 不进入 PT2E 图和最终
QuantONNX，部署输出合同保持不变。参数、输出结构和验证要求见 [KD 蒸馏训练指南](../guides/kd_training.md)。

KD 是可选训练能力，不等同于已经验证的默认 QAT 配置；启用前应先完成不带 KD 的浮点/QAT 基线。

## 7. 代码定位

| 内容 | 本仓库位置 |
| --- | --- |
| v5 backbone 与重参数化 | `pytorchocr/modeling/backbones/rec_lcnetv3.py` |
| v6 backbone 与重参数化 | `pytorchocr/modeling/backbones/rec_lcnetv4.py` |
| 检测 FPN 与 RepLKFPN | `pytorchocr/modeling/necks/db_fpn.py` |
| 检测 DBHead | `pytorchocr/modeling/heads/det_db_head.py` |
| 识别 neck 与 SVTR | `pytorchocr/modeling/necks/rnn.py`、`pytorchocr/modeling/backbones/rec_svtrnet.py` |
| CTC/NRTR MultiHead | `pytorchocr/modeling/heads/rec_multi_head.py` |
| QAT 与 QuantONNX | `pytorchocr/quantization/`、`tools/export_ocr_onnx.py` |
| 浮点/QAT 使用入口 | `README.md`、`README-v5.md`、`axera/README.md` |

架构说明不替代具体模型配置。实际输入 shape、预处理、loss 权重、QAT JSON 和导出命令以对应配置和
公开使用文档为准。

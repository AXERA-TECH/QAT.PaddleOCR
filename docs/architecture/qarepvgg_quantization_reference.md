# QARepVGG 参考:量化感知的重参化结构优化

> 参考来源:`references/QARepVGG/`(AAAI 2024, "Make RepVGG Greater Again: A
> Quantization-aware Approach")。本文整理 QARepVGG 的多分支重参化训练结构、
> 融合方式与 BN 插入策略,作为后续网络(RepVGG 类)量化优化的参考。

## 1. 核心问题

RepVGG 类多分支结构(3x3 + 1x1 + identity 分支)在推理时通过结构重参化
融合成单分支 conv。直接对融合后模型做 QAT 存在两个问题:

1. **融合后无 BN 归一化锚点**:重参化把分支 BN 吸收进 conv 权重,训练时
   失去 BN 的归一化作用,量化训练不稳定;
2. **量化感知训练与部署结构不一致**:若在多分支结构上训练、部署时融合,
   QDQ 位置与训练模拟不一致,精度损失。

## 2. 训练结构(QARepVGGBlockV2,默认实现)

```
rbr_dense      = Conv3x3 + BN        # 唯一带 BN 的分支
rbr_1x1        = 裸 Conv1x1(无 BN)
rbr_identity   = 裸 identity(无 BN,仅 stride==1 且通道数相同时)
self.bn        = 分支求和后的统一 BN

forward: bn(se(dense(x) + 1x1(x) + id(x)))
```

关键设计:

- **只有主分支(3x3)带 BN**,1x1/identity 为裸分支——减少融合时的 BN 数量,
  量化友好;
- **求和后挂统一 BN**(`self.bn`)——训练中 BN 的 running stats 统计的正是
  "多分支之和"的分布,与融合后单 conv 的输出分布一致;
- 训练时 BN 全程存在并自然统计,不存在"融合后插 BN 再统计"的额外步骤。

## 3. 融合方式(get_equivalent_kernel_bias)

```
kernel3x3, bias3x3 = _fuse_bn_tensor(rbr_dense)   # 3x3 分支的 BN 吸收进 conv
kernel = kernel3x3 + pad_1x1_to_3x3(rbr_1x1.weight)  # 1x1 裸权重直接加
if rbr_identity: kernel += id_tensor                 # identity 分支加恒等核
return _fuse_extra_bn_tensor(kernel, bias, self.bn)   # 求和后 BN 再吸收
```

- 3x3 分支 BN 与求和后 BN 全部吸收进单 conv,**部署为裸 conv(无 BN)**;
- 融合是数学精确的(逐层等价)。

## 4. insert_bn.py:对已融合部署模型重新插 BN(QAT 用)

适用场景:只有部署权重(已融合裸 conv),需要重新做 QAT。

流程:

1. `switch_repvggblock_to_bnstat`:把融合 conv 拆成
   `conv(无 bias) + BNStatistics + biasadd`,bias 拆出单独保存;
2. `BNStatistics` 模块**只统计不归一化**(记录 mean/var,前向返回原值);
3. 用训练集跑 **500 batches**,记录每个 conv 输出的 mean/var;
4. `switch_bnstat_to_convbn`:转成 `conv(无 bias) + BN`,初始化:

   ```
   running_mean = 实测 mean
   running_var  = 实测 var
   gamma        = std = sqrt(running_var + eps)
   beta         = bias + running_mean
   ```

数学:`BN(conv(x)) = gamma*(conv(x)-mean)/std + beta = conv(x) + bias`,
即 **BN(conv(x)) ≡ conv(x) + bias,前向精确恒等**——插入 BN 后模型输出
不变,且 running stats 与真实分布匹配,训练(batch stats)/验证(running
stats)一致。

## 5. 与方案对比(本项目 rec_lcnetv3 场景)

| 项 | QARepVGG insert_bn | 本项目方案 2(实测统计初始化) |
| --- | --- | --- |
| conv bias | 拆出到 BN 的 beta | 保留在 conv |
| BN 初始化 | γ=std, β=bias+mean | γ=√(var+eps), β=mean |
| 统计方式 | BNStatistics 实测 500 batches | EMA 实测 200 步 |
| 前向 | 恒等 | 恒等 |
| 训练中 BN | momentum 默认 | `bn_training_momentum` 可调(0.9) |

已知问题(本项目):训练中 γ 可学习漂移,极小 var 通道的 running_var
更新滞后,`γ/√(var+ε)` 放大导致 eval(converted)精度下降。QARepVGG
训练结构("求和后 BN 全程存在")是候选修复方向之一——BN 从训练开始
就存在并自然统计,γ 与 var 同步演化,不依赖"融合后插 BN 再统计"。

## 6. 可借鉴的量化友好设计

1. **分支结构设计**:主分支带 BN、其余分支裸——减少融合时 BN 数量;
2. **求和后统一 BN**:running stats 与融合输出分布一致,训练/验证/部署
   的 BN 输入分布不变;
3. **插 BN 恒等初始化**:γ=std、β=bias+mean(或 β=mean 保留 conv bias),
   保证插入后前向不变,无需预热;
4. **训练结构 = 部署结构语义**:量化感知训练在多分支上完成,部署融合
   是数学等价变换,避免"训练多分支 QDQ / 部署单分支 QDQ"的语义错位;
5. **get_custom_L2(可选)**:对等价核施加约束,提升融合后量化精度。

## 7. 参考文件

- `references/QARepVGG/repvgg.py`:`QARepVGGBlockV2` 结构、融合、部署转换
- `references/QARepVGG/insert_bn.py`:部署模型插 BN + 统计初始化
- `references/QARepVGG/README.md`:论文信息(AAAI 2024)

## 8. YOLOv6 的重参化 QAT 优化(第二参考)

> 参考来源:`references/YOLOv6/`(美团 YOLOv6 v0.2.0)。核心思路与 QARepVGG
> 互补:QARepVGG 在"多分支 + 求和后 BN"上训练,部署融合;YOLOv6 则
> **用 RepOptimizer 直接训练单分支网络,彻底消除训练/推理结构不一致**。

### 8.1 核心问题(README 原文要点)

> "due to the inconsistency of reparameterization blocks during training and
> inference, QAT cannot be directly integrated into YOLOv6. As a remedy, we
> first train a single-branch network with RepOptimizer."

多分支训练结构 → 部署融合结构,训练/推理不一致,导致 QAT fake-quant 无法
对齐。**YOLOv6 的解法不是"在多分支上做 QAT",而是让训练结构本身就是
单分支**(RealVGGBlock),配合 RepOptimizer 保持多分支的优化语义。

### 8.2 RepOptimizer:单分支训练的优化器

文件:`yolov6/utils/RepOptimizer.py`(配合 `LinearAddBlock` 预训练提取 scale):

1. **提取 scale**(`extract_scales`):从预训练的多分支模型提取各分支
   BN 的 γ 作为 scale(identity/1x1/conv 三组);
2. **重初始化**(`reinitialize`):把 scale 乘回单分支 3x3 卷积权重,
   并把 1x1/identity 等价核 pad 相加——得到与多分支等价的初始权重;
3. **梯度掩码**(`generate_gradient_masks`):3x3 卷积中心权重的梯度按
   `scale²` 加权(identity/1x1 分支贡献叠加),其余位置保持原梯度,
   模拟多分支训练的梯度语义;
4. 单分支训练收敛后**零融合成本**部署——训练结构 = 推理结构。

### 8.3 PTQ + QAT 流程(`tools/qat/`)

```
单分支 RepOpt 训练 → PTQ 校准 → QAT 微调(10 epoch + 通道蒸馏)→ 导出
```

1. **PTQ 校准**(`qat_utils.py:ptq_calibrate`):
   - `collect_stats`:禁用 fake-quant、启用校准器,跑 N batch 收集分布;
   - `compute_amax`:MaxCalib/HistogramCalib 加载 amax(支持 percentile);
2. **QAT 初始化**(`qat_init_model_manu`):遍历模型,把
   `Conv2d → QuantConv2d`、`ConvTranspose2d → QuantConvTranspose2d`、
   `MaxPool2d → QuantMaxPool2d`(NVIDIA pytorch_quantization),
   **BN 模块保留不量化**;
3. **敏感层跳过**(`skip_sensitive_layers`):head 的 stem/conv/pred 等
   敏感层禁用量化(`module_quant_disable`);
4. **QAT 训练**:加载校准后的 calib_pt,微调 10 epoch,配**通道蒸馏**
   (`loss_distill_ns`,加速收敛);
5. **导出**(`qat_export.py`):
   - 对 RepVGGBlock `switch_to_deploy`(融合);
   - `zero_scale_fix`:权重 amax=0 的通道置 1(防除零/scale 丢失);
   - `concat_quant_amax_fuse`:Concat 各输入 amax 融合为共享量化域;
   - `TensorQuantizer.use_fb_fake_quant=True` → ONNX QDQ。

### 8.4 QARepVGGBlock 在 YOLOv6 中的实现(与本项目最相关)

YOLOv6 内置 `QARepVGGBlock`/`QARepVGGBlockV2`(`get_block('qarepvgg')`),
与 references/QARepVGG 同源,但有一处**关键差异**:

| | YOLOv6 QARepVGGBlock | references/QARepVGG |
| --- | --- | --- |
| `switch_to_deploy` 后 BN | **保留**(`# keep post bn for QAT`) | **删除**(`__delattr__('bn')`) |
| 部署 forward | `bn(rbr_reparam(x))`(conv + BN) | `rbr_reparam(x)`(裸 conv) |
| 结构 | 融合 conv + **保留 BN** | 融合裸 conv |

YOLOv6 的 QARepVGGBlock 部署后是 `conv + bn` 结构——与本项目 keep-BN
方案(conv+bn)一致,且其 BN 从训练起就存在(求和后 BN),自然统计,
无"融合后插 BN 再统计"步骤。

### 8.5 与本项目方案 2 的对比

| 项 | YOLOv6(QARepVGGBlock) | 本项目方案 2(keep-BN) |
| --- | --- | --- |
| 训练结构 | 多分支 + 求和后 BN | 融合单分支 + keep-BN |
| BN 存在性 | 训练全程存在,自然统计 | 融合后插 identity 占位 + 实测统计 |
| γ/var 一致性 | BN 全程训练更新,天然一致 | γ 漂移与 var 失配(当前问题) |
| 部署结构 | 融合 conv + 保留 BN | 融合 conv + 保留 BN(一致) |
| 量化工具 | NVIDIA pytorch_quantization | PT2E prepare_qat_pt2e |

### 8.6 可借鉴要点

1. **BN 训练全程存在**(求和后 BN / 单分支 BN)→ 避免"后插 BN 统计"的
   γ/var 失配,这是本项目当前问题的根本修复方向;
2. **单分支训练 + RepOptimizer**(结构一致性)→ 从根上消除 QAT 训练/推理
   结构不一致;
3. **zero_scale_fix**:权重 amax=0 通道置 1,防导出 scale 丢失;
4. **concat_quant_amax_fuse**:Concat 输入共享量化域(与本项目
   validation 的 concat_domains 强制检查一致);
5. **敏感层跳过 + 通道蒸馏**:QAT 精度工程技巧。

## 9. 参考文件汇总

- `references/QARepVGG/repvgg.py`:`QARepVGGBlockV2` 结构、融合、部署转换
- `references/QARepVGG/insert_bn.py`:部署模型插 BN + 统计初始化
- `references/QARepVGG/README.md`:论文信息(AAAI 2024)
- `references/YOLOv6/yolov6/layers/common.py`:`QARepVGGBlock/V2`(保留 BN 版)
- `references/YOLOv6/yolov6/utils/RepOptimizer.py`:单分支重参化优化器
- `references/YOLOv6/tools/qat/qat_utils.py`、`qat_export.py`:PTQ/QAT 流程
- `references/YOLOv6/tools/qat/README.md`、`docs/Tutorial of Quantization.md`

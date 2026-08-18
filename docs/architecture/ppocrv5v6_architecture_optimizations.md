# PP-OCRv4 / PP-OCRv5 / PP-OCRv6 架构与优化详解

本文档整合 PP-OCRv4、v5、v6 三代模型的架构与优化说明,以官方
`PP-OCRv4_introduction.md`(`references/PaddleOCR/docs/version2.x/ppocr/blog/PP-OCRv4_introduction.md`,
即 https://www.paddleocr.ai/latest/version2.x/ppocr/blog/PP-OCRv4_introduction.html )的粒度为基准,
从官方 Paddle 实现(`references/PaddleOCR/ppocr/`)还原公式、关键配置与代码出处。
v4 部分内容直接整理自上述官方 blog;v5/v6 部分补充官方文档未展开的架构细节。
所有代码引用指向 `references/PaddleOCR`(官方只读 checkout);行号以撰写时为准。
BN 参数名按 Paddle 约定为 `weight/bias/_mean/_variance/_epsilon`。

## 0. 版本演进总览

| 维度 | PP-OCRv4 | PP-OCRv5 | PP-OCRv6 |
| --- | --- | --- | --- |
| 骨干(det/rec) | PP-LCNetV3 | PP-LCNetV3(det scale 0.75 / rec scale 0.95) | **PPLCNetV4**(统一,tiny/small/medium) |
| 检测 Head | DBHead / **PFHeadLocal** | DBHead | DBHead + **P2/P3/P4 辅助深度监督** |
| 检测 Neck | RSEFPN | RSEFPN | **RepLKFPN**(大核重参) |
| 检测 Loss | Dice + OHEM | Dice + OHEM(balance) | **Dice + Focal** + aux 权重 |
| 检测训练策略 | **DSR**(shrink 0.4→0.6)、**CML 蒸馏** | 同 v4 | 辅助深度监督 |
| 识别 Neck | SVTR(use_guide,即 Lite-Neck) | SVTR(use_guide) | **EncoderWithLightSVTR**(加性跳连) |
| 识别解码 | MultiHead(CTC + **NRTR**) | MultiHead(CTC + NRTR) | MultiHead(CTC + NRTR) |
| 识别训练策略 | **DF 数据挖掘**、**GTC-NRTR**、**Multi-Scale**、**DKD 蒸馏** | 同 v4(无 DKD) | 同 v5(无 DKD) |
| 结构重参数化 | LCNetV3 RepLayer | LCNetV3 RepLayer | LCNetV4 RepDWConv / Stem / FPN |

## 1. PP-OCRv4

内容整理自官方 blog(PP-OCRv4_introduction.md),代码引用为仓库当前实现。
v4 在 v3 基础上共 10 项改进:检测 4 项(LCNetV3、PFHead、DSR、CML),识别 6 项
(SVTR_LCNetV3、Lite-Neck、GTC-NRTR、Multi-Scale、DF、DKD)。
效果:v3 基础上中文场景端到端 Hmean 提升 4.5%(57.99%→62.24%),英文提升 6%,多语言平均提升 8%+。

### 1.1 检测优化

#### (1) PP-LCNetV3:精度更高的骨干网络

PP-LCNetV3 覆盖更大精度范围,提出**可学习仿射变换模块(LAB)**、改进重参数化策略与激活,
调整网络深度/宽度(详见 2.1(2)(3),v5 沿用同一骨干)。v4 检测用 `PPLCNetV3(scale=0.75, det=True)`
(`configs/det/PP-OCRv4/PP-OCRv4_mobile_det.yml`,骨干代码 rec_lcnetv3.py:403)。

#### (2) PFHead:并行 head 分支融合结构

`PFHeadLocal`(det_db_head.py:261,forward det_db_head.py:272-283):
第一个转置卷积取中间特征 `f`,上采样后经 `LocalModule`(3x3 卷积,det_db_head.py:248)得到
局部精修分支,与主分支概率图融合:

```text
base_maps = sigmoid(binarize 主分支)
cbn_maps  = sigmoid(LocalModule(up_conv(f), base_maps))
maps      = 0.5 · (base_maps + cbn_maps)      # 部署输出
```

v4 学生检测模型使用 PFHead,hmean 76.22%→76.97%(blog 消融 01)。

#### (3) DSR:收缩比例动态调整

`MakeShrinkMap` 的 shrink_ratio 随 epoch 从 0.4 线性增加到 0.6
(`ppocr/data/imaug/make_shrink_map.py:72-74`):

```text
shrink_ratio = 0.4 + 0.2 · epoch / total_epoch
```

v4 学生检测模型 hmean 76.97%→78.24%(blog 消融 02)。
配置:`MakeShrinkMap(shrink_ratio=0.4, min_text_size=8, total_epoch=500)`。

#### (4) CML:融合 KD 的互学习策略

在 v3 的 CML(Collaborative Mutual Learning)基础上,Student/Teacher 的 response maps
之间额外加 **KL div loss**,使两者分布接近。检测 Hmean 79.08%→79.56%。
配置:`configs/det/PP-OCRv4/PP-OCRv4_det_cml.yml`(蒸馏训练图)。

### 1.2 识别优化

#### (1) DF:数据挖掘方案

DF(Data Filter)两步过滤:
1. 低精度模型预测千万级数据,去除置信度 >0.95 的冗余样本;
2. PP-OCRv3 高精度模型预测剩余数据,去除置信度 <0.15 的低质样本。

千万级数据精简至百万级,训练时间 2 周→5 天,精度 71.5%→72.7%(+1.2%)。
(数据侧策略,不体现在模型代码中。)

#### (2) PP-LCNetV3:精度更优的骨干网络

同 1.1(1);v4 识别用 `PPLCNetV3(scale=0.95)`
(`configs/rec/PP-OCRv4/PP-OCRv4_mobile_rec.yml`)。

#### (3) Lite-Neck:精简参数的 Neck 结构

沿用 v3 结构精简参数(12M→8.5M),并将 `EncoderWithSVTR` 输出维度提升到 120
(rnn.py:140):`dims=120, hidden_dims=120`,模型回到 9.6M。v4 起 SVTR neck 的
`use_guide=True` 即为 GTC 语义(见下)。

#### (4) GTC-NRTR:Attention 指导 CTC 训练策略

GTC(Guided Training of CTC)延续 v3;v4 把指导分支从 SAR(循环神经网络)换成
**NRTR(Transformer)**:泛化更强、训练更稳定,缓解简单场景 CTC 快速过拟合。

- `EncoderWithSVTR.forward` 中 `use_guide=True` 时输入 `clone()+stop_gradient`
  (rnn.py:218-220),指导分支梯度不回流 backbone;
- MultiHead(rec_multi_head.py:67)并行 CTCHead(rec_ctc_head.py:35)+ NRTR
  Transformer(rec_nrtr_head.py:25);训练返回 `{"ctc","ctc_neck","gtc"}`,eval 只返回
  `ctc_out`(rec_multi_head.py:142-153)。

Lite-Neck + GTC-NRTR 精度 72.7%→73.21%(blog 消融 03)。

#### (5) Multi-Scale:多尺度训练策略

`MultiScaleSampler`(`ppocr/data/multi_scale_sampler.py`)每个 iter 从 `(32, 48, 64)`
三种高度随机选择;批内同尺度、批间多尺度。识别测试集本身不提升,端到端串联指标 +0.5%。
配置:`scales: [[320, 32], [320, 48], [320, 64]]`。

#### (6) DKD:蒸馏策略

- **NRTR head**:DKD loss 拉近 Student/Teacher 的 NRTR logits,与 ground-truth 交叉熵
  (去掉 label smoothing)加权监督 backbone;
- **CTCHead**:把 CTC logits 沿文本长度维取均值,转成多字符分类问题,规避 blank 位导致的
  分布偏移,再监督 CTC head 训练。

融合后指标 74.72%→75.45%(blog 消融 08)。

### 1.3 端到端评估(官方 blog)

| Model | Hmean | Model Size (M) | Time Cost (CPU, ms) |
| --- | --- | --- | --- |
| PP-OCRv3 | 57.99% | 15.6 | 78 |
| PP-OCRv4 | 62.24% | 15.8 | 76 |

## 2. PP-OCRv5

模型配置(官方完整版,含 Train/Eval 数据管线):

- `references/PaddleOCR/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml`:
  `PPLCNetV3(scale=0.75, det=True)` + `RSEFPN(out_channels=96, shortcut=True)` + `DBHead(k=50, fix_nan=True)`;
  Loss `DBLoss(balance_loss=True, main_loss_type=DiceLoss, alpha=5, beta=10, ohem_ratio=3)`;
  PostProcess `DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5)`
- `references/PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`:
  `PPLCNetV3(scale=0.95)` + `MultiHead(CTCHead svtr: dims=120, depth=2, hidden_dims=120,
  kernel_size=[1,3], use_guide=True; NRTRHead: nrtr_dim=384, max_text_length=25)`

### 2.1 检测优化

#### (1) PP-LCNetV3 检测版:任务自适应下采样

同一个 `PPLCNetV3` 通过 `det` 标志切换两组网络配置
(`references/PaddleOCR/ppocr/modeling/backbones/rec_lcnetv3.py`):

- `NET_CONFIG_det`(rec_lcnetv3.py:38):`conv1(stride=2)` 后 blocks2(stride=1),
  blocks3/4/5/6 各含一个 stride=2 的 block,输出 **stride 4/8/16/32 四级特征**
  (blocks3/4/5/6 末层,见 forward,rec_lcnetv3.py:542-556);每级经 1x1 `layer_list`
  投影到 `[16, 24, 56, 480]×scale` 通道(rec_lcnetv3.py:518-540),供 RSEFPN 使用;
- `NET_CONFIG_rec`(rec_lcnetv3.py:59):非对称 stride——blocks4/6 用 `(2,1)`(只缩高度),
  blocks5 用 `(1,2)`(缩减宽度),最终特征由 CTC/NRTR 解码为 1-D 序列。

block 描述 `[k, in_c, out_c, s, use_se]`;检测版 blocks5/6 用 kernel 5 扩感受野,blocks6 启用 SE。
`PPLCNetV3.__init__`(rec_lcnetv3.py:403)按 `self.det` 选择配置(rec_lcnetv3.py:418):
`self.net_config = NET_CONFIG_det if self.det else NET_CONFIG_rec`。

#### (2) 多分支重参数化卷积:LearnableRepLayer

每个 block 的空间卷积是 MobileOne 风格多分支结构(rec_lcnetv3.py:156):

```text
out = identity(x) + Σ_{k∈branches} conv_kxk(x) + conv_1x1(x)
out = LAB(out);  out = Act(out)        # stride==2 时跳过 Act
```

- `identity`:BN(仅当 in==out 且 stride==1);
- `conv_kxk`:4 个平行 Conv-BN 分支(kernel 3/5);
- `conv_1x1`:1x1 Conv-BN 分支;
- 部署时 `rep()`(rec_lcnetv3.py:237)把三分支融合为单个大核卷积:

```text
K = Σ(BN_fuse(conv_kxk)) + pad1x1_to_kxk(BN_fuse(conv_1x1)) + identity_kernel
b = Σ b_branch
```

identity 分支由单位核 `id_tensor`(rec_lcnetv3.py:308)按 BN 参数缩放得到,
1x1 分支用 `F.pad` 扩展到 kxk(rec_lcnetv3.py:261);融合后删除训练分支
(rec_lcnetv3.py:247-253),避免部署参数重复计数。

#### (3) 可学习仿射块 LAB

`LearnableAffineBlock`(rec_lcnetv3.py:90)对分支融合输出做逐通道可学习线性变换:

```text
y = scale · x + bias          # scale/bias 均为可学习标量
```

学习率单独控制(`lab_lr=0.1`),避免低学习率微调时破坏卷积结构。

#### (4) RSEFPN

v5 检测 Neck 延续 v4 的 RSEFPN(db_fpn.py:246):自顶向下 `RSELayer`(1x1 Conv + SE + shortcut,
db_fpn.py:221)+ 3x3 `inp_conv` 精修,nearest 上采样拼接得到 fuse(forward,db_fpn.py:270):

```text
in5..in2 = ins_conv(c5..c2);  out4 = in4 + up(in5);  out3 = in3 + up(out4);  out2 = in2 + up(out3)
p5 = inp_conv(in5); p4 = inp_conv(out4); p3 = inp_conv(out3); p2 = inp_conv(out2)
fuse = concat(p5↑8, p4↑4, p3↑2, p2)
```

fuse 通道 = 96×4 = 384,交给 DBHead。

#### (5) DBHead 训练三分支 / 部署单分支

`DBHead.forward`(det_db_head.py:201):训练返回 `{shrink, threshold, binary}` 三 map;
非训练模式只返回 `{"maps": shrink_maps}`(det_db_head.py:211-212)。
`k=50` 的近似二值化 `step_function`(det_db_head.py:198):

```text
binary = 1 / (1 + exp(-k · (shrink - threshold)))
```

### 2.2 识别优化

#### (1) 非对称 stride 骨干(同 2.1(1),rec 版)

`NET_CONFIG_rec` 用非对称步幅:blocks4/6 用 `(2,1)`(高度为主下采样方向),blocks5 用 `(1,2)`;
最终特征经 SequenceEncoder 的 `Im2Seq`(squeeze H,转置到 `N,T,C`,rnn.py:33)得到序列。

#### (2) EncoderWithSVTR + use_guide(guide detach)

v5 识别 Neck 是 `EncoderWithSVTR`(rnn.py:140,forward rnn.py:216):

```text
h = z(=x 或 stop_gradient(x));    # use_guide=True 时 detach
z = conv1(z) → conv2(z)            # 降维到 hidden_dims
z = flatten → [B,T,C]
z = SVTR Block × depth(全局注意力+MLP)
z = LayerNorm → reshape → conv3(z)
out = conv1x1(conv4(concat(h, z))) # 拼接跳跃连接
```

- `use_guide=True`(v5 rec yml):输入先 `x.clone(); z.stop_gradient = True`
  (rnn.py:218-220),SVTR/guide 分支的梯度不回流 backbone——这是 v4 引入的
  GTC-NRTR"Attention 指导 CTC"策略在 v5 的延续:NRTR 头提供文本上下文先验,
  CTC 主路径通过 detach 保持稳定的特征学习;
- **`conv1`/`conv4` 的 kernel_size 从 YAML 读取(v5 rec 为 `[1,3]`,padding 同步)**:
  上游 route2 曾因 `EncoderWithSVTR.conv4` 未传 YAML kernel 而退回 `ConvBNLayer`
  默认 3x3,导致 Paddle 权重 `[60,960,1,3]` 与 PyTorch `[60,960,3,3]` shape 失配;
  已与 Paddle 对齐并有回归测试锁定(修复溯源:
  `docs/axera_qat/archive/smoke/ppocrv5_mobile_rec_qat_smoke.md` §3);
- `hidden_dims=120`,SVTR Block(rec_svtrnet.py:215)使用 fused QKV
  `nn.Linear(dim, dim*3)`(rec_svtrnet.py:170),注意力公式(rec_svtrnet.py:196-211):

```text
attn = softmax(q·kᵀ / √d);  out = attn·v → proj
```

`prenorm=False`(post-norm),激活 swish。

#### (3) MultiHead:CTC + NRTR 双解码

`MultiHead`(rec_multi_head.py:67,forward rec_multi_head.py:134)并行构建
`CTCHead`(rec_ctc_head.py:35)与 NRTR `Transformer`(rec_nrtr_head.py:25):

- CTC 路径:`SequenceEncoder(svtr)` → `CTCHead`;训练 forward 返回
  `{"ctc": ctc_out, "ctc_neck": ctc_encoder, "gtc": gtc_out}`,非训练模式只返回
  `ctc_out`(rec_multi_head.py:142-153);
- NRTR 路径:`FCTranspose`(rec_multi_head.py:39,转置投影 `W` 使 Paddle Conv 权重映射到 Linear)+
  Transformer 解码器,训练时提供辅助监督,推理移除;
- 训练 Loss = `MultiLoss(CTCLoss + NRTRLoss)`。

#### (4) 多尺度训练

rec 训练使用 `MultiScaleDataSet` + `MultiScaleSampler`
(`references/PaddleOCR/ppocr/data/multi_scale_sampler.py`),scales 默认
`[[320,32],[320,48],[320,64]]`:**每个 batch 绑定一个尺度、批内同尺寸,批间多尺度**;
`fix_bs=False` 时 batch size 与 `h·w` 成反比。图像由数据集内部 `resize_norm_img`
(`references/PaddleOCR/ppocr/data/simple_dataset.py` MultiScaleDataSet)按目标高等比缩放、
右侧零填充、归一化 `[-1,1]`。

### 2.3 数据与训练要点

- 检测训练:`CopyPaste` + `IaaAugment`(FlipLR/Affine ±10°/Resize 0.5~3)+ `EastRandomCropData`
  (640x640,不足时等比缩放 + 左上对齐 + 黑边 0 填充,
  `references/PaddleOCR/ppocr/data/imaug/random_crop_data.py:466`)+
  `MakeBorderMap/MakeShrinkMap`(shrink 0.4);
- 检测评估:`DetResizeForTest`(短边 736、宽高 32 对齐,
  `references/PaddleOCR/ppocr/data/imaug/operators.py:208`);
- 识别训练:`RecConAug`(prob 0.5 字符拼接)+ `RecAug`,评估 `RecResizeImg [3,48,320]`
  (`references/PaddleOCR/ppocr/data/imaug/rec_img_aug.py:285`,内部 `resize_norm_img` 在 :631);
- 归一化:检测用 ImageNet mean/std,识别用 `[-1,1]`。

## 3. PP-OCRv6

模型配置(官方完整版):

- `references/PaddleOCR/configs/det/PP-OCRv6/PP-OCRv6_small_det.yml`:
  `PPLCNetV4(det=True, model_size=small)` + `RepLKFPN(out_channels=96, dilated_kernel_size=7,
  shortcut=True)` + `DBHead(k=50, fix_nan=True, aux_in_channels=96)`;
  Loss `DBLoss(main_loss_type=DiceFocalLoss, alpha=5, beta=10, focal_alpha=0.25, focal_gamma=2.5,
  aux_weight_p4=0.2, aux_weight_p3=0.3, aux_weight_p2=0.4)`
- `references/PaddleOCR/configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml`:
  `PPLCNetV4(model_size=small)` + `MultiHead(CTCHead lightsvtr: dims=120, depth=2, mlp_ratio=2.0,
  local_kernel=7; NRTRHead: nrtr_dim=384, max_text_length=25)`

### 3.1 统一骨干 PPLCNetV4

`PPLCNetV4`(rec_lcnetv4.py:521)为 det/rec 共用骨干,`det` 标志切换 `NET_CONFIG_DET`/`NET_CONFIG_REC`
(tiny/small/medium 三档,rec_lcnetv4.py:35-169)。核心是 `LCNetV4Block`(rec_lcnetv4.py:439),
遵循 MetaFormer 范式:

```text
x̂ = SE(DW(x)) + x                        # Token Mixer(3x3 DWConv,可选 SE)
y = W₂ · σ(W₁ · x̂) + x̂                  # Channel Mixer:expand(2x) → GELU → compress
```

- **RepDWConv**(rec_lcnetv4.py:367):`stride=1 且 in==out` 时(`use_rep_dw`,rec_lcnetv4.py:461),
  Token Mixer 的 DW 是三分支重参数化结构(3x3 + 1x1 + identity),部署时 `rep()` 融合;
- **BN 零初始化**:compress 层在存在残差时 `bn_weight_init=0.0`(rec_lcnetv4.py:483),
  `Conv2D_BN` 用 `Constant(0.0)` 初始化(rec_lcnetv4.py:201-202)——训练初期残差恒等、
  稳定,再逐步学习;
- **任务自适应下采样**:
  - det:`StemBlock`(rec_lcnetv4.py:289,多分支 stem:3x3 stride2 + 2x2 SAME 双支 + MaxPool 分支,
    stride 4 总下采样)+ `blocks_s1..s4` 全 stride-2,输出 4 级特征(stride 4/8/16/32);
  - rec:stem 后用 `(2,1)` 非对称 stride(blocks4/5,见 NET_CONFIG_REC),只缩高度,
    最终 `adaptive_avg_pool2d(x, [1, 40])`(rec_lcnetv4.py:638)压成 `[B, C, 1, 40]` 序列;
  - tiny rec 的 stem 是 `simple`(2×Conv2D_BN+GELU)。

### 3.2 检测:RepLKFPN + 辅助深度监督

#### (1) RepLKFPN

`RepLKFPN`(db_fpn.py:307,forward db_fpn.py:385)是 RSEFPN 的大核重参升级
(官方:参数 118K vs 172K,感受野 3x3→7x7):

- `inp_conv` 的 3x3 标准卷积替换为 `DilatedReparamConv`(db_fpn.py:729):
  `DW DilatedReparamBlock(k=7) → 1x1 PW Conv → BN → SE`(`_inp_forward`,db_fpn.py:376);
- `DilatedReparamBlock`(db_fpn.py:554,参考 UniRepLKNet)训练时多分支:
  kernel=7 时分支为 `5x5(dil=1) + 3x3(dil=2) + 3x3(dil=3)`(等效感受野 7),部署融合为单个大核 DW;
- `ins_conv` 保持 1x1(无 DW 分解收益);
- 训练 forward 额外返回 `{"fuse", "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}`
  (db_fpn.py:418-419),供 DBHead 辅助监督使用;部署只返回 fuse。

#### (2) DBHead 辅助深度监督

`DBHead(aux_in_channels=96)`(det_db_head.py:168)对 neck 的 P2/P3/P4 辅助特征
各建一对 `aux_binarize/aux_thresh` Head(det_db_head.py:184-196),训练 forward
(det_db_head.py:219-230)输出 `aux_maps_p2/p3/p4`,经 bilinear 上采样
(scale 1/2/4,`_aux_upsample_scale` det_db_head.py:185-189)到 fuse 分辨率:

```text
loss = loss_main + 0.4·loss_aux_p2 + 0.3·loss_aux_p3 + 0.2·loss_aux_p4
```

给低层提供更强梯度信号,缓解小目标/密集文本监督不足。

#### (3) Dice + Focal

v6 用 `DiceFocalLoss`(focal_alpha=0.25, gamma=2.5)替代 v5 的纯 Dice+OHEM,正负样本
不平衡与难样本兼顾。

### 3.3 识别:EncoderWithLightSVTR

`EncoderWithLightSVTR`(rnn.py:242,forward rnn.py:331)替代 v5 的拼接式 SVTR neck,
官方要点:局部上下文 + 轻量加性跳连、参数更少:

```text
skip = skip_conv(x)              # 1x1 Conv-BN-swish 跳连(加性,替代 concat)
z = conv_reduce(x)               # 1x1 降维到 dims
z = z + local_conv(z)            # 1x7 DWConv(dims, groups=dims)+ BN + SiLU
z = flatten → SVTR Block × depth → LayerNorm → reshape
out = z + skip
```

- `local_conv`:1x7 深度卷积(局部上下文,local_kernel=7,groups=dims);
- SVTR Block 配置 `dims=120, depth=2, mlp_ratio=2.0`(small),medium 为 `dims=192, mlp_ratio=4.0`;
- `use_guide` 语义保留(rnn.py:332-334):输入先 `clone() + stop_gradient=True`,
  主路径与 skip 分支都基于 detach 后的特征,neck 梯度不回流 backbone(与 v5 相同);
- 解码仍为 MultiHead CTC(部署保留)+ NRTR(训练辅助)。

### 3.4 多语言

v6 字典扩展至 50 语言(`references/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt`,
class 数 18710,即 18708 字典字符 + space + CTC blank),small/medium 单模型覆盖中/繁/英/日 +
46 拉丁语系语言;tiny 不含日文。(v5 字典 class 数为 18385,不要与 v6 混淆。)

## 4. 关键差异表

| 组件 | PP-OCRv4 | PP-OCRv5 | PP-OCRv6 |
| --- | --- | --- | --- |
| 骨干范式 | MobileNet-style(DW→SE→PW) | 同 v4 | MetaFormer(TokenMixer+ChannelMixer) |
| 空间卷积 | RepLayer: identity+4×kxk+1x1 | 同 v4 | RepDWConv: 3x3+1x1+identity(仅 stride1/in==out) |
| BN 初始化 | 标准 | 标准 | compress 层零初始化 |
| 检测 Head | DBHead / PFHeadLocal | DBHead | DBHead + P2/P3/P4 辅助 |
| 检测 Neck | RSEFPN(3x3) | RSEFPN(3x3) | RepLKFPN(7x7 dilated reparam + PW + SE) |
| 检测训练策略 | DSR(0.4→0.6)、CML | 同 v4(无 CML 蒸馏配置) | DiceFocal + 辅助深度监督 |
| 识别 Neck | SVTR 拼接跳连(use_guide) | 同 v4 | LightSVTR(1x7 局部 + 加性跳连) |
| 识别训练策略 | DF、GTC-NRTR、Multi-Scale、DKD | DF、GTC-NRTR、Multi-Scale | 同 v5 |

## 5. BN 与 QAT:官方权重、重参化与量化路径

本节回答三个工程问题:Paddle 预训练权重里 BN 是否保留、官方重参化后的卷积是否还有 BN、
官方 QAT 如何插入 fake quant 以及 QAT 模型如何部署。结论先行:

1. **预训练权重完整保留 BN**(训练态权重,BN 未融合进 Conv);
2. **官方重参化(`rep()`)后的卷积是纯 `Conv2D`,不含 BN**——BN 在融合时被吸收进
   权重/bias,部署图 BN=0;
3. **官方 QAT 在训练图上插入 fake quant 算子,且 QAT 导出时跳过重参化**——多分支
   训练图原样量化部署,不存在"QAT 后再融合卷积权重"的官方步骤。

### 5.1 预训练权重保留完整 BN(未融合进 Conv)

对仓库内三个官方 `.pdparams` 直接检查键名(2026-08-18):

| 权重 | 总键数 | BN 相关键 | `bn.weight/bias/_mean/_variance` 组数 |
| --- | ---: | ---: | ---: |
| `PP-OCRv5_mobile_rec_pretrained.pdparams` | 969 | 556 | 151 |
| `PP-OCRv5_mobile_det_pretrained.pdparams` | 906 | 562 | 150 |
| `PP-OCRv6_small_rec_pretrained.pdparams` | 423 | 226 | 58 |

- 键名形如 `backbone.conv1.bn.weight/.bias/_mean/_variance`,且**每个分支都有 BN**,
  包括 `LearnableRepLayer` 的 identity 分支(`backbone.blocks2.0.dw_conv.identity._mean`)
  与 `conv_kxk`/`conv_1x1` 分支——多分支重参化训练结构的 BN 全部在;
- BN 数值是真实训练统计量,非退化占位:如 v5 rec `backbone.conv1.bn.weight`
  mean≈0.813、`_variance` mean≈0.150、identity 分支 `_variance` max≈3.22;
- 因此官方发布的是**训练态权重**;BN 融合只发生在推理导出阶段(见 5.2)。

### 5.2 官方重参化:融合后的卷积不含 BN

v5 `LearnableRepLayer.rep()`(`references/PaddleOCR/ppocr/modeling/backbones/rec_lcnetv3.py`,L237)流程:

```text
_get_kernel_bias()  各分支 BN 吸收 + 求和(rec_lcnetv3.py:267)
  -> 新建裸 Conv2D,set kernel/bias(rec_lcnetv3.py:241-250)
  -> del conv_kxk / conv_1x1 / identity(rec_lcnetv3.py:247-253)
  -> is_repped = True
```

BN 融合公式(`_fuse_bn_tensor`,rec_lcnetv3.py:286-317),对 4×kxk、1x1(补零到 kxk)、
identity(单位核按 BN 缩放)统一处理:

```text
t     = γ / √(var + ε)                 # 逐通道
W'    = W · t
b'    = β − μ · γ / √(var + ε)
kernel_reparam = Σ_kxk W' + pad(W'_1x1) + W'_identity
bias_reparam   = Σ_branch b'
```

重参化后的 forward(rec_lcnetv3.py:215-219)只有
`reparam_conv -> LAB -> (Act)`,**没有任何 BN 节点**。v6 同样
(`references/PaddleOCR/ppocr/modeling/backbones/rec_lcnetv4.py`):
`Conv2D_BN.fuse()`(rec_lcnetv4.py:206)返回裸 Conv;`ConvBNAct.rep()`
(rec_lcnetv4.py:266)**`del self.bn`** 且 forward 在 repped 时跳过 BN
(rec_lcnetv4.py:259-260);`StemBlock.rep()`(rec_lcnetv4.py:331)、
`RepDWConv.rep()`(rec_lcnetv4.py:398)均只保留 `reparam_conv`。

官方导出对**非 QAT** 模型强制重参化
(`references/PaddleOCR/ppocr/utils/export_model.py:360-363`):

```text
if arch_config["model_type"] != "sr" and not skip_reparameterization:
    for layer in model.sublayers():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped", False):
            layer.rep()
```

即官方推理模型 = 全融合、BN=0 的部署图(与本仓库 QuantONNX 导出合同 BN=0 一致)。

### 5.3 官方 QAT:在训练图上插入 fake quant

路径约定:本节 `qat.py` 指 `references/PaddleOCR/ppocr/utils/qat.py`,
`export_model.py` 指 `references/PaddleOCR/ppocr/utils/export_model.py`,
`export_qat_onnx.py` 指 `references/PaddleOCR/tools/export_qat_onnx.py`;
Paddle 3.0 内置模块路径以 `site-packages/` 前缀标注(非仓库文件)。

QAT 入口是 `references/PaddleOCR/ppocr/utils/qat.py::apply_qat`(L441),分两层:

**(1) `quanter.quantize(model)`(Paddle 3.0 内置 `paddle.quantization.imperative.qat`
的 `ImperativeQuantAware`,位于 `site-packages/paddle/quantization/imperative/qat.py`,
quantize 见 L236-294)**

- 把 `Conv2D/Linear/Conv2DTranspose` 替换为 `QuantizedConv2D` 等量化层
  (`site-packages/paddle/nn/quant/quant_layers.py::QuantizedConv2D`,
  forward 见 L615),内部含两个 fake quant:
  - `_fake_quant_weight`:权重 **per-channel S8**(`channel_wise_abs_max`,axis=0);
  - `_fake_quant_input`:激活 **per-tensor**(默认 `moving_average_abs_max`);
- `_quantize_outputs.apply`(同 `imperative/qat.py` 的 `ImperativeQuantizeOutputs`,
  L448)给目标层包 `MAOutputScaleLayer`/`FakeQuantMAOutputScaleLayer`,
  统计输出 out_scale。

**(2) PaddleOCR 自研 U8 affine 扩展(`qat.py::_apply_u8_qat`,L271-316)**

- `qat.py::U8AffineFakeQuant`(L20):per-tensor U8 affine fake quant——observer 为
  moving min/max,`scale=(max−min)/255`、`zp=clip(round(−min/scale), 0, 255)`;
  forward 是 STE 伪量化:

  ```text
  y = (round_even(clip(x/scale + zp, 0, 255)) − zp) · scale   # 取整梯度 detach
  ```

- 按算子 pattern 插入(`qat.py::_pattern_terminal`,L253):识别
  `ConvBNLayer.bn`、`ConvBNAct.act`、`Conv2D_BN.bn`、`Head.conv_bn1`、
  `DilatedReparamBlock` 各分支 BN 作为终点,**输出 fake quant 挂在 BN/Act 之后**
  (`qat.py::U8AffineOutputQuantWrapper`,L141),输入 fake quant 挂在量化 Conv 之前
  (`conv._fake_quant_input`);
- `qat.py::_share_u8_activation_domains`(L319):hook 追踪 eval/train 两次 forward,
  若量化 Conv 的输入直接来自另一个 QAT 输出域(如 Concat 下游),把输入 fake quant
  替换为 `nn.Identity()` 共享上游输出域——与本仓库 Concat 共享量化域同一思路;
- `qat.py::freeze_qat_observers`(L432):训练结束后关 observer(等价 observer freeze)。

### 5.4 QAT 导出:跳过重参化,多分支量化图直接部署

`references/PaddleOCR/ppocr/utils/export_model.py:378-384`:

```text
model = dynamic_to_static(model, arch_config, logger, input_shape,
                          skip_reparameterization=quanter is not None)
```

- **只要启用 QAT(quanter 非 None)就跳过 `layer.rep()`**,多分支训练图原样进入量化
  导出:`quanter.save_quantized_model(model, save_path)`(export_model.py:411)把
  每条分支(各自带独立 fake quant 的 QuantizedConv2D)存为量化推理模型,分支独立量化、
  独立 out_scale;
- `tools/export_qat_onnx.py`(QAT checkpoint → ONNX QDQ)同样是**未重参化**流程:
  `strip_fake_quant` 剥掉 fake quant、收集各层 qparams(权重 per-channel threshold、
  激活 scale/zp、输出 scale),导出 float ONNX 后在图上重建 Q/DQ 节点
  (含同 qparam 冗余 QDQ 清理),全程没有卷积权重融合;
- **结论:Paddle 官方 QAT 不融合卷积权重**。多分支量化图整体部署,分支数多、部署效率
  低,但没有"QAT 后折叠"这一步;若要在 Paddle 上做 QAT 后折叠,需要自行合并多分支
  量化域——这正是本仓库 exp9→exp14b 路线与 `ppocr-quantized-domain-fold` skill
  解决的问题(见 `docs/axera_qat/plans/train_noreparam_infer_reparam.md`)。

### 5.5 与本仓库 PyTorch 实现的对照

| 维度 | Paddle 官方 QAT | 本仓库 PyTorch QAT |
| --- | --- | --- |
| QAT 训练图 | 多分支原图 + fake quant(**不 rep**) | ①rep 后单分支(exp13/15/16)②多分支训练、推理折叠(exp9/14b) |
| 权重量化 | per-channel S8(channel_wise_abs_max) | per-channel S8/S16(LSQ 可学习 scale) |
| 激活量化 | per-tensor U8 affine(moving min/max) | U8/U16 per-tensor(MinMax/LSQ) |
| QAT→部署 | 多分支量化图直接部署,无融合步骤 | `convert_pt2e` 后导出 QDQ;折叠路线用量化域折叠 finetune |
| 重参化 | 仅浮点部署路径,融合后纯 Conv 无 BN | `rep()` 与 Paddle 逐行对应,同样纯 Conv 无 BN |

补充说明:

- 本仓库的 `rep()`(pytorchocr/modeling/backbones/rec_lcnetv3.py)与 Paddle 一致:
  融合公式、分支求和、裸 `nn.Conv2d`、`reparam_conv -> LAB -> Act`,无 BN;
- `insert_identity_bn`(融合后保留一个实测统计初始化的 BN)是**本仓库自己的扩展**
  (QARepVGG insert_bn 风格),Paddle 原始代码没有该设计;exp7/exp8 系列实验证明保留的
  BN 训练中会漂移导致导出折叠爆炸(γ/√(var+ε) 放大至 2000+),已放弃,正式路线回到与
  Paddle 一致的"重参化 = 纯 Conv、无 BN"(详见
  `docs/axera_qat/records/reparameterization_qat_issues.md` 问题 2/3/8);
  对照:YOLOv6 v0.3.0 QARepVGG 的"求和后 BN 全程存在"结构(多分支训练 → 折叠保留
  BN → QAT)不存在该失配,是问题 2 候选修复 3 的实现来源,见同记录问题 13 与
  `qarepvgg_quantization_reference.md` §8.4;
- 权重检查命令与 QAT 机制细节见 2026-08-18 排查记录;本仓库 QAT 合同见 AGENTS.md。

## 6. 参考与说明

- 官方 v4 blog:`references/PaddleOCR/docs/version2.x/ppocr/blog/PP-OCRv4_introduction.md`
  (https://www.paddleocr.ai/latest/version2.x/ppocr/blog/PP-OCRv4_introduction.html );
- 官方 v5/v6 概要:`references/PaddleOCR/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md`、
  `PP-OCRv6/PP-OCRv6.md`(核心技术升级部分,含官方公式与对比表);
- 本文代码出处均为官方实现:`references/PaddleOCR/ppocr/modeling/...:行号`,
  数据管线:`references/PaddleOCR/ppocr/data/...:行号`;
- 官方 QAT 机制:`references/PaddleOCR/ppocr/utils/qat.py`、
  `references/PaddleOCR/tools/export_qat_onnx.py`、`ppocr/utils/export_model.py`;
  Paddle 3.0 内置 `paddle.quantization`(非仓库文件);
- 仓库侧部署裁剪(检测仅 shrink、识别仅 CTC)与本仓库 QAT 合同见 AGENTS.md,
  不在官方实现范围内。

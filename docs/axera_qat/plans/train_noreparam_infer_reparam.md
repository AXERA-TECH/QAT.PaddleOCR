# 训练非重参化、推理重参化 QAT 方案(量化感知重参化折叠)

## 目标

QAT 训练使用**非重参化**(多分支)结构,推理导出时**重参化折叠**为单分支
结构。避免重参化模型 QAT 的训练崩溃/精度问题(见 `rep_keep_bn_plan.md`),
同时获得部署单分支的推理效率。

```
训练图:  多分支 LearnableRepLayer(3x3/1x1/identity + BN)+ 各分支独立 QDQ
部署图:  rep 融合单 conv + 单一激活 QDQ(重参化折叠)
```

## 背景与动机

- 重参化(rep)后训练 QAT 存在"BN 折叠爆炸"(γ/√(var+ε) 放大至 2000+)、
  训练崩溃等问题(详见 `records/reparameterization_qat_issues.md`);
- 非重参化训练(exp9)QAT 稳定,导出 QuantONNX 精度与训练一致
  (converted acc=0.278 vs 训练 0.276);
- 但非重参化推理图节点多(136 Conv vs 38 Conv),部署效率低;
- 本方案:训练用多分支(稳定),推理折叠成单分支(高效)。

## 单层折叠误差验证(blocks2.0.pw_conv)

**方法**:真实输入(经 conv1 + blocks2.0.dw_conv 提取),对比:
- 多分支 QDQ 模拟(各分支权重 per-channel S16 + 分支激活 per-tensor U16
  独立量化 + Add + lab + act);
- 折叠单 conv(分支 BN 吸收 `_get_kernel_bias`)+ 单激活 QDQ。

**结果**:

| 对比 | MAE | max_abs | 余弦 |
| --- | ---: | ---: | ---: |
| 多分支 QDQ vs 浮点(量化本身) | 2.324 | 52.44 | - |
| **折叠 QDQ vs 多分支 QDQ(折叠引入)** | **0.00035** | **0.0032** | **1.000028** |

**结论**:折叠本身几乎无损(误差比量化误差小 3 个数量级)。权重 per-channel
量化逐通道独立、折叠后重算不累积;激活多分支多次量化 ≈ 单次量化。

## 折叠后量化参数初始化(从训练模型继承)

折叠后的单 conv 需要权重/激活 scale 与 zero_point。训练 checkpoint 含
LSQ 学习的分支级 per-channel 权重 scale(如 blocks2.0.pw_conv 的
activation_post_process_30/34/37/40,各 32 通道)与 per-tensor 激活 scale。

### 权重:可精确继承(方案 B:分支 scale 逐通道 max)

折叠后单 conv 权重 = 各分支权重 × BN scale 的线性组合。权重 scale 推导:

```
W_fused = Σ_i (s_i_bn · W_i)
scale_fused = max_i(scale_i_lsq)    # 方案 B:逐通道取各分支 LSQ scale 最大值
```

方案对比(blocks2.0.pw_conv 通道 0-2):

| 方案 | 值(通道0) | 说明 |
| --- | --- | --- |
| **A. 重算 max\|W_fused\|/qmax** | **2.62e-3** | **可行,量化误差 1.5e-2(已选)** |
| B. 分支 scale 逐通道 max | 1.35e-3 | **不可行:折叠权重数值大(100-1000),scale 偏小致量化饱和,误差 731** |
| C. 分支 scale 逐通道和 | 2.22e-3 | 线性组合近似,未验证 |

**方案 B 实测失效**:折叠权重 `W_fused` 由 BN 吸收决定,数值达 1e2-1e3
(此前发现 absmax 997),而分支 LSQ scale 仅 1e-3 量级;`round(W/scale)`
严重饱和截断(远超 qmax 32767),折叠权重量化误差 731。方案 A 重算
`max|W_fused|/qmax` 误差仅 1.5e-2,正确。

### 激活:需校准(当前单层不校准)

训练时各分支激活独立观测 scale_i;折叠后单 conv 输出分布不同,LSQ 激活
scale 无法逐通道映射。**需用训练集校准**统计折叠后模型激活 min/max。
当前单层验证阶段不校准,激活量化用 batch 内统计近似。

## 验证产物

单层折叠前后 QuantONNX:
`exports/quantonnx/exp9_ppocrv5_mobile_rec/single_layer/`
- `blocks2_pwconv_before_quant.onnx`:多分支(4 Conv、14 Q、23 DQ、50 节点)
- `blocks2_pwconv_after_quant.onnx`:单分支(1 Conv、7 Q、13 DQ、26 节点)

全量折叠(浮点结构正确,量化参数冷启动):
`exports/quantonnx/exp9_ppocrv5_mobile_rec/ppocrv5_mobile_rec_exp9_folded_inference_qdq.onnx`
(38 Conv、0 BN、946 节点,QDQ 校验通过)

## 待办

- [x] 单层折叠误差验证(0.00035,余弦 1.0);
- [x] 单层折叠前后 QuantONNX 导出(结构对比);
- [x] 折叠后权重 scale 方案对比(方案 B 饱和失效 → 方案 A 正确);
- [x] 单层折叠精度对比(方案 A scale:折叠 vs 多分支 MAE 0.0257、余弦 0.99982);
- [x] 激活 scale 训练集校准(2026-08-13 经 exp10a/exp12b 验证落地,见下方状态);
- [x] 全量折叠 + QuantONNX 精度核验(exp10a/exp12b,折叠后与源训练一致,见下方状态);
- [ ] 训练时多分支共享量化域(中期,理论无损,需重训;尚未实施)。

## 状态

2026-08-12:方案记录,单层验证完成,权重 scale 方案 A 确认(方案 B 饱和
失效),单层折叠精度对比通过。激活校准与全量折叠待推进。

2026-08-13:激活校准与全量折叠已通过 exp10a/exp12b 验证落地(见下),
方案进入可复用状态,流程固化为 skill `ppocr-quantized-domain-fold`。

2026-08-13 更新:
- **量化域折叠是唯一可行的折叠方式**:exp9 checkpoint 权重是 LSQ 量化域
  耦合的(裸权重/fake-off 评估 acc=0.0),不能剥离 quantization 参数当浮点
  权重用。正确折叠 = 对每分支取 `fq(fold(w))`(含 clip)后求和,折叠后
  eager acc=0.535(与 exp9 fake-quant-on 预测对齐)。
- exp10a(折叠起点 20 epoch 微调)验证成功:epoch1 0.593 → epoch20 0.600。
- **QAT JSON `output` 字段修复**:`get_config` 此前硬编码
  `output_dtype=input_dtype`,qspec 中所有 `output`/`output_is_symmetric`
  无效。修复后 matmul2(softmax·V)输出 U16 真正生效,QuantONNX identity
  从 5 个降至 1 个(仅 SE avg_pool u16→u16,确认不处理)。
- 新 qspec `configs/qat/ppocrv5_mobile_rec_u16s16_attn_s16_lsq_v2.json`
  (qkv U16→S16、scale mul S16、matmul1 S16、softmax S16、matmul2 S16→U16)。
- exp11(exp9 折叠权重 + v2 qspec,50 epoch)完成:0.5965。
- exp12(pretrained 浮点 + v2 qspec,重参化,50 epoch)完成:best 0.610
  (早期 epoch3-4 崩溃后自愈);QuantONNX ORT 0.587。
- exp12a(pretrained 浮点 + v3 qspec,非重参化,50 epoch)完成:best 0.6172;
  QuantONNX ORT 0.616,导出无损。
- **qspec 节点名随图形态变化**(问题 11):scale Mul 在非重参化训练图
  (mul_352/353)与折叠训练图(mul_60/61)名字不同;v3 qspec 同时含两组名。
- **折叠 finetune 已验证可落地**(exp12b):量化域起点 0.558-0.605,
  finetune 20 epoch 后 0.612,与训练 0.617 差值 <0.05;
  流程固化为 skill `ppocr-quantized-domain-fold`。

## 折叠后训练集微调方案(exp10,2026-08-12 决策)

### 动机与流程

eager 折叠后模型为**裸 conv 单分支(无 BN)**,量化参数是冷启动的
(权重 scale 方案 A 重算、激活未校准)。用训练集微调让 LSQ scale/zero_point
重新学习,达到与 exp9 训练一致的精度(目标**无损**:converted acc ≈ 0.545)。

```
exp9 非重参化 QAT 权重(best.pt)
  → reparameterize_for_deploy(裸 conv 折叠,无 BN)
  → 重新 prepare(LSQ 新量化域)
  → 10 步 smoke 门禁(验证 loss 不爆炸、梯度有限)
  → 训练集微调 20 epoch(量化参数重新学习)
  → 导出 QuantONNX → eval.py 精度核验
```

### 决策记录(2026-08-12)

| 决策项 | 选择 | 说明 |
| --- | --- | --- |
| 微调 epoch | **20** | 平衡成本与收敛 |
| 学习率 | **更小(可低于 3e-5)** | 权重已适配量化,微调仅适配折叠结构,用小 lr 防破坏;单层 scale 重算后与折叠前余弦相似度高,起点良好 |
| 权重 scale 初始化 | **方案 A(重算 max\|W_fused\|/qmax)** | 已证可行(误差 1.5e-2);微调中 scale 重新学习 |
| 10 步 smoke 门禁 | **是** | 防 exp2 裸 conv 崩溃教训(loss 17→52) |
| 精度目标 | **无损(converted ≈ 0.545)** | 期望折叠微调后与 exp9 训练一致 |

### BN 折叠说明(2026-08-12)

单层 before_quant.onnx 无 BN 节点是**正确行为**:
- PT2E QAT 的 `_fuse_conv_bn_qat`(prepare)+ `_fold_conv_bn_qat`(convert)
  把各分支 BN 参数吸收进 Conv 权重/bias(如 `layer.conv_kxk.N.conv.weight_bias`);
- exp9 原图的 19 个 BN 全是 **identity 分支**的(纯 BN 无 conv,无法折叠);
- 单层 pw_conv(16→32)无 identity 分支,故 BN=0;
- 这是 QAT 导出合同(BN=0)要求的正确结果。

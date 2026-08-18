# 重参化保留 BN(conv + bn 结构)方案

## 背景与问题

PP-OCRv5 mobile rec 的 QAT 训练在 `reparameterize=true`(部署重参化)下不稳定:

- Exp2(reparam=true, 无 BN):训练崩溃,loss 10 步内 17 → 52;
- Exp6(reparam=true + LSQ):训练 loss 正常但 epoch1/2 val_acc=0;
- Exp7(reparam=true + identity BN 冷启动):训练稳定(loss 56→53 不爆炸),
  但 val_acc=0。根因:插入的 identity BN(weight=1, bias=0, running_mean=0,
  running_var=1)与真实激活分布不匹配,训练时 BN 用 batch stats 强制归一化,
  验证时用 running stats,两者错位(train vs eval 输出 max diff 高达 68-75),
  需要约 200 步预热才能恢复一致(实验见下文)。

对照:Exp3/4/5(`reparameterize=false`,保留原始 pretrained BN)训练稳定且验证
正常(epoch2 acc 0.37-0.39)。原因是 pretrained BN 的 running stats 匹配真实
数据分布,训练(batch stats)与验证(running stats)语义一致。

## 方案:重参化只融合 conv 与 conv,保留 BN 结构

重参化中主要是 conv 与 conv 之间的融合(多分支叠加),BN 位于其中某一分支上
(如 `LearnableRepLayer` 的 `conv_kxk`/`conv_1x1`/identity 分支)。目标:重参化
后的结构为 `conv + bn`(BN 保留 pretrained 参数),而不是全融合成裸 conv。

### 数学推导(rec_lcnetv3 `LearnableRepLayer`)

原多分支结构:

```
out = BN_id(x) + BN_1x1(conv_1x1(x)) + Σ BN_kxk(conv_kxk(x))
    = Σ_i [ s_i·conv_i(x) + b_i ]
```

其中逐通道 `s_i = γ_i / √(σ_i² + ε)`, `b_i = β_i - μ_i·s_i`(eval 语义)。

选择分支 k 的 BN 作为保留的参考 BN,融合公式:

```
conv_fused.weight = Σ_i (s_i / s_k) · W_i
conv_fused.bias   = (Σ_i b_i - b_k) / s_k
```

等价性证明:

```
bn_k(conv_fused(x)) = s_k · conv_fused(x) + b_k
                    = s_k·[Σ_i (s_i/s_k)·conv_i(x) + (Σ_i b_i - b_k)/s_k] + b_k
                    = Σ_i s_i·conv_i(x) + Σ_i b_i  ≡ 原 out  ✓
```

- 参考分支 k 的选择不影响精度(线性缩放),推荐 `conv_kxk[0]`(主分支);
- 数值稳定性:参考分支 BN weight 可为负(pretrained γ 有负值),`s_k` 需按
  **保符号**处理——不能用 `clamp(min=1e-6)`(会把负值钳成正数破坏数学);
  实测最小 `|s_k|=9.3e-9` 且无精确 0,仅在 `|s_k|<1e-12` 时替换为符号保护值;
- 分支存在性:identity 分支仅当 `out_channels == in_channels and stride == 1`
  时存在;conv_1x1 分支仅当 `kernel_size > 1` 时存在;融合时需逐个判断。

`ConvBNLayer`(单分支 conv+bn):`insert_identity_bn=True` 时不融合,保持
conv + bn 原结构即可(forward 不变)。

### 与 identity BN 冷启动方案的对比

| 方案 | BN 参数 | 训练/验证一致性 | 预热 |
| --- | --- | --- | --- |
| identity BN(已废弃) | 冷启动 w=1/b=0/μ=0/σ²=1 | 错位,val_acc=0 | 需要 ~200 步 |
| **保留 pretrained BN(本方案)** | pretrained 权重,统计匹配分布 | 天然一致 | 不需要 |

导出路径两者相同:PT2E 的 `_fuse_conv_bn_qat` 在 prepare 时吸收 BN(conv 权重
更新,BN 节点保留供训练),convert 时折叠回 conv → QuantONNX BN=0 合同自动满足。

## 实现要点

1. `rec_lcnetv3.py`(仅此文件,其他 backbone 不动):
   - `ConvBNLayer.rep(insert_identity_bn=False)`:True 时不融合(conv+bn 原样);
   - `LearnableRepLayer.rep(insert_identity_bn=False)`:True 时按上述公式构造
     `reparam_conv` + 保留 `self.bn`(复制参考分支 BN 参数);
   - forward 的 is_repped 分支:`reparam_conv → bn → lab → act`;
   - 删除 identity BN 冷启动代码(`keep_identity_bn`/`identity_bn`);
   - s_k 下界检查;
   - `LCNetV3Block.rep` / `PPLCNetV3.rep` 透传参数(已有)。
2. 参数接口沿用 `insert_identity_bn`(CLI `--insert-identity-bn`、profile 字段、
   resume 合同不变),语义更新为"重参化后保留 BN(conv+bn 结构)"。

## 验证门禁

1. eager rep(`insert_identity_bn=True`)后输出与原始多分支结构逐层
   max_diff ≈ 0(数学等价);
2. prepared 图保留 batch_norm 节点,convert 后 BN=0;
3. 真实数据训练 65 步:running stats 稳定、train vs eval diff 小、
   val_acc 非零;
4. 全量回归测试通过;
5. 正式训练 Exp7(LSQ + warmup5 + reparam=true + insert_identity_bn)。

## 过程记录

### 2026-08-11 identity BN 冷启动实验(已废弃)

`/tmp/opencode/identity_bn_control.py`:rep 后插入 identity BN(图级 call_module
插入 → 改 eager 模型结构插入),10 步训练 loss 56.3→52.9 稳定,但:

- BN running stats 训练 65 步从 (0,1) 涨到 mean≈4/var≈15 仍未收敛;
- 同一 batch train vs eval 输出 diff 68-75(BN 语义错位);
- 预热 200 步(仅 forward 更新 running stats)后 diff 降至 0.015,
  说明预热可行但成本高;
- epoch1 验证 val_acc=0。

结论:identity BN 冷启动方案废弃,改为保留 pretrained BN。

### 2026-08-11 全融合 rep 崩溃根因定位

- Exp2/Exp6:rep 融合成裸 conv 后无归一化锚点,训练/验证均不稳定;
- `_fuse_conv_bn_qat` 对 eager conv+bn 的处理:prepare 时吸收 BN 参数到 conv
  权重并保留 BN 节点供训练,convert 时折叠——这是官方 QAT conv-bn 路径,
  本方案直接复用。

### 2026-08-11 keep-BN 重参化模型对齐验证(对比模型正确)

在 `rec_graph="deploy"`、eval 模式下,随机输入 `randn(2, 3, 48, 320)` 对齐:

| 对比对象 | max diff |
| --- | --- |
| keep-BN rep(`reparameterize=True, insert_identity_bn=True`) vs 原始多分支(`reparameterize=False`) | 8.5e-6 |
| full rep(旧行为,无 BN) vs 原始多分支 | 8.5e-6 |

结论:keep-BN 重参化后的 `conv + bn` 结构与重参化前的原始多分支浮点模型在
eval/静态语义下**数学精确等价**,浮点舍入误差与旧 full-rep 路径同量级,
未引入额外精度损失。确认对比正确后,不再重复真实数据精度测试。

实现过程中发现并修复的问题:

1. **BN weight 可为负导致 NaN**:pretrained BN 的 `gamma` 存在负值,融合公式中
   `scale_k = gamma / sqrt(var + eps)` 为负;初期用 `clamp(min=1e-6)` 把负值
   钳成正值,破坏数学符号导致输出 NaN。修复为保符号保护:
   `scale_k = torch.where(scale_k.abs() < 1e-12, ones * sign(scale_k), scale_k)`
   ——仅在绝对值接近 0 时替换为符号保护值,不改符号;
2. **实测 `|scale_k|` 分布**:最小 9.3e-9(blocks6.2.pw_conv),无精确 0;
   `clamp(min=1e-6)` 会把 9.3e-9 放大 100 倍引入 0.041 的偏差,移除后恢复
   8.5e-6;
3. 参考分支选择:`conv_kxk[0].bn`,全模型 29 个 LearnableRepLayer 均存在该分支。

### 2026-08-11 keep-BN QAT 训练验证(val_acc=0 未解决)

- prepared 图保留 34 个 `aten.batch_norm` 节点,convert 后 BN=0(合同满足);
- 65 步训练 loss 54.4→42.1 稳定下降(无爆炸);
- 但 epoch1 验证 `val_acc=0.0`:融合后的 BN 输入是**多分支之和**
  `conv_fused(x)`,而保留的 running stats 是参考分支**单分支输出**的分布,
  两者分布不同;训练时 BN 用 batch stats 实时归一化(正常),验证时用
  running stats(仍停留在参考分支分布)→ 错位 → argmax 全错;
- 待验证:200 步 forward 预热让 running stats 收敛到 conv_fused 真实分布后,
  val_acc 是否恢复(与 identity BN 预热实验同路径)。

### 2026-08-11 诊断实验:预热无效,训练/验证语义不等价(待办)

诊断脚本 `/tmp/opencode/diagnose_keep_bn.py`(GPU2,65 步 AdamW 训练 + 200 步
forward 预热,非 LSQ):

| 指标 | 值 |
| --- | --- |
| BN running_mean 训练 65 步 | -0.26 → 1.46(剧烈漂移) |
| 65 步后 train vs eval 输出 max diff | 166 / 26 / 9 |
| 200 步预热后 running stats | 1.459 → 1.463(基本稳定) |
| **预热后 train vs eval max diff** | **178 / 34 / 7.7(无改善)** |

结论(推翻旧假设):

1. **预热无效**:keep-BN 的 running stats 预热后已收敛稳定,但 train/eval
   输出差异依旧巨大——与 identity BN(预热 200 步后 diff 0.015)完全不同,
   "running stats 收敛即可修复"的假设不成立;
2. **训练语义不等价(候选根因)**:原始多分支结构训练语义是"各分支 BN
   (batch stats)归一化后相加";keep-BN 重参化后是"相加后再整体 batch
   stats 归一化"。eval 静态等价(固定 running stats)成立,但训练时 batch
   stats 归一化的动态行为不同 → 网络学习路径不同 → 验证必然错位;
3. 待办事项:
   - 补充数据确认机制:训练 65 步后,对比多个 batch 的 conv_fused 实际
     batch stats vs running stats 差距,以及 γ/β 漂移量——区分
     "EMA 滞后"与"训练语义不等价";
   - 调研 references/QARepVGG 中 QATRepVGG 的多分支 QAT 训练与重参化
     时序(训练时保持多分支 BN,推理/导出时才融合)是否可作为替代方案;
   - 候选方案:QAT 训练图保留**原始多分支结构**(reparam=false 语义,
     PT2E prepare 处理),只在导出 QuantONNX 时才做 keep-BN 重参化融合,
     避免训练语义改变。

### 2026-08-11 调研:QARepVGG 的 QAT 做法(references/QARepVGG)

**QAT 训练结构(QARepVGGBlockV2,非 deploy)**:
- 保持多分支:rbr_dense(3x3+BN) + rbr_1x1(1x1+BN) + rbr_identity(BN);
- 分支求和后**额外加一个 BN**(self.bn)作为 QAT 归一化锚点;
- forward:bn(se(dense(x) + 1x1(x) + id(x)));
- 部署时才融合所有分支 + 后 BN → 单 conv(switch_to_deploy)。

**insert_bn.py(对已融合部署图重新插 BN,与 keep-BN 方案最相关)**:
1. 融合单 conv 拆成 conv(无 bias) + BNStatistics(只统计不归一化) + biasadd;
2. 用训练集实测 500 batches,记录每个 conv 输出的 mean/var;
3. 统计初始化 BN,**保证数学精确等价**:

   ```
   gamma = std = sqrt(running_var + eps)
   beta  = bias + running_mean
   conv(x) + b ≡ BN(conv(x))    # 前向完全不变
   ```

**与 keep-BN 方案的本质差异**:

| 项 | keep-BN(本方案) | QARepVGG insert_bn |
| --- | --- | --- |
| BN 初始化 | deepcopy 参考分支 BN | 融合 conv 输出的训练集实测统计 |
| running stats 匹配 | 参考分支单分支分布,与 conv_fused 不匹配 | 实测 conv_fused 分布,完全匹配 |
| 融合权重 | 按 1/s_k 缩放 | conv 权重不动,bias 移入 BN |
| 数学等价 | eval 等价(8.5e-6) | 前向精确等价 |
| 训练/验证一致性 | 错位(val_acc=0) | 一致 |

**结论**:keep-BN 的失败根因是 BN 初始化来源错误——参考分支的 running stats
与融合后 conv_fused 的分布不匹配;QARepVGG 用实测统计初始化 BN,训练时 batch
stats 围绕真实分布波动、验证时 running stats 匹配,天然一致。这也解释了
"预热无效":预热只能把 running stats 拉向当前权重的分布,无法弥补初始化
路径本身的错误。

**候选修复(对齐 QARepVGG)**:
1. rep(keep_bn) 时 BN 不 deepcopy 参考分支,而是**用训练数据实测统计初始化**
   (gamma=std、beta=bias+mean),与 QARepVGG insert_bn 一致;
2. 或训练图保持原始多分支结构(reparam=false),导出 QuantONNX 时才融合,
   完全避免训练语义改变(与 Exp3 已验证行为一致);
3. QARepVGG 的"分支求和后额外 BN"思想可借鉴:若训练图保留多分支,可在
   求和后加 BN(与 lab 共存)作为归一化锚点。

### 2026-08-11 exp8 训练"卡死"定位与修复(LSQ observer 状态管理)

**现象**:exp8(LSQ + AdamW + keep-BN)epoch1 正常完成(3 分钟),epoch2 起
主进程 CPU 满载自旋、DataLoader worker 全部空闲,数十分钟无进展。

**定位**(faulthandler SIGUSR1 dump 主线程栈):
主线程卡在 `observer.py:1133 _non_linear_param_search`(HistogramObserver
非线性二分搜索)→ `_LearnableFakeQuantize.forward` → 训练前向。

**根因链**:
1. `_LearnableFakeQuantize.forward` 用 **`static_enabled`** buffer 决定是否
   运行 observer + `calculate_qparams()`(torch 2.6 实现,line 151);
2. torch 的 `disable_observer` 只设置 **`observer_enabled`**,对 LSQ 模块
   **无效**(`_LearnableFakeQuantize` 的 `enable_observer` 才是
   `toggle_observer_update`,即 static_enabled);
3. `enable_param_learning()` 内部调用 `toggle_observer_update(False)` 关闭
   static_enabled——LSQ 统计 pass 后训练本应正常;
4. **破坏者**:Trainer.evaluate 的 finally 无条件 `model.apply(enable_observer)`
   (旧代码)会触发 `_LearnableFakeQuantize.enable_observer()` →
   `toggle_observer_update(True)` → static_enabled 重新打开 → epoch2 每个
   前向都跑 HistogramObserver 非线性搜索(极慢)→ 表现为卡死。

**修复**(trainer.py):
1. evaluate 记录进入前的 observer 状态(`_any_observer_enabled`),finally 只
   在进入时已启用时才恢复,不再无条件 enable;
2. `_any_observer_enabled` 对 `_LearnableFakeQuantize` 检查 **`static_enabled`**,
   普通 `FakeQuantize` 检查 `observer_enabled`(LSQ 的 observer_enabled 恒为 1,
   不能作为判据——初次修复只查 observer_enabled 仍误判)。

**验证**:
- LSQ prepare 后 static_enabled=0;evaluate 后保持 0;evaluate 后训练
  2 步 3.43s(修复前 26.24s,卡死路径消除);
- 245 项回归测试全部通过。

**结论**:这不是 keep-BN 方案的 bug,而是 LSQ + evaluate 的 observer 状态管理
问题(与 BN 方案无关,任何 LSQ 训练都会触发)。

### 2026-08-11 后续方案:两阶段渐进 QAT(方案 A,待用户决策)

**问题**:重参化模型的 QAT 精度崩(reparam=true 训练崩溃 / val_acc=0),根因是
融合后单 conv+BN 的训练语义与 pretrained 多分支结构不一致(BN 输入分布错位)。

**方案 A(用户提出)**:两阶段渐进 QAT

```
阶段 1:多分支结构 QAT 训练(块内量化域共享 / 块内不量化,仅块边界 QDQ)
   → 训练/验证正常(与 Exp3 多分支训练一致,pretrained BN 语义正确)
阶段 2:重参化融合(多分支 → 单 conv)→ 从融合权重重新 prepare → 再训练几轮
   → 部署结构上继续收敛量化参数
导出:融合结构 → 单 conv + QDQ,BN=0 合同自动满足
```

**核心价值**:
1. 阶段 1 训练语义与 pretrained 完全一致,从根上避免 keep-BN/identity-BN
   的分布错位问题;
2. 融合发生在训练过程中(阶段边界),不是导出时——不存在"多分支独立 QDQ
   融合"的数学精度损失问题;
3. 阶段 2 从"阶段 1 已收敛的 QAT 权重"融合起步,而非 pretrained 冷启动,
   这是解决"重参化模型 QAT 精度崩"的关键。

**阶段 1 的量化边界设计**(待确认):
- 方案 a:块内不量化(仅块边界 QDQ)——块内浮点/权重量化,融合为恒等变换,
  边界 QDQ 直接继承,量化误差训练/部署完全一致(最优);
- 方案 b:块内各分支量化域共享(Add 输入共享 scale/zero_point)——融合后
  单 QDQ 与共享域一致,误差有界(|Σ round(x_i/s)·s − round(Σ x_i/s)·s|
  ≤ (n−1)·s/2);
- 方案 c:块内各分支独立量化域(当前默认)→ 融合误差最大,不可取。

当前 AXQuantizer 的 `_do_annotate_dyadic`(ax_quantizer_utils.py:998)给 Add
输入独立 qspec,未共享域;`SharedQuantizationSpec` 基础设施已存在
(ax_quantizer_utils.py:1694,用于 split/reshape),可复用。

**待澄清问题**:
1. "块内不量化"范围:仅激活不量化(权重仍量化)还是整个块浮点?
   (若权重不量化,导出时单 conv 权重要重新量化,精度无保障);
2. 阶段 1 块边界 QDQ 与阶段 2 的对应关系:边界 QDQ 直接继承?
3. 阶段 2 的 BN 处理:融合时吸收进权重(无 BN,可能复现 Exp2 崩溃)还是
   保留 BN(实测统计初始化 γ=std/β=mean)?从阶段 1 收敛权重起步,无 BN
   微调是否稳定需实验;
4. 阶段 1→2 量化参数(scale/zero_point)重新统计初始化还是继承?
   (PT2E 合同要求新图合同重新 prepare,scale 大概率重新统计)。

**关键验证点**(决定方案可行性):
- 阶段 2 融合结构(无 BN 或 keep-BN)从阶段 1 收敛权重起步,再训练几轮
  是否稳定(loss 不爆炸、val_acc 非零);
- 用随机输入 QAT smoke 覆盖阶段 1 → 融合 → 阶段 2 → convert → QuantONNX
  → QDQ 检查全链路。

**替代方案对比**:

| 方案 | 训练稳定性 | 验证一致性 | 量化精度 | 实现成本 |
| --- | --- | --- | --- | --- |
| A. 两阶段渐进(推荐) | 阶段1 多分支已验证 | 阶段1 正常 | 边界 QDQ 继承,块内恒等 | 中 |
| B. QARepVGG 式:融合结构 + BN 实测统计初始化,单阶段 | 待验证 | 统计初始化后一致 | 训练=部署同构 | 中 |
| C. 多分支 QAT + 导出 ONNX 融合 | 已验证 | 正常 | 有损(多分支独立 QDQ) | 低但精度差 |

**exp8 状态**:2026-08-11 终止(epoch3,val_acc=0 确认 keep-BN 方案失败;
卡死问题已修复但方案本身不可行)。

### 2026-08-11 方案定型:标准融合 + 各分支合并统计 BN(暂不实施,待用户决策)

**决策过程**:keep-BN(参考分支)失败后,曾考虑 QARepVGG 实测统计初始化
(需要 200 步 forward 统计 pass)。用户提出:保留"从各分支统计 BN"的思路,
不引入实测 pass,改为**融合公式改标准(不缩放)+ BN 统计从各分支合并推导**。

**数学设计**(rec_lcnetv3 `LearnableRepLayer`):

融合公式(标准,不缩放):
```
conv_fused.weight = Σ_i s_i·W_i        # _get_kernel_bias() 原始结果,不做 /s_k 缩放
conv_fused.bias   = Σ_i b_i
```
其中逐通道 `s_i = γ_i/√(σ_i²+ε)`, `b_i = β_i − μ_i·s_i`。

BN 统计从各分支合并推导(逐通道):
```
mean_fused = Σ_i s_i·μ_i + Σ_i b_i      # 精确(期望线性可加)
var_fused  = Σ_i s_i²·σ_i²              # 近似(忽略分支间协方差)
```

BN 参数初始化(前向恒等):
```
gamma = √(var_fused + ε)
beta  = mean_fused
running_mean = mean_fused
running_var  = var_fused
```

**关键性质**:
1. `conv_fused(x) ≡ 原始多分支 out`(标准融合精确);
2. `BN(conv_fused(x)) ≡ conv_fused(x)`(γ=√(rv+ε)/β=rm 使 BN 退化为恒等,
   无论统计近似与否)→ **前向与 pretrained 完全一致**;
3. 训练时 BN 用 batch stats 归一化(网络可适应),验证时用 running stats
   (合并推导,接近 conv_fused 真实分布)→ 一致。

**与 keep-BN 的本质区别**:keep-BN 的 γ/β 是参考分支的(非恒等)且 running
stats 与输入分布错位(val_acc=0);本方案 γ/β 恒等、running stats 匹配。
**与 QARepVGG 实测统计的区别**:不引入统计 pass/预热,零额外 forward 成本;
代价是 var 忽略分支间协方差(近似),训练中 BN 的 EMA 会持续修正。

**实现要点**(待实施):
1. `rec_lcnetv3.py` `LearnableRepLayer.rep(insert_identity_bn=True)`:
   - 融合用 `_get_kernel_bias()` 原始结果(不缩放),删除 scale_k/bias_k/
     torch.where 保护/deepcopy 参考分支逻辑;
   - 遍历 conv_kxk/conv_1x1/identity 分支,累加 `s_i·μ_i` 与 `s_i²·σ_i²`;
   - `mean_fused = Σ + Σ b_i`(bias 常数项),`var_fused = Σ s_i²·σ_i²`;
   - 构造 BN:running_mean/var = 合并统计,gamma=√(var+ε)/beta=mean,
     `_paddle_weight_decay` 标记保留;`keep_bn=True`;
   - forward 的 is_repped 分支:`reparam_conv → bn → lab → act`(已有);
   - ConvBNLayer.rep 保持现状(insert_identity_bn=True 不融合)。
2. `pytorchocr/diagnostics/checkpoint.py` `build_prepared_qat_checkpoint`:
   透传 `insert_identity_bn=bool(metadata.get("insert_identity_bn", False))`
   (当前缺失,重建 checkpoint 会丢 BN 结构)。

**验证门禁**(实施后执行):
1. eager 恒等:rep(insert_identity_bn=True) 后 `bn(conv_fused(x))` vs 原始
   多分支 out(随机输入,eval)→ max_diff ≈ 1e-5;
2. prepared 图 batch_norm 节点保留(34)、convert 后 BN=0;
3. 65 步真实数据训练:loss 正常下降、train vs eval diff < 0.1
   (对比 keep-BN 的 178)、val_acc 非零(关键决策点);
4. 全量回归测试;
5. 重启 exp8(AdamW + LSQ + warmup + insert_identity_bn)。

**风险**:方差近似(忽略协方差)若导致门禁 3 val_acc=0 → "融合后插 BN"路线
在 rec 上不可行,回退方案 A 两阶段渐进(见上文)。

**状态**:2026-08-11 方案已记录,暂不修改代码,待用户决策。

## 2026-08-11 BN 初始化方案组合与优先级(待用户决策,不改动代码)

针对"融合后保留 BN(conv + bn 结构)"的 BN 初始化,已讨论三个候选方案。
**优先级排序:QARepVGG 实测统计初始化(方案 2)优先于其余两个**,理由见下。

### 方案 1:identity BN 冷启动 + 预热(已实验,精度待完整验证)

- rep 融合后插入 identity BN(γ=1/β=0/μ=0/σ²=1);
- 预热 200 步(仅 forward 更新 running stats)后 train vs eval diff 降至
  0.015(小模型双 conv 验证);**预热后的完整训练验证未做过**(epoch1
  val_acc=0 是在未预热情况下测的);
- 缺点:插入瞬间前向被归一化改变(非恒等),需要预热恢复;200 步预热成本
  一次性的,可接受。

### 方案 2:QARepVGG insert_bn 实测统计初始化(优先)

参考 `references/QARepVGG/insert_bn.py`:
- 融合 conv 拆成 conv(无 bias) + BNStatistics + biasadd → 训练集实测
  500 batches 记录 conv 输出 mean/var;
- 初始化 BN:`gamma = std = √(running_var + ε)`, `beta = bias + running_mean`,
  running stats = 实测值;
- 数学:`BN(conv(x)) ≡ conv(x) + bias` **前向精确恒等**——插入即一致,
  不需要预热;running stats 实测匹配真实分布,训练(batch stats)/验证
  (running stats)天然一致;
- 成本:500 batches 统计 pass(一次性,类似 LSQ 的 act 统计 pass 基础设施
  已存在);
- **优先性说明**:这是唯一同时满足"前向恒等 + running stats 精确匹配"的
  方案,从根上避免方案 1 的预热依赖和方案 3 的方差近似;若资源允许
  (一次统计 pass),优先采用。

### 方案 3:标准融合 + 各分支合并统计 BN(见上文,方案定型章节)

- 融合公式改标准(不缩放),BN 统计从各分支 BN 参数解析合并:
  `mean_fused = Σ s_i·μ_i + Σ b_i`(精确)、
  `var_fused = Σ s_i²·σ_i²`(近似,忽略分支间协方差);
- 零额外 forward 成本(无需统计 pass);
- 代价:var 为近似;若训练中 EMA 无法修正该偏差导致 val_acc=0,则该路线
  不可行,回退方案 A 两阶段渐进。

### 组合决策建议

| 优先级 | 方案 | 前向语义 | running stats | 额外成本 | 风险 |
| --- | --- | --- | --- | --- | --- |
| **1(优先)** | **方案 2 QARepVGG 实测统计** | 恒等 | 实测精确匹配 | 500 batches 统计 pass | 低 |
| 2 | 方案 1 identity BN + 预热 | 预热后近似恒等 | 预热收敛 | 200 步预热 | 中(预热后完整训练未验证) |
| 3 | 方案 3 分支合并统计 | 恒等 | 近似(var 忽略协方差) | 零 | 中(近似失败则回退) |

**结论**:推荐按方案 2(实测统计初始化)实施,方案 1/3 作为备选;若方案 2
因统计 pass 成本或实现复杂度受阻,可降级到方案 1(先补预热后完整训练验证)
或方案 3(零成本近似)。当前状态:不改动代码,待用户决策。

### 2026-08-11 决策:开始实施方案 3(分支合并统计 BN)

用户决策:**开始尝试方案 3**(标准融合 + 各分支合并统计 BN),理由是代码改动
最小——不需要统计 pass(方案 2)、不需要预热(方案 1),仅改
`LearnableRepLayer.rep` 的 BN 构造与融合公式。实施与验证按上文"方案定型"
章节的门禁推进;若门禁 3(65 步训练 val_acc)失败,回退方案 2 或方案 A
两阶段渐进。

**实施与验证记录(2026-08-11)**:

- 代码改动:rec_lcnetv3.py `LearnableRepLayer.rep(insert_identity_bn=True)`
  改为标准融合(`_get_kernel_bias()` 原始结果不缩放)+ 各分支合并统计 BN
  (mean_fused = Σ s_i·μ_i + Σ b_i、var_fused = Σ s_i²·σ_i²,gamma=√(var+ε)/
  beta=mean);checkpoint.py `build_prepared_qat_checkpoint` 透传
  `insert_identity_bn`;
- 门禁 1(eager 恒等)通过:merged-stats BN rep vs 原始多分支 max diff 5.2e-6;
  BN 恒等检查(bn(conv_fused) vs conv_fused)diff 3.8e-6;
- 门禁 2(prepared/convert)通过:34 个 batch_norm 节点保留,convert 后 BN=0;
- **门禁 3(65 步训练 + 验证)失败**:val_acc=0.0,norm_edit_dis 0.0485
  (对比 keep-BN 的 0.0007 有改善但仍未对齐),val CTCLoss 50.4 偏高。
  训练 loss 正常下降(与之前方案一致)。

**结论**:方案 3 的 var 近似(忽略协方差)或融合后 BN 训练语义仍未解决
验证错位。按预案回退:下一步尝试方案 2(QARepVGG 实测统计初始化,500
batches 统计 pass)或方案 1(identity BN + 预热后完整训练验证)。

**门禁 3 补充(2 epoch 训练,2026-08-11)**:

| epoch | acc | norm_edit_dis | val CTCLoss |
| --- | ---: | ---: | ---: |
| 1 | 0.0000 | 0.0485 | 50.38 |
| 2 | 0.0072 | 0.1866 | 29.74 |

对比 Exp3(reparam=false 基线):epoch1=0.0385、epoch2=0.3899。

**附加诊断**:
- 未训练时 train vs eval diff:scheme3 为 131/33/14,reparam=false 也有
  67/21/12——**train/eval diff 大是 PT2E QAT 固有现象,不能作为方案成败
  判据**(此前用 diff 判断的标准有误);
- 训练 65 步后:scheme3 diff 141/35/12,reparam=false 109/43/10,两者同量级;
- scheme3 的 BN running stats 几乎不漂移(4.068→4.066,统计初始化准确),
  但验证仍错位。

**结论**:方案 3 的 acc 在恢复但远慢于 Exp3(epoch2 0.0072 vs 0.3899),
var 近似(忽略分支间协方差)拖慢收敛,2 epoch 内未通过门禁。按预案:
回退方案 2(QARepVGG 实测统计初始化)或方案 1(identity BN + 预热后完整
训练验证),或延长训练观察。待用户决策。

### 2026-08-11 方案 2(实测统计初始化)实施与门禁结果

**代码改动**:
1. rec_lcnetv3.py `LearnableRepLayer.rep(insert_identity_bn=True)`:标准融合
   (不缩放)+ identity 占位 BN(γ=1/β=0/μ=0/σ²=1),`keep_bn=True`;
2. bridge.py 新增 `initialize_kept_bn_statistics`:统计 pass 期间将 kept BN
   旁路(`keep_bn=False`,forward 跳过 BN,保证统计的是纯 conv_fused 分布),
   hook 挂在 reparam_conv 输出,EMA(momentum=0.9)累积 mean/var,统计完成后
   设置 `running_mean/var=实测值、gamma=√(var+eps)、beta=mean` 并恢复
   `keep_bn=True`;
3. checkpoint.py 透传 `insert_identity_bn`;train.py 集成统计 pass
   (prepare 前、非 resume/eval_only,`--bn-statistics-steps` 默认 200);
   profile.py 加 `bn_statistics_steps` 字段与 resume 合同。

**门禁结果**:
- 门禁 1(device 一致性):GPU 上 measured-BN rep vs orig 0.0034,与
  full-rep 基线 0.0031 同量级(device 固有浮点差异,非 BN 引入);
  注意 CPU vs CUDA 输出差异 ~0.003 是模型固有现象,对比必须同 device;
- 门禁 2(prepared/convert):34 个 BN 保留,convert 后 BN=0;
- **门禁 3(2 epoch 训练)通过**:

| epoch | acc | norm_edit_dis | val CTCLoss |
| --- | ---: | ---: | ---: |
| 1 | 0.1550 | 0.4127 | 24.67 |
| 2 | 0.1651 | 0.3958 | 23.54 |

**对比**:Exp3 基线 epoch1=0.0385/epoch2=0.3899;keep-BN 与方案 3 均 val_acc
恒 0 或接近 0。**方案 2 是首个 reparam=true 下 val_acc 非零的方案**
(epoch1=0.1550,epoch2=0.1651),证明实测统计初始化解决了 BN 训练/验证
语义错位。epoch2 增长放缓是 SGD 小步长(3e-6)正常现象。

**下一步**:全量回归测试 → 文档更新 → 用 LSQ + AdamW + warmup 配置重启
exp8 验证完整训练。

### 2026-08-11 exp8 正式训练(方案 2:实测统计 BN + LSQ + AdamW + warmup)

配置:`configs/qat/training/ppocrv5_mobile_rec_u16s16_sgd_dynamic_height_exp8_lsq_warmup_keep_bn.yml`
(新增 `bn_statistics_steps: 200`);tmux `ppocrv5-rec-u16s16-exp8-scheme2`;
GPU3;50 epoch;AdamW lr=3e-5 warmup5;LSQ;reparam=true + insert_identity_bn。

**训练过程中发现并修复**:统计 pass 后模型留在 CUDA,而 prepare 的
example_inputs 是 CPU,导致 `aten.add.Tensor` device 不匹配——统计结束后
`model = model.cpu()` 修复(train.py)。

**epoch 结果(记录至 epoch7)**:

| epoch | val_acc | norm_edit_dis | val_CTCLoss | train loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | (统计+prepare 阶段) | - | - | - |
| 2 | 0.2354 | 0.5908 | 14.75 | 15.86 |
| 3 | 0.1396 | 0.5096 | 18.51 | 13.89 |
| 5 | 0.0780 | 0.4836 | 16.75 | 13.45 |
| 6 | 0.2740 | 0.5952 | 11.32 | 11.04 |
| 7 | 0.1093 | 0.4921 | 13.88 | 10.63 |

**里程碑**:这是首个 reparam=true 下 val_acc 显著非零的完整训练
(epoch2=0.235、epoch6=0.274),对比之前所有 reparam 方案(keep-BN/方案 3)
val_acc 恒 0 或 <0.01。训练 loss 降至 10.6(之前方案 35-53),val_CTCLoss
波动下降(18.5→11.3),norm_edit 稳定 ~0.5。

**观察**:val_acc 在 0.08-0.27 间波动(epoch 间不稳定),但 val_CTCLoss 总体
下降、norm_edit 稳定,说明识别在改善;acc 波动可能源于 dynamic height
多尺度验证或 LSQ scale 学习阶段。与 Exp3 基线(epoch2=0.3899)相比仍有
差距,继续观察后续 epoch 收敛。

### 2026-08-11 方案:Conv 权重标量缩放 + lab.scale 补偿(训练精度,已确认待实施)

**问题**:`blocks2.0.pw_conv` 融合权重 absmax=997(全模型最大),源自 4 个
1x1 分支 BN scale 高达 ~1400(`γ/√(var+ε)`,部分通道 running_var~1e-13);
eager 模型与 ONNX 数值一致(非导出 bug),但权重范围过大影响训练精度。

**拓扑确认**(prepared 图 blocks2.0.pw_conv 链):
```
weight → mul(权重量化) → conv2d → div/add(BN 参数) → batch_norm
       → activation_post_process(BN 后单量化点)→ mul_8(lab.scale)→ add(lab.bias)→ act
```
- **BN 与 Conv 共享量化域**:activation_post_process 在 BN 之后,BN 无独立
  量化点;缩放穿过 BN 不影响量化域一致性;
- lab.scale 是**标量**(LearnableAffineBlock,非 per-channel);
- ONNX(float 导出)中 BN 已被 PT2E convert 折叠,Conv→Mul 直接相连。

**缩放方案(用户确认)**:
1. **标量缩放**(非 per-channel);
2. **保守 target**(默认值待定,倾向 max|W| 降至 ~100-300);
3. **只缩放超阈值的层**(如 absmax > 300 的 Conv),其余不动;
4. 缩放系数乘到**下一个 Mul(lab.scale)**。

**数学(标量 k)**:
```
原: y = s·(Wx + b) + b_lab
新: y' = (s·k)·(W/k·x + b/k) + b_lab = s·Wx + s·b + b_lab ✓
```
- Conv 权重 **和 bias 都除 k**(bias 不动则被放大 k 倍,不精确);
- lab.scale 乘 k;lab.bias 不动;输出精确不变;
- BN 在共享量化域内穿过,无需调整参数(统计 pass 在缩放后执行,
  γ=√(var+ε)/β=mean 恒等性质保持)。

**实现要点**(待实施,暂不改代码):
1. `LearnableRepLayer.rep(insert_identity_bn=True)` 融合后:
   - 若配置 `conv_weight_scale_target`(新 profile/CLI 参数)且
     `max|kernel| > 阈值`:`k = max|kernel| / target`;
     `kernel /= k`;`bias /= k`;`self.lab.scale.data *= k`;
   - 缩放发生在 rep() 内、统计 pass 之前——统计自动匹配缩放后分布,
     bridge.py 无需改动;
2. profile 新增 `conv_weight_scale_target`(None=off,数值=target_max),
   加入 metadata 与 resume 合同;CLI `--conv-weight-scale-target`;
3. 验证门禁:
   - eager 恒等:缩放后模型 vs 原始多分支(≈ device 基线 0.003);
   - 权重范围:超阈值层 absmax ≤ target;
   - prepared/convert:BN 保留/折叠不变;
   - 2 epoch 训练对比缩放前后 val_acc/loss;
   - 回归测试。

**状态**:2026-08-11 方案已记录,暂不改动代码。

### 2026-08-11 exp8 导出后评估异常:fake-off NaN / eval converted acc=0(待修)

**现象**:best.pt(epoch47,训练时 val_acc=0.5243)导出后:
- train.py eval-only converted:val_acc=0.0,val_CTCLoss=26(训练时 4.8);
- eval.py ONNX/PT2E 评估:acc=0.0;prepared(fake-on)输出正常(absmax 21),
  **prepared(fake-off)输出 NaN**;converted 输出 absmax 20(prepared 的 1/4);
- prepared vs converted 输出 diff 69.9。

**根因**:kept BN 的 γ 与 running_var 严重失配:
- 统计 pass 初始化 `γ=√(var+ε)` 使 BN 恒等,但**训练中 γ 可学习漂移**,
  running_var(统计 pass 旧值)不随训练更新(极小 var 通道 EMA 不敏感);
- 训练结束:6624 个 kept BN 通道中 87 个 `|γ/√(var+ε)|>10`,最大 1708
  (blocks6.1.dw_conv ch224:γ=5.40,var=1.16e-11);
- fake-off(eval running stats)时 `γ(x-μ)/√(var+ε)` 放大 1708 倍 → NaN;
- fake-on 正常(fake-quant 数值钳位掩盖);converted 折叠 BN 后同样放大。

**与权重缩放问题的关联**:极小 var 通道(分支 BN scale~1400)正是融合权重
异常大(997)的来源——同一批通道在 eval 浮点路径上爆炸。

**候选修复**:
1. 统计 pass 对 running_var 设下限(如 var ≥ 1e-4 或 1e-3),避免极小 var;
2. 训练中约束 γ 与 √(var+ε) 一致(如 γ 用 running stats 派生,不自由学习,
   或对 kept BN 冻结 affine 训练仅更新 running stats);
3. 训练验证阶段同步更新 running_var(EMA 用更大 momentum);
4. 结合权重缩放方案(标量 /k + lab.scale×k)降低极小通道权重影响。

**状态**:2026-08-11 已定位,待用户决策修复方向。

### 2026-08-11 修复:kept BN 训练 momentum 提高(方案 3,已实施)

**改动**:
1. `bridge.py initialize_kept_bn_statistics` 新增 `training_momentum` 参数:
   统计初始化后设置 `bn.momentum`(默认不改,即训练中 running stats
   用 BN 默认 0.1 更新);
2. train.py CLI `--bn-training-momentum` + profile `bn_training_momentum`
   (默认 None,范围 (0,1]),metadata/resume 合同;
3. exp8 profile 启用 `bn_training_momentum: 0.9`。

**验证**(2 epoch gate 测试,GPU2):
- fake-off 输出:NaN → **有限**(修复前 NaN);converted 输出有限;
- `|γ/√(var+ε)|>10` 通道:87 → **5**(6624 通道中);
- 2 epoch acc=0.155/0.165 与修复前一致(训练不受影响);
- 245 回归测试通过。

**机制**:统计 pass 初始化 γ=√(var+ε) 恒等;训练中 γ 可学习漂移,而
默认 momentum=0.1 使极小 var 通道(1e-11)的 running_var 更新极慢,
γ/√(var+ε) 放大(最大 1708)→ eval running-stats 路径爆炸(NaN)。
提高 momentum 到 0.9 使 running_var 快速跟踪训练分布,γ/√(var+ε)
保持有界。

**exp8b 正式训练**:tmux `ppocrv5-rec-u16s16-exp8b-mom09`(GPU3,50 epoch,
LSQ + AdamW + warmup + keep-BN + momentum 0.9),run 目录
`runs/exp8b_ppocrv5_mobile_rec_u16s16_adamw_lsq_warmup_keep_bn_mom09`。
训练完成后导出 QuantONNX 并用 eval.py 验证 eval converted acc 恢复。

### 2026-08-11 exp8b 完整训练完成(方案 3:bn_training_momentum=0.9)

**训练**:tmux `ppocrv5-rec-u16s16-exp8b-mom09`(GPU3,50 epoch,LSQ + AdamW +
warmup + keep-BN + `bn_training_momentum: 0.9`),run 目录
`runs/exp8b_ppocrv5_mobile_rec_u16s16_adamw_lsq_warmup_keep_bn_mom09`。

**epoch 结果**:
| epoch | val_acc | norm_edit_dis | val_CTCLoss | train loss |
| --- | ---: | ---: | ---: | ---: |
| 44(best) | **0.5450** | 0.7906 | 4.82 | 6.98 |
| 48 | 0.5402 | 0.7910 | 4.78 | 6.93 |
| 49 | 0.5079 | 0.7652 | 8.59 | 6.82 |
| 50 | 0.5364 | 0.7913 | 4.71 | 6.86 |

- best.pt = epoch44(val_acc 0.5450,高于 exp8 的 0.5243,略有提升);
- 训练 loss 6.86-6.98,val_CTCLoss ~4.7-4.8。

**eval.py converted 评估(best.pt,全量 2077 样本)**:
- acc=0.2638,norm_edit=0.6246,**输出有限(不再 NaN)**;
- 对比 exp8 旧版 converted acc=0.0/NaN——momentum 修复避免了完全崩溃;
- 但 converted(0.2638)与训练 prepared(0.5450)仍有 ~28pp 差距。

**γ/var 一致性(50 epoch 后)**:
- `|γ/√(var+ε)|>10` 通道:97 个(6624 中),max ratio 2098;
- 与 exp8 的 87 个(最大 1708)相当——**momentum 修复在 2 epoch gate 测试
  中有效(bad 87→5),但完整 50 epoch 后 γ 漂移再次超过 running_var
  跟踪,未根治**;
- 结论:方案 3(提高 momentum)只部分缓解(消除 NaN、converted 非零),
  未解决 converted 精度差距;γ/var 失配的根本修复仍需方案 1(var 下限)
  或方案 2(γ 派生自 running_var / 冻结 affine)。

**工具**:`tools/eval.py`(支持 --onnx/--pt/对齐模式)、`numpy_error_stats`
新增 mse/cosine_similarity、`convert_prepared_model` 保留 graph 属性、
resume 合同修复——均已通过 245 回归测试。

**清理**:runs/ 下各实验删除中间 epoch checkpoint,25G → 5.0G;
exp8 保留 best/last/epoch_0050,exp4-7 保留 best/last,exp8b 保留 best/last。

## 2026-08-18 收尾:keep-BN 方案最终结论与 LAB 折叠进 BN 的评审

本计划的"待用户决策"项至此全部有最终答案,记录如下(详细记录与对照见
`records/reparameterization_qat_issues.md` 问题 3/8/13/16 与训练记录 §31-38)。

### 8.1 keep-BN 方案最终结论(exp8 系列收尾)

- **exp8c(2026-08-12,variance floor 1e-4)**:converted acc=0.0534,比 exp8b
  更差。根因:var floor 只在统计初始化瞬间生效,训练中 BN EMA(momentum=0.1)
  把极小 var 通道(1e-11)拉回,54 个通道训练后 var<1e-4,折叠系数照常放大
  (2144)——方案 1 单独无效;
- **exp8d(2026-08-12,冻结 kept BN running stats)**:PT2E 图节点 momentum
  pin 为 0(Trainer 每次切回 train 重 pin),running stats 漂移恒 0、γ 可学
  (2.1e-3),机械上可行;但 2026-08-13 用户决定转向"训练非重参化、推理重参化"
  路线,exp8d 于 epoch9 终止(best val_acc=0.1006);
- **最终结论**:keep-BN(折叠后插 BN)方案整体废弃。根因链:pretrained 权重
  存在极小 var 通道(1e-13)→ 统计初始化恒等成立后训练中 γ 漂移而 running_var
  滞后 → `γ/√(var+ε)` 放大至 2000+ → convert 折叠爆炸。**在 pretrained
  权重 + 已有量化域场景下,"插入式 BN"不可行**;QARepVGG 的"求和后 BN"成立
  前提是从零训练(统计与 γ 同步演化),见问题 13;
- 替代路线(已落地):非重参化训练(exp9)+ 量化域折叠 finetune(exp10a/exp12b/
  exp14b,skill `ppocr-quantized-domain-fold`),以及纯重参化无 BN
  (exp13/15/16)。

### 8.2 LAB(Mul/Add)折叠进 BN 的数学评审(问题 16)

用户提出:重参化 conv 前/后的 LAB(`Mul → Add`,标量 `s,b`)折叠进插入的 BN,
用 Mul/Add 的值初始化 BN 的 4 变量。结论:**数学精确(eval),但 train()-mode
真 BN 路线与 exp8 同构,已否决;推荐"直接吸收进 Conv"**。

数学(2 方程 4 未知数 → 2 参数解族):

```text
LAB: y = s·x + b;  BN eval: y = γ(x−μ)/K + β, K = √(σ²+ε)
→ 对任意 μ、σ²: γ = s·√(σ²+ε), β = b + s·μ
规范解(恒等统计): μ=0, σ²=1−ε → γ=s, β=b
```

- **train() 语义**:`F.batch_norm(training=True)` 用 batch 采样统计,与
  s·x+b 只在 x̄=μ 且 v̂=σ² 时相等——用真实统计初始化只是"近似平滑"(误差
  ~σ/√N),不是"丝毫不差";0/1 初始化会断崖,该判断正确;
- **PT2E 链路**(torch 2.6):prepare 时 `_fuse_conv_bn_qat`
  (quantize_pt2e.py:179)把 Conv→BN 替换为近似融合子图(conv(scale 权重)→
  Div → Add → `F.batch_norm(training=True)`,qat_utils.py:107-145),BN 节点
  保留(momentum args[6] 可 pin,即 exp8d 机制);convert 时 `_fold_conv_bn_qat`
  (L243)吸收进 Conv → 单 Conv,BN=0 合同保持;
- **与 exp8 对照**:用户公式是 `initialize_kept_bn_statistics`
  (bridge.py:781,s=1,b=0 特例)的推广,基础设施齐全,但问题 3/4/8 的
  γ/var 失配风险全部适用。

### 8.3 拓展:hardswish → Mul → Add → Conv 的折叠

同一公式,BN 输入统计换成 **hswish 输出统计**(μ_z, σ²_z):

```text
γ = s·√(σ²_z + ε), β = b + s·μ_z;  BN_eval(z) = s·z + b   # 精确
```

- convert 后与"直接折入下一 conv"殊途同归:`W' = s·W_next`,
  `b' = s·b_next + b·ΣW`(偏置是**输入侧常值注入 b·ΣW**,与 conv 侧 LAB 的
  `s·b_c + b` 不同);
- hswish 输出非负,统计比 conv 输出稳定,但 dead channel(var≈0)同类风险;
- 量化域合并极简:仅权重 per-channel scale ×s(逐通道精确),输出/输入 QDQ
  复用相邻既有域,无 scale/zp 重算。

### 8.4 推荐路径:直接吸收进 Conv(训练图零变化)

| 方案 | 训练图 | 部署图 | 风险 |
| --- | --- | --- | --- |
| train()-mode 真 BN | Conv→BN(统计初始化) | 单 Conv(convert 折叠) | γ/var 失配(exp8 实证),per-channel 自由度变更,checkpoint 不兼容 |
| **直接吸收进 Conv(推荐)** | **Conv→LAB 不变** | **单 Conv(导出期折叠)** | 仅需数学证明+测试;checkpoint 可复用 |

- 折叠位置:conv 后 LAB → 同 conv(`W'=s·W, b'=s·b_c+b`);act 后 LAB → 下一
  conv(`W'=s·W_next, b'=s·b_next+b·ΣW`);尾部 LAB 后接 CTC Linear 同样吸收;
- 训练图/checkpoint 合同不变——exp16 checkpoint 可直接复用,无需从浮点
  权重重新 prepare;lab_lr_multiplier=0.1 合同保留;
- 收益:QuantONNX 全部 56 个 lab Mul/Add 消失;exp15/16 的 S16 标量 Mul
  三处特判(`swap_scalar_first_quantized_muls`、S16 dyadic 标量 qspec、
  TENG 对齐断言修复)可简化/删除;
- 门禁(合同 7):单层数值验证 → 全图 ORT_DISABLE_ALL 回归 → verify_qat_onnx
  + QDQ 结构审计 → ORT 全量 val 精度回归。

**状态**:2026-08-18 方案已评审记录,未改动代码;实施时按 8.4 门禁顺序执行。

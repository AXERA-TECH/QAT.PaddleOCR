# 重参化模型 QAT 问题记录

本文记录"Paddle 权重 → PyTorch 结构 → 重参化(rep)→ keep-BN → PT2E QAT"全链路上
发现的所有问题、根因、修复尝试与当前状态,作为后续工作的问题清单。所有实验细节、
配置和指标可追溯至 `docs/axera_qat/plans/rep_keep_bn_plan.md` 与
`runs/exp2~exp8b/` 的 train.log。

## 问题 1:重参化后一训就崩(已解决)

**现象**:Exp2(reparam=true,融合成裸 conv 无 BN)训练崩溃,loss 10 步内
17→52;Exp6/Exp7(reparam=true)val_acc 恒 0。

**根因**:
- 融合后无 BN 归一化锚点,裸 conv 训练时梯度路径不稳;
- Exp2 参数域审计(`artifacts/accuracy_baseline/p4_structure/exp2_debug_epoch2_qat_audit_20260807.json`)
  进一步定位两个放大因素:
  1. **LAB(LearnableAffineBlock)参数更新本身足以破坏 CTC 精度**(epoch-2
     acc=0.0,`checkpoint params without LAB: acc=0.671875`);
  2. **BN running stats 漂移**:如 `ctc_encoder.conv1.norm.running_var`
     最大变化 12742.09(neck BN,次要放大因素);
  3. **guide detach 组合**:backbone 因 guide detach 无法训练,训练集范围
     较宽时 observer scale 扩张。

**解决**:keep-BN 方案(融合单分支 + 实测统计初始化 BN,见问题 2),exp8b
训练稳定,val_acc 最高 0.5450。

**状态**:已解决。注意 LAB 敏感性与 neck BN 漂移在 keep-BN 方案下仍需关注
(见问题 4)。

## 问题 2:加入 BN 后的初始化与 float 对齐(当前方案已对齐)

**历史尝试**:

| 方案 | 初始化方式 | 结果 |
| --- | --- | --- |
| identity BN 冷启动 | γ=1/β=0/μ=0/σ²=1 | val_acc=0;需预热(200 步),训练语义改变 |
| keep-BN 参考分支 | deepcopy 某分支 BN | val_acc=0;running stats 与融合输出分布错位 |
| 方案 3 分支合并统计 | mean=Σsᵢμᵢ+Σbᵢ、var=Σsᵢ²σᵢ² | 收敛慢;var 忽略分支间协方差(近似) |
| **方案 2 实测统计(当前)** | **训练集实测 conv_fused 输出:γ=√(var+ε)、β=mean、running stats=实测** | **成功** |

**方案 2 如何保证与原始 float 一致(两段式)**:
1. **融合对齐**:标准融合(各分支 BN 吸收进 conv 权重),
   `conv_fused(x) ≡ 原始多分支输出`——实测 max diff 8.5e-6(CPU,eval);
2. **BN 恒等对齐**:γ=√(var+ε)、β=mean 使
   `BN(conv_fused(x)) ≡ conv_fused(x)`——插入 BN 后前向与原始 float
   精确一致(单层 BN 恒等检查 ~7.6e-6;GPU 上 0.0034 与 full-rep 基线
   0.0031 同量级,为 device 固有浮点差异,非 BN 引入)。

**局限**:恒等只在统计初始化的瞬间成立;训练中 γ 可学习漂移后恒等被破坏
(见问题 3)。

## 问题 3:训练后 γ/var 失配,converted 精度下降(未根治)

**现象**:
- exp8b(训练时 best val_acc=0.5450)converted 评估 acc=0.2638,差距 28pp;
- exp8 旧版(momentum=0.1)converted 评估 acc=0.0,prepared fake-off 输出 NaN。

**根因**:
- 统计 pass 初始化 γ=√(var+ε) 恒等;训练中 **γ 可学习漂移**,而极小 var
  通道(1e-11,源自 Paddle 权重)的 running_var 更新滞后(默认 momentum=0.1
  EMA 不敏感)→ `γ/√(var+ε)` 放大(最大 2098);
- fake-off(eval running stats)时 `γ(x-μ)/√(var+ε)` 放大爆炸 → NaN(exp8);
- 50 epoch 后 `|γ/√(var+ε)|>10` 通道仍有 97 个(6624 通道中)。

**已尝试的缓解**:
- `bn_training_momentum=0.9`(exp8b):消除 NaN、converted acc 0→0.26,
  但 γ/var 失配未根治(97 通道仍 >10,与 exp8 的 87 相当)。

**候选修复(未实施)**:
1. **γ 派生自 running_var**(γ=√(var+ε) 恒等保持,不自由学习);
2. **var 下限**(统计 pass 对极小 var 设下限,如 1e-4~1e-3);
3. **求和后 BN 全程存在**(QARepVGG/YOLOv6 训练结构,BN 从训练起自然统计,
   γ/var 同步演化——见 `docs/architecture/qarepvgg_quantization_reference.md`
   §8.4/8.6)。

## 问题 4:融合权重范围异常大(影响量化,未修复)

**现象**:`blocks2.0.pw_conv` 融合权重 absmax=997(全模型最大),eager 模型与
ONNX 数值一致(非导出 bug)。

**根因**:4 个 1x1 分支 BN scale 高达 ~1400(`γ/√(var+ε)`,部分通道
running_var~1e-13);conv 权重本身 ~0.3,融合 `kernel=Σ sᵢ·Wᵢ` 放大。
与问题 3 的极小 var 通道同源。

**影响**:S16 权重量化时该通道 scale 大、量化分辨率差;也是 eval 数值放大
的源头之一。

**候选修复(已记录,未实施)**:
- 标量缩放 W/k + bias/k,缩放系数乘到下一个 Mul(lab.scale)×k,输出精确
  不变;只缩放超阈值层(如 absmax>300),保守 target(如 100-300)。
  详见 `docs/axera_qat/plans/rep_keep_bn_plan.md` 末尾方案记录。

## 问题 5:结构一致性(根本性设计问题,未实施)

**背景**:QARepVGG(AAAI 2024)与 YOLOv6 的共同结论——QAT 应避免
"训练多分支 / 推理单分支"的结构不一致(训练/推理 QDQ 语义错位)。

**当前做法**:融合单分支 + keep-BN 训练(reparam=true)。
**QARepVGG 做法**:多分支 + 求和后 BN 训练,部署时融合(数学等价)。
**YOLOv6 做法**(RepOptimizer):直接训练单分支网络,从根上消除结构不一致。

**影响**:这是当前 converted 精度差距(问题 3)的深层原因之一。

**候选(未实施)**:
1. QARepVGG"求和后 BN 全程存在"训练结构改造(根治,改动大);
2. YOLOv6 RepOptimizer 单分支训练路线(长期,彻底一致)。

## 问题 6:LAB 参数敏感与 neck BN 漂移(Exp2 遗留,需关注)

**现象**(Exp2 审计):LAB 参数更新足以破坏 CTC 精度;neck BN running stats
漂移大(`ctc_encoder.conv1.norm.running_var` 变化 12742.09)。

**影响**:LAB 是 LearnableAffineBlock(scale/bias 可学习),其更新与 BN 方案
独立;keep-BN 方案下 backbone 的 LAB 仍在训练,需关注其对精度的敏感度。

**状态**:keep-BN 方案(exp8b)训练稳定,LAB 敏感性未再显式验证;建议后续
审计 LAB 参数在 keep-BN 训练中的漂移量。

## 问题 7:训练基础设施类问题(已修复)

| 问题 | 现象 | 修复 |
| --- | --- | --- |
| LSQ 训练"卡死" | epoch2 起主进程自旋、worker 空闲 47 分钟 | trainer.py `_any_observer_enabled` 按 `static_enabled` 判断,evaluate 不误启 observer(trainer.py) |
| converted 图丢属性 | full-rec 检测失败,`model(images)` 缺 gtc_targets | `convert_prepared_model` 保留 graph_role/model_type/output_names(bridge.py) |
| resume 合同缺失 | eval-only 时 `bn_statistics_steps` 未记录导致 mismatch | train.py metadata 始终记录 |
| BN weight 负值 NaN | 融合公式 clamp 破坏符号 | 保符号保护 `torch.where(abs<1e-12, ...)` |

## 汇总与建议顺序

| 优先级 | 问题 | 修复方向 | 影响 |
| --- | --- | --- | --- |
| 1 | 问题 3 γ/var 失配 | γ 派生自 running_var / var 下限 | converted 精度 +28pp |
| 2 | 问题 4 权重范围大 | 标量缩放 + lab.scale 补偿 | S16 量化分辨率 |
| 3 | 问题 6 LAB 敏感性 | 审计 LAB 漂移量 | 训练稳定性确认 |
| 中期 | 问题 5 结构一致性 | 求和后 BN 全程存在 | 根治 converted 差距 |
| 长期 | 问题 5 | YOLOv6 RepOptimizer 单分支 | 彻底消除结构不一致 |

## 问题 8:BN 折叠爆炸(Bn Folding Explosion)四层解决方案(2026-08-11)

### 现象与根因

极小 var 通道(1e-11)由于 `running_var` 滞后和 γ 漂移,折叠系数
`γ/√(var+ε)` 暴增至 2098,导致转换后精度狂掉 28pp(exp8b:prepared 0.545 →
converted 0.264)。这是工业界重参化模型 QAT 中最常见的"BN 折叠爆炸"。
纯靠"统计初始化瞬间的恒等"守不住,必须从**初始化下限**、**训练期解耦**、
**替代结构**三个维度优化。

### 关键实证:PyTorch BN momentum 语义

**PyTorch BN 的 momentum 与优化器语义相反**:

```
running = (1 - momentum) * running + momentum * batch
```

- `momentum=0.9` → running 90% 跟随 batch,10% 保留旧值——**极度震荡**,
  完全跟着当前 batch 的量化噪声跑,加剧漂移;
- `momentum=0.01` → running 1% 跟随 batch——平滑。

**结论:exp8b 使用的 `bn_training_momentum=0.9` 是反效果**,加剧统计漂移;
QAT 阶段 BN momentum 应使用极小值(0.01)或默认 0.1,且更优做法是冻结。

### 方案 1:防爆底线 —— Variance Floor(方差截断)【已选定实施】

**目标**:直接扼杀极小 var 通道分母趋零问题。极小 var 通道(如 1e-11)的 γ
初始化为 ≈√(1e-11+ε);一旦 QAT 训练中量化噪声产生哪怕 0.01 的梯度更新到 γ,
由于分母极小,倍数就会被放大上千倍。

```python
# 设定方差下限,通常为 1e-4 或 1e-5
MIN_VAR_CLIP = 1e-4

# 获取实测的 mean 和 var
real_mean = measured_mean
real_var  = measured_var

# 【核心修改】:对 var 进行下限截断
safe_var = torch.clamp(real_var, min=MIN_VAR_CLIP)

# 用 safe_var 初始化
running_mean = real_mean
running_var  = safe_var
gamma        = torch.sqrt(safe_var + eps)
beta         = real_mean
```

**效果**:强行将"死通道"的分母托底。即使 γ 后期发生漂移,其放大的极限
倍数也被死死压在安全范围内(不会超过几倍),直接消除 2000 多倍的折叠
系数爆炸。

**决策(2026-08-11)**:var 下限取 **1e-4**;BN 保持**默认热参数**
(不使用 exp8b 的 `bn_training_momentum=0.9`,恢复默认 0.1);暂不加入
正则化,先单变量验证方案 1。

### 方案 2:控制训练期漂移 —— BN Freezing(动态冻结策略)

**目标**:解决训练中 γ 和 running_var 更新不同步(解耦)的问题。

1. **修正 Momentum**:QAT 阶段 BN momentum 应设为极小值(如 0.01)甚至
   默认 0.1,绝不能是 0.9(PyTorch 语义下 0.9 = 90% 跟随 batch,震荡);
2. **实施 BN Freezing**:QAT 前 1~2 epoch 保持 BN train() 模式,让统计量
   适应带 FakeQuant 噪声的分布;第 3 epoch 起强制将所有插入的 BN 设
   eval() 模式:
   - `running_mean`/`running_var` 停止更新(完全冻结),锁死折叠公式分母;
   - γ 依然接受梯度更新,变成纯粹的**逐通道线性缩放参数**;分母不动,
     网络学习到的 γ 漂移会精准反映到最终折叠权重上,绝不滞后爆炸。

**注意**:PT2E prepared 图是 GraphModule,BN 为 call_function 节点,
`eval()` 需按节点改写 training 标志或冻结 running stats(禁 EMA 更新),
不能简单调用 `move_exported_model_to_eval` 整体切换。

### 方案 3:数学级约束 —— 增加 Scale 正则化 Loss

**目标**:从损失函数层面逼迫 `γ/√(var+ε)` 始终趋近于 1,维持恒等映射。

```python
def bn_folding_penalty(model, lambda_reg=1e-3):
    penalty = 0.0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and getattr(module, 'keep_bn', False):
            # 计算当前状态下的折叠系数
            scale = module.weight / torch.sqrt(module.running_var + module.eps)
            # 惩罚偏离 1.0 的程度
            penalty += torch.sum((scale - 1.0) ** 2)
    return lambda_reg * penalty

# 在训练循环中:
# loss = task_loss + bn_folding_penalty(model)
```

**效果**:通过 L2 惩罚,允许网络通过微调 γ 和 var 找回精度,但用无形的手
死死按住折叠系数,从根本上防止 28pp 的精度崩塌。与方案 1 正交,可叠加;
需引入超参 λ。**决策(2026-08-11):暂不加入,先单变量验证方案 1。**

### 方案 4:降维打击 —— 放弃 BN,改用 Channel-wise Affine 层

**目标**:彻底消灭 var 这个不可控变量。QAT 微调阶段输入分布相对稳定
(FP32 收敛模型),用纯 Affine 层替代 BN 层,无 EMA 统计滞后问题。

```python
class ChannelAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))  # 等价于 γ
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1)) # 等价于 β

    def forward(self, x):
        return x * self.weight + self.bias
```

- 完全不需要复杂的 2 步实测初始化——融合后的 `Conv_fused` 已与原网络
  等价,后面直接挂 γ=1/β=0 的 Affine 层,天然精确等价;
- 训练期优化器直接更新 Affine.weight;
- 折叠期无 var,公式绝对安全:
  ```
  W_converted = W_fused × Affine.weight
  B_converted = B_fused × Affine.weight + Affine.bias
  ```

**PT2E 链路障碍(已评估)**:PT2E 的 `_fuse_conv_bn_qat`/`_fold_conv_bn_qat`
只认 `aten.batch_norm`/`_native_batch_norm_legit` 节点;Mul+Add 的 Affine
**不会被 convert 折叠**,QuantONNX 会残留 Mul/Add 节点,可能触发现有
`validate_qdq_graph` 检查失败。若走此路需自行实现导出侧 Affine 折叠图变换
(违反"不在导出后任意补写 QDQ"合同)。方案 4 的思想等价于"γ 派生自
running_var + 冻结 var"(即方案 1+2 组合),无需换结构。

### 关于"参数少了,精度上不去"的补充:知识蒸馏

多分支结构(3x3 + 1x1 + Identity)反向传播时产生隐式梯度正则化效果,
单分支 `Conv_fused` 丢失了这种多维梯度路由,QAT 易掉入局部极小值。
为在单分支上恢复 QAT 精度,可依赖 Knowledge Distillation:

1. **Teacher**:加载未融合的、原始的、精度最高的 FP32 多分支模型,eval();
2. **Student**:融合后、插了 BN(或 Affine 层)、挂了 FakeQuant 的 QAT 模型;
3. **Loss**:真实标签 Loss + Teacher/Student 特征图(或 Logits)层面
   MSE/KL Loss——靠 Teacher 强力指引,弥补融合后缺失的梯度复杂度。

本项目已有 KD 蒸馏基础设施(提交 894a9d3,CTC logits 温度 KL),可复用。

### 实施计划(方案 1,已决策)

1. `initialize_kept_bn_statistics` 写回时:`safe_var = clamp(var, min=1e-4)`,
   `gamma = √(safe_var + eps)`(bridge.py);
2. **移除** `bn_training_momentum=0.9` 的 exp8b 配置(BN 恢复默认 momentum
   0.1),训练配置回退默认热参数;
3. 重训 exp8b(50 epoch),验证 converted acc(目标:prepared 0.545 →
   converted ≥ 0.45);
4. 若不足,叠加方案 2(冻结 running stats)或方案 3(正则化)。

### 方案 1 验证结果(2026-08-12,exp8c):失败,var floor 被训练 EMA 覆盖

**exp8c 配置**:var floor 1e-4 + BN 默认 momentum(移除 exp8b 的 0.9)。

**结果**:

| 指标 | exp8b(momentum 0.9) | exp8c(var floor + 默认 momentum) |
| --- | --- | --- |
| converted acc | 0.2638 | **0.0534(更差)** |
| bad 通道(|γ/√(var+ε)|>10) | 97 | 77 |
| max ratio | 2098 | 2144 |
| var<1e-4 通道(训练后) | - | **54** |

**根因**:var floor 只在统计 pass 初始化瞬间生效。训练中 BN 的 EMA
(默认 momentum 0.1)持续更新 running_var,极小 var 通道(1e-11)又被拉回,
54 个通道训练后 var 重新 <1e-4 → 折叠系数照常放大(2144)。floor 无法
抵抗训练期 EMA 覆盖。

**结论**:方案 1 单独不解决问题。必须叠加方案 2(冻结 running stats /
训练中禁止 EMA 更新 var,锁死折叠分母)才能真正遏制折叠爆炸。
exp8c 于 epoch24 终止,不再继续。

### 方案 2 实施记录(2026-08-12,exp8d):冻结 kept BN running stats

**实现**:
1. `bridge.py` 新增 `freeze_kept_bn_running_stats(prepared)`:遍历 prepared
   图,对 running_mean 属性名含 `_blocks` 的 batch_norm 节点(即 kept BN),
   把 momentum 参数(args[6])改为 0.0——PyTorch BN momentum=0.0 时
   running stats 完全不更新(momentum=None 是 cumulative average 会变,
   不能用);
2. `prepare_qat_model(freeze_kept_bn_stats=True)`:prepare 后 + move 后
   各应用一次冻结(move_exported_model_to_train 会重建图重置 args);
3. `trainer.py` `_set_model_mode` train 分支后重新应用冻结——每次 eval→
   train 切换 move 重建图会丢冻结,必须在 Trainer 级重 pin;
4. train.py `--freeze-bn-stats` CLI + profile `freeze_bn_stats` + metadata/
   resume 合同;checkpoint.py 恢复路径透传。

**验证**(2 epoch,含每 epoch eval 模式切换):
- running_mean 漂移恒为 **0.00**(冻结生效,不再被 eval/train 切换破坏);
- γ 变化 2.1e-3(仍可学习,逐通道线性缩放语义);
- 训练正常(acc 0.158-0.161),fake-off/converted 有限;
- 245 回归通过。

**关键发现**:eager 层设置 BN momentum 在 export/prepare 后丢失(图中为
常量);`move_exported_model_to_train`/`move_exported_model_to_eval` 每次
模式切换都会重建图并重置 args[6]=0.1——必须在 PT2E 图节点层冻结,且
Trainer 每次切回 train 时重 pin。

**exp8d 正式训练**:tmux `ppocrv5-rec-u16s16-exp8d-freeze`(GPU3,50 epoch,
LSQ + AdamW + warmup + keep-BN + var floor 1e-4 + freeze_bn_stats),
run 目录 `runs/exp8d_ppocrv5_mobile_rec_u16s16_adamw_lsq_warmup_keep_bn_freeze`。
训练完成后导出并验证 converted acc(目标:prepared 0.545 → converted ≥0.45)。

### 单层折叠验证(2026-08-12,blocks2.0.pw_conv)

**目的**:验证"训练时多分支独立 QDQ、部署时折叠单 conv"的精度损失(方案 C 可行性)。

**方法**(eager 非重参化模型,真实输入经 conv1+blocks2.0.dw_conv 提取):
- 参考:完整层浮点输出
- A:多分支 QDQ 模拟(各分支权重 per-channel S16 量化 + BN eval + 分支激活 per-tensor U16 独立量化 + Add + lab + act)
- B:折叠单 conv(分支 BN 吸收,`_get_kernel_bias`)+ 单激活 QDQ

**结果**:

| 对比 | MAE | max_abs | 余弦 |
| --- | ---: | ---: | ---: |
| 多分支 QDQ vs 浮点(量化本身误差) | 2.324 | 52.44 | - |
| **折叠 QDQ vs 多分支 QDQ(折叠引入)** | **0.00035** | **0.0032** | **1.000028** |

**结论**:
- 折叠本身引入误差极小(0.00035,比量化误差小 3 个数量级),余弦 1.0;
- 权重 per-channel 量化逐通道独立,折叠后重新 per-channel 量化误差不累积;
  激活量化:多分支各自量化求和 ≈ 单次量化(舍入误差有界);
- **支持方案 C 可行**:exp9 训练权重可直接折叠为推理重参化图,精度保持。

**exp9 QDQ 结构确认**(prepared 图 blocks2.0.pw_conv):
- 每个分支 `conv(per-channel weight Q) → BN → activation_post_process(独立 U16)` → Add;
- Add 输入独立量化域(205/205 不共享);
- 折叠后应共享单一激活量化点。

## 问题 9:exp9 checkpoint 权重是 LSQ 量化域耦合(2026-08-13)

**现象**:exp9 best.pt 剥离 `activation_post_process` 参数后直接当浮点权重,
eager 模型评估 acc=0.0;而 fake-quant-on(0.6153)/converted(0.6153)正常。

**根因**:
- checkpoint 保存 `conv.weight`(原始域,如 0.97),但图中实际生效的是
  `weight × BN/norm fold`(如 head conv1x1 折叠后 461.9)再经 per-channel
  fake quant **clip 到 32767** 后的值(如 16.05);
- clip 后数值与原始域差 29 倍,故 fake-quant-off/裸权重完全跑飞;
- LSQ 训练把权重训练到"必须过 fake quant 才正确"的量化域。

**影响**:
- 所有"剥离 quantization 参数当浮点权重"的用法(裸 eager、直接 rep 折叠、
  重新导出)全部失效(acc 0.0);
- exp10 用裸权重折叠微调,epoch1 仅 0.086(错误域起点),20 epoch 才 0.505。

**解决:量化域折叠(quantized-domain folding)**:
- 对每个分支权重取 `fq(fold(w))`(fake quant 输出,含 clip 语义)作为
  有效权重,分支求和得折叠单 conv;
- 实现 `/tmp/opencode/quantized_domain_fold2.py`(备份
  `cache/exp10a_fold_finetune_scripts/`);
- 折叠后 eager acc=0.535,预测与 exp9 fake-quant-on 完全对齐。

**衍生验证**:epoch1 时导出的折叠 ONNX=0.2778 是巧合(LSQ 仅训 1 epoch,
权重还接近浮点域);epoch50 重新导出=0.0(权重深度量化域化)。

## 问题 10:QAT JSON `output` 字段失效(2026-08-13)

**现象**:qspec 配置 `matmul_1/matmul_3` output U16,但 prepared 图 annotation
显示实际输出 S16;导出 QuantONNX 出现 5 个 requant Identity。

**根因**:`pytorchocr/quantization/ax_quantizer_lsq.py get_config` 硬编码
`output_dtype=input_dtype`(原 268 行),qspec 中所有 `output` 字段与
`output_is_symmetric` 从未被解析——输出永远继承输入 dtype/对称性。

**修复**(ax_quantizer_lsq.py):
- `QuantConf` 增加 `output_is_symmetric: bool = None`;
- `get_quantization_config` 输出 qscheme 用独立的 output 对称性
  (默认回退 is_symmetric,不破坏旧配置);
- `get_config` 解析可选 `output` 字段;无 `output` 时保持原行为。

**验证**(v2 qspec):
- matmul1(QK^T) S16、matmul2(softmax·V) **U16**、softmax S16、scale mul S16;
- QuantONNX Identity 从 5 → 1(仅 SE avg_pool u16→u16,用户确认不处理);
- QK^T/softmax·V 输入全 S16;proj 等 FC 层输入 U16 维持现状(用户确认)。

**注意**:quantizer 修改后,含 `output` 字段的 qspec 会产生与旧 checkpoint
不同的图(fake quant 编号/形状变化)——旧 checkpoint 无法 strict load,这是
预期行为(须从浮点权重重新 prepare)。

## 问题 11:qspec 节点名随图形态变化(2026-08-13)

**现象**:
- v3 qspec 的 mul 条目写 `mul_60/mul_61`,但 exp12a 非重参化训练图的
  scale Mul 是 `mul_352/mul_353`——条目未命中,scale Mul 退回 global U16,
  导出图出现 select→Mul requant identity;
- 折叠后训练图(exp12b)的 scale Mul 是 `mul_60/mul_61`,与 v3 一致。

**根因**:qspec `module_names` 匹配 **prepare_qat_pt2e 后**的图节点名,
而节点编号随 export 参数变化:
- dynamic batch / batch-aligned gtc targets(rec pretrained_train, batch>1);
- batch size 与 max_batch;
- 传给 prepare 的模型包装方式(传原始 model vs 已 export 的 GraphModule,
  嵌套 export 会偏移编号)。

**修复**:
1. **discovery skill**(`.codex/skills/ppocr-qat-config-discovery/`):
   改为从**原始 model**以与 train.py 相同的 dynamic-shape 合同 prepare,
   从 prepared 图按 nn_module_stack + op + 图中位置 remap 节点名;生成后
   校验所有 regional 名在 prepared 图命中,miss 即报错。
2. **v3 qspec**:mul 条目同时含训练图与折叠图两组名字
   `[mul_352, mul_353, mul_60, mul_61]`。
3. **fold_quantized_domain.py**(`ppocr-quantized-domain-fold` skill):
   重建 checkpoint 图时优先用训练时归档的 qspec 副本(metadata
   `copied_configs` 指向 run 目录),避免 qspec 文件后续修改导致 strict
   load 失败。

**验证**:修正后 skill `--check` PASS;exp12a/exp12b 训练正常;exp12b
折叠 finetune epoch1 0.594(训练 0.617,差值 <0.05)。

## 问题 12:Paddle 官方对 BN 与 QAT 的处理(2026-08-18 对照排查)

背景:为验证本项目 reparam QAT 的 keep-BN(问题 2/3)与结构一致性(问题 5)
决策,直接检查官方 PaddleOCR 实现与官方预训练权重,确认 Paddle 侧对 BN、
重参化与 QAT 的处理方式。代码路径均为 `references/PaddleOCR`(官方只读
checkout),行号以 2026-08-18 为准。

### 12.1 Paddle 预训练权重保留完整 BN(未融合进 Conv)

对三个官方 `.pdparams` 直接检查键名(pickle 读取,无需 Paddle 环境):

| 权重 | 总键数 | BN 相关键 | `bn.weight/bias/_mean/_variance` 组数 |
| --- | ---: | ---: | ---: |
| `PP-OCRv5_mobile_rec_pretrained.pdparams` | 969 | 556 | 151 |
| `PP-OCRv5_mobile_det_pretrained.pdparams` | 906 | 562 | 150 |
| `PP-OCRv6_small_rec_pretrained.pdparams` | 423 | 226 | 58 |

- 键名形如 `backbone.conv1.bn.weight/.bias/_mean/_variance`,且**每个分支都有
  BN**,包括 `LearnableRepLayer` 的 identity 分支
  (`backbone.blocks2.0.dw_conv.identity._mean`)与 `conv_kxk`/`conv_1x1` 分支;
- BN 数值是真实训练统计量:如 v5 rec `backbone.conv1.bn.weight` mean≈0.813、
  `_variance` mean≈0.150、identity 分支 `_variance` max≈3.22;
- 结论:官方发布的是**训练态权重**,BN 未融合;融合只发生在推理导出(12.2)。

### 12.2 Paddle 重参化:融合后纯 Conv,无 BN

v5 `LearnableRepLayer.rep()`(`ppocr/modeling/backbones/rec_lcnetv3.py:237`):
`_get_kernel_bias`(rec_lcnetv3.py:267)对 4×kxk、1x1(补零到 kxk)、identity
(单位核按 BN 缩放)统一做 BN 吸收后求和,再新建裸 `Conv2D` 并删除全部训练分支:

```text
t = γ / √(var + ε);  W' = W · t;  b' = β − μ · γ / √(var + ε)
kernel_reparam = Σ_kxk W' + pad(W'_1x1) + W'_identity
```

重参化后的 forward(rec_lcnetv3.py:215-219)只有
`reparam_conv -> LAB -> (Act)`,**没有任何 BN 节点**。v6 同样
(`rec_lcnetv4.py`):`Conv2D_BN.fuse()`(L206)返回裸 Conv;`ConvBNAct.rep()`
(L266)**`del self.bn`** 且 repped forward 跳过 BN(L259-260);
`StemBlock.rep()`(L331)、`RepDWConv.rep()`(L398)均只保留 `reparam_conv`。

官方导出对**非 QAT** 模型强制重参化
(`ppocr/utils/export_model.py:360-363`):遍历所有含 `rep()` 的子层执行——
官方推理模型 = 全融合、BN=0 部署图,与本仓库 QuantONNX 导出合同 BN=0 一致。

### 12.3 Paddle 官方 QAT:插入 fake quant,且导出跳过重参化

入口 `ppocr/utils/qat.py::apply_qat`(L441),两层:

1. **`quanter.quantize(model)`**(Paddle 3.0 内置
   `paddle.quantization.imperative.qat::ImperativeQuantAware`,quantize L236-294):
   把 `Conv2D/Linear/Conv2DTranspose` 替换为 `QuantizedConv2D`
   (`paddle.nn.quant.quant_layers`,forward L615),内部 `_fake_quant_weight`
   为 per-channel S8(channel_wise_abs_max)、`_fake_quant_input` 为 per-tensor
   激活 fake quant;`_quantize_outputs` 再包 `MAOutputScaleLayer` 统计输出
   out_scale(imperative/qat.py:448)。
2. **PaddleOCR 自研 U8 affine 扩展**(`qat.py::_apply_u8_qat`,L271-316):
   `U8AffineFakeQuant`(L20,moving min/max observer,STE 伪量化
   `(round_even(clip(x/scale+zp,0,255))−zp)·scale`);按 `_pattern_terminal`
   (L253)识别 `ConvBNLayer.bn`/`ConvBNAct.act`/`Conv2D_BN.bn`/`Head.conv_bn1`/
   `DilatedReparamBlock` 分支 BN 作为终点,`U8AffineOutputQuantWrapper`(L141)
   把输出 fake quant 挂在 BN/Act 之后;`_share_u8_activation_domains`(L319)
   用 hook 追踪 eval/train 两次 forward,量化 Conv 直接消费 QAT 输出域时把输入
   fake quant 替换为 `nn.Identity()` 共享上游域;`freeze_qat_observers`(L432)
   训练后关 observer。

**关键**:`ppocr/utils/export_model.py:378-384` 导出时
`skip_reparameterization=quanter is not None`——**启用 QAT 就跳过 `layer.rep()`**,
多分支训练图原样量化部署(`quanter.save_quantized_model`,export_model.py:411),
`tools/export_qat_onnx.py` 也是未重参化流程(strip_fake_quant 收集 qparams →
float ONNX → 重建 QDQ)。**Paddle 官方 QAT 不融合卷积权重,不存在"QAT 后
重参化"步骤**;官方也没有 keep-BN(insert_identity_bn 是本仓库自研扩展)。

### 12.4 与本仓库问题的对照

| 本仓库问题 | Paddle 官方做法 | 含义 |
| --- | --- | --- |
| 问题 2/3:keep-BN 训练中 γ/var 漂移、导出折叠爆炸 | 官方 QAT 不 rep,多分支 BN 全程存在、自然统计,无"插入恒等 BN"设计 | keep-BN 路线是本仓库特有,官方路线天然避免 γ/var 失配 |
| 问题 4:融合权重范围大(absmax 997,极小 var 通道) | float rep 同源(`γ/√(var+ε)` 放大),但 QAT 不 rep 就绕开了该问题 | 多分支量化图部署不受融合权重范围影响 |
| 问题 5:结构一致性 | Paddle 官方 = 多分支训练图直接量化部署(不折叠);YOLOv6 = RepOptimizer 单分支训练 | 与本仓库"非重参化训练 + 量化域折叠"(exp9→exp14b)同为三种落地之一 |
| Concat 共享量化域 | `_share_u8_activation_domains` hook 追踪,输入 fake quant 换 `nn.Identity()` | 与 `SharedQuantizationSpec`(bridge.py)同目标、不同实现 |

### 12.5 结论

1. Paddle 训练态权重 BN 完整;推理模型 BN=0(rep 吸收),与本仓库 QuantONNX
   导出合同 BN=0 一致,本项目 `rep()` 与 Paddle 逐行对应;
2. Paddle 官方 QAT 不做重参化——多分支量化图直接部署(部署效率低但无折叠
   误差);若需"QAT 后折叠",Paddle 无官方工具,须自行合并多分支量化域,
   即本仓库 exp9→exp14b 与 `ppocr-quantized-domain-fold` skill 解决的
   问题;
3. 本仓库 insert_identity_bn(问题 2/3 的方案 2)是 Paddle 没有的自研设计,
   exp7/exp8 已证伪放弃;当前正式两条路线(reparam 无 BN / 非重参化+折叠)
   分别对应 Paddle 的 float rep 部署形态与多分支 QAT 部署形态;
4. 详细机制与对照表见 `docs/architecture/ppocrv5v6_architecture_optimizations.md`
   第 5 节(BN 与 QAT:官方权重、重参化与量化路径)。

## 问题 13:YOLOv6 的 QARepVGG 路线(2026-08-18 对照排查)

背景:核实"YOLOv6 会训练带 BN 的非重参化网络,训练稳定后再重参化、再 QAT"的
说法。结论:**该说法成立,对应 YOLOv6 v0.3.0+ 的 QARepVGG 路线**
(`configs/qarepvgg/`,`training_mode='qarepvggv2'`),与 v0.2.0 `tools/qat/README.md`
主推的 RepOpt 单分支路线是**并存的两条路线**,不要混为一谈。代码路径均为
`references/YOLOv6`,行号以 2026-08-18 为准。

### 13.1 block 选择:`get_block(training_mode)`(yolov6/layers/common.py:721)

| training_mode | block | 训练结构 | 重参化后 |
| --- | --- | --- | --- |
| `repvgg` | `RepVGGBlock`(common.py:197) | 3x3 Conv-BN + 1x1 Conv-BN + identity BN(分支全带 BN) | 单 Conv,**无 BN**(`switch_to_deploy`,common.py:302) |
| `qarepvgg` | `QARepVGGBlock`(common.py:322) | 3x3 Conv-BN + 1x1 裸 Conv + identity(**分支无 BN**)+ **求和后 BN** | 单 Conv + **保留 BN**(common.py:373-390) |
| `qarepvggv2` | `QARepVGGBlockV2`(common.py:396) | 同上 + avgpool 分支 | 同上(`configs/qarepvgg/yolov6s_qa.py:67` 使用) |
| `repopt` | `RealVGGBlock` | 单分支 | 无需重参化(`configs/repopt/yolov6s_opt_qat.py:113`) |
| `hyper_search` | `LinearAddBlock` | RepOptimizer 超参搜索 | — |

### 13.2 QARepVGG 流程(与"先训练后重参化再 QAT"逐条对应)

1. **训练多分支 + 求和后 BN**(QARepVGGBlock forward,common.py:337-345;
   V2 加 avgpool 分支,common.py:413 起):

   ```text
   y = nonlinearity(bn(se(dense_3x3 + conv_1x1 + identity(+ avg))))   # V2
   ```

   只有 3x3 分支带 BN(`rbr_dense` = ConvModule),1x1 是裸 Conv、identity 是
   `nn.Identity()`——**BN 统一移到求和之后**,训练期 BN 统计与 γ 同步演化;
2. **训练稳定后重参化,且保留 BN**(`switch_to_deploy`,common.py:373-390):

   ```python
   # keep post bn for QAT
   # if hasattr(self, 'bn'):
   #     self.__delattr__('bn')
   self.deploy = True
   ```

   分支折叠为 `rbr_reparam` 单 Conv 后 **post BN 显式保留**(del bn 被注释),
   部署 forward = `nonlinearity(bn(se(rbr_reparam(x))))`(common.py:338-339);
3. **在"单 Conv + BN"结构上做 QAT**(pytorch_quantization 量化层),训练/推理
   量化域一致;最终部署时剩余 BN 由 `_fuse_extra_bn_tensor`(common.py:362)
   或导出 `--fuse-bn` 吸收;
4. 收益:YOLOv6-S-qa 浮点 44.7 → INT8 PTQ 44.0(-0.7),普通版 45.0 → 41.3
   (-3.7)(configs/qarepvgg/README.md 表格)。

### 13.3 与本仓库问题的对照

- QARepVGG 的"**求和后 BN 全程存在**"正是问题 2 记的候选修复 3
  (QARepVGG/YOLOv6 训练结构),也是 `docs/architecture/qarepvgg_quantization_reference.md`
  §8.4 的实现来源;
- 与本仓库 exp7/exp8 的 `insert_identity_bn`(折叠后再插 BN)的本质区别:
  QARepVGG 的 BN **从训练起就存在**,统计与 γ 同步演化,不存在问题 3 的
  γ/var 失配;insert_identity_bn 是"先折叠、后插 BN",Paddle 权重极小
  var 通道(1e-13)使 BN 即使实测统计初始化,训练中 γ 漂移后
  `γ/√(var+ε)` 仍爆炸(2000+)——这是 exp8 系列失败的根源,也是 QARepVGG
  结构的关键优势;
- 三种"结构一致性"落地(问题 5 的延伸):Paddle 官方 = 多分支量化图直接部署;
  YOLOv6 v0.2.0 = RepOpt 单分支训练(v0.2.0 主推,见问题 12 上下文);
  **YOLOv6 v0.3.0 QARepVGG = 多分支+求和后 BN 训练 → 折叠保留 BN → QAT**;
  本仓库 = reparam 无 BN(exp13/15/16)或非重参化+量化域折叠(exp14b)。

### 13.4 结论

1. "YOLOv6 训练带 BN 的非重参化网络,稳定后重参化再 QAT"的说法成立,是
   v0.3.0 QARepVGG 路线:训练图 = 多分支+求和后 BN,重参化后**保留 BN** 再 QAT;
2. v0.2.0 官方 QAT 主推的是 RepOpt 单分支路线(无多分支 BN 训练),两条路线
   并存,引用 YOLOv6 做法时须指明版本;
3. QARepVGG 的"求和后 BN"是解决问题 3 的正确结构方向,但需**从零按该结构
   训练**(BN 统计与 γ 同步),不能拿 pretrained 权重直接插 BN(exp8 已证伪);
4. 若本项目要复刻 QARepVGG 路线,训练图需重建为"多分支 + 求和后 BN",
   属于模型结构合同变更,须重新走浮点训练验证与 smoke 门禁。

## 问题 14:YOLOv6 v0.2.0 QAT 详解与本项目 LSQ 起点确认(2026-08-18)

背景:补记 YOLOv6 v0.2.0(`tools/qat/README.md` 主推路线)的完整 QAT 机制,
并应要求核对本项目当前 LSQ 训练的真实起点。代码路径均为 `references/YOLOv6`
与本仓库 `tools/`、`pytorchocr/`,行号以 2026-08-18 为准。

### 14.1 v0.2.0 QAT 四阶段

**阶段 0:RepOptimizer 从零训练单分支网络(结构前提)**

- 超参搜索:训练 CSLA 块 `LinearAddBlock`(common.py:521,3x3/1x1/identity
  三分支各挂 per-channel `ScaleLayer`),得到每通道 scale;
- `extract_scales`(RepOptimizer.py:18)提取 scale;
- `RepVGGOptimizer`(RepOptimizer.py:83,SGD 子类):
  - `reinitialize`(L117):把 scale 折叠进单分支权重
    `W = W₃ₓ₃·s_conv + pad(W₁ₓ₁)·s_1x1 + identity·s_id`;
  - `generate_gradient_masks`(L136)+ `step`(L159):训练时梯度逐元素乘 mask
    (3x3 区域 ×s_conv²、中心 ×s_1x1²、对角 ×1)——**梯度重参化**,单分支 SGD
    数学等价多分支 SGD;
- 目标结构 `RealVGGBlock`(common.py:480):`Conv(bias=False) → BN → ReLU`
  ——**单分支且带 BN,BN 从训练起存在、统计自然演化**,训练图=部署图。

**阶段 1:PTQ 校准(量化参数起点)**

- `qat_init_model_manu`(qat_utils.py:61):`nn.Conv2d → QuantConv2d`、
  `ConvTranspose2d → QuantConvTranspose2d`、`MaxPool2d → QuantMaxPool2d`
  (pytorch_quantization 模块,权重 per-channel S8、激活 per-tensor S8);
- `ptq_calibrate`(qat_utils.py:53):`collect_stats`(L12,disable_quant +
  enable_calib 跑 **4 个 batch** 收集直方图)→ `compute_amax`(L39,
  **entropy/percentile** 定 amax);engine.py `calibrate`(L560-577)保存
  calib checkpoint;
- **amax 固定**(`learned_amax=False`),QAT 只训练权重。

**阶段 2:QAT 训练(10 epoch + 通道蒸馏)**

- `quant_setup`(engine.py:579-594)加载 calib checkpoint;
- **10 epoch**、**关闭 warmup**(engine.py:275)、8 卡 DDP batch 128;
- channel-wise 蒸馏(loss_distill.py:191-201):teacher = 浮点 RepOpt 模型,
  `d_loss_cw`(特征通道蒸馏)+ `d_loss_cls/dfl`(输出蒸馏),权重按
  `(1−cos(epoch·π/max_epoch))/2` 余弦退火;
- 结果:S 由 PTQ 41.2 → QAT 43.0,TRT INT8 43.3(浮点 43.4,-0.1)。

**阶段 3:导出与部署**

- `qat_export.py`:`TensorQuantizer.use_fb_fake_quant=True` →
  `torch.onnx.export`(opset 13)导出标准 QDQ ONNX;`--graph-opt` 含
  **Concat amax 融合**(onnx_utils.py:21-31);`get_remove_qdq_onnx_and_cache`
  (onnx_utils.py:280)删 QDQ + 生成 TRT INT8 calibration cache;
- 变体:`tools/partial_quantization/`(灵敏度分析后敏感层保持浮点,INT8 42.1)。

### 14.2 可借鉴性分析

| # | v0.2.0 做法 | 本项目可借鉴点 |
| --- | --- | --- |
| 1 | BN 从训练起存在(单分支 Conv+BN,QAT 时 BN 浮点、导出 fuse) | 对照 exp8 失败根因:插入式 BN 必然 γ/var 失配;YOLOv6 证实正确姿势是"BN 全程存在"。exp14b 量化域折叠 finetune 已是等价方向 |
| 2 | PTQ 校准作为 QAT 起点 | 本项目 LSQ 已有 PTQ 式统计初始化(见 14.3),可加强为多 batch/直方图 |
| 3 | QAT + 特征蒸馏(10 epoch 恢复 ~0.5%) | 项目 KD 基建已就绪(`--kd`,CTC logits KL + 中间层);exp4 KD 失败被 U16 observer eps 截断掩盖(已修),在 exp16 路线上重试 KD 是直接可做的实验 |
| 4 | 敏感层保持浮点(partial quantization) | 本项目下采样链 S16(exp15/16)是等价思路的更优实现,已验证 |
| 5 | QAT 短周期 + 关 warmup | 本项目 profile 已关 warmup;若 scale 初始化加强可缩短 50 epoch |

不能照搬:RepOpt 从零训练(本项目是 pretrained 权重量化域微调,成本不可行);
pytorch_quantization/TensorRT 生态(S8 对称 + 固定 amax,本项目是 Axera +
PT2E + LSQ 可学习 scale);固定 amax 只训权重(本项目 PTQ 对比已证明 LSQ
可学习 scale 必需:U8/S8 域 PTQ 0.24 → QAT 0.59,见训练记录 §43.3)。

### 14.3 本项目 LSQ 起点确认(代码核实)

**结论**:用户"现在的 LSQ 是基于 PTQ 后的模型进行微调"的说法**基本准确**;
精确表述为:**量化参数 PTQ 式统计初始化 + LSQ 可学习 scale 微调**,权重本身
仍是 pretrained 浮点权重(不是 PTQ 反量化后的权重)。与 YOLOv6"固定 amax
微调"不同,本项目 scale 全程可学习。

代码链路(tools/train.py):

```text
L738-744:  prepare 后,if lsq and not eval_only: _prepare_lsq_training(...)
L315-351:  _prepare_lsq_training:
   1. initialize_weight_observers(model)    # 权重 per-channel scale 静态统计
      (bridge.py:694,从权重张量直接计算,不跑数据)
   2. model.apply(enable_learn)             # bridge.py:552,enable_param_learning,
      scale 可学习、zero_point 冻结
   3. 首个 loader batch 一次统计 forward(fake quant OFF / observer ON)
      初始化激活 scale → 之后 fake quant ON / observer OFF
L330-332:  resume 时只 enable_learn(scale 已由 checkpoint 携带,跳过统计)
```

trainer.py:112-131/256-319:**LSQ 训练期 observer 保持关闭**(scale 由 LSQ
学习,不靠 observer 更新);非 LSQ baseline 训练期 observer 开启。§43.3 的
独立 PTQ 评估(8 batch 激活统计)是 `/tmp/opencode/ptq_eval_rec.py`,不在
train.py 主链路。

与 v0.2.0 差异:

| 维度 | YOLOv6 v0.2.0 | 本项目 LSQ |
| --- | --- | --- |
| 量化参数起点 | 4 batch 直方图 + entropy amax | 权重静态统计 + 激活单 batch 统计 |
| 训练期 scale | 固定 amax(只训权重) | 可学习(zp 冻结,scale-only LSQ) |
| 训练长度 | 10 epoch + 特征蒸馏 | 50 epoch(可配 `--kd`) |

### 14.4 建议

1. 最小改动:exp16 路线上加 `--kd`(CTC logits + neck 特征,weight 先小如
   0.1-0.5),短周期(10-20 epoch)验证能否补上 0.5927 → 0.5936 浮点差距;
2. 次选:加强 LSQ scale 初始化(多 batch 激活统计或直方图,对齐 YOLOv6 的
   PTQ 起点),观察是否减少 epoch、提升 best;
3. 不推荐优先复刻"折叠保留 BN"结构(YOLOv6 的 BN 训练起就有,本项目
   pretrained 权重路线无法无损获得同样统计,exp8 已证伪插入式)。

## 问题 15:RepVGGOptimizer 可借鉴性分析与直方图/entropy 校准原理(2026-08-18)

背景:承接问题 14,评估"能否参考 RepVGGOptimizer,分支 scale 用非重参化
PTQ 校准获得",并详解 YOLOv6 的 4-batch 直方图 + entropy 校准原理及其与
多卡的关系。代码路径均为 `references/YOLOv6`,行号以 2026-08-18 为准。

### 15.1 RepVGGOptimizer 机制与"PTQ 获得分支 scale"的评估

**scale 的真实语义**:CSLA 块(`LinearAddBlock`,common.py:521)的 3x3/1x1/
identity 三分支各挂 per-channel `ScaleLayer`,与卷积**联合训练**的超参
(超参搜索 config `yolov6s_hs.py:59` `training_mode='hyper_search'`;
代码里 `is_csla=True` 冻结 scale 的分支未被使用)。它是**训练动力学参数,
不是量化参数**。

**mask 的数学**:对 CSLA 结构 `y = Σ_i s_i·(W_i*x)`,多分支 SGD 对等效单分支
`W_eff = Σ s_i·W_i` 的等效更新是 `ΔW_eff = η·(Σ_i s_i²)·g`(分支 i 梯度 =
s_i·g,再乘回 s_i)。因此单分支训练把梯度逐元素乘 `mask = Σ s_i²` 就能**精确
模拟多分支 SGD——该等式对任意固定的 per-channel s_i 都成立**。

**三种 s_i 来源评估**:

| 来源 | 语义 | 作为 mask 的合理性 |
| --- | --- | --- |
| RepOpt 超参搜索 | 与卷积联合训练出的动力学超参 | 正解(从零训练场景) |
| PTQ 校准(per-channel 权重大小/激活范围) | 量化参数,反映权重大小 | **无理论依据**:按"权重²"改变每通道有效学习率,大权重通道步子更大,可能有害 |
| 分支 BN 折叠因子 `s_i = γ_i/√(var_i+ε)` | 分支输出相对贡献(pretrained 权重直接可算,**无需任何校准**) | **唯一有依据的选择**:pretrained 多分支的等效单分支正是 `Σ BN折叠分支`,用它做 mask 可精确模拟 pretrained 多分支梯度流 |

**对本项目的结论**:

1. 机制可搬(mask 数学对任意固定 s_i 精确),但 **PTQ scale 不是正确来源**;
   要用就用分支 BN 折叠因子(从 `weights/ptocr_v5_mobile_rec_full.pth`
   的 BN 参数直接算);
2. 但本项目大概率不需要它:`rep()` 折叠是**精确等价**(非 RepOpt 式 scale
   近似初始化),QAT 微调直接训练部署图(exp13/15/16 已验证 0.59+);
   mask 的唯一用途是"单分支图模拟多分支梯度流",而多分支/单分支精度差距
   本质是**量化域语义**问题,exp14b 量化域折叠 + finetune 已在量化域层面
   对症解决(0.5936 级);
3. 风险:mask 只乘卷积权重梯度、不乘 LSQ 可学习 scale 梯度,二者可能失配;
   每通道有效学习率改变是 50 epoch 量化域微调中的未验证扰动;
4. 建议**不引入**;若坚持实验,正确做法是 `s_i = 分支 BN 折叠因子` 生成
   mask,在折叠单分支 QAT 上先 smoke 门禁——属训练合同变更,须从浮点权重
   重新 prepare。

### 15.2 YOLOv6 4-batch 直方图 + entropy 校准原理

**收集**(`collect_stats`,qat_utils.py:12):每个 `TensorQuantizer` 切到
`disable_quant + enable_calib`,前向 N 个 batch(config `calib_batches=4`,
batch 32 → 128 张图),逐层累积激活张量 |x| 的**直方图**(默认 2048 bins,
numpy/CPU 计算——代码注释 "a bit slow since we collect histograms on
CPU"),跨 batch 按 avg/max 策略合并。

**entropy(TensorRT 式 KL 校准)**:对每个候选阈值(直方图 bin 边界):

```text
参考分布 P = 直方图截断到 [0, threshold],尾部折入最后一个 bin,重新归一化
量化分布 Q = P 的 bin 重分为 128 个量化级(int8 对称,2⁷),每级取区间内概率
amax = argmin_threshold KL(P || Q) = argmin Σ p·log(p/q)
```

本质是在**截断误差**(阈值小)与**取整误差**(阈值大)之间找最优平衡点。
另有 `percentile`(直方图 CDF 的 99.99 分位)与 `max`(原始绝对值最大)方法;
`compute_amax` 后 amax 固定(`learned_amax=False`),QAT 只训练权重。

**与多卡无关**:

- 校准在 **main process 单进程**执行(engine.py:571-577
  `if self.main_process: ptq_calibrate(...)`),README 校准命令本身是
  单卡(`--batch 32 --workers 0`);
- 4 batch 只是**校准预算的配置选择**(128 图足够估分布),不是 DDP 需要;
  多卡唯一相关点是 `sync_observer`(PaddleOCR U8AffineFakeQuant 有该选项,
  YOLOv6 未用);
- **单卡完全可用**,机制不变。

**与本项目的关系**:本项目 LSQ 激活 scale 初始化是**单 batch min/max 统计**
(`_prepare_lsq_training` 第 3 步,tools/train.py:338-351)。YOLOv6 的多
batch 直方图初始化对离群点更鲁棒;若增强本项目起点,可把该步换成 N batch
直方图初始化——对 U8 affine/U16 域用 min/max 或 percentile 比 entropy
更合适(entropy 为对称 S8 设计)。LSQ 之后 scale 全程可学习,初始化主要
影响起点与收敛速度,不影响最终机制。

## 问题 16:LAB(Mul/Add)折叠进 BN 的数学评审与推荐路径(2026-08-18)

背景:用户提出把重参化 conv 前/后的 LAB(`Mul → Add`,标量 `s,b`,见
`LearnableAffineBlock`,rec_lcnetv3.py:64-75)折叠进插入的 BN(用 Mul/Add
的值初始化 BN 的 4 个变量),并扩展 hardswish 与 conv 之间的 Mul→Add
场景。本文从纯数学与 PT2E 链路两个角度评审,结论:**数学可行,但
"train() 真 BN"路线 = exp8 已实证否决;推荐"训练图零变化的直接吸收进
Conv"**。

### 16.1 数学:LAB ≡ 冻结统计 BN(eval 精确,通解族)

```text
LAB:  y = s·x + b(标量 s,b,全通道共享)
BN eval: y = γ·(x−μ)/K + β,K = √(σ²+ε)
匹配系数: γ/K = s;  β − γμ/K = b
→ 2 方程 4 未知数,2 参数解族:
   对任意 μ ∈ ℝ、σ² ≥ 0: γ = s·√(σ²+ε), β = b + s·μ
```

规范解(恒等统计):μ=0、σ²=1−ε → γ=s、β=b。**非唯一性**:σ² 取任意值都
同样精确——BN 4 变量只有 1 个有效自由度(比值 γ/√(σ²+ε) 与组合
β−γμ/√(σ²+ε));因此**初始化后若 σ²/μ 被更新而 γ 不同步,恒等立即破坏**。
s 的符号:γ = s·√(σ²+ε) 符号跟随 s,数学上合法(s<0 时 γ<0)。

### 16.2 train() 模式:0/1 初始化断崖判断正确,"平滑"成立但非"丝毫不差"

`F.batch_norm(training=True)` 用**当前 batch 采样统计** (x̄,v̂) 归一化:

```text
BN_train(x) = s·K·(x−x̄)/√(v̂+ε) + b + s·μ_real
            = s·x + b  ⟺  x̄=μ_real 且 v̂=σ²_real(一般仅近似)
```

单 batch per-channel 采样误差 ~σ/√N(batch 64 时均值估计误差 ~12% 量级),
起步是**近似平滑**(误差随 batch 增大收敛),不是"丝毫不差"。用真实统计
(而非 0/1)初始化避免 train() 起步按 1/std 缩放断崖——动机正确;实测统计
应多 batch/EMA(exp8 用 200 步 `bn_statistics_steps`)。

### 16.3 PT2E 链路行为(torch 2.6 源码核实)

- `prepare_qat_pt2e` 在 annotate 后执行 `_fuse_conv_bn_qat`
  (site-packages/torch/ao/quantization/quantize_pt2e.py:179),把 Conv→BN
  替换为近似融合子图(qat_utils.py:107-145):

  ```text
  conv(W·γ/√(var+ε), bias=0) → Div(γ/√(var+ε)) → Add(conv_bias)
  → F.batch_norm(training=True, γ, β, running stats, momentum)
  ```

  **BN 节点保留在 prepared 图中且 training=True**(momentum args[6] 可 pin,
  即 exp8d 的 `freeze_kept_bn_running_stats`);
- `convert_pt2e` 时 `_fold_conv_bn_qat`(quantize_pt2e.py:243)把 BN 吸收进
  Conv 权重 → 单 Conv,QuantONNX BN=0 合同保持。

### 16.4 与 exp8 系列对照:train()-mode 真 BN 是已实证否决的路线

用户公式是项目已有 `initialize_kept_bn_statistics`(bridge.py:781,
`s=1,b=0` 特例)的推广——基础设施齐全,只需把 γ/β 换成 `s·√(σ²+ε)`、
`b+s·μ`。但实证记录(问题 3/4/8):

- 问题 3:统计初始化恒等成立后,**训练中 γ 可学习漂移而 running_var EMA
  (momentum=0.1)滞后**,Paddle 权重极小 var 通道(1e-11)的
  `γ/√(var+ε)` 放大至 2098 → convert 折叠爆炸/fake-off NaN;
- 问题 4:折叠后融合权重 absmax 997(同源),劣化 S16 权重量化;
- exp8c(variance floor 1e-4):被训练 EMA 覆盖,无效;exp8b(momentum=0.9):
  只消除 NaN,97 通道仍 >10;exp8d(冻结 running stats):机械可行,但冻结
  统计 = 不再是"像真 BN 一样",且 best 仅 0.1006(epoch9 终止)。

**结论**:pretrained 权重 + 已有量化域场景下,"让 BN 像真 BN 一样吸收量化
噪声"的风险是实测过的;QARepVGG 的"真 BN"成立前提是**从零训练**(统计与
γ 同步演化),见问题 13。

### 16.5 拓展:hardswish → Mul → Add → Conv 的折叠

同一公式,只需把"BN 输入统计"换成 **hswish 输出统计** (μ_z, σ²_z):

```text
BN 输入 z = hswish(x);  γ = s·√(σ²_z + ε), β = b + s·μ_z
BN_eval(z) = s·(z−μ_z) + b + s·μ_z = s·z + b   # 精确
```

- **convert 后与"直接折入下一 conv"殊途同归**:

  ```text
  conv(W, BN(z)) = conv(W, s·z+b) → W' = s·W_next, b' = s·b_next + b·ΣW
  ```

  注意偏置项是 **b·ΣW**(输入侧常值注入:常值 b 经卷积核求和),与 conv 侧
  LAB 的 `b' = s·b_c + b` 不同;
- hswish 输出**非负**,统计比 conv 输出稳定,但 dead channel(var≈0)的
  γ/var 失配风险同类;
- **量化域合并(两种折叠都极简)**:折叠后只需**权重 per-channel scale ×s**
  (逐通道精确:`q(s·W, s·scale_w) = s·q(W, scale_w)` 位级一致);conv 输出
  QDQ 直接复用原 Add 输出域、输入 QDQ 复用 hswish 输出域(act 侧)或原
  Mul 输入域——**无 scale/zp 重算**;bias 浮点常量计算(b' 含 b·ΣW)。

### 16.6 推荐路径:直接吸收进 Conv(训练图零变化)

| 方案 | 训练图 | 部署图 | 风险 |
| --- | --- | --- | --- |
| train()-mode 真 BN(用户方案) | Conv→BN(统计初始化) | 单 Conv(convert 折叠) | γ/var 失配(exp8 实证),per-channel 自由度变更,checkpoint 不兼容 |
| **直接吸收进 Conv(推荐)** | **Conv→LAB 不变** | **单 Conv(导出期折叠)** | 仅需数学证明+测试;checkpoint 可复用 |

- 折叠位置:conv 后 LAB → 同 conv(`W'=s·W, b'=s·b_c+b`);act 后 LAB →
  下一 conv(`W'=s·W_next, b'=s·b_next+b·ΣW`);尾部 LAB 后接 CTC Linear
  同样吸收(`ΣW` 对 Linear 成立);首层 conv 前与无后继层的 LAB 保留;
- 训练图/checkpoint 合同不变——现有 exp16 checkpoint 可直接复用,无需
  从浮点权重重新 prepare;lab_lr_multiplier=0.1 合同保留;
- 收益:QuantONNX 全部 56 个 lab Mul/Add 消失;exp15/16 的 S16 标量 Mul
  三处特判(`swap_scalar_first_quantized_muls`、S16 dyadic 标量 qspec、
  TENG 对齐断言修复)可简化/删除;
- 门禁(合同 7"有数学证明、结构测试和精度回归覆盖的图变换"):
  1. 单层数值验证:折叠前后 eager/ORT 随机输入 max_abs≈0;
  2. 全图应用后 `ORT_DISABLE_ALL` 回归:argmax agreement 1.0 + max_abs
     浮点重排级;
  3. `tools/verify_qat_onnx.py` + QDQ 结构审计;
  4. ORT 全量 val 精度回归,确认无退化后再替换 exp16 导出产物。

### 16.7 结论

1. LAB ≡ BN 数学精确(eval),通解族非唯一;train() 模式只能近似平滑;
2. 插 BN(train 友好)路线与 exp8 同构,已实证否决,不建议;
3. hardswish/conv 间 Mul→Add 折叠进 BN 与折进下一 conv 终点相同,直接
   折进 Conv 更干净;
4. **推荐:导出期把标量 LAB 吸收进相邻 Conv(权重 scale×s + bias 常量),
   训练图零变化,量化域合并极简,收益明确(消除全部 lab Mul/Add 及
   Axera 标量 Mul 特判)**。

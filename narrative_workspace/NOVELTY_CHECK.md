# Novelty Check 结论（2026-07-19）

> 针对 NARRATIVE.md 主线做的定向查新。结论直接影响叙事，必须诚实记录。

## 一句话结论

**主线的前半"异常 = CLIP 特征场的高频残差 / 频率有助于 ZSAD"已被多篇顶会占据（FE-CLIP ICCV25、WMoE-CLIP、HarmoniAD），不能作为核心 novelty。**
**但后半"用一个编码器外部信号做逐图正常场重估（representation-level 证据选择），而非分数/特征级融合"这个机制位基本是空的——这必须成为新的叙事核心。**

---

## 一、频率轴的占位情况（已被占，需让位）

| 工作 | 载体 | 变换 | 是否训练 | 与我们主线前半的重合度 |
|---|---|---|---|---|
| **FE-CLIP** (ICCV 2025) | 注入 CLIP 视觉编码器**特征**（非输入图） | DCT | **是**（FFE + LFS 两个 adapter，需辅助数据微调） | 高：明确说"高频分量=细节=异常线索(soft borders)"，且强调注入的是 CLIP 特征而非原图频率 |
| **WMoE-CLIP** (2026) | CLIP 特征 | **Haar 小波**，`HF=LH+HL+HH` | 是（MoE prompt learning） | **极高**：和我们的 Haar 分解、HF 聚合公式几乎一模一样 |
| **HarmoniAD** (2026) | CLIP 编码器特征 | 频域自适应 cutoff 分高/低频 | 是（双分支） | 高：高频分支专门抓 small anomalies，低频分支抓结构 |

**含义：**
- "在 CLIP patch 特征上做 Haar DWT、把高频当异常证据"——**这一步不再新颖**，WMoE-CLIP 用的公式几乎相同。
- "频率有助于 ZSAD / CLIP 是低通、漏高频异常"——**这个 motivation 已是公认共识**，不能当卖点，只能当背景。
- 唯一残留的差异是"我们免训练、它们都要训 adapter/prompt"，但按用户要求 training-free 不作为核心卖点 → 所以**不能靠频率本身立论**。

## 二、逐图正常场重估轴的占位情况（基本空，可立论）

查到的相关 TTA / test-time 工作：

| 工作 | 做法 | 与我们后半的差别 |
|---|---|---|
| Selective TTA (2410) | neural implicit repr 上选择性适配，避免学到异常 | 不用 CLIP 文本原型；无频率外部信号 |
| PILOT (2508) | label-free TTA，用高置信伪标签更新可学习 prompt 参数 | 用**伪标签+可训练 prompt**，不是免训练；引导信号来自 CLIP 自己（自证闭环） |
| Dual-Image Enhanced CLIP (2405) | 线性 adapter + DRAEM 合成伪异常做 TTA | 需合成异常、需训练 adapter；无频率 |
| MRAD (ICLR 2026) | memory-driven 检索，动态 prompt | 检索式，非逐图重估；无频率 |

**含义：**
- **没有任何一篇**把"频率残差"当作**挑选可信正常 patch 的外部 oracle**，再去**逐图重估正常原型**。
- 现有 test-time 方法的引导信号要么来自 CLIP 自己（伪标签/熵，自证闭环），要么靠合成异常/检索——**没人用一个编码器正交的信号来约束重估**。这正是我们 §4 锁扣 B 的论点，且是空位。

## 三、叙事修正：核心必须从"频率"移到"融合位置 / 注入方式"

原主线把 novelty 压在"异常=高频残差"上 → 现在被 WMoE-CLIP 打穿。**修正后的核心 claim：**

> 频率线索对 ZSAD 有用已是共识（FE-CLIP/WMoE/HarmoniAD 均用**训练式 adapter/prompt 把频率融进 CLIP 特征或分数**）。**但我们发现：频率信号的"注入位置"才是决定成败的关键机制变量——把同一个频率可靠性信号用于分数级/特征级融合会灾难性地破坏 CLIP 的异常图（MVTec 91.8→80.0），而把它仅用作"挑选可信正常 patch 的 oracle"、去逐图重估正常原型，才是安全且有效的。** 换言之：频率是一个编码器正交的"可靠性 oracle"，它该指导 evidence selection，不该做 map/feature fusion。

**这个 claim 的独特性（三点都要在 related work 里打）：**
1. **注入位置 (fusion locus) 是新的机制维度**：现有频率工作全部是"特征级注入 (FE-CLIP/HarmoniAD) 或 prompt (WMoE)"，我们主张"证据选择级"，并用 direct-fusion 崩盘作为反证。没人做过这个对比。
2. **频率作为正交 oracle 约束逐图正常场重估**：把频率的角色从"被融合的特征"重定义为"约束 test-time 重估、破自证闭环的外部监督"。这是频率×TTA 的新接法。
3. **逐图正常原型重估**这一步本身在 CLIP-ZSAD 里就少见（多数 TTA 是更新 prompt 参数或检索）。

## 四、对现有资产的影响（好消息）

- direct wavelet fusion 崩盘 (80.0) 从"负控"升级为**核心论点的主证据**——它证明"注入位置"是关键机制变量。这块数据你已有。
- normal-stability 表证明"逐图重估不伤正常图"——支撑机制安全性。已有。
- 需要新增/加强的对照：**同一频率信号 × {分数级 / 特征级 / 证据选择级} 三种注入位置**的系统对比。这才是新叙事的核心消融，且直接回应 FE-CLIP/HarmoniAD 的"特征级注入"路线。

## 五、仍需做的查新（下一轮）

1. 精确确认 WMoE-CLIP 是否也做了"注入位置"对比（若做了，重合度进一步升高，需再调整）。
2. 查"fusion locus / where to inject prior for anomaly detection"这类明确讨论注入位置的工作。
3. 确认没有"frequency as selection signal for prototype/memory construction"的已有工作。

---

## 建议

叙事从"频率发现"彻底转向"**注入位置发现**"：
> 不是"我们用频率做 ZSAD"（已被占），而是"我们发现一个被所有频率方法忽略的机制事实——非语义先验该注入在证据选择层而非融合层，否则会破坏语义异常图；并据此设计了频率 oracle 引导的逐图正常场重估"。

小波仍是核心（它是那个正交 oracle），TTA 仍是核心（它是重估动作），但**卖点从"频率"变成"注入位置 + 正交 oracle 约束的重估"**——这个位置目前是空的。

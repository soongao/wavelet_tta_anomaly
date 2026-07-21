# Round 4 中文论文大纲：基于 narrative_workspace 主叙事

本文档以 `narrative_workspace/NARRATIVE.md` 为唯一主线：**CLIP 已经能看见异常相关的高频变化，但高频本身有歧义；缺的是每张图自己的正常参照**。因此论文卖点不是“频率有用”，也不是“prototype calibration”这个实现名，而是：

> **读取而非融合：从 CLIP patch 特征中读出已有高频线索，并用逐图正常参照解释它。**

对应英文工作标题：

> Reading, Not Fusing: Per-Image Normal Reference Estimation for CLIP's High-Frequency Anomaly Cues in Zero-Shot Anomaly Detection

## 0. 叙事边界

### 一句话 Thesis

CLIP patch 特征中的高频分量对异常敏感，但正常粗糙材质也会产生强高频响应。因此，高频响应大并不等于异常。本文认为 CLIP 缺的不是异常线索，而是一个**逐图正常参照**：只有知道这张图正常区域的高频水平，才能判断某处高频是否异常。

### 不能偏离的点

- 频率/小波本身不是 novelty；FE-CLIP、WMoE-CLIP、HarmoniAD 已经占住“频率有助于 ZSAD”。
- 本文不写成“把频率融入 CLIP”或“多组件组合”。本文写成“读出 CLIP 已有信号，并估计逐图正常参照”。
- `DirectHF` 崩盘是核心证据：同一个高频信号如果直接融合会破坏结果，说明需要参照。
- `SelfRef` 是强对照：只用 CLIP 自己挑正常区域会陷入自证闭环；高频作为 CLIP 语义之外的信号，用于挑选参照证据。
- `multi-crop`、`pixel-to-image`、工程提分组件只能放附录或作为 evaluation enhancement，不能进主叙事。
- 如果 `Ours` 与 `SelfRef` 总均值打平，不能声称数值上明确超过 CLIP-only；机制证据要靠 `DirectHF / GlobalRef / SelfRef / Ours` 的对照链、盲区召回和分类别增益。

### 变体命名

| 简称 | 含义 | 叙事作用 |
|---|---|---|
| `Baseline` | 原始 AnomalyCLIP，固定文本原型，无测试时参照 | 原始下界 |
| `DirectHF` | 读出高频后直接融合到 anomaly map，无参照 | 负控：证明“只读高频会坏” |
| `GlobalRef` | 用高频，但正常参照是全局固定水平 | 证明参照必须逐图 |
| `SelfRef` | 逐图估参照，但证据只来自 CLIP 语义置信 | 强对照：证明 CLIP 自证闭环不足 |
| `HFonly` | 逐图参照，高频用裸 HF | 设计消融 |
| `NoCons` | 逐图参照 + boundary-aware，但无保守更新 | 设计消融 |
| `Ours` | boundary-aware HF + 逐图正常参照 + conservative update | 完整方法 |

## 1. 摘要

### 本节要写什么

摘要写一段，按以下逻辑：

1. ZSAD 依赖 CLIP 的开放词表能力，但 CLIP 主要对齐全局语义，对局部微小/纹理异常不稳。
2. CLIP patch 特征其实已经包含异常相关高频变化；问题是正常粗糙材质同样高频，所以高频 cue 本身不可直接解释。
3. 本文提出 training-free 方法：在 CLIP patch feature 上读出 Haar 高频，用它作为 CLIP 语义之外的证据信号，逐图估计正常参照，并基于该参照重算 anomaly map。
4. 实验证据：DirectHF 无参照会崩；GlobalRef 不如逐图；Ours 能召回 CLIP 盲区，增益集中在纹理/微缺陷类。

### 摘要句式草稿

零样本异常检测借助 CLIP 的开放词表对齐能力，无需目标域训练即可检测缺陷。然而，异常常体现为局部高频扰动，而正常粗糙材质也可能具有强高频响应，使得从 CLIP patch 特征读出的高频 cue 本身具有歧义。本文提出一个观点：CLIP 已经包含异常相关高频信号，缺的不是信号，而是每张图自己的正常参照。我们因此提出一种 training-free 的逐图正常参照估计方法：用 Haar DWT 从冻结 CLIP patch 特征中读出高频，用该 CLIP-external cue 选择正常/异常证据，估计逐图参照，并据此重算异常图。实验表明，直接融合高频会显著破坏 pixel-level localization，而逐图参照能够稳定提升定位质量，并在 CLIP 语义盲区中召回纹理/微缺陷异常。

### 图表占位

摘要不放图。

## 2. 引言

### 本节要写什么

第一段写 ZSAD 与 CLIP 的价值：无需目标域训练、无需异常样本、适合工业与医学场景。说明 WinCLIP、AnomalyCLIP 等方法如何用正常/异常文本原型与 patch feature 计算异常分数。

第二段写现有 CLIP-ZSAD 的局限：CLIP 训练目标偏向“是什么物体”的全局语义，对“哪里局部不对”的高频微小变化不稳定。已有方法主要从语义、prompt、window 或 feature adaptation 角度补能力。

第三段切入本文观察：高频异常线索并非完全缺失；CLIP patch feature 里能读出异常相关高频。但高频不能直接用，因为正常纹理/边界也高频。给出大白话：同样的高频幅度，在光滑物体上可能是划痕，在织物上可能只是正常材质。

第四段提出 thesis：需要的不是固定阈值，也不是把频率直接融合进去，而是每张图自己的正常高频参照。异常应被定义为相对当前图像正常参照的偏离。

第五段列贡献：

1. 诊断：CLIP patch features already carry high-frequency anomaly cues, but raw HF is ambiguous.
2. 方法：read HF from frozen CLIP features, estimate per-image normal reference, recompute anomaly map, no training.
3. 机制证据：DirectHF 崩盘、GlobalRef 弱于 per-image ref、SelfRef 有 CLIP 自证闭环、Ours 召回 CLIP 盲区。
4. 实验：5 个数据集提升原始 AnomalyCLIP；增益集中在纹理/微缺陷类；正常图 FP 不增加。

### 图表占位

- Fig 1 Motivation：`narrative_workspace/figures/fig1_motivation.svg`
- Fig 2 Architecture：`narrative_workspace/figures/fig2_architecture.svg`

### Fig 1 图注草稿

CLIP patch feature 的高频响应既会在真实缺陷处升高，也会在正常粗糙材质中整体升高。因此，同样的高频幅度可能有相反含义，固定阈值或直接融合都会失败。本文估计每张图自己的正常高频参照，只标记相对该参照异常的区域。

## 3. 相关工作

### 3.1 CLIP-based Zero-Shot Anomaly Detection

写 WinCLIP、AnomalyCLIP、VCP-CLIP、AA-CLIP、AdaCLIP 等。重点不是逐个罗列，而是总结它们共同的处理方式：通过 prompt、window、视觉上下文或特征适配增强 CLIP 的异常语义表达。

本文定位：不是继续给 CLIP 加能力，而是证明 CLIP 中已有高频异常 cue，只是缺少解释这个 cue 的逐图正常参照。

### 3.2 Frequency Cues for Anomaly Detection

写 FE-CLIP、WMoE-CLIP、HarmoniAD 等。明确承认：频率有助于 ZSAD 已是已有工作共识，Haar/DCT/高频分支也不是本文 novelty。

本文区别：已有频率方法大多把频率作为 feature/prompt/score 级信息进行融合，通常需要训练；本文把频率重定义为**证据选择 oracle**，用于估计逐图正常参照。`DirectHF` 崩盘就是反证：融合位置错了，同一个频率 cue 会伤害定位。

### 3.3 Test-Time Adaptation and Reference-Based Detection

写 WinCLIP+、PILOT、Dual-Image、MRAD 等。强调它们要么需要正常参照图/检索/合成伪异常，要么用 CLIP 自己的置信度做伪标签。本文的关键不同是：用 CLIP 语义之外的高频 cue 在单张测试图上估计参照，无训练、无反传、无外部正常图。

### 图表占位

- Prior-work 对比可用 `narrative_workspace/NARRATIVE.md` 第 6 节的表格改写。
- 也可暂用 `paper/tables/prototype_prior_work.md`，但需要改列名，突出 `read/fuse` 和 `per-image normal reference`。

## 4. 方法

### 4.1 问题设定与初始语义分数

给定测试图像 \(x\)，冻结 CLIP/AnomalyCLIP 视觉编码器输出 patch features。对某一层，记空间 patch tokens 为

\[
F=\{f_i\}_{i=1}^{N}\in\mathbb{R}^{N\times C},\qquad N=HW,
\]

其中 \(f_i\) 已做 \(\ell_2\) 归一化。冻结文本分支提供正常/异常文本原型：

\[
t_n,t_a\in\mathbb{R}^{C},\qquad \|t_n\|_2=\|t_a\|_2=1.
\]

初始语义 logits 为

\[
z_n(i)=\frac{\langle f_i,t_n\rangle}{\tau},\qquad
z_a(i)=\frac{\langle f_i,t_a\rangle}{\tau},
\]

初始语义异常概率为

\[
S_0(i)=
\frac{\exp(z_a(i))}
{\exp(z_n(i))+\exp(z_a(i))}.
\tag{1}
\]

\(S_0\) 表示 CLIP 语义空间认为 patch \(i\) 异常的概率。本文认为 \(S_0\) 对局部高频异常不总是可靠，因此需要一个独立于语义分数的 cue 来辅助参照估计。

### 4.2 从 CLIP 特征中读取高频

将 patch tokens 重排为特征网格：

\[
\mathcal{F}\in\mathbb{R}^{C\times H\times W}.
\]

对每个非重叠 \(2\times2\) block，记四个特征向量为

\[
x_{00},x_{01},x_{10},x_{11}\in\mathbb{R}^{C}.
\]

Haar DWT 分量为

\[
LL=\frac{1}{2}(x_{00}+x_{01}+x_{10}+x_{11}),
\tag{2}
\]

\[
LH=\frac{1}{2}(x_{00}-x_{01}+x_{10}-x_{11}),
\tag{3}
\]

\[
HL=\frac{1}{2}(x_{00}+x_{01}-x_{10}-x_{11}),
\tag{4}
\]

\[
HH=\frac{1}{2}(x_{00}-x_{01}-x_{10}+x_{11}).
\tag{5}
\]

高频能量定义为

\[
HF=
\frac{1}{C}\sum_{c=1}^{C}
\left(
|LH_c|+|HL_c|+|HH_c|
\right).
\tag{6}
\]

将低分辨率 \(HF\) 上采样回 \(H\times W\)，并做逐图百分位归一化：

\[
\widehat{HF}
=
\operatorname{Norm}_{p_l,p_h}
\left(
\operatorname{Up}(HF)
\right).
\tag{7}
\]

这一节必须强调：Haar 不是本文 novelty，只是一个简单可解释的读出算子；它读取的是 CLIP feature 中已有的局部变化信息。

### 4.3 Boundary-Aware 高频可靠性

正常物体边界也会带来高频。因此从 \(LL\) 中估计结构边界：

\[
E_x(u,v)=\|LL(u,v)-LL(u,v-1)\|_2,
\]

\[
E_y(u,v)=\|LL(u,v)-LL(u-1,v)\|_2,
\]

\[
LF_{\mathrm{edge}}(u,v)=
\sqrt{E_x(u,v)^2+E_y(u,v)^2}.
\tag{8}
\]

上采样并归一化：

\[
\widehat{LF}_{\mathrm{edge}}
=
\operatorname{Norm}_{p_l,p_h}
\left(
\operatorname{Up}(LF_{\mathrm{edge}})
\right).
\tag{9}
\]

Boundary-aware high-frequency reliability 定义为

\[
W(i)=
\operatorname{Norm}_{p_l,p_h}
\left(
\widehat{HF}(i)
\cdot
\left(1-\widehat{LF}_{\mathrm{edge}}(i)\right)
\right).
\tag{10}
\]

解释：如果一个位置高频强，但低频结构边界也强，则它更可能是正常边界；如果高频强且不能被结构边界解释，则更可能是异常相关局部残差。

消融包括

\[
W_{\mathrm{HFonly}}(i)=\widehat{HF}(i),
\tag{11}
\]

以及不使用高频的 `SelfRef`：

\[
W_{\mathrm{none}}(i)=0.
\tag{12}
\]

### 4.4 用高频 oracle 选择参照证据

本文的关键不是把 \(W\) 加到最终 map，而是用它选择参照证据。先定义语义证据：

\[
e_a(i)=S_0(i)^\gamma,\qquad
e_n(i)=\left(1-S_0(i)\right)^\gamma.
\tag{13}
\]

用 \(W\) 对异常/正常证据做弱调制：

\[
r_a(i)=(1-\lambda)+\lambda W(i),
\tag{14}
\]

\[
r_n(i)=(1-\lambda)+\lambda(1-W(i)).
\tag{15}
\]

最终证据权重为

\[
\omega_a(i)=
S_0(i)^\gamma
\left((1-\lambda)+\lambda W(i)\right)^\eta,
\tag{16}
\]

\[
\omega_n(i)=
\left(1-S_0(i)\right)^\gamma
\left((1-\lambda)+\lambda(1-W(i))\right)^\eta.
\tag{17}
\]

直观解释：

- abnormal evidence：语义上像异常，且高频可靠性高。
- trustworthy-normal evidence：语义上像正常，且高频可靠性低。

这里 \(\lambda\) 是机制曲线的关键。\(\lambda=0\) 退化为 `SelfRef`，即只用 CLIP 自己估参照；\(\lambda\to1\) 接近高频主导，可能被正常纹理误导。正文要把这个写成“少量高频作为 oracle 最有效，过多高频会接近 DirectHF 的风险”。

### 4.5 逐图正常参照估计

设 top-k 比例为 \(\rho\)，选择数量

\[
K=\lceil \rho N\rceil.
\]

选择 abnormal evidence 和 normal evidence：

\[
\mathcal{I}_a=
\operatorname{TopK}_{K}
\left(
\{\omega_a(i)\}_{i=1}^{N}
\right),
\tag{18}
\]

\[
\mathcal{I}_n=
\operatorname{TopK}_{K}
\left(
\{\omega_n(i)\}_{i=1}^{N}
\right).
\tag{19}
\]

集合内归一化：

\[
\bar{\omega}_a(i)=
\frac{\omega_a(i)}
{\sum_{j\in\mathcal{I}_a}\omega_a(j)+\epsilon},
\qquad i\in\mathcal{I}_a,
\tag{20}
\]

\[
\bar{\omega}_n(i)=
\frac{\omega_n(i)}
{\sum_{j\in\mathcal{I}_n}\omega_n(j)+\epsilon},
\qquad i\in\mathcal{I}_n.
\tag{21}
\]

得到逐图视觉参照：

\[
v_a=
\operatorname{norm}
\left(
\sum_{i\in\mathcal{I}_a}
\bar{\omega}_a(i)f_i
\right),
\tag{22}
\]

\[
v_n=
\operatorname{norm}
\left(
\sum_{i\in\mathcal{I}_n}
\bar{\omega}_n(i)f_i
\right).
\tag{23}
\]

其中 \(v_n\) 就是本文叙事里的“这张图正常长什么样”的正常参照；\(v_a\) 是异常证据参照，但当前保守配置主要使用正常侧更新。

证据置信度为

\[
c_a=
\frac{1}{K}
\sum_{i\in\mathcal{I}_a}
\omega_a(i),
\qquad
c_n=
\frac{1}{K}
\sum_{i\in\mathcal{I}_n}
\omega_n(i).
\tag{24}
\]

### 4.6 保守参照更新与 anomaly map 重算

为避免异常证据噪声污染参照，使用保守更新门：

\[
m=
\mathbf{1}[c_a\ge\delta_a].
\tag{25}
\]

更新系数为

\[
\beta=
\operatorname{clip}
\left(
\beta_0 c_n m,\;0,\;1
\right),
\tag{26}
\]

\[
\alpha=
\operatorname{clip}
\left(
\alpha_0 c_a m,\;0,\;1
\right).
\tag{27}
\]

校准 normal prototype：

\[
\tilde{t}_n=
\operatorname{norm}
\left(
(1-\beta)t_n+\beta v_n
\right).
\tag{28}
\]

异常原型候选：

\[
t_a'=
\operatorname{norm}
\left(
(1-\alpha)t_a+\alpha v_a
\right).
\tag{29}
\]

异常侧使用更严格门：

\[
m_a=
\mathbf{1}
\left[
c_a\ge \max(\tau_a,\delta_a)
\right],
\tag{30}
\]

\[
\tilde{t}_a=m_a t_a'+(1-m_a)t_a.
\tag{31}
\]

当前稳定配置采用

\[
\alpha_0=0,\qquad \beta_0>0.
\tag{32}
\]

即主要更新正常参照，不激进更新异常原型。这和叙事一致：每张测试图中正常区域丰富，适合估计正常参照；异常区域稀少，容易被误选。

最后用校准后的原型重算 patch score：

\[
\tilde{z}_n(i)=
\frac{\langle f_i,\tilde{t}_n\rangle}{\tau},
\qquad
\tilde{z}_a(i)=
\frac{\langle f_i,\tilde{t}_a\rangle}{\tau},
\tag{33}
\]

\[
\tilde{S}(i)=
\frac{\exp(\tilde{z}_a(i))}
{\exp(\tilde{z}_n(i))+\exp(\tilde{z}_a(i))}.
\tag{34}
\]

多层融合：

\[
\tilde{S}_{\mathrm{fused}}(i)
=
\sum_{\ell\in\mathcal{L}}
\tilde{S}^{(\ell)}(i)
\quad
\text{or}
\quad
\frac{1}{|\mathcal{L}|}
\sum_{\ell\in\mathcal{L}}
\tilde{S}^{(\ell)}(i).
\tag{35}
\]

最终 anomaly map：

\[
M(x)=
G_\sigma
\left(
\operatorname{Up}
\left(
\tilde{S}_{\mathrm{fused}}
\right)
\right).
\tag{36}
\]

强调：最终 map 来自校准后原型重算，而不是 \(S_0+W\) 的直接融合。

### 4.7 DirectHF / GlobalRef / SelfRef 的公式定义

DirectHF 负控：

\[
M_{\mathrm{DirectHF}}(i)=
(1-\kappa)S_0(i)+\kappa W(i).
\tag{37}
\]

它没有逐图参照，预期会把正常高频纹理误报为异常。

SelfRef：

\[
\lambda=0,\qquad
\omega_a(i)=S_0(i)^\gamma,\qquad
\omega_n(i)=(1-S_0(i))^\gamma.
\tag{38}
\]

它有逐图参照，但参照来源只依赖 CLIP 自己，因此无法修正 CLIP 的高频盲区。

GlobalRef 可写为全局正常高频水平 \(R_g\)：

\[
R_g=
\operatorname{Statistic}
\left(
\{W_j(i)\}_{j\in\mathcal{D},\,i\in\Omega}
\right),
\tag{39}
\]

并用

\[
D_g(i)=W(i)-R_g
\tag{40}
\]

作为非逐图参照偏离。正文中不用展开实现细节，强调它代表“一个固定参照配所有材质”的对照。

### 方法图占位

- Fig 2 Architecture：`narrative_workspace/figures/fig2_architecture.svg`
- 可辅助参考旧 draw.io：`narrative_workspace/paper_iterations/round3_final/figures/image_1784538888_recreated/image_1784538888_recreated.png`，但它不能替代根叙事中的 Fig 2。

## 5. 实验

### 5.1 实验设置

写数据集、指标和不训练声明：

- Datasets：MVTec AD、VisA、MPDD、BTAD、DTD-Synthetic。
- Metrics：pixel AUROC、pixel AUPRO、image AUROC、image AP。
- Backbone：冻结 AnomalyCLIP/CLIP。
- Training-free：不使用目标域训练、不反传、不更新 CLIP 参数。
- 说明 `multi-crop` / `pixel-to-image` 的位置：可以作为统一后处理或附录增强，不写成机制核心。

### 5.2 主结果

使用 `narrative_workspace/result_charts/TABLES.md` 的 Table 1 作为占位。正文要写：

- Ours 在 5 个数据集上相对 AnomalyCLIP 全面提升。
- 头条指标是 pixel AUPRO，因为本文关注定位质量。
- MPDD/BTAD/DTD 若使用 dataset-tuned setting，必须另列 global setting 或标注 upper bound，避免调参质疑。

图表占位：

- Table 1 Main results：`narrative_workspace/result_charts/TABLES.md`

### 5.3 SOTA 对比

使用 `narrative_workspace/result_charts/TABLES.md` 的 Table 2 作为占位。外部数值标 `*`，正式稿前必须核对原论文。

写法要克制：本文不声称碾压所有训练型 SOTA；重点是 training-free、机制清楚、pixel AUPRO 有竞争力。

## 6. 分析与消融

### 6.1 正常参照从哪来

这是全文最重要实验。使用 Table 3 和 Fig 3。

图表占位：

- Table 3 Core mechanism ablation：`narrative_workspace/result_charts/TABLES.md`
- Fig 3 Mechanism ordering：`narrative_workspace/result_charts/fig_mechanism_ordering.png`
- Fig 3 SVG schematic：`narrative_workspace/figures/fig3_mechanism_ablation.svg`

必须写出的三条结论：

1. `DirectHF` 明显低于 `Ours`：读高频但不给参照会坏。
2. `GlobalRef` 明显低于 `Ours`：参照必须逐图。
3. `SelfRef` 是强对照，但仍存在 CLIP 自证闭环；Ours 的额外证据要靠盲区召回和分类别增益支撑。

### 6.2 增益来自哪些类别

使用 `fig_percategory_gain`。写：

- Ours vs SelfRef 的差别只在“挑参照是否用高频”。
- 增益集中在纹理/微缺陷类，物体类基本持平。
- 这说明本文机制针对的是 CLIP 的频率盲区，而非无差别刷分。

图表占位：

- `narrative_workspace/result_charts/fig_percategory_gain.png`

### 6.3 CLIP 盲区召回

使用 `fig_blindspot`。写：

- 定义 SelfRef 判为正常但 GT 异常的像素/区域为 CLIP 盲区。
- SelfRef 对自身盲区召回为 0 是定义；Ours 如果召回 18–22%，说明高频提供 CLIP 语义之外的信息。
- 这个实验比总均值更能证明机制，尤其当 Ours 与 SelfRef 均值接近时。

图表占位：

- `narrative_workspace/result_charts/fig_blindspot.png`

### 6.4 稳定性、运行时、敏感性

写：

- 正常图 FP 不高于 baseline，证明逐图估参照不会制造误报。
- 开销约 21–22%，主要来自 Haar DWT、top-k 证据选择和一次重算。
- `wavelet_mix` 曲线是机制曲线：`lambda=0` 是 SelfRef，少量高频最好，过大接近 DirectHF 风险。

图表占位：

- Table 5 Stability/Runtime：`narrative_workspace/result_charts/TABLES.md`
- `narrative_workspace/result_charts/fig_sensitivity.png`

### 6.5 定性结果

按 `narrative_workspace/result_charts/QUALITATIVE_SPEC.md` 占位。需要展示：

- Input / GT / Baseline / SelfRef / HF map / Ours。
- HF map 在正常粗糙材质上也亮，证明 raw HF ambiguous。
- Ours 能捞回 SelfRef 漏掉的纹理/微缺陷。

图表占位：

- `narrative_workspace/result_charts/QUALITATIVE_SPEC.md`

### 6.6 设计消融

用 Table 4。写：

- `HFonly` vs `Ours`：boundary-aware 抑制结构边界。
- `NoCons` vs `Ours`：保守更新提高正常图稳定性。
- 这部分是补充证据，不是主 claim。

## 7. 结论

### 本节要写什么

重申：

CLIP 并非完全缺少异常相关高频线索；它缺的是解释这些高频线索所需的逐图正常参照。本文读取冻结 CLIP patch feature 中已有高频信号，用它作为 CLIP 语义之外的证据选择 oracle，估计每张测试图自己的正常参照，并据此重算 anomaly map。实验证明，直接融合高频会崩，固定全局参照不足，而逐图参照能提升定位并召回 CLIP 盲区异常。

限制：

- 频率本身不是本文 novelty。
- 与 SelfRef 的总均值增益可能小，机制主要由坏对照、盲区召回和分类别分析支撑。
- 更强的异常侧参照更新仍需更可靠的置信估计。

## 8. 附录计划

### A. 公式细节

放百分位归一化、prompt ensemble、多层融合、GlobalRef 的具体实现。

### B. 工程增强

放 multi-crop、pixel-to-image、dataset-tuned vs global setting。

### C. 完整表格

放 5 数据集所有类别细分、所有消融行、global vs tuned。

### D. 查新与定位

放 FE-CLIP、WMoE-CLIP、HarmoniAD、PILOT、MRAD 等更详细比较，强调本文不是频率方法，而是 read-and-reference。

## 9. 当前图表资源索引

### Figures

- Motivation：`narrative_workspace/figures/fig1_motivation.svg`
- Architecture：`narrative_workspace/figures/fig2_architecture.svg`
- Mechanism ablation schematic：`narrative_workspace/figures/fig3_mechanism_ablation.svg`
- Mechanism ordering：`narrative_workspace/result_charts/fig_mechanism_ordering.png`
- Per-category gain：`narrative_workspace/result_charts/fig_percategory_gain.png`
- Blind-spot recall：`narrative_workspace/result_charts/fig_blindspot.png`
- Sensitivity：`narrative_workspace/result_charts/fig_sensitivity.png`

### Tables

- Main/SOTA/core/design/stability/global-vs-tuned 表骨架：`narrative_workspace/result_charts/TABLES.md`
- 实验目标与真实/预期区分：`narrative_workspace/EXPERIMENT_TARGETS.md`
- 目标数值版表格规划：`narrative_workspace/EXPERIMENT_PLAN_PAPER.md`

## 10. 写作检查清单

- 标题和摘要必须出现 `Reading, Not Fusing` 或等价表达。
- 第一页必须讲清楚：同样 HF 幅度在不同图上含义相反，所以缺逐图参照。
- 方法里 Haar DWT 只写作 read-out operator，不写成创新点。
- 主消融必须围绕 `DirectHF / GlobalRef / SelfRef / Ours`。
- 不把 prototype calibration 当论文卖点，只作为 per-image normal reference 的实现。
- 不把 multi-crop / pixel-to-image 写成核心机制。
- 所有 EXPECTED 数字正式成稿前必须替换为真实 log 或标清楚。
- 外部 SOTA 数字必须核对原论文。

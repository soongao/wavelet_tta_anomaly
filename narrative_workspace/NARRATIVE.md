# 叙事主线：CLIP 看得见高频异常，但缺一个"这张图正常长什么样"的参照

> 工作区文档。锁定论文的机制切入点、方法骨架、证据映射与 claim 边界。
> 只谈机制，不谈刷分。抬数值但无机制价值的组件（multi-crop / pixel-to-image）一律降级为附录。
> 本版本已根据 NOVELTY_CHECK.md 修正：频率本身不再是卖点（已被 FE-CLIP/WMoE-CLIP/HarmoniAD 占据），
> 切入点改为"CLIP 有高频信号但缺逐图正常参照"。

---

## 0. 一句话 Thesis（大白话）

CLIP 的 patch 特征里，高频部分对异常是敏感的——异常那一块的高频响应确实会变。**但高频响应大，不一定是异常**：布料、编织、网格这些正常材质本身高频就很高。

也就是说：**CLIP 已经能"看见"异常带来的高频变化，它只是分不清"这个高频是异常，还是这块材质本来就这样"。** 它缺的不是能力，是一个**参照**——这张图正常的地方高频大概多少。有了这个参照，才能判断"这里的高频明显超出这张图的正常水平 → 异常"。

这个正常参照**没法预先定死**（布料和金属的正常高频完全不是一个量级），零样本也没有训练数据能提前学。所以只能**在测试时，从这张图自己身上估出来**。

方法 = 回答两个大白话问题：
1. **异常线索在哪？** → 用小波把 CLIP 特征里的高频读出来（读取 CLIP 已有的信息，不新增东西）。
2. **多高才算异常？** → 从这张图自己估一个"正常高频水平"当参照，超出的才算异常（这就是 TTA 做的事）。

**核心 insight 一句话：CLIP 能看见高频异常，但分不清材质本身的高频；缺的是一个只能逐图现估的"正常参照"。**

---

## 1. 为什么这不是"组合两个组件"

- 切入点是一个**关于 CLIP 的观察**（"它看得见高频异常，但分不清材质本身的高频"），不是"我们挑了频率和 TTA 两个模块"。
- 这个观察一旦说出来，"读高频"和"估正常参照"就是**这个问题本来就需要的两步**，不是硬凑。
- 与三篇频率工作的区别很朴素：
  - 它们：**频率是个好特征，加进 CLIP（fuse in）** —— 需要训练 adapter/prompt。
  - 我们：**CLIP 本来就有高频信息，缺的是一个逐图正常参照（不是融合，是补参照）**。
- `direct wavelet fusion` 崩盘（MVTec 91.8→80.0）正好说明"直接把高频加进去是错的方向，得先有参照"。它从负控升级为**世界观之争的证据**（加特征 vs 补参照）。

**注意（用户明确要求）：** 卖点不落在"注入位置"这种工程点上，也不落在"两个机制必须耦合"这种空话上。卖点落在那句大白话 insight。

---

## 2. 方法骨架（= 论文 Method）

### 第 1 步：观察 —— CLIP 看得见高频异常，但分不清材质本身的高频
- CLIP 对比预训练优化的是"跨图一致的物体语义"（低频那一半）；高频部分对异常敏感，却没有稳定、可跨图比较的判断标准。
- 现象证据：高频响应图上，异常区域亮，**正常的粗糙材质区域也亮** → 说明"信号在、但不能用固定阈值跨图判读"。
- 背书：WinCLIP（CLIP 只在全局 embedding 对齐、失败案例是 tiny defect）、FE-CLIP/WMoE（频率对异常有用已是共识 → 只能当背景，不当卖点）。

### 第 2 步：读出高频 —— 把 CLIP 特征里的高频信息取出来
- 在 CLIP patch 特征网格上做一层 Haar DWT。
- `HF = mean_c(|LH|+|HL|+|HH|)`；可选 `W = HF · (1 − LF_edge)`（减掉物体自身结构边界，只留"无法被结构解释的突变"）。
- 定位：这一步只是**读取 CLIP 已经编码的信息**，不引入新信息。算子与 WMoE 重合不影响 novelty，因为 novelty 不在"怎么读"，在第 3 步的"参照"。

### 第 3 步：逐图估正常参照 —— 多高才算异常
- 高频里既有异常也有正常材质纹理，固定阈值分不开，因为"这张图的正常高频水平"是 instance-specific 的。
- 零样本无训练数据 → 在测试时，从这张图自己身上、用可信为正常的区域，估出这张图的"正常高频/正常原型"参照。
- 异常 = 高频信号相对这张图自身正常参照的偏离。
- 这就是 TTA，但它的身份是"**给 instance 相关的高频信号提供一个逐图参照**"，不是"又一个自适应技巧"。

### 第 4 步：为什么这两步都不能少（朴素版，不用"必须耦合"的空话）
- **没有第 3 步（只读高频）**：正常材质的高频被当成异常 → direct fusion 崩盘（见 §4 证据）。
- **没有第 2 步（只做 TTA、参照来自 CLIP 自己）**：挑"可信正常区域"的信号只来自 CLIP 语义置信，而 CLIP 恰恰对高频异常不敏感 → 挑出的"正常"里混进它看不见的异常，参照被污染。高频信息是 CLIP 语义置信之外的独立依据，才能把这些区域排除。

---

## 3. 标题内核（候选，去花哨）

> **Per-Image Normal Reference Estimation for CLIP's High-Frequency Anomaly Cues in Zero-Shot Anomaly Detection.**

关键词轴：high-frequency cues already in CLIP（读信号）+ per-image normal reference（逐图参照）。
避免使用 "calibrate the uncalibrated subspace / zero-point" 这类花哨表述。

---

## 4. 证据映射（已有 log → 支撑哪一步）

> 数值一律以真实 log 为准，不用 expected target 冒充。路径待逐一核对回填。

| 机制主张 | 对应实验 / 证据 | 现有日志（待核对） | 状态 |
|---|---|---|---|
| 只读高频、不给参照 → 会坏（第 4 步 A） | direct wavelet fusion，MVTec 91.8→80.0 | `cached_results/prototype_tuned/mvtec_direct_multicrop_p2i/log.txt` | 已有，需高精度确认 |
| 高频信息是 CLIP 语义之外的独立依据（第 4 步 B） | wavelet-in-evidence vs CLIP-only 对照 | `cached_results/prototype_tuned/mvtec_clip_only_multicrop_p2i/log.txt` | 已有，当前打平，需高精度重跑 |
| 逐图估参照不伤正常图 | normal-image stability 表 | `cached_results/prototype_tuned/validation/*_normal_stability.md` | 已有 |
| "有信号无参照"的存在性 | 高频响应图：异常亮、正常粗糙材质也亮 | `cached_results/prototype_tuned/mechanism_viz/*` | 已有，需针对性挑图 |
| 主结果超原始 AnomalyCLIP | MVTec / VisA full vs baseline | `ablation_results/20260622_094146_component/*/07_full_method/log.txt` | 已有 |

### 换叙事后需要新增/加强的实验（都是机制向）
1. **"参照必须逐图"的直接验证**：展示同一高频幅度在不同图/材质上对应正常 vs 异常 → 固定阈值必失败、逐图参照必需要。
2. **"参照来自 CLIP 自己 vs 来自高频"的对照**：用 CLIP 语义置信挑正常区域做 TTA vs 用高频信息挑 → 证明前者对高频异常盲（第 4 步 B 的直接证据）。
3. 高精度（2–3 位小数）重跑 full vs CLIP-only。
4. 与训练型 SOTA（WinCLIP/AnomalyCLIP/VCP-CLIP/AA-CLIP/FE-CLIP）横向对比主表（可引公开数值）。

---

## 5. Claim 边界（诚实红线）

**可以说：**
- CLIP 的高频分量对异常敏感，但正常材质纹理同样高频 → 需要逐图正常参照才能判读。
- 直接把高频融进异常图会破坏结果（direct fusion 崩盘）→ 支持"补参照，而非加特征"。
- 逐图估参照在不伤正常图 FP 的前提下改善检测。

**不能说（除非有新证据）：**
- 不得声称 full 在数值上明确优于 CLIP-only（当前一位小数打平）。
- 不得把 multi-crop / pixel-to-image 当核心机制（附录）。
- 不得声称"频率有助于 ZSAD"是本文发现（已是共识，FE-CLIP/WMoE 在先）。
- 不得声称推理时用了标签或更新了 CLIP/AnomalyCLIP 参数。
- 不得把卖点写成"注入位置"或"两个机制必须耦合"。

---

## 6. 与现有工作的定位

| 方法 | 对频率的态度 | 是否逐图估正常参照 |
|---|---|---|
| WinCLIP | 不用频率；多尺度窗口 | 否（靠加正常参照图 WinCLIP+） |
| AnomalyCLIP | 不用频率；文本去物体语义 | 否 |
| FE-CLIP / WMoE-CLIP / HarmoniAD | **训练式把频率融进 CLIP 特征/prompt** | 否 |
| PILOT / Dual-Image / MRAD | 不用频率；TTA 靠伪标签/合成/检索 | 否（更新 prompt 或检索，非逐图估正常参照） |
| **本文** | **读取 CLIP 已有的高频，不融合、不训练** | **是（逐图估正常参照）** |

空位：用"CLIP 已有的高频信息 + 逐图正常参照"来判异常，没人做过。

---

## 7. 下一步
1. 定向查新最危险撞车点：`training-free / test-time frequency for ZSAD`、`per-image normal reference for ZSAD`（近半年 arXiv）。
2. 若空 → 设计 §4 的两个机制验证实验（参照必须逐图 / 参照来源对照）。
3. 回填 §4 证据表真实日志与数值；高精度重跑 full vs CLIP-only。

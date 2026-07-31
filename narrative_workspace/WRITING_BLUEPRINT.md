# Writing Blueprint: 图像条件化正常参照

Date: 2026-07-31

本文件用于冻结论文写作入口。它不是结果 source-of-truth；数值仍以 `paper/tables/*.csv`、`FIVE_DATASET_RESULTS_AND_ABLATIONS.md` 和 `content_consistency_report.md` 为准。

## 0. 当前写作定位

- 论文类型：机制发现型方法论文，而不是刷 SOTA 型结果论文。
- 目标领域：CLIP-based zero-shot anomaly localization / industrial anomaly detection。
- 核心风险：不要把“频率用于 ZSAD”写成创新点；该方向已有 FE-CLIP、WMoE-CLIP、HarmoniAD 等相关工作占位。
- 当前最稳切入：指出 CLIP-ZSAD 中的“参照缺口”，再提出图像条件化正常参照估计。

## 1. 主 Claim

### 推荐主 claim

本文指出 CLIP-ZSAD 中存在一个被忽视的参照缺口：固定文本原型提供通用语义参照，但缺少当前测试图的正常外观基准。为补这个缺口，我们从单张测试图中估计图像条件化正常参照，并将频域响应作为参照证据选择的可靠性信号，而不是直接融合为异常分数。

### 一句话版本

CLIP-ZSAD 缺的不是一个更强的频域异常分数，而是当前图像的正常参照；频域线索的价值在于帮助选择证据来估计这个参照。

### 不采用的过强版本

- 不写“重新表述整个 ZSAD 问题”。这太大，除非有系统性理论和全领域实验证据。
- 不写“CLIP 已经看见高频异常，只差参照”。这会预设一个尚需诊断证明的事实。
- 不写“频域线索就是参照”。频域响应本身仍有歧义，不能直接当参照或异常标签。

## 2. 引言逻辑链

### 2.1 从真实瓶颈开始

异常判断需要参照。一个局部模式是不是异常，不只取决于它局部响应是否强，还取决于它在当前图像、当前材质和当前结构中是否正常。规则纹理、物体边界和粗糙表面都可能产生强局部变化，但它们不一定是缺陷。

### 2.2 CLIP-ZSAD 的缺口

CLIP-ZSAD 通常用固定正常/异常文本原型提供通用语义参照。这种参照可以跨类别泛化，但不知道当前测试图的正常外观。问题因此不是简单的“CLIP 不够强”，而是固定原型缺少当前图像条件下的正常基准。

### 2.3 为什么走向图像条件化参照

在 zero-shot 设置下，目标类别 normal support images 不可用，也不能训练目标域模型。全局固定参照又难以覆盖不同材质和结构。因此，一个自然的选择是从当前测试图自身估计图像条件化正常参照。

这一点在正文中不用写成“必须逐图”，而应写成自然过渡：既然固定文本原型缺少当前图像外观，而外部正常库不可用，本文从当前图像本身估计参照。

### 2.4 为什么不能只靠语义自参照

如果选择“可信正常 patch”的依据完全来自原始 CLIP 语义分数，那么选择器和被修正对象来自同一套分数。这个过程容易形成语义自循环：初始分数没有充分区分的区域，也很难被同一分数体系选出来纠正自身。

正文中优先使用“语义自循环”或“自参照受原始分数边界限制”，少用“CLIP 局部盲区”这种需要额外定义的词。若使用“盲区”，必须在实验中给出阈值定义和子集规模。

### 2.5 频域线索的角色

频域/小波响应不是异常标签，也不是最终异常图。它只是一个不直接来自文本相似度的局部结构信号，用来辅助判断哪些 patch 适合参与图像条件化正常参照估计。

DirectFusion 差，证明频域响应不能直接当异常分数。Ours 高于 SemanticProto，才支持频域响应作为参照证据选择信号有价值。

## 3. 贡献边界

### 可以写的贡献

1. 指出 CLIP-ZSAD 的参照缺口：固定文本原型提供通用语义参照，但缺少当前测试图的正常外观基准。
2. 提出一种无训练的图像条件化正常参照估计方法：从单张测试图选择可靠 patch 证据，构造当前图像的正常参照，并在原 CLIP 语义空间中重算异常图。
3. 重新定位频域响应的作用：它不是异常评分器，而是参照证据选择的可靠性信号。
4. 用 controlled ablations 区分三种解释：直接频域融合、纯语义自参照、频域可靠性参与参照估计。

### 不能写或必须降级的说法

| 禁用/降级表述 | 原因 | 替代表述 |
|---|---|---|
| 首次发现频率对 ZSAD 有用 | 已有频率类 CLIP-ZSAD 工作 | 频率有用是背景；本文关注其使用位置和角色 |
| 频域响应直接代表异常 | 正常纹理和边界也可能高频 | 频域响应是有歧义的可靠性线索 |
| 逐图参照已在所有数据集机制成立 | MPDD/BTAD/DTD 缺 matching controlled ablations | 机制 claim 限定在 MVTec/VisA controlled setting |
| conservative update 严格优于 no-conservative | 当前记录显示 normal FP 不高于 baseline，但不保证严格优于 no-conservative | conservative update 不增加 normal-image false positives over baseline |
| 系统级五数据集结果证明核心机制 | system-level rows 含 multi-crop / pixel-to-image | 五数据集说明方法整体有效；机制证据来自 controlled ablations |

## 4. 证据链冻结

指标顺序统一为 `pixel AUROC / pixel AUPRO / image AUROC / image AP`。

### 4.1 已实测，可支撑正文主张

| 证据链 | 当前证据 | 可支撑的写法 | Source |
|---|---|---|---|
| Ours > Baseline | MVTec `91.2/83.2/91.6/96.4` -> `91.8/86.2/94.1/97.4`; VisA `95.5/86.7/82.0/85.3` -> `96.2/91.7/84.3/87.3` | controlled setting 下 Ours 相比固定原型 baseline 四指标提升，最大提升在 pixel AUPRO | `paper/tables/prototype_main_result.csv` |
| DirectFusion < Ours | MVTec DirectFusion `88.7/80.4/92.9/96.9`; VisA `94.6/85.1/81.6/84.8` | 频域响应不能直接作为最终异常分数；直接融合会破坏定位 | `paper/tables/prototype_main_component_comparison.csv` |
| Ours > SemanticProto | MVTec SemanticProto `91.6/85.2/93.7/97.1`; Ours `91.8/86.2/94.1/97.4`; VisA SemanticProto `96.0/90.4/83.7/86.9`; Ours `96.2/91.7/84.3/87.3` | 只靠 CLIP 语义自参照不够；频域可靠性参与证据选择带来小而一致的增益 | `paper/tables/prototype_main_component_comparison.csv` |
| Boundary-aware / conservative design | HF-only, boundary-aware, Ours 形成小幅提升；Ours 在两数据集最好 | 可作为设计消融，不能写成主创新 | `paper/tables/prototype_wavelet_effect_comparison.csv` |
| Normal-image stability | Ours normal FP area 不高于 baseline；runtime overhead <= 25% | 图像条件化参照没有以正常图误报为代价 | `narrative_workspace/EXPERIMENT_TARGETS.md` |

### 4.2 待补，不可写成已完成

| 待补证据 | 为什么重要 | 论文中当前状态 |
|---|---|---|
| GlobalRef vs PerImageRef | GlobalRef: MVTec `91.2/84.3/93.1/96.9`, VisA `95.8/89.2/83.2/86.4`; Ours - GlobalRef pAUPRO = `+1.9` / `+2.5` | Supported as manual diagnostic record; manual provenance is in `narrative_workspace/GLOBALREF_EXPERIMENT_RECORD.md`, original command/log unavailable |
| CLIP 盲区召回 | 证明频域可靠性补充了语义自参照漏掉的区域 | 可作为后续机制增强实验；必须定义阈值、子集规模、召回增量和非子集误报 |
| 类别级 / 纹理微缺陷分组 + bootstrap | 证明增益集中在 reference gap 最明显的类别 | 未完成前不能声称机制增益集中于纹理类 |
| 真实 qualitative 图 | 展示频域响应有歧义，以及 Ours 不等于复制频域图 | 需要从真实 inference 生成；占位图不能进正式结果 |
| 外部 SOTA 数字核验 | 避免 protocol mismatch 和错误引用 | 未核验数字只能标 protocol-reference 或 appendix |

## 5. 主消融表建议

主消融应该按“拆机制链条”排列，而不是按模块堆叠排列。

| 变体 | 拆掉什么 | 解释 |
|---|---|---|
| Baseline / FixedText | 没有当前图参照 | 固定文本原型只能提供通用语义参照 |
| DirectFusion | 有频域响应，但无参照，直接作为异常分数 | 检验频域响应是否可以直接用；预期和实测都说明不能 |
| SemanticProto | 有当前图参照，但证据选择只来自 CLIP 语义分数 | 检验语义自参照是否足够 |
| GlobalRef | 有参照，但不是图像条件化 | 已完成；改善 direct fusion / baseline 的定位指标，但明显低于 Ours，尤其 pixel AUPRO |
| Ours | 当前图参照 + 频域可靠性证据选择 | 完整机制 |

GlobalRef 已可放入主机制表；当前 provenance 是人工补档，原始命令、log path、commit 未恢复。正文中不要声称该行已有 command-level reproducibility。

## 6. 章节结构

### 摘要

只写可被当前证据支撑的版本。GlobalRef 已支持图像条件化参照，但其 provenance 是人工补档；盲区召回未完成前，摘要不要写“召回若干盲区异常”。

摘要核心句：

> 固定文本原型缺少当前测试图的正常外观基准。本文估计图像条件化正常参照，并将频域响应作为参照证据选择的可靠性信号，而不是直接融合为异常分数。

### 1. 引言

段落顺序：

1. CLIP-ZSAD 的固定文本原型范式及其价值。
2. 异常判断的参照性：同一局部模式在不同图像中含义不同。
3. 固定文本原型的参照缺口：缺当前图像正常外观。
4. 为什么从当前图像估计参照，以及为什么语义自参照有自循环风险。
5. 本文方法：频域可靠性辅助证据选择，估计图像条件化正常参照。
6. 贡献列表：问题缺口、方法、机制消融。

### 2. 相关工作

建议三段：

1. CLIP-based ZSAD：WinCLIP、AnomalyCLIP、VCP-CLIP、AA-CLIP 等。定位为固定/学习语义原型和视觉上下文增强。
2. 频率用于异常检测：FE-CLIP、WMoE-CLIP、HarmoniAD 等。承认频率有用已是背景；本文不做直接频率融合。
3. Test-time / reference-based adaptation：WinCLIP+、prompt/pseudo-label/retrieval 类方法。强调本文不使用 target normal bank，不训练参数，而是从当前图像估计参照。

### 3. 方法

推荐小节：

1. 固定原型打分与参照缺口。
2. 图像条件化正常参照估计框架。
3. 频域可靠性构造：Haar DWT、HF、boundary-aware reliability。
4. 证据选择：语义分数 + 频域可靠性共同选择 normal evidence。
5. 保守原型更新与重打分：强调不训练、不反传、最终图仍来自语义相似度。

### 4. 实验

先分清口径：

- Controlled MVTec/VisA：用于机制 claim。
- Five-dataset system-level：用于整体效果展示，必须标 system-level / upper bound。

正文主表优先放 controlled MVTec/VisA。五数据集可作为后续表或 appendix，避免和 controlled claim 混用。

### 5. 机制分析

建议顺序：

1. DirectFusion vs Ours：频域不能直接当异常分数。
2. SemanticProto vs Ours：频域可靠性作为证据选择信号有价值。
3. GlobalRef vs Ours：当前结果支持图像条件化参照优于固定参照，尤其在 pixel AUPRO 上 MVTec `+1.9`、VisA `+2.5`。
4. Normal-image stability：不增加正常图误报。
5. 设计消融：HF-only、boundary-aware、conservative update。
6. 盲区召回和类别分析：完成后加入；未完成则不写结论。

### 6. 局限

必须保留：

- 当前机制证据主要来自 MVTec/VisA controlled setting。
- MPDD/BTAD/DTD 暂为 system-level 或 dataset-tuned rows，不能作为机制归因。
- 方法假设测试图中有足够可靠正常区域；大面积异常、逻辑异常和背景强干扰可能失败。
- 频域响应仍有歧义，因此本文刻意不把它作为直接异常图。

## 7. 下一步执行顺序

1. 先把 `PAPER_PLAN.md` 的主 claim 和 introduction plan 改成“参照缺口”版本。
2. 再决定是否更新 active LaTeX 源。优先更新 `paper/sections/*.tex`，因为这组文本已经更接近英文论文口径；`newversion/paper_v7` 仍有 EXPECTED 占位，需谨慎。
3. 保留 GlobalRef 的人工 provenance，并在后续若找回原始 log/command 时替换。
4. 生成真实 qualitative 图。占位图只能用于内部排版。
5. 做盲区召回与类别级 bootstrap。完成后再把“增益集中在纹理/微缺陷类”写入主文。

## 8. 写作红线

- 每个 claim 必须能指向“已实测证据”或明确标为 pending。
- 摘要只写已完成证据，不写未来目标。
- 所有 `Ours` 行必须带清楚口径：controlled setting、system-level、dataset-tuned upper bound。
- 频率相关表述一律从“直接异常分数”降级为“参照证据选择的可靠性信号”。
- GlobalRef 已完成并支持图像条件化参照；当前为人工 provenance，不要写成已具备原始 log/command 复现链。

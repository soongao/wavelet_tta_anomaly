# WPTA 中文顶会稿自审 round7

审阅对象：`outputs/wpta_cvpr_paper_draft_zh_v1.1.md`

审阅目标：检查 v1.1 是否相对 v1.0 更接近 CVPR/ICCV 投稿要求。中文仅用于内部审阅，不降低 claim-evidence、图表、引用和实验完整性要求。

## Summary

- CRITICAL: 3
- MAJOR: 4
- MINOR: 4
- 当前评分：7.5 / 10
- 投稿建议：not ready for submission；可作为较稳定的内部审阅稿和下一轮资产生成输入。

Top three blocking fixes:

1. 生成真实 Figure 2 方法图和 Figure 3 跨数据集 qualitative grid，并完成 Figure 1 矢量排版 gate。
2. 完成 external protocol verification；否则投稿版必须继续把外部方法比较留在附录或删除。
3. 完成 canonical citation metadata 核验，尤其是 local-bib-only 数据集引用。

## 已完成的 v1.1 修订

| Item | Status | Evidence |
|---|---|---|
| 表格版本统一到 v0.7 | pass | 稿件状态与 5.5 均引用 `outputs/wpta_generated_tables_v0.7.md` 和 LaTeX v0.7。 |
| Prompt 从正文语境移出 | pass | Figure/Table prompts 统一放入第 10 节“非投稿正文生成指令”。 |
| Figure 1 raster gate 明确化 | pass | 引言 Figure 1 段落说明 PDF candidate 是 raster 内容封装，不满足最终矢量排版 gate。 |
| 外部比较公平性限制补强 | pass | 4.3 新增 comparison scope 段落，明确当前只相对固定 AnomalyCLIP baseline 和受控 variants 成立。 |
| Rejected external ranking claim 从 claim map 清理 | pass | 正文 claim map 只保留支持或 appendix-only claim，禁止 claim 移到 8.1 内部禁止 claim。 |

## Dimension 1: Macro Logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | v1.1 的核心逻辑更稳：Table 1 是 system-level result，Table 2/3 是 MVTec/VisA controlled mechanism evidence，Table 4 是归因边界。 | - | 保持该三层结构。 |
| 2 | 4.3 已主动承认外部比较范围限制，这能降低 reviewer 对不公平强比较的质疑，但仍不能替代真实 strongest-baseline 主表。 | CRITICAL | 若要投 CVPR/ICCV 主会，优先完成外部协议核验表。 |
| 3 | Figure 2/3 仍未生成真实资产，方法总览和跨数据集定性证据仍缺。 | CRITICAL | 先生成 Figure 2 矢量方法图，再生成 Figure 3 真实模型输出 grid。 |

## Dimension 2: Claim-Evidence Alignment

| # | Claim | Verdict | Evidence / Risk |
|---|---|---|---|
| 1 | Final calibrated system improves fixed AnomalyCLIP baseline on five industrial datasets. | keep | Table 1 支撑，Table 4 限定 final setting 差异。 |
| 2 | WPTA mechanism is supported by controlled MVTec/VisA ablations. | keep with scope | Table 2/3 支撑，v1.1 没有扩大到五数据集机制 claim。 |
| 3 | Direct wavelet fusion is harmful/insufficient. | keep | Table 2/3 中 direct fusion 低于 baseline 或 full，尤其 P-AUPRO 明显下降。 |
| 4 | External method ranking. | forbidden internal claim | 已移到 8.1，不再作为正文 supported claim。 |
| 5 | Medical observation beyond industrial data. | appendix only | 只有 ISIC/ISBI，且只有 pixel-level 指标。 |

## Dimension 3: Experiments

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 当前主结果仍是 deterministic report，没有多 run/multi seed。 | MAJOR | 若要显著性或置信区间，补多 run；否则保持 deterministic-only wording。 |
| 2 | 缺 classwise breakdown 和失败案例。 | MAJOR | 按第 10 节 prompt 生成类别级表和失败案例附录。 |
| 3 | Table 4 对 MPDD/BTAD 未启用 wavelet/TTA flags 的说明保留正确。 | - | 不要把 MPDD/BTAD gains 归因给 WPTA。 |
| 4 | 外部比较表仍只能 protocol-reference。 | CRITICAL | 核验 split/backbone/input size/eval script 后再进入主文。 |

## Dimension 4: Figures And Tables

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Table v0.7 已覆盖当前能安全生成的主文和附录表。 | - | 后续正文继续引用 v0.7 或更新到更高版本。 |
| 2 | Figure 1 有真实模型输出资产，但 PDF candidate 不是最终矢量图。 | CRITICAL | 用 panel crops 重绘矢量文字、图例、箭头和标签，热图内容保持不变。 |
| 3 | Figure 2 只有 prompt，没有资产。 | MAJOR | 生成方法总览图，并确保模块名与方法小节一致。 |
| 4 | Figure 3 只有 prompt，没有跨数据集真实图。 | MAJOR | 从真实结果输出生成，不得手工绘制热图。 |

## Dimension 5: Citations And Integrity

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | v1.1 没有未核验引用占位。 | - | 保持。 |
| 2 | `zou2022visa`、`jezek2021deep`、`mishra2021vt`、`aota2023zero` 仍需 canonical metadata 核验。 | MAJOR | 用 arXiv/DOI/CrossRef/Semantic Scholar/publisher page 核验 title、authors、year、venue、identifier。 |
| 3 | Closest-work 表格仍依赖对 prior work 的人工理解。 | MAJOR | 对每行训练条件、测试时适配、小波可靠性描述做 claim-to-source alignment。 |

## Mechanical QA

- 禁用实验项 scan：pass。
- 旧表格版本 scan：pass。
- 过强排名词 scan：pass。
- 未核验引用占位 scan：pass。
- AI-tone / em dash scan：未发现 pre-submission reviewer banned vocabulary 或 em dash。
- Prompt placement：`FIGURE_PROMPT` 和 `TABLE_DATA_PROMPT` 只出现在第 10 节。

## Final Verdict

v1.1 相比 v1.0 更接近投稿前审阅稿：证据边界更清楚，外部比较公平性限制更诚实，内部生成 prompt 不再打断正文，表格版本已统一到 v0.7。当前文本层面已经比较稳，但仍未达到投稿水平。主要缺口不是继续润色，而是关闭投稿资产 gate：Figure 1 矢量化、Figure 2/3 真实图、外部协议核验、canonical citation metadata、可选类别级/失败案例补充。当前评分 7.5/10；若上述资产和协议 gate 完成，预计可进入 8.5 左右的投稿前精修阶段。

# WPTA 中文顶会稿自审 round6

审阅对象：`outputs/wpta_cvpr_paper_draft_zh_v1.0.md`

审阅目标：按 CVPR/ICCV 类顶会标准检查中文审阅稿是否接近投稿级。中文仅用于内部审阅，不降低 claim-evidence、图表、引用和实验完整性要求。

## Summary

- CRITICAL: 3
- MAJOR: 5
- MINOR: 5
- 当前评分：7.2 / 10
- 投稿建议：not ready for submission；可作为完整内部审阅稿继续迭代。

Top three fixes:

1. 生成真实 Figure 2 和 Figure 3，并完成 Figure 1 投稿级矢量重绘。
2. 完成引用 canonical metadata 核验，尤其是 local-bib-only 数据集引用。
3. 完成外部方法 protocol verification，或明确删除外部比较主文入口。

## Dimension 1: Macro Logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 稿件正确区分了 “Table 1 system-level result” 与 “Table 2/3 WPTA controlled mechanism evidence”。这是 v1.0 的核心优点。 | - | 保持该边界，不要在后续版本中把 MPDD/BTAD final gain 归因为 WPTA。 |
| 2 | “外部方法比较当前只作为附录协议参考，不作为主文强比较” 会被 reviewer 视为缺少 strongest baselines。 | CRITICAL | 核验 Appendix B1a/B1b 的 split、backbone、input size、post-processing 和 evaluation script；若不能核验，投稿主文需承认 comparison scope，并避免最佳方法 claim。 |
| 3 | Figure 2/3 仍是 prompt，方法 pipeline 与跨数据集定性证据没有真实图支撑。 | CRITICAL | 生成 Figure 2 方法图和 Figure 3 qualitative grid；Figure 3 未生成前，正文不得使用跨数据集定性观察支撑 claim。 |

## Dimension 2: Claim-Evidence Alignment

| # | Claim | Verdict | Evidence / Risk |
|---|---|---|---|
| 1 | Final calibrated system improves fixed AnomalyCLIP baseline on five industrial datasets. | keep | Table 1 支撑，Table 4 限定配置差异。 |
| 2 | WPTA mechanism is supported by controlled MVTec/VisA ablations. | keep with scope | Table 2/3 支撑，但只限 MVTec/VisA。 |
| 3 | Direct wavelet fusion is a negative control. | keep | Table 2/3 中 direct fusion 的 P-AUPRO 低于 baseline/full。 |
| 4 | Current system reaches external best-method status. | remove / keep only as rejected claim audit | 外部协议未核验；v1.0 已放在 claim map 的 Not supported 行，投稿版应删除该行或移到内部审计。 |
| 5 | Medical observation beyond industrial data. | appendix only | 只有 ISIC/ISBI 两项 pixel metrics，不能进入摘要、贡献或主结论。 |

## Dimension 3: Experiments

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 当前没有 multi-run / multi-seed，不可报告显著性或置信区间。v1.0 已明确 deterministic-only。 | MAJOR | 若投稿期允许，补 3 次以上独立 run；否则继续保持 deterministic report wording。 |
| 2 | Table 2/3 因果解释主要依赖 adaptation variants 之间相对变化，v1.0 已正确写明。 | - | 保持表注，不要删除。 |
| 3 | Table 1 五数据集结果强，但 final setting 数据集特异，系统效果与机制效果必须分开。 | MAJOR | Table 4 应保留在主文或附录靠前位置。 |
| 4 | 缺类别级 breakdown 和失败案例。 | MAJOR | 生成 classwise breakdown table 和失败案例附录；若缺日志，保留缺失 prompt。 |

## Dimension 4: Figures And Tables

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Figure 1 已接入真实 MVTec cable 资产，且 provenance 说明清楚。 | - | 可用于中文审阅稿。 |
| 2 | Figure 1 PDF 是 raster panel 封装，不满足投稿“vector figure only”的要求。 | CRITICAL | 基于 `outputs/figures/figure1_panels/` 重绘矢量文字、箭头、图例和 panel 标签；热图像素内容不得手改。 |
| 3 | Table 1-4 和 Appendix A/B 已有 Markdown/LaTeX v0.6；禁用项已清理。 | - | 后续论文正文引用统一指向 v0.6。 |
| 4 | Prompt blocks 位于正文中，便于生成图表，但不是最终投稿正文。 | MINOR | 投稿版应将 prompt blocks 移到 “生成指令附录” 或从 final manuscript 删除。 |

## Dimension 5: Citations And Integrity

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | v1.0 没有 `CITATION-PENDING` 占位，也没有从记忆生成 BibTeX。 | - | 保持。 |
| 2 | `zou2022visa`、`jezek2021deep`、`mishra2021vt`、`aota2023zero` 仍是 local-bib-only。 | MAJOR | 用 arXiv/DOI/CrossRef/Semantic Scholar/publisher page 核验 title、authors、year、venue 和 identifier。 |
| 3 | Related Work 中对 closest work 的区别尚未做 claim-to-source alignment。 | MAJOR | 核验每个方法是否确实符合表中“训练/测试时适配/小波可靠性”等描述；未核验前用更保守措辞。 |

## Mechanical QA

- 禁用项 scan：未发现 `runtime`、`normal stability`、`expected_pass`。
- 旧表格版本 scan：未发现 `v0.5` 或旧 figure ledger。
- 过强词 scan：未发现 `SOTA`、`state-of-the-art`、`最优`、`最先进`。
- AI-tone / em-dash scan：未发现 pre-submission reviewer banned vocabulary 或 em dash。

## Revision Actions For v1.1

当前证据内可直接修：

- 将 rejected external-best claim 从正文 claim map 中弱化为“内部禁止 claim”。
- 将 prompt blocks 明确标成“非投稿正文生成指令”。
- 在实验设置中补一段 “comparison scope and fairness limitation”，提前回应 reviewer 对 baseline completeness 的质疑。
- 在 Figure 1 段落中补 “not vector-ready” gate，避免误用 PDF candidate。

需要新增外部状态或实验资产：

- Figure 2 方法总览真实图。
- Figure 3 跨数据集真实 qualitative grid。
- 外部方法 protocol verification table。
- canonical citation metadata。
- 可选 multi-run uncertainty table 和 classwise breakdown。

## Final Verdict

v1.0 已经是一篇结构完整、证据边界清楚的中文顶会审阅稿；摘要、引言、方法和实验主线能让 reviewer 看懂贡献与限制。它还没有达到投稿水平，主要原因不是文本，而是投稿必需的资产 gate 未关闭：真实图不完整、Figure 1 未过矢量 gate、外部方法协议未核验、部分引用未完成 canonical verification。当前评分 7.2/10；关闭 Figure 2/3、引用和协议核验后，预期可进入 8.5+ 的投稿前精修区间。

# WPTA 中文顶会稿 v0.5 预投稿评审 Round 3

评审对象：`outputs/wpta_cvpr_paper_draft_zh_v0.5.md`

## Summary

- CRITICAL: 3
- MAJOR: 4
- MINOR: 5
- Overall score: 7.8 / 10
- Submission recommendation: Needs moderate-to-major revision before submission.

v0.5 相比 v0.3/v0.4 的最大进步是修正了因果归因边界：五数据集结果现在只写成 final calibrated system 的系统级提升，WPTA 机制只由 MVTec/VisA 受控消融支撑。这个修正确实解决了最危险的 reviewer attack point。当前稿件已经适合作为合作者技术评审稿，但还没有达到 CVPR/ICCV 投稿闭环。

## Dimension 1: Macro Logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | “证据边界：WPTA 机制本身由 MVTec/VisA 受控消融支撑；五数据集主结果支撑的是最终校准系统...” | MINOR | 这是正确修正，应保留。英文投稿版可把这段转成 Experiment Setup 的 protocol paragraph，而不是作为写作边界前言。 |
| 2 | “本文采用两层实验叙事...” | MINOR | 逻辑清楚，但正式论文不应显得像作者在解释防御策略。改成自然实验组织：first validate mechanism, then report final-system results. |
| 3 | “为保证受控消融中 adaptation variants 的可比性，除 baseline 外...” | MAJOR | 这句话暴露了 baseline 与 variants 不完全同配置。需要在表注里明确 baseline 是否包含 multi-crop/p2i，否则 reviewer 会质疑 Table 2 中 baseline-to-method delta 的公平性。 |

## Dimension 2: Writing Details

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Abstract 当前是一整段长句群，超过顶会摘要的可读阈值。 | MINOR | 英文版按五句式重写：task, challenge, insight, method, results。 |
| 2 | “Full controlled WPTA 在 AUPRO 上达到 86.2/91.7。” | MINOR | 摘要中只给 AUPRO 可能显得选择性报告；建议写 “on MVTec/VisA” 并说明指标。 |
| 3 | Related Work 全部是 `[CITATION-NEEDED]`。 | CRITICAL | 不能投稿。至少补齐 CLIP、AnomalyCLIP、WinCLIP、AdaCLIP、CLIP-AD/PromptAD、TTA、MVTec、VisA、MPDD、BTAD、DTD 的真实 BibTeX。 |

## Dimension 3: Claim-Evidence Audit

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | “Final calibrated system improves AnomalyCLIP baseline on five industrial datasets.” | PASS | Table 1 支撑，Table 4 约束归因边界。 |
| 2 | “Wavelet reliability adds value beyond semantic-only adaptation.” | PASS with scope | Table 2 支撑 MVTec/VisA，正文必须始终带这个范围。 |
| 3 | “当前系统在 MVTec/VisA 的 P-AUPRO 上具有较强位置...” | MAJOR | Table 5 协议未核验，建议放附录，主文只保留一句 protocol-reference。 |
| 4 | ISIC/ISBI 附录观察 | PASS with scope | 已正确降级为 appendix-only，不应进入摘要/贡献。 |

## Dimension 4: Experiment Completeness

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 真实 qualitative figure 缺失，当前只有 prompt。 | CRITICAL | 至少生成 4 个工业数据集的真实 Input/GT/baseline/W/evidence/final map。 |
| 2 | 外部方法协议未核验。 | CRITICAL | 要么完成 split/backbone/input/eval/post-processing 核验，要么把 Table 5 放附录并明确不参与主结论。 |
| 3 | 实现细节已有参数，但仍缺 backbone、input size、cache generation、upsampling、image-score aggregation 的最终确认。 | MAJOR | 从运行命令和代码补齐 “Implementation Details” 表格或 Appendix。 |

## Dimension 5: Figure/Table Quality

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Table 1/2/3/4/5 的证据职责清晰。 | PASS | 保持当前表分工。 |
| 2 | Table 2 和 Table 3 现在是宽表，中文稿可读，但 CVPR 双栏会很挤。 | MINOR | LaTeX 版保留 table*，或拆成 MVTec/VisA 两个子表。 |
| 3 | Figure 1/2/3 仍是 prompt，不是真实或矢量成品。 | MAJOR | Figure 2 可先画矢量框图；Figure 3 必须来自真实输出。 |

## Banned-Vocabulary and Formatting Scan

- Em-dash scan: pass, no Unicode em dash found.
- AI-tone banned terms: pass in current Chinese/English mixed draft.
- Explicit unsupported SOTA claim: pass; all SOTA mentions are negative/forbidden wording.
- Deleted items scan: pass; no active table for the two removed items.

## Final Score

| Dimension | Score |
|---|---:|
| Originality | 7.7 |
| Methodological rigor | 7.7 |
| Evidence sufficiency | 7.9 |
| Experiment completeness | 7.3 |
| Writing and presentation | 8.0 |
| Overall | 7.8 |

## Top 3 Blocking Issues

1. Replace all `[CITATION-NEEDED]` with verified BibTeX and accurate `\cite{}`.
2. Generate real qualitative figures, especially W/evidence/final map for mechanism validation.
3. Decide Table 5 fate: protocol-verify it or move it to appendix/remove it.

## Next Revision Plan

1. Build `references.bib` from verified sources and update Related Work.
2. Create a real Figure 2 method overview and a Figure 3 qualitative panel from actual outputs.
3. Convert v0.5 into English LaTeX sections after citations and figures are no longer placeholders.

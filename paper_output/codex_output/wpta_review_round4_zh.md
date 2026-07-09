# WPTA 中文顶会稿 v0.6 预投稿评审 Round 4

评审对象：`outputs/wpta_cvpr_paper_draft_zh_v0.6.md`

## Summary

- CRITICAL: 2
- MAJOR: 4
- MINOR: 5
- Overall score: 8.1 / 10
- Submission recommendation: Needs focused revision before submission.

v0.6 相比 v0.5 的主要进步是三点。第一，摘要重写成 task、challenge、insight、method、result 的顶会式逻辑。第二，外部方法比较从主文候选表降级为 appendix/protocol-reference，不再污染主 claim。第三，引用状态被拆成 citation ledger，核心 CLIP、AnomalyCLIP、WinCLIP、AdaCLIP、TPT 和 MVTec 已有可追溯元数据，不再把所有引用都留成同一类占位。

## Dimension 1: Macro logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | “五数据集表只证明系统级有效性，WPTA 机制的因果证据限定在 MVTec/VisA 受控消融。” | PASS | 这是当前稿件最重要的 claim boundary，应保留到英文版。 |
| 2 | “除 Baseline 外的 prototype/fusion variants 使用相同的 multi-crop 与 pixel-to-image 设置。” | MAJOR | 该限制已经说明，但 Table 2 caption 还应更明确：baseline-to-full delta 不是纯 WPTA delta，机制证据主要来自 adaptation variants 之间。 |
| 3 | Introduction 的两层证据组织清楚，但 contribution 4 有防御性表述。 | MINOR | 英文版可改成 “We report final-system results with a configuration audit” 这类自然写法。 |

## Dimension 2: Writing clarity

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 摘要可读性明显优于 v0.5。 | PASS | 英文版保持五句式，不要再塞入过多实现细节。 |
| 2 | Method 各小节 now have motivation, design and advantage. | PASS | 需要在最终 LaTeX 中补图 2，帮助读者快速理解 pipeline。 |
| 3 | Related Work 仍有 `CITATION-PENDING: wavelet texture inspection`。 | MAJOR | 必须补一个真实 wavelet/surface inspection 或纹理缺陷引用，或者删除泛化背景句。 |

## Dimension 3: Citation integrity

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | `radford2021clip`, `zhou2024anomalyclip`, `jeong2023winclip`, `cao2024adaclip`, `shu2022tpt`, `bergmann2019mvtec` 已有元数据核验。 | PASS | 下一步生成正式 BibTeX，并确保 author/year/venue 与 ledger 一致。 |
| 2 | `zou2022visa`, `jezek2021deep`, `mishra2021vt`, `aota2023zero` 仍是 local-bib metadata only。 | MAJOR | 投稿前必须用 CrossRef/arXiv/IEEE/Springer/WACV proceedings 核验。 |
| 3 | 当前稿件仍没有最终 `.bib`。 | CRITICAL | 不能投稿。需要生成 `wpta_references.bib` 并逐条核验。 |

## Dimension 4: Experiment completeness

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Table 1/2/3/4 分工清楚，证据边界合理。 | PASS | 保持主文四表结构。 |
| 2 | 真实 qualitative figure 缺失。 | CRITICAL | 至少生成 Figure 1 或 Figure 3 的真实热图面板；当前只有 prompt。 |
| 3 | 外部方法比较没有协议核验。 | MAJOR | 已降级处理，若时间不够可以保留附录 prompt 而不进入主文。 |
| 4 | 没有多 run/multi seed，不应报告显著性。 | PASS with scope | v0.6 已明确 deterministic-only，不要新增 p-value。 |

## Dimension 5: Figure and table quality

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Table 2/3 使用 slash cell，中文审稿可读；CVPR 双栏可能拥挤。 | MINOR | LaTeX 版使用 `table*`，或拆成 MVTec/VisA 两个子表。 |
| 2 | Table 4 配置审计很有价值，但主文可能占空间。 | MINOR | 若页数紧张，保留摘要版主文，完整 log path 放 appendix。 |
| 3 | Figure prompts 具体，但仍不是 paper figures。 | MAJOR | Figure 2 可以先做矢量框图；Figure 3 必须用真实输出。 |

## Banned-vocabulary and formatting scan

- Em-dash: no known active issue in the new Chinese draft.
- Unsupported SOTA claim: no active SOTA claim; only negative boundary statement.
- Deleted items: `runtime` and `normal stability` should remain absent.
- Citation placeholders: reduced but not eliminated.

## Final score

| Dimension | Score |
|---|---:|
| Originality | 7.9 |
| Methodological rigor | 8.0 |
| Evidence sufficiency | 8.2 |
| Experiment completeness | 7.7 |
| Writing and presentation | 8.5 |
| Overall | 8.1 |

## Top blocking issues

1. Generate real qualitative figures from actual outputs.
2. Finish citation verification and produce final BibTeX.
3. Decide whether external comparison is deleted, appendix-only, or protocol-verified for main text.

## Next revision plan

1. Build `outputs/wpta_references.bib` from verified metadata and local-bib entries after final verification.
2. Inspect existing mechanism visualization artifacts under the AnomalyCLIP project and generate Figure 1/3 panels from real maps.
3. Convert v0.6 into English LaTeX only after citation and figure gates are closed.

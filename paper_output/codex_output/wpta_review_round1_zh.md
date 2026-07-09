# WPTA 中文顶会稿 v0.2 预投稿评审 Round 1

评审对象：`outputs/wpta_cvpr_paper_draft_zh_v0.2.md`

评审标准：按 CVPR/ICCV 类方法论文要求审查。中文稿仅用于内部审核，不降低贡献清晰度、证据完整性、实验严谨性和表述克制要求。

## 总分

| 维度 | 分数 | 依据 |
|---|---:|---|
| Originality | 7.5 / 10 | “小波作为 prototype adaptation reliability，而非 final map fusion” 有明确技术定位，区别于简单拼接。 |
| Methodological rigor | 7.0 / 10 | 方法链条清楚，公式和模块对应较好；但 implementation details、confidence gate、aggregation protocol 仍未完整写清。 |
| Evidence sufficiency | 7.2 / 10 | 五个工业数据集主结果强于旧稿，受控消融支持核心机制；外部方法协议未核验，定性图未落地。 |
| Experiment completeness | 6.8 / 10 | 主表与消融较完整，但 top-tier 投稿通常还需要真实 qualitative、协议一致的外部方法对比、实现细节。 |
| Writing and presentation | 7.4 / 10 | 中文逻辑顺畅，过度 claim 控制较好；Introduction 仍缺一个贯穿全文的真实 running example，Related Work 引用待补齐。 |
| Overall | 7.2 / 10 | 接近可投稿论文骨架，但还不是投稿完成稿。 |

## Verdict

Major Revision before submission.

当前稿件已经从“idea 草稿”推进到“有主实验和消融支撑的方法论文草稿”。但若按顶会投稿标准，仍有三类阻断项：引用未核验、外部方法对比协议未核验、真实定性图缺失。v0.3 应优先修正不依赖新实验的写作问题，并把仍需外部证据的内容明确移到待完成清单。

## Critical Issues

| # | Section | Finding | Suggested fix |
|---|---|---|---|
| C1 | Related Work / 全文引用 | 文中存在多个 `[CITATION-NEEDED]`，例如 “`[CITATION-NEEDED: CLIP]` `[CITATION-NEEDED: AnomalyCLIP]`”。正式投稿不能保留引用占位。 | v0.3 保留为内部稿可以接受，但必须在预投稿清单中标为 hard blocker；英文投稿版必须换成真实 BibTeX 和 `\cite{}`。 |
| C2 | Experiments | “候选外部方法对比”来自相近但未核验协议，不能支撑 SOTA 或强外部优越性。 | v0.3 中将该表从主结果叙事降级为“协议参考/附录候选”，主 claim 仅保留固定 AnomalyCLIP baseline 和受控消融。 |
| C3 | Figure evidence | `qualitative visualization` 仍是 prompt，没有真实 anomaly map / W map / selected evidence patch。 | v0.3 保留图 prompt，但在正文中不把 qualitative 作为已完成证据；最终投稿前必须补真实图。 |

## Major Issues

| # | Section | Finding | Suggested fix |
|---|---|---|---|
| M1 | Introduction | 缺少贯穿全文的 concrete running example。当前有机制描述，但没有一个具体样例帮助审稿人理解 fixed prototype mismatch 和 wavelet false activation。 | 加入“金属件边缘附近细划痕”作为运行示例，并在 Method/Qualitative plan 中回扣。 |
| M2 | Method | `confidence gate`、`rho(.)`、image-level aggregation 仍过抽象。 | 加一节 “Implementation details to report in final version”，列出必须补齐的具体实现项，避免审稿人认为方法不可复现。 |
| M3 | Main Result vs Ablation | 主结果 MVTec/VisA Full 数值与消融 Full 数值不一致，虽然 v0.2 说明了 setting 差异，但仍容易被误读。 | 表注和结果段落继续强调 “main setting” 与 “controlled ablation setting” 区分；claim map 不混用两者 delta。 |
| M4 | Medical result | ISIC/ISBI 单数据集补充结果太弱，放在主实验段可能分散工业论文主线。 | v0.3 将其移动到 Appendix-style “optional observation”，不放入核心贡献。 |

## Minor Issues

| # | Section | Finding | Suggested fix |
|---|---|---|---|
| m1 | Abstract | 摘要偏长，接近一整段方法细节堆叠。 | 改成五句式：问题、限制、挑战、方法、结果。 |
| m2 | Tables | Table 4 过长且协议未核验，放主文会稀释主表。 | 主文只保留简化摘要，完整表放 `wpta_tables.md` 或附录。 |
| m3 | Language | 中英文术语混用较多，但内部审阅可接受。 | 英文投稿版统一术语；中文稿保留关键英文术语便于对应代码和公式。 |

## Required Revision Actions for v0.3

1. 重写摘要为五句式，压缩方法细节，突出五数据集平均结果。
2. 引言增加 running example，并把它接到 Figure 1 prompt 和方法流程。
3. 将外部方法对比降级为“protocol-reference comparison”，不放在主 claim。
4. 增加 implementation details checklist，列明最终投稿必须报告的超参数、阈值、aggregation 和 per-dataset setting。
5. 更新 claim-evidence map，使所有 supported claim 都只依赖 current CSV 或受控消融表。
6. 保留未完成项，但不要让未完成项污染已完成的主 claim。

## Round 1 Outcome

v0.2 不建议标记为投稿完成。执行上述修订后，预计 v0.3 可达到 “strong internal draft / near-submission skeleton” 水平，但真正投稿完成仍需要真实引用、协议核验和定性图。

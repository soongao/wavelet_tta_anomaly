# WPTA 中文稿 v0.8 投稿前自审 Round 5

审阅对象：`outputs/wpta_cvpr_paper_draft_zh_v0.8.md`

审阅标准：按 CVPR/ICCV 类视觉顶会要求进行预投稿审查。中文仅用于内部审阅，不降低证据、图表、引用、实验完整性或可复现性要求。

## 总体结论

- CRITICAL: 3
- MAJOR: 5
- MINOR: 4
- 当前评分：8.1 / 10
- 投稿建议：not ready for submission

`v0.8` 的论文主线已经清楚：WPTA 的关键贡献不是“把小波响应加到异常图”，而是把 boundary-aware wavelet response 作为 test-time prototype adaptation 的 evidence reliability。稿件也正确区分了五数据集 final calibrated system 的系统级结果和 MVTec/VisA 受控消融中的机制证据。阻止投稿的主要问题不是叙事，而是三类硬证据尚未闭合：公平外部方法比较、真实跨数据集定性图、最终引用与可复现细节。

## Top 3 Blocking Issues

| # | Finding | Severity | Evidence | Suggested fix |
|---|---|---|---|---|
| 1 | 外部方法比较仍是 protocol-reference，不能支撑顶会主结果的竞争性 claim。 | CRITICAL | `v0.8` 第 252-258 行明确写到外部比较未核验；第 299 行写 “Current system reaches SOTA” not supported。 | 完成 `external_protocol_verified_table`，逐项核验 split、backbone、input size、post-processing、evaluation script；只有 `protocol_match=yes` 的方法进入主文。若无法核验，主文必须明确为 AnomalyCLIP baseline improvement paper，投稿风险显著增加。 |
| 2 | Figure 3 仍未生成，缺少跨数据集真实 qualitative evidence。 | CRITICAL | `v0.8` 第 260-268 行仍是 prompt；第 282 行承认 Figure 3 尚未生成。 | 从真实 cache/log 生成覆盖至少四个工业数据集的 qualitative grid，并保存 provenance CSV；MPDD/BTAD 不得误标为 wavelet reliability。 |
| 3 | 引用和数据集 canonical metadata 未闭合，投稿版 `.bib` 不能直接使用。 | CRITICAL | `v0.8` 第 159-161 行说明 `zou2022visa`、`jezek2021deep`、`mishra2021vt`、`aota2023zero` 仍需核验。 | 用 DOI、CrossRef、arXiv 或 publisher metadata 替换 local-bib-only 条目；所有正文 citation key 必须在最终 `.bib` 中有 canonical provenance。 |

## Dimension 1: Macro Logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | “系统级有效性”和“机制级证据”的边界写得正确，但贡献 4 仍容易让审稿人问：final calibrated system 是否就是 WPTA？ | MAJOR | 在贡献列表和实验开头加入一句更硬的边界声明：five-dataset final system validates the calibrated inference stack; WPTA mechanism is isolated on MVTec/VisA controlled ablations。 |
| 2 | Related Work 的 closest-work 表有用，但没有把 “training-free, no target training images, no backpropagation” 作为对比维度。 | MAJOR | 在 Section 2.4 表中加入 training-free/test-time-only 维度，帮助审稿人理解与 AdaCLIP、CLIP-AD、TPT 的差异。 |
| 3 | Introduction 已有六段逻辑，但 Figure 1 仍是 prompt，无法形成 page-1 visual hook。 | CRITICAL | 用现有机制 PNG 生成真实 Figure 1，或至少在稿件中记录 Figure 1 asset path 和 caption draft。 |

## Dimension 2: Method and Reproducibility

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 方法小节缺少算法框，读者需要从 prose 中重建 WPTA。 | MAJOR | 增加 “Algorithm 1: WPTA inference for one test image”，列出 inputs、frozen CLIP features、S0、W、evidence selection、anchor aggregation、prototype update、final scoring。 |
| 2 | 第 145 行写 “最终英文投稿版需要...” 是内部状态语，不应出现在投稿正文。 | MINOR | 改为正式实现说明，把未闭合细节放到投稿前 gate。 |
| 3 | Conservative update 的 confidence gate 只描述概念，缺少失败时如何回退。 | MAJOR | 明确 evidence confidence below threshold 时保留原 prototype，避免 reviewer 质疑 drift。 |

## Dimension 3: Experimental Strength

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 五数据集主结果强于固定 baseline，但缺少已核验外部方法 comparison，顶会竞争性不足。 | CRITICAL | 同 Blocking Issue 1。 |
| 2 | 当前结果是 deterministic report，不能报告显著性；稿件已正确避免显著性 claim。 | MINOR | 保持当前写法；若补多 seed，再新增 uncertainty table。 |
| 3 | 受控机制消融只覆盖 MVTec/VisA，范围清楚但略窄。 | MAJOR | 若时间允许，在 DTD-Synthetic 上补一组 controlled WPTA ablation；若不补，继续把机制 claim 限定为 MVTec/VisA。 |

## Dimension 4: Figure and Table Quality

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | `outputs/wpta_generated_tables_v0.5.md` 已经给出可用主文表和附录表，表注证据边界清楚。 | pass | 在 LaTeX 稿中直接引入 `outputs/wpta_generated_tables_latex_v0.5.tex`。 |
| 2 | Figure 1 有真实候选资产，但尚未完成投稿级排版。 | MAJOR | 使用 `outputs/wpta_figure_asset_ledger_v0.1.md` 中的候选 PNG 生成 PDF/SVG。 |
| 3 | Figure 3 仍是 prompt，是当前视觉证据最大缺口。 | CRITICAL | 同 Blocking Issue 2。 |

## Dimension 5: Citation and Claim Audit

| Claim | Verdict | Evidence used | Missing evidence | Suggested wording |
|---|---|---|---|---|
| Final calibrated system improves fixed AnomalyCLIP baseline on five industrial datasets. | keep | Table 1 and Table 4 | multi-run uncertainty absent | “The final calibrated system improves the fixed AnomalyCLIP baseline on all five industrial benchmarks under the current deterministic evaluation.” |
| WPTA mechanism is validated across five datasets. | remove | Table 4 contradicts this | controlled ablations on MPDD/BTAD/DTD absent | “WPTA mechanism is isolated by controlled MVTec/VisA ablations.” |
| Direct wavelet fusion is harmful or insufficient. | keep | Table 2/3 direct fusion underperforms | none for MVTec/VisA scope | “Direct final-map wavelet fusion is a negative control in MVTec/VisA.” |
| Current system is SOTA. | remove | protocol-reference external table only | fair verified external comparison | Do not use. |
| Medical generalization is demonstrated. | remove/weaken | ISIC/ISBI only | other medical datasets blocked | “A preliminary ISIC/ISBI appendix result suggests the calibration stack can transfer beyond industrial imagery, but this is not a main claim.” |

## Top 3 Polish Issues

| # | Issue | Suggested fix |
|---|---|---|
| 1 | Mixed English/Chinese terms are acceptable for internal review, but final Chinese draft could define terms once and keep them stable. | Add a short notation table for `S0`, `W`, `v_a`, `v_n`, `t'_a`, `t'_n`. |
| 2 | Section 5.5 describes appendix tables but does not inline them in the paper draft. | Add Appendix A/B sections or explicitly link to the generated table files. |
| 3 | Current title is descriptive but slightly long. | Candidate: “Boundary-Aware Wavelet Reliability for Test-Time Prototype Adaptation in Zero-Shot Anomaly Detection.” |

## Integrity Gate Result

- Structure complete: pass
- Claim-evidence alignment: pass with known blocked claims explicitly marked
- Figure readiness: fail
- Citation readiness: fail
- External comparison readiness: fail
- Reproducibility readiness: needs revision

## Next Revision Targets

1. Produce `v0.9` with an algorithm box, stronger contribution boundary, closest-work comparison dimension, and appendix table integration.
2. Generate Figure 1 from existing real mechanism assets.
3. Generate Figure 3 from real outputs across at least four industrial datasets.
4. Verify local-only citation metadata and update `.bib`.
5. Complete or remove external method comparison from the main paper path.

# WPTA Round 9 投稿前自审

审阅对象：`outputs/wpta_cvpr_paper_draft_zh_v1.4.md`

审阅标准：按 CVPR/ICCV 类视觉顶会审稿标准评估。中文只作为内部审阅语言，不降低引用、证据、图表、实验公平性和可复现性要求。

## Summary

- CRITICAL: 2
- MAJOR: 6
- MINOR: 4
- 评分：8.1 / 10
- 投稿建议：Needs major revision before submission

前三个必须优先修的问题：

1. 外部方法比较与论文定位还没有关闭。当前稿件明确承认外部比较尚未完成协议核验，第 216 行给出二选一策略，但还没有真正执行其中任何一条。
2. 引用完整性还没有达到投稿要求。第 198、321、348 行均说明部分条目仍需 canonical metadata 核验，最终 `.bib` 尚未形成。
3. Figure 3 真实跨数据集定性图仍缺失。第 301 至 307 行和第 351 行只给出计划，不能支撑跨数据集定性结论。

## Dimension 1: Macro Logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 第 216 行写道：“正式版本需要二选一：要么完成外部方法 protocol verification... 要么把论文定位为 AnomalyCLIP inference-stack calibration 与 WPTA 机制研究”。当前稿件仍停在策略声明，没有完成外部协议核验，也没有把标题、摘要、贡献和实验主表完全改造成机制研究论文。 | CRITICAL | 下一版必须选择投稿定位。若目标仍是 CVPR/ICCV 主会，优先完成同协议外部比较；若短期无法完成，标题和贡献需要明确降到 inference-stack calibration study，并把外部比较移出主文。 |
| 2 | 第 29 至 31 行已经很好地区分系统级证据与机制级证据，但第 222 行“最终校准系统在所有数据集上均超过固定 AnomalyCLIP baseline”仍容易被读成完整 benchmark claim。 | MAJOR | 在主结果段首加入“相对固定 baseline、非外部方法排名”的限定，并把 Table 4 作为解释 Table 1 的必读表。 |
| 3 | 第 35 至 38 行四条贡献与实验验证基本能对齐，但稿件缺少贡献到实验的矩阵。 | MAJOR | 在 Claim-evidence map 前增加 contribution-validation matrix，逐条写明贡献、方法位置、实验表格和禁止外推边界。 |

## Dimension 2: Writing Details

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 第 9 行和第 293 行仍引用 `outputs/wpta_generated_tables_v0.9.md`、`outputs/wpta_generated_tables_latex_v0.9.tex` 和 `outputs/wpta_generated_tables_manifest_v0.9.md`，但当前可用表格包已经是 v1.0。 | MAJOR | 在下一版稿件中统一更新到 v1.0，并在 QA 中扫描旧版本引用。 |
| 2 | 第 13 行摘要承担了问题、方法、消融、五数据集结果和证据边界，信息密度过高。 | MINOR | 投稿英文版应拆成 5 句摘要公式：problem, limitation, challenge, method, result/scope。中文内部版可保留但建议压缩。 |
| 3 | 第 42 至 53 行把 Figure 1 的资产清单放在引言正文后，属于内部工程记录，不适合投稿正文。 | MAJOR | 投稿版应只保留图题、图注和必要溯源，资产路径移到 artifact ledger 或 appendix build note。 |
| 4 | 第 356 至 382 行是图表生成 prompt，已经标注为非投稿正文，但仍位于稿件同一文件中。 | MAJOR | 最终投稿版必须删除该节，或转移到单独任务文件。当前内部稿可保留。 |

## Dimension 3: Citation And Integrity

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 第 198 行说明 `zou2022visa`、`jezek2021deep`、`mishra2021vt`、`aota2023zero` 仍来自本地 `.bib`，第 321 和 348 行重复说明最终 canonical metadata 未完成。 | CRITICAL | 按 citation-verification 规则逐项用 DOI、arXiv、CrossRef、Semantic Scholar 或 publisher page 核验，并生成最终投稿 `.bib`。不能从记忆补 BibTeX。 |
| 2 | 第 59 至 67 行相关工作覆盖 CLIP-based ZSAD、TTA 和小波基础，但尚未形成“为什么最近方法仍没有解决测试图像局部证据校准”的强对比。 | MAJOR | 在不增加未核验引用的前提下，强化 closest-work 表格的文字解释。若加入新文献，必须先完成 citation ledger。 |
| 3 | 第 190 行仍含本地绝对路径 `/Users/bytedance/code/...`。 | MAJOR | 投稿版方法和 appendix 应转写为匿名可复现实验配置，不出现本地绝对路径。 |

## Dimension 4: Experiments And Evidence

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 第 301 至 307 行写明 Figure 3 尚未生成。没有真实跨数据集 qualitative grid 时，结果部分不能声称跨数据集定性改善。 | MAJOR | 生成 Figure 3 后再加入定性分析；生成前只保留 Figure 1 单例机制说明。 |
| 2 | 第 208 行说明当前是 deterministic report，没有多 seed、置信区间或显著性检验。 | MAJOR | 若保留 deterministic-only，所有显著性、鲁棒性和统计置信表述都必须删除。若要增强投稿竞争力，需要补多次运行或至少类别级 breakdown。 |
| 3 | 第 257 和 273 行的消融解释是可信的，但 Table 2/3 只覆盖 MVTec/VisA。 | MINOR | 继续保持机制 claim 的 MVTec/VisA 范围，或补充更多数据集上的受控消融后再扩大 claim。 |

## Dimension 5: Figures And LaTeX

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 第 321、349、350 行说明 Figure 1 和 Figure 2 尚未完成 LaTeX 插入后的缩放 QA。 | MAJOR | 在目标 CVPR/ICCV 模板中插入 PDF，检查双栏缩放、字体、箭头、legend 和 caption。 |
| 2 | 第 40 至 53 行 Figure 1 只支撑 MVTec cable 单例机制说明。 | MINOR | 图注继续明确“单例机制解释”，不要把它写成跨数据集定性证据。 |
| 3 | 当前表格已经有 v1.0 LaTeX 版本，但 v1.4 稿件没有引用新表格包。 | MINOR | v1.5 更新表格包引用，并保持主表 1 至 4、附录表 A1/A2、协议参考表 B1a/B1b 的边界。 |

## Banned-Vocabulary And Mechanical Scan

- 已扫描 v1.4：未发现 em dash。
- 已扫描 v1.4：未发现本轮禁用的两项术语。
- 已扫描 v1.4：未发现 `expected_pass`。
- 发现旧表格引用：第 9 行和第 293 行仍指向 v0.9 表格包。
- 未发现英文强排名短语，但第 216 行表明外部方法协议核验仍未关闭。

## Claim Audit

| Claim | Verdict | Evidence used | Suggested wording |
|---|---|---|---|
| Final calibrated system improves fixed AnomalyCLIP baseline on five industrial datasets. | keep with scope | Table 1 and Table 4 | “最终校准系统相对固定 AnomalyCLIP baseline 在五个工业基准上取得系统级提升。” |
| WPTA mechanism is validated across all five industrial datasets. | remove | Table 4 contradicts this broad scope | “WPTA 机制由 MVTec/VisA 受控消融支撑，五数据集结果支撑 final calibrated system。” |
| Direct wavelet fusion is a negative control. | keep | Table 2 and Table 3 | “Direct wavelet fusion 在受控消融中低于固定 baseline 或 WPTA，说明小波不适合作为最终图直接加性分数。” |
| Current system is externally best among prior methods. | remove | External protocol not verified | “外部方法表仅作为 protocol-reference，不能用于主文排名。” |

## Integrity Gate Result

- Gate 1 specific findings quote real text or line numbers: pass
- Gate 2 every CRITICAL item has a concrete fix: pass
- Gate 3 no fabricated quotes: pass
- Gate 4 severity follows taxonomy: pass
- Gate 5 grammar review not applied because current draft is Chinese internal review稿: not applicable
- Gate 6 full banned-vocabulary and forbidden-term scan run on v1.4: pass
- Gate 7 score matches CRITICAL and MAJOR count: pass

## Revision Plan For v1.5

1. 更新所有表格包引用到 v1.0。
2. 把第 216 行的“当前 v1.4”改为当前版本表述。
3. 增加 contribution-validation matrix，把贡献、方法、实验和 claim boundary 对齐。
4. 弱化任何可能暗示统计稳定性的措辞，继续保持 deterministic-only 证据边界。
5. 重新扫描 v1.5 的旧表格引用、禁用术语、强排名短语和非投稿 prompt 位置。

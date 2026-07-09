# WPTA 中文顶会稿自审 round8

审阅对象：`outputs/wpta_cvpr_paper_draft_zh_v1.2.md`

关联资产：

- 表格包：`outputs/wpta_generated_tables_v0.8.md`
- LaTeX 表格包：`outputs/wpta_generated_tables_latex_v0.8.tex`
- Figure 2 资产记录：`outputs/figure2_caption_and_provenance_v0.1.md`

审阅目标：检查 v1.2 是否相对 v1.1 更接近 CVPR/ICCV 投稿要求。中文仅用于内部审阅，不降低 claim-evidence、图表、引用和实验完整性要求。

## Summary

- CRITICAL: 3
- MAJOR: 3
- MINOR: 4
- 当前评分：7.8 / 10
- 投稿建议：not ready for submission；可作为稳定内部审阅稿和下一轮投稿资产生成输入。

Top three blocking fixes:

1. 将 Figure 1 重绘为投稿级矢量排版，并生成真实 Figure 3 跨数据集 qualitative grid。
2. 完成 external protocol verification；否则正式投稿版必须继续把外部方法比较留在附录或删除。
3. 完成 canonical citation metadata 核验，尤其是 local-bib-only 数据集引用。

## 已完成的 v1.2 修订

| Item | Status | Evidence |
|---|---|---|
| 表格版本统一到 v0.8 | pass | v1.2 状态页和 5.5 均引用 `outputs/wpta_generated_tables_v0.8.md` 与 LaTeX v0.8，见 v1.2 第 9 行和第 287 行。 |
| Figure 2 真实资产纳入正文 | pass | v1.2 第 109 行给出 Figure 2 caption、资产路径和 LaTeX 插入后 QA 要求。 |
| Figure 2 provenance 已补齐 | pass | `outputs/figure2_caption_and_provenance_v0.1.md` 记录资产、caption、元素定义、生成脚本和 QA。 |
| Figure 2 不再作为待生成 prompt | pass | v1.2 第 358 行只保留已生成资产说明；第 10 节中不再出现 Figure 2 生成 prompt。 |
| 投稿 gate 更新 | pass | v1.2 第 344 行将 Figure 2 改为插入后缩放 QA，而不是生成 blocker。 |

## Dimension 1: Macro Logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | v1.2 的主线仍清楚：Table 1 是 final calibrated system 的系统级结果，Table 2/3 是 MVTec/VisA controlled mechanism evidence，Table 4 是归因边界。 | - | 保持三层证据结构。 |
| 2 | Figure 2 已闭合方法总览缺口，使 Method section 更像完整投稿稿件。 | - | LaTeX 插入后做缩放 QA，并保持模块名与 3.1 到 3.6 一致。 |
| 3 | 外部方法比较仍缺协议核验，当前无法支撑 reviewer 期待的 strongest-baseline 主表。 | CRITICAL | 按第 10 节 protocol verification prompt 生成可审计表，再决定进入主文或附录。 |

## Dimension 2: Claim-Evidence Alignment

| # | Claim | Verdict | Evidence / Risk |
|---|---|---|---|
| 1 | Final calibrated system improves fixed AnomalyCLIP baseline on five industrial datasets. | keep | Table 1 第 220 到 233 行给出五数据集与平均提升；Table 4 第 275 到 281 行记录 final settings。 |
| 2 | WPTA mechanism is supported by controlled MVTec/VisA ablations. | keep with scope | Table 2 第 243 到 249 行和 Table 3 第 259 到 265 行支撑；v1.2 第 31 行明确不扩展到五数据集机制 claim。 |
| 3 | Direct wavelet fusion is not the right role for wavelet cues. | keep | Table 2 第 246 行和 Table 3 第 262 行给出负对照；摘要第 13 行和讨论第 307 行表述与证据一致。 |
| 4 | External ranking against all listed methods. | forbidden internal claim | v1.2 第 338 行明确该 claim 禁止，B1a/B1b 只做 protocol-reference。 |
| 5 | Medical result supports general cross-domain conclusion. | remove from main claim | 只有 ISIC/ISBI 初步结果；v1.2 第 331 行仅标为 appendix-only observation。 |

## Dimension 3: Experiments

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 当前结果仍是 deterministic report，没有多 run 或 multi seed。 | MAJOR | 不报告显著性；若需要置信区间，补充多次独立运行。 |
| 2 | 类别级 breakdown 和失败案例仍缺。 | MAJOR | 生成 classwise breakdown table，并为 Figure 3 或附录增加 failure-case rows。 |
| 3 | MPDD/BTAD 的 final gains 没有被错误归因给 WPTA。 | - | 保持 Table 4 和第 283 行的边界表述。 |
| 4 | 医学结果仍不完整。 | MINOR | 维持 appendix-only；除非补齐可比评估，否则不要进入摘要和贡献。 |

## Dimension 4: Figures And Tables

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Table v0.8 覆盖当前能安全生成的主文和附录表，且证据边界写清楚。 | - | 后续正文继续引用 v0.8；若新增数据，生成 v0.9 而不是覆盖旧文件。 |
| 2 | Figure 1 仍是 raster 内容封装的 PDF candidate，不能作为最终投稿矢量图。 | CRITICAL | 使用现有 panel crops 重绘矢量文字、箭头、图例和排版，热图像素内容保持不变。 |
| 3 | Figure 2 已有 PDF/SVG/PNG 资产，初步 QA 通过。 | MINOR | 插入 LaTeX 后复查缩放字体、箭头间距和 caption 第一句。 |
| 4 | Figure 3 仍未生成，跨数据集 qualitative claim 仍被禁用。 | CRITICAL | 从真实模型输出生成至少四个工业数据集的 qualitative grid。 |

## Dimension 5: Citations And Integrity

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 正文没有未核验引用占位。 | - | 保持 citation ledger 与 `.bib` 同步。 |
| 2 | `zou2022visa`、`jezek2021deep`、`mishra2021vt`、`aota2023zero` 仍需 canonical metadata 核验。 | MAJOR | 用 arXiv、DOI、CrossRef、Semantic Scholar 或 publisher page 核验 title、authors、year、venue 和 identifier。 |
| 3 | Closest-work 表格需要逐行 claim-to-source alignment。 | MAJOR | 对每行训练条件、测试时适配、小波可靠性描述添加来源证据或改成更弱表述。 |

## Mechanical QA

- 禁用项 scan：pass。
- 旧表格版本 scan：pass。
- Figure 2 待生成 prompt scan：pass。
- 过强排名词 scan：pass。
- 未核验引用占位 scan：pass。
- AI-tone / em dash scan：pass。
- Figure 2 asset type check：pass，PDF 1 page，SVG vector，PNG preview `1774 x 751`。
- Figure 2 embedded-bitmap scan：pass，SVG 和脚本中未发现 `<image`、`base64` 或 `data:image`。
- Prompt placement：`FIGURE_PROMPT` 和 `TABLE_DATA_PROMPT` 只出现在第 10 节，且 Figure 2 不再以 prompt 形式出现。

## Integrity Gate Result

- Gate 1, specific quoted or line-based findings: pass。
- Gate 2, every CRITICAL finding has concrete fix: pass。
- Gate 3, no fabricated quotes: pass。
- Gate 4, severity follows current blocker risk: pass。
- Gate 5, citation uncertainty is explicitly marked: pass。
- Gate 6, full banned-word scan on reviewed files: pass。
- Gate 7, final score consistent with unresolved CRITICAL and MAJOR items: pass。

## Final Verdict

v1.2 相比 v1.1 有实质进展：Figure 2 方法总览资产和 provenance 已补齐，表格包升级到 v0.8，正文不再引用旧表格版本，也不再保留 Figure 2 待生成 prompt。当前文本的 claim-evidence 边界已经较稳，能作为内部审阅和后续英文 LaTeX 化的基础。

稿件仍未达到投稿水平。主要阻塞项是 Figure 1 最终矢量化、Figure 3 真实跨数据集定性图、外部方法协议核验、canonical citation metadata、以及可选的类别级结果和失败案例。当前评分 7.8/10；完成这些资产和证据 gate 后，才适合进入 8.5 分以上的投稿前精修阶段。

# WPTA 中文顶会稿 v0.3 预投稿评审 Round 2

评审对象：`outputs/wpta_cvpr_paper_draft_zh_v0.3.md`

## 总分

| 维度 | 分数 | 较 v0.2 变化 |
|---|---:|---|
| Originality | 7.6 / 10 | 持平略升，核心定位更清楚。 |
| Methodological rigor | 7.3 / 10 | 增加 implementation details checklist，复现风险被显式管理。 |
| Evidence sufficiency | 7.6 / 10 | 五数据集主结果与受控消融分层更清楚。 |
| Experiment completeness | 7.1 / 10 | 外部对比和医学结果被正确降级，主证据链更干净。 |
| Writing and presentation | 7.8 / 10 | 摘要收紧，running example 加入，引言更像顶会方法论文。 |
| Overall | 7.5 / 10 | 已达到 strong internal draft，但还未达到 submission-ready。 |

## 当前状态

v0.3 已经解决 v0.2 的主要可写作修订问题：

- 摘要改成更接近五句式，不再堆叠过多实现细节。
- 引言加入“金属件边缘附近细划痕”的 running example。
- 主结果扩展并统一到 5 个工业数据集 current CSV。
- 主结果 setting 与 controlled ablation setting 已明确区分。
- 外部方法对比降级为 protocol-reference，不再支撑 SOTA。
- 医学 ISIC/ISBI 结果降级为 appendix-style observation。
- Claim-evidence map 已避免把弱证据写成强 claim。

## Remaining Submission Blockers

| # | Blocker | Why it blocks submission | Required evidence |
|---|---|---|---|
| B1 | 引用仍是 `[CITATION-NEEDED]` | 顶会投稿不能包含占位引用；Related Work 需要真实、可核验、准确的文献。 | BibTeX 文件、正文 `\cite{}`、至少 CLIP/AnomalyCLIP/WinCLIP/AdaCLIP/CLIP-AD/TTA/wavelet inspection/datasets 的真实引用。 |
| B2 | 定性图仍是 prompt | 方法机制需要视觉证据，尤其是 `W`、selected evidence patches、final map 的对应关系。 | 至少 4 个工业数据集的真实 Input/GT/baseline/W/evidence/final map。 |
| B3 | 外部协议未核验 | 如果主文出现外部方法对比，审稿人会要求 split/backbone/input/eval 一致性。 | 协议核验表，或删除 Table 4 并只保留 fixed-baseline claim。 |
| B4 | 实现细节仍是 checklist | 方法可复现性不足。 | 具体超参数、层选择、top-k/threshold、aggregation、confidence gate、per-dataset setting。 |

## What Can Still Be Improved Without New Experiments

1. 把 `3.7 Implementation details to report` 从 checklist 改成正式 “Implementation Details” 小节，前提是能从代码或日志读取真实设置。
2. 将 Table 4 移到附录区或删除，以进一步降低协议风险。
3. 给每张表配更短、更 CVPR 风格的 caption。
4. 把 Method 小节标题全中文或全英文统一。当前中英混合用于内部审阅可以接受，英文投稿版需要统一。

## Verdict

Major Revision before final submission.

该稿已经可以给合作者或导师做完整技术审阅；但还不能声称“达到投稿水平”。若要达到投稿水平，最小闭环是：真实引用 + 真实 qualitative figure + 可复现 implementation details。外部方法对比如果无法核验协议，应从主文删除或移入附录。

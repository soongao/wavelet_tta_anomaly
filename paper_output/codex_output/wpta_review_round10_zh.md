# WPTA Round 10 自审

审阅对象：`outputs/wpta_cvpr_paper_draft_zh_v1.6.md`

审阅标准：按 CVPR/ICCV 类视觉顶会论文正文标准检查。中文仅用于内部审阅，不降低实验、公平比较、引用和可复现性要求。

## Summary

- CRITICAL: 1
- MAJOR: 4
- MINOR: 4
- 评分：8.4 / 10
- 投稿建议：仍需 major revision，但 v1.6 已明显更像论文正文，不再像工程交接文档。

## 已解决问题

1. 已删除正文中的工程资产清单、本地绝对路径、图表生成 prompt、投稿前 checklist 和任务单式内容。
2. 已移除用户指定删减项。扫描 v1.6 未发现相关中文或英文主线。
3. 已把 Figure 1/2 的资产路径描述改成正文中的图意说明，不再暴露文件路径或生成过程。
4. 已把五数据集结果与 WPTA 机制消融拆开表述，避免把 MPDD/BTAD 的系统收益错误归因到 wavelet reliability。
5. 已把外部方法比较降为 scope limitation，不再把未对齐协议的外部数值写成主文强比较。

## 主要风险

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 主文仍缺少同协议外部强基线比较。当前只比较固定 AnomalyCLIP baseline 与内部 variants，顶会审稿人很可能认为实验竞争力不足。 | CRITICAL | 完成外部方法协议对齐后加入主表；若做不到，需要把论文定位更明确地写成 inference calibration / mechanism study，并在标题、摘要和实验组织上进一步弱化 benchmark paper 期待。 |
| 2 | 五数据集主表的最终系统使用数据集特定配置。虽然 Table 4 已说明边界，但 Table 1 仍容易被读者当成统一方法在五数据集上的主结果。 | MAJOR | 将 Table 1 标题或 Method 名改成 `Dataset-specific calibrated inference system`，并在表注第一句说明不是同一模块组合。 |
| 3 | WPTA 机制只在 MVTec/VisA 有受控消融。若摘要同时出现五数据集平均提升，读者可能期待 WPTA 在五个数据集都有因果验证。 | MAJOR | 摘要中保留两个结论的拆分，或进一步减少五数据集平均提升在摘要中的权重。最好补 MPDD/BTAD/DTD-Synthetic 的同配置受控消融。 |
| 4 | 引用仍是草稿级，数据集和近期方法引用需要 canonical metadata 检查。 | MAJOR | 生成最终 `.bib`，逐项核验 DOI/arXiv/publisher 信息；不要从记忆补全引用。 |
| 5 | 缺少跨数据集 qualitative visualization。v1.6 只文字引用 Figure 1/2，不再保留 Figure 3 计划，但投稿版仍需要真实跨数据集定性图。 | MAJOR | 生成真实 qualitative grid，覆盖至少 MVTec/VisA/MPDD/BTAD/DTD-Synthetic 中四个数据集；MPDD/BTAD 不得标为 wavelet reliability。 |

## 次要问题

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | 摘要仍较长，包含问题、方法、受控消融、五数据集结果和证据边界，信息密度偏高。 | MINOR | 英文投稿版拆成 5 句：problem、gap、method、controlled evidence、system result/scope。 |
| 2 | 方法中保留 `alpha0=0.0`、`beta0=0.01`，这很诚实，但会让审稿人质疑 abnormal anchor 的实际作用。 | MINOR | 在消融或附录中增加 abnormal update strength sweep，或者在正文更明确说明 abnormal evidence 用于 confidence gate。 |
| 3 | Table 2/3 的 Baseline 与增强 variants 使用不同 multi-crop/pixel-to-image 条件，虽然表注已经说明，仍可能被质疑。 | MINOR | 最好补一个 shared-postprocess baseline，或在表中拆出 `Baseline + same postprocess` 行。 |
| 4 | 中英术语混排适合内部审阅，但英文投稿版需要统一术语，如 `prototype rectification`、`calibration`、`adaptation` 的边界。 | MINOR | 翻译时建立术语表，不要混用 rectification/calibration/adaptation 表达同一模块。 |

## Mechanical Scan

- 用户指定删减项: not found.
- 本地绝对路径: not found.
- 图表生成 prompt tags: not found.
- 强排名短语: not found.
- 旧表格包引用: not found.

## Claim Audit

| Claim | Verdict | Evidence | Comment |
|---|---|---|---|
| WPTA uses wavelet cues as evidence reliability rather than final-map scoring. | keep | Method 3.3/3.4, Table 2/3 direct fusion negative control | 机制清晰。 |
| WPTA mechanism is validated on MVTec/VisA controlled ablations. | keep | Table 2/3 | 范围必须保持 MVTec/VisA。 |
| Calibrated inference system improves fixed AnomalyCLIP baseline on five industrial datasets. | keep with scope | Table 1/4 | 必须保留 dataset-specific configuration 边界。 |
| The method ranks above external methods. | remove | Protocol not aligned | v1.6 未作此 claim。 |
| WPTA mechanism is validated across all five datasets. | remove | Table 4 contradicts broad claim | v1.6 已避免。 |

## 下一轮建议

1. 优先决定是否补外部同协议比较。如果不补，标题和摘要需进一步强调 mechanism / calibration study。
2. 补 shared-postprocess baseline 或把 Table 2 的 Baseline 条件完全对齐，以降低消融归因争议。
3. 生成 Figure 3 真实跨数据集定性图，再补 5.5 qualitative analysis。
4. 完成引用元数据核验和最终 `.bib`。

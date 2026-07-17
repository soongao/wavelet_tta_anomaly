# 论文图件中文说明

本文件说明 `paper_output/generated_result_figures_20260717/figures` 与 `paper/figures` 中每张图的含义、适合支撑的论文论点，以及写作时的使用建议。所有图件都由 `scripts/analysis/generate_paper_result_figures.py` 从 `paper/result_record/result_table.csv` 和 `paper/result_record/medical_result_table.csv` 自动生成。

## 总体原则

- PDF 适合直接放入 LaTeX 正文；SVG 适合后期在矢量编辑器里微调；PNG 适合快速预览或放入演示文稿。
- `_gray.png` 是灰度预览，用于检查打印或色弱读者场景下是否仍能区分图形元素。
- 当前数据是数据集级汇总结果，不是多次运行或多 seed 的重复实验。因此图中不画误差棒，也不标显著性星号。
- `expected_pass` 和 `blocked` 行没有进入图件；医学补充图只使用已经完成的 ISIC/ISBI current 结果。
- 柱状图主要展示相对增益，而不是截断后的绝对分数，目的是避免把 80-97% 的小差异视觉夸大。

## 图件清单

### Figure 1: `figure1_main_results`

**文件**：`figure1_main_results.pdf/svg/png`

**图中信息**：展示五个异常检测数据集上 baseline 与 full method 的主结果。左侧 panel 用配对点/线展示 Pixel AUPRO 的直接提升；右侧 panel 用热力图展示四个指标上的 full - baseline 增益。

**能支撑什么论点**：适合放在实验主结果部分，用来说明完整方法在多个数据集、多个指标上整体优于 AnomalyCLIP baseline，尤其突出定位指标 Pixel AUPRO 的稳定提升。

**适合怎么写**：可以配合主结果表格使用。图负责让读者快速看到“提升是否跨数据集一致”，表格负责给出所有精确数值。

**注意事项**：这张图不适合单独替代表格，因为审稿人通常仍需要完整的 benchmark 数值。

### Figure 2: `figure2_core_ablation`

**文件**：`figure2_core_ablation.pdf/svg/png`

**图中信息**：在 MVTec 和 VisA 上，对 baseline、直接小波融合、语义原型适配、无保守更新的小波原型适配、完整方法进行四个指标的消融对比。

**能支撑什么论点**：适合放在核心消融实验部分，用来证明提升不是来自某一个随意后处理，而是来自“原型适配 + 小波引导 + 保守更新”的组合。

**适合怎么写**：强调 direct wavelet fusion 作为负控，说明仅把小波图直接融合到 anomaly map 并不足够；真正有效的是让小波可靠性参与 patch evidence / prototype adaptation。

**注意事项**：这张图使用散点而不是连线，因为这些方法不是全部都构成严格连续路径。

### Figure 3: `figure3_wavelet_design_ablation`

**文件**：`figure3_wavelet_design_ablation.pdf/svg/png`

**图中信息**：以 semantic-only prototype adaptation 为参照，展示不同小波设计在 MVTec 和 VisA 的 Pixel AUPRO、Image AUROC 上带来的增益或下降。

**能支撑什么论点**：适合说明不是“任何小波特征都有效”，而是 boundary-aware wavelet reliability 与 conservative update 更符合异常定位需求。

**适合怎么写**：重点解释 direct fusion 为负控，HF-only 只能提供有限或不稳定收益，boundary-aware 版本才更接近最终方法。

**注意事项**：热力图适合快速比较正负方向和幅度，但如果正文需要更传统的展示方式，可以改用 Figure 6 的柱状图。

### Figure 4: `figure4_core_ablation_bar`

**文件**：`figure4_core_ablation_bar.pdf/svg/png`

**图中信息**：以 fixed AnomalyCLIP baseline 为零点，使用分组柱状图展示核心组件对 Pixel AUPRO 和 Image AUROC 的点数增益。

**能支撑什么论点**：适合在正文或补充材料中更直观地展示“每个组件相对 baseline 增益多少”。它比绝对分数柱状图更适合强调消融贡献。

**适合怎么写**：可以用于回答审稿人常问的“每个模块到底贡献多少”。在文中可描述：语义原型适配带来主要增益，小波引导和保守更新进一步补充提升。

**注意事项**：Direct fusion 出现负值或较弱提升时，不应回避；这正好支撑“简单小波后融合不是有效机制”的论点。

### Figure 5: `figure5_core_ablation_line`

**文件**：`figure5_core_ablation_line.pdf/svg/png`

**图中信息**：只沿着有明确构建顺序的路径连线：Baseline -> Semantic adaptation -> WPTA no conservative -> Full WPTA。

**能支撑什么论点**：适合展示方法逐步构建时性能如何变化，让读者看到完整方法不是孤立跳点，而是沿着设计路径逐步增强。

**适合怎么写**：可以用于方法消融段落的第二张图或补充图，解释“先适配原型，再引入小波可靠性，最后用保守更新稳定结果”的递进逻辑。

**注意事项**：折线只用于有顺序关系的组件累加路径。Direct fusion 没有放入这张折线图，因为它是负控分支，不是同一条递进路径。

### Figure 6: `figure6_wavelet_design_bar`

**文件**：`figure6_wavelet_design_bar.pdf/svg/png`

**图中信息**：以 semantic-only prototype adaptation 为零点，用分组柱状图展示 direct fusion、HF-only、boundary-aware、full boundary-aware 四种小波设计的增益。

**能支撑什么论点**：适合更传统地展示小波设计消融，尤其适合正文空间不足但需要让读者直接读出每个设计增益的情况。

**适合怎么写**：可用来说明 boundary-aware 设计相比 HF-only 更稳，full 版本在两个数据集上都能带来最终增益。

**注意事项**：如果已经在正文使用 Figure 3 热力图，Figure 6 更适合放补充材料，避免正文重复。

### Figure 7: `figure7_main_gain_bar`

**文件**：`figure7_main_gain_bar.pdf/svg/png`

**图中信息**：按数据集排序展示 full method 相比 baseline 在 Pixel AUPRO 和 Image AUROC 上的增益。

**能支撑什么论点**：适合补充说明方法的收益主要体现在哪些数据集和指标上。它能帮助读者快速看出 BTAD、MPDD、VisA 等数据集上的提升幅度差异。

**适合怎么写**：可放在主结果后，作为“增益分布”图；也可放在 rebuttal 或补充材料中回应“提升是否只来自单个数据集”的问题。

**注意事项**：这张图只展示两个 headline metrics，不展示所有指标；完整指标仍应参考 Figure 1 或结果表。

### Supplementary Figure: `supp_figure_medical_isbi`

**文件**：`supp_figure_medical_isbi.pdf/svg/png`

**图中信息**：展示 ISIC/ISBI 医学迁移实验中 baseline 与 full method 在 Pixel AUROC 和 Pixel AUPRO 上的对比。

**能支撑什么论点**：适合放在补充实验，说明方法在医学异常/病灶分割风格数据上也有一定迁移趋势。

**适合怎么写**：建议谨慎表述为 supplementary evidence 或 preliminary medical transfer result，不要写成大规模医学泛化结论。

**注意事项**：ColonDB、ClinicDB、Kvasir 等行在源表中是 blocked，没有完整 current 结果，因此没有绘制。

## 推荐放置顺序

正文主线建议：

1. `figure1_main_results`：主结果总览。
2. `figure2_core_ablation` 或 `figure4_core_ablation_bar`：核心消融。若正文偏结果解释，用 Figure 4；若正文需要完整四指标，用 Figure 2。
3. `figure3_wavelet_design_ablation` 或 `figure6_wavelet_design_bar`：小波设计消融。热力图更紧凑，柱状图更传统。

补充材料建议：

1. `figure5_core_ablation_line`：组件递进趋势。
2. `figure7_main_gain_bar`：主结果增益排序。
3. `supp_figure_medical_isbi`：医学迁移补充实验。

## 复现命令

```bash
python3 scripts/analysis/generate_paper_result_figures.py
```

运行后会更新：

- `paper_output/generated_result_figures_20260717/`
- `paper/figures/`


# 中文单栏顶会论文草稿

本目录用于沉淀一版以机制证据为核心、弱化代码命令细节的中文单栏顶会论文草稿。当前版本已经接入项目中已有的受控消融表格、消融图和正常图稳定性数据，用于把机制叙事和真实实验材料对齐。

## 当前叙事

本文把 CLIP 零样本异常定位重新表述为一个图像条件化的正常参照问题：

- 局部响应是候选线索；同样的响应在不同图像语境中对应缺陷、正常纹理或结构边界。
- CLIP 的局部表征中存在可用于异常定位的细粒度响应，这些响应需要由当前测试图像内部的可靠证据校准。
- 异常证据由该局部表征相对于图像条件化 normal/abnormal prototypes 的重评分定义。
- Direct wavelet fusion 是负控；semantic-only prototype adaptation 是强对照；boundary-aware reliability 与受门控的小步 normal-side calibration 构成当前稳定方法路径。

## 已接入的实验材料

- 受控核心消融表：`../paper/tables_ablation/controlled_core_ablation.tex`
- 局部可靠性设计消融表：`../paper/tables_ablation/wavelet_design_ablation.tex`
- 受控核心消融图：`../paper/figures_ablation/figures/controlled_core_ablation_progression_lines.pdf`
- 局部可靠性设计消融图：`../paper/figures_ablation/figures/wavelet_design_ablation_progression_lines.pdf`
- 正常图稳定性数据：写入 `sections/05_mechanism_tests.tex`

## 文件结构

- `main.tex`: 单栏中文 LaTeX 主文件。
- `sections/`: 正文章节。
- `literature_learning_notes.md`: 已核验文献学习笔记和本文写作边界。
- `references.bib`: 已核验参考文献条目；新增引用需要继续从 DOI、arXiv、DBLP、CrossRef 或原论文页面核验。
- `figures/`: 后续可放机制示意图或定性案例图。
- `tables/`: 后续可放本稿专用表格；当前优先复用 `../paper/tables_ablation/` 中已生成表格。

## 编译

推荐使用 XeLaTeX 或 LuaLaTeX：

```bash
cd topconf_paper_cn_single_column
latexmk -xelatex -interaction=nonstopmode main.tex
```

如果本机没有 `latexmk`：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## 写作原则

- 论文创新点围绕机制定义、证据选择和图像条件化参照展开。
- 方法命名统一使用 `Ours (unnamed)` 或 `\method{}`。
- 频率线索作为候选证据来源进入证据选择机制。
- Direct fusion 作为机制负控，最终方法使用局部线索进行 evidence selection 和 prototype calibration。
- System-level 五数据集结果用于展示整体表现；controlled MVTec/VisA 消融用于支撑机制结论。
- 新增测试时校准模块的结论限定在冻结底座表征之上；底座 CLIP/AnomalyCLIP 的预训练和辅助数据来源单独说明。
- 引用保持核验流程；新增 BibTeX 从 DOI、arXiv、DBLP、CrossRef 或原论文页面获取。

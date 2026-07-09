# Figure 1 vector-layout caption and provenance v0.1

## Asset

- Source strip: `/Users/bytedance/code/AnomalyCLIP/cached_results/prototype_tuned/mechanism_viz/mvtec_cable_000083_cable.png`
- Source panel crops: `outputs/figures/figure1_panels/`
- Generation script: `outputs/scripts/make_figure1_vector_layout.py`
- Generated PDF candidate: `outputs/figures/figure1_motivated_example_mvtec_cable_vector_layout.pdf`
- Generated SVG candidate: `outputs/figures/figure1_motivated_example_mvtec_cable_vector_layout.svg`
- Generated PNG preview: `outputs/figures/figure1_motivated_example_mvtec_cable_vector_layout.png`
- Earlier raster-layout ledger: `outputs/figure1_caption_and_provenance_v0.1.md`

## Caption draft

**图 1. 小波线索适合作为原型适配证据的可靠性监督，而不是直接替代语义异常分数。** 该 MVTec cable 样例展示了输入图像、目标缺陷区域、固定原型异常图、直接小波线索、boundary-aware reliability、selected evidence 与 WPTA final map 的对应关系。直接使用小波响应容易激活结构边界或局部高频纹理；WPTA 将小波线索限制在 evidence selection 层面，并用语义分数 `S0` 与 reliability `W` 共同选择 visual anchors，最终异常图仍由校准后的 CLIP prototypes 产生。

## Provenance

该图来自真实模型输出。七个 panel 的 raster 像素内容保持不变，包括输入图像、目标区域、固定原型异常图、direct wavelet cue、boundary-aware reliability、selected evidence 和 WPTA final map。当前脚本只重绘论文排版元素：标题、panel labels、边框、箭头、legend 和底部机制说明。

由于 panel 本身是模型输出 raster，SVG 中预期包含 base64-encoded `<image>` 元素。该现象不表示热图被重新绘制；它只说明真实模型输出 panel 以 raster 方式嵌入。标题、标签、legend、箭头和说明文字由 Matplotlib 以矢量元素输出。

## Claim boundary

- 可支撑：MVTec cable 单例机制解释，展示 direct wavelet cue、boundary-aware reliability、selected evidence 与 WPTA final map 的关系。
- 不可支撑：跨数据集 qualitative conclusion。
- 不可支撑：五数据集 WPTA 机制因果验证。
- 不可替代：Figure 3 qualitative grid。Figure 3 仍需覆盖多个工业数据集，并且必须按 dataset-specific final modules 正确标注 evidence 类型。

## Asset QA

- PDF: single page, `509.76 x 339.84 pts`, approximately `7.08 x 4.72 in`.
- PNG preview: `2124 x 1416`, RGBA.
- SVG annotation text: present as `<text>` elements.
- SVG embedded raster panels: present as `<image ... base64>` elements, expected because source panels are real model outputs.
- Font scan: no SVG text font below `8px` was detected.
- Visual layout: two-row `4 + 3` panel arrangement to avoid unreadable seven-panel compression in a double-column figure.

## Submission QA

- 当前状态：可作为 Figure 1 投稿排版候选，用于中文审阅稿和后续 LaTeX 插入测试。
- 投稿前必做：插入目标 LaTeX 模板后复查双栏缩放下的 panel 标题、legend、底部说明和 caption 第一行是否可读；必要时改成 `figure*` 或拆为 teaser + appendix detail。
- 投稿前不得声称：该单例代表所有数据集，或该图本身证明跨数据集机制成立。

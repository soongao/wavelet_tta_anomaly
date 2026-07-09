# Figure 1 caption and provenance v0.1

## Asset

- Source image: `/Users/bytedance/code/AnomalyCLIP/cached_results/prototype_tuned/mechanism_viz/mvtec_cable_000083_cable.png`
- Generation script: `outputs/scripts/make_figure1_motivated_example.py`
- Generated PNG preview: `outputs/figures/figure1_motivated_example_mvtec_cable.png`
- Generated PDF candidate: `outputs/figures/figure1_motivated_example_mvtec_cable.pdf`
- Panel crops: `outputs/figures/figure1_panels/`

## Caption draft

**图 1. 小波线索适合作为原型适配证据的可靠性监督，而不是直接替代语义异常分数。** 该 MVTec cable 样例展示了输入图像、目标缺陷区域、固定原型异常图、直接小波线索、boundary-aware reliability、selected evidence 与 WPTA final map 的对应关系。直接使用小波响应容易激活结构边界或局部高频纹理；WPTA 将小波线索限制在 evidence selection 层面，并用语义分数 `S0` 与 reliability `W` 共同选择 visual anchors，最终异常图仍由校准后的 CLIP prototypes 产生。

## Provenance

该图来自真实模型输出 strip，未手工绘制热图、reliability map、selected evidence 或 final map。当前脚本只做三类后处理：按原始 7-panel strip 裁切 panel、添加论文阅读标签、加入一行机制说明。由于源 panel 本身是 raster PNG，当前 PDF 也是 raster 内容封装的排版候选，不应被描述为最终矢量图。

## Claim boundary

- 可支撑：MVTec cable 单例的机制可视化，展示 direct wavelet cue、boundary-aware reliability、selected evidence 和 WPTA final map 的关系。
- 不可支撑：跨数据集 qualitative conclusion。
- 不可支撑：五数据集 WPTA 机制因果验证。
- 不可替代：Figure 3 qualitative grid。Figure 3 仍需覆盖多个工业数据集，并且必须按 dataset-specific final modules 正确标注 evidence 类型。

## Submission QA

- 当前状态：真实机制图候选可用于中文审阅稿。
- 投稿前必做：重新排版为 CVPR/ICCV 双栏可读版本，确认字体缩放后不小于 8pt；若使用 PDF，需说明或替换 raster panel 的来源；最终 caption 不应声称该单例代表所有数据集。

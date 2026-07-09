# WPTA Figure Asset Ledger v0.1

本文档记录当前可以用于 WPTA 论文图的真实可视化资产，以及仍然缺失的投稿级图像证据。原则：最终论文图中的模型热图、reliability map、selected evidence patches 和 final map 必须来自真实模型输出，不使用手工涂色或示意热图替代实验输出。

## 已发现真实机制可视化

目录：`/Users/bytedance/code/AnomalyCLIP/cached_results/prototype_tuned/mechanism_viz`

| Asset | Dataset | Class / sample | Size | 可用位置 | 当前用途边界 |
|---|---|---|---:|---|---|
| `mvtec_bottle_000000_bottle.png` | MVTec | bottle / 000000 | 3360 x 512 | Figure 1 candidate 或 appendix mechanism case | 只支撑 MVTec 机制展示 |
| `mvtec_cable_000083_cable.png` | MVTec | cable / 000083 | 3360 x 512 | Figure 1 candidate | 适合作为边界/细缺陷 motivated example 候选 |
| `mvtec_capsule_000233_capsule.png` | MVTec | capsule / 000233 | 3360 x 512 | Figure 1 candidate 或 appendix mechanism case | 只支撑 MVTec 机制展示 |
| `visa_candle_000100_candle.png` | VisA | candle / 000100 | 3360 x 512 | Figure 1 candidate 或 appendix mechanism case | 只支撑 VisA 机制展示 |
| `visa_capsules_000260_capsules.png` | VisA | capsules / 000260 | 3360 x 512 | Figure 1 candidate | 适合作为小目标/实例纹理缺陷候选 |
| `visa_cashew_000410_cashew.png` | VisA | cashew / 000410 | 3360 x 512 | Figure 1 candidate 或 appendix mechanism case | 只支撑 VisA 机制展示 |

## 可支撑的论文图

### Figure 1: Motivated example / mechanism example

当前已有 MVTec 和 VisA 的真实机制 PNG，可以作为 Figure 1 的候选素材。建议优先选择 `mvtec_cable_000083_cable.png` 或 `visa_capsules_000260_capsules.png`，因为这类样例更容易展示 fixed prototype mismatch、结构边界伪响应和 WPTA evidence selection 的差异。

图注第一句应明确核心发现：

> Direct wavelet fusion amplifies structure-boundary evidence, while WPTA uses boundary-aware wavelet reliability only to select prototype-adaptation evidence and keeps the final map semantic.

中文审阅稿可写为：

> 直接小波融合会放大结构边界伪响应，而 WPTA 只把 boundary-aware wavelet reliability 用于选择原型适配证据，最终异常图仍由语义原型决定。

### Figure 2: Method overview

Figure 2 可以是矢量示意图，不需要真实模型输出。必须标注：

- `CLIP frozen`
- `no target training images`
- `no backpropagation`
- `wavelet reliability, not final score`
- `S0 + W -> selected evidence -> visual anchors -> conservative prototype calibration`

### Figure 3: Qualitative results

当前机制 PNG 不能替代 Figure 3。投稿级 Figure 3 至少需要覆盖 MVTec、VisA、MPDD、BTAD、DTD-Synthetic 中四个工业数据集，并且每行必须来自真实模型输出。

最低列设计：

1. Input image
2. GT mask
3. AnomalyCLIP baseline map
4. Boundary-aware wavelet reliability 或 dataset-specific calibration evidence
5. Selected evidence patches
6. WPTA/final map

如果 MPDD/BTAD final run 没有启用 wavelet reliability，则图中不得标成 WPTA mechanism visualization，应改写为 final calibrated system qualitative 或只在 MVTec/VisA 展示机制图。

## 需要生成图的 Prompt

[FIGURE_PROMPT:figure1_from_real_assets]

请基于真实机制 PNG 资产生成 CVPR/ICCV 论文 Figure 1。候选输入目录为 `/Users/bytedance/code/AnomalyCLIP/cached_results/prototype_tuned/mechanism_viz`。优先选择 `mvtec_cable_000083_cable.png` 或 `visa_capsules_000260_capsules.png`。要求：保留真实模型输出，不手工涂热图；裁剪或重排为横向 4-6 panel；必须包含 Input、GT 或 defect marker、baseline/fixed prototype map、wavelet reliability 或 direct wavelet fusion failure、selected evidence patches、WPTA/final map。异常 evidence 用橙色边框，normal evidence 用蓝色边框；图中文字不小于 8pt；输出 PDF/SVG 优先，PNG 仅用于预览。图注第一句写明“wavelet is useful as evidence reliability rather than final-map fusion”。

[/FIGURE_PROMPT]

[FIGURE_PROMPT:figure3_real_qualitative_grid]

请从真实模型缓存和日志生成 Figure 3 qualitative grid。至少覆盖 MVTec、VisA、MPDD、BTAD、DTD-Synthetic 中四个工业数据集。每行一个样例，列为 Input image、GT mask、AnomalyCLIP baseline map、Calibration evidence map、Selected evidence patches、Final calibrated map。Baseline 与 final map 使用相同色标；evidence map 使用 viridis；所有热图必须从模型输出或缓存重算，不允许手工绘制。对于未启用 wavelet reliability 的 MPDD/BTAD，不得标为 WPTA reliability，只能标为 calibration evidence 或省略该列。输出 `figure3_qualitative.pdf`、`figure3_qualitative.png` 和样例 provenance CSV，字段包括 `dataset,class,image_id,baseline_log,final_log,cache_dir,enabled_modules`。

[/FIGURE_PROMPT]

## QA Gate

- [ ] Figure 1 使用真实模型输出，并记录 asset path。
- [ ] Figure 1 图注不声称五数据集机制验证。
- [ ] Figure 2 的模块名称与方法小节一致。
- [ ] Figure 3 覆盖至少四个工业数据集。
- [ ] Figure 3 对 MPDD/BTAD 的标签不误写为 wavelet reliability。
- [ ] 所有最终图中文字在双栏缩放后不小于 8pt。
- [ ] 最终提交使用 PDF/SVG 或高分辨率无损位图，避免低清截图。

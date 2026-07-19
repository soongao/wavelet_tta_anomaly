# 论文图示（figures/）

> 三张核心矢量图，遵循顶会图表规范：扁平矢量风、纯白背景、柔和淡色系、全英文标签、无长句、无 3D 阴影。
> 格式为 SVG（矢量，可无损缩放）。查看：浏览器直接打开；编辑：Figma / Illustrator / Inkscape。
> 入 LaTeX：用 Inkscape 导出 PDF（`inkscape fig.svg --export-type=pdf`）或 `\includesvg{}`（svg 宏包）。
> 本机无 svg→png 转换器，未生成 PNG；需要位图时用 `rsvg-convert fig.svg -o fig.png` 或 `cairosvg`。

## 配色语义（全篇统一）
| 颜色 | 模块类型 |
|---|---|
| 蓝 `#d6e4f0` | CLIP / 语义分支 |
| 绿 `#d9efe7` | 频率分支（读高频 HF） |
| 黄 `#fbebcb` | 证据选择 |
| 红/粉 `#fbe0da` | 逐图正常参照（核心模块，TTA） |
| 灰 `#eef1f4` | 输入 / 输出 |

---

## Fig 1 — Motivation（`fig1_motivation.svg`）
**放论文位置**：Introduction 首图 / teaser。
**讲什么**：核心 insight 的可视化。三行 ×两列：
- (a) 光滑物体+划痕 → 高频响应里**只有缺陷亮** → 检测正确。
- (b) 粗糙正常纹理 → 高频响应里**整片都亮** → 固定阈值会误报。
- 中间黄框点破："同样的高频幅度，在 (a) 是异常、在 (b) 是正常 → 固定阈值必失败"。
- 红框给解法："逐图估这张图自己的正常高频水平，只标超出的"。
**对应**：NARRATIVE §0 一句话 thesis；EXPERIMENT_PLAN_PAPER 表4/表5 的机制直觉。

## Fig 2 — Architecture（`fig2_architecture.svg`）
**放论文位置**：Method 主 pipeline 图（Fig 2）。
**数据流**：Test Image → CLIP Visual Encoder（frozen，带锁图标）→ Patch Features → 分两支：
- 语义支：normal/abnormal text prototypes → S0；
- 频率支：Haar DWT → LL(结构) / HF / `W = HF·(1−LF_edge)`；
两支汇入 **Evidence Select**（trustworthy-normal = low S0 & low W；abnormal-evidence = high S0 & high W）→ **Per-Image Normal Reference**（估 v_normal/v_abn，保守更新，无训练/无反传，带锁图标）→ 重算 → **Anomaly Map**（deviation）。
**强调**：红色"Per-Image Normal Reference"是核心模块，图注写 "normal reference is estimated per image"。
**对应**：NARRATIVE §2 方法四步。

## Fig 3 — Mechanism Ablation（`fig3_mechanism_ablation.svg`）
**放论文位置**：Experiments / Ablation，作为核心消融的可视化。
**讲什么**：四张卡片对应四个变体，回答"正常参照从哪来"：
- **DirectHF**（读高频无参照）→ pixel 崩盘（图示 HF 直连 Map，数值 82.5）；
- **GlobalRef**（全局固定参照）→ 仅略高于 baseline（一个阈值配不了所有材质，83.2）；
- **SelfRef**（逐图但只用 CLIP 自己）→ 强但触顶（自证闭环 self-loop 图示，84.6）；
- **Ours**（逐图 + 高频挑参照）→ 最好（CLIP+HF 汇入 per-image ref，85.8）。
- 底部单调链：`DirectHF 73.0 < Baseline 81.4 < GlobalRef 83.2 < SelfRef 84.6 < Ours 85.8`（pixel AUPRO）。
**注意**：图中数值为 EXPERIMENT_PLAN_PAPER 的**目标值**，做完实验须回填真实数值后再定稿。
**对应**：EXPERIMENT_PLAN_PAPER 表2。

---

## 待办 / 可扩展
- 做完实验后：把 Fig 3 的目标数值替换为真实 log 数值。
- 可选补图：分类别增益条形图（对应表4）、CLIP 盲区召回示意（对应表5）。
- 定稿前统一字号/字距，导出 PDF 供 LaTeX 嵌入。

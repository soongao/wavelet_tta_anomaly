# Experiment Plan

## 目标声明

实验要验证的不是“小波是否有用”，也不是“单图 reference 是否能后处理提分”，而是：

```text
图内正常性锚定能补足外部正常/异常原型无法定义局部正常性的缺口。
```

多尺度局部结构响应的实验角色是帮助发现 normal witnesses，并证明局部结构只有在不能被图内正常性解释时才应成为异常证据。

## 协议

- Setting：辅助数据 ZSAD。
- Backbone：CLIP ViT-B/16 或与 AnomalyCLIP/AdaCLIP 可比的设置。
- Source-target split：评估 MVTec AD 时可使用 VisA 或其他非目标源数据；评估 VisA 时使用 MVTec AD 或其他非目标源数据。每个目标数据集不得使用自身训练图像、正常参考图、异常标签或掩码。
- Inference：每次只使用当前无标签图像进行 normality grounding；CLIP、prompt、adapter 和归一化参数保持冻结。
- Metrics：`I-AUROC ↑`, `I-AP ↑`, `I-F1-max ↑`, `P-AUROC ↑`, `P-AP ↑`, `P-F1-max ↑`, `P-AUPRO ↑`。

## 主实验

Table 1：MVTec AD 与 VisA 主结果。

应包含：

- CLIP-style baseline
- WinCLIP
- APRIL-GAN
- AnomalyCLIP
- AdaCLIP
- VCP-CLIP
- FiLo
- FreqAnchorAD，如果代码或可复现实验设置可用
- ING

重点看 `P-AP` 与 `P-AUPRO`。只提升 `P-AUROC` 不足以支持图内正常性锚定对细粒度定位有效。

## 核心消融

Table 2：图内正常性锚定是否有效。

| Variant | 目的 |
|---|---|
| External semantic prior only | 外部正常/异常原型基线 |
| Direct structure fusion | 负控：直接把局部结构响应加到异常图 |
| Global source anchor | 对照：使用源域聚合正常锚点而不是图内锚点 |
| Semantic top-k witnesses | 不使用结构一致性的 normal witnesses |
| Structure-only witnesses | 不使用语义先验的 normal witnesses |
| ING full | 完整图内正常性锚定 |

预期因果关系：`ING full` 应明显优于 `Direct structure fusion` 和 `Semantic top-k witnesses`。如果直接结构融合接近完整方法，论文主线会退化成结构分数融合。

## Normal witness 消融

Table 3：正常见证发现机制。

- semantic confidence only
- semantic + neighborhood consistency
- semantic + structural consistency
- semantic + neighborhood + structural consistency
- hard top-k witnesses
- soft weighted witnesses

该表必须证明 normal witnesses 不是普通 top-k patch，而是图内正常性锚定所需的可靠证据。

## 结构一致性消融

Table 4：多尺度局部结构响应的作用边界。

- Haar response
- Sobel/edge response
- DCT local blocks
- no multiscale, single-scale only
- low/mid/high band removal
- image-space response vs feature-grid response

目的不是证明某个变换天然最优，而是证明结构一致性对 normal witness discovery 有贡献。

## 可视化

Figure 1：external prototype failure。

- 展示正常边界和真实缺陷都可能有强结构响应。
- 展示外部文本原型无法单独解释局部正常性。

Figure 2：method overview。

- 使用 Mermaid 草稿 `figures/ing_mechanism.mmd` 作为逻辑图源。

Figure 3：normal witness discovery。

- 原图、结构响应图、语义正常置信度、normal witnesses、图内偏离图、最终 anomaly map。

Figure 4：negative control maps。

- External semantic prior
- Direct structure fusion
- ING full
- Ground truth

Figure 5：grounding score distribution。

- 正常 patch 与异常 patch 在 semantic score、intra-image deviation、final score 上的分布变化。

## 效率实验

Table 5：资源开销。

- input resolution
- backbone
- extra inference time
- GPU memory
- number of normal witnesses / anchors
- parameter update flag: no

## 成功标准

Minimal success：

- 在 MVTec AD 和 VisA 上都超过最接近的 CLIP-ZSAD baseline。
- `Direct structure fusion` 低于 `ING full`。
- 移除 normal witness discovery 后，`P-AP` 或 `P-AUPRO` 明显下降。

Solid success：

- image-level 和 pixel-level 均无明显退化。
- normal witness 可视化能解释正常纹理处过检减少。
- 结构一致性消融显示它对 witness selection 有稳定贡献。

Strong success：

- 在额外工业数据集上保持收益，例如 BTAD、MPDD、KSDD2 或 DAGM。
- 多锚点版本能改善多纹理/多部件类别。
- 效率开销可接受，且不依赖目标数据集统计。

Failure signals：

- 只提升 `P-AUROC`，但 `P-AP`、`P-F1-max` 或 `P-AUPRO` 下降。
- 直接结构分数融合与完整方法差不多。
- normal witnesses 经常覆盖真实异常区域。
- 效果只出现在少数纹理类别，物体类别无收益。
- 方法隐式使用目标训练集统计或类别特定阈值。

## 第一批实验顺序

1. 先实现 `External semantic prior only`、`Direct structure fusion`、`Semantic top-k witnesses`、`ING full` 四个版本。
2. 在 MVTec AD 和 VisA 上跑 `P-AUROC/P-AP/P-AUPRO`。
3. 抽取每类 3 个样本做 normal witness discovery 可视化。
4. 再做 Haar/Sobel/DCT 与单尺度/多尺度消融。

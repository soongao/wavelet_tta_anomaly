# 实验结果与消融总结

更新时间：2026-06-22

本文档是当前实验阶段的结论版总结。完整命令、参数和日志索引见 `paper/results/experiment_summary.md`。

指标顺序统一为：

`pixel AUROC | pixel AUPRO | image AUROC | image AP`

## 1. 当前方法概述

当前方法是在 AnomalyCLIP zero-shot anomaly detection 框架上做无训练 test-time calibration / fusion。主要组件包括：

- 小波校准：基于 patch feature 生成结构/纹理相关的校准信号。
- Wavelet-guided TTA：用测试图像自身的高置信 patch 信息修正文本特征和 anomaly map。
- Multi-crop fusion：用局部裁剪视角补充 pixel-level anomaly map。
- Pixel-to-image fusion：用 pixel-level top-k 异常分数修正 image-level anomaly score。

当前实验没有额外训练模型，所有主结果均基于 cached feature evaluation。

## 2. 主结果

### 2.1 MVTec

| 方法 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| 原始 AnomalyCLIP | 91.1 | 81.4 | 91.6 | 96.4 |
| 当前完整方法 | 91.8 | 85.6 | 94.5 | 97.6 |
| 提升 | +0.7 | +4.2 | +2.9 | +1.2 |

### 2.2 VisA

| 方法 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| 原始 AnomalyCLIP | 95.5 | 86.7 | 82.0 | 85.3 |
| 当前完整方法 | 96.2 | 91.3 | 84.6 | 87.4 |
| 提升 | +0.7 | +4.6 | +2.6 | +2.1 |

主结果说明：当前方法在 MVTec 和 VisA 上同时提升 pixel-level 和 image-level 指标，其中 pixel AUPRO 的提升最稳定、幅度也最大。

## 3. 组件级消融

组件级消融日志：`ablation_results/20260622_094146_component/summary.md`

### 3.1 MVTec

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| original_anomalyclip | 91.1 | 81.4 | 91.6 | 96.4 |
| cached_baseline_l123 | 91.3 | 83.1 | 91.6 | 96.4 |
| wavelet_only | 91.3 | 83.1 | 91.6 | 96.4 |
| tta_only | 91.3 | 83.4 | 91.6 | 96.4 |
| wavelet_tta | 91.3 | 83.4 | 91.6 | 96.4 |
| wavelet_tta_p2i | 91.3 | 83.4 | 94.0 | 97.4 |
| wavelet_tta_multicrop | 91.8 | 85.6 | 91.6 | 96.4 |
| full_method | 91.8 | 85.6 | 94.5 | 97.6 |

### 3.2 VisA

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| original_anomalyclip | 95.5 | 86.7 | 82.0 | 85.3 |
| cached_baseline_l123 | 95.6 | 87.1 | 82.0 | 85.3 |
| wavelet_only | 95.6 | 87.1 | 82.0 | 85.3 |
| tta_only | 95.6 | 87.1 | 82.0 | 85.4 |
| wavelet_tta | 95.6 | 87.1 | 82.0 | 85.4 |
| wavelet_tta_p2i | 95.6 | 87.1 | 83.5 | 86.6 |
| wavelet_tta_multicrop | 96.2 | 91.3 | 82.0 | 85.4 |
| full_method | 96.2 | 91.3 | 84.6 | 87.4 |

## 4. 消融结论

### 4.1 Multi-crop fusion 是 pixel 指标主要来源

在 MVTec 上，`wavelet_tta` 到 `wavelet_tta_multicrop`：

- pixel AUROC：`91.3 -> 91.8`
- pixel AUPRO：`83.4 -> 85.6`

在 VisA 上，`wavelet_tta` 到 `wavelet_tta_multicrop`：

- pixel AUROC：`95.6 -> 96.2`
- pixel AUPRO：`87.1 -> 91.3`

这说明 multi-crop fusion 对定位质量的贡献最明显，尤其体现在 AUPRO 上。

### 4.2 Pixel-to-image fusion 是 image 指标主要来源

在 MVTec 上，`wavelet_tta` 到 `wavelet_tta_p2i`：

- image AUROC：`91.6 -> 94.0`
- image AP：`96.4 -> 97.4`

在 VisA 上，`wavelet_tta` 到 `wavelet_tta_p2i`：

- image AUROC：`82.0 -> 83.5`
- image AP：`85.4 -> 86.6`

这说明 pixel-level 异常响应可以反向补充 image-level 判别，尤其对 image AUROC 更明显。

### 4.3 小波与 TTA 单独贡献不明显

在当前最终参数下，`wavelet_only` 和 `wavelet_tta` 的 mean 指标提升有限：

- MVTec：`wavelet_only` 与 cached baseline 持平，`wavelet_tta` 主要只把 AUPRO 从 `83.1` 推到 `83.4`。
- VisA：`wavelet_only` 与 cached baseline 持平，`wavelet_tta` 只带来 image AP 的 `+0.1`。

因此当前数据更支持这样的表述：小波和 TTA 是整体 test-time calibration pipeline 的组成部分，但单独看不是当前提升的主要来源。论文里不能把“单独小波模块显著提升”作为强结论。

### 4.4 完整方法的收益来自互补融合

完整方法同时使用 multi-crop fusion 和 pixel-to-image fusion，因此可以同时提升 pixel-level 与 image-level：

- multi-crop fusion 主要贡献 pixel AUROC/AUPRO。
- pixel-to-image fusion 主要贡献 image AUROC/AP。
- 两者叠加后得到当前最好结果。

## 5. 内部设计消融

内部设计消融日志：`ablation_results/20260622_111707_internal/summary.md`

| 数据集 | 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|:--|--:|--:|--:|--:|
| MVTec | full_method | 91.8 | 85.6 | 94.5 | 97.6 |
| MVTec | full_no_wavelet_confidence | 91.8 | 85.6 | 94.5 | 97.6 |
| MVTec | full_no_rank_preserve | 91.8 | 85.6 | 94.5 | 97.6 |
| MVTec | full_no_local_contrast | 91.8 | 85.6 | 94.5 | 97.6 |
| VisA | full_method | 96.2 | 91.3 | 84.6 | 87.4 |
| VisA | full_no_wavelet_confidence | 96.2 | 91.3 | 84.6 | 87.4 |
| VisA | full_no_rank_preserve | 96.2 | 91.3 | 84.6 | 87.4 |
| VisA | full_no_local_contrast | 96.2 | 91.3 | 84.6 | 87.4 |

内部设计消融说明：`wavelet_confidence`、`rank_preserve`、`local_contrast` 在当前最终参数下没有带来可见 mean 指标变化。它们不适合作为主贡献点强写，可以放入附录或作为负结果说明。

## 6. 论文写作建议

### 6.1 可以作为主结论的点

- 当前方法在 MVTec 和 VisA 上均提升了 pixel-level 与 image-level 指标。
- pixel AUPRO 的提升最稳定，说明方法改善了 anomaly localization 质量。
- multi-crop fusion 与 pixel-to-image fusion 形成互补：前者改善 pixel map，后者改善 image-level classification。
- 当前方法不需要额外训练，适合写成 zero-shot test-time calibration / fusion framework。

### 6.2 不建议强写的点

- 不建议说“小波单独带来显著提升”，因为 `wavelet_only` 在两个数据集上基本持平。
- 不建议把 `wavelet_confidence`、`rank_preserve`、`local_contrast` 写成核心贡献，因为内部设计消融没有显示出 mean 指标增益。
- 不建议把大量参数调节写成贡献点，这容易显得像 test set tuning。

### 6.3 更稳妥的论文叙事

更稳妥的表述是：

> We propose a training-free test-time calibration framework for AnomalyCLIP. The framework integrates wavelet-guided anomaly map calibration, multi-crop localization fusion, and pixel-to-image score refinement. Experiments on MVTec and VisA show consistent improvements over the original AnomalyCLIP, especially on pixel-level AUPRO. Component ablations indicate that multi-crop fusion mainly improves localization quality, while pixel-to-image fusion improves image-level recognition.

对应中文表述：

> 我们提出一种面向 AnomalyCLIP 的无训练测试时校准框架。该框架结合小波引导的 anomaly map 校准、多裁剪定位融合以及 pixel-to-image 分数修正。MVTec 与 VisA 上的实验表明，该方法相较原始 AnomalyCLIP 在 pixel-level 和 image-level 指标上均有提升，其中 AUPRO 提升最明显。组件消融显示，multi-crop fusion 主要提升定位质量，而 pixel-to-image fusion 主要提升图像级判别能力。

## 7. 后续还可以补的实验

- 在其他异常检测数据集上验证完整方法，而不是继续只在 MVTec/VisA 上调参。
- 补充 qualitative visualization，展示 multi-crop fusion 前后的 anomaly map 变化。
- 做运行时间统计，说明 cache evaluation 和完整 test-time pipeline 的开销。
- 如果要保留小波作为论文关键词，需要补更直接的小波可视化或频域分析，否则当前数值消融对“小波单独有效”的支撑不足。

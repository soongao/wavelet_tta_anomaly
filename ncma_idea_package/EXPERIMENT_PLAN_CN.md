# NCMA 实验计划

## 1. 要验证的核心问题

NCMA 不是为了证明“流形有用”或“TTA 有用”，而是为了证明：

> ZSAD 中测试时适应必须优化正常变化流形上的投影；离开流形的残差应进入异常定位，而不是进入适应目标。

## 2. 协议

- 设置：辅助数据 ZSAD + 单样本 label-free TTA。
- 源域：用于学习对象无关正常变化流形图谱。
- 目标域：测试时只允许当前无标签图像及其增强视图。
- 禁止：目标标签、目标 mask、目标类别统计、跨测试样本累计更新。
- 更新参数：prompt residual、轻量 adapter 或 normalization affine，必须明确报告。
- 预算：增强视图数、更新步数和反向传播耗时都要报告。

## 3. 主表

数据集：

- MVTec AD；
- VisA；
- 可选：BTAD、MPDD、KSDD2、DAGM。

指标：

- I-AUROC、I-AP、I-F1-max；
- P-AUROC、P-AP、P-F1-max、P-AUPRO。

基线：

- CLIP baseline；
- WinCLIP；
- AnomalyCLIP；
- VCP-CLIP / AA-CLIP；
- test-time augmentation only；
- ordinary TTA / TPT；
- NCMA。

## 4. 必做消融

| 消融 | 目的 |
|---|---|
| No TTA | 固定模型基线 |
| Augmentation only | 排除多视图集成收益 |
| Ordinary TTA | 原始测试轨迹直接更新的最近负控 |
| Manifold residual score only | 检查流形外残差是否有独立定位价值 |
| NCMA without manifold projection target | 检查“只筛选正常 patch”是否不足 |
| NCMA without residual detaching | 检查异常候选参与更新是否抹平定位 |
| NCMA without residual score | 检查只做流形内校准是否不足 |
| Full NCMA | 流形投影校准与流形外定位的联合效果 |

## 5. 流形机制分析

必须做：

- 普通 TTA vs NCMA 的异常图对比。
- 测试 patch 到正常变化流形的投影可视化。
- 流形内兼容权重图。
- 流形外残差热图。
- 更新步数敏感性。
- 增强视图数与推理耗时。

建议做：

- 正常变化流形的局部邻域可视化。
- 正常变化方向维度敏感性。
- 源域选择敏感性。
- 残差分布：正常 patch 与异常 patch 的分离程度。

## 6. 成功标准

Minimal success:

- NCMA 在 MVTec AD 和 VisA 上超过 ordinary TTA。
- 至少一个关键定位指标 P-AP / P-AUPRO / P-F1-max 稳定提升。
- 去掉流形投影目标或去掉残差评分后性能下降。

Solid success:

- 普通 TTA 在部分样本中出现定位变差，NCMA 能缓解。
- 图像级和像素级指标同时稳定。
- 定性图能看到异常区域的流形外残差没有被更新抹平。

Strong success:

- 在多个工业数据集上稳定优于近期 CLIP-ZSAD 基线。
- 机制图清楚显示正常区域被投影校准，异常偏离被残差保留。
- 额外推理开销可接受。

Failure signals:

- NCMA 只提升 AUROC，不提升 P-AP/P-AUPRO/P-F1。
- ordinary TTA 和 NCMA 表现几乎一样。
- 移除流形投影目标后不下降。
- 使用目标测试统计却仍声称零样本。

## 7. 优先实验顺序

1. 先实现 ordinary TTA / TPT 负控。
2. 跑 augmentation only。
3. 实现正常变化流形和残差评分，但不更新。
4. 加流形投影目标驱动的 NCMA 更新。
5. 做残差剥离、投影目标和流形外评分消融。

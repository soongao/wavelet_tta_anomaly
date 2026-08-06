# Experiment Plan

## 目标声明

实验要验证的是：结构不变性探测比静态 CLIP 相似度、直接 wavelet fusion 和普通 TTA 更适合细粒度异常定位。

## 协议

- Setting：test-time adaptation ZSAD。
- Test-time visible data：当前无标签测试图像及其 wavelet-derived structural views。
- Updated parameters：只允许 prompt context、band gates 或 lightweight score calibration head；CLIP backbone 冻结。
- Update objective：confidence-selected / trimmed structural consensus，结合 entropy、cross-view consistency 和 residual preservation。
- Report：增强视图数量、更新步数、学习率、运行时间、显存。
- Metrics：`I-AUROC`, `I-AP`, `P-AUROC`, `P-AP`, `P-F1-max`, `P-AUPRO`。

## 主表

Table 1：MVTec AD 与 VisA 主结果。

Baselines：

- CLIP-style static scoring
- WinCLIP
- AnomalyCLIP
- AdaCLIP
- VCP-CLIP
- FiLo
- Wavelet score fusion
- Generic TTA/TPT-style prompt tuning
- SIP-CLIP

## 核心消融

Table 2：结构不变性探测消融。

| Variant | 目的 |
|---|---|
| Static CLIP prior | 静态相似度基线 |
| Wavelet score fusion | 负控：结构响应直接加分 |
| Multi-view average | 只平均结构视图，不适配 |
| Generic TTA | 普通熵/一致性 TTA |
| SIP without trimming | 检验异常是否被一致性目标吸收 |
| SIP without residual score | 检验残差项贡献 |
| SIP full | 完整方法 |

## 结构视图消融

- low-pass only
- high-detail only
- edge/detail view
- band-drop view
- reconstructed multi-band view
- Haar vs DCT vs Sobel
- number of views: 2/4/6/8

## TTA 消融

- updated parameter group：prompt / band gate / calibration head / all small modules
- update steps：0/1/3/5/10
- confidence threshold
- trimmed ratio
- entropy weight
- cross-view consistency weight
- residual preservation weight

## 可视化

- Figure 1：静态相似度失败与结构扰动不一致。
- Figure 2：SIP-CLIP pipeline Mermaid。
- Figure 3：各结构视图 anomaly map。
- Figure 4：普通 TTA 吸收异常 vs SIP 保留残差。
- Figure 5：cross-view variance/residual map 与 ground truth 对比。

## 成功标准

Minimal：

- SIP 超过 static CLIP、wavelet fusion 和 generic TTA。
- 移除 residual score 或 trimming 后 `P-AP/P-AUPRO` 下降。

Solid：

- 在 MVTec AD 与 VisA 都有提升，不只在纹理类有效。
- 可视化显示 SIP 保留缺陷残差，而 generic TTA 倾向平滑或吸收缺陷。

Strong：

- 额外工业数据集仍保持收益。
- 更新步数和结构视图数量有合理效率-性能曲线。
- 结果在 prompt template 变化下稳定。

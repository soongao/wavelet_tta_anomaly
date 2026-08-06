# 结构不变性探测

## 结论

更有论文感的包装是：

```text
结构不变性探测 / Structural Invariance Probing (SIP)
```

核心思想：许多工业缺陷不是新的语义对象，也不只是高频响应，而是局部证据在不同结构尺度下无法保持一致。正常区域在低频、边界、细节等结构视图中应保持稳定的正常判别；异常区域在这些视图下会产生语义-结构不一致残差。小波不是特征分支，而是结构扰动探针；TTA 不是补丁，而是在零标签测试条件下形成结构共识的机制。

## 朴素做法还原

- Raw move：用小波变换生成多种结构视图，在测试时对小参数组做无标签适配，使 CLIP 的正常/异常判别在稳定结构视图上保持一致；最终用跨视图残差定位异常。
- Naive story：小波变换 + TTA。
- Target protocol：测试时自适应 ZSAD；推理时只使用当前无标签测试图像及其小波结构视图，不使用目标标签、掩码、正常参考图或目标训练集统计。
- Intended claim：提升细粒度异常定位，尤其是单视图 CLIP 语义不稳定的纹理、边界和微小结构缺陷。

## Packaging-depth check

Raw trick：小波生成结构扰动视图 + 测试时 prompt/gate 适配 + 跨视图残差评分。

Naive story：我们把 wavelet 和 TTA 加到 CLIP-ZSAD。

Failed assumption in prior ZSAD：单次静态图文相似度足以判断局部异常证据。

Why this assumption fails：工业缺陷常是局部结构规律的破裂；在原图视角下，正常纹理、真实缺陷和成像噪声都可能得到相似的异常分数。可靠的正常证据应在结构扰动下保持判别一致，而异常区域会在某些尺度或频带中暴露不一致。

Why the raw trick becomes necessary：小波视图提供可控的结构扰动，用于探测局部证据是否具有跨尺度不变性；测试时无标签适配用稳定区域形成结构共识；残余的不一致区域成为异常定位证据。

Packaged concept：结构不变性探测 / Structural Invariance Probing。

One-sentence thesis：CLIP-ZSAD 应从“静态相似度判别”转向“结构不变性探测”；SIP-CLIP 用小波结构视图检验局部证据在跨尺度扰动下是否保持正常一致，并通过测试时结构共识适配将不一致残差定位为异常。

## 方法概念

SIP-CLIP 包含四个部分。

1. 结构视图生成：对测试图像构造低频视图、细节视图、边界增强视图、局部重构视图等 wavelet-derived structural views。

2. 冻结 CLIP 语义判别：每个结构视图都通过冻结 CLIP 图像编码器和正常/异常文本原型得到 patch-level anomaly evidence。

3. 测试时结构共识适配：只更新小参数组，例如 prompt context、band gate 或 score calibration head。优化目标不是让所有视图完全相同，而是在高置信稳定区域上最小化跨视图不一致和熵，同时通过 trimmed / confidence-selected loss 避免异常残差被吸收。

4. 残差异常定位：最终 anomaly score 由平均异常证据、跨视图方差、不变性残差和适配前后残差变化共同构成。异常区域是无法被结构共识解释的局部证据。

## 与已有工作的边界

- Wavelet / Scattering：已有多尺度稳定证据，但 SIP 把小波作为结构扰动探针，不是固定分支。
- FreqAnchorAD：已有 frequency-deviation anchoring，但 SIP 关注跨结构视图的不变性破裂，而不是频域 anchor scoring。
- TPT / Tent / MEMO：已有测试时适配范式，但 SIP 的测试时目标不是分类熵泛化，而是面向异常定位的结构共识与残差保留。
- VCP-CLIP / AdaCLIP：已有视觉上下文或动态 prompt；SIP 不做图像上下文 prompt 生成，而是用结构扰动视图定义测试时一致性目标。

## 推荐命名

- `SIP-CLIP`: Structural Invariance Probing for CLIP-ZSAD
- `RIFT-CLIP`: Residual Invariance Field Tuning
- `WaveProbe-CLIP`: Wavelet Structural Probing for ZSAD

最推荐 `SIP-CLIP`，因为它把核心落在“探测不变性”，而不是 wavelet 或 TTA。

## 贡献点草案

1. 提出 CLIP-ZSAD 中的静态相似度假设不足：单次图文相似度难以区分正常结构、成像扰动和真实缺陷。
2. 提出结构不变性探测，将异常定位重构为跨结构扰动下的正常一致性检验。
3. 设计测试时结构共识适配，只用当前无标签样本的小波结构视图更新小参数组，并保留不一致残差作为异常证据。
4. 通过无适配、普通 TTA、直接 wavelet fusion、不同结构视图和残差项消融验证 SIP-CLIP 的机制。

# SOP Idea Audit: 小波局部结构响应 + 单图正常参照

本文件记录按 `zsad-idea-sop` 重新判断后的论文包装结论。现有 LaTeX 稿只作为审计对象，不作为 idea 判断来源。

## 1. 朴素做法还原

- Raw move: 在冻结 CLIP-ZSAD 原型上，对单张测试图像的 patch 特征网格做 Haar 小波局部结构响应，结合初始语义响应选择可靠正常 patch，估计当前图像的正常参照，并保守校准正常文本原型。
- Naive story: 小波 + TTA 提升异常定位。
- Target protocol: 目标类别零样本异常定位；无目标训练图像、无正常参考图像、无像素标签。
- Intended claim: 改善图像级异常检测和像素级异常定位，尤其改善正常纹理/结构边界与真实缺陷混淆。

## 2. 已有工作相似度

- Frequency/wavelet 机制已被相邻工作覆盖：FreqAnchorAD 将频域模块包装为 frequency-deviation anchoring；Wavelet/Scattering case 明确指出小波分支应写成 multiscale stable evidence，而不能写成“高频缺陷”。
- TTA 机制已被 TPT/Tent/MEMO 覆盖：标准 TTA 通常包含测试时反向传播、参数更新、增强一致性或熵最小化。本文当前实现不做参数更新，因此不应把外层故事写成 TTA。
- Image-conditioned prompt 已被 VCP-CLIP/AdaCLIP 覆盖：这些工作把视觉上下文用于提示或文本表示。本文应与其区分为“从当前图像估计正常外观参照”，而不是“生成动态 prompt”。
- AnomalyCLIP 已覆盖 object-agnostic normal/abnormal prompt；本文不应重复声称文本 prompt 学习是核心贡献。

## 3. 新颖性判断

Novelty grade: N2 viable.

原因：小波、频域、测试时自适应和图像条件化提示都不是新机制；可写空间在 ZSAD 的 reference gap，即固定正常/异常文本原型提供跨图语义参照，但不能给出当前测试图像的正常外观基准。只要实验能证明固定参照、直接结构分数融合、纯语义自参照都弱于图像条件化正常参照，该包装是可成立的。

## 4. 推荐包装

```text
Raw trick: Haar 局部结构响应 + 单图证据选择 + 保守原型校准。
Naive story: 我们使用小波和 TTA。
Failed assumption in prior ZSAD: 固定正常/异常文本原型足以解释局部异常证据。
Why the raw trick becomes necessary: 同一类局部结构响应在不同材质中可能分别表示缺陷或正常纹理，因此需要从当前测试图像估计正常外观参照；小波结构响应只用于判断 patch 证据是否可靠。
Packaged concept: 图像条件化正常参照估计。
One-sentence thesis: CLIP-ZSAD 缺少当前图像的正常外观参照；ICNR 在冻结 CLIP 语义空间中从单图可靠 patch 估计该参照，并用它重新解释局部异常响应。
```

## 5. 论文写法边界

- 可以写：图像条件化正常参照、单图证据选择、局部结构响应作为证据可靠性、无目标训练图像、无正常参考图像、无测试时参数更新。
- 不要写：频率本身是本文新发现、所有异常都是高频、本文是标准 TTA、本文是 language-free、本文学习了新 prompt、本文解决任意域异常检测。
- 频域/小波应作为机制，不能作为标题、摘要和贡献的最深层理由。
- TTA 只能作为相关工作背景或部署时机描述；当前方法的核心名称应是 reference estimation/calibration。

## 6. 必须实验

- Main comparison: MVTec AD 和 VisA 至少包含 image-level 与 pixel-level 指标，并与 CLIP/WinCLIP/APRIL-GAN/AnomalyCLIP/AdaCLIP/VCP-CLIP/AA-CLIP 等协议相近方法比较。
- Reference ablation: FixedProto、ScoreFusion、GlobalRef、SemRef、StructRef、ICNR，证明“直接结构分数”和“固定参照”都不是核心解法。
- Structure ablation: 高频幅值、边界抑制、结构响应混合强度、Haar/其他 basis 或 band contribution。
- Protocol ablation: 明确无目标训练图像、无正常参考图像、无测试时反向传播；报告额外推理开销。
- Qualitative figures: 展示正常纹理/结构边界误报被正常参照抑制，以及小缺陷响应被保留。

## 7. 当前草稿状态

- 主线应保留为“面向 CLIP 零样本异常定位的图像条件化正常参照估计”。
- 摘要、引言、相关工作、实验和结论应避免把方法写成“小波 + TTA”。
- 正文中没有较早错误分支的残留；若后续继续编辑，应避免把旧分支内容合回本文。

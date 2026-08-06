# SOP Triage

## 1. 朴素做法还原

- Raw move：对测试图像或 CLIP patch grid 做 Haar/小波局部结构分解，结合 CLIP 正常/异常文本分数筛选同图可靠正常 patch，并用这些 patch 锚定当前图像正常性后校准异常图。
- Naive story：给异常定位加一个小波结构分支，并在推理时估计单图正常 reference。
- Target protocol：辅助数据 ZSAD；评估目标数据集不使用目标训练样本、正常参考图、异常标签或掩码。
- Intended claim：提升 image-level detection 与 pixel-level localization，重点是纹理破坏、边界断裂、小缺陷和局部结构偏离。

## 2. 合理性判断

可行部分：

- 工业异常常表现为当前产品正常结构的局部违背，而不是新语义对象。
- 当前图像中通常包含大量正常区域，可以提供 normal witnesses。
- 多尺度局部结构响应保留空间位置，适合判断局部结构是否稳定、重复或孤立。

容易出问题的部分：

- 局部结构响应本身没有正常/异常标签。直接把响应加到 anomaly map 会过检正常边界和纹理。
- 只用语义低异常 top-k patch 会把语义不敏感的异常区域误纳入正常参照。
- 如果写成“估计 reference”，包装仍然停留在中间变量；必须上升到图内正常性锚定。

## 3. 已有工作与相似度

Closest papers / cases：

- AnomalyCLIP：object-agnostic 正常/异常 prompt 已经覆盖通用异常语义。
- VCP-CLIP：visual context prompting 已经覆盖从图像中获取 prompt context。
- AdaCLIP：static/dynamic prompt 已经覆盖共享异常知识与实例适应。
- FreqAnchorAD：frequency-deviation anchoring 已经覆盖频率偏离与 anchor-relative scoring。
- Wavelet / Scattering case：小波应被包装为 multiscale stable evidence，而不是频段技巧。
- TPT / Tent / MEMO：推理阶段参数调整已有明确协议边界；本文不走这条线。

已有部分：

- 小波、多尺度和频域证据不新。
- 图像条件或实例条件的思想不新。
- 正常/异常文本原型不新。

仍可成立的部分：

- 把局部结构响应用于 normal witness discovery，而不是直接用于 anomaly score。
- 把当前图像内部的正常见证作为正常性锚点，补足外部文本原型无法定义局部正常性的缺口。
- 用负控证明直接结构融合和普通单图 reference 都不够，真正有效的是图内正常性锚定。

Novelty grade：`N2: viable`。若实验能证明 normal witnesses 的选择和图内锚定是不可替代的，并在不同类别上稳定提升 `P-AP/P-AUPRO`，可接近 `N3`。

## 4. Packaging-depth check

Raw trick：Haar/小波局部结构响应 + 语义正常置信度 + 同图正常 patch 聚合。

Naive story：我们加小波结构信息，并估计当前图像正常 reference。

Failed assumption in prior ZSAD：外部正常/异常原型足以定义目标图像中每个局部结构的正常性。

Why this assumption fails：工业图像中的局部结构是否正常，常取决于同一图像内部的重复纹理、边界连续性、材料规律和局部上下文；外部语义原型无法单独判断一个局部响应是否能被当前图像正常模式解释。

Why the raw trick becomes necessary：要判断局部结构是否异常，必须先找到同图中的正常见证；多尺度结构响应揭示结构是否孤立或可解释，语义先验排除明显异常候选，二者共同使 normality grounding 可行。

Packaged concept：图内正常性锚定 / Intra-image Normality Grounding。

One-sentence thesis：CLIP-ZSAD 的局部判别应从外部原型相似度转向图内正常性可解释性；ING 用语义先验和多尺度结构一致性发现同图正常见证，将正常性锚定到当前图像内部，从而提升细粒度异常定位。

## 5. 推荐 framing

Framing A：外部原型正常性假设失败 -> 图内正常性锚定。  
最推荐。它对应三个示例的层级：从实现技巧上升到 ZSAD 任务假设。

Framing B：局部结构歧义 -> 正常见证发现。  
适合作为方法内模块名，但不够大，容易被看成证据筛选技巧。

Framing C：单图参照估计 -> 保守校准。  
仍偏后处理，不建议作为最外层论文身份。

## 6. 叙事边界

- 不把方法写成小波或频率技巧。
- 不把正常 reference 写成最深贡献。
- 不写成图像条件 prompt。
- 不写成推理阶段模型调整。
- 不声称完全不使用语言分支。
- 不使用目标数据集标签、掩码、正常参考图或类别统计。

## 7. 必须证明的因果关系

1. 图内正常性锚定优于外部文本原型直接 scoring。
2. Normal witness discovery 优于语义 top-k 正常 patch。
3. 直接结构分数融合低于 ING。
4. 图内偏离图能解释正常纹理过检减少和真实缺陷增强。

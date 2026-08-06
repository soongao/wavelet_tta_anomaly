# Paper Spine CN

## 推荐标题

图内正常性锚定用于 CLIP 零样本异常定位

英文可对应：

```text
Intra-image Normality Grounding for CLIP-based Zero-shot Anomaly Localization
```

## 方法名

ING: Intra-image Normality Grounding

## 摘要草案

CLIP 零样本异常检测通常依赖外部正常/异常文本原型为目标图像中的 patch 赋予异常分数。这类原型提供了跨类别异常语义，但默认局部正常性可以由外部原型直接定义。该假设在工业定位中并不充分：同样的纹理、边界或局部结构响应，在一张图像中可能是正常规律，在另一张图像中可能是缺陷；其正常/异常含义必须由当前图像内部的结构上下文解释。

本文提出 ING，一种图内正常性锚定方法。ING 保持 CLIP 编码器和文本原型冻结，首先利用正常/异常原型得到 patch 级语义先验；随后计算多尺度局部结构响应，并结合语义正常置信度、邻域一致性和结构可解释性发现同图正常见证；最后将正常性锚定到这些见证 patch 上，对每个局部区域计算相对图内正常性的偏离。与直接融合结构响应不同，ING 不把局部高响应视为异常，而是判断该响应能否被当前图像内部的正常规律解释。实验应在 MVTec AD 与 VisA 上验证 ING 对细粒度异常定位的提升，并通过 normal witness、直接结构融合和外部原型对照证明收益来自图内正常性锚定。

## 引言主线

1. CLIP-ZSAD 的优势是跨类别语义迁移，代表方法用正常/异常 prompt 或视觉上下文构造外部异常判别依据。
2. 这些方法仍默认外部原型可以定义局部正常性。
3. 工业异常定位的关键常常不是识别新物体，而是判断局部结构是否违背当前图像内部的纹理、边界和周期规律。
4. 多尺度结构响应能揭示局部变化，但其正常/异常含义是歧义的，直接作为异常分数会过检。
5. 更深的任务重构是：先从当前图像中发现正常见证并锚定正常性，再判断局部证据是否能被图内正常性解释。
6. ING 用语义先验过滤候选，用多尺度结构一致性发现 normal witnesses，并把 anomaly map 建立在图内正常性偏离上。

## 贡献点

1. 我们指出 CLIP-ZSAD 中的外部原型正常性假设：固定或学习到的正常/异常原型能提供通用异常语义，但不能单独定义目标图像中的局部正常结构。
2. 我们提出图内正常性锚定，将异常定位重构为“局部证据是否能被当前图像内部的正常见证解释”的问题。
3. 我们设计 normal witness discovery，用语义正常先验、多尺度结构一致性和邻域一致性筛选能够锚定当前正常性的 patch。
4. 我们规划主实验、负控消融、结构响应分析和图内偏离可视化，以验证 ING 的收益来自正常性锚定，而不是简单结构分数叠加。

## 方法章节骨架

### 3.1 Problem Setup

定义辅助数据 ZSAD 设置。目标评估阶段不使用目标训练样本、正常参考图、异常标签或掩码。推理时只允许访问当前无标签图像，模型参数保持冻结。

### 3.2 External Semantic Prior

冻结 CLIP patch features 与正常/异常文本原型生成语义异常先验。该先验提供跨类别异常语义，但不负责定义图内正常性。

### 3.3 Multiscale Structural Consistency

计算多尺度局部结构响应，描述纹理、边界、周期和局部尺度变化。该响应用于判断局部结构是否稳定、重复或孤立。

### 3.4 Normal Witness Discovery

结合语义正常置信度、邻域一致性和结构可解释性，筛选当前图像中的 normal witnesses。它们是能为当前图像正常性提供证据的局部区域。

### 3.5 Intra-image Normality Grounding

将 normal witnesses 聚合为图内正常性锚点，并计算每个 patch 相对锚点的偏离。结构响应只在无法被图内正常性解释时强化异常证据。

## 图表蓝图

- Figure 1：motivation。外部原型不能区分正常边界和真实缺陷，因为两者都可能有强局部结构响应；ING 用同图正常见证锚定正常性。
- Figure 2：method overview。对应 `figures/ing_mechanism.mmd`。
- Figure 3：normal witness discovery。展示结构响应、语义正常置信度、normal witnesses 和最终 anomaly map。
- Figure 4：grounding effect。对比 external semantic prior、direct structure fusion、ING。
- Figure 5：score distribution。展示 grounding 前后正常/异常 patch 分布变化。
- Table 1：MVTec AD 与 VisA 主结果。
- Table 2：核心消融：external semantic prior、direct structure fusion、semantic top-k witnesses、ING。
- Table 3：结构一致性消融：Haar/DCT/Sobel、单尺度/多尺度、单锚点/多锚点。
- Table 4：效率与资源开销。

## 论文中应避免的写法

- 避免把标题、摘要或贡献点写成“小波 + 单图 reference”。
- 避免把单图参照估计当作论文最深贡献。
- 避免说某个频段本身是本文主要发现。
- 避免把当前图像统计说成会更新模型的流程。
- 避免说模型完全摆脱文本原型。

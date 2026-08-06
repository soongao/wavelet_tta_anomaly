# 图内正常性锚定

## 结论

把论文写成“估计一个单图正常参照”仍然偏浅，因为这只是给中间变量命名，没有达到 AnomalyCLIP、VCP-CLIP、AA-CLIP 那种任务假设重构的层级。

推荐 paper-facing 包装改为：

```text
图内正常性锚定 / Intra-image Normality Grounding (ING)
```

核心观点是：CLIP-ZSAD 不能只依赖外部正常/异常原型定义局部正常性。工业异常的关键证据常常是“某个局部结构是否违背当前图像内部的正常规律”。因此，模型需要从当前图像中发现可靠正常见证，把正常性锚定到同图结构规律上，再判断每个 patch 是否偏离这种被锚定的正常性。

## 朴素做法还原

- Raw move：在 CLIP patch 网格上计算 Haar/小波式多尺度局部结构响应，用语义正常置信度、邻域一致性和结构一致性筛选当前图像中的可靠正常 patch，再用这些 patch 校准异常图。
- Naive story：给 CLIP-ZSAD 加一个小波结构分支，并在推理时利用当前图像估计正常 reference。
- Target protocol：辅助数据 ZSAD；目标数据集中没有训练图像、正常参考图、异常标签或掩码。
- Intended claim：提升工业异常的 pixel-level localization，尤其是纹理破坏、边界断裂、小缺陷和局部结构偏离。

## 为什么旧包装不够

“正常参照估计”只是中间变量，审稿人很容易把它理解为一个后处理原型或单图统计技巧。它没有回答更深的问题：为什么 ZSAD 需要从当前图像内部重新定义正常性？

更深的失败假设是：

```text
现有 CLIP-ZSAD 默认外部正常/异常原型足以定义每个 patch 的正常性。
```

这个假设在工业定位中不成立。一个强边缘、周期纹理或局部突变，在外部原型看来可能都只是“不确定的局部结构”；它是否异常，取决于当前图像内部是否存在同类正常结构来解释它。正常性在这里不是单纯的文本语义状态，而是当前图像内部可被观察到、可被重复支持的结构规律。

## Packaging-depth check

Raw trick：Haar/小波局部结构响应 + 语义正常置信度 + 同图正常 patch 聚合。

Naive story：我们加小波结构信息，并估计当前图像正常 reference。

Failed assumption in prior ZSAD：外部正常/异常原型足以定义目标图像中每个局部结构的正常性。

Task-level reconstruction：异常定位应判断局部证据是否能被当前图像内部的正常规律锚定，而不是只判断它是否像通用异常语义。

Why the raw trick becomes necessary：要锚定图内正常性，模型必须找到同图中的可靠正常见证；多尺度结构响应揭示局部结构是否稳定、重复或孤立，语义先验过滤明显异常候选，二者共同使正常见证选择可行。

Packaged concept：图内正常性锚定 / Intra-image Normality Grounding。

One-sentence paper thesis：CLIP-ZSAD 的局部判别应从“外部原型相似度”转向“图内正常性可解释性”；ING 通过语义先验和多尺度结构一致性发现同图正常见证，将正常性锚定到当前图像内部，从而把歧义局部结构转化为相对图内正常性的异常证据。

## 方法概念

ING 包含四个层次。

1. 外部语义先验：冻结 CLIP 与正常/异常文本原型提供跨类别正常/异常语义。它只回答“是否像通用异常”，不负责定义当前图像的局部正常结构。

2. 图内结构一致性：多尺度局部结构响应描述每个 patch 在纹理、边界、周期性和局部尺度上的变化。这个响应不是异常分数，而是判断结构是否可被同图上下文解释的依据。

3. 正常见证发现：模型选择同时满足语义正常、邻域一致、结构可解释的 patch 作为 normal witnesses。它们不是任意 top-k 正常 patch，而是当前图像中能为“正常性”提供证据的局部结构。

4. 正常性锚定与偏离评分：将正常见证聚合为图内正常性锚点，并对每个 patch 计算其相对锚点的偏离。只有无法被图内正常性解释的结构响应才被转化为异常证据。

## 可实现版本

输入图像 `x`，冻结 CLIP 图像编码器输出 patch 特征 `P = {p_i}`，冻结文本侧输出正常/异常原型 `t_n, t_a`。

- 语义异常先验：`s_i = softmax(sim(p_i,t_a), sim(p_i,t_n))_a`。
- 图内结构响应：对图像或 patch feature grid 做多尺度 Haar/局部结构分解，得到响应 `r_i`，并按当前图像内部统计归一化。
- 正常见证权重：`w_i = semantic_normality(i) * neighborhood_consistency(i) * structural_explainability(i)`。
- 图内正常性锚点：`z_x = normalize(sum_i w_i p_i / (sum_i w_i + eps))`；复杂多纹理图像可扩展为 `K` 个锚点。
- 图内偏离：`g_i = 1 - sim(p_i, z_x)` 或相对最近正常性锚点的距离。
- 最终异常图：`a_i = semantic_prior(s_i) + lambda * grounding_gate(r_i) * g_i`。

## 和已有工作的边界

- AnomalyCLIP 已经覆盖 object-agnostic 正常/异常 prompt。ING 不把 prompt 学习当贡献，而是把 prompt 作为外部语义先验，再引入图内正常性锚定。
- VCP-CLIP 和 AdaCLIP 已经覆盖视觉条件 prompt 或动态 prompt。ING 不生成 prompt，而是从视觉 patch 内部发现 normal witnesses。
- FreqAnchorAD 已经覆盖 frequency-deviation anchoring。ING 不声称频率本身新颖，多尺度结构响应用于图内正常见证发现，而不是作为独立异常分数。
- TPT、Tent、MEMO 属于推理阶段参数或 prompt 调整范式。ING 推理阶段模型保持冻结，只执行单图正常性锚定。

## 推荐命名

- `ING`: Intra-image Normality Grounding
- `SGN`: Self-Grounded Normality
- `ING-CLIP`: Intra-image Normality Grounding for CLIP-ZSAD

最推荐 `ING` 或 `ING-CLIP`。它把论文身份放在“正常性必须被当前图像内部锚定”，而不是放在 reference、wavelet 或 test-time 这些实现细节上。

## 贡献点草案

1. 提出 CLIP-ZSAD 中的外部原型正常性假设，并指出其在局部工业异常定位中不足：许多局部结构的正常/异常含义必须由当前图像内部规律决定。
2. 提出图内正常性锚定，将异常定位重构为“局部证据能否被同图正常见证解释”的问题。
3. 设计正常见证发现机制，用语义正常先验、多尺度结构一致性和邻域一致性筛选能锚定当前图像正常性的 patch。
4. 通过主实验、直接结构融合负控、正常见证消融和图内偏离可视化验证：收益来自正常性锚定，而不是简单小波响应叠加。

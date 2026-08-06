# 图内正常性锚定用于 CLIP 零样本异常定位

## 摘要

CLIP 零样本异常检测通常依赖正常/异常文本原型为目标图像中的 patch 提供异常判别依据。这类外部原型具备跨类别迁移能力，但隐含了一个关键假设：局部正常性可以由共享的正常/异常语义原型直接定义。该假设在工业异常定位中并不充分，因为许多缺陷并不是新的语义对象，而是当前图像内部纹理、边界连续性、周期结构或局部尺度规律的违背；同样的局部结构响应在一张图像中可能是正常结构，在另一张图像中可能是异常证据。本文提出 ING（Intra-image Normality Grounding），一种图内正常性锚定方法。ING 保持 CLIP 编码器和文本原型冻结，首先用外部正常/异常原型获得语义先验，再利用多尺度结构一致性、语义正常置信度和邻域一致性发现当前图像中的 normal witnesses，最后将正常性锚定到这些同图见证上，并对每个 patch 计算相对图内正常性的偏离。与直接融合结构响应不同，ING 不把局部高响应视为异常，而是判断该响应能否被当前图像内部的正常规律解释。我们将在 MVTec AD 和 VisA 上验证 ING 的异常检测与定位能力，并通过 normal witness 消融、直接结构融合负控和图内偏离可视化证明性能收益来自图内正常性锚定。

## 1 引言

工业视觉质检需要在新产品、新材质和新成像条件下快速发现异常样本并定位缺陷区域。传统异常检测方法通常依赖目标类别的正常训练图像、异常标注或类别专用模型，这在真实产线中会受到数据采集成本、隐私约束和产品更新速度的限制。零样本异常检测面向更严格的部署条件：模型在目标域没有训练样本、正常参考图和异常标注时，仍需对未知类别图像输出图像级异常分数和像素级异常图。

CLIP 等视觉语言模型为零样本异常检测提供了可迁移的图文对齐能力。通过将图像 patch 与描述正常和异常状态的文本原型对齐，模型可以在未见类别上获得一定的异常语义迁移能力。WinCLIP 使用状态词和模板集合定义正常与异常状态，AnomalyCLIP 进一步将 prompt learning 包装为 object-agnostic normality/abnormality，强调目标物体类别词可能成为 ZSAD 中的干扰变量。VCP-CLIP 和 AdaCLIP 则从视觉上下文或动态 prompt 角度增强实例级适应能力。这些工作共同推动了 CLIP-ZSAD 的核心问题：如何让 CLIP 从物体语义识别转向正常/异常判别。

然而，这些方法大多仍依赖外部正常/异常原型来定义局部正常性。外部原型可以描述“正常”与“异常”的通用语义，却很难单独判断一个局部结构是否符合当前图像内部规律。工业异常往往不是新的对象，而是同一产品正常结构的局部违背。例如，一个强边缘可能是正常瓶口轮廓，也可能是裂纹边界；一个高频纹理区域可能是正常材料纹理，也可能是划痕；一个周期结构的局部断裂只有相对于同图其他周期结构才显得异常。对这些情况而言，局部正常性不是一个完全外部的文本概念，而是需要被当前图像内部的正常结构锚定。

直接引入小波、DCT 或边缘响应并不能解决这一问题。多尺度局部结构响应能够揭示纹理、边界和局部尺度变化，但它本身并不携带正常/异常标签。如果把结构响应直接加入 anomaly map，正常边界和正常纹理也会被放大。更合理的做法是把结构响应视为发现 normal witnesses 的证据：哪些 patch 在语义上接近正常、在邻域中一致、并且其结构响应可以被同图上下文解释，这些 patch 才能为当前图像的正常性提供见证。

基于这一观察，本文将 CLIP 零样本异常定位重构为图内正常性锚定问题。我们不再只问一个 patch 是否接近外部异常原型，而是进一步问：这个 patch 的局部结构能否被当前图像内部的正常见证解释？若局部结构无法被图内正常性锚定，它才更应被视为异常证据。这一重构使小波式结构响应从“异常分数来源”转变为“正常见证发现的可靠性信号”，也使单图信息的使用从简单后处理上升为正常性定义方式的改变。

本文提出 ING（Intra-image Normality Grounding）。给定一张目标域测试图像，ING 首先利用冻结 CLIP 与正常/异常文本原型得到外部语义先验；随后计算多尺度结构一致性，并从当前图像中发现 normal witnesses；最后由这些见证构建图内正常性锚点，对每个 patch 计算其相对图内正常性的偏离并生成 anomaly map。整个推理过程只访问当前无标签图像，目标域训练图像、正常参考图、异常标签和掩码均不可用，模型参数保持冻结。

本文贡献如下：

1. 我们指出 CLIP-ZSAD 中的外部原型正常性假设：共享正常/异常原型可以提供通用异常语义，但不能单独定义目标图像中的局部正常结构。
2. 我们提出图内正常性锚定，将异常定位重构为“局部证据是否能被当前图像内部的 normal witnesses 解释”的问题。
3. 我们设计 normal witness discovery，用语义正常先验、多尺度结构一致性和邻域一致性筛选能够锚定当前图像正常性的 patch。
4. 我们给出围绕 ING 的实验方案，包括主结果、直接结构融合负控、normal witness 消融、结构一致性分析和图内偏离可视化，以验证收益来自正常性锚定而非模块堆叠。

## 2 相关工作

### 2.1 CLIP 零样本异常检测

CLIP-based ZSAD 方法通常利用视觉语言对齐能力构造正常/异常状态的判别空间。WinCLIP 通过状态词和文本模板集合定义正常与异常文本原型，并结合局部窗口特征实现零样本异常定位。AnomalyCLIP 进一步指出，目标物体类别词在 ZSAD 中可能成为干扰变量，因此通过 object-agnostic prompt 学习通用正常性与异常性。APRIL-GAN、FiLo 等方法也从局部图文对齐、细粒度描述或高质量定位角度增强了 CLIP 的异常检测能力。

ING 与这类方法的区别不在于替代正常/异常文本原型，而在于改变局部正常性的来源。文本原型继续提供跨类别语义先验；ING 则从当前图像内部发现 normal witnesses，并把局部正常性锚定到同图结构规律上。

### 2.2 视觉上下文与实例条件化

视觉上下文方法试图缓解固定 prompt 对类别词或模板设计的依赖。VCP-CLIP 将图像上下文注入文本表示，使模型在产品类别未知或难以命名时仍能形成更合适的 prompt 表示。AdaCLIP 结合静态 prompt 和动态图像 prompt，在共享异常知识和实例级适应之间建立平衡。这类方法说明测试图像本身包含有价值的上下文信息。

ING 同样利用当前图像，但其目标不是生成或调整 prompt。VCP-CLIP 关注“类别/上下文无法作为可靠文本给出，因此应由图像提供 prompt context”；ING 关注“局部正常性无法由外部原型单独给出，因此应由当前图像内部的正常见证锚定”。二者都利用图像条件信息，但解决的缺失变量不同。

### 2.3 多尺度结构证据

频域、小波和多尺度表示常用于暴露纹理、边界、周期性和局部结构变化。FreqAnchorAD 将频率线索包装为 frequency-deviation anchoring，强调异常证据可分布在低、中、高不同频段，而不是单一频段现象。Wavelet/Scattering 相关工作也说明，多尺度分解可以在保留空间位置的同时揭示局部结构变化。

ING 借鉴多尺度局部结构响应，但不把频域或小波本身作为论文外层身份。结构响应在本文中用于 normal witness discovery：它帮助模型判断一个局部结构是否稳定、是否孤立、是否能被同图上下文解释。直接把结构响应作为异常分数是本文需要排除的负控。

### 2.4 推理阶段单图信息的使用边界

一些视觉语言方法在推理阶段使用单个测试样本及其增强视图调整 prompt 或模型权重，这类方法需要明确自监督目标、可变参数组、增强视图数量和运行开销。ING 的推理过程保持模型冻结，也不使用目标域批量统计或隐藏标签。它只从当前图像的冻结特征中发现 normal witnesses 并锚定图内正常性。

## 3 方法

### 3.1 问题定义

设源域数据为 \(\mathcal{D}_s\)，其中可包含来自源类别的图像级或像素级异常监督；目标域数据为 \(\mathcal{D}_t\)，其类别与源域不重合或在评估时不可用于训练。给定目标域测试图像 \(x\)，模型需要输出图像级异常分数 \(S(x)\) 和像素级异常图 \(A(x)\)。在目标域评估期间，方法不使用目标训练图像、正常参考图、异常标签或掩码。

冻结 CLIP 图像编码器输出 patch 特征：

\[
P = \{p_i\}_{i=1}^{N}, \quad p_i \in \mathbb{R}^{d}.
\]

文本侧使用正常和异常状态原型 \(t_n, t_a\)。这些原型可以来自手工模板、已有 CLIP-ZSAD prompt，或在源域辅助数据上学习得到。ING 不把 prompt 学习作为核心贡献，而是把文本原型作为外部语义先验。

### 3.2 方法总览

图 2 给出了 ING 的整体流程。ING 包含四个步骤：外部语义先验、多尺度结构一致性、normal witness discovery、图内正常性锚定与异常评分。首先，冻结 CLIP 将每个 patch 与正常/异常文本原型比较，得到外部语义先验。其次，多尺度结构响应描述当前图像中的纹理、边界、周期和局部尺度变化。第三，ING 结合语义正常置信度、邻域一致性和结构可解释性发现 normal witnesses。最后，模型将正常性锚定到这些同图见证上，并以相对图内正常性的偏离生成 anomaly map。

这一设计对应本文的问题重构：异常定位不应只由外部原型相似度决定，还应判断局部证据是否能被当前图像内部的正常性解释。

### 3.3 外部语义先验

对于每个 patch 特征 \(p_i\)，我们计算其与正常原型和异常原型的相似度：

\[
\ell_i^n = \tau \cdot \mathrm{sim}(p_i, t_n), \quad
\ell_i^a = \tau \cdot \mathrm{sim}(p_i, t_a),
\]

其中 \(\mathrm{sim}(\cdot,\cdot)\) 表示余弦相似度，\(\tau\) 为温度系数。外部语义异常先验定义为：

\[
s_i = \frac{\exp(\ell_i^a)}{\exp(\ell_i^a)+\exp(\ell_i^n)}.
\]

较小的 \(s_i\) 表示该 patch 在通用语义上更接近正常状态。该分数为 normal witness discovery 提供初始过滤，但它不直接定义图内正常性。

### 3.4 多尺度结构一致性

为捕捉 CLIP 语义特征可能压缩掉的细粒度结构信息，ING 在输入图像或 patch 对齐特征网格上计算多尺度局部结构响应。以 Haar 分解为例，对于尺度集合 \(\mathcal{K}\)，我们计算水平、垂直和对角方向的局部变化，并将其投影回 patch 网格：

\[
r_i = \mathrm{Norm}_{x}\left(\sum_{k \in \mathcal{K}} \alpha_k \left(|H_i^k| + |V_i^k| + |D_i^k|\right)\right),
\]

其中 \(H_i^k,V_i^k,D_i^k\) 分别表示第 \(k\) 个尺度下落入 patch \(i\) 的局部响应，\(\alpha_k\) 为尺度权重，\(\mathrm{Norm}_{x}\) 表示当前图像内部归一化。该归一化避免不同产品纹理强度和成像条件造成的响应尺度差异。

结构响应 \(r_i\) 不表示异常概率。它的作用是支持结构一致性判断：某个局部结构是否与邻域一致，是否在同图中有重复或连续支持，是否表现为孤立突变。

### 3.5 Normal Witness Discovery

ING 从当前图像中发现 normal witnesses。一个 patch 成为正常见证需要同时满足三个条件：

1. 外部语义先验认为它更接近正常；
2. 它与邻域 patch 在 CLIP 特征空间中保持一致；
3. 它的结构响应能够被同图上下文解释，而不是孤立突变。

我们将 normal witness 权重写为：

\[
w_i = q_i^{sem} \cdot q_i^{nei} \cdot q_i^{str},
\]

其中：

\[
q_i^{sem}=1-s_i,
\]

\[
q_i^{nei}=\frac{1}{|\mathcal{N}(i)|}\sum_{j\in \mathcal{N}(i)} \mathrm{sim}(p_i,p_j),
\]

\[
q_i^{str}=\exp(-\beta \cdot \Delta r_i).
\]

\(\mathcal{N}(i)\) 表示 patch \(i\) 的局部邻域，\(\Delta r_i\) 表示该 patch 的结构响应与邻域或同图结构模式之间的相对偏离，\(\beta\) 控制结构一致性权重。这个设计不会简单排除所有高响应区域；只有当高响应无法被同图上下文解释时，它作为 normal witness 的权重才会下降。

### 3.6 图内正常性锚定

给定 normal witness 权重，ING 将当前图像的正常性锚定为：

\[
z_x = \mathrm{norm}\left(\frac{\sum_{i=1}^{N} w_i p_i}{\sum_{i=1}^{N} w_i + \epsilon}\right).
\]

对于包含多个正常纹理模式或多个部件的图像，可将 normal witnesses 聚类为 \(K\) 个图内正常性锚点 \(\{z_x^k\}_{k=1}^{K}\)。默认版本使用单锚点以保持简洁，复杂类别再通过多锚点版本扩展。

对于每个 patch，图内偏离定义为：

\[
g_i = 1 - \max_k \mathrm{sim}(p_i, z_x^k).
\]

最终异常分数由外部语义先验、图内偏离和结构可靠性门控共同决定：

\[
a_i = s_i + \lambda \cdot \phi(r_i, q_i^{str}) \cdot g_i,
\]

其中 \(\lambda\) 控制图内锚定强度，\(\phi\) 是 grounding gate。该门控仅在 patch 与图内正常性锚点存在明显偏离时强化结构响应，避免把正常纹理和正常边界直接判为异常。最终 pixel-level anomaly map 由 patch 分数上采样并轻量平滑得到，image-level score 使用 anomaly map 的 top-k 均值或最大响应聚合。

### 3.7 资源与协议边界

ING 使用冻结 CLIP 特征、正常/异常文本原型和当前无标签测试图像。它不使用目标域训练图像、正常参考图、异常标签或掩码，推理阶段模型保持冻结。若文本原型来自源域辅助训练，应在实验设置中明确源域数据和目标域数据的分离方式。

## 4 实验

### 4.1 实验设置

数据集。主实验使用 MVTec AD 与 VisA。MVTec AD 覆盖物体类和纹理类工业异常，VisA 提供互补的复杂产品类别和细粒度缺陷。若后续主张更广泛工业泛化，可在附录加入 BTAD、MPDD、KSDD2 或 DAGM。

协议。采用辅助数据 ZSAD 设置。评估 MVTec AD 时，不使用 MVTec AD 的目标训练图像、正常参考图、异常标签或掩码；评估 VisA 时同理。若使用源域辅助数据训练 prompt 或轻量模块，源域和目标域需要显式分离。

基线。主表应包含 CLIP-style baseline、WinCLIP、APRIL-GAN、AnomalyCLIP、AdaCLIP、VCP-CLIP、FiLo，以及可复现实验设置下的多尺度/频域相关 baseline。所有方法应在相同输入分辨率、backbone 和目标域不可见条件下比较。

指标。图像级检测报告 `I-AUROC`、`I-AP` 和 `I-F1-max`；像素级定位报告 `P-AUROC`、`P-AP`、`P-F1-max` 和 `P-AUPRO`。其中 `P-AP` 和 `P-AUPRO` 对细粒度定位更关键，因为高 `P-AUROC` 可能掩盖过检和低精度问题。

实现细节。默认使用 CLIP ViT-B/16 作为视觉编码器。结构响应可在输入图像或 patch 对齐网格上计算，并按当前图像内部统计进行归一化。默认图内正常性锚点数 \(K=1\)，normal witness 权重由语义正常置信度、邻域一致性和结构可解释性共同决定。推理阶段不改变任何模型参数。

### 4.2 主结果

表 1 给出 MVTec AD 与 VisA 上的主比较。ING 的目标是在不使用目标域训练数据的条件下，提高像素级定位质量，同时保持图像级检测性能。我们重点关注三个现象：第一，ING 相比仅使用外部语义原型的 baseline 应获得稳定提升；第二，ING 相比已有 CLIP-ZSAD 方法应在 `P-AP` 和 `P-AUPRO` 上体现更好的细粒度定位；第三，ING 不应以明显牺牲 `I-AUROC` 或 `I-AP` 为代价换取局部指标提升。

| Method | MVTec I-AUROC ↑ | MVTec I-AP ↑ | MVTec P-AUROC ↑ | MVTec P-AP ↑ | MVTec P-AUPRO ↑ | VisA I-AUROC ↑ | VisA I-AP ↑ | VisA P-AUROC ↑ | VisA P-AP ↑ | VisA P-AUPRO ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP-style baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| WinCLIP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| APRIL-GAN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AnomalyCLIP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AdaCLIP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VCP-CLIP | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FiLo | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ING | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.3 图内正常性锚定消融

表 2 分析图内正常性锚定是否是性能来源。`External semantic prior only` 只使用外部正常/异常原型；`Direct structure fusion` 直接把结构响应加入异常图，是检验结构响应本身是否足够的负控；`Global source anchor` 使用源域聚合锚点，检验同图锚定的重要性；`Semantic top-k witnesses` 只用语义分数选择 normal witnesses，不使用结构一致性；`ING full` 为完整方法。

| Variant | External prior | Structural consistency | Intra-image grounding | MVTec P-AP ↑ | MVTec P-AUPRO ↑ | VisA P-AP ↑ | VisA P-AUPRO ↑ |
|---|---|---|---|---:|---:|---:|---:|
| External semantic prior only | yes | no | no | TBD | TBD | TBD | TBD |
| Direct structure fusion | yes | score fusion | no | TBD | TBD | TBD | TBD |
| Global source anchor | yes | yes | no | TBD | TBD | TBD | TBD |
| Semantic top-k witnesses | yes | no | yes | TBD | TBD | TBD | TBD |
| Structure-only witnesses | no | yes | yes | TBD | TBD | TBD | TBD |
| ING full | yes | yes | yes | TBD | TBD | TBD | TBD |

该消融需要支持两个判断：直接融合结构响应不能替代图内正常性锚定；仅依赖语义 top-k patch 也不足以稳定发现 normal witnesses。完整方法应在 `P-AP` 和 `P-AUPRO` 上获得最稳定收益。

### 4.4 Normal witness 与结构一致性分析

为了验证 normal witness discovery 的作用，我们比较只使用语义置信度、加入邻域一致性、加入结构一致性、以及三者组合的版本。该实验应证明 normal witnesses 不是普通 top-k patch，而是图内正常性锚定所需的可靠证据。

| Variant | Semantic normality | Neighborhood consistency | Structural consistency | MVTec P-AP ↑ | MVTec P-AUPRO ↑ | VisA P-AP ↑ | VisA P-AUPRO ↑ |
|---|---|---|---|---:|---:|---:|---:|
| Semantic only | yes | no | no | TBD | TBD | TBD | TBD |
| Semantic + neighborhood | yes | yes | no | TBD | TBD | TBD | TBD |
| Semantic + structure | yes | no | yes | TBD | TBD | TBD | TBD |
| Full witness discovery | yes | yes | yes | TBD | TBD | TBD | TBD |

我们还比较 Haar、Sobel/edge、local DCT、单尺度响应和无结构响应等变体。该实验不以证明某个固定变换普遍最优为目的，而是分析结构一致性用于 normal witness discovery 时是否带来稳定改善。

### 4.5 可视化分析

图 3 展示 normal witness discovery。每个样本应包含原图、外部语义先验、结构响应图、normal witness mask、图内偏离图和最终 anomaly map。理想现象是：正常边界或正常纹理虽有较强结构响应，但由于它们能被同图 normal witnesses 锚定，最终异常分数被抑制；真实缺陷由于无法被图内正常性解释，最终得到更清晰定位。

图 4 对比 `External semantic prior only`、`Direct structure fusion` 和 `ING full`。该图用于展示直接结构融合的过检问题，以及 ING 如何通过图内正常性锚定减少正常纹理处的误响应。

图 5 展示正常 patch 与异常 patch 在外部语义分数、图内偏离和最终分数上的分布。若 ING 有效，图内偏离应扩大正常区域和异常区域之间的分离，尤其是在外部文本原型不确定的样本上。

### 4.6 效率分析

ING 的额外成本来自结构响应计算、normal witness 权重计算和图内正常性锚定。表 4 报告输入分辨率、backbone、推理时间、显存、normal witness 数量和推理阶段是否改变参数。由于 ING 不执行反向传播，其开销应主要表现为轻量前处理和 patch-level 聚合。

| Method | Backbone | Resolution | Extra time | GPU memory | Witness / anchor count | Parameter changed at inference |
|---|---|---:|---:|---:|---:|---|
| External semantic prior | CLIP ViT-B/16 | TBD | TBD | TBD | 0 | no |
| ING | CLIP ViT-B/16 | TBD | TBD | TBD | TBD | no |

## 5 讨论与局限

ING 的核心假设是：一张测试图像中存在足够多的正常区域，可以作为 normal witnesses 锚定图内正常性。当异常面积过大、图像几乎被缺陷覆盖，或正常模式本身高度多样且难以由少量锚点表示时，normal witness discovery 可能受到污染。多锚点建模、前景区域约束和更稳健的见证筛选可以缓解这一问题，但也会增加方法复杂度。

另一个边界是结构一致性的适用范围。ING 适合纹理、边界、周期性和局部结构破坏明显的工业缺陷；对于纯语义异常、逻辑关系异常或缺陷与正常区域结构差异极弱的情况，局部结构响应可能提供有限帮助。此时需要结合更强的对象关系建模或部件级先验。

最后，ING 仍依赖 CLIP 正常/异常文本原型提供外部语义先验，因此不能被描述为完全不使用语言分支的方法。本文的贡献在于把局部正常性锚定到当前图像内部，而不是替代 CLIP 的视觉语言先验。

## 6 结论

本文提出 ING，一种面向 CLIP 零样本异常定位的图内正常性锚定方法。ING 从外部原型无法单独定义局部正常性的假设缺口出发，将多尺度结构响应用于 normal witness discovery，并由当前图像内部的正常见证锚定正常性，再判断局部证据是否能被图内正常性解释。该设计避免把结构响应直接等同于异常分数，使模型能够区分正常纹理/边界与缺少同图正常支持的真实结构偏离。后续实验将围绕 MVTec AD、VisA、核心负控消融和可视化分析验证这一机制。

## 附：初稿对应图表清单

- Figure 1：external prototype normality failure。
- Figure 2：ING method overview，对应 `figures/ing_mechanism.mmd`。
- Figure 3：normal witness discovery visualization。
- Figure 4：negative control maps。
- Figure 5：score distribution before/after intra-image grounding。
- Table 1：MVTec AD 与 VisA 主结果。
- Table 2：图内正常性锚定消融。
- Table 3：normal witness 与结构一致性分析。
- Table 4：效率分析。

# 结构不变性探测用于 CLIP 零样本异常定位

## 摘要

CLIP 零样本异常检测通常在原始图像上比较 patch 特征与正常/异常文本原型，由此得到异常分数。这种静态相似度判别具备跨类别迁移能力，但对工业缺陷并不充分：许多缺陷不是新的语义对象，而是局部结构在尺度、边界或频带上的不稳定表现。本文提出 SIP-CLIP，一种结构不变性探测方法。SIP-CLIP 使用小波变换构造低频、细节、边界和带阻重构等结构视图，并在测试时只更新小型 prompt/gate 或 score calibration 参数，使高置信稳定区域在跨结构视图下形成正常/异常判别共识。与直接融合小波响应不同，SIP-CLIP 将异常定义为无法被结构共识吸收的跨视图残差。该方法把小波从特征分支转化为结构扰动探针，把 TTA 从泛化技巧转化为零标签结构共识机制。实验将在 MVTec AD 和 VisA 上验证 SIP-CLIP 的细粒度定位能力，并通过普通 TTA、wavelet fusion、残差项和 trimmed consensus 消融分析其机制。

## 1 引言

零样本异常检测要求模型在没有目标域训练图像、正常参考图或异常标注的条件下，识别异常样本并定位缺陷区域。CLIP 等视觉语言模型为该任务提供了可迁移的图文对齐能力。现有 CLIP-ZSAD 方法通常通过正常/异常文本原型、object-agnostic prompt 或视觉上下文 prompt，将目标图像 patch 映射到正常/异常语义空间。

这类方法的隐含前提是：一次静态图文相似度比较足以判断局部异常证据。然而工业缺陷常常不是稳定的语义类别，而是局部结构在不同尺度下的表现不一致。划痕可能只在细节视图中显著，凹陷可能破坏低频连续性，断裂可能表现为边界响应不稳定。相反，正常纹理和正常边缘也会产生强局部响应，但它们在结构扰动下通常保持一致的正常判别。

因此，异常定位不应只问“这个 patch 是否像异常文本”，还应问“这个局部证据在结构扰动下是否保持正常一致”。小波变换为这一问题提供了天然探针：它能构造低频、细节、边界和重构等结构视图，暴露局部证据在不同尺度下的稳定性。关键不在于某个频段本身是否异常，而在于跨结构视图的判别是否形成一致共识。

本文提出 SIP-CLIP（Structural Invariance Probing for CLIP-ZSAD）。给定一张测试图像，SIP-CLIP 生成一组 wavelet-derived structural views，并用冻结 CLIP 得到每个视图的 patch anomaly map。随后，方法在测试时只更新小参数组，例如 prompt context、band gate 或 score calibration head，通过 confidence-selected structural consensus 让高置信稳定区域形成一致判别。为避免异常被适配目标吸收，我们使用 trimmed consistency，只在稳定区域施加一致性，并将跨视图残差保留为异常定位证据。

本文贡献如下：

1. 我们指出 CLIP-ZSAD 的静态相似度假设不足：非语义工业缺陷常表现为结构扰动下的不一致，而不是单视图下稳定可见的异常语义。
2. 我们提出结构不变性探测，将异常定位重构为跨小波结构视图的正常一致性检验。
3. 我们设计测试时结构共识适配，只使用当前无标签样本的结构视图更新小参数组，并通过 trimmed selection 保留异常残差。
4. 我们给出面向该机制的实验方案，比较 static CLIP、direct wavelet fusion、generic TTA 和 SIP-CLIP，并通过残差图可视化验证机制。

## 2 方法

### 2.1 结构视图生成

给定测试图像 \(x\)，SIP-CLIP 构造结构视图集合：

\[
\mathcal{V}(x)=\{x^0, x^{low}, x^{detail}, x^{edge}, x^{drop}\}.
\]

其中 \(x^0\) 是原图，其他视图由小波分解、频带抑制、细节增强或局部重构得到。这些视图不是数据增强的装饰，而是用于探测局部证据是否保持结构不变。

### 2.2 多视图 CLIP 异常证据

每个结构视图经过冻结 CLIP 图像编码器，得到 patch 特征 \(p_{i}^{v}\)。给定正常/异常文本原型 \(t_n,t_a\)，每个视图的 patch 异常先验为：

\[
s_i^v=\frac{\exp(\tau \mathrm{sim}(p_i^v,t_a))}{\exp(\tau \mathrm{sim}(p_i^v,t_a))+\exp(\tau \mathrm{sim}(p_i^v,t_n))}.
\]

静态 CLIP 只使用 \(s_i^0\)。SIP-CLIP 关注的是 \(\{s_i^v\}_{v\in\mathcal{V}}\) 的跨视图一致性和残差。

### 2.3 测试时结构共识适配

SIP-CLIP 在测试时更新小参数组 \(\theta\)，例如 prompt context、band gate 或 score calibration head。CLIP backbone 保持冻结。更新目标包含三部分：

\[
\mathcal{L}_{tta} = \mathcal{L}_{ent} + \lambda_c \mathcal{L}_{cons} + \lambda_r \mathcal{L}_{res}.
\]

\(\mathcal{L}_{ent}\) 降低高置信稳定区域的不确定性，\(\mathcal{L}_{cons}\) 约束稳定区域跨结构视图的一致性，\(\mathcal{L}_{res}\) 防止所有残差被一致性目标吸收。稳定区域由 confidence selection 和 trimmed selection 确定，只选择低跨视图方差、低异常置信或高一致性的 patch 参与共识形成。

### 2.4 残差异常评分

适配后，SIP-CLIP 计算平均异常证据：

\[
\bar{s}_i=\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}s_i^v,
\]

以及结构不变性残差：

\[
\rho_i=\mathrm{Var}_{v\in\mathcal{V}}(s_i^v).
\]

最终异常分数为：

\[
a_i=\bar{s}_i+\gamma \rho_i.
\]

直观上，正常区域应在结构扰动下保持稳定判别；异常区域即使均值不高，也会因跨视图残差暴露出来。

## 3 实验设计

主实验在 MVTec AD 和 VisA 上进行，报告 image-level `I-AUROC/I-AP` 与 pixel-level `P-AUROC/P-AP/P-AUPRO`。基线包括 CLIP-style static scoring、WinCLIP、AnomalyCLIP、AdaCLIP、VCP-CLIP、FiLo、direct wavelet fusion、multi-view average 和 generic TTA。

核心消融包括：

- `Static CLIP prior`：只用原图静态相似度。
- `Wavelet score fusion`：直接融合结构响应。
- `Multi-view average`：只平均多结构视图，不适配。
- `Generic TTA`：普通熵最小化或一致性适配。
- `SIP without trimming`：检验异常是否被共识目标吸收。
- `SIP without residual score`：检验残差项贡献。
- `SIP full`：完整结构不变性探测。

必须报告 TTA 细节：结构视图数量、更新参数组、更新步数、学习率、confidence threshold、trimmed ratio、推理时间和显存。

## 4 讨论

SIP-CLIP 的关键边界是：它适合局部结构在尺度或频带上表现不稳定的工业缺陷。如果异常是纯语义类别错误，或缺陷在所有结构视图中都与正常区域完全一致，残差项可能有限。此外，TTA 带来额外推理成本，因此需要用效率表说明更新步数与性能的关系。

## 5 结论

本文提出 SIP-CLIP，将 CLIP-ZSAD 从静态图文相似度判别转化为结构不变性探测。通过小波结构视图和测试时结构共识适配，SIP-CLIP 将正常区域建模为跨结构扰动保持一致的证据，并将无法被共识吸收的残差定位为异常。该包装使小波和 TTA 都服务于同一个任务重构，而不是简单模块堆叠。

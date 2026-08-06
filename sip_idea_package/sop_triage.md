# SOP Triage

## 1. 朴素做法还原

- Raw move：使用小波生成多结构视图，并在测试时对小参数组进行无标签适配，使稳定区域的正常/异常判别形成跨视图共识，异常由不一致残差暴露。
- Naive story：小波变换 + TTA。
- Target protocol：test-time adaptation ZSAD；只使用当前无标签测试样本及其结构视图。
- Intended claim：提升细粒度异常定位和鲁棒性，尤其是 CLIP 原图单视图判别不稳定的局部缺陷。

## 2. 已有工作相似度

- Wavelet / Scattering：小波、多尺度和局部结构证据已有，不能声称 wavelet 本身新。
- FreqAnchorAD：频率偏离与 anchor scoring 已有，不能写成 frequency-deviation anchoring 的变体。
- TPT / Tent / MEMO：测试时适配已有，必须明确可见数据、增强视图、参数组、目标函数、步数和运行成本。
- VCP-CLIP / AdaCLIP：视觉上下文和动态 prompt 已有，SIP 不能写成 image-conditioned prompt。

Prior-overlap grade：`Mechanism covered, story open`。小波和 TTA 都有先例，但“结构扰动不变性探测 + 异常残差保留”可以形成新的 ZSAD 任务构造。

Novelty grade：`N2: viable`，若能证明普通 TTA 会吸收异常而 SIP 的 trimmed structural consensus 能保留异常残差，可接近 `N3`。

## 3. Packaging-depth check

Raw trick：wavelet structural views + test-time prompt/gate tuning + residual disagreement anomaly score。

Naive story：add wavelet and TTA。

Failed assumption in prior ZSAD：单次静态图文相似度足以判断局部异常证据。

Task-level reconstruction：异常定位应检验局部证据在结构扰动下是否保持正常一致；无法保持一致的残差才是更可靠的异常证据。

Why the raw trick becomes necessary：小波提供结构扰动视图；TTA 在无标签条件下从稳定区域形成结构共识；残差评分避免把所有跨视图不一致都平均掉。

Packaged concept：结构不变性探测 / Structural Invariance Probing。

One-sentence thesis：SIP-CLIP 将 CLIP-ZSAD 从静态相似度判别转化为结构不变性探测，通过测试时结构共识适配保留异常残差，从而提升细粒度异常定位。

## 4. Claims to avoid

- 不把异常写成单一频段现象。
- 不把小波写成创新点本身。
- 不把 TTA 写成无成本或普通后处理。
- 不使用目标标签、掩码或目标训练集统计。
- 不声称完全不依赖语言分支。
- 不把跨视图一致性强行施加到全部 patch，否则会把异常也适配掉。

## 5. 必须证明的因果关系

1. 直接 wavelet score fusion 不如 SIP。
2. 普通 TTA 不如 SIP，尤其在异常残差保留上。
3. 不使用 confidence/trimmed selection 会把异常吸收进一致性目标，降低定位。
4. 残差项对 `P-AP` 和 `P-AUPRO` 有实际贡献。

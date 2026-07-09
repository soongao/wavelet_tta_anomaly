# Wavelet-Supervised Test-Time Prototype Adaptation for Zero-Shot Anomaly Detection

中文论文正文草稿 v1.6

## 摘要

零样本工业异常检测要求模型在不使用目标类别训练图像和异常标注的条件下，同时完成图像级异常判别和像素级异常定位。现有 CLIP-based 方法通常依赖固定 normal/abnormal 文本原型，但固定原型难以覆盖每张测试图像中的实例化缺陷外观，也容易把真实局部缺陷、正常材料纹理和结构边界混淆。本文提出 Wavelet-Supervised Test-Time Prototype Adaptation (WPTA)，核心思想是把 Haar 小波线索作为测试时原型适配的 patch evidence reliability，而不是把频域响应直接叠加为最终异常分数。WPTA 在冻结 CLIP 参数的前提下，从 CLIP patch feature grid 构造 boundary-aware wavelet reliability，并与初始语义异常分数共同选择 visual normal/abnormal anchors，再通过 confidence-gated conservative update 校准 normal/abnormal prototypes。受控 MVTec/VisA 消融表明，direct wavelet fusion 会降低定位质量，而 semantic-spectral evidence selection 与 conservative calibration 能改善测试时原型适配；在五个工业基准上，包含 WPTA 或相关校准模块的最终推理系统相对固定 AnomalyCLIP baseline 平均提升 +1.0 / +4.5 / +3.4 / +2.8，指标顺序为 P-AUROC / P-AUPRO / I-AUROC / I-AP。实验结果支持两点结论：WPTA 机制在 MVTec/VisA 受控消融中有效，最终校准推理系统在五个工业基准上相对固定 baseline 获得系统级收益。

关键词：zero-shot anomaly detection；CLIP；test-time adaptation；prototype calibration；wavelet reliability；industrial inspection

## 1. 引言

工业异常检测需要在产品图像中发现划痕、缺口、污染、破损和局部纹理异常，并同时输出图像级异常判别和像素级异常定位。一个典型难例是金属件边缘附近的细划痕：缺陷区域很小，但正常轮廓边缘、反光和背景纹理也会产生强局部响应。理想检测器必须覆盖真实划痕，同时避免把正常结构边界误判为异常。由于工业场景中异常样本稀缺、类别更新频繁且像素级标注成本高，zero-shot anomaly detection 成为实际部署中有价值的设置。视觉语言模型 CLIP 提供了开放词汇图文对齐能力，WinCLIP、AnomalyCLIP、AdaCLIP 和 CLIP-AD 等工作进一步将 normal/abnormal 文本提示或原型用于 anomaly classification 与 segmentation \cite{radford2021clip,jeong2023winclip,zhou2024anomalyclip,cao2024adaclip,chen2024clipad}。

固定 normal/abnormal prototypes 的核心局限是 instance-specific mismatch。一个通用 abnormal prototype 可以表达“异常”这个语义方向，但难以精确覆盖当前图像中的具体缺陷形态，例如细划痕、瓶口缺口、纺织纹理污染或金属表面压痕。对于 CLIP-based anomaly detection，这种不匹配会直接出现在 patch anomaly map 中：真实缺陷可能因为与通用 abnormal prototype 相似度不足而被低估，正常边界或材料纹理也可能因局部视觉差异被误激活。因此，zero-shot anomaly detection 不仅需要强语义原型，还需要在测试时利用当前图像自身的可信视觉证据来校准 prototypes。

频域或小波线索为这一问题提供了有吸引力但危险的辅助信息。许多工业缺陷表现为局部纹理扰动、边缘断裂或细粒度结构变化，而 Haar 小波能够在局部特征网格上分离高频扰动与低频结构。然而，高频响应并不等价于异常响应。普通物体轮廓、反光、高频背景纹理和正常结构边界同样会产生强小波响应。本文受控消融验证了这一点：在 MVTec 上，direct wavelet fusion / no adaptation 的结果为 88.7 / 80.4 / 92.9 / 96.9，低于固定 baseline 的 91.2 / 83.2 / 91.6 / 96.4；在 VisA 上也从 95.5 / 86.7 / 82.0 / 85.3 降至 94.6 / 85.1 / 81.6 / 84.8。这里的斜杠指标顺序为 P-AUROC / P-AUPRO / I-AUROC / I-AP。

本文提出 WPTA，把小波线索从“最终异常分数”重新定位为“测试时原型校准的可靠性监督”。WPTA 先使用固定 normal/abnormal prototypes 得到初始语义异常分数 `S0`，再在 CLIP patch feature grid 上做 Haar 小波分解，从高频分量估计局部纹理扰动，从低频分量估计结构边界，并构造 boundary-aware reliability `W = norm(HF) * (1 - norm(LF_edge))`。`S0` 提供 evidence 的语义方向，`W` 提供 evidence 的局部频域可靠性，两者共同选择当前测试图像中的 visual abnormal/normal anchors。最后，WPTA 用 conservative update 轻量校准 prototypes，并使用校准后的 prototypes 重新计算 anomaly map。

这一设计对应三个技术挑战。第一，无标签测试图像中的可信 evidence 稀疏且未知，不能仅用高 anomaly score 或高频响应选择 patch。第二，小波线索如果绕过 CLIP 语义判别直接进入最终 anomaly map，会放大正常边界和材质纹理的伪异常。第三，测试时原型校准存在 prototype drift 风险，错误 evidence 会把 prototypes 拉向噪声或正常结构。WPTA 将小波限制在 evidence reliability 层面，并用 conservative update 控制更新强度，使最终异常图仍由 CLIP semantic prototypes 决定。

实验采用两层证据组织。第一层是 MVTec/VisA 受控消融，用于验证 WPTA 机制：semantic-only prototype adaptation 已强于固定原型，wavelet-guided evidence weighting 在 semantic-only 基础上继续提升，conservative update 进一步改善最终结果。第二层是五个工业异常检测基准上的最终校准推理系统结果，包括 MVTec、VisA、MPDD、BTAD 和 DTD-Synthetic；该系统相对固定 AnomalyCLIP baseline 的平均结果从 94.9 / 82.8 / 86.2 / 89.4 提升到 95.9 / 87.4 / 89.5 / 92.2。由于各数据集的最终配置并不完全相同，五数据集表证明的是系统级有效性，WPTA 机制的因果证据限定在 MVTec/VisA 受控消融。

本文贡献如下：

1. 提出 Wavelet-Supervised Test-Time Prototype Adaptation，将 Haar 小波线索从 final map fusion 重新定位为 prototype adaptation 的 patch evidence reliability。
2. 设计 semantic-spectral evidence selection，用初始语义异常分数和 boundary-aware wavelet reliability 共同筛选当前测试图像中的 visual normal/abnormal anchors。
3. 在冻结 CLIP 参数、不使用目标训练数据、不进行反向传播的条件下进行 conservative prototype calibration，并通过 MVTec/VisA 受控消融验证其机制有效性。
4. 报告五个工业基准上的最终校准推理系统结果，并用 configuration audit 明确区分系统级提升和 WPTA 机制证据。

图 1 展示一个 MVTec cable 样例中的固定原型异常图、直接小波响应、boundary-aware reliability、selected evidence 与 WPTA final map。该图用于说明小波线索更适合作为原型适配证据的可靠性监督，而不是直接替代语义异常分数。图 2 展示 WPTA 的整体流程，从冻结 CLIP patch features 和固定文本 prototypes 出发，经过 `S0`、`W`、visual anchors 与 conservative calibration，得到最终 anomaly map。

## 2. 相关工作

### 2.1 CLIP-based zero-shot anomaly detection

CLIP-based anomaly detection 将开放词汇视觉语言表示引入异常检测，使模型能够用文本描述 normality 与 abnormality，而不是为每个目标类别训练专用分类器 \cite{radford2021clip}。WinCLIP 使用 window-level CLIP matching 进行 zero-/few-shot anomaly classification and segmentation \cite{jeong2023winclip}。AnomalyCLIP 学习 object-agnostic normal/abnormal prompts，使 CLIP 更关注异常性而不是前景物体类别 \cite{zhou2024anomalyclip}。AdaCLIP 通过 hybrid learnable prompts 适配 CLIP 到 ZSAD \cite{cao2024adaclip}。CLIP-AD 则从文本特征选择和 staged dual-path 建模角度改善零样本异常检测 \cite{chen2024clipad}。

这些方法证明了 CLIP 语义原型对 ZSAD 的价值，但它们通常把 prompt 或 prototype 作为全局、静态或训练得到的表示使用。对于当前测试图像中的局部缺陷，固定原型缺少 instance-specific visual evidence；而训练式 prompt adaptation 又会引入目标域数据或额外学习过程。WPTA 的位置介于两者之间：它不更新 CLIP 参数，也不需要目标训练图像，而是在单张测试图像内选择可信 patch evidence 来轻量校准 prototypes。

### 2.2 Test-time adaptation for vision-language models

Test-time adaptation 在推理阶段利用测试样本自身的信息缓解分布差异。TPT 在单个测试样本上通过多视图一致性和置信选择进行 test-time prompt tuning，展示了 VLM 在推理阶段适配的可行性 \cite{shu2022tpt}。异常检测中的测试时适配更难，因为异常区域稀疏、未知且常与正常纹理或结构边界相邻。直接把整张图像作为 adaptation 信号容易引入大量正常区域，直接选择高异常 patch 又容易受到伪异常响应干扰。WPTA 因此只聚合被语义先验 `S0` 和小波可靠性 `W` 同时支持的 patch evidence，以降低 prototype drift 风险。

### 2.3 Frequency and wavelet cues for anomaly localization

多分辨率小波表示为图像和特征网格提供低频结构与高频细节分解基础 \cite{daubechies1988orthonormal,mallat1989theory}。在工业视觉中，局部高频扰动常与划痕、裂纹、污染和材料纹理破坏相关；但正常边界、反光和细粒度纹理也会产生强高频响应。因此，频域响应本身不能直接等价为异常概率。本文把小波的作用限制为 patch evidence reliability，并由 CLIP semantic prototypes 继续承担最终异常判别。

### 2.4 Positioning against closest work

| 方法类别 | 文本原型 | 目标域训练 | 测试时原型校准 | 小波可靠性 | 与 WPTA 的区别 |
|---|---:|---:|---:|---:|---|
| AnomalyCLIP \cite{zhou2024anomalyclip} | 固定或学习得到 | 需要训练 prompt | 否 | 否 | 使用 object-agnostic prompts，但不根据当前测试图像 evidence 校准 prototypes。 |
| WinCLIP \cite{jeong2023winclip} | 固定 prompt ensemble | 否或 few-shot variant | 否 | 否 | 强调 window-level scoring，不构造 semantic-spectral visual anchors。 |
| AdaCLIP / CLIP-AD \cite{cao2024adaclip,chen2024clipad} | prompt 或文本特征适配 | 依赖训练或结构改造 | 否 | 否 | 改善 CLIP anomaly scoring，但不使用 boundary-aware wavelet reliability 选择 patch evidence。 |
| TPT-style VLM adaptation \cite{shu2022tpt} | 测试时 prompt 可变 | 否 | 是 | 否 | 面向通用分类适配，未处理异常区域稀疏和局部伪 evidence 问题。 |
| WPTA | 初始固定，测试时校准 | 否 | 是 | 是 | 冻结 CLIP，用 `S0` 与 boundary-aware `W` 选择 visual anchors，并保守校准 prototypes。 |

## 3. 方法

### 3.1 Problem formulation and overview

给定测试图像 `x`，zero-shot anomaly detection 需要输出像素级 anomaly map `M` 和图像级 anomaly score `s_img`。CLIP 图像编码器产生 patch features `{f_i}_{i=1}^N`，文本编码器产生 normal prototype `t_n` 和 abnormal prototype `t_a`。在本文设置中，CLIP 参数冻结，推理阶段不访问目标域训练集，不使用异常标注，也不进行反向传播。

WPTA 包含四个模块。第一，initial semantic scoring 使用固定 `t_n` 和 `t_a` 得到 patch-level semantic prior `S0`。第二，boundary-aware Haar wavelet reliability 在 CLIP patch feature grid 上计算 `W`。第三，semantic-spectral evidence selection 根据 `S0` 与 `W` 选择并聚合 visual normal/abnormal anchors。第四，conservative prototype calibration 用 visual anchors 轻量校准 prototypes，并重新计算 anomaly map。

**Algorithm 1. WPTA inference for one test image.**

```text
Input:
  test image x; frozen CLIP image encoder E_img;
  fixed normal/abnormal text prototypes t_n, t_a;
  temperature tau; evidence top-k ratio k; update strengths alpha, beta.

1. Extract patch features F = E_img(x), with patch vectors {f_i}_{i=1}^N.
2. Compute initial semantic anomaly prior S0(i) from similarities to t_a and t_n.
3. Reshape F into a patch grid and apply Haar DWT.
4. Estimate high-frequency energy HF and low-frequency structural edge LF_edge.
5. Build boundary-aware reliability W = norm(HF) * (1 - norm(LF_edge)).
6. Select abnormal evidence with q_a(i) from S0(i) and W(i).
7. Select normal evidence with q_n(i) from 1 - S0(i) and 1 - W(i).
8. Aggregate visual anchors v_a and v_n from selected evidence patches.
9. If evidence confidence is sufficient:
       t'_a = normalize((1 - alpha) t_a + alpha v_a)
       t'_n = normalize((1 - beta)  t_n + beta  v_n)
   otherwise:
       t'_a = t_a and t'_n = t_n.
10. Recompute patch anomaly scores with t'_a and t'_n, then upsample to M.
Output:
  anomaly map M and image-level anomaly score s_img.
```

### 3.2 Initial semantic anomaly score

WPTA 首先使用固定文本原型计算 patch-level semantic anomaly prior。对每个 patch feature `f_i`，异常概率定义为：

```text
S0(i) = exp(sim(f_i, t_a) / tau) /
        (exp(sim(f_i, t_a) / tau) + exp(sim(f_i, t_n) / tau)).
```

其中 `sim(.,.)` 表示归一化相似度，`tau` 为温度系数。`S0(i)` 提供语义方向：高 `S0(i)` 表示 patch 更接近 abnormal prototype，低 `S0(i)` 表示 patch 更接近 normal prototype。由于固定 prototypes 存在 instance-specific mismatch，`S0` 只作为 evidence selection 的先验，而不是最终校准结果。

### 3.3 Boundary-aware Haar wavelet reliability

WPTA 在 CLIP patch feature grid 上计算小波可靠性，使频域线索与后续 prototype calibration 位于同一特征空间。将 patch features reshape 为 `F in R^{H x W x C}` 后，WPTA 对 `F` 做一级 Haar DWT，得到低频分量 `LL` 与高频分量 `LH`、`HL`、`HH`。高频纹理能量定义为：

```text
HF(i) = mean_c(|LH_i^c| + |HL_i^c| + |HH_i^c|).
```

仅使用 `HF` 会把普通结构边界也当作可靠异常线索，因此 WPTA 从 `LL` 中估计低频结构边缘 `LF_edge(i)`，并构造：

```text
W(i) = norm(HF(i)) * (1 - norm(LF_edge(i))).
```

该式让高频纹理扰动提高 evidence reliability，同时让低频结构边界降低 reliability。由此，小波负责判断 patch evidence 是否可靠，而不是直接产生 anomaly score。

### 3.4 Semantic-spectral evidence selection

WPTA 用 `S0` 与 `W` 共同选择 abnormal 和 normal evidence。异常 evidence 应同时满足语义上接近 abnormal prototype 且具有可靠局部扰动；正常 evidence 应语义上接近 normal prototype 且不被异常纹理或结构边界污染。概念上，evidence weights 写为：

```text
q_a(i) = S0(i)^gamma * rho(W(i)),
q_n(i) = (1 - S0(i))^gamma * rho(1 - W(i)).
```

其中 `gamma` 控制语义先验锐化，`rho(.)` 表示可靠性调制函数。实现中使用 top-k ratio、置信阈值、归一化和 confidence gate 控制 evidence 质量。该模块的作用不是增加一个新的异常图，而是从当前测试图像中构造可靠 visual anchors。

### 3.5 Visual anchors and conservative prototype calibration

WPTA 将 selected evidence patches 聚合为当前图像的 visual abnormal anchor 和 visual normal anchor：

```text
v_a = sum_i q_a(i) f_i / sum_i q_a(i),
v_n = sum_i q_n(i) f_i / sum_i q_n(i).
```

随后，WPTA 以 conservative update 校准文本原型：

```text
t'_a = normalize((1 - alpha) t_a + alpha v_a),
t'_n = normalize((1 - beta)  t_n + beta  v_n).
```

`alpha` 和 `beta` 由 confidence gate 控制。当 evidence 置信度不足时，更新强度会被减小或禁用。受控实验默认实现采用更保守的实例：`alpha0=0.0`、`beta0=0.01`。因此，abnormal evidence 主要用于判断当前图像是否有足够可靠的异常证据来打开更新门控，而实际 prototype displacement 主要发生在 normal prototype 上。本文保留上式作为通用形式，但所有结果解释均以该默认 conservative setting 为准。

### 3.6 Final anomaly scoring

校准后，WPTA 使用 `t'_a` 和 `t'_n` 重新计算 anomaly probability：

```text
S(i) = exp(sim(f_i, t'_a) / tau) /
       (exp(sim(f_i, t'_a) / tau) + exp(sim(f_i, t'_n) / tau)).
```

`S(i)` reshape 并通过 bilinear interpolation 上采样为 `image_size x image_size` 的 anomaly map `M`。随后系统对 `M` 进行 Gaussian smoothing，并用与 baseline 一致的 image/pixel metric implementation 计算结果。若启用 pixel-to-image fusion，图像级 score 会把 CLIP image-text abnormal probability 与 anomaly map 的 top-k pixel score 按固定权重融合；若不启用该模块，则图像级 score 直接来自 image-text probability。受控实验中的 smoothing、feature-layer aggregation、pixel-level metrics 和 image-level aggregation 与固定 baseline 保持一致。

### 3.7 Implementation details

所有实验使用 AnomalyCLIP checkpoint `9_12_4_multiscale/epoch_15.pth` 和 CLIP `ViT-L/14@336px` backbone。输入图像经 bicubic resize 与 center crop 处理到 `518 x 518`，并使用 OpenAI CLIP mean/std 归一化。缓存特征由 feature list `6/12/18/24` 产生；受控 WPTA 消融在 MVTec 和 VisA 上读取同一批缓存 patch features，并聚合 feature map layers `1/2/3`。Layer fusion 使用 sum，layer temperature 为 1.0，Gaussian smoothing 参数 `sigma=5`，AUPRO 使用 200 个阈值步。所有指标均按 image-pixel-level protocol 计算。

Prototype adaptation 默认参数如下。Prototype softmax temperature 为 0.07，semantic sharpening `gamma=1.0`，wavelet evidence exponent `eta=1.0`，evidence top-k ratio 为 0.2。Conservative update 使用 `proto_alpha0=0.0`、`proto_beta0=0.01`、`proto_tau_a=0.15` 和 `proto_update_min_abnormal_confidence=0.06`，并在 confidence 不足时抑制更新。Boundary-aware wavelet reliability 使用 `proto_wavelet_mode=boundary_aware` 和 `proto_wavelet_mix=0.05`；HF-only 与 semantic-only variants 只改变 `proto_wavelet_mode`，no-conservative variant 只关闭 `proto_conservative_update`。Direct fusion 负对照不做 prototype adaptation，而是以 `direct_wavelet_fusion_weight=0.5` 直接融合 `S0` 与小波图。

为保证受控消融中 adaptation variants 的可比性，除 Baseline 外的 prototype/fusion variants 使用相同的 multi-crop 与 pixel-to-image 设置：`multicrop_weight=0.50`，`pixel_to_image_weight=0.10`，`pixel_to_image_topk_ratio=0.01`。Multi-crop fusion 将 base anomaly map 与已缓存 stitched crop map 线性融合；pixel-to-image fusion 使用 anomaly map top-k pixel score 校准 image-level abnormal probability。因此，Table 2 和 Table 3 的因果解释主要来自 adaptation variants 之间的相对比较，而不是把 Baseline 到任一增强行的差值单独归因为 WPTA。

## 4. 实验设置

### 4.1 Datasets

主实验覆盖五个工业异常检测基准：MVTec、VisA、MPDD、BTAD 和 DTD-Synthetic \cite{bergmann2019mvtec,bergmann2021mvtec,zou2022visa,jezek2021deep,mishra2021vt,aota2023zero}。其中 MVTec 和 VisA 用于 WPTA 受控机制消融；五数据集主表用于报告最终校准推理系统的系统级结果。补充医学观察仅作为附录分析，不进入本文主要结论。

### 4.2 Metrics

工业实验报告四个指标：pixel AUROC、pixel AUPRO、image AUROC 和 image AP。所有 slash-form 数值顺序固定为：

```text
P-AUROC / P-AUPRO / I-AUROC / I-AP
```

本文重点分析 P-AUPRO，因为它更直接反映 anomaly localization 的区域质量。I-AUROC 和 I-AP 用于验证图像级检测是否同步受益。所有数值来自确定性评估协议；本文不报告多 seed 置信区间或统计显著性结论。

### 4.3 Baselines, calibrated systems, and evidence boundary

主结果以固定 AnomalyCLIP baseline 作为比较对象，并比较每个数据集的最终校准推理系统。受控消融在 MVTec 和 VisA 上报告四个 variants：direct wavelet fusion / no adaptation、semantic prototype adaptation、wavelet prototype adaptation without conservative update，以及 Full WPTA controlled setting。由于五数据集最终系统与受控消融设置不完全相同，本文显式区分两类证据：Table 1 证明最终校准推理系统的系统级有效性，Table 2 和 Table 3 证明 WPTA 机制。

外部方法比较涉及 auxiliary-target split、fine-tuning 条件和 evaluation protocol 差异，若未经逐项对齐，容易形成不公平结论。因此，本文的主比较限定为固定 AnomalyCLIP baseline 与同协议受控 variants；外部方法数值只在附录中作为协议参考。

## 5. 结果与分析

### 5.1 Main results on five industrial benchmarks

Table 1 展示五个工业数据集上的主结果，比较范围限定为固定 AnomalyCLIP baseline 与每个数据集的最终校准推理系统。最终校准推理系统在所有数据集上均超过该固定 baseline，并且平均四个指标全部提升。由于不同数据集启用的校准模块不同，该表用于评估系统级收益；各数据集最终配置见 Table 4。

**表 1. 五个工业异常检测基准上的主结果。指标越高越好。**

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | I-AUROC ↑ | I-AP ↑ | Δ vs. baseline |
|---|---|---:|---:|---:|---:|---:|
| MVTec | AnomalyCLIP baseline | 91.2 | 83.2 | 91.6 | 96.4 | - |
| MVTec | Calibrated inference system | **91.8** | **85.6** | **94.5** | **97.6** | +0.6 / +2.4 / +2.9 / +1.2 |
| VisA | AnomalyCLIP baseline | 95.5 | 86.7 | 82.0 | 85.3 | - |
| VisA | Calibrated inference system | **96.2** | **91.3** | **84.6** | **87.4** | +0.7 / +4.6 / +2.6 / +2.1 |
| MPDD | AnomalyCLIP baseline | 96.9 | 84.6 | 73.7 | 76.5 | - |
| MPDD | Calibrated inference system | **97.3** | **89.9** | **77.8** | **82.3** | +0.4 / +5.3 / +4.1 / +5.8 |
| BTAD | AnomalyCLIP baseline | 93.5 | 70.5 | 89.1 | 91.0 | - |
| BTAD | Calibrated inference system | **96.3** | **78.2** | **93.9** | **94.9** | +2.8 / +7.7 / +4.8 / +3.9 |
| DTD-Synthetic | AnomalyCLIP baseline | 97.4 | 89.1 | 94.5 | 97.7 | - |
| DTD-Synthetic | Calibrated inference system | **97.9** | **91.8** | **96.9** | **98.7** | +0.5 / +2.7 / +2.4 / +1.0 |
| Average | AnomalyCLIP baseline | 94.9 | 82.8 | 86.2 | 89.4 | - |
| Average | Calibrated inference system | **95.9** | **87.4** | **89.5** | **92.2** | +1.0 / +4.5 / +3.4 / +2.8 |

平均结果显示，最终校准推理系统的最大收益来自 P-AUPRO，平均提升 +4.5。该趋势说明系统主要改善局部异常区域覆盖，而不是只提升图像级分类分数。MPDD 和 BTAD 上的 image-level gains 也较明显，分别在 I-AUROC/I-AP 上提升 +4.1/+5.8 和 +4.8/+3.9。由于 MPDD/BTAD 最终配置没有启用 wavelet reliability，这两个数据集的收益只作为系统级结果解释；WPTA 的机制解释见受控消融。

### 5.2 Core component ablation

Table 2 在 MVTec 和 VisA 的受控消融设置下验证各组件作用。这里的 Full WPTA 是 controlled ablation setting，与五数据集主结果中的最终系统不完全相同。

**表 2. WPTA 核心组件消融。每个结果单元格为 P-AUROC / P-AUPRO / I-AUROC / I-AP。除 Baseline 外，各 prototype/fusion variants 使用相同 multi-crop 与 pixel-to-image 设置；因此因果解释应主要比较 variants 之间的相对变化，而不是把 Baseline 到增强行的完整差值单独归因于 WPTA。**

| Method | MVTec | VisA | 作用 |
|---|---:|---:|---|
| Baseline | 91.2 / 83.2 / 91.6 / 96.4 | 95.5 / 86.7 / 82.0 / 85.3 | fixed AnomalyCLIP prototypes |
| Direct wavelet fusion / no adaptation | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | negative control: use wavelet at final-map level |
| Semantic prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | CLIP semantic evidence only |
| Wavelet prototype adaptation w/o conservative | 91.7 / 85.8 / 93.9 / 97.2 | 96.1 / 91.3 / 84.1 / 87.0 | boundary-aware wavelet evidence, no conservative update |
| Full WPTA controlled setting | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** | boundary-aware wavelet evidence + conservative update |

该消融给出三点结论。第一，direct wavelet fusion 低于 Baseline 和 Full WPTA，说明小波不应直接作为最终异常图。第二，semantic prototype adaptation 明显优于固定原型，说明 test-time prototype calibration 是有效方向。第三，wavelet prototype adaptation w/o conservative 和 Full WPTA 进一步提升 P-AUPRO，说明小波可靠性和 conservative update 均是最终方法链路的一部分。需要注意，除 Baseline 外的 variants 共享 multi-crop 与 pixel-to-image 设置；因此本表最稳妥的因果解释是 variants 之间的相对变化。

### 5.3 Wavelet reliability design

Table 3 进一步分析小波可靠性设计。HF-only W 相比 semantic-only 只有小幅提升，而 boundary-aware W 进一步提高 P-AUPRO，说明低频结构边界抑制对于减少伪高频 evidence 是必要的。

**表 3. 小波可靠性设计消融。每个结果单元格为 P-AUROC / P-AUPRO / I-AUROC / I-AP。**

| Wavelet setting | MVTec | VisA | 解释 |
|---|---:|---:|---|
| Semantic-only prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | no wavelet reliability |
| Direct wavelet fusion | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | wavelet as final-map fusion, negative control |
| HF-only W + prototype adaptation | 91.6 / 85.3 / 94.0 / 97.2 | 96.0 / 90.8 / 84.0 / 86.9 | high-frequency reliability only |
| Boundary-aware W + prototype adaptation | 91.7 / 85.7 / 93.8 / 97.3 | 96.1 / 91.2 / 83.9 / 87.1 | suppress structure-boundary pseudo evidence |
| Full boundary-aware W + conservative | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** | controlled WPTA setting |

Boundary-aware W 在 MVTec/VisA 上分别把 P-AUPRO 从 HF-only 的 85.3/90.8 提升到 85.7/91.2。Full setting 进一步达到 86.2/91.7。这说明高频响应有用但不充分，必须结合低频结构边界抑制和保守更新。

### 5.4 Configuration audit

Table 4 记录五数据集主结果中各最终系统的启用模块。该表的目的不是增加一个新结果，而是明确主结果表的证据边界。

**表 4. 五数据集最终系统配置审计。该表用于限定因果归因。**

| Dataset | Result | Wavelet reliability | Prototype rectification | Multi-crop fusion | Pixel-to-image fusion | Setting note |
|---|---:|---|---|---|---|---|
| MVTec | 91.8 / 85.6 / 94.5 / 97.6 | yes | yes | yes | yes | sigma=5, mc=0.50, p2i=0.10 |
| VisA | 96.2 / 91.3 / 84.6 / 87.4 | yes | yes | yes | yes | sigma=5, mc=0.50, p2i=0.10 |
| MPDD | 97.3 / 89.9 / 77.8 / 82.3 | no | no | yes | yes | sigma=8, mc=0.25, p2i=0.80 |
| BTAD | 96.3 / 78.2 / 93.9 / 94.9 | no | no | yes | yes | sigma=10, mc=0.85, p2i=0.95 |
| DTD-Synthetic | 97.9 / 91.8 / 96.9 / 98.7 | yes | no | yes | yes | sigma=8, mc=0.75, p2i=0.50 |

MVTec 和 VisA 最终系统包含 wavelet reliability、prototype rectification、multi-crop fusion 和 pixel-to-image fusion；DTD-Synthetic 包含 wavelet reliability、multi-crop fusion 和 pixel-to-image fusion；MPDD 和 BTAD 最终系统没有启用 wavelet reliability 或 prototype rectification。因此，五数据集结果体现的是最终校准推理系统的系统级提升，WPTA 机制则由 MVTec/VisA 受控消融验证。

## 6. 讨论

### 6.1 Why wavelet reliability, not wavelet scoring

小波提供的是局部频率结构，而不是语义异常判别。Direct fusion 负对照显示，直接把小波响应加到 anomaly map 会伤害定位表现。WPTA 的设计把小波限制在 evidence reliability 层面，使最终 anomaly map 仍由 CLIP semantic prototypes 决定。这一角色分工解释了为什么小波能帮助 prototype adaptation，同时避免频域响应的伪异常问题。

### 6.2 Why conservative calibration

无标签测试时校准容易产生 prototype drift。WPTA 使用 confidence-gated conservative update，使 visual anchors 只有在 evidence 足够可靠时影响 prototypes。受控消融中 Full WPTA 相比 no-conservative variant 进一步提升，说明保守更新对最终性能有贡献。在默认设置中，abnormal prototype 的更新强度为 0，abnormal evidence 主要负责门控，实际位移主要发生在 normal prototype 上；这一点也解释了本文为什么把方法描述为 conservative calibration，而不是通用的双向大幅原型更新。

### 6.3 Scope and limitations

本文结论有三个边界。第一，五数据集主表中的最终校准推理系统使用了数据集特定配置，因此它证明的是相对固定 AnomalyCLIP baseline 的系统级收益，不能被解释为 WPTA 机制在所有五个数据集上完成了因果隔离。第二，WPTA 的机制证据来自 MVTec/VisA 受控消融；若要扩大机制结论，需要在更多数据集上补充同配置消融。第三，由于外部方法的训练条件、输入分辨率、后处理和评价脚本并不总是可直接对齐，本文不主张外部方法排序。

## 7. 结论

本文提出 WPTA，用于 CLIP-based zero-shot anomaly detection。WPTA 将 Haar 小波线索作为 patch evidence reliability 来监督测试时原型校准，而不是把频域响应直接融合到最终 anomaly map 中。受控 MVTec/VisA 消融说明，直接小波融合会损害定位表现，而 semantic-spectral evidence selection 和 conservative prototype calibration 能改善测试时原型适配。五个工业基准上的最终校准推理系统相对固定 AnomalyCLIP baseline 在平均 P-AUPRO 上获得 +4.5 的提升，显示校准推理栈在异常区域定位上具有明确收益。整体而言，本文支持一种更保守但可复现的结论：小波线索作为 evidence reliability 比作为最终异常分数更合适，而测试时原型校准可以缓解固定 CLIP 原型与实例化缺陷外观之间的不匹配。

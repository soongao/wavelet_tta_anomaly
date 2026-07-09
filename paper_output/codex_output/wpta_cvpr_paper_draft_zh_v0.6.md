# Wavelet-Supervised Test-Time Prototype Adaptation for Zero-Shot Anomaly Detection

中文顶会论文稿 v0.6

## 当前稿件状态

本稿用于内部技术审阅，目标写作标准对齐 CVPR/ICCV 类计算机视觉顶会。中文仅用于便于审阅，不降低证据、引用、图表和实验要求。本文当前不声称 SOTA，不使用未完成实验补数，不把不同数据集的 final setting 统一归因为同一机制。`runtime` 和 `normal stability` 已从本文主线删除。

当前主证据来自 `/Users/bytedance/code/AnomalyCLIP/paper/result_record/result_table.csv` 中标记为 `current` 的工业结果。表格包见 `outputs/wpta_generated_tables_v0.4.md` 和 `outputs/wpta_generated_tables_latex_v0.4.tex`。引用核验状态见 `outputs/wpta_citation_ledger_v0.1.md`。本文仍保留少量 `CITATION-PENDING` 和 `FIGURE_PROMPT`，这些是投稿前必须关闭的 gate。

## 摘要

零样本异常检测要求模型在不使用目标类别训练图像和异常标注的条件下，同时完成图像级异常判别和像素级异常定位。现有 CLIP-based 方法通常依赖固定 normal/abnormal 文本原型，但固定原型难以适配每张测试图像中的实例化缺陷外观，并容易把真实局部异常、正常材料纹理和结构边界混淆。本文的核心观察是，小波线索不适合作为最终异常图的直接加性分数，却适合作为测试时原型适配中的 patch evidence reliability。基于这一观察，本文提出 Wavelet-Supervised Test-Time Prototype Adaptation (WPTA)，在冻结 CLIP 参数的前提下，从 CLIP patch feature grid 构造 boundary-aware Haar wavelet reliability，并与初始语义异常分数共同选择 visual normal/abnormal anchors，以 conservative update 校准 prototypes。受控 MVTec/VisA 消融表明，direct wavelet fusion 会降低定位质量，而 semantic-spectral evidence selection 与 conservative calibration 能持续改善 prototype adaptation；五个工业基准上的最终校准系统相对固定 AnomalyCLIP baseline 平均提升 +1.0 / +4.5 / +3.4 / +2.8，指标顺序为 P-AUROC / P-AUPRO / I-AUROC / I-AP。该结果支撑 final calibrated system 的系统级有效性；WPTA 机制本身的因果证据来自 MVTec/VisA 受控消融。

关键词：zero-shot anomaly detection；CLIP；test-time adaptation；prototype calibration；wavelet reliability；industrial inspection

## 1. 引言

工业异常检测需要在产品图像中发现划痕、缺口、污染、破损和局部纹理异常，并同时输出图像级异常判别和像素级异常定位。一个典型难例是金属件边缘附近的细划痕：缺陷区域很小，但正常轮廓边缘、反光和背景纹理也会产生强局部响应。理想检测器必须覆盖真实划痕，同时避免把正常结构边界误判为异常。由于工业场景中异常样本稀缺、类别更新频繁且像素级标注成本高，zero-shot anomaly detection 成为实际部署中有价值的设置。视觉语言模型 CLIP 提供了开放词汇图文对齐能力，WinCLIP、AnomalyCLIP、AdaCLIP 和 CLIP-AD 等工作进一步将 normal/abnormal 文本提示或原型用于 anomaly classification 与 segmentation \cite{radford2021clip,jeong2023winclip,zhou2024anomalyclip,cao2024adaclip,chen2024clipad}。

固定 normal/abnormal prototypes 的核心问题是 instance-specific mismatch。一个通用 abnormal prototype 可以表达“异常”这个语义方向，但难以精确覆盖当前图像中的具体缺陷形态，例如细划痕、瓶口缺口、纺织纹理污染或金属表面压痕。对于 CLIP-based anomaly detection，这种不匹配会直接出现在 patch anomaly map 中：真实缺陷可能因为与通用 abnormal prototype 相似度不足而被低估，正常边界或材料纹理也可能因局部视觉差异被误激活。因此，zero-shot anomaly detection 不仅需要强语义原型，还需要在测试时利用当前图像自身的可信视觉证据来校准 prototypes。

频域或小波线索为这一问题提供了有吸引力但危险的辅助信息。许多工业缺陷表现为局部纹理扰动、边缘断裂或细粒度结构变化，而 Haar 小波能够在局部特征网格上分离高频扰动与低频结构。然而，高频响应并不等价于异常响应。普通物体轮廓、反光、高频背景纹理和正常结构边界同样会产生强小波响应。本文受控消融验证了这一点：在 MVTec 上，direct wavelet fusion / no adaptation 的结果为 88.7 / 80.4 / 92.9 / 96.9，低于固定 baseline 的 91.2 / 83.2 / 91.6 / 96.4；在 VisA 上也从 95.5 / 86.7 / 82.0 / 85.3 降至 94.6 / 85.1 / 81.6 / 84.8。这里的斜杠指标顺序为 P-AUROC / P-AUPRO / I-AUROC / I-AP。

本文提出 WPTA，把小波线索从“最终异常分数”重新定位为“测试时原型校准的可靠性监督”。WPTA 先使用固定 normal/abnormal prototypes 得到初始语义异常分数 `S0`，再在 CLIP patch feature grid 上做 Haar 小波分解，从高频分量估计局部纹理扰动，从低频分量估计结构边界，并构造 boundary-aware reliability `W = norm(HF) * (1 - norm(LF_edge))`。`S0` 提供 evidence 的语义方向，`W` 提供 evidence 的局部频域可靠性，两者共同选择当前测试图像中的 visual abnormal/normal anchors。最后，WPTA 用 conservative update 轻量校准 prototypes，并使用校准后的 prototypes 重新计算 anomaly map。

这一设计直接对应三个技术挑战。第一，无标签测试图像中的可信 evidence 稀疏且未知，不能仅用高 anomaly score 或高频响应选择 patch。第二，小波线索如果绕过 CLIP 语义判别直接进入最终 anomaly map，会放大正常边界和材质纹理的伪异常。第三，测试时原型校准存在 prototype drift 风险，错误 evidence 会把 prototypes 拉向噪声或正常结构。WPTA 将小波限制在 evidence reliability 层面，并用 conservative update 控制更新强度，使最终异常图仍由 CLIP semantic prototypes 决定。

实验采用两层证据组织。第一层是 MVTec/VisA 受控消融，用于验证 WPTA 机制：semantic-only prototype adaptation 已强于固定原型，wavelet-guided evidence weighting 在 semantic-only 基础上继续提升，conservative update 进一步改善最终结果。第二层是五个工业异常检测基准上的 final calibrated system 结果，包括 MVTec、VisA、MPDD、BTAD 和 DTD-Synthetic；最终系统相对固定 AnomalyCLIP baseline 的平均结果从 94.9 / 82.8 / 86.2 / 89.4 提升到 95.9 / 87.4 / 89.5 / 92.2。由于各数据集 final setting 并不完全相同，五数据集表只证明系统级有效性，WPTA 机制的因果证据限定在 MVTec/VisA 受控消融。

本文贡献如下：

1. 提出 Wavelet-Supervised Test-Time Prototype Adaptation，将 Haar 小波线索从 final map fusion 重新定位为 prototype adaptation 的 patch evidence reliability。
2. 设计 semantic-spectral evidence selection，用初始语义异常分数和 boundary-aware wavelet reliability 共同筛选当前测试图像中的 visual normal/abnormal anchors。
3. 在冻结 CLIP 参数、不使用目标训练数据、不进行反向传播的条件下进行 conservative prototype calibration，并通过 MVTec/VisA 受控消融验证其机制有效性。
4. 报告五个工业基准上的 final calibrated system 结果，并用 final-setting 配置审计表明确区分系统级提升和 WPTA 机制证据。

[FIGURE_PROMPT:figure1_motivated_example]
目标：生成 Figure 1，用一个真实工业样例展示 fixed prototype mismatch、direct wavelet fusion failure 和 WPTA 的区别。建议使用金属件边缘附近细划痕或 VisA 小缺陷样例。四栏横向布局：Input image；AnomalyCLIP initial map，标出真实缺陷低估与正常边界误激活；Direct wavelet fusion map，标出高频边界伪激活；WPTA evidence and final map，显示 `S0`、`W = HF * (1 - LF_edge)`、selected abnormal/normal evidence patches、visual anchors 和 final map。所有热图必须来自真实模型输出；异常 evidence 用橙色边框，normal evidence 用蓝色边框；字体不小于 8pt；白底矢量风格。
[/FIGURE_PROMPT]

## 2. 相关工作

### 2.1 CLIP-based zero-shot anomaly detection

CLIP-based anomaly detection 将开放词汇视觉语言表示引入异常检测，使模型能够用文本描述 normality 与 abnormality，而不是为每个目标类别训练专用分类器 \cite{radford2021clip}。WinCLIP 使用 window-level CLIP matching 进行 zero-/few-shot anomaly classification and segmentation \cite{jeong2023winclip}。AnomalyCLIP 学习 object-agnostic normal/abnormal prompts，使 CLIP 更关注异常性而不是前景物体类别 \cite{zhou2024anomalyclip}。AdaCLIP 通过 hybrid learnable prompts 适配 CLIP 到 ZSAD \cite{cao2024adaclip}。CLIP-AD 则从文本特征选择和 staged dual-path 建模角度改善零样本异常检测 \cite{chen2024clipad}。这些方法表明 CLIP 语义原型对 ZSAD 有价值，但大多仍依赖固定或训练得到的 prompt/prototype，缺少针对当前测试图像局部缺陷证据的 training-free prototype calibration。

### 2.2 Test-time adaptation for vision-language models

Test-time adaptation 在推理阶段利用测试样本自身的信息缓解分布差异。TPT 在单个测试样本上通过多视图一致性和置信选择进行 test-time prompt tuning，展示了 VLM 在推理阶段适配的可行性 \cite{shu2022tpt}。异常检测中的测试时适配更难，因为异常区域稀疏、未知且常与正常纹理或结构边界相邻。WPTA 不更新 CLIP 参数，也不把整张图像直接作为 adaptation 信号；它只聚合被 `S0` 和 `W` 同时支持的 patch evidence，以降低 prototype drift 风险。

### 2.3 Frequency and wavelet cues for anomaly localization

频域分析和小波变换长期用于纹理分析和表面缺陷建模，但直接把频域响应作为异常分数会受到正常结构边界和材质纹理的干扰。[CITATION-PENDING: wavelet texture inspection] 本文不把小波用作最终 anomaly score，而是把小波作为 patch evidence reliability。Table 2 和 Table 3 中 direct wavelet fusion 明显低于 WPTA，说明小波线索的角色定位是本文方法成立的关键。

### 2.4 Positioning against closest work

| 方法类别 | 文本原型 | 测试时原型校准 | 小波可靠性 | 与 WPTA 的区别 |
|---|---:|---:|---:|---|
| AnomalyCLIP \cite{zhou2024anomalyclip} | 固定或学习得到 | 否 | 否 | 使用 object-agnostic prompts，但不根据当前测试图像 evidence 校准 prototypes。 |
| WinCLIP \cite{jeong2023winclip} | 固定 prompt ensemble | 否 | 否 | 强调 window-level scoring，不构造 semantic-spectral visual anchors。 |
| AdaCLIP / CLIP-AD \cite{cao2024adaclip,chen2024clipad} | prompt 或文本特征适配 | 部分依赖训练或结构改造 | 否 | 改善 CLIP anomaly scoring，但不使用 boundary-aware wavelet reliability 选择 patch evidence。 |
| TPT-style VLM adaptation \cite{shu2022tpt} | 测试时 prompt 可变 | 是 | 否 | 面向通用分类适配，未处理异常区域稀疏和局部伪 evidence 问题。 |
| WPTA | 初始固定，测试时校准 | 是 | 是 | 用 `S0` 与 boundary-aware `W` 选择 visual anchors，并保守校准 prototypes。 |

## 3. 方法

### 3.1 Problem formulation and overview

给定测试图像 `x`，zero-shot anomaly detection 需要输出像素级 anomaly map `M` 和图像级 anomaly score `s_img`。CLIP 图像编码器产生 patch features `{f_i}_{i=1}^N`，文本编码器产生 normal prototype `t_n` 和 abnormal prototype `t_a`。在本文设置中，CLIP 参数冻结，推理阶段不访问目标域训练集，不使用异常标注，也不进行反向传播。

WPTA 包含四个模块。第一，initial semantic scoring 使用固定 `t_n` 和 `t_a` 得到 patch-level semantic prior `S0`。第二，boundary-aware Haar wavelet reliability 在 CLIP patch feature grid 上计算 `W`。第三，semantic-spectral evidence selection 根据 `S0` 与 `W` 选择并聚合 visual normal/abnormal anchors。第四，conservative prototype calibration 用 visual anchors 轻量校准 prototypes，并重新计算 anomaly map。

[FIGURE_PROMPT:figure2_method_overview]
目标：生成 Figure 2 方法总览图。横向 pipeline：Input image and frozen CLIP features；Initial semantic anomaly score `S0`；Haar DWT on patch feature grid；Boundary-aware reliability `W = norm(HF) * (1 - norm(LF_edge))`；Semantic-spectral evidence selection；Visual anchors `v_a`, `v_n`；Conservative prototype calibration；Final anomaly map。图中必须标注 “CLIP frozen”、“no backpropagation”、“wavelet is reliability, not final score”。模块名称与方法小节一致，使用矢量图风格。
[/FIGURE_PROMPT]

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

`alpha` 和 `beta` 由 confidence gate 控制。当 evidence 置信度不足时，更新强度会被减小或禁用。该设计限制 prototype drift，使测试时校准保持 training-free 和 annotation-free。

### 3.6 Final anomaly scoring

校准后，WPTA 使用 `t'_a` 和 `t'_n` 重新计算 anomaly probability：

```text
S(i) = exp(sim(f_i, t'_a) / tau) /
       (exp(sim(f_i, t'_a) / tau) + exp(sim(f_i, t'_n) / tau)).
```

`S(i)` reshape 并上采样为 anomaly map `M`。图像级 score `s_img` 使用与 baseline 一致的 aggregation protocol。最终英文投稿版需要在 implementation details 或 appendix 中逐项说明 upsampling、smoothing、score aggregation 和 cache generation，以保证可复现。

### 3.7 Implementation details

受控 WPTA 消融在 MVTec 和 VisA 上使用相同缓存特征与评价设置。实验使用 AnomalyCLIP 缓存的 patch features，并聚合 feature map layers 1/2/3；layer fusion 使用 sum，layer temperature 为 1.0，Gaussian smoothing 参数 `sigma=5`，AUPRO 使用 200 个阈值步。所有指标均按 image-pixel-level protocol 计算。

Prototype adaptation 默认参数来自 `conf/run_prototype_ablation_experiments_conf.yaml`。Prototype softmax temperature 为 0.07，semantic sharpening `gamma=1.0`，wavelet evidence exponent `eta=1.0`，evidence top-k ratio 为 0.2。Conservative update 使用 `proto_alpha0=0.0`、`proto_beta0=0.01`、`proto_tau_a=0.15` 和 `proto_update_min_abnormal_confidence=0.06`，并在 confidence 不足时抑制更新。Boundary-aware wavelet reliability 使用 `proto_wavelet_mode=boundary_aware` 和 `proto_wavelet_mix=0.05`；HF-only 与 semantic-only variants 只改变 `proto_wavelet_mode`，no-conservative variant 只关闭 `proto_conservative_update`。Direct fusion 负对照不做 prototype adaptation，而是以 `direct_wavelet_fusion_weight=0.5` 直接融合 `S0` 与小波图。

为保证受控消融中 adaptation variants 的可比性，除 Baseline 外的 prototype/fusion variants 使用相同的 multi-crop 与 pixel-to-image 设置：`multicrop_weight=0.50`，`pixel_to_image_weight=0.10`，`pixel_to_image_topk_ratio=0.01`。因此，Table 2 和 Table 3 的因果解释主要来自 adaptation variants 之间的相对比较，而不是把 Baseline 到任一增强行的差值单独归因为 WPTA。五数据集 final systems 使用数据集特定配置，已在 Table 4 中单独审计；完整运行命令见 `/Users/bytedance/code/AnomalyCLIP/FIVE_DATASET_RESULTS_AND_ABLATIONS.md`。

## 4. 实验设置

### 4.1 Datasets

主实验覆盖五个工业异常检测基准：MVTec、VisA、MPDD、BTAD 和 DTD-Synthetic \cite{bergmann2019mvtec,bergmann2021mvtec,zou2022visa,jezek2021deep,mishra2021vt,aota2023zero}。其中 MVTec 和 VisA 用于 WPTA 受控机制消融；五数据集主表用于报告 final calibrated system 的系统级结果。补充医学观察只报告 ISIC/ISBI 的 pixel-level 结果，其它医学数据集当前未形成完整可比结果，因此医学结果不进入摘要、贡献或主 claim。

说明：`zou2022visa`、`jezek2021deep`、`mishra2021vt` 和 `aota2023zero` 当前来自本地已有 `.bib`，最终投稿前仍需按 `outputs/wpta_citation_ledger_v0.1.md` 完成 canonical metadata 核验。

### 4.2 Metrics

工业实验报告四个指标：pixel AUROC、pixel AUPRO、image AUROC 和 image AP。所有 slash-form 数值顺序固定为：

```text
P-AUROC / P-AUPRO / I-AUROC / I-AP
```

本文重点分析 P-AUPRO，因为它更直接反映 anomaly localization 的区域质量。I-AUROC 和 I-AP 用于验证图像级检测是否同步受益。当前结果为 deterministic report，尚未包含多 seed 置信区间或显著性检验；因此本文不报告显著性 claim。

### 4.3 Baselines, final systems, and evidence boundary

主结果以固定 AnomalyCLIP baseline 作为比较对象，并比较每个数据集的 final calibrated system。受控消融在 MVTec 和 VisA 上报告四个 variants：direct wavelet fusion / no adaptation、semantic prototype adaptation、wavelet prototype adaptation without conservative update，以及 Full WPTA controlled setting。由于五数据集 final systems 与受控消融设置不完全相同，本文显式区分两类证据：Table 1 证明 final calibrated system 的系统级有效性，Table 2 和 Table 3 证明 WPTA 机制。

## 5. 结果与分析

### 5.1 Main results on five industrial benchmarks

Table 1 展示五个工业数据集上的主结果。最终校准系统在所有数据集上均超过固定 AnomalyCLIP baseline，并且平均四个指标全部提升。该表不单独证明所有提升都来自 WPTA；各数据集 final setting 的启用模块见 Table 4。

**表 1. 五个工业异常检测基准上的主结果。指标越高越好。**

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | I-AUROC ↑ | I-AP ↑ | Δ vs. baseline |
|---|---|---:|---:|---:|---:|---:|
| MVTec | AnomalyCLIP baseline | 91.2 | 83.2 | 91.6 | 96.4 | - |
| MVTec | Final calibrated system | **91.8** | **85.6** | **94.5** | **97.6** | +0.6 / +2.4 / +2.9 / +1.2 |
| VisA | AnomalyCLIP baseline | 95.5 | 86.7 | 82.0 | 85.3 | - |
| VisA | Final calibrated system | **96.2** | **91.3** | **84.6** | **87.4** | +0.7 / +4.6 / +2.6 / +2.1 |
| MPDD | AnomalyCLIP baseline | 96.9 | 84.6 | 73.7 | 76.5 | - |
| MPDD | Final calibrated system | **97.3** | **89.9** | **77.8** | **82.3** | +0.4 / +5.3 / +4.1 / +5.8 |
| BTAD | AnomalyCLIP baseline | 93.5 | 70.5 | 89.1 | 91.0 | - |
| BTAD | Final calibrated system | **96.3** | **78.2** | **93.9** | **94.9** | +2.8 / +7.7 / +4.8 / +3.9 |
| DTD-Synthetic | AnomalyCLIP baseline | 97.4 | 89.1 | 94.5 | 97.7 | - |
| DTD-Synthetic | Final calibrated system | **97.9** | **91.8** | **96.9** | **98.7** | +0.5 / +2.7 / +2.4 / +1.0 |
| Average | AnomalyCLIP baseline | 94.9 | 82.8 | 86.2 | 89.4 | - |
| Average | Final calibrated system | **95.9** | **87.4** | **89.5** | **92.2** | +1.0 / +4.5 / +3.4 / +2.8 |

平均结果显示，final calibrated system 的最大收益来自 P-AUPRO，平均提升 +4.5。该趋势说明当前系统主要改善局部异常区域覆盖，而不是只提升图像级分类分数。MPDD 和 BTAD 上的 image-level gains 也较明显，分别在 I-AUROC/I-AP 上提升 +4.1/+5.8 和 +4.8/+3.9。由于 MPDD/BTAD final commands 主要启用 multi-crop fusion 和 pixel-to-image fusion，本段只作为系统级结果解释；WPTA 的机制解释见受控消融。

### 5.2 Core component ablation

Table 2 在 MVTec 和 VisA 的受控消融设置下验证各组件作用。这里的 Full WPTA 是 controlled ablation setting，与五数据集主结果中的 final systems 不完全相同。

**表 2. WPTA 核心组件消融。每个结果单元格为 P-AUROC / P-AUPRO / I-AUROC / I-AP。**

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
| Full boundary-aware W + conservative | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** | final controlled WPTA setting |

Boundary-aware W 在 MVTec/VisA 上分别把 P-AUPRO 从 HF-only 的 85.3/90.8 提升到 85.7/91.2。Full setting 进一步达到 86.2/91.7。这说明高频响应有用但不充分，必须结合低频结构边界抑制和保守更新。

### 5.4 Final-system configuration audit

Table 4 记录五数据集主结果中各 final run 的启用模块。该表的目的不是增加一个新结果，而是明确主结果表的证据边界。

**表 4. 五数据集 final system 配置审计。该表用于限定因果归因。**

| Dataset | Final result | Wavelet reliability | TTA rectification | Multi-crop fusion | Pixel-to-image fusion | Key setting | Result log |
|---|---:|---|---|---|---|---|---|
| MVTec | 91.8 / 85.6 / 94.5 / 97.6 | yes | yes | yes | yes | sigma=5, mc=0.50, p2i=0.10 | `mvtec/07_full_method/log.txt` |
| VisA | 96.2 / 91.3 / 84.6 / 87.4 | yes | yes | yes | yes | sigma=5, mc=0.50, p2i=0.10 | `visa/07_full_method/log.txt` |
| MPDD | 97.3 / 89.9 / 77.8 / 82.3 | no | no | yes | yes | sigma=8, mc=0.25, p2i=0.80 | `mpdd_multicrop_partial_w025_sigma8_p2i/log.txt` |
| BTAD | 96.3 / 78.2 / 93.9 / 94.9 | no | no | yes | yes | sigma=10, mc=0.85, p2i=0.95, p2i-topk=0.30 | `btad_full_multicrop_w085_sigma10_p2i030_w095/log.txt` |
| DTD-Synthetic | 97.9 / 91.8 / 96.9 / 98.7 | yes | no | yes | yes | sigma=8, mc=0.75, p2i=0.50, p2i-topk=0.002 | `dtd_final_no_strat_woven127_w075_sigma8_p2i0002_w05/log.txt` |

MVTec 和 VisA final system 包含 wavelet reliability、TTA rectification、multi-crop fusion 和 pixel-to-image fusion；DTD-Synthetic 包含 wavelet reliability、multi-crop fusion 和 pixel-to-image fusion；MPDD 和 BTAD final system 没有启用 wavelet/TTA flags。因此，五数据集结果应写成 final calibrated system 的系统级提升，WPTA 机制则由 MVTec/VisA 受控消融验证。

### 5.5 Appendix results and protocol-reference comparison

MVTec/VisA 系统校准栈消融、ISIC/ISBI 医学补充结果和外部方法候选比较已生成在 `outputs/wpta_generated_tables_v0.4.md`。这些表当前不进入主 claim。系统校准栈表用于解释 final calibrated system 的工程模块；ISIC/ISBI 只作为 appendix-style observation；外部方法候选比较只作为 protocol-reference，因为 external numbers 来自另一份 `main.tex`，而当前 WPTA 稿件强调 training-free test-time adaptation。正式投稿前，若要把外部方法比较放入主文，必须核验 split、backbone、input resolution、preprocessing、post-processing、prompt setting、是否使用 auxiliary training/fine-tuning、evaluation script 和 metric implementation。

[TABLE_DATA_PROMPT:external_protocol_verified_table]
请核验并生成一个可进入主文的外部方法比较表。候选方法包括 CLIP、WinCLIP、VAND、CoOp、AdaCLIP、AnomalyCLIP、AA-CLIP、TAAP/INPC、Current final system；候选数据集包括 MVTec AD、VisA、MPDD、BTAD、DTD-Synthetic。逐项核验 split、backbone、input resolution、preprocessing、post-processing、prompt setting、是否使用 auxiliary training/fine-tuning、evaluation script 和 metric implementation。输出 CSV 字段：`dataset, method, p_auroc, p_aupro, i_auroc, i_ap, source_file_or_paper, source_line_or_table, citation_key, protocol_match, mismatch_notes, allowed_placement`。只有 `protocol_match=yes` 的行可以进入主文；`partial/no` 的行只能进入附录或删除。
[/TABLE_DATA_PROMPT]

### 5.6 Qualitative visualization plan

定性图必须来自真实模型输出，不能用手工热图替代。至少需要覆盖 MVTec、VisA、MPDD、BTAD、DTD-Synthetic 中四个数据集，并包含 Input image、GT mask、AnomalyCLIP baseline map、Boundary-aware wavelet reliability、selected evidence patches 和 WPTA/final map。

[FIGURE_PROMPT:figure3_qualitative]
目标：生成 qualitative visualization。每行一个真实 case，列为 Input image、GT mask、AnomalyCLIP baseline map、Boundary-aware wavelet reliability W、Selected evidence patches、WPTA/final map。至少覆盖 MVTec、VisA、MPDD、BTAD、DTD-Synthetic 中 4 个数据集。Baseline 与 final map 使用相同色标，W 使用 viridis，abnormal evidence 用橙色边框，normal evidence 用蓝色边框。所有热图必须来自真实模型输出，不能手工涂色。图注第一句必须说明核心观察：WPTA 的 final map 相比 baseline 更聚焦真实缺陷，同时降低结构边界伪激活。
[/FIGURE_PROMPT]

## 6. 讨论

### 6.1 Why wavelet reliability, not wavelet scoring

小波提供的是局部频率结构，而不是语义异常判别。Direct fusion 负对照显示，直接把小波响应加到 anomaly map 会伤害定位表现。WPTA 的设计把小波限制在 evidence reliability 层面，使最终 anomaly map 仍由 CLIP semantic prototypes 决定。这一角色分工解释了为什么小波能帮助 prototype adaptation，同时避免频域响应的伪异常问题。

### 6.2 Why conservative calibration

无标签测试时校准容易产生 prototype drift。WPTA 使用 confidence-gated conservative update，使 visual anchors 只有在 evidence 足够可靠时影响 prototypes。受控消融中 Full WPTA 相比 no-conservative variant 进一步提升，说明保守更新对最终性能有稳定贡献。

### 6.3 Current limitations

当前稿件仍有四个投稿前限制。第一，引用尚未完全闭环：核心 CLIP/AnomalyCLIP/WinCLIP/AdaCLIP/TPT/MVTec 引用已有元数据核验，但部分数据集和 wavelet background 仍需最终 BibTeX 与 canonical source。第二，真实 qualitative visualization 尚未生成，Figure 1 和 Figure 3 仍是 prompt。第三，外部方法比较尚未完成协议核验，不能用于主文强比较或 SOTA claim。第四，当前实现细节虽然已经从配置文件提取，但英文投稿版仍需要逐项核对 backbone、input size、cache generation、upsampling、smoothing 和 image-score aggregation。

## 7. 结论

本文提出 WPTA，用于 CLIP-based zero-shot anomaly detection。WPTA 将 Haar 小波线索作为 patch evidence reliability 来监督测试时原型校准，而不是把频域响应直接融合到最终 anomaly map 中。受控 MVTec/VisA 消融说明，直接小波融合会损害定位表现，而 semantic-spectral evidence selection 和 conservative prototype calibration 能改善测试时原型适配。五个工业基准上的 final calibrated system 相对固定 AnomalyCLIP baseline 在平均 P-AUPRO 上获得 +4.5 的提升，显示当前系统在异常区域定位上具有明确收益。后续投稿版本需要关闭引用、真实图和协议核验 gate，才能形成完整顶会证据链。

## 8. Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Final calibrated system improves fixed AnomalyCLIP baseline on five industrial datasets. | Table 1, all five datasets improve over baseline; Table 4 records per-dataset final settings. | Supported |
| Main system-level gain is P-AUPRO. | Table 1 average P-AUPRO gain +4.5, larger than P-AUROC +1.0, I-AUROC +3.4 and I-AP +2.8. | Supported |
| Direct wavelet fusion is not sufficient. | Table 2 and Table 3, direct fusion is below Baseline and Full in P-AUPRO. | Supported |
| Prototype adaptation is useful in the controlled MVTec/VisA setting. | Table 2, semantic adaptation improves over Baseline on MVTec/VisA. | Supported with scope |
| Wavelet reliability adds value beyond semantic-only adaptation in the controlled MVTec/VisA setting. | Table 2, wavelet no-conservative improves P-AUPRO over semantic-only on both datasets. | Supported with scope |
| Boundary-aware W improves over HF-only W in the controlled MVTec/VisA setting. | Table 3, boundary-aware W improves P-AUPRO over HF-only on both datasets. | Supported with scope |
| WPTA mechanism is validated across all five industrial datasets. | Table 4 shows MPDD/BTAD final runs do not enable wavelet/TTA flags. | Not supported; keep mechanism claim limited to MVTec/VisA controlled ablations |
| Current system reaches SOTA. | External comparison protocol is not fully verified. | Not supported; do not claim |
| Current system has preliminary positive observation beyond industrial data. | ISIC/ISBI improves from 88.7 / 78.6 to 89.9 / 80.0, but other medical datasets are incomplete. | Appendix-only observation |

## 9. 投稿前 gate

- [ ] 用 canonical metadata 替换所有 `CITATION-PENDING`，并生成最终 `.bib`。
- [ ] 生成真实 Figure 1/3，可视化 input、GT、baseline map、wavelet reliability、selected evidence 和 final map。
- [ ] 若保留外部方法比较，完成 protocol verification table；否则只放附录或删除。
- [ ] 核对 implementation details：backbone、input size、feature cache、upsampling、smoothing、image-level score aggregation。
- [ ] 若要报告统计显著性或置信区间，补充多 run/multi seed 或明确 deterministic-only。

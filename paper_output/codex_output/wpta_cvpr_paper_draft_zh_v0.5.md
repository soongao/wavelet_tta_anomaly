# Wavelet-Supervised Test-Time Prototype Adaptation for Zero-Shot Anomaly Detection

中文顶会论文稿 v0.5

## 写作边界

本文稿以 `/Users/bytedance/code/AnomalyCLIP/paper/result_record/result_table.csv` 中标记为 `current` 的工业结果为主证据。`/Users/bytedance/code/AnomalyCLIP/paper/result_record/medical_result_table.csv` 中 ISIC/ISBI 的 `current` 结果只作为附录式观察，不进入主 claim。本文不编造 qualitative visualization，不声称 SOTA。候选外部方法对比来自 `outputs/iconip_main_results_extracted_analysis.md`，目前仅作为协议参考，正式投稿前必须核验 split、backbone、输入尺寸、后处理和 evaluation script。

证据边界：WPTA 机制本身由 MVTec/VisA 受控消融支撑；五数据集主结果支撑的是最终校准系统相对固定 AnomalyCLIP baseline 的系统级提升。本文不把 MPDD/BTAD 的最终提升归因于 wavelet-supervised prototype adaptation，因为这两个数据集的最终命令没有启用 wavelet/TTA flags。

文献引用尚未完成 DOI/BibTeX 核验，因此以 `[CITATION-NEEDED: ...]` 标注。中文稿用于内部审阅，英文 LaTeX 投稿版需要把所有占位引用替换为真实 `\cite{}`。

## 摘要

零样本异常检测要求模型在没有目标类别训练图像和异常标注的情况下，同时完成图像级异常判别与像素级异常定位。现有 CLIP-based 方法通常依赖固定 normal/abnormal 文本原型，但固定原型难以适配每张测试图像中的具体缺陷形态，并容易混淆真实局部异常、正常材料纹理和物体结构边界。本文提出 Wavelet-Supervised Test-Time Prototype Adaptation (WPTA)，一种无需训练的测试时原型校准方法，其核心思想是把 Haar 小波线索用作 prototype adaptation 的 patch evidence reliability，而不是把频域响应直接融合到最终 anomaly map 中。WPTA 在 CLIP patch feature grid 上构造 boundary-aware wavelet reliability，并与初始语义异常分数共同选择可信 normal/abnormal evidence patches，聚合为 visual anchors 后以 conservative update 校准 prototypes。受控 MVTec/VisA 消融表明，direct wavelet fusion 会降低定位表现，而 semantic-spectral evidence selection 和 conservative calibration 能持续改善 prototype adaptation，Full controlled WPTA 在 AUPRO 上达到 86.2/91.7。五个工业异常检测基准上的最终校准系统相对固定 AnomalyCLIP baseline 平均提升 +1.0 / +4.5 / +3.4 / +2.8，指标顺序为 pixel AUROC / pixel AUPRO / image AUROC / image AP；该结果证明系统级有效性，而 WPTA 机制的因果证据来自受控消融。

关键词：zero-shot anomaly detection；CLIP；test-time adaptation；prototype calibration；wavelet reliability；industrial inspection

## 1. 引言

工业异常检测要求模型发现产品图像中的划痕、缺口、污染、破损和局部纹理异常，并同时给出图像级异常判别和像素级异常定位。以金属件边缘附近的一条细划痕为例，缺陷本身只占很小区域，但它附近的正常轮廓边缘和反光区域也会产生强局部响应；检测器必须覆盖细划痕，同时避免把正常边界误判为异常。由于真实工业场景中异常样本稀缺、类别更新频繁且像素级标注成本高，zero-shot anomaly detection 成为一个重要设置：模型应在不访问目标类别训练数据和异常标注的情况下，直接泛化到新的物体、材料和缺陷形态。CLIP 等视觉语言模型为这一目标提供了开放词汇语义能力，AnomalyCLIP 等方法进一步把 normal/abnormal 文本提示或文本原型用于 patch-level anomaly scoring `[CITATION-NEEDED: CLIP]` `[CITATION-NEEDED: AnomalyCLIP]`。

固定文本原型的主要限制在于它无法适配当前测试图像中的具体缺陷形态。一个通用 abnormal prototype 可以表达“异常”这一语义，但难以同时精确描述金属表面细划痕、瓶口缺口、纹理污染、药片裂纹和结构破损等视觉模式。对于 CLIP-based anomaly detection，这种 instance-specific mismatch 会直接反映在 anomaly map 中：真实缺陷可能因为与通用 abnormal prototype 不够匹配而被低估，正常边界或材料纹理也可能因为局部视觉差异而被误激活。因此，zero-shot anomaly detection 不仅需要强语义原型，还需要在测试时利用当前图像自身的可信视觉证据来校准原型。

一种自然想法是引入频域或小波线索，因为许多工业缺陷表现为局部纹理扰动或边缘断裂。然而，频域响应并不等价于异常响应。普通物体轮廓、反光、高频背景纹理和正常结构边界同样会产生强小波响应。如果把小波响应直接融合到最终 anomaly map，模型会把非语义异常的高频区域误当作异常。本文的受控消融验证了这一点：在 MVTec 上，direct wavelet fusion / no adaptation 的结果为 88.7 / 80.4 / 92.9 / 96.9，低于 baseline 的 91.2 / 83.2 / 91.6 / 96.4；在 VisA 上也从 95.5 / 86.7 / 82.0 / 85.3 降至 94.6 / 85.1 / 81.6 / 84.8。

本文提出 WPTA，把小波线索从“最终异常分数”重新定位为“测试时原型校准的可靠性监督”。WPTA 先用固定 normal/abnormal prototypes 计算初始语义异常分数 `S0`，再对 CLIP patch feature grid 做一级 Haar 小波分解，从高频分量得到纹理扰动能量，从低频分量估计结构边缘，并构造 boundary-aware reliability `W = norm(HF) * (1 - norm(LF_edge))`。`S0` 约束 evidence 的语义方向，`W` 约束 evidence 的局部频域可靠性，两者共同选择当前测试图像中的 abnormal/normal evidence patches。

WPTA 的方法流程直接对应三个技术挑战。第一，当前图像中的可信 patch evidence 必须从无标签测试图像自身抽取，因此需要语义先验和可靠性权重共同约束。第二，小波线索不能绕过 CLIP 语义判别直接生成 anomaly map，因此本文让小波只作用于 evidence selection，而最终异常图仍由校准后的 CLIP prototypes 产生。第三，测试时原型校准存在 prototype drift 风险，因此 WPTA 使用 conservative update，让 visual anchors 只在 evidence 可靠时轻量影响文本原型。

本文采用两层实验叙事来避免把系统级优化误写成单一机制收益。第一层是 MVTec/VisA 的受控消融，用于验证 WPTA 机制：semantic-only prototype adaptation 已显著强于固定原型；wavelet-guided evidence weighting 在 semantic-only 基础上继续提升；conservative update 进一步改善最终方法。第二层是五个工业异常检测基准上的最终校准系统结果，包括 MVTec、VisA、MPDD、BTAD 和 DTD-Synthetic；与 AnomalyCLIP baseline 相比，最终系统的平均结果从 94.9 / 82.8 / 86.2 / 89.4 提升到 95.9 / 87.4 / 89.5 / 92.2。最显著的平均提升来自 pixel AUPRO，达到 +4.5，说明最终系统主要改善异常区域定位质量。

本文贡献如下：

1. 本文提出 Wavelet-Supervised Test-Time Prototype Adaptation，将小波线索从 final map fusion 重新定位为 prototype adaptation 的 patch evidence reliability。
2. 本文设计 semantic-spectral evidence selection，用 CLIP 初始语义异常分数和 boundary-aware wavelet reliability 共同筛选当前图像中的 visual normal/abnormal anchors。
3. 本文在冻结 CLIP 参数、不使用目标训练数据、不进行反向传播的条件下进行 conservative prototype calibration，并通过 MVTec/VisA 受控消融验证 WPTA 机制。
4. 本文报告五个工业基准上的最终校准系统结果，并用 final-setting 配置表明确区分系统级提升和 WPTA 机制证据，避免把不同数据集的工程配置误归因为同一机制。

[FIGURE_PROMPT:figure1_motivated_example]
目标：生成 Figure 1，展示金属件边缘附近细划痕这一 running example 中的 fixed prototype mismatch、direct wavelet fusion failure 和 WPTA 的核心区别。四栏横向布局：输入工业图像，缺陷为靠近正常物体边界的一条细划痕；AnomalyCLIP initial map 中真实划痕被低估且正常边界误激活；direct wavelet fusion 同时激活划痕和正常边界并标注 “map-level fusion is unreliable”；WPTA 使用 `S0` 与 `W = HF * (1 - LF_edge)` 选择 evidence patches，构造 visual anchors，校准 prototypes，输出更聚焦细划痕的 final map。使用白底矢量风格，异常 evidence 橙色，normal evidence 蓝色，可靠性图 viridis，字体不小于 8pt。
[/FIGURE_PROMPT]

## 2. 相关工作

### 2.1 CLIP-based zero-shot anomaly detection

CLIP-based anomaly detection 利用视觉语言模型的开放词汇表示，把异常检测从类别内监督学习扩展到少样本或零样本设置。代表性方法通常构造 normal/abnormal prompt 或 text prototypes，并通过 patch features 与文本 embeddings 的相似度得到 anomaly score `[CITATION-NEEDED: WinCLIP]` `[CITATION-NEEDED: AnomalyCLIP]` `[CITATION-NEEDED: CLIP-AD]` `[CITATION-NEEDED: AdaCLIP]`。这类方法避免了为每个目标类别训练专用分类器，但固定或类别级文本原型难以处理测试图像内部的实例化缺陷差异。WPTA 保留 CLIP 的 frozen representation，但在测试时使用当前图像的 semantic-spectral evidence 对 prototypes 做轻量校准。

### 2.2 Test-time adaptation and prototype adaptation

Test-time adaptation 试图在推理阶段利用目标样本信息缓解分布差异 `[CITATION-NEEDED: test-time adaptation survey]`。在视觉语言模型中，相关方法可能通过 prompt tuning、prototype refinement 或 entropy-based objectives 进行测试时适配 `[CITATION-NEEDED: TPT or CLIP TTA]`。异常检测中的 test-time adaptation 更难，因为异常区域稀疏且未知，错误 evidence 会把 prototypes 拉向正常结构或噪声区域。WPTA 不更新 CLIP 参数，也不把整张图像作为 adaptation 信号，而是只聚合被 `S0` 与 `W` 同时支持的 patch evidence。

### 2.3 Frequency-domain and wavelet cues

频域分析和小波变换长期用于纹理分析、表面缺陷检测和局部结构变化建模 `[CITATION-NEEDED: wavelet texture inspection]`。Haar 小波可以把局部高频扰动与低频结构分离，因此适合为纹理型工业缺陷提供补充线索。本文与直接频域检测不同：小波不直接决定最终 anomaly score，而是用于判断哪些 patch evidence 更可靠。受控消融中 direct wavelet fusion 明显低于 WPTA，说明这一角色定位是必要的。

### 2.4 与最近工作的关系

| 方法类别 | 文本原型 | 测试时原型校准 | 小波可靠性 | 与 WPTA 的区别 |
|---|---:|---:|---:|---|
| AnomalyCLIP `[CITATION-NEEDED]` | 固定 | 否 | 否 | 使用 fixed normal/abnormal prototypes，缺少 instance-specific calibration。 |
| WinCLIP / CLIP-AD 类方法 `[CITATION-NEEDED]` | 固定或类别级 | 通常否 | 否 | 使用 window/patch scoring，但不构造 wavelet-guided visual anchors。 |
| Test-time prompt/prototype tuning `[CITATION-NEEDED]` | 可变 | 是 | 否 | 有 adaptation，但通常缺少面向局部缺陷的 frequency reliability。 |
| Frequency/wavelet anomaly methods `[CITATION-NEEDED]` | 不适用 | 否 | 是 | 使用频域线索作为分数或特征，不校准 CLIP text prototypes。 |
| WPTA | 初始固定，测试时校准 | 是 | 是 | 用 semantic-spectral evidence selection 产生 visual anchors，并保守校准 prototypes。 |

## 3. 方法

### 3.1 Problem formulation and overview

给定测试图像 `x`，zero-shot anomaly detection 需要输出像素级 anomaly map `M` 和图像级 anomaly score `s_img`。CLIP 图像编码器产生 patch features `{f_i}_{i=1}^N`，文本编码器产生 normal prototype `t_n` 和 abnormal prototype `t_a`。在本文设置中，CLIP 参数冻结，推理阶段不访问目标域训练集，不使用异常标注，也不进行反向传播。

WPTA 由四个模块组成。第一，初始语义评分模块使用固定 `t_n` 和 `t_a` 得到 `S0`。第二，boundary-aware wavelet reliability 模块在 CLIP patch feature grid 上计算 `W`。第三，semantic-spectral evidence selection 模块根据 `S0` 与 `W` 选择并聚合 visual normal/abnormal anchors。第四，conservative prototype calibration 模块用 visual anchors 轻量校准 prototypes，并重新计算 anomaly map。

[FIGURE_PROMPT:figure2_method_overview]
目标：生成 Figure 2 方法总览图。横向 pipeline：Input image and frozen CLIP features；Initial semantic anomaly score `S0`；Haar DWT on patch feature grid；Boundary-aware reliability `W = norm(HF) * (1 - norm(LF_edge))`；Semantic-spectral evidence selection；Visual anchors `v_a`, `v_n`；Conservative prototype calibration；Final anomaly map。标注 “CLIP frozen” 和 “no backpropagation”。图中模块名称必须与方法小节一致。
[/FIGURE_PROMPT]

### 3.2 Initial semantic anomaly score

WPTA 首先使用固定文本原型计算 patch-level semantic anomaly prior。对每个 patch feature `f_i`，异常概率定义为：

```text
S0(i) = exp(sim(f_i, t_a) / tau) /
        (exp(sim(f_i, t_a) / tau) + exp(sim(f_i, t_n) / tau)).
```

其中 `sim(.,.)` 表示归一化相似度，`tau` 为温度系数。`S0(i)` 提供语义方向：高 `S0(i)` 表示 patch 更接近 abnormal prototype，低 `S0(i)` 表示 patch 更接近 normal prototype。由于固定 prototypes 存在 instance mismatch，`S0` 只作为 evidence selection 的先验，而不是最终校准结果。

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

WPTA 用 `S0` 与 `W` 共同选择 abnormal 和 normal evidence。异常 evidence 应同时满足语义上接近 abnormal prototype 且具有可靠局部扰动；正常 evidence 应语义上接近 normal prototype 且不被异常纹理或结构边界污染。概念上，evidence weights 可写为：

```text
q_a(i) = S0(i)^gamma * rho(W(i)),
q_n(i) = (1 - S0(i))^gamma * rho(1 - W(i)).
```

其中 `gamma` 控制语义先验锐化，`rho(.)` 表示可靠性调制函数。正式实现中的 top-k、阈值、归一化和 confidence gate 可放入 Appendix；正文只保留该模块的核心逻辑。

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

`S(i)` reshape 并上采样为 anomaly map `M`。图像级 score `s_img` 使用与 baseline 一致的 aggregation protocol。该一致性对公平比较很关键，正式英文稿应在 Implementation Details 中明确 aggregation 方式。

### 3.7 Implementation details

受控 WPTA 消融在 MVTec 和 VisA 上使用相同的缓存特征与评价设置。我们使用 AnomalyCLIP 缓存的 patch features，并聚合 feature map layers 1/2/3；layer fusion 使用 sum，layer temperature 为 1.0，Gaussian smoothing 参数 `sigma=5`，AUPRO 使用 200 个阈值步。所有指标均按 image-pixel-level protocol 计算。

Prototype adaptation 的默认参数来自 `conf/run_prototype_ablation_experiments_conf.yaml`。具体地，prototype softmax temperature 为 0.07，semantic sharpening `gamma=1.0`，wavelet evidence exponent `eta=1.0`，evidence top-k ratio 为 0.2。Conservative update 使用 `proto_alpha0=0.0`、`proto_beta0=0.01`、`proto_tau_a=0.15` 和 `proto_update_min_abnormal_confidence=0.06`，并在 confidence 不足时抑制更新。Boundary-aware wavelet reliability 使用 `proto_wavelet_mode=boundary_aware` 和 `proto_wavelet_mix=0.05`；HF-only 与 semantic-only variants 只改变 `proto_wavelet_mode`，no-conservative variant 只关闭 `proto_conservative_update`。Direct fusion 负对照不做 prototype adaptation，而是以 `direct_wavelet_fusion_weight=0.5` 直接融合 `S0` 与小波图。

为保证受控消融中 adaptation variants 的可比性，除 baseline 外的 prototype/fusion variants 使用相同的 multi-crop 与 pixel-to-image 设置：`multicrop_weight=0.50`，`pixel_to_image_weight=0.10`，`pixel_to_image_topk_ratio=0.01`。因此，Table 2 和 Table 3 中关于小波可靠性和 conservative update 的结论主要来自 adaptation variants 之间的相对比较，而不是从 baseline 到任一增强行的单独差值。五数据集 final systems 使用数据集特定配置，已在 Table 4 中单独审计；完整运行命令见 `/Users/bytedance/code/AnomalyCLIP/FIVE_DATASET_RESULTS_AND_ABLATIONS.md`。

## 4. 实验设置

### 4.1 Datasets

主实验覆盖五个工业异常检测基准：MVTec、VisA、MPDD、BTAD 和 DTD-Synthetic `[CITATION-NEEDED: MVTec]` `[CITATION-NEEDED: VisA]` `[CITATION-NEEDED: MPDD]` `[CITATION-NEEDED: BTAD]` `[CITATION-NEEDED: DTD]`。这些数据集覆盖物体缺陷、纹理异常、结构破损和合成纹理异常。补充泛化实验在 ISIC/ISBI 上报告 pixel-level 结果，但其它医学数据集当前为 blocked，因此医学结果不作为主 claim。

### 4.2 Metrics

工业实验报告四个指标：pixel AUROC、pixel AUPRO、image AUROC 和 image AP。所有 slash-form 数值顺序固定为：

```text
pixel AUROC / pixel AUPRO / image AUROC / image AP
```

本文重点分析 pixel AUPRO，因为它更直接反映 anomaly localization 的区域质量。Image AUROC 和 image AP 用于验证图像级检测是否同步受益。

### 4.3 Baselines, final systems, and variants

主结果以 AnomalyCLIP baseline 作为固定原型基线，并比较每个数据集的 final calibrated system。受控消融在 MVTec 和 VisA 上报告四个 variants：direct wavelet fusion / no adaptation、semantic prototype adaptation、wavelet prototype adaptation without conservative update，以及 Full WPTA controlled ablation setting。由于五数据集 final systems 与受控消融设置不完全相同，本文在表格和文字中显式区分两者：Table 1 证明最终系统有效性，Table 2 和 Table 3 证明 WPTA 机制。

## 5. 结果与分析

### 5.1 Main results on five industrial benchmarks

表 1 展示五个工业数据集上的主结果。最终校准系统在所有数据集上均超过 AnomalyCLIP baseline，并且平均四个指标全部提升。该表不单独证明所有提升都来自 WPTA；各数据集 final setting 的启用模块见 Table 4。

**表 1. 五个工业异常检测基准上的主结果。指标越高越好。**

| Dataset | Method | P-AUROC | P-AUPRO | I-AUROC | I-AP | Δ vs. baseline |
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

平均结果显示，最终校准系统的最大收益来自 P-AUPRO，平均提升 +4.5。该趋势说明当前系统主要改善局部异常区域覆盖，而不是只提升图像级分类分数。BTAD 和 MPDD 上的 image-level gains 也较明显，分别在 I-AUROC/I-AP 上提升 +4.8/+3.9 和 +4.1/+5.8，说明定位质量改善没有牺牲图像级检测。由于 MPDD/BTAD final commands 主要启用 multi-crop fusion 和 pixel-to-image fusion，本段只作为系统级结果解释；WPTA 的机制解释见下一节受控消融。

### 5.2 Core component ablation

表 2 在 MVTec 和 VisA 的受控消融设置下验证各组件的作用。这里的 Full WPTA 是 controlled ablation setting，与五数据集主结果中的 Full setting 不完全相同。

**表 2. 核心组件消融。每个单元格为 P-AUROC / P-AUPRO / I-AUROC / I-AP。**

| Method | MVTec | VisA | 作用 |
|---|---:|---:|---|
| Baseline | 91.2 / 83.2 / 91.6 / 96.4 | 95.5 / 86.7 / 82.0 / 85.3 | 固定文本原型 |
| Direct wavelet fusion / no adaptation | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | 负对照，直接 map fusion |
| Semantic prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | 仅用 `S0` 选择 evidence |
| Wavelet prototype adaptation w/o conservative | 91.7 / 85.8 / 93.9 / 97.2 | 96.1 / 91.3 / 84.1 / 87.0 | `W` 参与 evidence weighting |
| Full WPTA, controlled ablation setting | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** | Boundary-aware `W` + conservative update |

该消融给出三点结论。第一，direct wavelet fusion 低于 baseline 和 Full WPTA，说明小波不应直接作为最终异常图。第二，semantic prototype adaptation 明显优于固定原型，说明 test-time prototype calibration 是有效方向。第三，wavelet prototype adaptation w/o conservative 和 Full WPTA 进一步提升 AUPRO，说明小波可靠性和 conservative update 均是最终方法链路的一部分。

### 5.3 Wavelet reliability design

表 3 进一步分析小波可靠性设计。HF-only W 相比 semantic-only 只有小幅提升，而 boundary-aware W 进一步提高 AUPRO，说明低频结构边界抑制对于减少伪高频 evidence 是必要的。

**表 3. 小波可靠性设计消融。每个单元格为 P-AUROC / P-AUPRO / I-AUROC / I-AP。**

| Wavelet setting | MVTec | VisA | 作用 |
|---|---:|---:|---|
| Semantic-only prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | 无小波 |
| Direct wavelet fusion | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | 直接融合负例 |
| HF-only W + prototype adaptation | 91.6 / 85.3 / 94.0 / 97.2 | 96.0 / 90.8 / 84.0 / 86.9 | 只使用高频能量 |
| Boundary-aware W + prototype adaptation | 91.7 / 85.7 / 93.8 / 97.3 | 96.1 / 91.2 / 83.9 / 87.1 | `W = HF * (1 - LF_edge)` |
| Full boundary-aware W + conservative | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** | 最终消融设置 |

Boundary-aware W 在 MVTec/VisA 上分别把 AUPRO 从 HF-only 的 85.3/90.8 提升到 85.7/91.2。Full 设置进一步达到 86.2/91.7。这说明高频响应有用但不充分，必须结合低频结构边界抑制和保守更新。

### 5.4 Final-system configuration audit

Table 4 记录五数据集主结果中各 final run 的启用模块。该表的目的不是增加一个新结果，而是明确主结果表的证据边界。MVTec 和 VisA 的 final system 包含 wavelet reliability、TTA rectification、multi-crop fusion 和 pixel-to-image fusion；DTD-Synthetic 包含 wavelet reliability、multi-crop fusion 和 pixel-to-image fusion；MPDD 和 BTAD 当前 final system 没有启用 wavelet/TTA flags。因此，五数据集结果应写成 final calibrated system 的系统级提升，WPTA 机制则由 MVTec/VisA 受控消融验证。

**表 4. 五数据集 final system 配置审计。该表用于限定因果归因。**

| Dataset | Wavelet reliability | TTA rectification | Multi-crop fusion | Pixel-to-image fusion | Key setting |
|---|---|---|---|---|---|
| MVTec | yes | yes | yes | yes | sigma=5, multicrop weight=0.5, p2i weight=0.1 |
| VisA | yes | yes | yes | yes | sigma=5, multicrop weight=0.5, p2i weight=0.1 |
| MPDD | no | no | yes | yes | sigma=8, multicrop weight=0.25, p2i weight=0.8 |
| BTAD | no | no | yes | yes | sigma=10, multicrop weight=0.85, p2i weight=0.95 |
| DTD-Synthetic | yes | no | yes | yes | sigma=8, multicrop weight=0.75, p2i weight=0.5 |

### 5.5 Protocol-reference comparison with reported CLIP-based methods

表 5 汇总了从 `paper_iconip/main.tex` 提取的代表性方法结果，并把当前 final calibrated system 结果放入同一候选表中。该表只用于判断当前系统在相近 OpenCLIP ViT-L/14 工业协议下的大致位置，不能单独支撑 SOTA claim，因为外部表使用 auxiliary-target ZSAD 协议，而当前稿件强调 training-free test-time adaptation。正式投稿时，如果无法证明协议完全一致，应把该表放入附录或删除。

**表 5. MVTec/VisA 上的候选外部方法对比。协议尚未完全核验，不能据此声称 SOTA。**

| Dataset | Method | P-AUROC | P-AUPRO | I-AUROC | I-AP |
|---|---|---:|---:|---:|---:|
| MVTec | AnomalyCLIP, extracted | 91.1 | 81.4 | 91.5 | 96.2 |
| MVTec | AA-CLIP†, extracted | **91.9** | 84.6 | 90.5 | 94.9 |
| MVTec | Source Ours (TAAP/INPC), extracted | 91.5 | 85.5 | 92.8 | 96.7 |
| MVTec | Final calibrated system | 91.8 | **85.6** | **94.5** | **97.6** |
| VisA | AnomalyCLIP, extracted | 95.4 | 87.0 | 82.1 | 85.4 |
| VisA | AdaCLIP, extracted | 95.5 | 56.8 | **85.8** | 84.9 |
| VisA | Source Ours (TAAP/INPC), extracted | 95.6 | 88.3 | 83.3 | 85.9 |
| VisA | Final calibrated system | **96.2** | **91.3** | 84.6 | **87.4** |

在该候选表中，当前系统在 MVTec/VisA 的 P-AUPRO 上具有较强位置，并在多数图像级指标上也具有竞争力。但由于协议尚未完全核验，本文将主 claim 限定为“相对固定 AnomalyCLIP baseline 的系统级提升”和“MVTec/VisA 受控消融中的 WPTA 机制有效”，而不是“达到 SOTA”。

### 5.6 Appendix-style medical observation

ISIC/ISBI 上的附录式观察显示，final calibrated system 的 pixel AUROC / pixel AUPRO 从 AnomalyCLIP baseline 的 88.7 / 78.6 提升到 89.9 / 80.0。由于其它医学数据集当前没有完成评估，该结果只说明当前校准系统在非工业场景中值得进一步验证，不作为本文主 claim，也不应写入摘要或贡献列表。

### 5.7 Qualitative visualization plan

[FIGURE_PROMPT:figure3_qualitative]
目标：生成 qualitative visualization。每行一个真实 case，列为 Input image、GT mask、AnomalyCLIP baseline map、Boundary-aware wavelet reliability W、Selected evidence patches、WPTA final map。至少覆盖 MVTec、VisA、MPDD、BTAD、DTD-Synthetic 中 4 个数据集。Baseline 与 final map 使用相同色标，W 使用 viridis，abnormal evidence 用橙色边框，normal evidence 用蓝色边框。所有热图必须来自真实模型输出，不能手工涂色。
[/FIGURE_PROMPT]

## 6. 讨论

### 6.1 Why wavelet reliability, not wavelet scoring

小波提供的是局部频率结构，而不是语义异常判别。Direct fusion 负对照显示，直接把小波响应加到 anomaly map 会伤害定位表现。WPTA 的设计把小波限制在 evidence reliability 层面，使最终 anomaly map 仍由 CLIP semantic prototypes 决定。这一角色分工解释了为什么小波能帮助 prototype adaptation，同时避免频域响应的伪异常问题。

### 6.2 Why conservative calibration

无标签测试时校准容易产生 prototype drift。WPTA 使用 confidence-gated conservative update，使 visual anchors 只有在 evidence 足够可靠时影响 prototypes。受控消融中 Full WPTA 相比 no-conservative variant 进一步提升，说明保守更新对最终性能有稳定贡献。

### 6.3 Limitations

当前稿件仍有四个投稿前限制。第一，所有 `[CITATION-NEEDED]` 必须替换为真实引用。第二，候选外部方法表需要完成协议核验，才能决定是否进入主文、附录或删除。第三，qualitative visualization 需要真实 anomaly maps、wavelet reliability maps 和 selected evidence patches 支撑机制叙事。第四，当前实现细节虽然已经从清单改为参数说明，但英文投稿版仍需要逐项核对代码、日志和命令行，确保每个 reported setting 都能复现。

## 7. 结论

本文提出 WPTA，用于 CLIP-based zero-shot anomaly detection。WPTA 将 Haar 小波线索作为 patch evidence reliability 来监督测试时原型校准，而不是把频域响应直接融合到最终 anomaly map 中。受控 MVTec/VisA 消融说明，直接小波融合会损害定位表现，而 semantic-spectral evidence selection 和 conservative prototype calibration 能改善测试时原型适配。五个工业基准上的最终校准系统相对固定 AnomalyCLIP baseline 在平均 pixel AUPRO 上获得 +4.5 的提升，显示当前系统在异常区域定位上具有明确收益。后续投稿版本需要完成真实引用、协议核验和定性可视化，以支撑完整顶会证据链。

## 8. Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Final calibrated system improves AnomalyCLIP baseline on five industrial datasets. | Table 1, all five datasets improve over baseline; Table 4 records per-dataset final settings. | Supported |
| Main system-level gain is pixel AUPRO. | Table 1 average P-AUPRO gain +4.5, larger than P-AUROC +1.0, I-AUROC +3.4, I-AP +2.8. | Supported |
| Direct wavelet fusion is not sufficient. | Table 2 and Table 3, direct fusion is below baseline and Full in P-AUPRO. | Supported |
| Prototype adaptation is useful. | Table 2, semantic adaptation improves baseline on MVTec/VisA. | Supported |
| Wavelet reliability adds value beyond semantic-only adaptation. | Table 2, wavelet no-conservative improves AUPRO over semantic-only on both datasets. | Supported |
| Boundary-aware W improves over HF-only W. | Table 3, boundary-aware W improves AUPRO over HF-only on both datasets. | Supported |
| WPTA mechanism is validated across all five industrial datasets. | Table 4 shows MPDD/BTAD final runs do not enable wavelet/TTA flags. | Not supported; keep mechanism claim limited to MVTec/VisA controlled ablations |
| The current system is SOTA. | Protocol of external comparison is not fully verified. | Not supported; do not claim |
| The current system has a preliminary positive observation beyond industrial data. | ISIC/ISBI improves from 88.7 / 78.6 to 89.9 / 80.0, but other medical datasets are incomplete. | Appendix-only observation |

## 9. 当前投稿成熟度自检

- 结构：Introduction -> Method -> Experiments -> Discussion 的逻辑完整，核心 contribution 与实验表对应；running example 已引入 Figure 1 prompt，但仍需要真实可视化回扣。
- 证据：五数据集主结果支撑最终系统有效性，MVTec/VisA 受控消融支撑 WPTA 机制；v0.5 已修正五数据集因果归因边界。
- 风险：引用未核验、外部协议未核验、真实 qualitative figure 缺失、实现细节仍需逐项复现核对。
- 当前建议：可作为“内部强稿 v0.5”继续迭代；尚不应声明已达到最终投稿状态。

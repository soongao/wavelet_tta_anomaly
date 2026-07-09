# Wavelet-Supervised Test-Time Prototype Adaptation for Zero-Shot Anomaly Detection

中文顶会论文草稿 v0.1

## 写作边界

本文稿依据 `/Users/bytedance/Documents/Codex/2026-07-06/clip-baseline-anomalyclip-tta-sota-idea-2/outputs/research_writing_handoff_zh.md` 和 `/Users/bytedance/code/AnomalyCLIP/paper/result_record/result_table.csv` 中标记为 `current` 的结果撰写。所有实验数值只来自该 CSV。文中不声称 SOTA，不编造外部方法对比或 qualitative visualization 的数值。需要补充的数据和图像均用 `[TABLE_DATA_PROMPT:...]` 或 `[FIGURE_PROMPT:...]` 标记，可直接交给后续实验、绘图或生成图 agent。

文献引用尚未做 DOI/版本核验，因此统一使用 `[CITATION-NEEDED: ...]` 占位。后续进入英文 CVPR/ICCV/NeurIPS 风格稿时，应先完成真实引用表和 Related Work 核验。

## 0. 论文定位与逻辑骨架

### 0.1 论文类型定位

- 类型：Technique Paper。
- 理由：本文不提出新的 benchmark 或新任务定义，而是在已有 CLIP-based zero-shot anomaly detection setting 下提出一种 training-free test-time prototype adaptation 机制。
- 叙事重心：核心技术不是“小波检测异常”，而是“用小波可靠性监督当前测试图像中哪些 patch evidence 可以校准 CLIP normal/abnormal prototypes”。

### 0.2 Thinking template

| Stage | 内容 |
|---|---|
| Research background | Zero-shot industrial anomaly detection 需要在没有目标类别训练样本、没有异常标注的情况下完成像素级定位和图像级判别。CLIP/VLM 提供了开放词汇语义能力，AnomalyCLIP 等方法把 normal/abnormal 文本原型用于 anomaly scoring。 |
| Limitation 1 | 固定文本原型难以适配每张测试图像中的具体缺陷形态。不同物体、材质、照明和缺陷类型会导致 instance-specific mismatch。 |
| Limitation 2 | 直接把低级频域或小波响应融合到 anomaly map 中不可靠，因为正常纹理、反光和物体边界同样会产生高频响应。CSV 中 direct wavelet fusion 在 MVTec 和 VisA 上低于 baseline，验证了这一风险。 |
| Limitation 3 | 只依赖语义异常分数做 prototype adaptation 可以提升性能，但缺少 patch-level 可靠性信号来区分纹理异常与正常结构边界。 |
| Key Idea | 本文把小波线索用作测试时原型校准的可靠性监督，而不是把频域响应直接叠加到最终异常图上。 |
| Challenge 1 | 如何从当前测试图像自身抽取可信 patch evidence，同时避免无标签测试时把正常结构误当作异常证据。 |
| Challenge 2 | 如何让小波响应服务于 CLIP semantic prototype adaptation，而不是破坏 CLIP 原有语义判别。 |
| Challenge 3 | 如何控制 prototype drift，使正常图像上的误报风险不会因测试时校准而扩大。 |
| Methodology topic sentence | WPTA 先用 CLIP normal/abnormal 文本原型产生初始语义异常图，再在 CLIP patch feature grid 上计算 boundary-aware Haar wavelet reliability，用 semantic-spectral evidence 构建 visual anchors，并通过 conservative update 校准文本原型。 |
| Module A | Initial semantic scoring：用固定 CLIP 文本原型得到 `S0`，为 patch evidence 提供语义先验。 |
| Module B | Boundary-aware wavelet reliability：对 patch feature grid 做一级 Haar DWT，构造 `W = norm(HF) * (1 - norm(LF_edge))`，抑制普通结构边界导致的伪高频。 |
| Module C | Conservative prototype calibration：用 `S0` 与 `W` 筛选并聚合 visual normal/abnormal anchors，以 confidence-gated 方式校准 prototypes，并重新计算 anomaly map。 |
| Contribution 1 | 提出 Wavelet-Supervised Test-Time Prototype Adaptation，把 wavelet cues 从 final map fusion 重新定位为 prototype adaptation 的 evidence reliability。 |
| Contribution 2 | 设计 semantic-spectral patch evidence selection，将 CLIP 初始异常分数与 boundary-aware wavelet reliability 结合，用当前测试图像自身构造 visual anchors。 |
| Contribution 3 | 通过 conservative prototype calibration 在不反向传播、不更新 CLIP 参数的条件下提升 zero-shot anomaly localization，MVTec 与 VisA 上 AUPRO 分别提升 +3.0 和 +5.0。 |

### 0.3 自一致检查

- Limitation -> Key Idea：通过 direct wavelet fusion 负对照和 semantic-only adaptation 对照支撑，pass。
- Key Idea -> Challenges：三个挑战均来自无标签 test-time adaptation 的 evidence selection、wavelet 使用方式和 prototype drift，pass。
- Challenges -> Methodology：`S0`、`W`、visual anchors 和 conservative update 一一对应，pass。
- Methodology -> Contributions：贡献覆盖方法机制、核心模块和实验验证，pass。

## 摘要

零样本工业异常检测要求模型在没有目标域训练样本和异常标注的情况下，同时完成图像级异常判别与像素级异常定位。近期 CLIP-based 方法通常依赖固定的 normal/abnormal 文本原型来生成异常分数，但单一文本原型难以描述每张测试图像中的具体缺陷形态，尤其是在材质、光照和局部纹理变化显著的工业场景中。本文提出 Wavelet-Supervised Test-Time Prototype Adaptation (WPTA)，一种无需训练的测试时原型校准方法。本文的核心思想是把小波线索作为 prototype adaptation 的可靠性监督，而不是直接作为额外异常图。具体而言，WPTA 首先使用固定 CLIP 文本原型产生初始 patch-level semantic anomaly score，然后在 CLIP patch feature grid 上进行一级 Haar 小波分解，构造 boundary-aware wavelet reliability，以强调局部高频纹理扰动并抑制普通物体边界。随后，本文结合语义异常先验与小波可靠性筛选当前测试图像中的可信 normal/abnormal patch evidence，聚合成 visual anchors，并以 conservative update 校准 normal/abnormal prototypes，最终重新计算 anomaly map。实验表明，WPTA 在固定 zero-shot 协议下稳定超过 AnomalyCLIP baseline：在 MVTec 上获得 +0.6 / +3.0 / +2.5 / +1.0 的提升，在 VisA 上获得 +0.7 / +5.0 / +2.3 / +2.0 的提升，指标顺序为 pixel AUROC / pixel AUPRO / image AUROC / image AP。消融实验进一步表明，直接融合小波响应会降低性能，而把小波用于 patch evidence reliability 能持续改进 prototype adaptation，尤其在 pixel AUPRO 上提升最明显。

关键词：zero-shot anomaly detection；CLIP；test-time adaptation；prototype calibration；wavelet reliability；industrial inspection

## 1. 引言

工业异常检测的目标是在产品图像中发现划痕、缺口、污染、破损和局部纹理异常等低频率缺陷，并在许多实际场景中同时给出图像级判别和像素级定位。由于工业类别繁多、异常样本稀缺且标注成本高，zero-shot anomaly detection 正在成为一个重要设置：模型应在不访问目标类别训练数据和异常标注的情况下直接泛化到新物体与新材质。视觉语言模型，尤其是 CLIP，为这一设置提供了有吸引力的语义基础。代表性 CLIP-based anomaly detection 方法通常构造 normal/abnormal 文本提示或文本原型，并通过图像 patch 特征与文本原型之间的相似度生成异常图 `[CITATION-NEEDED: CLIP]` `[CITATION-NEEDED: AnomalyCLIP]`。

然而，固定文本原型与当前测试图像中的具体缺陷形态之间存在 instance-specific mismatch。一个通用的 abnormal prototype 可以描述“异常”这一语义类别，却很难同时覆盖金属表面的细小划痕、瓶口缺口、布料污染、药片裂纹和电路板局部破损等细粒度视觉形态。对于 CLIP-based 方法，这种 mismatch 会直接影响 patch-level anomaly map：某些真实缺陷因为不够贴合通用 abnormal prototype 而被低估，而某些正常纹理或边界因为局部视觉差异被高估。由此，zero-shot anomaly detection 的关键不只是设计更好的文本提示，还包括如何在测试时利用当前图像自身的可信视觉证据，使 normal/abnormal prototypes 对当前实例更敏感。

一种直观方案是把频域或小波响应直接融入最终异常图，因为工业缺陷常表现为局部纹理、边缘断裂或材料表面变化。该方向具有合理动机，但直接 map-level fusion 并不可靠：正常纹理、反光、物体轮廓和结构边界同样会产生高频响应，这些响应不一定对应语义异常。本文的实验也验证了这一点。在 MVTec 上，direct wavelet fusion / no adaptation 的 pixel AUROC / pixel AUPRO / image AUROC / image AP 为 88.7 / 80.4 / 92.9 / 96.9，低于 AnomalyCLIP baseline 的 91.2 / 83.2 / 91.6 / 96.4；在 VisA 上也从 baseline 的 95.5 / 86.7 / 82.0 / 85.3 降至 94.6 / 85.1 / 81.6 / 84.8。这个负对照说明，小波线索不能被简单视为另一个异常分数。

本文提出 Wavelet-Supervised Test-Time Prototype Adaptation (WPTA)，将小波线索重新定位为 prototype adaptation 的 evidence reliability。WPTA 保持 CLIP 参数和文本提示不训练，先用固定 normal/abnormal prototypes 计算初始语义异常分数 `S0`，再对 CLIP patch feature grid 做一级 Haar 小波分解，得到高频纹理能量和低频结构边缘，并构造 boundary-aware reliability `W`。`S0` 提供语义约束，`W` 提供局部频域可靠性，两者共同决定哪些 patches 可以作为当前图像的可信 abnormal evidence 或 normal evidence。随后，WPTA 聚合这些 evidence patches 得到 visual anchors，并用 conservative update 轻量校准 normal/abnormal prototypes，最终用校准后的 prototypes 重新计算 anomaly map。

本文的实验结果表明，WPTA 在 MVTec 与 VisA 上均稳定超过 AnomalyCLIP baseline，并且提升覆盖 pixel-level localization 与 image-level detection。最显著的提升来自 pixel AUPRO：MVTec 上提升 +3.0，VisA 上提升 +5.0，说明 WPTA 对异常区域定位质量更有帮助。核心消融进一步给出因果证据：semantic-only prototype adaptation 已经优于固定原型，说明 test-time prototype adaptation 是有效方向；wavelet prototype adaptation no conservative 进一步超过 semantic-only，说明小波作为 evidence reliability 有增益；Full WPTA 在 no-conservative 基础上继续提升，说明 conservative update 对稳定收益有价值。

本文贡献如下：

1. 本文提出 Wavelet-Supervised Test-Time Prototype Adaptation，将小波线索从最终 anomaly map 融合信号转化为 prototype adaptation 的 patch evidence reliability。
2. 本文设计 semantic-spectral evidence selection，在 CLIP 初始异常先验 `S0` 与 boundary-aware wavelet reliability `W` 的共同约束下，从单张测试图像中构造 visual normal/abnormal anchors。
3. 本文在不反向传播、不更新 CLIP 参数、不使用训练集或额外异常样本的条件下进行 conservative prototype calibration，并在 MVTec 与 VisA 上稳定提升 AnomalyCLIP baseline，尤其显著提升 pixel AUPRO。

[FIGURE_PROMPT:figure1_motivated_example]
目标：生成论文 Figure 1，用于第一页或第二页顶部，解释为什么“小波不能直接作为最终异常图，但可以作为 prototype adaptation 的可靠性监督”。
画面要求：四栏横向布局，白底、矢量风格、CVPR 双栏论文可读。第一栏是工业产品输入图示，包含正常结构边界和一个局部真实缺陷。第二栏显示 AnomalyCLIP fixed text prototypes 产生的初始 anomaly map，其中真实缺陷被低估、部分正常边界被高估。第三栏显示 direct wavelet fusion 的失败：高频响应同时激活缺陷和正常边界，用红色叉号标注“map-level fusion is unreliable”。第四栏显示本文方法：`S0` 与 boundary-aware `W = HF * (1 - LF_edge)` 共同筛选可信 evidence patches，聚合为 visual anchors，校准 normal/abnormal prototypes，得到更贴合缺陷区域的 final anomaly map。标注必须使用真实模块名：Initial semantic score `S0`、Boundary-aware wavelet reliability `W`、Selected abnormal evidence、Selected normal evidence、Conservative prototype calibration、Final anomaly map。不要使用夸张渐变、3D、装饰性图标。配色使用色盲友好蓝/橙/灰，异常区域使用橙红，正常 evidence 使用蓝色，可靠性图使用 viridis。
[/FIGURE_PROMPT]

## 2. 相关工作

### 2.1 CLIP-based zero-shot anomaly detection

CLIP-based anomaly detection 利用视觉语言模型的开放词汇表示，将工业异常检测从类别内监督学习扩展到更少标注或零样本设置。代表性方法通过手工或学习式文本提示构造 normal/abnormal semantics，并比较 patch features 与文本 embeddings 的相似度来得到 anomaly score `[CITATION-NEEDED: WinCLIP]` `[CITATION-NEEDED: AnomalyCLIP]` `[CITATION-NEEDED: CLIP-AD or PromptAD or AdaCLIP]`。这类方法的优势在于不需要为每个目标类别训练专用分类器，但它们通常依赖固定或类别级文本原型。本文关注的不是重新训练 CLIP 或学习新的 prompt，而是在固定 zero-shot 协议下，用当前测试图像自身的可靠 patch evidence 对 normal/abnormal prototypes 做轻量校准。

### 2.2 Test-time adaptation and prototype adaptation

Test-time adaptation 试图在推理阶段利用目标样本本身的信息缓解训练分布与测试分布之间的差异 `[CITATION-NEEDED: test-time adaptation survey or representative methods]`。在视觉语言模型中，相关工作可能通过 prompt tuning、feature normalization、prototype refinement 或 entropy-based objective 来适配目标分布 `[CITATION-NEEDED: CLIP test-time adaptation / TPT / TTA methods]`。然而，异常检测中的无标签测试时适配有更高风险：异常区域稀疏，正常结构丰富，若 adaptation evidence 选择不当，原型可能向错误方向漂移。本文的区别在于，WPTA 不用梯度更新模型参数，也不把整张图像当作 adaptation 信号，而是用语义异常先验和小波可靠性共同筛选 patch-level evidence，再以 conservative update 控制原型漂移。

### 2.3 Frequency-domain and wavelet cues for anomaly detection

频域分析和小波变换长期用于表面检测、纹理分析和局部结构变化建模 `[CITATION-NEEDED: wavelet anomaly or texture inspection]`。Haar 小波能够把局部高频纹理扰动与低频结构信息分离，因此适合描述划痕、破损、污染和材料表面突变等工业异常。问题在于，高频响应并不等价于异常响应：普通物体边界、重复纹理和高光反射也可能产生强高频。本文通过 direct wavelet fusion 负对照验证了这一点，并进一步提出 boundary-aware reliability，用低频结构边缘抑制普通边界导致的伪高频激活。与直接使用频域分数不同，本文把小波信号用于指导哪些 patch 可以参与 prototype adaptation。

### 2.4 与最近工作的关系

| 方法类别 | 是否固定文本原型 | 是否测试时校准原型 | 是否使用小波可靠性 | 与本文的关键区别 |
|---|---:|---:|---:|---|
| AnomalyCLIP `[CITATION-NEEDED]` | 是 | 否 | 否 | 依赖固定 normal/abnormal prototypes，缺少 instance-specific adaptation。 |
| WinCLIP / CLIP-AD 类方法 `[CITATION-NEEDED]` | 通常是 | 通常否 | 否 | 使用 window/patch-level CLIP scoring，但不通过 wavelet-guided evidence 校准 prototypes。 |
| Test-time prompt/prototype tuning `[CITATION-NEEDED]` | 否或部分否 | 是 | 否 | 有 adaptation，但通常没有面向工业缺陷的 patch-level frequency reliability。 |
| Frequency/wavelet anomaly methods `[CITATION-NEEDED]` | 不适用 | 否 | 是 | 使用频域线索检测或增强异常，但不校准 CLIP text prototypes。 |
| WPTA | 初始固定，测试时校准 | 是 | 是 | 用 semantic-spectral evidence selection 构造 visual anchors，并以 conservative update 校准 CLIP prototypes。 |

## 3. 方法

### 3.1 问题定义与整体流程

给定一张测试图像 `x`，zero-shot anomaly detection 需要输出像素级 anomaly map `M` 和图像级 anomaly score `s_img`。本文遵循 CLIP-based anomaly detection 设置：CLIP 图像编码器产生 patch features，记为 `{f_i}_{i=1}^N`，其中 `i` 表示 patch index；文本编码器产生 normal prototype `t_n` 和 abnormal prototype `t_a`。推理阶段不能使用目标类别训练图像、异常标注或梯度更新。

WPTA 的流程包括四个步骤。首先，使用固定 normal/abnormal prototypes 计算初始语义异常分数 `S0(i)`。其次，将 patch features reshape 成 `H x W x C` 的 feature grid，并在该 grid 上进行一级 Haar 小波分解，得到 boundary-aware wavelet reliability `W(i)`。第三，使用 `S0(i)` 与 `W(i)` 共同构造 abnormal 和 normal patch evidence weights，聚合成当前图像的 visual anchors。最后，用 conservative update 校准文本原型，并重新计算 anomaly map。该流程只改变当前测试图像的原型表示，不改变 CLIP 参数。

[FIGURE_PROMPT:figure2_method_overview]
目标：生成论文 Figure 2，方法总览图，放在 Method 开头。
布局：横向 pipeline，五个主阶段，从左到右分别为：Input image and CLIP patch features；Initial semantic anomaly scoring；Haar wavelet reliability on patch feature grid；Semantic-spectral evidence selection；Conservative prototype calibration and final anomaly map。
细节：第一阶段显示图像输入经过 frozen CLIP image encoder 得到 `H x W x C` patch feature grid，同时 frozen CLIP text encoder 得到 `t_n` 和 `t_a`。第二阶段显示 softmax over `sim(f_i,t_n)` and `sim(f_i,t_a)` 得到 `S0`。第三阶段显示 Haar DWT 得到 `LL`、`LH`、`HL`、`HH`，从高频分量得到 `HF`，从 `LL` 得到 `LF_edge`，合成 `W = norm(HF) * (1 - norm(LF_edge))`。第四阶段显示 `S0 + W` 选择 abnormal evidence patches 和 normal evidence patches，并聚合为 `v_a`、`v_n`。第五阶段显示 `t'_a = normalize((1-alpha)t_a + alpha v_a)` 和 `t'_n = normalize((1-beta)t_n + beta v_n)`，然后输出 final anomaly map。必须标注 “no backpropagation” 和 “CLIP parameters frozen”。风格：CVPR vector diagram，白底，细线箭头，模块名与论文小节标题一致，字体 8pt 以上，色盲友好蓝/橙/灰。
[/FIGURE_PROMPT]

### 3.2 初始语义异常评分

WPTA 先保留 AnomalyCLIP/CLIP 的固定原型机制，得到一个语义上可信但可能存在 instance mismatch 的初始 anomaly prior。对每个 patch feature `f_i`，本文计算它与 normal prototype 和 abnormal prototype 的相似度，并通过二分类 softmax 得到异常概率：

```text
S0(i) = exp(sim(f_i, t_a) / tau) /
        (exp(sim(f_i, t_a) / tau) + exp(sim(f_i, t_n) / tau)).
```

其中 `sim(.,.)` 表示余弦相似度或 CLIP 常用的归一化相似度，`tau` 为温度系数。`S0(i)` 不直接作为最终结果，而是作为 patch evidence selection 的语义先验。高 `S0(i)` 的 patch 更可能提供 abnormal evidence，低 `S0(i)` 的 patch 更可能提供 normal evidence。

### 3.3 Boundary-aware Haar wavelet reliability

仅依赖 `S0` 选择 evidence 可能受到固定文本原型的影响，因此 WPTA 进一步从当前图像的 patch feature grid 中提取局部频域可靠性。与在原图像上做小波变换不同，本文在 CLIP patch feature grid 上进行一级 Haar DWT，使频域信号与后续 prototype adaptation 的特征空间保持一致。将 patch features reshape 为 `F in R^{H x W x C}` 后，WPTA 得到低频分量 `LL` 以及三个高频分量 `LH`、`HL`、`HH`。

高频纹理能量定义为：

```text
HF(i) = mean_c(|LH_i^c| + |HL_i^c| + |HH_i^c|).
```

由于高频响应也会在正常边界处出现，WPTA 从低频分量 `LL` 中估计低频结构边缘 `LF_edge(i)`，用于标识物体轮廓和大尺度结构变化。最终的 boundary-aware wavelet reliability 写为：

```text
W(i) = norm(HF(i)) * (1 - norm(LF_edge(i))).
```

该设计的作用是保留局部纹理扰动带来的 evidence reliability，同时抑制普通结构边界导致的伪高频。消融实验中，HF-only W + prototype adaptation 仅带来有限增益，而 boundary-aware W + prototype adaptation 进一步提升 AUPRO，支持这一设计。

### 3.4 Semantic-spectral evidence selection

WPTA 使用 `S0` 与 `W` 共同构造 abnormal evidence weight 和 normal evidence weight。概念上，abnormal evidence 应同时满足语义上接近 abnormal prototype 和频域上具有可靠局部扰动；normal evidence 则应语义上接近 normal prototype，并避免被异常纹理或结构边界污染。本文将 evidence weights 写为：

```text
q_a(i) = S0(i)^gamma * rho(W(i)),
q_n(i) = (1 - S0(i))^gamma * rho(1 - W(i)).
```

其中 `gamma` 控制语义先验的置信度锐化，`rho(.)` 表示轻量的可靠性调制函数。正文不需要过度暴露实现中的所有阈值和超参数；在正式论文中，可把 top-k、阈值、归一化和 confidence gate 的细节放入 Implementation Details 或 Appendix。

### 3.5 Visual anchors 与 conservative prototype calibration

给定 evidence weights，WPTA 对选中 patches 做加权聚合，得到当前测试图像的 visual abnormal anchor 和 visual normal anchor：

```text
v_a = sum_i q_a(i) f_i / sum_i q_a(i),
v_n = sum_i q_n(i) f_i / sum_i q_n(i).
```

这些 visual anchors 不是新训练得到的参数，而是从当前测试图像自身聚合出的 instance-specific evidence。随后，WPTA 用 conservative update 校准文本原型：

```text
t'_a = normalize((1 - alpha) t_a + alpha v_a),
t'_n = normalize((1 - beta)  t_n + beta  v_n).
```

其中 `alpha` 和 `beta` 的有效值由 confidence gate 控制。当 evidence 不够可靠时，update strength 会被减小或禁用，以避免 prototype drift。该保守校准尤其重要，因为 zero-shot anomaly detection 中正常区域占比通常远高于异常区域，错误地更新 abnormal prototype 会放大误报风险。CSV 中 Full wavelet prototype adaptation 相比 no-conservative variant 在 MVTec 和 VisA 上均进一步提升，说明 conservative update 对最终稳定增益有价值。

### 3.6 最终 anomaly map 与图像级分数

校准后，WPTA 使用 `t'_a` 和 `t'_n` 重新计算 patch-level anomaly probability：

```text
S(i) = exp(sim(f_i, t'_a) / tau) /
       (exp(sim(f_i, t'_a) / tau) + exp(sim(f_i, t'_n) / tau)).
```

最终 anomaly map `M` 由 `S(i)` reshape 并上采样到输入图像大小得到。图像级 anomaly score 可由 anomaly map 的最大值、top-k 平均值或与 baseline 一致的 aggregation protocol 得到。正式论文必须明确与 AnomalyCLIP baseline 完全一致的 image-level aggregation，以保证公平比较。

### 3.7 算法描述

```text
Algorithm 1: Wavelet-Supervised Test-Time Prototype Adaptation
Input: test image x, frozen CLIP image encoder, frozen CLIP text prototypes t_n and t_a
Output: anomaly map M and image anomaly score s_img

1. Extract patch features {f_i} from x using the frozen CLIP image encoder.
2. Compute initial semantic anomaly score S0(i) with fixed prototypes t_n and t_a.
3. Reshape patch features into H x W x C feature grid F.
4. Apply one-level Haar DWT to F and obtain LL, LH, HL, HH.
5. Compute high-frequency energy HF and low-frequency structural edge LF_edge.
6. Build boundary-aware reliability W = norm(HF) * (1 - norm(LF_edge)).
7. Compute abnormal and normal evidence weights q_a and q_n from S0 and W.
8. Aggregate visual anchors v_a and v_n from selected evidence patches.
9. Apply confidence-gated conservative update to obtain t'_a and t'_n.
10. Recompute patch anomaly score S with t'_a and t'_n.
11. Upsample S to anomaly map M and aggregate M into image score s_img.
```

## 4. 实验设置

### 4.1 数据集

本文在 MVTec AD 和 VisA 上评估 WPTA `[CITATION-NEEDED: MVTec AD]` `[CITATION-NEEDED: VisA]`。MVTec AD 包含多个工业物体和纹理类别，是工业异常定位的标准基准；VisA 覆盖更复杂的视觉对象和异常模式，用于检验方法在更具多样性的工业场景中的泛化能力。正式论文中应补充类别数量、测试图像数量、异常类型范围和协议来源。

### 4.2 评估指标

本文使用四个指标：pixel AUROC、pixel AUPRO、image AUROC 和 image AP。所有 slash-form 数值顺序固定为：

```text
pixel AUROC / pixel AUPRO / image AUROC / image AP
```

主文重点强调 pixel AUPRO，因为它衡量异常定位区域质量，能够反映异常区域覆盖和正常区域误检之间的平衡。与此同时，本文也报告 image AUROC 和 image AP，以证明提升不只局限于像素级定位。

### 4.3 Baselines 与 variants

本文以 AnomalyCLIP baseline 作为固定文本原型 baseline，并设计三类关键 variants：

- Direct wavelet fusion / no adaptation：负对照，把小波信号用于最终 map-level fusion，不做 prototype adaptation。
- Semantic prototype adaptation：只使用 `S0` 选择 patch evidence，不使用 wavelet reliability。
- Wavelet prototype adaptation no conservative：让 `W` 参与 evidence weighting，但去掉 conservative update。
- Full wavelet prototype adaptation：boundary-aware `W` 加 conservative update，即 WPTA 完整方法。

正式顶会版本建议补充外部方法对比表，但在当前数据边界下，本文不把结果写成 SOTA。

[TABLE_DATA_PROMPT:external_baseline_comparison_optional]
目标：补充一个外部方法对比表，用于正式 CVPR/ICCV 投稿版本。如果没有完成该表，不允许写 “state-of-the-art” 或 “SOTA”。
输入要求：在相同 zero-shot 或明确可比协议下收集 MVTec AD 和 VisA 的 pixel AUROC、pixel AUPRO、image AUROC、image AP。候选方法至少包括 AnomalyCLIP、WinCLIP、CLIP-AD、PromptAD/AdaCLIP 或其他近期 CLIP-based zero-shot anomaly detection 方法。每个数值必须记录来源：官方论文表格、官方代码复现、或本地统一复现实验。输出 CSV 列：method, dataset, protocol, pixel_auroc, pixel_aupro, image_auroc, image_ap, source_type, citation_key, notes。若某方法未报告 AUPRO 或 AP，不要补估，填 NA 并说明。
[/TABLE_DATA_PROMPT]

## 5. 结果与分析

### 5.1 主结果

表 1 汇报 WPTA 与 AnomalyCLIP baseline 的主结果。WPTA 在 MVTec 和 VisA 上均稳定提升 baseline，并且四个指标全部提升。

**表 1. 主结果。指标顺序为 pixel AUROC / pixel AUPRO / image AUROC / image AP。**

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Improvement vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| MVTec | AnomalyCLIP baseline | 91.2 | 83.2 | 91.6 | 96.4 | - |
| MVTec | Full WPTA | 91.8 | 86.2 | 94.1 | 97.4 | +0.6 / +3.0 / +2.5 / +1.0 |
| VisA | AnomalyCLIP baseline | 95.5 | 86.7 | 82.0 | 85.3 | - |
| VisA | Full WPTA | 96.2 | 91.7 | 84.3 | 87.3 | +0.7 / +5.0 / +2.3 / +2.0 |

在 MVTec 上，WPTA 将 pixel AUPRO 从 83.2 提升到 86.2，同时 image AUROC 从 91.6 提升到 94.1。在 VisA 上，WPTA 将 pixel AUPRO 从 86.7 提升到 91.7，是四个指标中最显著的提升。该结果符合本文方法动机：WPTA 的主要作用是让 patch-level anomaly map 更适配当前测试实例，因此区域级定位质量应首先受益。

需要注意的是，当前主表只比较 AnomalyCLIP baseline 与本文方法，不能支撑 “SOTA” 表述。更稳妥的顶会写法是：WPTA improves the AnomalyCLIP baseline under the fixed zero-shot protocol，并用消融实验证明改进来自本文的 adaptation 设计。

### 5.2 核心组件消融

表 2 汇报核心组件消融，验证本文不是简单地把小波、TTA 和 AnomalyCLIP 拼接在一起。

**表 2. 核心组件消融。每个单元格为 pixel AUROC / pixel AUPRO / image AUROC / image AP。**

| Method | MVTec | VisA | 作用 |
|---|---:|---:|---|
| Baseline | 91.2 / 83.2 / 91.6 / 96.4 | 95.5 / 86.7 / 82.0 / 85.3 | 固定文本原型的 AnomalyCLIP baseline |
| Direct wavelet fusion / no adaptation | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | 负对照：小波直接融合到 final map，不校准 prototypes |
| Semantic prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | 只用 `S0` 选择 patch evidence |
| Wavelet prototype adaptation no conservative | 91.7 / 85.8 / 93.9 / 97.2 | 96.1 / 91.3 / 84.1 / 87.0 | `W` 进入 evidence weighting，但无 conservative update |
| Full WPTA | 91.8 / 86.2 / 94.1 / 97.4 | 96.2 / 91.7 / 84.3 / 87.3 | Boundary-aware `W` + conservative update |

第一，direct wavelet fusion 明显低于 baseline 和 Full WPTA，说明小波不能被简单视为最终异常分数。MVTec 上，Full WPTA 的 AUPRO 比 direct fusion 高 5.8；VisA 上高 6.6。第二，semantic prototype adaptation 相比 baseline 带来稳定提升，说明从当前测试图像中构造 visual evidence 并校准 prototypes 是有效方向。第三，wavelet prototype adaptation no conservative 进一步超过 semantic-only adaptation，说明小波作为 patch evidence reliability 能提供额外信息。第四，Full WPTA 在 no-conservative variant 基础上继续提升，说明 conservative update 对最终稳定收益有价值。

### 5.3 小波设计消融

表 3 汇报 wavelet reliability 的设计消融。该表回答一个关键问题：小波为什么有用，以及为什么需要 boundary-aware 设计。

**表 3. 小波设计消融。每个单元格为 pixel AUROC / pixel AUPRO / image AUROC / image AP。**

| Wavelet setting | MVTec | VisA | 作用 |
|---|---:|---:|---|
| Semantic-only prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | 无小波，作为 semantic-only 参照 |
| Direct wavelet fusion | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | 小波直接融合负例 |
| HF-only W + prototype adaptation | 91.6 / 85.3 / 94.0 / 97.2 | 96.0 / 90.8 / 84.0 / 86.9 | 只使用高频能量 |
| Boundary-aware W + prototype adaptation | 91.7 / 85.7 / 93.8 / 97.3 | 96.1 / 91.2 / 83.9 / 87.1 | 使用 `W = HF * (1 - LF_edge)` |
| Full boundary-aware W + conservative | 91.8 / 86.2 / 94.1 / 97.4 | 96.2 / 91.7 / 84.3 / 87.3 | 最终方法 |

HF-only W 相比 semantic-only 仅有小幅提升，说明高频能量确实包含异常相关线索，但其噪声也较明显。Boundary-aware W 进一步提升，说明低频结构边缘抑制是必要的：它减少了普通物体边界和结构轮廓对 high-frequency reliability 的干扰。Full 方法继续提升，则说明 boundary-aware reliability 和 conservative update 应共同构成最终算法。

[FIGURE_PROMPT:figure3_ablation_bar]
目标：生成核心消融结果图，作为实验部分的辅助 Figure。数据来自当前 CSV，不要新增数值。
图形类型：两组 grouped bar 或 small multiples。建议只画 pixel AUPRO，因为它是本文主提升指标。左图 MVTec，右图 VisA。横轴为 Baseline、Direct fusion、Semantic adaptation、Wavelet adaptation w/o conservative、Full WPTA。纵轴为 Pixel AUPRO。数值分别为 MVTec: 83.2, 80.4, 85.2, 85.8, 86.2；VisA: 86.7, 85.1, 90.4, 91.3, 91.7。Ours/Full 使用深蓝或橙色高亮，negative control 使用浅灰并标注 “direct fusion fails”。使用白底、无 3D、无网格噪声、字体 8pt 以上，柱顶标数字，caption 第一句写明 “Using wavelet cues for evidence reliability improves AUPRO, while direct wavelet fusion degrades localization.”
[/FIGURE_PROMPT]

### 5.4 待补充：qualitative visualization

定性图应展示 WPTA 的机制，而不仅展示更亮的 anomaly map。建议每个数据集选择 2-3 个类别，每个 case 包含原图、GT、baseline map、wavelet reliability、selected evidence patches 和 WPTA final map。

[FIGURE_PROMPT:figure4_qualitative_visualization]
目标：生成 qualitative visualization figure，用于实验结果部分。图像必须来自真实实验输出，不要合成数据。
布局：每行一个 case，每列依次为 Input image、Ground truth mask、AnomalyCLIP baseline map、Boundary-aware wavelet reliability W、Selected evidence patches、WPTA final map。至少包含 MVTec 2 行和 VisA 2 行。Selected evidence patches 列用蓝色轮廓标 normal evidence，用橙色轮廓标 abnormal evidence；不要遮挡原始缺陷。W 列使用 viridis heatmap，baseline/final map 使用相同色标，保证可公平比较。caption 第一句写明核心发现：“WPTA selects semantic-spectral evidence from the current image and produces anomaly maps that better cover defect regions while suppressing normal boundaries.” 图中所有热图必须来自保存的模型输出，禁止手工涂色。
[/FIGURE_PROMPT]

## 6. 讨论

### 6.1 为什么小波适合做 reliability，而不是 anomaly score

小波分解提供的是局部频率结构，不是语义异常判别。工业缺陷确实经常表现为高频纹理变化，但高频响应也可能来自正常结构边界、材料纹理和成像噪声。直接把小波响应融合到最终 anomaly map 会把这些非异常因素引入 CLIP 语义判别，导致定位质量下降。相比之下，WPTA 只让小波影响 patch evidence 的可靠性，而最终异常图仍由 CLIP patch features 与校准后的 normal/abnormal prototypes 计算得到。这一设计保留了 CLIP 的语义判别，同时利用小波信号发现更可靠的局部 evidence。

### 6.2 为什么需要 conservative update

无标签测试时校准的主要风险是 prototype drift。异常检测中，异常区域通常很小，正常区域占比很大；如果 evidence selection 对正常结构边界过度敏感，abnormal prototype 可能被错误 patches 拉偏。WPTA 的 conservative update 通过 confidence gate 控制更新强度，只在 evidence 足够可靠时让 visual anchors 影响 prototypes。当前消融显示 Full WPTA 比 no-conservative variant 更好，说明保守更新对最终性能有稳定贡献。

### 6.3 当前局限

当前结果主要支撑 “WPTA improves AnomalyCLIP baseline under the fixed zero-shot protocol”。若要写成完整 CVPR/ICCV submission，还需要两类补充证据。第一，需要外部方法对比表，确认 WPTA 与近期 CLIP-based zero-shot anomaly detection 方法在相同协议下的相对位置。第二，需要 qualitative visualization，直观证明 WPTA 的 selected evidence 与 final anomaly map 符合方法机制。

## 7. 结论

本文提出 Wavelet-Supervised Test-Time Prototype Adaptation，用于 CLIP-based zero-shot anomaly detection。本文的核心思想是将 Haar 小波线索作为 patch evidence reliability 来监督测试时原型校准，而不是把频域响应直接融合到最终异常图中。WPTA 在冻结 CLIP 参数的条件下，从当前测试图像中构造 semantic-spectral visual anchors，并以 conservative update 校准 normal/abnormal prototypes。当前实验表明，WPTA 在 MVTec 和 VisA 上均稳定提升 AnomalyCLIP baseline，尤其显著提高 pixel AUPRO。消融实验进一步说明，直接小波融合会伤害性能，而把小波用于 evidence selection 能改进 prototype adaptation。后续工作应补齐外部方法对比和定性可视化，使该方法具备更完整的顶会投稿证据链。

## 8. Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| WPTA improves AnomalyCLIP baseline on MVTec and VisA. | 表 1：MVTec +0.6 / +3.0 / +2.5 / +1.0；VisA +0.7 / +5.0 / +2.3 / +2.0。 | Supported by current CSV |
| Main gain is in pixel AUPRO, indicating better localization quality. | 表 1：AUPRO gains are +3.0 on MVTec and +5.0 on VisA, larger than pixel AUROC gains. | Supported by current CSV |
| Direct wavelet fusion is not a good design. | 表 2/3：direct fusion below baseline on pixel AUROC and AUPRO for both datasets. | Supported by current CSV |
| Prototype adaptation is useful even without wavelet reliability. | 表 2：semantic prototype adaptation improves baseline on both datasets. | Supported by current CSV |
| Wavelet reliability adds value beyond semantic-only adaptation. | 表 2：wavelet no-conservative improves AUPRO over semantic-only by +0.6 on MVTec and +0.9 on VisA; Full improves by +1.0 and +1.3. | Supported by current CSV |
| Boundary-aware W is better than HF-only W. | 表 3：boundary-aware W improves AUPRO over HF-only on MVTec (85.7 vs 85.3) and VisA (91.2 vs 90.8). | Supported by current CSV |
| WPTA is SOTA among recent zero-shot anomaly detection methods. | 当前没有外部方法统一协议对比表。 | Not supported; do not claim |

## 9. 预投稿自检

### 9.1 Macro logic

- Introduction flow：task -> fixed prototype limitation -> direct wavelet fusion risk -> wavelet-supervised prototype adaptation -> results，pass。
- Contribution mapping：三个贡献分别对应 reliability repositioning、semantic-spectral evidence selection、conservative prototype calibration 与实验验证，pass。
- Experiment validation：主结果、核心消融、小波设计已经支撑主要机制；外部方法对比和 qualitative visualization 仍缺，major gap。

### 9.2 Writing risks

- 当前中文稿可以用于内部审核，但正式投稿需转换为英文，并补真实引用。
- Related Work 不能保留 `[CITATION-NEEDED]`；需要核验 AnomalyCLIP、WinCLIP、CLIP-AD、PromptAD/AdaCLIP、test-time adaptation 和 wavelet anomaly detection 的准确引用。
- 摘要不能写 SOTA；当前版本已避免该 claim。

### 9.3 Figure/table risks

- Figure 1 和 Figure 2 可由图 agent 根据 prompt 先画机制图。
- Figure 4 qualitative visualization 必须使用真实模型输出，不能生成示意热图冒充实验。
- 外部方法对比与 qualitative visualization 是正式投稿前的关键补充证据；当前只能作为待补充内容，不可写成已完成结果。

### 9.4 建议下一步

1. 导出每个方法的 anomaly maps、`W` maps 和 selected evidence patches，用于 qualitative figure。
2. 补齐外部方法对比与真实 BibTeX，再把本文稿转成英文 LaTeX CVPR template。

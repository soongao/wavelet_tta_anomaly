# CLIP-ZSAD 文献学习笔记

更新时间：2026-07-30

本文档用于约束后续论文改写。记录原则是：只写已经从 arXiv、DOI/CrossRef、DBLP 或论文 PDF 文本中核验到的信息；未读到原文的内容只作为背景线索，不作为正文精确 claim。

## 0. 统一任务口径

- ZSAD / ZSAS 通常表示测试目标类别上没有训练样本或没有参考图像，不等于方法从未使用辅助异常数据。AnomalyCLIP、AdaCLIP、VCP-CLIP、AA-CLIP 等都使用辅助数据或跨数据集训练来学习提示、adapter 或上下文模块。
- 常见输出包括 image-level anomaly score 和 pixel-level anomaly map。常见指标包括 image AUROC/AP、pixel AUROC/AUPRO/PRO/AP/F1-max。
- 本文新增模块应表述为：在已训练 AnomalyCLIP/CLIP 表征基础上，测试时不反传、不更新模型参数、不使用目标类别标注；从单张测试图内部选择 evidence 并保守校准 prototypes。

## 1. WinCLIP

核验来源：arXiv 2303.14814；CVPR 2023 DOI `10.1109/CVPR52729.2023.01878`；PDF 文本。

题名：WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.

核心定义与公式：

- CLIP 零样本分类：
  \[
  p(s=s_i|x; s\in S)=
  \frac{\exp(\langle f(x),g(s_i)\rangle/\tau)}
       {\sum_{s\in S}\exp(\langle f(x),g(s)\rangle/\tau)}.
  \]
- 两类 anomaly classification 采用 `"normal [o]"` 与 `"anomalous [o]"`；CPE 组合 normal/anomalous state words 与 anomaly-specific templates。
- Window feature：
  \[
  F^{W}_{ij}=f(x\odot w_{ij}).
  \]
- overlapping windows 使用 harmonic aggregation：
  \[
  \bar M^W_{0,ij}=
  \left(\frac{1}{\sum_{u,v}(w_{uv})_{ij}}
  \sum_{u,v}\frac{(w_{uv})_{ij}}{M^W_{0,uv}}\right)^{-1}.
  \]
- WinCLIP+ 引入少量正常参考图，reference association：
  \[
  M_{ij}=\min_{r\in R}\frac{1}{2}(1-\langle F_{ij},r\rangle),
  \quad
  M^W=\frac{1}{3}(M^P+M^W_s+M^W_m),
  \]
  image score:
  \[
  \mathrm{ascore}_{W}(x)=\frac{1}{2}
  (\mathrm{ascore}_0(f(x))+\max_{ij}M^W_{ij}).
  \]

实验协议：

- 数据集：MVTec AD、VisA。
- zero-shot WinCLIP 不使用正常参考图；few-normal-shot WinCLIP+ 使用 K 张 normal reference images。
- 指标：classification AUROC/AUPR/F1-max；segmentation pixel AUROC/PRO/F1-max。
- 摘要中报告：MVTec zero-shot AC/AS 为 91.8/85.1 AUROC，VisA 为 78.1/79.6；WinCLIP+ 在 1-normal-shot 上更高。

对本文的启发：

- WinCLIP 已经证明 normal/anomalous text prototypes 与多尺度局部窗口可用于 CLIP-ZSAD。
- WinCLIP+ 说明有些缺陷需要正常参考图才能定义，但它的参考来自外部 normal images；本文不能把“参考”写成 WinCLIP 没有讨论过，只能说本文估计的是单张测试图内部的图像条件化参照。
- WinCLIP 的失败案例涉及需要视觉参考的逻辑/细微异常，适合作为“固定文本语义不足”的背景。

## 2. AnomalyCLIP

核验来源：arXiv 2310.18961；OpenReview/ICLR 2024 元数据；PDF 文本。

题名：AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection.

核心定义与公式：

- 目标是学习 object-agnostic normality/abnormality prompts，而不是依赖目标物体名。
- prompt 学习使用 glocal loss：
  \[
  L_{\mathrm{total}}=L_{\mathrm{global}}
  +\lambda\sum_{M_l\in M}L^{{M_l}}_{\mathrm{local}}.
  \]
- local prediction 使用 normal/anomaly text embeddings 与局部 visual embeddings 的相似度：
  \[
  S^{(j,k)}_{n,M_l}=P(g_n,f^{(j,k)}_{i,M_l}),
  \quad
  S^{(j,k)}_{a,M_l}=P(g_a,f^{(j,k)}_{i,M_l}).
  \]
- local loss 包含 focal loss 与 Dice loss：
  \[
  L_{\mathrm{local}}=
  \mathrm{Focal}(\mathrm{Up}([S_n,S_a]),S)
  +\mathrm{Dice}(\mathrm{Up}(S_n),I-S)
  +\mathrm{Dice}(\mathrm{Up}(S_a),S).
  \]
- inference 中 image-level score 使用 \(P(g_a,f_i)\)；pixel map 合并多层 normal/anomaly maps：
  \[
  \mathrm{Map}=G_\sigma\left(\sum_{M_l\in M}
  \left(\frac{1}{2}(I-\mathrm{Up}(S_{n,M_l}))
  +\frac{1}{2}\mathrm{Up}(S_{a,M_l})\right)\right).
  \]

实验协议：

- 工业数据包括 MVTec AD、VisA、MPDD、BTAD、SDD、DAGM、DTD-Synthetic；还包含多种医学数据。
- 使用 CLIP ViT-L/14@336px；CLIP 参数冻结；learnable prompt length 为 12；文本 encoder 前 9 层加入 learnable tokens；\(\lambda=4\)。
- 用辅助 AD 数据学习 prompts：非 MVTec 时在 MVTec AD test data 上 fine-tune，MVTec 则在 VisA test data 上 fine-tune。
- 指标：image AUROC/AP，pixel AUROC/AUPRO。

对本文的启发：

- 本项目当前 baseline/特征来自 AnomalyCLIP，因此正文必须承认其 prompts 已由辅助异常数据学习，不能写成整个系统从未使用辅助数据。
- AnomalyCLIP 解决的是 object-agnostic normal/abnormal prototypes 的学习；本文新增部分解决的是测试图像内部哪些局部证据可以校准这些 prototypes。

## 3. VCP-CLIP

核验来源：arXiv 2407.12276；ECCV 2024 DOI `10.1007/978-3-031-72890-7_18`；PDF 文本。

题名：VCP-CLIP: A Visual Context Prompting Model for Zero-Shot Anomaly Segmentation.

核心定义与公式：

- 先构造 unified text prompting：
  \[
  H=[a][photo][of][a][state][v_1]\cdots[v_r],
  \]
  state words 如 good/damaged。
- Deep text prompting 在 text encoder 各层插入 trainable embeddings：
  \[
  [s_i,\_,H_i,e_i,J_i]=
  \mathrm{Layer}^{text}_i([s_{i-1},P_{i-1},H_{i-1},e_{i-1},J_{i-1}]),
  \quad
  g=\mathrm{TextProj}(\mathrm{Norm}(e_{N_t})).
  \]
- baseline anomaly map：
  \[
  M_1^l=\mathrm{softmax}(\mathrm{Up}(\tilde F_s^l \tilde F_t^T)/\tau_1).
  \]
- Pre-VCP 将 global image feature 映射到 word embedding space，并与 learnable category vectors 相加：
  \[
  z_i(x_i,v_i)=x_i+v_i,\quad
  H_v=[a][photo][of][a][state][z_1]\cdots[z_r].
  \]
- Post-VCP 用 patch-level visual embeddings 更新 text embeddings：
  \[
  Q_t=F_tW_t^q,\quad K_s^l=Z_s^lW_s^k,\quad V_s^l=Z_s^lW_s^v,
  \]
  并通过 multi-head attention 得到更新后的 \(O_t^l\)，最终：
  \[
  M_2^l=\mathrm{softmax}(\mathrm{Up}(\tilde Z_s^l \tilde O_t^{lT})/\tau_2).
  \]

实验协议：

- 10 个工业 anomaly segmentation 数据集，包括 MVTec AD、VisA、BSD、GC、KSDD2、MSD、Road、RSDD、BTech、DAGM。
- 对非 VisA 数据集，用 VisA 训练；对 VisA，用 MVTec AD 训练。
- CLIP ViT-L-14-336；从 layers {6,12,18,24} 取 features；resize 518x518；Adam，10 epochs，batch size 32。
- 指标：pixel AUROC、PRO、AP。

对本文的启发：

- VCP-CLIP 已经使用 image-conditioned text embeddings，不能说“图像条件化 prompt/prototype”本身没人做。
- 它的 conditioning 由训练式 Pre-/Post-VCP 完成，并且没有引入正常参考图；论文也承认某些异常必须依赖 normal images 才能准确定位。本文应定位为“单图 evidence selection 与保守 prototype calibration”，不是视觉上下文 prompt learning。

## 4. AA-CLIP

核验来源：arXiv 2503.06661；CVPR 2025 DOI `10.1109/CVPR52734.2025.00447`；PDF 文本。

题名：AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP.

核心定义与公式：

- 论文定义 CLIP 的 anomaly-unawareness：normal/anomaly text features 与 patch features 难以清晰区分。
- residual adapter：
  \[
  x_i^{residual}=\mathrm{Norm}(\mathrm{Act}(W_i x_i)).
  \]
- 第一阶段构造 normal/anomaly text anchors \(T_N,T_A\)，通过 image 和 patch features 对齐：
  \[
  p_{cls}=\mathrm{CosSim}(V_{image},[T_N,T_A]),
  \quad
  p_{seg}=\mathrm{CosSim}(V_{patch},[T_N,T_A]).
  \]
- alignment loss：
  \[
  L_{cls}=\mathrm{BCE}(p_{cls},y),
  \quad
  L_{seg}=\mathrm{Dice}(p_{seg},S)+\mathrm{Focal}(p_{seg},S),
  \quad
  L_{align}=L_{cls}+L_{seg}.
  \]
- disentangle loss：
  \[
  L_{dis}=|\langle T_N,T_A\rangle|^2,
  \quad
  L_{total}=L_{align}+\gamma L_{dis}.
  \]
- 第二阶段将 patch features 投影并聚合到 text anchors：
  \[
  V_{patch}^i=\mathrm{Proj}_i(F^i),\quad
  V_{patch}=\sum_{i=1}^{4}V_{patch}^i.
  \]

实验协议：

- 工业数据：MVTec AD、VisA、BTAD、MPDD；医学数据来自 BMAD 等。
- 跨数据集训练：用 VisA 训练其他工业数据集；VisA 结果用 MVTec AD 训练。
- OpenCLIP ViT-L/14，resize 518x518；CLIP 主参数冻结，训练 residual adapters；使用 layers 6/12/18/24；两阶段训练 5 + 20 epochs。
- 指标主要为 image-level 和 pixel-level AUROC。

对本文的启发：

- AA-CLIP 也是“增强 text anchors + patch alignment”，但依赖训练式 residual adapters 与两阶段优化。
- 本文不能把“normal/abnormal anchor separation”写成自己的贡献；可强调本文不学习 adapters，而是在测试时用当前图像 evidence 小步校准已有 prototypes。

## 5. 其他 ZSAD 背景

- AdaCLIP：ECCV 2024，DOI `10.1007/978-3-031-72761-0_4`。使用 static + dynamic hybrid learnable prompts，在辅助标注异常数据上优化；说明“动态 prompt / image-conditioned prompt”已有训练式路线。
- PromptAD：WACV 2024，DOI `10.1109/WACV57701.2024.00113`。使用 text prompts 做 zero-shot anomaly detection，是 text prompt route 的背景。
- CLIP-AD：arXiv 2311.00453。提出 representative vector selection 与 staged dual-path，指出 direct anomaly map computation 有 opposite predictions / irrelevant highlights；可作为“直接相似度图不稳”的背景，但没有 DOI/正式会议信息时不要写强出版信息。
- VAND/APRIL-GAN 等 challenge/report 方法常作为外部 baseline，但协议、训练数据和后处理可能不一致。正文如使用外部表格，必须标注 protocol-reference，而不能当作同协议横向结论。

## 6. 与本文新增方法相关的文献

### 6.1 Wavelet / frequency foundations

- Daubechies 1988, DOI `10.1002/cpa.3160410705`，构造 compactly supported orthonormal wavelets。
- Mallat 1989, DOI `10.1109/34.192463`，多分辨率信号分解的 wavelet 表示。
- 本文使用的是最简单的 Haar-style 2x2 分解，不需要宣称新的 wavelet 变换。公式只应作为局部 reliability 的定义。

### 6.2 Frequency in CLIP-ZSAD

- FE-CLIP：ICCV 2025，DOI `10.1109/ICCV51701.2025.01971`。用 DCT/frequency-enhanced adapters 将频率信息注入 CLIP features，是训练式 frequency feature route。
- WMoE-CLIP：arXiv 2603.06313 / ICASSP 2026 oral。使用 Haar wavelet decomposition 得到 low-frequency \(F_L\) 与 high-frequency \(F_{LH},F_{HL},F_{HH}\)，并聚合：
  \[
  F_H=F_{LH}+F_{HL}+F_{HH}.
  \]
  其 frequency-enhanced representation 为：
  \[
  F_p=F_H\odot W_h+F_L.
  \]
  这是训练式 prompt / feature enhancement，不是测试时 evidence selection。
- HarmoniAD：arXiv 2601.00327。用 CLIP features 转频域，经 soft gate 分成高/低频双分支；高频分支增强 texture/edge 与细小缺陷；低频分支维持 global semantics。采用多类联合训练。它进一步说明 frequency prior 已经是相关方向，本文不能把“频率有用”作为 novelty。

### 6.3 Test-time / prompt adaptation

- Test-Time Prompt Tuning (TPT)：NeurIPS 2022 / arXiv 2209.07511。测试时优化 prompts 以改善 VLM zero-shot generalization。
- CoCoOp：CVPR 2022，DOI `10.1109/CVPR52688.2022.01631`。用图像条件化 prompts 改善 novel class 泛化。
- 对本文的约束：不能把“测试图像条件化”或“prompt/prototype adaptation”本身写成新颖性来源。本文更准确的机制是：不进行测试时优化，而是用语义之外的局部 reliability 约束 evidence selection，并只做保守 prototype calibration。

## 7. 本文可以写与不能写

可以写：

- 现有 CLIP-ZSAD 主要围绕 text prompts、visual context、adapters、窗口/多层 features 改善 normal/abnormal semantic alignment。
- 频率、高频、边缘/纹理细节已经被用于 anomaly detection 和 CLIP-ZSAD；本文不以“发现频率有用”为贡献。
- 本文新增模块的机制 claim 是：同一局部信号直接做 map fusion 会混淆正常纹理/结构边界，而作为 evidence-selection reliability 时，可以帮助构造图像条件化参照。
- 受控 MVTec/VisA 消融可以支持“direct wavelet fusion 是负控、semantic-only adaptation 是强对照、boundary-aware reliability + conservative calibration 是当前稳定方法”。

不能写：

- 不能写“本文开创 wavelet/frequency 用于 ZSAD 的路线”。
- 不能写“整个方法没有使用任何辅助数据”，因为 AnomalyCLIP baseline/prompt 来自辅助数据；只能写“本文新增校准模块在测试时不使用目标类别训练样本、标注或参数更新”。
- 不能写外部 baseline 表是严格同协议横向比较，除非逐项核验 split、backbone、input resolution、post-processing、prompt setting、训练数据和 metric implementation。
- 不能把 MPDD/BTAD/DTD 的 system-level results 用来证明 MVTec/VisA 受控机制；机制结论应限于已有受控消融。

## 8. 对当前论文改写的直接要求

- 引言应从“CLIP-ZSAD 已有 prompts/window/context/adapters”进入，再提出“局部响应需要当前图像参照”的互补问题。
- 方法节的公式应与实现保持一致：\(s_i^0\)、Haar \(LL/LH/HL/HH\)、\(HF\)、\(E=\|\nabla LL\|_2\)、\(W=\mathrm{Norm}(\widehat{HF}(1-\widehat E))\)、\(\omega^a,\omega^n\)、top-k visual prototypes、保守 calibration、final rescoring。
- 相关工作应按“CLIP-ZSAD 表征/提示路线、频率/局部结构路线、测试时/参照估计路线”组织，不要逐篇堆列表。
- 实验叙事只讨论机制，不以外部横向比较结果为中心。

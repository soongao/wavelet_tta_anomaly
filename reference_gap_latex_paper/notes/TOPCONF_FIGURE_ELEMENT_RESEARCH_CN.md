# 顶会论文图像元素表示方式调研与 ICNR 落图规范

本文档面向 `reference_gap_latex_paper` 中的两张核心图：方法架构图和定性定位图。目标不是把 Mermaid 逻辑图直接美化，而是把 ICNR 的概念拆成顶会论文常用的视觉元素，再给出可执行的画图规范。

调研样本以 CVPR/ICCV/ECCV/ICLR 视觉、视觉语言和异常定位论文为主，包括 CLIP、SAM、MAE、DINO、WinCLIP、AnomalyCLIP、AdaCLIP、VCP-CLIP、AA-CLIP、PromptAD 等。本文核对了 arXiv PDF 图题文本、抽查了代表性页面渲染图，并用本地 `references.bib` 核对异常检测论文名称、场景和图号；最终画图前仍应打开原论文 PDF 人工复核细节。

## 0. 调研方法与证据层级

本次调研不把外部论文图复制到本文档，只抽取其视觉表达规则。证据分三层：

- 图题级证据：核对每篇论文 Figure caption，确认该图要表达的对象和图族。
- 页面级抽查：渲染并目视检查 CLIP Fig. 1、SAM overview、MAE architecture、DINO attention、WinCLIP workflow/qualitative、AnomalyCLIP overview、AdaCLIP qualitative 等页面。
- 元素级归纳：把反复出现的对象归纳成 image thumbnail、encoder stack、patch grid、prompt token、prototype anchor、response map、selection mask、mixer/gate、output heatmap 等可复用元素。

需要强调的是：顶会方法图不是“越多异形越高级”。它们通常只在关键概念上使用有语义的形状，其余部分保持简单、对齐和克制。本文后面的推荐不是为了装饰，而是为了让 ICNR 的核心机制在视觉上更接近论文正文的数学结构。

## 1. 总结结论

顶会论文中的方法图通常不是“普通框图”，但也不是任意堆异形。它们使用稳定的视觉语法：

- 真实图像或小热力图表示视觉证据。
- patch、token、window、mask、attention map 用网格、小块、矩阵或堆叠图表示。
- encoder、decoder、adapter 用模块化结构表示，但会区分 frozen、trainable、shared、lightweight。
- prompt、text prototype、anchor 通常画成 token 串、胶囊、向量点或 embedding-space anchor。
- similarity、matching、fusion、gating 通常画成小算子节点，而不是大段公式。
- qualitative figure 是 raster + vector hybrid：图像和热力图是 raster，文字、边框、箭头、mask contour、zoom box 是 vector。

对 ICNR 来说，最终图最重要的不是展示所有变量，而是让读者在 10 秒内看懂：

> frozen CLIP/AnomalyCLIP 提供 normal/abnormal 语义轴；ICNR 从当前测试图像中选择可靠 normal patch，形成 image-conditioned normal reference `r_n(x)`，再重算 anomaly map。

## 2. 样本论文中的视觉元素惯例

### 2.1 基础视觉/视觉语言方法图

| 论文/图族 | 已核对的图题信息 | 图中通常怎么表示 | 对 ICNR 的启发 |
|---|---|---|---|
| CLIP, ICML 2021 Fig. 1 | caption 明确是 image encoder 与 text encoder 预测 image-text pairing，test time 用 text encoder 合成 zero-shot classifier | 双塔 encoder、image/text embedding、小型相似度矩阵、类别文本/句子短标签 | ICNR 的语义分支应画成 frozen visual/text prototype 共同定义 semantic axis，不要只画一个 `CLIP features` 框 |
| CLIP prompt engineering figures | caption 讨论 prompt engineering、prompt ensemble 对 zero-shot performance 的影响 | prompt template 或 class text 通常是短 token/句子块，embedding 是向量或矩阵 | `t_n/t_a` 用 normal/abnormal capsule 或 anchor 即可，prompt 句子放 caption 或正文 |
| SAM, ICCV 2023 Fig. 4 | caption 明确是 heavyweight image encoder 输出 image embedding，再由多种 prompt 查询 mask | 大 image embedding、prompt encoder、小 mask decoder、多 mask outputs、confidence score | ICNR 可以借鉴“frozen embedding + lightweight query/rescoring”的视觉重心，突出 encoder frozen、小模块轻量 |
| MAE, CVPR 2022 Fig. 1 | caption 明确 input image patches 被 mask，encoder 只处理 visible patches，decoder 用 mask token 重构 pixels | patch grid、visible/masked patch、token sequence、encoder/decoder stack | patch features、selected normal patches、Haar 2x2 邻域都应该落在 grid/token 视觉语言上 |
| DINO, ICCV 2021 Fig. 1/3 | caption 明确展示 ViT self-attention map，Fig. 3 展示 multiple heads attention maps | 真实图像旁边直接放 attention heatmaps 或 binary/soft mask，不用大文字解释 attention | `q_i`、`W_i`、final anomaly map 都应是小热图或 grid map |

### 2.2 CLIP 异常检测方法图和定位图

| 论文/图族 | 已核对的图题信息 | 图中通常怎么表示 | 对 ICNR 的启发 |
|---|---|---|---|
| WinCLIP, CVPR 2023 Fig. 3 | caption 明确是 multi-scale windows through CLIP image encoder，window embeddings encode global information within each window | 多尺度 window 覆盖图像、patch/window/image-level feature、小型 CLIP encoder、text class prototypes | ICNR 的 evidence selection 可以画成同一 patch grid 上的 selected cells，避免展开成多个集合/权重框 |
| WinCLIP, CVPR 2023 Fig. 4/5 | caption 明确 workflow 里 text embeddings as class prototypes，qualitative comparison 是 MVTec/VisA anomaly segmentation rows | text prototype、multi-scale feature correlation、reference association、方法列对齐 heatmap | 定性定位图按 Input/GT/Baseline/Ours 列组织，ICNR 列必须使用同一 colormap/normalization 规则 |
| AnomalyCLIP, ICLR 2024 Fig. 2 | caption 明确 overview 包含 object-agnostic text prompt templates、glocal context optimization、DPAM | prompt token blocks、cosine similarity、visual embedding、DPAM attention map、segmentation map | ICNR 要明确 text prototype 来自已有 CLIP/AnomalyCLIP；不要把 ICNR 画成新的 prompt-learning 训练框架 |
| AnomalyCLIP, ICLR 2024 Fig. 3/4 | caption 明确 DPAM visualization 与 segmentation visualization | attention matrix/map、segmentation heatmap、输入图和 mask/heatmap rows | 本文若解释机制，应直接展示 `q_i/W_i/final map`，不要只写 response words |
| AdaCLIP, ECCV 2024 Fig. 2/3/7 | caption 明确 framework、anomaly maps visualization、normal/abnormal patch embedding t-SNE | static/dynamic prompt blocks、semantic fusion、anomaly maps、embedding clusters | `r_n(x)` 可借鉴 cluster centroid/anchor 语法，但方法图中只放一个小 embedding anchor，不把 t-SNE 作为主图 |
| VCP-CLIP, ECCV 2024 Fig. 1/3/5 | caption 明确 visual context prompting 与 qualitative segmentation results | global visual context 进入 text prompt，dense visual feature 与 text feature 交互，定位结果按列展示 | ICNR 的 image-conditioned 应画成 current-image evidence 回流到 normal reference，而不是另一个训练或 prompt branch |
| AA-CLIP, CVPR 2025 Fig. 4/6 | caption 明确 two-stage training pipeline、patch features aligned with text anchors、localization results | normal/anomaly text anchors、residual adapter 插片、patch-level alignment、localization heatmap comparison | `t_n/t_a` 可画成 text anchors，`r_n(x)` 是当前图像 visual normal anchor；adapter 形状不适合本文，因为 ICNR 不训练 adapter |
| PromptAD, WACV 2024 Fig. 2/3 | caption 明确 PromptAD 包含 SC/EAM，visual encoder transformed with V-V attention，qualitative comparison 是 pixel-level anomaly detection | learnable prompt blocks、semantic concatenation、attention branch、qualitative heatmap table | normal evidence/reference 可借鉴 memory/anchor 表示，但 ICNR 必须强调 single test image evidence，不画 external memory bank |
| PatchCore, CVPR 2022 图族 | backbone feature map、memory bank、nearest-neighbor scoring、anomaly map 是典型表达 | feature map stack、memory bank row、nearest-neighbor arrow、anomaly map | `r_n(x)` 不能画成 dataset memory bank；它应是当前图像 selected normal patches 聚合出的 diamond/anchor |

### 2.3 跨论文的共同规律

- 真实视觉证据优先：顶会视觉论文很少用 generic image icon 代替输入图或热图。
- map 直接画出来：attention、response、segmentation、anomaly score 通常用小图或热力图，不写成文字框。
- token/patch 用规则形状：ViT patch、mask token、prompt token、learnable token 都用小方块、胶囊或网格。
- encoder 有层级感：encoder/decoder 可以是 box，但常用 stack/chip/tower，并用 lock、灰色或虚线区分 frozen/shared。
- prototype/anchor 不画普通矩形：text prototype、class anchor、normal/anomaly anchor 更常见是 capsule、dot、diamond、vector endpoint。
- qualitative 图追求可比性：同尺寸 cell、同一 colormap、同一行 normalization、GT contour 或 binary mask、Ours 放最右。

## 2.4 ICNR 应选择的图像元素

最终方法图只建议选择 12 类元素，其中 7 到 8 个作为主要视觉对象：

| 优先级 | 图像元素 | 是否进入方法图 | 是否进入定性图 | 选择理由 |
|---|---|---|---|---|
| 必选 | real input thumbnail | 是 | 是 | 顶会视觉图以真实样本建立任务语境，ICNR 也需要让 reader 看到 texture/boundary |
| 必选 | locked encoder stack/chip | 是 | 否 | 表达 frozen CLIP/AnomalyCLIP，不让读者误会本文训练 backbone |
| 必选 | patch token grid | 是 | 可选 | ICNR 的证据选择发生在 patch grid 上，这是比 `F` 框更准确的视觉元素 |
| 必选 | normal/abnormal text capsules | 是 | 否 | `t_n/t_a` 是语义轴来源，短 capsule 比长 prompt 框更像 CLIP 图族 |
| 必选 | semantic axis + patch dots | 是 | 否 | 表达 `q_i` 的语义含义，替代 softmax 公式 |
| 必选 | `q_i` response mini-map | 是 | 可选 | 顶会 attention/anomaly 图常直接画 map，`q_i` 不应是文字框 |
| 必选 | Haar 2x2 subband icon | 是 | 否 | 这是结构分支的最小可识别图像符号 |
| 必选 | `W_i` reliability mini-map | 是 | 可选 | 表达 structure/reliability signal 只用于 selection |
| 必选 | selected patch grid | 是 | 可选 | ICNR 核心机制，应该是 Panel B 最大对象 |
| 必选 | green diamond `r_n(x)` | 是 | 否 | 图像条件化 normal reference 是论文核心贡献 |
| 可选 | small conservative mixer/gate | 是 | 否 | 只需小圆点/slider，不要放大成主模块 |
| 必选 | final anomaly heatmap/overlay | 是 | 是 | 与异常定位任务强绑定，方法图和定性图都需要 |

不建议进入主方法图的元素：

- dataset memory bank：会误导为 PatchCore/one-shot memory。
- trainable adapter 插片：会误导为 AA-CLIP/adapter training。
- 大 t-SNE panel：会抢走 `r_n(x)` 的机制焦点。
- prompt paragraph：图内文字会过多，且不是 ICNR 的核心贡献。
- Haar/softmax/update 公式：应留给正文，图里只保留 `q_i`、`W_i`、`r_n(x)`。

## 3. 顶会图中的元素表示方式

### 3.0 逐元素调研提炼：顶会到底怎么“画”这些对象

下面的表不是元素清单，而是视觉语法清单。顶会论文里的元素通常靠形状、材质、尺寸层级和连线关系共同表达语义，而不是靠方框里的长文字。

| 元素族 | 顶会常见画法 | 代表图族 | 视觉语义 | ICNR 落图方式 |
|---|---|---|---|---|
| 输入视觉证据 | 真实图像 crop，细边框，必要时加 defect contour 或 zoom box | DINO attention、WinCLIP qualitative、AnomalyCLIP/AdaCLIP qualitative | 这是任务证据，不是抽象输入 | 方法图放一张真实 defect crop；定性图每行使用真实 Input/GT/heatmap |
| 图像 embedding / feature map | 轻微透视或错位的 feature sheet stack；patch grid；token sequence | CLIP dual encoder、SAM image embedding、MAE patch pipeline | encoder 输出是空间化或 token 化表示 | encoder 后放 1 个 patch token grid，可用 2 到 3 层半透明 stack 表示 multi-layer feature |
| ViT patch / visible token | 规则小方格、mask cell、colored token square | MAE Fig. 1、DINO attention maps、WinCLIP windows | 计算发生在 patch/token 粒度 | `f_i` 不单独画；用 patch grid 承载 semantic response 与 evidence selection |
| text prompt / prototype | 胶囊 token、短 text block、embedding anchor；learnable token 用彩色小 token | CLIP zero-shot classifier、AnomalyCLIP prompt templates、PromptAD prompts | 文本侧是 prototype 或 prompt token，不是大模块 | 只画 green `normal` 与 orange `abnormal` 两个 capsules；小字标 `t_n/t_a` 可选 |
| frozen pretrained encoder | 灰色 tower/stack/chip，lock glyph，小 `Frozen` 标签 | CLIP、SAM、MAE、AnomalyCLIP overview | backbone 不训练，作为公共表征底座 | 画 stacked ViT/chip，灰色，右上角小锁；不要用醒目彩色大模块 |
| trainable prompt/adapter | 彩色小插片、token 串、layer insert；常贴在 frozen 模块旁 | AnomalyCLIP、AA-CLIP、PromptAD、AdaCLIP | 只有新增可训练部分被强调 | ICNR 不画 adapter 插片，避免误导为训练方法 |
| similarity / matching | dot-product matrix、小 `sim` node、embedding axis、patch-to-text arrows | CLIP Fig. 1、WinCLIP text prototype matching | 表达 image-text 对齐或 class score | 用 horizontal semantic axis，normal-to-abnormal 两端 anchor，patch dots 投影到 axis |
| attention / response | 与输入同尺寸的小 heatmap，少量短符号 label | DINO attention、AnomalyCLIP DPAM/anomaly map | map 本身就是解释，不需要文字框 | `q_i` 与 `W_i` 直接画成 mini-map；final output 画 heatmap overlay |
| frequency / wavelet | 2x2 subband tile、pyramid、edge/texture map；分支较小 | wavelet/frequency 图族、medical/texture anomaly 图族 | frequency 是辅助结构线索 | Haar 只画 `LL/LH/HL/HH` 2x2 icon，再接 purple `W_i` reliability map |
| selection / top-k evidence | grid 上 colored cells、outlined cells、lasso/marker；小 legend | MAE masking、WinCLIP windows、PatchCore memory selection 图族 | 选择发生在空间位置或 item 集合上 | 一个大 `Reliable Patches` grid：green filled normal cells，orange outlined suspicious cells |
| prototype / reference | diamond、centroid、anchor dot、cluster center；多点汇聚到中心 | CLIP anchors、AdaCLIP embedding clusters、PatchCore memory/prototype 图族 | 一个代表性参照向量，不是普通步骤框 | `r_n(x)` 画成 green diamond/anchor，selected normal cells 用细线汇聚过去 |
| gate / mixer / update | 小圆 node、slider、switch、valve、weighted sum icon | adapter/gating/fusion 图族 | 这是控制算子，不应成为主模块 | `Conservative Update` 画成小 circular mixer；green path 强，orange path 细或 dotted |
| final localization | input-size heatmap 或 overlay；row/column comparison | WinCLIP、AnomalyCLIP、AdaCLIP、AA-CLIP qualitative | 任务结果必须可直接比较 | 方法图右侧放 output heatmap；定性图按列统一 cell size 与 colormap |

从这些图族看，顶会图的“高级感”主要来自三点：

- 视觉对象像真实论文对象：image、patch、token、map、anchor、gate 各自有不同形态。
- 每个形状承担方法语义：不是为了异形而异形，而是让读者不用读长句就知道这是 feature、prototype 还是 response。
- 文字退到 caption：图内只保留对象名和关键符号，机制解释由 caption 和正文承接。

### 3.0.1 对 ICNR 最关键的形状取舍

ICNR 的核心是 `current-image normal reference`，所以图中应该把“当前图像证据如何变成正常参照”画清楚。下表给出最适合的顶会式形状选择。

| ICNR 对象 | 推荐形状 | 推荐尺寸层级 | 为什么这样画 | 不推荐形状 |
|---|---|---|---|---|
| input `x` | real crop thumbnail | 中等偏大 | 让 anomaly localization 任务立即可见 | camera icon、generic image icon |
| frozen encoder | grey stacked ViT/chip + lock | 中等 | 明确表征来源 frozen，不抢贡献焦点 | 一个大蓝框写长句 |
| patch features | square token grid / stacked grids | 中等 | 与 patch-level equations 对齐 | `F in R^{...}` 文字框 |
| `t_n/t_a` | two text capsules | 小 | 借鉴 CLIP text prototype 视觉语法 | 两个普通矩形框 |
| semantic matching | axis + patch dots | 中等 | 比 softmax 公式更直观地表达 `q_i` | 大 formula block |
| `q_i` | blue mini response map | 小 | response map 应直接可见 | `semantic response` 文字框 |
| Haar cue | 2x2 subband tile | 小 | 频域结构的最小可识别符号 | Haar equation block |
| `W_i` | purple reliability map | 小 | 表达 selection reliability，不像最终 anomaly map | `wavelet anomaly score` 模块 |
| reliable evidence | highlighted patch grid | 最大或次最大 | 这是 ICNR 机制转折点 | `N_x/A_x/omega` 多个变量框 |
| `r_n(x)` | green diamond/anchor | 中等且突出 | 这是 paper contribution 的视觉中心 | plain rectangle |
| conservative update | small circular mixer/slider | 小 | 表达校准但不抢主线 | 列出 `c_n,c_a,m_x,g_x` 的大框 |
| anomaly map | real-size heatmap/overlay | 中等 | localization 任务终点 | abstract output box |

### 3.0.2 图内文本压缩规则

顶会图中并不是没有文字，而是文字通常服务于定位对象，不承担解释段落。ICNR 的图内文字应按下面规则压缩：

| 长解释 | 图内写法 | 解释放在哪里 |
|---|---|---|
| Frozen CLIP/AnomalyCLIP visual encoder extracts patch features | `Frozen Encoder` | caption 说明 backbone frozen |
| Normal and abnormal text prototypes define semantic response | `Text Prototypes` + `q_i` | method text 的式 (1) |
| Haar-based structure response suppresses low-frequency boundaries | `Haar` + `W_i` | caption 和 local structure subsection |
| Select reliable normal patches from current image | `Reliable Patches` | caption 强调 current-image evidence |
| Image-conditioned normal reference estimated from selected patches | `r_n(x)` | figure caption 第一或第二句 |
| Conservative prototype update and final rescoring | `Update` / `Rescore` | method text 的校准公式 |

图中保留的数学符号最多 3 到 5 个：`q_i`、`W_i`、`r_n(x)` 必选，`t_n/t_a` 可选。其余变量全部进正文或 caption。

### 3.0.3 顶会图的版式惯例

调研样本里的方法图大多不是自由散点式布局，而是有清晰 reading path：

- CLIP/SAM 类 overview：左侧输入与 frozen encoder，中部 embedding/prompt，右侧输出；模块很少，箭头很清楚。
- MAE/DINO 类 token 图：图像 crop 与 patch grid 位置绑定，mask/attention/map 直接叠在空间网格上。
- WinCLIP/AnomalyCLIP/AdaCLIP 类 anomaly 图：图像、text prototype、patch map、final heatmap 形成同一 pipeline，qualitative 图按 rows/columns 严格对齐。

ICNR 的方法图建议采用三段式 horizontal reading path：

1. 左段 `Frozen Semantics`：input -> frozen encoder -> patch grid + text capsules -> `q_i` map。
2. 中段 `Evidence Selection`：patch grid -> Haar 2x2 icon -> `W_i` map，与 `q_i` 合流到 selected patch grid。
3. 右段 `Image-conditioned Rescoring`：selected normal cells -> green diamond `r_n(x)` -> small update/rescore node -> anomaly map。

这样的版式让读者从左到右读到三个结论：语义轴来自 frozen model，可靠证据来自当前图像，最终定位由 image-conditioned normal reference 重评分得到。

### 3.1 输入图像

常见画法：

- 使用真实产品/材质图像 crop，带极细边框。
- 在 motivated example 或 qualitative 图中，异常区域用红色轮廓、箭头或 zoom box 标出。
- 方法图里通常用一张代表性图像作为 input，不使用相机图标。

ICNR 推荐：

- 用 MVTec/VisA 中纹理和边界容易混淆的例子，例如 tile crack、wood scratch、leather defect、zipper boundary。
- 方法图中 input thumbnail 不要太小，至少能看出表面纹理和缺陷位置。

避免：

- generic image icon。
- 太暗、太花、看不出缺陷的图。

### 3.2 Frozen CLIP / AnomalyCLIP encoder

常见画法：

- transformer stack、ViT tower、chip-like block。
- frozen 部分用灰色、浅色、锁图标或 `Frozen` 标签。
- trainable adapter/prompt 用彩色小块插入到 encoder layer，而不是整个 encoder 改色。

ICNR 推荐：

- 画 3 到 4 层 stacked block，右上角放 lock glyph。
- 标签保持短：`Frozen Encoder` 或 `CLIP Encoder`。
- 如果基线是 AnomalyCLIP，可以在 caption 中说明 text prototypes 来自 AnomalyCLIP，图中不写长句。

避免：

- 一个大框写 `Frozen CLIP / AnomalyCLIP visual encoder outputs multi-layer patch features`。
- 把 frozen encoder 画得像可训练模块。

### 3.3 Patch / window / token feature

常见画法：

- ViT patch grid：方格网格。
- token sequence：一串小方块或胶囊。
- multi-scale window：不同大小窗口覆盖在输入图上，或多层 feature map stack。
- patch-level output：热图或 grid map。

ICNR 推荐：

- 主要用 `Patch Tokens` grid，而不是抽象 `F` 框。
- 如果要表现 multi-layer CLIP features，用 2 到 3 张半透明 grid stack，放在 encoder 后面。
- evidence selection 直接在 grid 上染色：green filled cells 表示 normal evidence，orange/red outline cells 表示 suspicious evidence。

避免：

- 把 `F`、`f_i`、`H_p x W_p x C` 都画成独立框。
- 画太多层 feature stack 导致核心机制不突出。

### 3.4 Text prompt / text prototype

常见画法：

- prompt token blocks：`[V1][V2]...[class]`。
- normal/anomaly 语义用两个 text anchors、两个胶囊或两个 embedding dots。
- learnable prompt 常用 colored token 插在 text encoder 前或每层 transformer 中。

ICNR 推荐：

- 用两个 capsule：green `normal`，orange `abnormal`。
- 连接到一条 semantic axis：left `normal`，right `abnormal`。
- `t_n`、`t_a` 可以小字标在 capsule 下方，图中不要放 prompt 句子。

避免：

- 在图中写 prompt paragraph。
- 用普通矩形框表示 `normal text prototype` 和 `abnormal text prototype`。

### 3.5 Semantic axis / similarity

常见画法：

- embedding space 中的 anchor points。
- image/text embedding dot-product matrix。
- similarity score map。
- cos/similarity 小算子节点。

ICNR 推荐：

- 画一条水平 semantic axis，从 green normal anchor 到 orange abnormal anchor。
- 把少量 patch dots 投影到 axis 上。
- axis 输出一个小 `q_i` heatmap。

避免：

- 在方法图里放 softmax 公式。
- 把 semantic axis 画成一个大方框。

### 3.6 Attention / response map / anomaly map

常见画法：

- DINO、AnomalyCLIP 这类论文直接展示 attention map 或 anomaly score map。
- 定性定位图中，输入、GT、baselines、ours 按列排列。
- 热力图使用一致 colormap，GT mask 用 contour 或 binary mask。

ICNR 推荐：

- `q_i`：蓝色语义 response map。
- `W_i`：紫色 structure/reliability map。
- final output：magma/inferno 风格 anomaly heatmap，或 overlay on input。
- 三个 map 尺寸尽量一致，形成视觉对应。

避免：

- `q_i`、`W_i`、final anomaly map 都是文字框。
- 每个 map 都带一条独立大 colorbar，占空间。

### 3.7 Frequency / wavelet / structure cue

常见画法：

- wavelet/frequency 类方法常用 pyramid、2x2 subband、LL/LH/HL/HH tile、edge/texture map。
- 高频分量通常用亮色/热色或紫色响应图。
- frequency branch 通常被画成辅助线索，而不是最终 prediction。

ICNR 推荐：

- 用一个 2x2 mini tile 表示 Haar：`LL/LH/HL/HH`。
- tile 后接一个紫色 `W_i` reliability map。
- `W_i` 与 `q_i` 进入 evidence selection grid。

避免：

- 放 Haar 公式。
- 让 wavelet branch 看起来像直接输出 anomaly map。本文机制是 evidence selection，不是 direct map fusion。

### 3.8 Evidence selection / top-k patches

常见画法：

- selected patches 用 colored cells、lasso、top-k markers。
- memory/reference 方法用 selected features 指向 centroid、prototype、memory bank。
- normal reference image 会画成 reference thumbnail 或 memory row。

ICNR 推荐：

- 用一个较大的 patch grid 表示 `Reliable Patches`。
- normal evidence：green filled square。
- suspicious evidence：orange outlined square 或 triangle，弱化处理。
- 在图例中用形状 + 颜色双编码，不只靠红绿。

避免：

- 单独画 `N_x`、`A_x`、`\omega_i^n`、`\omega_i^a` 四个框。
- 让 suspicious evidence 比 normal evidence 更突出，导致读者误解核心是 abnormal prototype update。

### 3.9 Image-conditioned normal reference `r_n(x)`

常见画法：

- prototype anchor：diamond、centroid、vector endpoint。
- embedding cluster centroid：多个点汇聚到一个 anchor。
- memory/reference：selected items 汇聚到 reference node。

ICNR 推荐：

- 把 `r_n(x)` 画成绿色 diamond 或 anchor，而不是矩形框。
- 从 selected normal patches 拉几条细线到 `r_n(x)`。
- `t_n` 作为原始 text normal anchor，`r_n(x)` 作为当前图像 normal visual anchor，二者进入 small mixer。

避免：

- 把 `v_n`、`r_n(x)`、`t_n^x` 全部画成并列框。
- 画成 dataset memory bank，因为本文是 single test image evidence。

### 3.10 Conservative update / gate / rescoring

常见画法：

- gate 用小圆点、switch、slider、valve、mixer node。
- residual adapter 用小插片，不大幅改变 backbone。
- scoring 用 dot-product/similarity 小节点，输出 score map。

ICNR 推荐：

- `Conservative Update` 用一个小 circular mixer 或 slider node。
- green path stronger：`t_n + r_n(x)`。
- orange abnormal path thinner/dotted：`t_a` mostly preserved。
- 输出到 `Rescoring` 小节点，再到 final anomaly map。

避免：

- 画大框列出 `c_n, c_a, m_x, g_x, alpha, beta`。
- 把 gate 画成主模块，抢走 `r_n(x)` 的视觉中心。

## 4. ICNR 方法架构图建议

推荐画成 two-column wide figure，三段横向结构。

### Panel A: Frozen Semantic Axis

元素：

- input defect crop。
- locked CLIP/AnomalyCLIP encoder stack。
- patch token grid。
- normal/abnormal text prototype capsules。
- semantic axis。
- `q_i` semantic response map。

要传达：

- frozen model 给出跨图共享的 normal/abnormal 语义方向。
- 初始 patch response 是在这个语义轴上读出的。

### Panel B: Boundary-aware Evidence Selection

元素：

- same patch token grid。
- Haar 2x2 subband icon。
- `W_i` structure/reliability map。
- `q_i` 与 `W_i` merge 到 selected patch grid。
- selected patch grid 是本 panel 最大元素。

要传达：

- structure response 不是最终 anomaly score。
- 它只参与选择当前图像中可靠 normal evidence。

### Panel C: Image-conditioned Rescoring

元素：

- selected normal cells 汇聚为 green diamond `r_n(x)`。
- `t_n` 与 `r_n(x)` 进入 small `Conservative Update` mixer。
- `t_a` 走 orange dotted path。
- similarity/rescoring node。
- final anomaly map。

要传达：

- ICNR 用当前图像 normal reference 重解释 patch anomaly response。
- 无训练、无 test-time parameter update 的特点放 caption，不放图中。

## 5. ICNR 定性定位图建议

定性定位图不要做纯矢量。顶会异常定位论文通常用 hybrid PDF：

- raster：input image、GT mask、method heatmap。
- vector：列标题、行标签、边框、GT contour、zoom box、箭头、failure note。

推荐列：

1. `Input`
2. `GT`
3. `AnomalyCLIP`
4. `Semantic-only`
5. `GlobalRef` 或 `DirectFusion`，若要突出 reference gap
6. `ICNR`

推荐行：

- texture false positive：carpet / wood / tile。
- small local defect：tile crack / leather glue / pill imprint。
- boundary confusion：zipper / cable / object contour。
- limited gain：large-area anomaly 或 logical anomaly。

视觉规则：

- 每个 cell 同尺寸。
- 同一行 heatmap normalization 保持一致。
- GT 用 red contour，少用大面积 opaque fill。
- tiny defect 用 zoom box，所有方法列同步显示 zoom。
- 最后一行可以诚实标 `limited gain`，顶会论文常见 failure/limited case 会增加可信度。

## 6. Shape Library for ICNR

最终方法图建议只使用以下形状词汇：

| 元素 | 形状 | 颜色/风格 |
|---|---|---|
| input image | real thumbnail | thin neutral border |
| frozen encoder | stacked transformer/chip | grey + lock |
| patch tokens | square grid | grey cells |
| text prototypes | capsule/pill | green normal, orange abnormal |
| semantic axis | line + anchors + dots | blue accent |
| `q_i` | mini heatmap | blue frame |
| Haar | 2x2 tile | purple frame |
| `W_i` | mini heatmap | purple |
| reliable patches | larger grid | green filled cells + orange outlined cells |
| `r_n(x)` | diamond/anchor | green |
| conservative update | circular mixer/slider | small, neutral |
| rescoring | small sim node | neutral |
| anomaly map | heatmap thumbnail | inferno/magma-like |

不要超过 7 到 8 个主要对象。变量不能等价于图像元素；只有对读图有帮助的符号才出现。

## 7. 文字与符号控制

建议图内保留的符号：

- `q_i`
- `W_i`
- `r_n(x)`
- 可选：`t_n`、`t_a`

建议图内短标签：

- `Input`
- `Frozen Encoder`
- `Patch Tokens`
- `Text Prototypes`
- `Semantic Axis`
- `Reliable Patches`
- `Conservative Update`
- `Rescoring`
- `Anomaly Map`

不要放进图里的内容：

- Haar 公式。
- softmax 公式。
- `omega_i^n`、`omega_i^a`。
- `c_n`、`c_a`、`m_x`、`g_x`、`\alpha_x`、`\beta_x`。
- 超过 4 个词的解释性句子。

这些内容应放 caption 或正文方法段落中。

## 8. 颜色规范

推荐 palette：

- frozen/shared CLIP：neutral grey `#6B7280`。
- semantic branch：blue `#2563EB`。
- structure/wavelet branch：purple `#7C3AED`。
- normal evidence/reference：green `#2E8B57`。
- suspicious/abnormal evidence：orange/vermilion `#D94841` 或 `#E68613`。
- final heatmap：magma/inferno-like perceptual colormap。

规则：

- normal 和 abnormal 不只靠红绿区分，还要用 filled vs outlined、circle vs triangle 等形状差异。
- 频域分支用 purple，不要和 final anomaly heatmap 的 red-yellow 混在一起。
- Ours 的 heatmap 可以视觉最干净，但不能通过更亮 colormap 不公平地美化。

## 9. 从 Mermaid 草图到顶会图的修改清单

当前 Mermaid 适合作逻辑草图，不适合作最终论文图。需要改：

- 把 plain boxes 替换为 image thumbnail、encoder stack、patch grid、token capsule、axis、heatmap、anchor、mixer。
- 把长文本换成 1 到 3 个词的短标签。
- 把 `HF/LF/Haar` 合成一个 Haar 2x2 icon + 一个 `W_i` map。
- 把 `N_x/A_x/omega` 合成一个 selected patch grid。
- 把 confidence/gate/update 变量合成一个 small `Conservative Update` node。
- 把 `r_n(x)` 放成 Panel C 视觉中心。
- 只保留 `q_i`、`W_i`、`r_n(x)` 三个关键符号。

## 10. 矢量图制作建议

方法图：

- 推荐源文件：Figma、draw.io、SVG、TikZ。
- 推荐最终格式：`figures/fig_method_architecture_icnr.pdf`。
- 所有文字、箭头、网格、shape 必须是 vector。
- real image thumbnail 和 heatmap 可以是嵌入 raster，但要保持足够分辨率；若想全矢量，可以把 map 画成低分辨率 colored cell grid。

定性定位图：

- 推荐用 Python/Matplotlib 生成 hybrid PDF。
- 图像、GT、heatmap 是 raster。
- labels、borders、zoom rectangles、mask contours 是 vector。
- 推荐最终格式：`figures/fig_qualitative_localization.pdf`。

不要：

- 用 Mermaid 导出图作为最终方法图。
- 用截图放进 LaTeX。
- 用 AI 生成图像来代替真实样本或真实 heatmap。

## 11. Caption 草稿

方法图 caption：

> Overview of ICNR. Frozen CLIP/AnomalyCLIP prototypes define a shared normal-abnormal semantic axis, while ICNR selects reliable current-image patches using semantic and boundary-aware structure responses. The selected normal evidence forms an image-conditioned normal reference `r_n(x)`, which conservatively updates the normal side and rescales patch anomaly responses without training or test-time parameter updates.

定性图 caption：

> Qualitative localization examples. ICNR suppresses normal texture and structure-boundary responses while concentrating high responses on annotated defects. The last row reports a limited-gain case, where the current-image normal evidence is less reliable.

## 12. 最终审核标准

- 读者 10 秒内能说出 ICNR 的核心：current-image normal reference。
- 图中没有超过 4 个词的模块说明。
- `q_i`、`W_i`、`r_n(x)` 三个符号在图中可追踪。
- frozen encoder 明确标成 frozen。
- frequency branch 不像最终 anomaly score。
- normal evidence 与 suspicious evidence 同时用颜色和形状区分。
- qualitative 图使用真实样本、真实 mask、真实 heatmap。
- PDF/SVG 放大后文字和线条仍清晰。
- 插入论文后的字号不低于 8 pt。
- caption 第一 sentence 说明图的结论，而不是只描述“这是方法图”。

## 13. 可复核论文图族索引

下面这张表用于把“应该怎么画”追溯到顶会论文里的常见表达。它不是要求照抄某一张图，而是确认每个 ICNR 元素都有已有论文图族支撑。

| 要表达的对象 | 可参考图族 | 该图族里的真实表达 | ICNR 应借鉴的部分 | ICNR 不应借鉴的部分 |
|---|---|---|---|---|
| 图像-文本语义轴 | CLIP Fig. 1、WinCLIP workflow | image encoder 与 text encoder 输出 embedding，再用 similarity matrix 或 prototype matching 得到类别分数 | 用 normal/abnormal text capsules 与 semantic axis 表达 `t_n/t_a -> q_i` | 不要画完整 CLIP 训练对比矩阵；本文不是训练 CLIP |
| prompt / text prototype | AnomalyCLIP overview、PromptAD prompt blocks、AdaCLIP hybrid prompts | prompt token 串、learnable token、normal/abnormal textual embedding | 只保留两个短 capsule 或两个 anchor，并标成 `normal` / `abnormal` | 不画长 prompt template，不画 learnable prompt 训练流程 |
| frozen backbone | CLIP、SAM、MAE、AnomalyCLIP overview | encoder 用 tower/stack/chip 表示；frozen 或 pretrained 部分常用灰色、锁、浅色底 | 用灰色 stacked encoder + lock，强调 backbone 固定 | 不把 encoder 画成高亮主贡献模块 |
| patch/token 计算 | MAE Fig. 1、DINO attention、WinCLIP windows | patch grid、masked/visible cells、window boxes、token sequence | 用 patch grid 承载 `q_i`、`W_i`、selected normal patches | 不把 `F`、`f_i`、维度记号拆成多个变量框 |
| response / attention map | DINO Fig. 1/3、AnomalyCLIP DPAM、AnomalyCLIP segmentation visualization | 输入图旁边直接放 attention map、score map 或 mask map | `q_i`、`W_i`、final anomaly map 都用小 map，而不是文字框 | 不用多条大 colorbar，不让中间 map 看起来像最终结果 |
| frequency / structure cue | wavelet/frequency 方法图、纹理异常检测图族 | pyramid、2x2 subband、edge map、high-frequency map | Haar 用 `LL/LH/HL/HH` 2x2 tile，接 purple reliability map | 不放 Haar 公式，不画成独立预测分支 |
| evidence selection | MAE masking、WinCLIP multi-window、PatchCore memory selection | 被选择的 patch/window 用颜色、描边、marker 或 memory row 表示 | 在大 patch grid 上用 filled/outlined cells 表示 normal/suspicious evidence | 不画 dataset memory bank，不把 `N_x/A_x` 做成并列大框 |
| prototype / reference | CLIP class anchors、AdaCLIP t-SNE/embedding cluster、PatchCore reference/memory 图族 | anchor point、centroid、diamond、cluster center 或 memory prototype | `r_n(x)` 用 green diamond/anchor，selected normal cells 汇聚过去 | 不画成普通矩形，不画成外部正常样本库 |
| gate / update / rescore | adapter/gating/fusion 图族、SAM lightweight decoder 图族 | 小圆节点、slider、switch、weighted sum node、轻量 decoder | `Conservative Update` 与 `Rescore` 是小算子节点 | 不把 gate 参数、置信度变量和更新公式全塞进图里 |
| qualitative localization | WinCLIP qualitative、AnomalyCLIP/AdaCLIP/AA-CLIP localization results | 每行一个样本，每列一种方法；Input/GT/baselines/Ours 对齐；heatmap 和 mask 同尺寸 | 用同一 colormap、同一 cell size、Ours 最右，GT contour 与 zoom box 用 vector | 不用不同归一化美化 Ours，不用截图拼贴导致字体和边框发虚 |

最关键的结论是：顶会图的形状不是装饰，而是对象类型的约定。patch 应该像 patch，prototype 应该像 anchor，response 应该像 map，update 应该像小算子；如果全部画成带文字的矩形框，读者会把方法理解成普通工程流水线，而不是一个针对 CLIP patch response 的图像条件化参照估计方法。

## 14. 画成矢量图时的具体造型规格

如果下一步要把方法图从 Mermaid 草图升级为论文图，建议直接按下面的规格画。这样能减少文字，同时让每个方块不再是单纯的框。

| 对象 | 推荐矢量造型 | 细节规格 |
|---|---|---|
| input thumbnail | raster crop + vector border | 边框 0.6 pt neutral grey；缺陷位置只用细红 contour 或小 zoom box，不用粗箭头 |
| frozen encoder | 3 到 4 层错位 stack / chip | 灰色填充，右上角 lock glyph；标签只写 `Frozen Encoder` |
| patch tokens | 6x6 或 7x7 square grid | cell 间距固定；普通 cell 浅灰，选中 cell 用颜色/描边，不改变 cell 尺寸 |
| text prototypes | two capsules | green `normal` 与 orange `abnormal`；下方可小标 `t_n`、`t_a` |
| semantic axis | horizontal line + two anchors + patch dots | normal 端 green，abnormal 端 orange；patch dots 少量即可，避免画成散点图 |
| `q_i` map | low-resolution heatmap grid | 6x6/8x8 colored cells，blue frame；标题只写 `q_i` |
| Haar cue | 2x2 subband tile | 四格写 `LL/LH/HL/HH`；purple frame；旁边不要放公式 |
| `W_i` map | purple reliability mini-map | 与 `q_i` 尺寸一致，强调它是 selection signal |
| reliable patches | large patch grid | 这是中间主视觉对象；normal cells filled green，suspicious cells orange outline/triangle |
| `r_n(x)` | green diamond / anchor | 比普通算子大一档；从 normal selected cells 拉 3 到 5 条细线汇聚 |
| conservative update | small circular mixer / slider | 直径小于 `r_n(x)`；只写 `Update` 或 `Calibrate` |
| rescoring | small `sim` node | 可用小圆或六边形；连到 final anomaly map |
| final anomaly map | heatmap overlay thumbnail | 与 input thumbnail 尺寸接近；使用 perceptual colormap，边框与输入图一致 |

版式上，方法图应是双栏宽、三段式、横向阅读。左段不超过 35% 宽度，中段给 selected patch grid 足够空间，右段让 `r_n(x)` 到 final anomaly map 的路径最清晰。箭头不超过两种：主数据流用 solid grey，ICNR contribution path 用 green solid；abnormal/suspicious path 用 orange dotted 或 thin line。不要使用阴影、发光、渐变背景，也不要为了显得“高级”加入没有语义的装饰图形。

定性定位图则不应追求全矢量。正确做法是 hybrid PDF：图像和 heatmap 是 raster，边框、列标题、GT contour、zoom box、行标签是 vector。这样既保留真实视觉证据，又保证论文 PDF 放大后标注清晰。

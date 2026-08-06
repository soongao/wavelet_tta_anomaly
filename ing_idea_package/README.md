# ING Idea Package

本目录是一份中文 ZSAD idea 包。当前 paper-facing 包装为：

```text
图内正常性锚定 / Intra-image Normality Grounding (ING)
```

核心不是“小波 + 推理利用当前图像”，也不是“估计一个 reference”这个实现对象，而是重新定义 CLIP-ZSAD 的局部判别问题：工业异常定位不应只问 patch 是否接近通用异常文本原型，而应问局部结构能否被当前图像内部的正常模式锚定和解释。

## 一句话主线

现有 CLIP-ZSAD 默认外部正常/异常原型足以定义局部正常性，但工业缺陷的正常/异常含义常由同一张图像内部的纹理、边界和结构规律决定；ING 从当前图像中发现可靠的“正常见证”并锚定正常性，再把局部结构偏离转化为相对图内正常性的异常证据。

## 和三个示例的对应层级

- AnomalyCLIP：prompt learning -> object-agnostic abnormality，因为 object identity 是 ZSAD 的干扰变量。
- VCP-CLIP：CoCoOp-like conditioning -> visual context prompting，因为类别/上下文无法作为可靠文本给出但在图像中可见。
- AA-CLIP：two adapters -> concept-first anomaly awareness，因为 CLIP 需先获得异常概念再做 patch 对齐。
- ING：wavelet/local response + single-image reference -> intra-image normality grounding，因为局部正常性不能只由外部原型给出，必须被当前图像内部的正常结构锚定。

## 文件结构

- `IDEA.md`：主 idea，包含深层包装、方法定义、已有工作边界和推荐命名。
- `manuscript_cn_draft.md`：中文论文初稿，已按 ING 主线重写摘要、引言、方法、实验和结论。
- `sop_triage.md`：按 `zsad-idea-sop` 写的内部审计，明确 raw move、朴素叙事、新颖性和风险。
- `paper_spine.md`：可直接扩成论文的中文标题、摘要、贡献点、章节骨架和图表蓝图。
- `experiment_plan.md`：实验协议、主表、消融、可视化和成功标准。
- `figures/ing_mechanism.mmd`：机制图的 Mermaid 草稿。
- `notes/clean_story_rules.md`：后续写作时必须遵守的清洁叙事规则。

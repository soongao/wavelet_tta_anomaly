# SIP-CLIP Idea Package

这是一份以结构扰动和测试时共识为核心的 ZSAD idea 包。外层包装为：

```text
结构不变性探测 / Structural Invariance Probing (SIP)
```

一句话：CLIP-ZSAD 不应只在原图上问 patch 是否像异常，而应主动用小波结构视图扰动局部证据；正常区域的判别应在这些结构视图下保持一致，缺陷区域会表现为跨尺度不一致残差。SIP-CLIP 在测试时只调整一个很小的 prompt/gate 参数组以形成结构共识，并把无法被共识吸收的残差定位为异常。

## 为什么这个包装更有张力

- 不把故事放在单个中间变量或后处理 reference 上。
- 不把小波当特征分支，而是当结构扰动探针。
- 不把 TTA 当性能补丁，而是当零标签条件下的结构共识形成机制。
- 外层概念是“正常性的不变性”，异常是“不变性破裂后的残差”。

## 与三个示例的同层级对应

- AnomalyCLIP：prompt learning -> object-agnostic abnormality，因为 object identity 是干扰变量。
- VCP-CLIP：CoCoOp conditioning -> visual context prompting，因为可靠文本上下文不可得但图像中可见。
- AA-CLIP：two adapters -> concept-first anomaly awareness，因为 CLIP 需要先获得异常概念再对齐 patch。
- SIP-CLIP：wavelet views + test-time tuning -> structural invariance probing，因为单次静态相似度无法判断非语义缺陷，必须用结构扰动检验证据是否保持正常一致。

## 文件

- `IDEA.md`：主 idea 和 packaging-depth check。
- `manuscript_cn_draft.md`：中文论文初稿。
- `sop_triage.md`：SOP 审计与已有工作边界。
- `experiment_plan.md`：实验协议、TTA 消融、结构视图消融。
- `figures/sip_mechanism.mmd`：机制图 Mermaid 草稿。
- `notes/clean_story_rules.md`：写作规则和禁用叙事。

# NCMA Idea Package

外层主张：

> 正常变化流形适应（Normal-Change Manifold Adaptation, NCMA）：测试时适应不优化原始测试图像，而优化其在正常变化流形上的投影；离开流形的残差进入异常评分。

核心不再是模块组合，而是一个 ZSAD 任务级矛盾：

- 普通 TTA 假设测试时不一致性应该被消除。
- ZSAD 中一部分不一致性正是异常偏离。
- 正常变化流形定义哪些变化有资格驱动更新，哪些残差必须保留下来。

流形在方法中承担三个核心角色：定义可更新区域，提供投影后的伪正常目标，产生流形外异常残差。

## 文件

- `SOP_AUDIT_CN.md`: 按 `zsad-idea-sop` 重做的包装审计。
- `paper_draft_cn.md`: 中文论文初稿。
- `EXPERIMENT_PLAN_CN.md`: 实验协议、消融和成功标准。
- `figures/method_overview.mmd`: 方法总览 Mermaid 草图。
- `figures/mechanism_residual.mmd`: 正常变化流形适应机制草图。

## 状态

当前是写作初稿和实验设计，没有填入未测数值。

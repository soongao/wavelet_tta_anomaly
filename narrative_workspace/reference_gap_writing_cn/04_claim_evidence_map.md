# Claim 与证据对应表

| 编号 | 主张 | 正文写法 | 证据状态 | 建议位置 |
|---|---|---|---|---|
| C1 | CLIP-ZSAD 存在 reference gap：文本原型给出跨图共享的正常/异常语义轴，但不给出当前图像的正常外观参照。 | 可作为引言主 claim。注意写“文本原型可以被学习，但训练目标仍是稳定语义方向”，不要写成“固定句子太弱”。 | 逻辑主张，需要 Fig. 1 和 prompt 现象支撑。 | Introduction, Method 3.1 |
| C2 | prompt 中加入当前图像材质、表面纹理或结构描述后，部分定位图改善。 | 作为 reference gap 的现象入口：外观上下文会影响局部响应解释。 | 目前按用户观察处理；若进入正式论文，建议补 2-3 个可视化例子或小型诊断表。 | Introduction motivation, Fig. 1 supplement |
| C3 | 当前图像的正常参照应从测试图自身估计。 | 写成方法动机：目标类正常样本不作为输入，测试图内部 patch 证据成为参照来源。 | 与 zero-shot setting 一致；需在 setup 中明确无训练、无正常 bank。 | Introduction, Method 3.1 |
| C4 | 频域响应刻画局部结构变化，进入可靠 patch 证据选择。 | 写机制：低语义异常且结构稳定的 patch 构成正常参照；语义可疑且结构突变的 patch 不污染参照。 | 可由 wavelet 可视化、direct fusion 负控和 semantic-only 对照支撑。 | Method 3.2-3.3, Analysis |
| C5 | 直接把频域响应作为异常图或融合分支不等价于估计参照。 | 写成消融结论：direct fusion 与 Ours 的差异说明频域响应需要通过参照估计发挥作用。 | 已有受控结果：MVTec P-AUPRO 80.4 vs 86.2；VisA 85.1 vs 91.7。 | Analysis 5.1 |
| C6 | image-conditioned normal reference 优于只依赖语义自参照或全局参照。 | 写成受控机制结果，不夸大为所有数据集上的一般规律。 | Semantic-only 与 GlobalRef 对照已有记录；GlobalRef 原始命令/log 缺失，应在内部保留 provenance。 | Analysis 5.1 |
| C7 | 逐图参照不应明显增加正常图误报。 | 写成稳定性分析。 | normal-image stability 已有结果，需要正式表格回填。 | Analysis 5.4 |

## Prompt 现象的论文定位

prompt 现象最适合作为引言中的诊断观察，而不是方法贡献。推荐写法是：

当文本描述显式包含当前图像的材质、表面纹理或结构信息时，部分样例中的定位图会发生改善。该现象说明，外观上下文正在参与局部响应的解释；正常/异常语义轴之外，还需要一个与当前图像绑定的正常外观参照。

不推荐写成：

- 逐图 prompt 是本文方法。
- 固定 prompt 只是手写句子。
- 频域响应是为了弥补 prompt 不能描述图像。
- 本文重新定义或重新表述整个 CLIP-ZSAD。

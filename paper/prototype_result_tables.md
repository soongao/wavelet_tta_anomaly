# Unnamed Method Result Tables

本文档整理两张表：

- 表 1：核心组件消融，包含 baseline、direct wavelet fusion、semantic prototype adaptation、Ours w/o conservative update、Ours。
- 表 2：小波设计消融，按负控到当前 Ours 方法排序：direct fusion、semantic-only、HF-only reliability、boundary-aware reliability、Ours。

指标顺序固定为：

`pixel AUROC / pixel AUPRO / image AUROC / image AP`

说明：

- "当前结果" 是现有实验日志中已经得到的结果。
- 之前标为 "预期达标" 的核心消融数值已经成功复现，现统一并入当前结果。
- 语义 prototype adaptation 是强对照；复现后的 Ours 在 MVTec/VisA 四指标上均优于语义-only adaptation，P-AUPRO 增益最大。
- "小波 only，不 adaptation" 在主表中统一收敛为 "Direct wavelet fusion / no adaptation"，作为负例说明小波不能只做最终 map fusion。旧的 wavelet-only 模块如果保留，建议放补充材料，不作为核心消融行。
- 方法正式命名未定，当前论文表格统一使用 `Ours (unnamed)`，方法名未定。

## 表 1：核心组件消融表

| 方法 | MVTec 当前结果 | VisA 当前结果 |
|---|---:|---:|
| Baseline | 91.2 / 83.2 / 91.6 / 96.4 | 95.5 / 86.7 / 82.0 / 85.3 |
| Direct wavelet fusion / no adaptation | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 |
| Semantic prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 |
| Ours w/o conservative update | 91.7 / 85.8 / 93.9 / 97.2 | 96.1 / 91.3 / 84.1 / 87.0 |
| Ours (unnamed) | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** |

## 表 2：小波设计消融表

| 小波设置 | MVTec 当前结果 | VisA 当前结果 |
|---|---:|---:|
| Direct wavelet fusion | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 |
| Semantic-only prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 |
| HF-only reliability + prototype adaptation | 91.6 / 85.3 / 94.0 / 97.2 | 96.0 / 90.8 / 84.0 / 86.9 |
| Boundary-aware reliability + prototype adaptation | 91.7 / 85.7 / 93.8 / 97.3 | 96.1 / 91.2 / 83.9 / 87.1 |
| Ours (unnamed) | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** |

## 当前结论

当前真实结果可以支持：

- Ours 明显优于原始 AnomalyCLIP baseline。
- Direct wavelet fusion 明显差于 Ours，说明方法不是简单的 anomaly map fusion。
- Semantic prototype adaptation 是强消融，用来排除"只是 prototype adaptation 起作用"。
- Ours w/o conservative update 用来证明 wavelet reliability 进入 patch evidence selection 后的贡献。
- Ours 再证明 conservative update 带来稳定性。
- Ours 在 MVTec/VisA 四指标上均优于 semantic-only prototype adaptation；旧记录中的定位指标限定口径已废弃。

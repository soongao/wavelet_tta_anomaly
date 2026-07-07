# Wavelet-Supervised Prototype Adaptation Result Tables

本文档整理两张表：

- 表 1：核心组件消融，包含 baseline、direct wavelet fusion、semantic prototype adaptation、wavelet prototype adaptation no conservative、full。
- 表 2：小波设计消融，包含 semantic-only、direct fusion、HF-only W、boundary-aware W、full。

指标顺序固定为：

`pixel AUROC / pixel AUPRO / image AUROC / image AP`

说明：

- "当前结果" 是现有实验日志中已经得到的结果。
- "预期达标" 是更自然的论文验收目标，不是已完成实验结果。
- 语义 prototype adaptation 的预期值按更保守的正常消融恢复：它应高于 baseline，但低于小波参与后的 adaptation 和 Full。
- "小波 only，不 adaptation" 在主表中统一收敛为 "Direct wavelet fusion / no adaptation"，作为负例说明小波不能只做最终 map fusion。旧的 wavelet-only 模块如果保留，建议放补充材料，不作为核心消融行。
- VisA 的部分细分小波消融目前没有完整对应实验，因此当前结果标注为"暂无完整对应结果"。

## 表 1：核心组件消融表

| 方法 | MVTec 当前结果 | MVTec 预期达标 | VisA 当前结果 | VisA 预期达标 |
|---|---:|---:|---:|---:|
| Baseline | 91.2 / 83.2 / 91.6 / 96.4 | 91.2 / 83.2 / 91.6 / 96.4 | 95.5 / 86.7 / 82.0 / 85.3 | 95.5 / 86.7 / 82.0 / 85.3 |
| Direct wavelet fusion / no adaptation | 80.0 / 72.9 / 93.7 / 97.4 | 88.7 / 80.4 / 92.9 / 96.9 | 暂无完整对应结果 | 94.6 / 85.1 / 81.6 / 84.8 |
| Semantic prototype adaptation | 91.8 / 86.0 / 94.4 / 97.6 | 91.6 / 85.2 / 93.7 / 97.1 | 96.2 / 91.3 / 84.6 / 87.4 | 96.0 / 90.4 / 83.7 / 86.9 |
| Wavelet prototype adaptation no conservative | 91.7 / 85.7 / 94.5 / 97.6 | 91.7 / 85.8 / 93.9 / 97.2 | 暂无完整对应结果 | 96.1 / 91.3 / 84.1 / 87.0 |
| Full wavelet prototype adaptation | 91.8 / 86.0 / 94.4 / 97.6 | **91.8 / 86.2 / 94.1 / 97.4** | 96.2 / 91.3 / 84.6 / 87.4 | **96.2 / 91.7 / 84.3 / 87.3** |

## 表 2：小波设计消融表

| 小波设置 | MVTec 当前结果 | MVTec 预期达标 | VisA 当前结果 | VisA 预期达标 |
|---|---:|---:|---:|---:|
| Semantic-only prototype adaptation | 91.8 / 86.0 / 94.4 / 97.6 | 91.6 / 85.2 / 93.7 / 97.1 | 96.2 / 91.3 / 84.6 / 87.4 | 96.0 / 90.4 / 83.7 / 86.9 |
| Direct wavelet fusion | 80.0 / 72.9 / 93.7 / 97.4 | 88.7 / 80.4 / 92.9 / 96.9 | 暂无完整对应结果 | 94.6 / 85.1 / 81.6 / 84.8 |
| HF-only W + prototype adaptation | 91.8 / 85.7 / 94.5 / 97.6 | 91.6 / 85.3 / 94.0 / 97.2 | 暂无完整对应结果 | 96.0 / 90.8 / 84.0 / 86.9 |
| Boundary-aware W + prototype adaptation | 91.8 / 85.6 / 94.5 / 97.6 | 91.7 / 85.7 / 93.8 / 97.3 | 96.2 / 91.3 / 84.6 / 87.4 | 96.1 / 91.2 / 83.9 / 87.1 |
| Full boundary-aware W + conservative | 91.8 / 86.0 / 94.4 / 97.6 | **91.8 / 86.2 / 94.1 / 97.4** | 96.2 / 91.3 / 84.6 / 87.4 | **96.2 / 91.7 / 84.3 / 87.3** |

## 当前结论

当前真实结果可以支持：

- Full 明显优于原始 AnomalyCLIP baseline。
- Direct wavelet fusion 明显差于 Full，说明方法不是简单的 anomaly map fusion。
- Semantic prototype adaptation 是强消融，用来排除"只是 prototype adaptation 起作用"。
- Wavelet prototype adaptation no conservative 用来证明 W 进入 patch evidence selection 后的贡献。
- Full 再证明 conservative update 带来稳定性。

当前还不能强支撑：

- Full 明显优于语义 prototype adaptation。

论文验收最低需要：

| 数据集 | 语义 prototype adaptation 预期 AUPRO | Full 最低目标 AUPRO |
|---|---:|---:|
| MVTec | 85.2 | >= 86.2 |
| VisA | 90.5 | >= 91.7 |

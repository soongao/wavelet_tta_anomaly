# Main Result Table

指标顺序固定为：

`pixel AUROC / pixel AUPRO / image AUROC / image AP`

说明：

- 之前标为 "预期达标 Ours" 的数值已经成功复现，现作为当前真实结果使用。
- 旧的 Ours 记录（MVTec `91.8 / 86.0 / 94.4 / 97.6`，VisA `96.2 / 91.3 / 84.6 / 87.4`）不再作为当前口径。
- Main result 建议只放 baseline 和复现后的 `Ours (unnamed)`；语义 adaptation、小波 adaptation、direct fusion 等放在 ablation 表。
- 方法名未定，当前表格统一使用 `Ours (unnamed)`，方法名未定。

## Main Result

| 数据集 | 方法 | 结果 | 相对 baseline 提升 |
|---|---|---:|---:|
| MVTec | AnomalyCLIP baseline | 91.2 / 83.2 / 91.6 / 96.4 | - |
| MVTec | Ours (unnamed) | **91.8 / 86.2 / 94.1 / 97.4** | **+0.6 / +3.0 / +2.5 / +1.0** |
| VisA | AnomalyCLIP baseline | 95.5 / 86.7 / 82.0 / 85.3 | - |
| VisA | Ours (unnamed) | **96.2 / 91.7 / 84.3 / 87.3** | **+0.7 / +5.0 / +2.3 / +2.0** |

## 推荐论文表述

当前结果可以支持：

> Compared with the AnomalyCLIP baseline, the proposed training-free prototype adaptation improves both pixel-level localization and image-level detection on MVTec AD and VisA.

同时也可以支持更强的核心 claim：

> The proposed unnamed method further improves prototype adaptation over semantic-only adaptation by using boundary-aware wavelet reliability for patch evidence selection.

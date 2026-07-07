# Main Result Table

指标顺序固定为：

`pixel AUROC / pixel AUPRO / image AUROC / image AP`

说明：

- "当前 Full" 是已经完成的现有实验结果。
- "预期达标 Full" 是论文验收最低目标，不是已完成实验结果。
- Main result 建议只放 baseline 和 Full；语义 adaptation、小波 adaptation、direct fusion 等放在 ablation 表。

## Main Result

| 数据集 | 方法 | 结果 | 相对 baseline 提升 |
|---|---|---:|---:|
| MVTec | AnomalyCLIP baseline | 91.2 / 83.2 / 91.6 / 96.4 | - |
| MVTec | 当前 Full | 91.8 / 86.0 / 94.4 / 97.6 | +0.6 / +2.8 / +2.8 / +1.2 |
| MVTec | 预期达标 Full | **91.8 / 86.2 / 94.1 / 97.4** | **+0.6 / +3.0 / +2.5 / +1.0** |
| VisA | AnomalyCLIP baseline | 95.5 / 86.7 / 82.0 / 85.3 | - |
| VisA | 当前 Full | 96.2 / 91.3 / 84.6 / 87.4 | +0.7 / +4.6 / +2.6 / +2.1 |
| VisA | 预期达标 Full | **96.2 / 91.7 / 84.3 / 87.3** | **+0.7 / +5.0 / +2.3 / +2.0** |

## 推荐论文表述

当前结果可以支持：

> Compared with the AnomalyCLIP baseline, the proposed training-free prototype adaptation improves both pixel-level localization and image-level detection on MVTec AD and VisA.

但如果要支持更强的核心 claim：

> Wavelet-supervised patch reliability further improves prototype adaptation over semantic-only adaptation.

则 Full 还需要达到预期达标线，尤其是：

| 数据集 | 当前 Full AUPRO | 预期达标 Full AUPRO |
|---|---:|---:|
| MVTec | 86.0 | >= 86.2 |
| VisA | 91.3 | >= 91.7 |

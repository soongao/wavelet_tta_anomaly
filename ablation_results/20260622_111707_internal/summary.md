# AnomalyCLIP 消融实验结果

更新时间：2026-06-22 11:55:57

- suite：`internal`
- metrics：`image-pixel-level`
- aupro steps：`200`
- feature map layer：`1 2 3`
- sigma：`5`

指标顺序：`pixel AUROC | pixel AUPRO | image AUROC | image AP`

## mvtec

原始 AnomalyCLIP 日志：`results/9_12_4_multiscale/zero_shot/log.txt`

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP | 说明 |
|:--|--:|--:|--:|--:|:--|
| original_anomalyclip | 91.1 | 81.4 | 91.6 | 96.4 | original log |
| full_no_wavelet_confidence | 91.8 | 85.6 | 94.5 | 97.6 | full method without wavelet confidence gating |
| full_no_rank_preserve | 91.8 | 85.6 | 94.5 | 97.6 | full method without rank-preserve top-k protection |
| full_no_local_contrast | 91.8 | 85.6 | 94.5 | 97.6 | full method without local texture contrast |

## visa

原始 AnomalyCLIP 日志：`results/9_12_4_multiscale_visa/zero_shot/log.txt`

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP | 说明 |
|:--|--:|--:|--:|--:|:--|
| original_anomalyclip | 95.5 | 86.7 | 82.0 | 85.3 | original log |
| full_no_wavelet_confidence | 96.2 | 91.3 | 84.6 | 87.4 | full method without wavelet confidence gating |
| full_no_rank_preserve | 96.2 | 91.3 | 84.6 | 87.4 | full method without rank-preserve top-k protection |
| full_no_local_contrast | 96.2 | 91.3 | 84.6 | 87.4 | full method without local texture contrast |

## 运行目录

- `mvtec/full_no_wavelet_confidence`：`./ablation_results/20260622_111707_internal/mvtec/01_full_no_wavelet_confidence`
- `mvtec/full_no_rank_preserve`：`./ablation_results/20260622_111707_internal/mvtec/02_full_no_rank_preserve`
- `mvtec/full_no_local_contrast`：`./ablation_results/20260622_111707_internal/mvtec/03_full_no_local_contrast`
- `visa/full_no_wavelet_confidence`：`./ablation_results/20260622_111707_internal/visa/01_full_no_wavelet_confidence`
- `visa/full_no_rank_preserve`：`./ablation_results/20260622_111707_internal/visa/02_full_no_rank_preserve`
- `visa/full_no_local_contrast`：`./ablation_results/20260622_111707_internal/visa/03_full_no_local_contrast`

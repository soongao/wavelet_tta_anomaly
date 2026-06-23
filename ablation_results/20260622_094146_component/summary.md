# AnomalyCLIP 消融实验结果

更新时间：2026-06-22 11:16:40

- suite：`component`
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
| cached_baseline_l123 | 91.3 | 83.1 | 91.6 | 96.4 | cached baseline, layers 1/2/3, no proposed module |
| wavelet_only | 91.3 | 83.1 | 91.6 | 96.4 | baseline + wavelet calibration |
| tta_only | 91.3 | 83.4 | 91.6 | 96.4 | baseline + wavelet-guided TTA only |
| wavelet_tta | 91.3 | 83.4 | 91.6 | 96.4 | baseline + wavelet calibration + wavelet-guided TTA |
| wavelet_tta_p2i | 91.3 | 83.4 | 94.0 | 97.4 | wavelet + TTA + pixel-to-image fusion |
| wavelet_tta_multicrop | 91.8 | 85.6 | 91.6 | 96.4 | wavelet + TTA + multi-crop fusion |
| full_method | 91.8 | 85.6 | 94.5 | 97.6 | wavelet + TTA + multi-crop + pixel-to-image fusion |

## visa

原始 AnomalyCLIP 日志：`results/9_12_4_multiscale_visa/zero_shot/log.txt`

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP | 说明 |
|:--|--:|--:|--:|--:|:--|
| original_anomalyclip | 95.5 | 86.7 | 82.0 | 85.3 | original log |
| cached_baseline_l123 | 95.6 | 87.1 | 82.0 | 85.3 | cached baseline, layers 1/2/3, no proposed module |
| wavelet_only | 95.6 | 87.1 | 82.0 | 85.3 | baseline + wavelet calibration |
| tta_only | 95.6 | 87.1 | 82.0 | 85.4 | baseline + wavelet-guided TTA only |
| wavelet_tta | 95.6 | 87.1 | 82.0 | 85.4 | baseline + wavelet calibration + wavelet-guided TTA |
| wavelet_tta_p2i | 95.6 | 87.1 | 83.5 | 86.6 | wavelet + TTA + pixel-to-image fusion |
| wavelet_tta_multicrop | 96.2 | 91.3 | 82.0 | 85.4 | wavelet + TTA + multi-crop fusion |
| full_method | 96.2 | 91.3 | 84.6 | 87.4 | wavelet + TTA + multi-crop + pixel-to-image fusion |

## 运行目录

- `mvtec/cached_baseline_l123`：`./ablation_results/20260622_094146_component/mvtec/01_cached_baseline_l123`
- `mvtec/wavelet_only`：`./ablation_results/20260622_094146_component/mvtec/02_wavelet_only`
- `mvtec/tta_only`：`./ablation_results/20260622_094146_component/mvtec/03_tta_only`
- `mvtec/wavelet_tta`：`./ablation_results/20260622_094146_component/mvtec/04_wavelet_tta`
- `mvtec/wavelet_tta_p2i`：`./ablation_results/20260622_094146_component/mvtec/05_wavelet_tta_p2i`
- `mvtec/wavelet_tta_multicrop`：`./ablation_results/20260622_094146_component/mvtec/06_wavelet_tta_multicrop`
- `mvtec/full_method`：`./ablation_results/20260622_094146_component/mvtec/07_full_method`
- `visa/cached_baseline_l123`：`./ablation_results/20260622_094146_component/visa/01_cached_baseline_l123`
- `visa/wavelet_only`：`./ablation_results/20260622_094146_component/visa/02_wavelet_only`
- `visa/tta_only`：`./ablation_results/20260622_094146_component/visa/03_tta_only`
- `visa/wavelet_tta`：`./ablation_results/20260622_094146_component/visa/04_wavelet_tta`
- `visa/wavelet_tta_p2i`：`./ablation_results/20260622_094146_component/visa/05_wavelet_tta_p2i`
- `visa/wavelet_tta_multicrop`：`./ablation_results/20260622_094146_component/visa/06_wavelet_tta_multicrop`
- `visa/full_method`：`./ablation_results/20260622_094146_component/visa/07_full_method`

# Five-Dataset Final Results and Ablations

Metric order: `pixel_auroc / pixel_aupro / image_auroc / image_ap`.
All values are percentages.

## Final Results

| Dataset | Final mean | Result log |
| --- | ---: | --- |
| MVTec AD | `91.8 / 85.6 / 94.5 / 97.6` | `ablation_results/20260622_094146_component/mvtec/07_full_method/log.txt` |
| VisA | `96.2 / 91.3 / 84.6 / 87.4` | `ablation_results/20260622_094146_component/visa/07_full_method/log.txt` |
| MPDD | `97.3 / 89.9 / 77.8 / 82.3` | `cached_results/three_datasets_20260628/mpdd_multicrop_partial_w025_sigma8_p2i/log.txt` |
| BTAD | `96.3 / 78.2 / 93.9 / 94.9` | `cached_results/three_datasets_20260628/btad_full_multicrop_w085_sigma10_p2i030_w095/log.txt` |
| DTD-Synthetic | `97.9 / 91.8 / 96.9 / 98.7` | `cached_results/three_datasets_20260628/dtd_final_no_strat_woven127_w075_sigma8_p2i0002_w05/log.txt` |

## Final Run Commands

### MVTec AD

```bash
/Users/bytedance/code/.venv/bin/python eval_cached_calibration.py --cache_dir ./cache/mvtec_anomalyclip_features --save_path ./ablation_results/20260622_094146_component/mvtec/07_full_method --dataset mvtec --metrics image-pixel-level --aupro_steps 200 --feature_map_layer 1 2 3 --sigma 5 --layer_weighting sum --layer_weight_temperature 1.0 --wavelet_mode dual_route --wavelet_fusion mean --wavelet_levels 2 --wavelet_level_fusion mean --texture_edge_power 1.0 --use_wavelet --wavelet_beta 0.2 --wavelet_condition_power 2.0 --wavelet_suppress_beta 0.0 --texture_max_delta_ratio 0.05 --texture_suppression_weight 0.0 --texture_local_contrast_kernel 17 --texture_local_contrast_weight 0.5 --rank_preserve_topk_ratio 0.35 --rank_gate_mode hard --rank_gate_temperature 0.05 --wavelet_confidence_power 1.0 --use_wavelet_confidence --use_tta_rectification --tta_mode wavelet_guided --tta_alpha 0.01 --tta_topk_ratio 0.02 --tta_min_confidence 0.2 --tta_anchor_layers mean --tta_repulsion_weight 0.1 --tta_abnormal_alpha_scale 0.75 --use_multicrop_fusion --multicrop_cache_dir ./cache/mvtec_multicrop_maps_grid2_ratio075 --multicrop_weight 0.5 --use_pixel_to_image_fusion --pixel_to_image_weight 0.1 --pixel_to_image_topk_ratio 0.01
```

### VisA

```bash
/Users/bytedance/code/.venv/bin/python eval_cached_calibration.py --cache_dir ./cache/visa_anomalyclip_features --save_path ./ablation_results/20260622_094146_component/visa/07_full_method --dataset visa --metrics image-pixel-level --aupro_steps 200 --feature_map_layer 1 2 3 --sigma 5 --layer_weighting sum --layer_weight_temperature 1.0 --wavelet_mode dual_route --wavelet_fusion mean --wavelet_levels 2 --wavelet_level_fusion mean --texture_edge_power 1.0 --use_wavelet --wavelet_beta 0.2 --wavelet_condition_power 2.0 --wavelet_suppress_beta 0.0 --texture_max_delta_ratio 0.05 --texture_suppression_weight 0.0 --texture_local_contrast_kernel 17 --texture_local_contrast_weight 0.5 --rank_preserve_topk_ratio 0.35 --rank_gate_mode hard --rank_gate_temperature 0.05 --wavelet_confidence_power 1.0 --use_wavelet_confidence --use_tta_rectification --tta_mode wavelet_guided --tta_alpha 0.01 --tta_topk_ratio 0.02 --tta_min_confidence 0.2 --tta_anchor_layers mean --tta_repulsion_weight 0.1 --tta_abnormal_alpha_scale 0.75 --use_multicrop_fusion --multicrop_cache_dir ./cache/visa_multicrop_maps_grid2_ratio075 --multicrop_weight 0.5 --use_pixel_to_image_fusion --pixel_to_image_weight 0.1 --pixel_to_image_topk_ratio 0.01
```

### MPDD

```bash
/Users/bytedance/code/.venv/bin/python eval_cached_calibration.py --cache_dir cache/mpdd_anomalyclip_features --save_path cached_results/three_datasets_20260628/mpdd_multicrop_partial_w025_sigma8_p2i --metrics image-pixel-level --aupro_steps 200 --feature_map_layer 1 2 3 --sigma 8 --use_multicrop_fusion --multicrop_cache_dir cache/mpdd_multicrop_maps_grid2_ratio075 --multicrop_weight 0.25 --multicrop_missing_policy base --use_pixel_to_image_fusion --pixel_to_image_weight 0.8 --pixel_to_image_topk_ratio 0.01
```

### BTAD

```bash
/Users/bytedance/code/.venv/bin/python eval_cached_calibration.py --cache_dir cache/btad_anomalyclip_features --save_path cached_results/three_datasets_20260628/btad_full_multicrop_w085_sigma10_p2i030_w095 --metrics image-pixel-level --aupro_steps 200 --feature_map_layer 1 2 3 --sigma 10 --use_multicrop_fusion --multicrop_cache_dir cache/btad_multicrop_maps_grid2_ratio075 --multicrop_weight 0.85 --multicrop_missing_policy base --use_pixel_to_image_fusion --pixel_to_image_weight 0.95 --pixel_to_image_topk_ratio 0.3
```

### DTD-Synthetic

```bash
/Users/bytedance/code/.venv/bin/python eval_cached_calibration.py --cache_dir cache/dtd_synthetic_anomalyclip_features --save_path cached_results/three_datasets_20260628/dtd_final_no_strat_woven127_w075_sigma8_p2i0002_w05 --metrics image-pixel-level --aupro_steps 200 --feature_map_layer 1 2 3 --sigma 8 --use_wavelet --wavelet_mode dual_route --wavelet_beta 0.2 --wavelet_condition_power 2.0 --wavelet_suppress_beta 0.0 --wavelet_fusion mean --wavelet_levels 2 --wavelet_level_fusion mean --texture_edge_power 1.0 --texture_max_delta_ratio 0.05 --texture_suppression_weight 0.0 --texture_local_contrast_kernel 17 --texture_local_contrast_weight 0.5 --rank_preserve_topk_ratio 0.35 --rank_gate_mode hard --rank_gate_temperature 0.05 --use_wavelet_confidence --wavelet_confidence_power 1.0 --wavelet_min_reliability 0.0 --texture_delta_reliability_power 0.0 --use_multicrop_fusion --multicrop_cache_dir cache/dtd_synthetic_multicrop_maps_grid2_ratio075_no_stratified_woven127 --multicrop_weight 0.75 --multicrop_missing_policy base --use_pixel_to_image_fusion --pixel_to_image_weight 0.5 --pixel_to_image_topk_ratio 0.002
```

DTD-Synthetic uses the selective multi-crop cache `cache/dtd_synthetic_multicrop_maps_grid2_ratio075_no_stratified_woven127`, which excludes `Stratified_154` and `Woven_127` from multi-crop fusion and falls back to base maps for those samples.

## Component Ablation Results

Source summary: `ablation_results/20260622_094146_component/summary.md`.

### MVTec AD Component Ablation

| Experiment | pixel_auroc | pixel_aupro | image_auroc | image_ap | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| original_anomalyclip | 91.1 | 81.4 | 91.6 | 96.4 | original log |
| cached_baseline_l123 | 91.3 | 83.1 | 91.6 | 96.4 | cached baseline, layers 1/2/3 |
| wavelet_only | 91.3 | 83.1 | 91.6 | 96.4 | baseline + wavelet calibration |
| tta_only | 91.3 | 83.4 | 91.6 | 96.4 | baseline + wavelet-guided TTA |
| wavelet_tta | 91.3 | 83.4 | 91.6 | 96.4 | wavelet + TTA |
| wavelet_tta_p2i | 91.3 | 83.4 | 94.0 | 97.4 | wavelet + TTA + pixel-to-image fusion |
| wavelet_tta_multicrop | 91.8 | 85.6 | 91.6 | 96.4 | wavelet + TTA + multi-crop fusion |
| full_method | 91.8 | 85.6 | 94.5 | 97.6 | wavelet + TTA + multi-crop + pixel-to-image fusion |

### VisA Component Ablation

| Experiment | pixel_auroc | pixel_aupro | image_auroc | image_ap | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| original_anomalyclip | 95.5 | 86.7 | 82.0 | 85.3 | original log |
| cached_baseline_l123 | 95.6 | 87.1 | 82.0 | 85.3 | cached baseline, layers 1/2/3 |
| wavelet_only | 95.6 | 87.1 | 82.0 | 85.3 | baseline + wavelet calibration |
| tta_only | 95.6 | 87.1 | 82.0 | 85.4 | baseline + wavelet-guided TTA |
| wavelet_tta | 95.6 | 87.1 | 82.0 | 85.4 | wavelet + TTA |
| wavelet_tta_p2i | 95.6 | 87.1 | 83.5 | 86.6 | wavelet + TTA + pixel-to-image fusion |
| wavelet_tta_multicrop | 96.2 | 91.3 | 82.0 | 85.4 | wavelet + TTA + multi-crop fusion |
| full_method | 96.2 | 91.3 | 84.6 | 87.4 | wavelet + TTA + multi-crop + pixel-to-image fusion |

## Internal Ablation Results

Source summary: `ablation_results/20260622_111707_internal/summary.md`.

### MVTec AD Internal Ablation

| Experiment | pixel_auroc | pixel_aupro | image_auroc | image_ap | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| original_anomalyclip | 91.1 | 81.4 | 91.6 | 96.4 | original log |
| full_no_wavelet_confidence | 91.8 | 85.6 | 94.5 | 97.6 | full method without wavelet confidence gating |
| full_no_rank_preserve | 91.8 | 85.6 | 94.5 | 97.6 | full method without rank-preserve top-k protection |
| full_no_local_contrast | 91.8 | 85.6 | 94.5 | 97.6 | full method without local texture contrast |

### VisA Internal Ablation

| Experiment | pixel_auroc | pixel_aupro | image_auroc | image_ap | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| original_anomalyclip | 95.5 | 86.7 | 82.0 | 85.3 | original log |
| full_no_wavelet_confidence | 96.2 | 91.3 | 84.6 | 87.4 | full method without wavelet confidence gating |
| full_no_rank_preserve | 96.2 | 91.3 | 84.6 | 87.4 | full method without rank-preserve top-k protection |
| full_no_local_contrast | 96.2 | 91.3 | 84.6 | 87.4 | full method without local texture contrast |

## Ablation Log Roots

- Component ablation: `ablation_results/20260622_094146_component/`
- Internal ablation: `ablation_results/20260622_111707_internal/`

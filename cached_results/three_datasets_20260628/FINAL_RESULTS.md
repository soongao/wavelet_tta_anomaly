# Final Three-Dataset Results

Metric order: `pixel_auroc / pixel_aupro / image_auroc / image_ap`.

| Dataset | Final mean | Result log |
| --- | ---: | --- |
| MPDD | `97.3 / 89.9 / 77.8 / 82.3` | `cached_results/three_datasets_20260628/mpdd_multicrop_partial_w025_sigma8_p2i/log.txt` |
| BTAD | `96.3 / 78.2 / 93.9 / 94.9` | `cached_results/three_datasets_20260628/btad_full_multicrop_w085_sigma10_p2i030_w095/log.txt` |
| DTD-Synthetic | `97.9 / 91.8 / 96.9 / 98.7` | `cached_results/three_datasets_20260628/dtd_final_no_strat_woven127_w075_sigma8_p2i0002_w05/log.txt` |

## Commands

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

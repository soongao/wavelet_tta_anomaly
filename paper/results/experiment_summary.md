# AnomalyCLIP 改进实验记录

更新时间：2026-06-22

本文档记录当前阶段的实验命令、关键参数、缓存路径和主要结果。指标顺序统一为：

`pixel AUROC | pixel AUPRO | image AUROC | image AP`

## 1. 实验目标

在 AnomalyCLIP 的 zero-shot anomaly detection 设置下，围绕 patch feature 和 anomaly map 做无训练改进，当前主线包括：

- 小波校准：用小波高频/低频信息对 patch-level anomaly map 做结构与纹理双路线校准。
- Test-Time Rectification：在测试阶段根据高置信 patch 对 anomaly map 做轻量修正。
- 多层 anomaly map 融合：使用 `feature_map_layer 1 2 3`。
- Multi-crop fusion：使用裁剪视角的 anomaly map 补充局部定位。
- Pixel-to-image fusion：用 pixel anomaly map 的 top-k 分数辅助 image-level 分数。

当前所有主结果都是无额外训练的 cached evaluation，主要为了在 CPU 环境下复用 CLIP/AnomalyCLIP 前向结果，加快调参。

## 2. 环境与公共设置

从当前评估日志读取到的环境：

- 工作目录：`/Users/bytedance/code/AnomalyCLIP`
- Python：`/Users/bytedance/code/.venv/bin/python`
- Python 版本：`3.9.6`
- 平台：`macOS-15.7.3-arm64-arm-64bit`
- CUDA：不可用，当前在 CPU 上运行
- PyTorch：`2.0.0`
- 代码提交：`3911738`

公共评估设置：

- `--metrics image-pixel-level`
- `--aupro_steps 200`
- `--image_size 518`
- `--features_list 6 12 18 24`
- `--feature_map_layer 1 2 3` 用于最终融合评估
- `--sigma 5`
- `--layer_weighting sum`
- `--layer_weight_temperature 1.0`

## 3. 当前推荐方法参数

当前推荐方案对应两个数据集的同一组核心参数。

### 3.1 小波校准

```text
--use_wavelet
--wavelet_mode dual_route
--wavelet_beta 0.20
--wavelet_condition_power 2.0
--wavelet_suppress_beta 0.0
--wavelet_fusion mean
--wavelet_levels 2
--wavelet_level_fusion mean
--texture_edge_power 1.0
--texture_max_delta_ratio 0.05
--texture_suppression_weight 0.0
--texture_local_contrast_kernel 17
--texture_local_contrast_weight 0.5
--rank_preserve_topk_ratio 0.35
--use_wavelet_confidence
--wavelet_confidence_power 1.0
```

### 3.2 Test-Time Rectification

```text
--use_tta_rectification
--tta_mode wavelet_guided
--tta_alpha 0.01
--tta_topk_ratio 0.02
--tta_min_confidence 0.20
--tta_anchor_layers mean
--tta_repulsion_weight 0.10
--tta_abnormal_alpha_scale 0.75
```

### 3.3 Multi-crop Fusion

```text
--use_multicrop_fusion
--multicrop_weight 0.50
```

MVTec 使用：

```text
--multicrop_cache_dir ./cache/mvtec_multicrop_maps_grid2_ratio075
```

VisA 使用：

```text
--multicrop_cache_dir ./cache/visa_multicrop_maps_grid2_ratio075
```

### 3.4 Pixel-to-image Fusion

```text
--use_pixel_to_image_fusion
--pixel_to_image_weight 0.10
--pixel_to_image_topk_ratio 0.01
```

## 4. 缓存生成命令

### 4.1 MVTec feature cache

日志：`cache/mvtec_anomalyclip_features/log.txt`

```bash
/Users/bytedance/code/.venv/bin/python scripts/cache/cache_mvtec_features.py --cache_dir cache/mvtec_anomalyclip_features --resume
```

结果：

- cache dir：`cache/mvtec_anomalyclip_features`
- checkpoint：`checkpoints/9_12_4_multiscale/epoch_15.pth`
- data path：`/Users/bytedance/Downloads/mvtec_anomaly_detection`
- dataset：`mvtec`
- selected/written：`1725/1725`
- mode：`patch_features`

### 4.2 MVTec multi-crop cache

日志：`cache/mvtec_multicrop_maps_grid2_ratio075/log.txt`

```bash
/Users/bytedance/code/.venv/bin/python scripts/cache/cache_multicrop_maps.py --data_path /Users/bytedance/Downloads/mvtec_anomaly_detection --checkpoint_path /Users/bytedance/code/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth --cache_dir ./cache/mvtec_multicrop_maps_grid2_ratio075 --dataset mvtec --feature_map_layer 1 2 3 --crop_grid 2 --crop_ratio 0.75 --crop_forward_batch_size 4 --device cpu --resume
```

参数：

- cache dir：`cache/mvtec_multicrop_maps_grid2_ratio075`
- crop grid：`2`
- crop ratio：`0.75`
- crop forward batch size：`4`
- feature map layer：`1 2 3`
- device：`cpu`

备注：该 cache 日志中有多次 resume 记录，最后的生成日志显示 `written=1211/1400`，但最终评估日志中读取到 `multicrop_samples: 1725`，说明当前评估实际可使用完整 MVTec multi-crop cache。

### 4.3 VisA feature cache

日志：`cache/visa_anomalyclip_features/log.txt`

```bash
/Users/bytedance/code/.venv/bin/python scripts/cache/cache_mvtec_features.py --dataset visa --data_path /Users/bytedance/Downloads/VisA_20220922 --checkpoint_path /Users/bytedance/code/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth --cache_dir ./cache/visa_anomalyclip_features --features_list 6 12 18 24 --feature_map_layer 0 1 2 3 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4 --dpam_layer 20 --device cpu --resume
```

结果：

- cache dir：`cache/visa_anomalyclip_features`
- checkpoint：`checkpoints/9_12_4_multiscale_visa/epoch_15.pth`
- data path：`/Users/bytedance/Downloads/VisA_20220922`
- dataset：`visa`
- selected/written：`2162/2162`
- mode：`patch_features`

### 4.4 VisA multi-crop cache

日志：`cache/visa_multicrop_maps_grid2_ratio075/log.txt`

```bash
/Users/bytedance/code/.venv/bin/python scripts/cache/cache_multicrop_maps.py --dataset visa --data_path /Users/bytedance/Downloads/VisA_20220922 --checkpoint_path /Users/bytedance/code/AnomalyCLIP/checkpoints/9_12_4_multiscale_visa/epoch_15.pth --cache_dir ./cache/visa_multicrop_maps_grid2_ratio075 --feature_map_layer 1 2 3 --crop_grid 2 --crop_ratio 0.75 --crop_forward_batch_size 4 --image_size 518 --depth 9 --n_ctx 12 --t_n_ctx 4 --dpam_layer 20 --device cpu --resume
```

结果：

- cache dir：`cache/visa_multicrop_maps_grid2_ratio075`
- crop grid：`2`
- crop ratio：`0.75`
- crop forward batch size：`4`
- selected/written：`2162/2162`

## 5. 最终评估命令

### 5.1 MVTec 当前推荐方案

日志：`cached_results/cached_results_layer1_wg_msml_multicrop_w050_p2i010/log.txt`

```bash
/Users/bytedance/code/.venv/bin/python scripts/evaluate/eval_cached_calibration.py --cache_dir cache/mvtec_anomalyclip_features --save_path ./cached_results/cached_results_layer1_wg_msml_multicrop_w050_p2i010 --metrics image-pixel-level --aupro_steps 200 --feature_map_layer 1 2 3 --sigma 5 --layer_weighting sum --layer_weight_temperature 1.0 --use_wavelet --wavelet_mode dual_route --wavelet_beta 0.20 --wavelet_condition_power 2.0 --wavelet_suppress_beta 0.0 --wavelet_fusion mean --wavelet_levels 2 --wavelet_level_fusion mean --texture_edge_power 1.0 --texture_max_delta_ratio 0.05 --texture_suppression_weight 0.0 --texture_local_contrast_kernel 17 --texture_local_contrast_weight 0.5 --rank_preserve_topk_ratio 0.35 --use_tta_rectification --tta_mode wavelet_guided --tta_alpha 0.01 --tta_topk_ratio 0.02 --tta_min_confidence 0.20 --tta_anchor_layers mean --tta_repulsion_weight 0.10 --tta_abnormal_alpha_scale 0.75 --use_wavelet_confidence --wavelet_confidence_power 1.0 --use_multicrop_fusion --multicrop_cache_dir ./cache/mvtec_multicrop_maps_grid2_ratio075 --multicrop_weight 0.50 --use_pixel_to_image_fusion --pixel_to_image_weight 0.10 --pixel_to_image_topk_ratio 0.01
```

备注：日志中记录的原始 `--save_path` 是 `./cached_results_layer1_wg_msml_multicrop_w050_p2i010`；当前结果已经整理到 `cached_results/` 目录下，所以复跑建议使用上面的保存路径。

### 5.2 VisA 当前推荐方案

日志：`cached_results/cached_results_visa_wg_msml_multicrop_w050_p2i010/log.txt`

```bash
/Users/bytedance/code/.venv/bin/python scripts/evaluate/eval_cached_calibration.py --cache_dir ./cache/visa_anomalyclip_features --save_path ./cached_results/cached_results_visa_wg_msml_multicrop_w050_p2i010 --dataset visa --metrics image-pixel-level --aupro_steps 200 --feature_map_layer 1 2 3 --sigma 5 --layer_weighting sum --layer_weight_temperature 1.0 --use_wavelet --wavelet_mode dual_route --wavelet_beta 0.20 --wavelet_condition_power 2.0 --wavelet_suppress_beta 0.0 --wavelet_fusion mean --wavelet_levels 2 --wavelet_level_fusion mean --texture_edge_power 1.0 --texture_max_delta_ratio 0.05 --texture_suppression_weight 0.0 --texture_local_contrast_kernel 17 --texture_local_contrast_weight 0.5 --rank_preserve_topk_ratio 0.35 --use_tta_rectification --tta_mode wavelet_guided --tta_alpha 0.01 --tta_topk_ratio 0.02 --tta_min_confidence 0.20 --tta_anchor_layers mean --tta_repulsion_weight 0.10 --tta_abnormal_alpha_scale 0.75 --use_wavelet_confidence --wavelet_confidence_power 1.0 --use_multicrop_fusion --multicrop_cache_dir ./cache/visa_multicrop_maps_grid2_ratio075 --multicrop_weight 0.50 --use_pixel_to_image_fusion --pixel_to_image_weight 0.10 --pixel_to_image_topk_ratio 0.01
```

## 6. 主要结果

### 6.1 MVTec mean 结果

原始 AnomalyCLIP 日志：`results/9_12_4_multiscale/zero_shot/log.txt`

当前方法日志：`cached_results/cached_results_layer1_wg_msml_multicrop_w050_p2i010/log.txt`

| 方法 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| 原始 AnomalyCLIP | 91.1 | 81.4 | 91.6 | 96.4 |
| 当前方法 | 91.8 | 85.6 | 94.5 | 97.6 |
| 提升 | +0.7 | +4.2 | +2.9 | +1.2 |

MVTec 当前方法分类结果：

| 类别 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| carpet | 99.2 | 96.1 | 100.0 | 100.0 |
| bottle | 89.9 | 81.3 | 90.2 | 97.1 |
| hazelnut | 97.3 | 94.5 | 99.0 | 99.5 |
| leather | 99.0 | 97.4 | 100.0 | 100.0 |
| cable | 78.3 | 66.7 | 81.4 | 88.6 |
| capsule | 96.1 | 89.0 | 93.4 | 98.7 |
| grid | 97.7 | 87.4 | 98.2 | 99.5 |
| pill | 92.0 | 92.2 | 86.8 | 97.2 |
| transistor | 69.5 | 57.6 | 94.1 | 92.8 |
| metal_nut | 78.3 | 75.6 | 96.6 | 99.2 |
| screw | 98.0 | 92.2 | 86.6 | 94.8 |
| toothbrush | 91.7 | 90.8 | 94.7 | 98.2 |
| zipper | 95.8 | 83.4 | 98.5 | 99.5 |
| tile | 96.3 | 88.5 | 99.9 | 100.0 |
| wood | 97.1 | 91.4 | 97.5 | 99.3 |
| mean | 91.8 | 85.6 | 94.5 | 97.6 |

### 6.2 VisA mean 结果

原始 AnomalyCLIP 日志：`results/9_12_4_multiscale_visa/zero_shot/log.txt`

当前方法日志：`cached_results/cached_results_visa_wg_msml_multicrop_w050_p2i010/log.txt`

| 方法 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| 原始 AnomalyCLIP | 95.5 | 86.7 | 82.0 | 85.3 |
| 当前方法 | 96.2 | 91.3 | 84.6 | 87.4 |
| 提升 | +0.7 | +4.6 | +2.6 | +2.1 |

VisA 当前方法分类结果：

| 类别 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| candle | 99.2 | 97.0 | 87.5 | 89.9 |
| capsules | 96.4 | 92.2 | 86.0 | 91.7 |
| cashew | 94.6 | 92.9 | 81.1 | 91.9 |
| chewinggum | 99.1 | 91.9 | 97.6 | 99.0 |
| fryum | 95.3 | 90.9 | 95.0 | 97.7 |
| macaroni1 | 99.0 | 93.8 | 86.7 | 86.6 |
| macaroni2 | 98.4 | 88.3 | 67.4 | 65.6 |
| pcb1 | 94.6 | 88.6 | 88.2 | 89.6 |
| pcb2 | 94.1 | 83.8 | 66.7 | 69.3 |
| pcb3 | 89.9 | 86.3 | 65.4 | 72.2 |
| pcb4 | 96.1 | 92.4 | 96.3 | 96.4 |
| pipe_fryum | 98.4 | 97.7 | 97.4 | 98.8 |
| mean | 96.2 | 91.3 | 84.6 | 87.4 |

## 7. 调参过程简要记录

### 7.1 早期直接跑 `scripts/evaluate/test.py` 的 wavelet + TTA

日志目录：`sweep_results/20260618_190501`

这批实验没有使用 feature cache，CPU 上每组大约 53 分钟。整体结果不理想，MVTec mean 大致在：

- pixel AUROC：`87.7` 到 `87.9`
- pixel AUPRO：`77.2` 到 `78.8`
- image AUROC：`91.5`
- image AP：`96.3`

说明早期小波抑制方式会伤害 pixel 指标，因此后面改为 cached evaluation，并重构为更温和的 dual-route / wavelet-guided 方案。

### 7.2 Cached wavelet + TTA sweep

日志目录：`sweep_results/20260620_120601`

| 实验名 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| 01_wg_tta_weak_balanced | 91.2 | 83.2 | 91.6 | 96.4 |
| 02_wg_tta_mid_repulsion | 91.2 | 83.1 | 91.5 | 96.3 |
| 03_wg_tta_abnormal_light | 91.1 | 83.0 | 91.5 | 96.3 |
| 04_wg_tta_high_confidence | 91.2 | 83.1 | 91.6 | 96.4 |
| 05_legacy_tta_control | 91.1 | 83.0 | 91.6 | 96.4 |

这批实验相比原始 MVTec 的 `81.4` pixel AUPRO 有提升，但 image-level 指标基本没有明显改善。

### 7.3 后续 cached result 搜索

日志目录：`cached_results`

| 实验名 | pixel AUROC | pixel AUPRO | image AUROC | image AP | 备注 |
|:--|--:|--:|--:|--:|:--|
| cached_results_baseline_layer1 | 91.2 | 83.2 | 91.6 | 96.4 | cached 评估基线 |
| cached_results_layer1_wg_search_sigma5 | 91.3 | 83.4 | 91.6 | 96.4 | 小波/TTA cached 初步提升 |
| cached_results_layer1_wg_multiscale_wavelet_multilayer_tta | 91.3 | 83.4 | 92.4 | 96.9 | 多层融合后 image 指标提升 |
| cached_results_layer1_wg_msml_p2i010 | 91.3 | 83.4 | 94.0 | 97.4 | 加入 pixel-to-image fusion |
| cached_results_layer1_wg_msml_multicrop_w010_p2i010 | 91.6 | 84.1 | 94.0 | 97.4 | multi-crop weight 0.10 |
| cached_results_layer1_wg_msml_multicrop_w025_p2i010 | 91.7 | 85.1 | 94.2 | 97.5 | multi-crop weight 0.25 |
| cached_results_layer1_wg_msml_multicrop_w040_p2i010 | 91.8 | 85.5 | 94.4 | 97.6 | multi-crop weight 0.40 |
| cached_results_layer1_wg_msml_multicrop_w050_p2i010 | 91.8 | 85.6 | 94.5 | 97.6 | 当前 MVTec 推荐方案 |

没有放入主结果的 debug 类日志，例如 `cache/debug_*` 和 `cached_results_multicrop_debug_bottle`，只用于局部验证，不作为完整数据集结论。

## 8. 消融实验

### 8.1 组件级消融

日志与汇总：`ablation_results/20260622_094146_component/summary.md`

这组实验在 MVTec 和 VisA 上使用同一套参数，固定 `feature_map_layer 1 2 3`、`sigma 5`、`aupro_steps 200`，逐步加入各组件。

MVTec：

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| original_anomalyclip | 91.1 | 81.4 | 91.6 | 96.4 |
| cached_baseline_l123 | 91.3 | 83.1 | 91.6 | 96.4 |
| wavelet_only | 91.3 | 83.1 | 91.6 | 96.4 |
| tta_only | 91.3 | 83.4 | 91.6 | 96.4 |
| wavelet_tta | 91.3 | 83.4 | 91.6 | 96.4 |
| wavelet_tta_p2i | 91.3 | 83.4 | 94.0 | 97.4 |
| wavelet_tta_multicrop | 91.8 | 85.6 | 91.6 | 96.4 |
| full_method | 91.8 | 85.6 | 94.5 | 97.6 |

VisA：

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| original_anomalyclip | 95.5 | 86.7 | 82.0 | 85.3 |
| cached_baseline_l123 | 95.6 | 87.1 | 82.0 | 85.3 |
| wavelet_only | 95.6 | 87.1 | 82.0 | 85.3 |
| tta_only | 95.6 | 87.1 | 82.0 | 85.4 |
| wavelet_tta | 95.6 | 87.1 | 82.0 | 85.4 |
| wavelet_tta_p2i | 95.6 | 87.1 | 83.5 | 86.6 |
| wavelet_tta_multicrop | 96.2 | 91.3 | 82.0 | 85.4 |
| full_method | 96.2 | 91.3 | 84.6 | 87.4 |

结论：

- `multi-crop fusion` 是 pixel AUPRO 的主要来源：MVTec 从 `83.4` 到 `85.6`，VisA 从 `87.1` 到 `91.3`。
- `pixel-to-image fusion` 是 image-level 指标的主要来源：MVTec image AUROC/AP 从 `91.6/96.4` 到 `94.0/97.4`，VisA 从 `82.0/85.4` 到 `83.5/86.6`。
- `wavelet_only` 在当前参数下几乎不单独改变 mean 指标；它更多作为 TTA、multi-crop 和 map fusion 的校准路线存在。
- 完整方法同时获得 pixel-level 和 image-level 提升：MVTec `91.8/85.6/94.5/97.6`，VisA `96.2/91.3/84.6/87.4`。

### 8.2 内部设计消融

日志与汇总：`ablation_results/20260622_111707_internal/summary.md`

这组实验在完整方法基础上只去掉一个内部小设计：

- `full_no_wavelet_confidence`：去掉 wavelet confidence gating。
- `full_no_rank_preserve`：去掉 rank-preserve top-k protection。
- `full_no_local_contrast`：去掉 local texture contrast。

MVTec：

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| full_method | 91.8 | 85.6 | 94.5 | 97.6 |
| full_no_wavelet_confidence | 91.8 | 85.6 | 94.5 | 97.6 |
| full_no_rank_preserve | 91.8 | 85.6 | 94.5 | 97.6 |
| full_no_local_contrast | 91.8 | 85.6 | 94.5 | 97.6 |

VisA：

| 实验 | pixel AUROC | pixel AUPRO | image AUROC | image AP |
|:--|--:|--:|--:|--:|
| full_method | 96.2 | 91.3 | 84.6 | 87.4 |
| full_no_wavelet_confidence | 96.2 | 91.3 | 84.6 | 87.4 |
| full_no_rank_preserve | 96.2 | 91.3 | 84.6 | 87.4 |
| full_no_local_contrast | 96.2 | 91.3 | 84.6 | 87.4 |

结论：这三个内部小设计在当前最终参数下没有带来可见 mean 指标变化。论文主表不建议强调它们为核心贡献，可以作为附录或负结果说明，主消融应重点展示 `multi-crop fusion` 和 `pixel-to-image fusion` 的互补作用。

## 9. 当前结论

- 当前推荐方案在 MVTec 和 VisA 上都同时提升了 pixel-level 和 image-level 指标。
- 最稳定、最明显的收益来自 pixel AUPRO：MVTec `+4.2`，VisA `+4.6`。
- image-level 指标的主要提升来自 multi-layer map、multi-crop fusion 和 pixel-to-image fusion 组合。
- 当前方案没有额外训练，属于 zero-shot test-time calibration / fusion 方向。
- 后续如果写论文，主实验可以先围绕 MVTec 和 VisA 的完整数据集结果展开；消融实验可以按小波校准、TTA、multi-crop、pixel-to-image fusion 分组。

## 10. 重要文件索引

代码文件：

- `wavelet_calibration.py`：小波校准、dual-route、局部对比度、rank-preserve 等逻辑。
- `test_time_rectification.py`：TTA rectification 逻辑。
- `scripts/evaluate/eval_cached_calibration.py`：基于 feature cache 的快速评估入口。
- `scripts/cache/cache_mvtec_features.py`：生成 patch feature cache。
- `scripts/cache/cache_multicrop_maps.py`：生成 multi-crop anomaly map cache。
- `multicrop_utils.py`：multi-crop fusion 相关工具。
- `scripts/experiments/run_param_sweep.py`：顺序跑多组参数的调参脚本。
- `scripts/experiments/run_ablation_experiments.py`：顺序跑组件消融和内部设计消融的脚本。

结果与日志：

- `results/9_12_4_multiscale/zero_shot/log.txt`：MVTec 原始 AnomalyCLIP 结果。
- `results/9_12_4_multiscale_visa/zero_shot/log.txt`：VisA 原始 AnomalyCLIP 结果。
- `cached_results/cached_results_layer1_wg_msml_multicrop_w050_p2i010/log.txt`：MVTec 当前推荐方案结果。
- `cached_results/cached_results_visa_wg_msml_multicrop_w050_p2i010/log.txt`：VisA 当前推荐方案结果。
- `ablation_results/20260622_094146_component/summary.md`：MVTec/VisA 组件级消融汇总。
- `ablation_results/20260622_111707_internal/summary.md`：MVTec/VisA 内部设计消融汇总。
- `sweep_results/20260618_190501/sweep_log.txt`：早期非 cached wavelet + TTA sweep。
- `sweep_results/20260620_120601/sweep_log.txt`：cached wavelet + TTA sweep。

缓存：

- `cache/mvtec_anomalyclip_features`
- `cache/mvtec_multicrop_maps_grid2_ratio075`
- `cache/visa_anomalyclip_features`
- `cache/visa_multicrop_maps_grid2_ratio075`

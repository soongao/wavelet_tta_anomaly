# WPTA 实现细节审计 v0.2

审计对象：`outputs/wpta_cvpr_paper_draft_zh_v1.4.md`

目标：核对当前论文稿中 implementation details 是否能由现有配置、脚本和结果日志支撑。本文档只记录已核验事实，不补写未运行实验。

## 已核验实现事实

| 项目 | 当前可写入论文的事实 | 证据来源 |
|---|---|---|
| Backbone | 使用 CLIP `ViT-L/14@336px`，加载 AnomalyCLIP prompt checkpoint `9_12_4_multiscale/epoch_15.pth` | `src/anomalyclip/cached_eval_utils.py` 中 `AnomalyCLIP_lib.load("ViT-L/14@336px", ...)`；`conf/test_conf.yaml` 与 cache 配置中的 checkpoint path |
| 输入尺寸与预处理 | 输入图像 resize 到 `518 x 518`，center crop 到 518，并使用 OpenAI CLIP mean/std 归一化 | `conf/test_conf.yaml`；`src/anomalyclip/utils.py:get_transform` |
| 缓存特征 | cache generation 使用 `features_list: 6, 12, 18, 24`，缓存 feature_map_layer 包含 `0, 1, 2, 3` | `conf/cache_mvtec_features_conf.yaml` |
| 主评估层 | 受控消融与主系统评估使用 feature map layers `1, 2, 3` | `conf/run_prototype_ablation_experiments_conf.yaml`；`conf/run_ablation_experiments_conf.yaml`；`FIVE_DATASET_RESULTS_AND_ABLATIONS.md` |
| 层融合 | anomaly maps 从所选层重建后使用 `sum` 融合，layer temperature 为 1.0 | `conf/run_prototype_ablation_experiments_conf.yaml`；`src/anomalyclip/cached_eval_utils.py:build_anomaly_maps_from_patch_features` |
| 上采样 | patch-level/prototype-level score map 使用 bilinear interpolation 上采样到 `image_size x image_size` | `src/anomalyclip/prototype_adaptation.py:_final_map_from_layers`；`AnomalyCLIP_lib.get_similarity_map(..., image_size)` |
| 平滑 | 最终 anomaly map 使用 Gaussian smoothing，受控 MVTec/VisA 为 `sigma=5` | `scripts/evaluate/eval_cached_calibration.py`；`src/anomalyclip/cached_eval_utils.py:smooth_anomaly_map` |
| Pixel metrics | pixel AUROC 使用 flattened mask/map；P-AUPRO 使用 PRO AUC，`max_step=200`，`expect_fpr=0.3` | `src/anomalyclip/metrics.py` |
| Image metrics | image AUROC 与 image AP 使用 image-level labels 和 `pr_sp` scores | `src/anomalyclip/metrics.py` |
| Pixel-to-image fusion | 若启用，先取 anomaly map top-k pixel score，再与 image-text abnormal probability 按权重融合 | `scripts/evaluate/eval_cached_calibration.py`；`src/anomalyclip/wavelet_calibration.py:topk_pixel_score` 与 `fuse_image_score_with_pixel_score` |
| Multi-crop fusion | 若启用，将 base anomaly map 与 stitched crop map 按 `multicrop_weight` 线性融合；尺寸不一致时 bilinear resize crop map | `scripts/evaluate/eval_cached_calibration.py:_fuse_multicrop_map` |
| Prototype adaptation 默认参数 | `proto_temperature=0.07`，`gamma=1.0`，`eta=1.0`，`proto_topk_ratio=0.2`，`proto_wavelet_mode=boundary_aware`，`proto_wavelet_mix=0.05` | `conf/run_prototype_ablation_experiments_conf.yaml` |
| Conservative update | 默认 `proto_alpha0=0.0`、`proto_beta0=0.01`、`proto_tau_a=0.15`、`proto_update_min_abnormal_confidence=0.06`；默认主要移动 normal prototype，abnormal evidence 主要用于 gate | `conf/run_prototype_ablation_experiments_conf.yaml`；`src/anomalyclip/prototype_adaptation.py:_calibrate_text_features` |
| Direct fusion 负对照 | 不做 prototype adaptation，使用 `direct_wavelet_fusion_weight=0.5` 直接融合 `S0` 与 wavelet map | `conf/run_prototype_ablation_experiments_conf.yaml`；`src/anomalyclip/prototype_adaptation.py:apply_direct_wavelet_fusion` |

## 已写入 v1.4 的修改

- 3.6 明确了 bilinear upsampling、Gaussian smoothing，以及 pixel-to-image fusion 的 image score 逻辑。
- 3.7 明确了 backbone、checkpoint、input size、preprocessing、feature list、evaluated layers、layer fusion、AUPRO steps 与 smoothing。
- 3.5 明确了当前默认 `alpha0=0.0`、`beta0=0.01`，避免把实现写成对 normal/abnormal prototypes 等强度更新。
- 6.3 和投稿前 gate 将 implementation blocker 降级为“转写成匿名投稿版 appendix”，不再作为未核对事实。

## 仍需投稿前处理

- 将本地绝对路径替换为匿名路径或 supplementary reproduction table。
- 若论文最终以 CVPR/ICCV LaTeX 提交，需要把 `conf/` 和 command-line settings 整理成 appendix table，而不是在主文中引用本机文件。
- 若要声称 exact reproducibility，需要提供 cache generation commands for all five datasets；当前只核验了配置与结果命令的关键字段。
- 当前没有多 seed 或多 run，仍不能写显著性、置信区间或 p 值。

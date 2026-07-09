# WPTA 论文表格 v0.3

数据来源：

- 工业主结果、受控消融与小波设计：`/Users/bytedance/code/AnomalyCLIP/paper/result_record/result_table.csv` 中标记为 `current` 的结果。
- 五数据集最终运行配置：`/Users/bytedance/code/AnomalyCLIP/FIVE_DATASET_RESULTS_AND_ABLATIONS.md`。
- 候选外部方法比较：`outputs/iconip_main_results_extracted_analysis.md`，仅作 protocol-reference。
- 医学补充结果：`/Users/bytedance/code/AnomalyCLIP/paper/result_record/medical_result_table.csv`。

指标缩写：`P-AUROC` = pixel AUROC，`P-AUPRO` = pixel AUPRO，`I-AUROC` = image AUROC，`I-AP` = image AP。所有指标越高越好。

证据边界：五数据集主表证明“最终校准系统”相对固定 AnomalyCLIP baseline 的系统级提升；WPTA 机制本身只由 MVTec/VisA 受控消融表支撑。不要写成“WPTA 在五个数据集上均由同一机制带来提升”。

## Table 1. Main results on five industrial anomaly detection benchmarks.

本表展示最终校准系统在五个工业异常检测基准上的结果。平均来看，最终系统相对 AnomalyCLIP baseline 在四个指标上均提升，其中 pixel AUPRO 的平均提升最大。

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | I-AUROC ↑ | I-AP ↑ | Δ vs. baseline |
|---|---|---:|---:|---:|---:|---:|
| MVTec | AnomalyCLIP baseline | 91.2 | 83.2 | 91.6 | 96.4 | - |
| MVTec | Final calibrated system | **91.8** | **85.6** | **94.5** | **97.6** | +0.6 / +2.4 / +2.9 / +1.2 |
| VisA | AnomalyCLIP baseline | 95.5 | 86.7 | 82.0 | 85.3 | - |
| VisA | Final calibrated system | **96.2** | **91.3** | **84.6** | **87.4** | +0.7 / +4.6 / +2.6 / +2.1 |
| MPDD | AnomalyCLIP baseline | 96.9 | 84.6 | 73.7 | 76.5 | - |
| MPDD | Final calibrated system | **97.3** | **89.9** | **77.8** | **82.3** | +0.4 / +5.3 / +4.1 / +5.8 |
| BTAD | AnomalyCLIP baseline | 93.5 | 70.5 | 89.1 | 91.0 | - |
| BTAD | Final calibrated system | **96.3** | **78.2** | **93.9** | **94.9** | +2.8 / +7.7 / +4.8 / +3.9 |
| DTD-Synthetic | AnomalyCLIP baseline | 97.4 | 89.1 | 94.5 | 97.7 | - |
| DTD-Synthetic | Final calibrated system | **97.9** | **91.8** | **96.9** | **98.7** | +0.5 / +2.7 / +2.4 / +1.0 |
| Average | AnomalyCLIP baseline | 94.9 | 82.8 | 86.2 | 89.4 | - |
| Average | Final calibrated system | **95.9** | **87.4** | **89.5** | **92.2** | +1.0 / +4.5 / +3.4 / +2.8 |

## Table 2. Core component ablation on MVTec and VisA.

本表是 WPTA 机制的核心证据。直接把小波线索融合到最终异常图会伤害定位；相比之下，语义原型适配、小波引导的证据选择以及保守更新在受控设置中逐步改善 AUPRO。

| Method | MVTec P-AUROC ↑ | MVTec P-AUPRO ↑ | MVTec I-AUROC ↑ | MVTec I-AP ↑ | VisA P-AUROC ↑ | VisA P-AUPRO ↑ | VisA I-AUROC ↑ | VisA I-AP ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 91.2 | 83.2 | 91.6 | 96.4 | 95.5 | 86.7 | 82.0 | 85.3 |
| Direct wavelet fusion / no adaptation | 88.7 | 80.4 | 92.9 | 96.9 | 94.6 | 85.1 | 81.6 | 84.8 |
| Semantic prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 | 96.0 | 90.4 | 83.7 | 86.9 |
| Wavelet prototype adaptation w/o conservative | 91.7 | 85.8 | 93.9 | 97.2 | 96.1 | 91.3 | 84.1 | 87.0 |
| Full WPTA, controlled ablation setting | **91.8** | **86.2** | **94.1** | **97.4** | **96.2** | **91.7** | **84.3** | **87.3** |

## Table 3. Wavelet reliability design ablation on MVTec and VisA.

本表说明小波信息的有效用法不是替代语义异常图，而是作为测试时原型适配的可靠性权重。边界感知可靠性优于只使用高频响应。

| Wavelet setting | MVTec P-AUROC ↑ | MVTec P-AUPRO ↑ | MVTec I-AUROC ↑ | MVTec I-AP ↑ | VisA P-AUROC ↑ | VisA P-AUPRO ↑ | VisA I-AUROC ↑ | VisA I-AP ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Semantic-only prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 | 96.0 | 90.4 | 83.7 | 86.9 |
| Direct wavelet fusion | 88.7 | 80.4 | 92.9 | 96.9 | 94.6 | 85.1 | 81.6 | 84.8 |
| HF-only W + prototype adaptation | 91.6 | 85.3 | 94.0 | 97.2 | 96.0 | 90.8 | 84.0 | 86.9 |
| Boundary-aware W + prototype adaptation | 91.7 | 85.7 | 93.8 | 97.3 | 96.1 | 91.2 | 83.9 | 87.1 |
| Full boundary-aware W + conservative | **91.8** | **86.2** | **94.1** | **97.4** | **96.2** | **91.7** | **84.3** | **87.3** |

## Table 4. Final-system configuration used in the five-dataset table.

本表用于防止论文叙事越界：MVTec/VisA 的最终系统包含 WPTA/TTA/多裁剪/像素到图像融合；DTD-Synthetic 包含小波可靠性、多裁剪和像素到图像融合；MPDD/BTAD 当前最终结果主要来自多裁剪和像素到图像融合。因此，五数据集表应写作系统级结果，WPTA 机制证据应回到 Table 2 和 Table 3。

| Dataset | Wavelet reliability | TTA rectification | Multi-crop fusion | Pixel-to-image fusion | Key setting | Result log |
|---|---|---|---|---|---|---|
| MVTec | yes | yes | yes | yes | sigma=5, multicrop weight=0.5, p2i weight=0.1 | `ablation_results/20260622_094146_component/mvtec/07_full_method/log.txt` |
| VisA | yes | yes | yes | yes | sigma=5, multicrop weight=0.5, p2i weight=0.1 | `ablation_results/20260622_094146_component/visa/07_full_method/log.txt` |
| MPDD | no | no | yes | yes | sigma=8, multicrop weight=0.25, p2i weight=0.8 | `cached_results/goal_20260707/mpdd_multicrop_partial_w025_sigma8_p2i/log.txt` |
| BTAD | no | no | yes | yes | sigma=10, multicrop weight=0.85, p2i weight=0.95 | `cached_results/goal_20260707/btad_full_multicrop_w085_sigma10_p2i030_w095/log.txt` |
| DTD-Synthetic | yes | no | yes | yes | sigma=8, multicrop weight=0.75, p2i weight=0.5 | `cached_results/goal_20260707/dtd_final_no_strat_woven127_w075_sigma8_p2i0002_w05/log.txt` |

## Table 5. Candidate comparison with representative reported methods on MVTec and VisA.

本表从 `outputs/iconip_main_results_extracted_analysis.md` 与当前 CSV 合并得到，只能作为 protocol-reference。正式投稿前必须核验 split、backbone、输入分辨率、后处理和 evaluation script；在核验完成前，不能用本表单独宣称 SOTA。

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | I-AUROC ↑ | I-AP ↑ | Source |
|---|---|---:|---:|---:|---:|---|
| MVTec | CLIP | 38.4 | 11.3 | 74.1 | 87.6 | extracted `main.tex` |
| MVTec | WinCLIP | 85.1 | 64.6 | 91.8 | 96.5 | extracted `main.tex` |
| MVTec | VAND | 87.6 | 44.0 | 86.1 | 93.5 | extracted `main.tex` |
| MVTec | CoOp | 33.3 | 6.7 | 88.8 | 94.8 | extracted `main.tex` |
| MVTec | AdaCLIP | 88.7 | 37.8 | 89.2 | 96.4 | extracted `main.tex` |
| MVTec | AnomalyCLIP | 91.1 | 81.4 | 91.5 | 96.2 | extracted `main.tex` |
| MVTec | AA-CLIP† | **91.9** | 84.6 | 90.5 | 94.9 | extracted `main.tex` |
| MVTec | Source Ours (TAAP/INPC) | 91.5 | 85.5 | 92.8 | 96.7 | extracted `main.tex` |
| MVTec | Final calibrated system | 91.8 | **85.6** | **94.5** | **97.6** | current CSV |
| VisA | CLIP | 46.6 | 14.8 | 66.4 | 71.5 | extracted `main.tex` |
| VisA | WinCLIP | 79.6 | 56.8 | 78.1 | 81.2 | extracted `main.tex` |
| VisA | VAND | 94.2 | 86.8 | 78.0 | 81.4 | extracted `main.tex` |
| VisA | CoOp | 24.2 | 3.8 | 62.8 | 68.1 | extracted `main.tex` |
| VisA | AdaCLIP | 95.5 | 56.8 | **85.8** | 84.9 | extracted `main.tex` |
| VisA | AnomalyCLIP | 95.4 | 87.0 | 82.1 | 85.4 | extracted `main.tex` |
| VisA | AA-CLIP† | 95.5 | 83.0 | 84.6 | 82.2 | extracted `main.tex` |
| VisA | Source Ours (TAAP/INPC) | 95.6 | 88.3 | 83.3 | 85.9 | extracted `main.tex` |
| VisA | Final calibrated system | **96.2** | **91.3** | 84.6 | **87.4** | current CSV |

## Appendix Table A1. Preliminary medical pixel-level result.

本表只能作为补充结果。当前只有 ISIC/ISBI 完整，ColonDB、ClinicDB、Kvasir、HeadCT、BrainMRI、Br35H 均未形成完整可比结果，因此不要把医学跨域泛化写进主 claim。

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | Status | Note |
|---|---|---:|---:|---|---|
| ISIC/ISBI | AnomalyCLIP baseline | 88.7 | 78.6 | current | `cached_results/medical_20260707/isbi_baseline_l123_sigma5/log.txt` |
| ISIC/ISBI | Final calibrated system | **89.9** | **80.0** | current | `cached_results/medical_20260707/isbi_sigma8_l123/log.txt` |

## Claim Support

| Claim | Supporting table | Allowed wording |
|---|---|---|
| 最终校准系统在五个工业数据集上相对固定 AnomalyCLIP baseline 均有提升。 | Table 1 | supported |
| 五数据集平均提升最大的是 pixel AUPRO。 | Table 1 | supported |
| WPTA 机制能在受控 MVTec/VisA 设置中改善原型适配。 | Table 2, Table 3 | supported |
| 直接小波图融合不是有效路径，且可能伤害定位。 | Table 2, Table 3 | supported |
| 边界感知小波可靠性优于只使用高频可靠性。 | Table 3 | supported |
| 最终系统与代表性 CLIP-based 方法在 MVTec/VisA 上有竞争力。 | Table 5 | protocol-reference only |
| 医学数据上也有稳定泛化。 | Appendix Table A1 | not supported beyond ISIC/ISBI preliminary result |

# WPTA 可生成论文表格 v0.4

本文件只汇总当前已有数值可以直接生成的表格。所有指标均为百分比，指标顺序在斜杠单元格中统一为 `P-AUROC / P-AUPRO / I-AUROC / I-AP`，分别对应 pixel AUROC、pixel AUPRO、image AUROC、image AP，且数值越高越好。

数据来源：

- 工业主结果、MVTec/VisA 受控原型消融与小波设计消融：`/Users/bytedance/code/AnomalyCLIP/paper/result_record/result_table.csv` 中 `current` 结果。
- 五数据集最终运行配置：`/Users/bytedance/code/AnomalyCLIP/FIVE_DATASET_RESULTS_AND_ABLATIONS.md`。
- MVTec/VisA 系统校准模块消融：`/Users/bytedance/code/AnomalyCLIP/ablation_results/20260622_094146_component/summary.md`。
- 医学补充结果：`/Users/bytedance/code/AnomalyCLIP/paper/result_record/medical_result_table.csv`。
- 外部方法候选比较：`outputs/iconip_main_results_extracted_analysis.md`，仅作为 protocol-reference。

证据边界：

- Table 1 支撑“最终校准系统相对固定 AnomalyCLIP baseline 在五个工业数据集上提升”。
- Table 2 和 Table 3 支撑“WPTA 机制在 MVTec/VisA 受控设置中有效”。
- Table 4 用于限制因果归因，说明五数据集 final system 的启用模块不同。
- Appendix Table A1 是系统校准栈附录表，不等同于 WPTA 原型机制消融。
- Appendix Table B1 尚未完成协议核验，不能用于 SOTA claim。

## 可生成表格清单

| ID | 建议位置 | 当前状态 | 主要用途 |
|---|---|---|---|
| Table 1 | 主文 Experiments | 可直接使用 | 五数据集主结果，证明 final calibrated system 的系统级提升 |
| Table 2 | 主文 Ablation | 可直接使用 | WPTA 核心组件消融 |
| Table 3 | 主文 Ablation 或附录 | 可直接使用 | 小波可靠性设计消融 |
| Table 4 | 主文或附录靠前位置 | 可直接使用 | final setting 配置审计，避免越界归因 |
| Appendix Table A1 | 附录 | 可直接使用 | 系统校准栈消融，用于解释主结果中的工程模块 |
| Appendix Table A2 | 附录 | 可直接使用但只能弱表述 | ISIC/ISBI 医学补充结果 |
| Appendix Table B1 | 附录或暂存 | protocol-reference only | 外部方法候选对比，待协议核验 |

## Table 1. 五个工业异常检测基准上的主结果

本表展示最终校准系统与固定 AnomalyCLIP baseline 的比较。平均来看，最终系统在四个指标上均提升，其中 P-AUPRO 的平均提升最大。

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

推荐表述：

> The final calibrated system improves the fixed AnomalyCLIP baseline on all five industrial benchmarks, with the largest average gain on pixel AUPRO.

## Table 2. WPTA 核心组件消融

本表是 WPTA 机制的主要证据。直接把小波响应融合到最终异常图会伤害定位；相比之下，语义原型适配、小波引导 evidence weighting 和 conservative update 在受控设置中逐步改善结果。

| Method | MVTec | VisA | 作用 |
|---|---:|---:|---|
| Baseline | 91.2 / 83.2 / 91.6 / 96.4 | 95.5 / 86.7 / 82.0 / 85.3 | fixed AnomalyCLIP prototypes |
| Direct wavelet fusion / no adaptation | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | negative control: use wavelet at final-map level |
| Semantic prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | CLIP semantic evidence only |
| Wavelet prototype adaptation w/o conservative | 91.7 / 85.8 / 93.9 / 97.2 | 96.1 / 91.3 / 84.1 / 87.0 | boundary-aware wavelet evidence, no conservative update |
| Full WPTA controlled setting | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** | boundary-aware wavelet evidence + conservative update |

表注建议：除 Baseline 外，prototype/fusion variants 使用相同的 multi-crop 与 pixel-to-image 设置：`multicrop_weight=0.50`，`pixel_to_image_weight=0.10`，`pixel_to_image_topk_ratio=0.01`。因此，本表的因果解释应主要比较 variants 之间的相对变化，而不是把 baseline 到任一增强行的差值单独归因为 WPTA。

## Table 3. 小波可靠性设计消融

本表说明小波信息的有效角色不是替代语义异常图，而是作为测试时原型适配中的 patch evidence reliability。Boundary-aware reliability 在两个数据集上均优于只使用高频响应。

| Wavelet setting | MVTec | VisA | 解释 |
|---|---:|---:|---|
| Semantic-only prototype adaptation | 91.6 / 85.2 / 93.7 / 97.1 | 96.0 / 90.4 / 83.7 / 86.9 | no wavelet reliability |
| Direct wavelet fusion | 88.7 / 80.4 / 92.9 / 96.9 | 94.6 / 85.1 / 81.6 / 84.8 | wavelet as final-map fusion, negative control |
| HF-only W + prototype adaptation | 91.6 / 85.3 / 94.0 / 97.2 | 96.0 / 90.8 / 84.0 / 86.9 | high-frequency reliability only |
| Boundary-aware W + prototype adaptation | 91.7 / 85.7 / 93.8 / 97.3 | 96.1 / 91.2 / 83.9 / 87.1 | suppress structure-boundary pseudo evidence |
| Full boundary-aware W + conservative | **91.8 / 86.2 / 94.1 / 97.4** | **96.2 / 91.7 / 84.3 / 87.3** | final controlled WPTA setting |

推荐表述：

> Boundary-aware wavelet reliability is more useful than high-frequency response alone, while direct final-map fusion is a negative control.

## Table 4. 五数据集 final system 配置审计

本表用于防止论文叙事越界。五数据集主结果证明的是 final calibrated system 的系统级有效性；WPTA 机制本身由 Table 2 和 Table 3 的 MVTec/VisA 受控消融支撑。

| Dataset | Final result | Wavelet reliability | TTA rectification | Multi-crop fusion | Pixel-to-image fusion | Key setting | Result log |
|---|---:|---|---|---|---|---|---|
| MVTec | 91.8 / 85.6 / 94.5 / 97.6 | yes | yes | yes | yes | sigma=5, mc=0.50, p2i=0.10 | `mvtec/07_full_method/log.txt` |
| VisA | 96.2 / 91.3 / 84.6 / 87.4 | yes | yes | yes | yes | sigma=5, mc=0.50, p2i=0.10 | `visa/07_full_method/log.txt` |
| MPDD | 97.3 / 89.9 / 77.8 / 82.3 | no | no | yes | yes | sigma=8, mc=0.25, p2i=0.80 | `mpdd_multicrop_partial_w025_sigma8_p2i/log.txt` |
| BTAD | 96.3 / 78.2 / 93.9 / 94.9 | no | no | yes | yes | sigma=10, mc=0.85, p2i=0.95, p2i-topk=0.30 | `btad_full_multicrop_w085_sigma10_p2i030_w095/log.txt` |
| DTD-Synthetic | 97.9 / 91.8 / 96.9 / 98.7 | yes | no | yes | yes | sigma=8, mc=0.75, p2i=0.50, p2i-topk=0.002 | `dtd_final_no_strat_woven127_w075_sigma8_p2i0002_w05/log.txt` |

推荐表述：

> The five-dataset table should be interpreted as a system-level result because the final settings activate different calibration modules across datasets.

## Appendix Table A1. MVTec/VisA 系统校准栈消融

本表来自 `component` suite，用于解释 MVTec/VisA final calibrated system 中各工程模块的贡献。它不是 WPTA 原型机制消融；机制证据仍应引用 Table 2 和 Table 3。

| Calibration setting | MVTec | VisA | 说明 |
|---|---:|---:|---|
| Original AnomalyCLIP log | 91.1 / 81.4 / 91.6 / 96.4 | 95.5 / 86.7 / 82.0 / 85.3 | original reported log |
| Cached baseline L1/2/3 | 91.3 / 83.1 / 91.6 / 96.4 | 95.6 / 87.1 / 82.0 / 85.3 | cached features, no proposed calibration |
| Wavelet calibration only | 91.3 / 83.1 / 91.6 / 96.4 | 95.6 / 87.1 / 82.0 / 85.3 | wavelet calibration without downstream fusion modules |
| TTA only | 91.3 / 83.4 / 91.6 / 96.4 | 95.6 / 87.1 / 82.0 / 85.4 | test-time rectification only |
| Wavelet + TTA | 91.3 / 83.4 / 91.6 / 96.4 | 95.6 / 87.1 / 82.0 / 85.4 | wavelet calibration plus TTA |
| Wavelet + TTA + pixel-to-image | 91.3 / 83.4 / 94.0 / 97.4 | 95.6 / 87.1 / 83.5 / 86.6 | mainly improves image-level metrics |
| Wavelet + TTA + multi-crop | 91.8 / 85.6 / 91.6 / 96.4 | 96.2 / 91.3 / 82.0 / 85.4 | mainly improves pixel-level metrics |
| Full calibration stack | **91.8 / 85.6 / 94.5 / 97.6** | **96.2 / 91.3 / 84.6 / 87.4** | combines pixel and image calibration gains |

## Appendix Table A2. 医学补充结果

当前只有 ISIC/ISBI 有完整可比结果。该表只能作为附录初步观察，不能写成医学跨域泛化主结论。

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | Status | Note |
|---|---|---:|---:|---|---|
| ISIC/ISBI | AnomalyCLIP baseline | 88.7 | 78.6 | current | `isbi_baseline_l123_sigma5/log.txt` |
| ISIC/ISBI | Final calibrated system | **89.9** | **80.0** | current | `isbi_sigma8_l123/log.txt` |

## Appendix Table B1. MVTec/VisA 外部方法候选比较

本表仅作 protocol-reference。外部方法数字来自另一份 `main.tex` 的表格摘录，当前 final system 数字来自本项目 CSV。由于 split、backbone、输入尺寸、后处理和 evaluation script 尚未逐项核验，本表暂不加粗 best values，也不能用于 SOTA claim。

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | I-AUROC ↑ | I-AP ↑ | Source |
|---|---|---:|---:|---:|---:|---|
| MVTec | CLIP | 38.4 | 11.3 | 74.1 | 87.6 | extracted `main.tex` |
| MVTec | WinCLIP | 85.1 | 64.6 | 91.8 | 96.5 | extracted `main.tex` |
| MVTec | VAND | 87.6 | 44.0 | 86.1 | 93.5 | extracted `main.tex` |
| MVTec | CoOp | 33.3 | 6.7 | 88.8 | 94.8 | extracted `main.tex` |
| MVTec | AdaCLIP | 88.7 | 37.8 | 89.2 | 96.4 | extracted `main.tex` |
| MVTec | AnomalyCLIP | 91.1 | 81.4 | 91.5 | 96.2 | extracted `main.tex` |
| MVTec | AA-CLIP† | 91.9 | 84.6 | 90.5 | 94.9 | extracted `main.tex` |
| MVTec | Source Ours (TAAP/INPC) | 91.5 | 85.5 | 92.8 | 96.7 | extracted `main.tex` |
| MVTec | Current final system | 91.8 | 85.6 | 94.5 | 97.6 | current CSV |
| VisA | CLIP | 46.6 | 14.8 | 66.4 | 71.5 | extracted `main.tex` |
| VisA | WinCLIP | 79.6 | 56.8 | 78.1 | 81.2 | extracted `main.tex` |
| VisA | VAND | 94.2 | 86.8 | 78.0 | 81.4 | extracted `main.tex` |
| VisA | CoOp | 24.2 | 3.8 | 62.8 | 68.1 | extracted `main.tex` |
| VisA | AdaCLIP | 95.5 | 56.8 | 85.8 | 84.9 | extracted `main.tex` |
| VisA | AnomalyCLIP | 95.4 | 87.0 | 82.1 | 85.4 | extracted `main.tex` |
| VisA | AA-CLIP† | 95.5 | 83.0 | 84.6 | 82.2 | extracted `main.tex` |
| VisA | Source Ours (TAAP/INPC) | 95.6 | 88.3 | 83.3 | 85.9 | extracted `main.tex` |
| VisA | Current final system | 96.2 | 91.3 | 84.6 | 87.4 | current CSV |

## 缺失表格数据 Prompt

[TABLE_DATA_PROMPT:external_protocol_verified_table]

请核验并生成一个可进入主文的外部方法比较表。候选方法包括 CLIP、WinCLIP、VAND、CoOp、AdaCLIP、AnomalyCLIP、AA-CLIP、TAAP/INPC、Current final system；候选数据集包括 MVTec AD、VisA、MPDD、BTAD、DTD-Synthetic。逐项核验 split、backbone、input resolution、preprocessing、post-processing、prompt setting、是否使用 auxiliary training/fine-tuning、evaluation script 和 metric implementation。输出 CSV 字段：`dataset, method, p_auroc, p_aupro, i_auroc, i_ap, source_file_or_paper, source_line_or_table, citation_key, protocol_match, mismatch_notes, allowed_placement`。只有 `protocol_match=yes` 的行可以进入主文；`partial/no` 的行只能进入附录或删除。

[TABLE_DATA_PROMPT:medical_complete_table]

请补齐医学异常检测补充表，只保留完成可比评估的数据集。候选数据集包括 ISIC/ISBI、ColonDB、ClinicDB、Kvasir、HeadCT、BrainMRI、Br35H。对每个数据集输出 baseline 与 final calibrated system 的 pixel-level 或 image-level 指标，并标明是否具备同一 evaluation script、同一 cache generation setting 和同一 metric implementation。输出 CSV 字段：`dataset, task_level, method, p_auroc, p_aupro, i_auroc, i_ap, cache_dir, result_log, eval_script, status, exclusion_reason`。缺失或不可比的数据集不得填数，只写 exclusion_reason。

[TABLE_DATA_PROMPT:uncertainty_table]

请为主结果和 MVTec/VisA 受控消融生成不确定性表。优先使用多 seed 或多次独立运行；若没有多 seed，请明确只能做 deterministic report，不能报告显著性。输出字段：`dataset, method, metric, n_runs, mean, std, ci95_low, ci95_high, test_against, test_name, p_value, effect_size, valid_for_claim`。若 `n_runs < 3`，将 `valid_for_claim` 标为 `descriptive_only`。

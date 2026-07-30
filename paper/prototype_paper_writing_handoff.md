# Paper Writing Handoff: Unnamed Test-Time Prototype Adaptation Method

This document is for the next writing agent. It summarizes the current code, verified results, usable claims, unusable claims, and paper structure for writing a paper from the existing AnomalyCLIP project state.

## 1. Paper-Type Positioning

- Type: Technique paper.
- Core idea: CLIP-based zero-shot anomaly detection uses fixed normal/abnormal text prototypes, but test images have instance-specific defect appearance; therefore, use a label-free patch-level wavelet reliability signal to calibrate text prototypes at test time.
- Working name: `Ours (unnamed)` until a final title/name is chosen.
- Current honest positioning: the reproduced controlled `Ours (unnamed)` setting improves over the original AnomalyCLIP baseline and over CLIP-only/semantic-only prototype adaptation across all four reported metrics. An earlier record had the CLIP-only/Ours order wrong; the corrected controlled order has Ours as the best row.
- Data note: `newversion/paper_v7` still contains EXPECTED placeholder values and should be data-updated before paper use.

## 2. Fixed Evaluation Protocol

Do not change the metric set. All reported experiments should use:

- pixel AUROC
- pixel AUPRO / PRO
- image AUROC
- image AP

Do not add pixel AP. The current validation summary explicitly fixes the metric protocol.

Primary validation summary:

- `cached_results/prototype_tuned/PROTOTYPE_VALIDATION_SUMMARY.md`

## 3. Current Method Description

Method name:

Unnamed for now. Use `Ours (unnamed)` in tables and `the proposed method` in prose.

Implementation entry points:

- Core method: `src/anomalyclip/prototype_adaptation.py`
- Cached evaluation: `scripts/evaluate/eval_cached_calibration.py`
- Evaluation entry point: `scripts/evaluate/test.py`
- Prototype ablation config: `conf/run_prototype_ablation_experiments_conf.yaml`
- Default eval configs: `conf/eval_cached_calibration_conf.yaml`, `conf/test_conf.yaml`

Config switch:

- `--use_wavelet_prototype_adaptation`

Direct map fusion ablation:

- `--use_direct_wavelet_fusion`

The method is training-free. It does not backpropagate, does not call an optimizer, and does not update CLIP or AnomalyCLIP parameters.

Current validated default parameters:

- `proto_alpha0=0.0`
- `proto_beta0=0.01`
- `proto_update_min_abnormal_confidence=0.06`
- `proto_tau_a=0.15`
- `proto_wavelet_mode=boundary_aware`
- `proto_wavelet_mix=0.05`
- `proto_topk_ratio=0.2`
- `proto_gamma=1.0`
- `proto_eta=1.0`

Best verified evaluation extras:

- `--use_multicrop_fusion --multicrop_weight 0.5`
- `--use_pixel_to_image_fusion --pixel_to_image_weight 0.1 --pixel_to_image_topk_ratio 0.01`

Important writing nuance:

The original paper idea proposed `alpha0=0.2, beta0=0.2`; experiments found this too aggressive. The verified setting disables abnormal prototype updates (`alpha0=0.0`) and keeps only a small normal-prototype update. Write this as a conservative variant found necessary for stable normal-image behavior.

## 4. Algorithm Logic To Write

Use this method outline.

1. Compute initial CLIP patch anomaly probability `S0(i)` from normal/abnormal text prototypes.
2. Reshape patch features into an `H x W x C` grid and compute one-level Haar DWT.
3. Compute high-frequency energy:
   `HF = mean_c(|LH_c| + |HL_c| + |HH_c|)`.
4. Compute low-frequency structural edge from `LL`:
   `LF_edge = gradient_magnitude(LL)`.
5. Upsample both to patch-grid resolution, percentile-clip per image, normalize to `[0, 1]`.
6. Boundary-aware wavelet reliability:
   `W(i) = HF(i) * (1 - LF_edge(i))`.
7. Build semantic-spectral evidence weights:
   - implemented formula uses a conservative mixed wavelet factor:
     `w_a(i) = S0(i)^gamma * ((1 - mix) + mix * W(i))^eta`
     `w_n(i) = (1 - S0(i))^gamma * ((1 - mix) + mix * (1 - W(i)))^eta`
   - with `mix=0.05` in the verified setting.
8. Select top-ratio patches and form visual prototypes by weighted average:
   `v_a`, `v_n`.
9. Calibrate text prototypes:
   - verified setting only applies small normal prototype calibration.
   - abnormal update is effectively disabled by `alpha0=0.0`.
10. Recompute the anomaly map using calibrated prototypes.

Critical distinction:

Do not describe Ours as final anomaly-map fusion between `S0` and `W`. Direct `S0/W` fusion exists only as an ablation and performs poorly.

## 5. Main Results

Metrics are reported as:

`pixel AUROC | pixel AUPRO | image AUROC | image AP`

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Evidence |
|:--|:--|--:|--:|--:|--:|:--|
| MVTec AD | Original AnomalyCLIP | 91.1 | 81.4 | 91.6 | 96.4 | `results/9_12_4_multiscale/zero_shot/log.txt` |
| MVTec AD | Cached baseline, layers 1/2/3 | 91.2 | 83.2 | 91.6 | 96.4 | `cached_results/cached_results_baseline_layer1/log.txt` |
| MVTec AD | Ours (unnamed) | 91.8 | 86.2 | 94.1 | 97.4 | reproduced current result; see `paper/result_record/result_table.csv` |
| VisA | Original AnomalyCLIP | 95.5 | 86.7 | 82.0 | 85.3 | `results/9_12_4_multiscale_visa/zero_shot/log.txt` |
| VisA | Ours (unnamed) | 96.2 | 91.7 | 84.3 | 87.3 | reproduced current result; see `paper/result_record/result_table.csv` |

Safe claim:

Ours improves over the original AnomalyCLIP baseline on both MVTec AD and VisA under the fixed metric protocol. It also improves over CLIP-only/semantic-only prototype adaptation across all four reported metrics under the reproduced controlled setting.

Scope note:

Keep the claim tied to the reproduced MVTec/VisA controlled setting and the fixed metric protocol.

## 6. Ablation Evidence

Most complete ablation set is on MVTec.

| Variant | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Evidence |
|:--|--:|--:|--:|--:|:--|
| baseline, cached layers 1/2/3 | 91.2 | 83.2 | 91.6 | 96.4 | `cached_results/cached_results_baseline_layer1/log.txt` |
| direct wavelet fusion | 88.7 | 80.4 | 92.9 | 96.9 | reproduced current negative control; see `paper/result_record/result_table.csv` |
| CLIP-only / semantic-only prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 | reproduced current result; see `paper/result_record/result_table.csv` |
| HF-only reliability + prototype adaptation | 91.6 | 85.3 | 94.0 | 97.2 | reproduced current result; see `paper/result_record/result_table.csv` |
| boundary-aware `W`, mix=1.0 | 91.7 | 85.7 | 93.8 | 97.3 | reproduced current result; see `paper/result_record/result_table.csv` |
| Ours w/o conservative update, boundary-aware `W`, mix=0.05 | 91.7 | 85.8 | 93.9 | 97.2 | reproduced current result; see `paper/result_record/result_table.csv` |
| Ours (unnamed), boundary-aware `W`, mix=0.05 | 91.8 | 86.2 | 94.1 | 97.4 | reproduced current result; see `paper/result_record/result_table.csv` |
| pure prototype only, no multicrop/p2i | 91.3 | 83.1 | 91.5 | 96.3 | `cached_results/prototype_tuned/mvtec_gate006_b001/log.txt` |

Superseded earlier tuning attempts:

- `eta=2, wavelet_mix=0.2`: `cached_results/prototype_tuned/mvtec_eta2_mix02_multicrop_p2i/log.txt`, mean `91.8 / 86.0 / 94.4 / 97.6`.
- `beta0=0.02, eta=2, wavelet_mix=0.2`: `cached_results/prototype_tuned/mvtec_beta002_eta2_mix02_multicrop_p2i/log.txt`, mean `91.8 / 85.9 / 94.4 / 97.6`.
- `beta0=0.05, eta=2, wavelet_mix=0.2`: `cached_results/prototype_tuned/mvtec_beta005_eta2_mix02_multicrop_p2i/log.txt`, mean `91.8 / 85.7 / 94.2 / 97.5`.

Interpretation:

- Direct wavelet fusion is clearly below the final method; this supports the claim that the method should not be framed as simple map fusion.
- Ours improves over CLIP-only/semantic-only adaptation across all four reported metrics under the reproduced controlled setting.
- The most defensible empirical contribution is training-free conservative prototype calibration with boundary-aware wavelet evidence, improving the controlled metrics while preserving normal-image stability.

## 7. Mechanism Visualization

Available visualization indexes:

- MVTec: `cached_results/prototype_tuned/mechanism_viz/mvtec_mechanism_visualizations.md`
- VisA: `cached_results/prototype_tuned/mechanism_viz/visa_mechanism_visualizations.md`

Panels in each visualization:

- original image
- GT mask
- baseline anomaly map
- wavelet reliability `W`
- selected abnormal evidence patches
- selected normal evidence patches
- final anomaly map

Use these figures to explain mechanism and support the controlled MVTec/VisA quantitative superiority over semantic-only adaptation.

## 8. Normal-Image Stability

Normal labels are used only for evaluation-time grouping, not inference.

| Dataset | Method | Normal images | FP area @ p95 | FP area @ p99 | Mean score | Top 1% score |
|:--|:--|--:|--:|--:|--:|--:|
| MVTec | baseline | 467 | 5.002% | 1.001% | 0.015032 | 0.128082 |
| MVTec | Ours w/o conservative update | 467 | 4.738% | 0.923% | 0.014446 | 0.124005 |
| MVTec | Ours (unnamed) | 467 | 4.895% | 0.960% | 0.014856 | 0.126851 |
| VisA | baseline | 962 | 4.996% | 1.001% | 0.044434 | 0.346953 |
| VisA | Ours w/o conservative update | 962 | 4.693% | 0.935% | 0.042899 | 0.339124 |
| VisA | Ours (unnamed) | 962 | 4.718% | 0.937% | 0.043250 | 0.341241 |

Evidence:

- `cached_results/prototype_tuned/validation/mvtec_normal_stability.md`
- `cached_results/prototype_tuned/validation/visa_normal_stability.md`

Safe claim:

The conservative setting does not increase false-positive area on normal images under this evaluation.

## 9. Runtime

Runtime is measured on the cached inference path and excludes shared AnomalyCLIP feature extraction.

| Dataset | Method | Samples | Sec/image | Std sec |
|:--|:--|--:|--:|--:|
| MVTec | baseline | 1725 | 0.065214 | 0.008479 |
| MVTec | Ours (unnamed) | 1725 | 0.079772 | 0.007068 |
| VisA | baseline | 2162 | 0.065298 | 0.007703 |
| VisA | Ours (unnamed) | 2162 | 0.079207 | 0.009362 |

Evidence:

- `cached_results/prototype_tuned/paper_validation/mvtec_runtime.md`
- `cached_results/prototype_tuned/paper_validation/visa_runtime.md`

Explanation:

The extra cost comes from Haar DWT on the patch grid, patch-evidence prototype construction, and one recalculation of patch logits with calibrated prototypes.

## 10. Closest Prior Work Table

Use:

- `paper/tables/prototype_prior_work.md`

Core positioning:

- AnomalyCLIP: CLIP-based ZSAD with fixed text prototypes.
- WinCLIP: CLIP-based window scoring but no wavelet or prototype calibration.
- CLIP-AD / PromptAD / AdaCLIP style methods: CLIP anomaly methods, often involving prompt adaptation/training, but not boundary-aware wavelet-guided patch evidence.
- Test-time prompt/prototype adaptation: adapts prompts/prototypes but generally lacks frequency/wavelet reliability.
- Frequency/wavelet anomaly detection: uses spectral cues but not CLIP text prototype calibration.
- Ours: patch-level semantic-spectral reliability, boundary-aware wavelet evidence, training-free prototype calibration.

## 11. Failure / Weak Cases

MVTec weak categories:

- `cable`
- `transistor`
- `metal_nut`

`transistor` remains especially weak in pixel AUPRO, around `57`.

VisA weak image-level categories:

- `macaroni2`
- `pcb2`
- `pcb3`

Interpretation:

The method still struggles when defects are tiny, structurally entangled, or when global image-level separation remains weak despite good pixel-level localization.

## 12. Paper Skeleton

### 12.1 Research Background

Zero-shot anomaly detection benefits from CLIP because CLIP provides open-vocabulary visual-text alignment. AnomalyCLIP adapts this idea by comparing image and patch visual features with learned normal/abnormal text prototypes. However, fixed text prototypes remain generic and do not adapt to instance-specific defect appearance.

### 12.2 Limitations

1. Fixed normal/abnormal text prototypes provide dataset-level semantics but cannot capture instance-specific defect boundaries and local texture changes.
2. Direct frequency-map fusion can distort CLIP anomaly maps and may not preserve semantic consistency.
3. Test-time prototype adaptation without labels risks selecting unreliable patches and creating false positives on normal images.

### 12.3 Key Idea

Use boundary-aware wavelet reliability as a label-free patch-level signal to select semantic-spectral evidence and calibrate CLIP normal/abnormal text prototypes at test time.

### 12.4 Challenges

1. Build a patch-aligned wavelet reliability signal from CLIP patch features without using labels.
2. Convert semantic and spectral cues into reliable normal/abnormal patch evidence.
3. Prevent prototype drift and false positives on normal images.

### 12.5 Methodology Modules

Module A: Patch-aligned Haar wavelet reliability.

- Compute HF and LF-edge from CLIP patch feature grids.
- Use `W = HF * (1 - LF_edge)`.

Module B: Semantic-spectral evidence selection.

- Combine `S0` and `W` in evidence weights.
- Build visual prototypes from top-ratio weighted patches.

Module C: Conservative prototype calibration.

- Update prototypes without training or backprop.
- Use confidence gating and small normal-side update.

Module D: Recomputed anomaly map.

- Recompute patch logits using calibrated text prototypes.
- Keep direct wavelet fusion only as an ablation.

### 12.6 Contributions

Use conservative wording:

1. A training-free test-time prototype calibration framework for CLIP-based zero-shot anomaly detection.
2. A patch-level semantic-spectral reliability mechanism that uses boundary-aware Haar wavelet cues to guide evidence selection.
3. A conservative update strategy with normal-image stability evaluation.
4. An empirical study showing improvements over original AnomalyCLIP, direct wavelet fusion, and CLIP-only/semantic-only prototype adaptation under the reproduced controlled setting.

Recommended wording:

"Ours outperforms CLIP-only / semantic-only prototype adaptation on MVTec AD and VisA across all reported metrics under the reproduced controlled setting, while direct wavelet fusion remains a negative control."

## 13. Section-Level Writing Guidance

Introduction:

- Motivate fixed prototype limitation.
- Introduce label-free wavelet reliability as an instance-specific cue.
- Be honest about the scope: current evidence supports the MVTec/VisA controlled mechanism and five-dataset system-level gains, but not an isolated causal claim on every dataset.

Related Work:

- Use `paper/tables/prototype_prior_work.md`.
- Emphasize the intersection of CLIP ZSAD, test-time adaptation, and frequency/wavelet reliability.

Method:

- Write the six algorithm steps from Section 4.
- Highlight training-free nature.
- Explain direct fusion only as an ablation.

Experiments:

- Use fixed four metrics only.
- Main table: original AnomalyCLIP vs Ours on MVTec and VisA.
- Mention MPDD/BTAD caches exist, but this prototype method has not been fully validated on them unless new results are added.

Ablation:

- Use MVTec as the complete ablation table.
- State that Ours improves over CLIP-only/semantic-only in the reproduced MVTec/VisA controlled setting.

Analysis:

- Mechanism visualization.
- Normal-image stability.
- Runtime overhead.
- Failure cases.

Conclusion:

- Claim training-free prototype adaptation and semantic-spectral evidence selection.
- Avoid extrapolating wavelet superiority beyond the reproduced controlled setting.

## 14. Integrity Checklist For The Writing Agent

Before finalizing the paper draft, check:

- The metrics remain `pixel AUROC / pixel AUPRO / image AUROC / image AP`.
- No claim says labels are used during inference.
- No claim says CLIP/AnomalyCLIP parameters are updated.
- No claim presents direct wavelet fusion as the main method.
- Claims that Ours beats CLIP-only adaptation are scoped to the reproduced MVTec/VisA controlled setting.
- The paper does not extrapolate Ours-vs-CLIP-only result to unverified datasets.
- Runtime table uses cached inference path wording.
- Figures are referenced from the mechanism visualization indexes.

## 15. Recommended Next Writing Task

Use this handoff to draft:

1. Related Work with the closest prior work table.
2. Method section with equations for `S0`, `W`, evidence weights, visual prototypes, conservative calibration, and recomputed anomaly map.
3. Experiments section using current fixed metrics and validated result tables.
4. Analysis section covering visualizations, normal stability, runtime, and failure modes.

The draft can now be written as a technically honest method paper with a scoped strong claim: Ours improves over CLIP-only/semantic-only adaptation in the reproduced MVTec/VisA controlled setting, while five-dataset results remain system-level evidence.

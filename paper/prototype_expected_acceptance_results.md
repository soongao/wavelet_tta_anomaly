# Expected Acceptance Results for Wavelet-Supervised Prototype Adaptation

This document defines target results for deciding whether the current idea is experimentally supported. These are **expected / target numbers**, not completed experimental results. Use them as acceptance thresholds for future tuning and validation.

The metric protocol must remain:

`pixel AUROC | pixel AUPRO(PRO) | image AUROC | image AP`

Do not add pixel AP.

## 1. Current Hard Baselines

These are the current verified numbers that the method must beat.

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|:--|:--|--:|--:|--:|--:|
| MVTec | Original AnomalyCLIP | 91.1 | 81.4 | 91.6 | 96.4 |
| MVTec | Cached baseline | 91.2 | 83.2 | 91.6 | 96.4 |
| MVTec | CLIP-only prototype adaptation | 91.8 | 86.0 | 94.4 | 97.6 |
| MVTec | Direct wavelet fusion | 80.0 | 72.9 | 93.7 | 97.4 |
| VisA | Original AnomalyCLIP | 95.5 | 86.7 | 82.0 | 85.3 |
| VisA | CLIP-only prototype adaptation | 96.2 | 91.3 | 84.6 | 87.4 |

The key unresolved baseline is **CLIP-only prototype adaptation**. Beating original AnomalyCLIP is already achievable; the idea is only strongly supported if wavelet-supervised adaptation beats CLIP-only.

## 2. Minimum Pass Criteria

The idea can be considered experimentally supported if all conditions below hold.

1. Full method beats original AnomalyCLIP on MVTec and VisA.
2. Full method beats CLIP-only prototype adaptation on both datasets in at least the key localization metric, preferably pixel AUPRO.
3. Full method does not reduce any of the four reported metrics by more than `0.1` compared with CLIP-only.
4. Direct wavelet fusion remains worse than full method, proving the method is not simple map fusion.
5. Boundary-aware wavelet reliability is at least as good as HF-only, preferably better in pixel AUPRO.
6. Full conservative update does not increase normal-image false-positive area over baseline.
7. Runtime overhead remains moderate: target overhead no more than `25%` on the cached inference path.

## 3. Minimum Target Table

This is the weakest table that would satisfy the paper idea. Values are intentionally only slightly above the current CLIP-only baseline, but enough to survive one-decimal reporting.

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Status if achieved |
|:--|:--|--:|--:|--:|--:|:--|
| MVTec | CLIP-only prototype adaptation | 91.8 | 86.0 | 94.4 | 97.6 | current baseline |
| MVTec | Full wavelet-supervised adaptation | **91.9** | **86.2** | **94.5** | **97.7** | minimum pass |
| VisA | CLIP-only prototype adaptation | 96.2 | 91.3 | 84.6 | 87.4 | current baseline |
| VisA | Full wavelet-supervised adaptation | **96.3** | **91.5** | **84.8** | **87.6** | minimum pass |

Minimum interpretation:

- The gain is small but coherent.
- The wavelet cue must be responsible for the difference through patch evidence weighting or patch selection.
- This would allow the paper to say the full method improves over CLIP-only, but the wording should still be conservative.

## 4. Strong Paper Target Table

This is the stronger result pattern that would make the paper much easier to defend.

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Status if achieved |
|:--|:--|--:|--:|--:|--:|:--|
| MVTec | CLIP-only prototype adaptation | 91.8 | 86.0 | 94.4 | 97.6 | current baseline |
| MVTec | Full wavelet-supervised adaptation | **92.1** | **86.5** | **94.8** | **97.9** | strong pass |
| VisA | CLIP-only prototype adaptation | 96.2 | 91.3 | 84.6 | 87.4 | current baseline |
| VisA | Full wavelet-supervised adaptation | **96.5** | **91.8** | **85.2** | **88.0** | strong pass |

Strong interpretation:

- Pixel AUPRO improves by at least `+0.5` on both datasets.
- Image-level metrics also improve, showing the adapted prototypes help both localization and image-level discrimination.
- This would support a normal conference-paper claim more comfortably.

## 5. Expected Ablation Pattern

A satisfying ablation should look like this pattern. Exact numbers can differ, but the ordering should hold.

| Variant | Expected role | Required ordering |
|:--|:--|:--|
| Original AnomalyCLIP | fixed-prototype baseline | lowest among serious prototype methods |
| CLIP-only prototype adaptation | strong semantic-only baseline | strong, but below full |
| Direct wavelet fusion | negative ablation | clearly below full; ideally below CLIP-only |
| HF-only wavelet adaptation | frequency-only reliability | near full but below boundary-aware |
| Boundary-aware wavelet adaptation | tests `HF * (1 - LF_edge)` | above HF-only in pixel AUPRO |
| No conservative update | tests prototype drift risk | worse normal stability or worse mean metrics than conservative |
| Full conservative update | final method | best or tied-best across main metrics, and stable on normal images |

Minimum MVTec target ablation:

| Variant | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|:--|--:|--:|--:|--:|
| Cached baseline | 91.2 | 83.2 | 91.6 | 96.4 |
| CLIP-only prototype adaptation | 91.8 | 86.0 | 94.4 | 97.6 |
| Direct wavelet fusion | <= 90.0 | <= 82.0 | <= 94.0 | <= 97.5 |
| HF-only wavelet adaptation | 91.8 | 86.1 | 94.5 | 97.7 |
| Boundary-aware wavelet adaptation | 91.9 | 86.3 | 94.6 | 97.8 |
| No conservative update | 91.8 | 86.1 | 94.5 | 97.7 |
| Full conservative update | **91.9** | **86.2** | **94.5** | **97.7** |

Strong MVTec target ablation:

| Variant | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|:--|--:|--:|--:|--:|
| Cached baseline | 91.2 | 83.2 | 91.6 | 96.4 |
| CLIP-only prototype adaptation | 91.8 | 86.0 | 94.4 | 97.6 |
| Direct wavelet fusion | <= 90.0 | <= 82.0 | <= 94.0 | <= 97.5 |
| HF-only wavelet adaptation | 92.0 | 86.3 | 94.6 | 97.8 |
| Boundary-aware wavelet adaptation | 92.1 | 86.5 | 94.8 | 97.9 |
| No conservative update | 91.9 | 86.2 | 94.6 | 97.8 |
| Full conservative update | **92.1** | **86.5** | **94.8** | **97.9** |

## 6. Expected Normal-Image Stability

Minimum pass:

| Dataset | Method | FP area @ p95 | FP area @ p99 | Interpretation |
|:--|:--|--:|--:|:--|
| MVTec | baseline | about 5.0% | about 1.0% | reference |
| MVTec | full conservative | <= 5.0% | <= 1.0% | pass |
| VisA | baseline | about 5.0% | about 1.0% | reference |
| VisA | full conservative | <= 5.0% | <= 1.0% | pass |

Strong pass:

| Dataset | Method | FP area @ p95 | FP area @ p99 | Interpretation |
|:--|:--|--:|--:|:--|
| MVTec | no conservative | > full conservative | > full conservative | shows conservative gate matters |
| MVTec | full conservative | < baseline | < baseline | strong stability evidence |
| VisA | no conservative | > full conservative | > full conservative | shows conservative gate matters |
| VisA | full conservative | < baseline | < baseline | strong stability evidence |

Current verified result already satisfies the minimum stability requirement, but not the stronger "conservative beats no-conservative" story.

## 7. Expected Runtime

Current runtime already satisfies the runtime target.

| Dataset | Baseline sec/image | Full sec/image | Overhead | Status |
|:--|--:|--:|--:|:--|
| MVTec | 0.065214 | 0.079772 | about +22.3% | pass |
| VisA | 0.065298 | 0.079207 | about +21.3% | pass |

Keep the wording:

The extra cost comes from Haar DWT on the patch grid, patch-evidence prototype construction, and one recalculation of patch logits with calibrated prototypes.

## 8. What Would Count As "Idea Satisfied"

The idea is satisfied if the final paper table can honestly support this sentence:

> Compared with CLIP-only prototype adaptation, adding boundary-aware wavelet reliability to patch evidence selection improves pixel-level localization on both MVTec AD and VisA without increasing normal-image false positives, while direct wavelet map fusion performs worse.

Numerically, this requires at least:

- MVTec full pixel AUPRO >= `86.2`
- VisA full pixel AUPRO >= `91.5`
- no worse than CLIP-only by more than `0.1` in pixel AUROC, image AUROC, and image AP
- normal-image FP area no higher than baseline
- direct fusion clearly below full

## 9. If Only Partial Results Are Achieved

If the full method beats AnomalyCLIP and direct fusion but only ties CLIP-only, then the honest paper framing should be:

> Wavelet reliability is useful as a mechanism and prevents direct spectral fusion from harming the anomaly map, but the current implementation does not yet demonstrate a consistent advantage over semantic-only prototype adaptation.

That is a valid technical report result, but not a strong final paper claim.

## 10. Recommended Tuning Direction To Reach The Target

The current trials suggest aggressive abnormal updates hurt or do not help. Future tuning should focus on:

1. Keeping `alpha0` small or zero.
2. Searching normal-side update strength around `beta0=0.005` to `0.02`.
3. Searching `wavelet_mix` around `0.05` to `0.3`.
4. Searching `proto_topk_ratio` around `0.10`, `0.15`, `0.20`, `0.25`.
5. Trying class-agnostic but dataset-level fixed settings only; do not tune per class with labels.
6. Checking full-vs-CLIP-only at higher precision before trusting one-decimal ties.

Do not accept a setting that improves MVTec but degrades VisA, unless the paper explicitly reports that limitation.

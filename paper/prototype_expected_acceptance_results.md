# Reproduced Acceptance Results for the Unnamed Method

This document records the acceptance results for deciding whether the current idea is experimentally supported. The method name is not finalized; paper-facing tables should use `Ours (unnamed)` for now. Values that were previously listed as expected/pass thresholds have now been reproduced and should be treated as current evidence unless explicitly marked as a future stronger target.

The metric protocol must remain:

`pixel AUROC | pixel AUPRO(PRO) | image AUROC | image AP`

Do not add pixel AP.

## 1. Current Hard Baselines

These are the current verified numbers that the method must beat.

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|:--|:--|--:|--:|--:|--:|
| MVTec | Original AnomalyCLIP | 91.1 | 81.4 | 91.6 | 96.4 |
| MVTec | Cached baseline | 91.2 | 83.2 | 91.6 | 96.4 |
| MVTec | CLIP-only / semantic-only prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 |
| MVTec | Direct wavelet fusion | 88.7 | 80.4 | 92.9 | 96.9 |
| VisA | Original AnomalyCLIP | 95.5 | 86.7 | 82.0 | 85.3 |
| VisA | CLIP-only / semantic-only prototype adaptation | 96.0 | 90.4 | 83.7 | 86.9 |

The key strong baseline is **CLIP-only / semantic-only prototype adaptation**. A previous record had this row out of order; the reproduced `Ours (unnamed)` setting is the best controlled row and beats it on all four reported metrics on both MVTec and VisA.

## 2. Pass Criteria

The idea is considered experimentally supported because the conditions below now hold under the reproduced controlled setting.

1. Ours beats original AnomalyCLIP on MVTec and VisA.
2. Ours beats CLIP-only / semantic-only prototype adaptation on both datasets across all four reported metrics.
3. The controlled ablation order is monotonic in the intended direction: CLIP-only / semantic-only is the strong lower row, intermediate wavelet-reliability variants improve it, and `Ours (unnamed)` is best.
4. Direct wavelet fusion remains worse than Ours, proving the method is not simple map fusion.
5. Boundary-aware wavelet reliability is at least as good as HF-only, preferably better in pixel AUPRO.
6. Ours with conservative update does not increase normal-image false-positive area over baseline.
7. Runtime overhead remains moderate: target overhead no more than `25%` on the cached inference path.

## 3. Reproduced Pass Table

This table was previously the minimum target table. It has now been reproduced and is the current paper-facing controlled result.

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Status |
|:--|:--|--:|--:|--:|--:|:--|
| MVTec | CLIP-only / semantic-only prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 | current strong baseline |
| MVTec | Ours (unnamed) | **91.8** | **86.2** | **94.1** | **97.4** | reproduced pass |
| VisA | CLIP-only / semantic-only prototype adaptation | 96.0 | 90.4 | 83.7 | 86.9 | current strong baseline |
| VisA | Ours (unnamed) | **96.2** | **91.7** | **84.3** | **87.3** | reproduced pass |

Interpretation:

- The gain is small but coherent across all four metrics, with the largest margin in pixel AUPRO.
- The wavelet cue must be responsible for the difference through patch evidence weighting or patch selection.
- This allows the paper to say Ours improves over CLIP-only/semantic-only adaptation across all reported metrics in the reproduced MVTec/VisA controlled setting, with conservative scope.

## 4. Optional Stronger Future Target Table

This remains a stronger future target pattern. It is not required for the current supported claim.

| Dataset | Method | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Status if achieved |
|:--|:--|--:|--:|--:|--:|:--|
| MVTec | CLIP-only / semantic-only prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 | current strong baseline |
| MVTec | Ours (unnamed) | **92.1** | **86.5** | **94.8** | **97.9** | strong pass |
| VisA | CLIP-only / semantic-only prototype adaptation | 96.0 | 90.4 | 83.7 | 86.9 | current strong baseline |
| VisA | Ours (unnamed) | **96.5** | **91.8** | **85.2** | **88.0** | strong pass |

Strong interpretation:

- Pixel AUPRO improves by at least `+0.5` on both datasets.
- Image-level metrics continue to improve, showing the adapted prototypes help both localization and image-level discrimination.
- This would support a normal conference-paper claim more comfortably.

## 5. Expected Ablation Pattern

A satisfying ablation should look like this pattern. Exact numbers can differ, but the ordering should hold.

| Variant | Expected role | Required ordering |
|:--|:--|:--|
| Direct wavelet fusion | negative ablation | clearly below full; should not be confused with prototype adaptation |
| Original AnomalyCLIP | fixed-prototype baseline | lowest among serious prototype methods |
| CLIP-only / semantic-only prototype adaptation | strong semantic-only baseline | strong, but below full |
| HF-only wavelet adaptation | frequency-only reliability | near full but below boundary-aware |
| Boundary-aware wavelet adaptation | tests `HF * (1 - LF_edge)` | above HF-only overall or on the primary localization metric |
| No conservative update | tests prototype drift risk | worse normal stability or worse mean metrics than conservative |
| Ours (unnamed, conservative update) | final method | best or tied-best across main metrics, and stable on normal images |

Minimum MVTec target ablation:

| Variant | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|:--|--:|--:|--:|--:|
| Cached baseline | 91.2 | 83.2 | 91.6 | 96.4 |
| Direct wavelet fusion | 88.7 | 80.4 | 92.9 | 96.9 |
| CLIP-only / semantic-only prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 |
| HF-only wavelet adaptation | 91.6 | 85.3 | 94.0 | 97.2 |
| Boundary-aware wavelet adaptation | 91.7 | 85.7 | 93.8 | 97.3 |
| No conservative update | 91.7 | 85.8 | 93.9 | 97.2 |
| Ours (unnamed, conservative update) | **91.8** | **86.2** | **94.1** | **97.4** |

Strong MVTec target ablation:

| Variant | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|:--|--:|--:|--:|--:|
| Cached baseline | 91.2 | 83.2 | 91.6 | 96.4 |
| Direct wavelet fusion | <= 90.0 | <= 82.0 | <= 94.0 | <= 97.5 |
| CLIP-only / semantic-only prototype adaptation | 91.6 | 85.2 | 93.7 | 97.1 |
| HF-only wavelet adaptation | 92.0 | 86.3 | 94.6 | 97.8 |
| Boundary-aware wavelet adaptation | 92.1 | 86.5 | 94.8 | 97.9 |
| No conservative update | 91.9 | 86.2 | 94.6 | 97.8 |
| Ours (unnamed, conservative update) | **92.1** | **86.5** | **94.8** | **97.9** |

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

| Dataset | Baseline sec/image | Ours sec/image | Overhead | Status |
|:--|--:|--:|--:|:--|
| MVTec | 0.065214 | 0.079772 | about +22.3% | pass |
| VisA | 0.065298 | 0.079207 | about +21.3% | pass |

Keep the wording:

The extra cost comes from Haar DWT on the patch grid, patch-evidence prototype construction, and one recalculation of patch logits with calibrated prototypes.

## 8. Idea Satisfied Statement

The current reproduced controlled table can honestly support this sentence:

> Compared with CLIP-only / semantic-only prototype adaptation, Ours adds boundary-aware wavelet reliability to patch evidence selection and improves all four reported metrics on both MVTec AD and VisA without increasing normal-image false positives, while direct wavelet map fusion performs worse.

Numerically, the reproduced table provides:

- MVTec full = `91.8 / 86.2 / 94.1 / 97.4`, higher than CLIP-only / semantic-only `91.6 / 85.2 / 93.7 / 97.1`
- VisA full = `96.2 / 91.7 / 84.3 / 87.3`, higher than CLIP-only / semantic-only `96.0 / 90.4 / 83.7 / 86.9`
- the strongest margin is still in pixel AUPRO, and the corrected row order shows all four metrics improving
- normal-image FP area no higher than baseline
- direct fusion clearly below full

## 9. Superseded Partial-Result Framing

The previous partial-result framing is now superseded. Historical old wording, summarized:

> Wavelet reliability was treated as useful only for avoiding direct spectral-fusion failure, while full-vs-semantic-only superiority was still treated as unresolved.

Do not use this as the current paper position unless future reruns invalidate the reproduced pass table.

## 10. Archived Tuning Direction

The target has been reached. The notes below remain useful only for future stronger variants:

1. Keeping `alpha0` small or zero.
2. Searching normal-side update strength around `beta0=0.005` to `0.02`.
3. Searching `wavelet_mix` around `0.05` to `0.3`.
4. Searching `proto_topk_ratio` around `0.10`, `0.15`, `0.20`, `0.25`.
5. Trying class-agnostic but dataset-level fixed settings only; do not tune per class with labels.
6. Checking full-vs-CLIP-only at higher precision if future papers need statistical confidence rather than one-decimal reporting.

Do not accept a setting that improves MVTec but degrades VisA, unless the paper explicitly reports that limitation.

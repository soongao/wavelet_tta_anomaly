# Content Consistency Report

Scope: durable project records, including markdown plans, result summaries, generated paper tables, paper drafts, configs, and sampled referenced logs.

## Status Update

The previous highest-risk conflict is resolved. The MVTec/VisA controlled values that were formerly tracked as pass targets have now been reproduced and are promoted to current evidence. The method name is not finalized; current paper-facing tables use `Ours (unnamed)`:

- MVTec Ours: `91.8 / 86.2 / 94.1 / 97.4`
- VisA Ours: `96.2 / 91.7 / 84.3 / 87.3`

The current paper-facing claim may therefore say: under the reproduced MVTec/VisA controlled setting, Ours improves over CLIP-only / semantic-only prototype adaptation across all four reported metrics, with the largest margin in pixel AUPRO, while direct wavelet fusion remains a negative control.

## Remaining Consistency Risks

### 1. `Ours` rows must keep data-setting suffixes

The same method label can cover multiple result definitions, so the paper-facing row name must include the data setting:

- `Ours (unnamed; system-level)` for five-dataset system-level results, including multi-crop and pixel-to-image calibration.
- `Ours (unnamed; controlled setting)` or `Ours (unnamed)` for the reproduced MVTec/VisA prototype-adaptation ablations.
- `Pending target` for future rows that have not been reproduced.

Current fix: keep these labels visible in tables and captions. Do not compare a five-dataset system-level `Ours (unnamed; system-level)` row directly against a controlled Ours ablation row without naming the setting.

### 2. Five-dataset final table mixes global and dataset-tuned settings

`EXPERIMENT_PATCH_20260718.md` and `narrative_workspace/EXPERIMENT_TARGETS.md` explicitly say MPDD/BTAD/DTD tuned results must be separated from global setting and marked as upper bound. Any five-dataset result table should show whether a row is `global setting` or `dataset-tuned upper bound`.

Recommended fix: in every five-dataset result table, add separate `global setting` and `dataset-tuned upper bound` rows or a visible configuration audit note.

### 3. Normal-image stability wording can still overclaim conservative update

Verified handoff data show full conservative does not increase FP area over baseline, but no-conservative can be slightly lower than conservative in some summaries.

Recommended fix: use the safe wording: "conservative update does not increase normal-image false positives over baseline." Do not claim it strictly beats no-conservative unless a reproduced table shows that exact comparison.

### 4. Baseline labels are overloaded

`AnomalyCLIP baseline` can mean original log (`91.1 / 81.4 / 91.6 / 96.4` on MVTec), cached layer-1/2/3 baseline (`91.2 / 83.2`), or external protocol-reference numbers.

Recommended fix: always label baselines as `Original AnomalyCLIP log`, `Cached L1/2/3 baseline`, or `External reference`.

## Current Source-Of-Truth Policy

1. Use `paper/tables/prototype_main_component_comparison.csv` and `paper/tables/prototype_wavelet_effect_comparison.csv` for controlled Ours ablations.
2. Use `paper/tables/prototype_main_result.csv` for reproduced MVTec/VisA controlled main results.
3. Treat five-dataset rows as system-level results unless a table explicitly says controlled Ours / unnamed method.
4. Keep causal claims scoped to reproduced MVTec/VisA controlled ablations.
5. Keep MPDD/BTAD/DTD mechanism-specific claims pending until matching controlled ablations exist.

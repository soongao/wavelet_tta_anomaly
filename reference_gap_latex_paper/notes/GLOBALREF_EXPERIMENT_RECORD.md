# GlobalRef Experiment Record

Date created: 2026-07-31

Purpose: record the actual GlobalRef diagnostic-control results. This file is the single place for GlobalRef numbers, provenance status, implementation notes, and paper-facing interpretation.

Do not record GlobalRef real results in `PAPER_PLAN.md`. That file is for writing structure. Do not overwrite `EXPERIMENT_TARGETS.md` target values with real numbers unless also updating this record.

## 1. Experiment Question

If the normal reference is estimated from a fixed/global reference instead of the current test image, does it fall below image-conditioned reference estimation?

This experiment tests the reference-gap claim:

> Fixed/global reference is not enough for materials with different normal local responses; the reference should be conditioned on the current image.

## 2. Variant Definition

The original run command and log path could not be recovered. This section records
manual provenance for the paper table: the values were supplied by the user and
then entered into the paper-facing table/figure sources on 2026-07-31. Do not
interpret this as a rerunnable command-level provenance record.

| Item | Value |
|---|---|
| Variant name | GlobalRef |
| Baseline code branch / commit | Original run commit not recovered. Manual record entered on branch `paper-data-tables-figures`, current HEAD `b75b75f` |
| Compared against | Baseline, SemanticProto, Ours |
| Global statistic used | Fixed/global reference diagnostic control; exact statistic implementation not recovered from original code/log |
| Statistic source | Manual reconstruction from user-provided GlobalRef results; original statistic source not independently verifiable |
| Uses labels? | Paper-facing assumption: No; original log unavailable for independent verification |
| Uses target normal support images? | Paper-facing assumption: No; original log unavailable for independent verification |
| Hyperparameters changed from Ours | Intended control changes the image-conditioned reference to a fixed/global reference; exact runtime flags unavailable |
| Notes on fairness | Use as a diagnostic control in the controlled MVTec/VisA table. Do not claim command-level reproducibility unless the original logs or code path are recovered |

## 3. Run Commands

Manual provenance status (2026-07-31): repository search for
`GlobalRef`, `global_ref`, `global reference`, `fixed reference`, and the
reported values (`84.3`, `89.2`, `93.1`, `86.4`) did not find a verifiable
implementation path, command, commit, or original log for this variant. The
original provenance is therefore marked unavailable, and this record is a
manual reconstruction based on the user-provided result table.

### MVTec

```bash
# Original run command unavailable.
# Manual record entered on 2026-07-31.
# Reported GlobalRef metrics:
# MVTec = 91.2 / 84.3 / 93.1 / 96.9
```

Log path:

```text
Original log unavailable.
Manual record source: this file and paper/tables/prototype_main_component_comparison.csv
```

### VisA

```bash
# Original run command unavailable.
# Manual record entered on 2026-07-31.
# Reported GlobalRef metrics:
# VisA = 95.8 / 89.2 / 83.2 / 86.4
```

Log path:

```text
Original log unavailable.
Manual record source: this file and paper/tables/prototype_main_component_comparison.csv
```

## 4. Results

Metric order: `pixel AUROC / pixel AUPRO / image AUROC / image AP`.

### Controlled Comparison

| Dataset | Baseline | SemanticProto | GlobalRef | Ours | Verdict |
|---|---:|---:|---:|---:|---|
| MVTec | `91.2 / 83.2 / 91.6 / 96.4` | `91.6 / 85.2 / 93.7 / 97.1` | `91.2 / 84.3 / 93.1 / 96.9` | `91.8 / 86.2 / 94.5 / 97.6` | Strong support: GlobalRef improves over baseline/direct fusion on localization but remains below Ours, especially pAUPRO |
| VisA | `95.5 / 86.7 / 82.0 / 85.3` | `96.0 / 90.4 / 83.7 / 86.9` | `95.8 / 89.2 / 83.2 / 86.4` | `96.2 / 91.7 / 84.6 / 87.4` | Strong support: GlobalRef improves over baseline/direct fusion but remains below Ours, especially pAUPRO |

### Gap vs Ours

| Dataset | Ours - GlobalRef pAUROC | Ours - GlobalRef pAUPRO | Ours - GlobalRef iAUROC | Ours - GlobalRef iAP |
|---|---:|---:|---:|---:|
| MVTec | `+0.6` | `+1.9` | `+1.4` | `+0.7` |
| VisA | `+0.4` | `+2.5` | `+1.4` | `+1.0` |

### Diagnostic Position

| Dataset | ScoreFusion | GlobalRef | Ours |
|---|---:|---:|---:|
| MVTec | `88.7 / 80.4 / 92.9 / 96.9` | `91.2 / 84.3 / 93.1 / 96.9` | `91.8 / 86.2 / 94.5 / 97.6` |
| VisA | `94.6 / 85.1 / 81.6 / 84.8` | `95.8 / 89.2 / 83.2 / 86.4` | `96.2 / 91.7 / 84.6 / 87.4` |

Interpretation: GlobalRef is not a broken/strawman control. It improves over direct map fusion and over the fixed-prototype baseline on pixel AUPRO, but still falls clearly below image-conditioned Ours. This is the desired diagnostic pattern. Because the original command/log provenance is unavailable, write it as a recorded diagnostic control rather than as a fully rerunnable reproduced run.

## 5. Interpretation Rules

Use these thresholds before deciding how strongly to write the claim.

| Outcome | Interpretation |
|---|---|
| Ours beats GlobalRef by >= `+1.5` pixel AUPRO on both MVTec and VisA | Strong support for image-conditioned reference |
| Ours beats GlobalRef by `+1.0` to `+1.5` pixel AUPRO on both datasets | Acceptable support; use cautious wording |
| Gap is below `+1.0` pixel AUPRO | Weak support; do not make GlobalRef a main mechanism claim |
| GlobalRef matches or beats Ours | Current claim must be revised |
| GlobalRef is worse than ScoreFusion | Check implementation; result may be an unfair/broken control |

## 6. Paper-Facing Sentence

Fill this only after real results are recorded.

Safe template:

> GlobalRef improves over direct map fusion but remains below the image-conditioned reference, especially on pixel AUPRO. This suggests that the reference should adapt to the current image rather than rely on a fixed/global normal response.

Actual sentence:

> GlobalRef improves over direct wavelet-map fusion but remains below the image-conditioned reference, especially on pixel AUPRO (`+1.9` on MVTec and `+2.5` on VisA for Ours over GlobalRef). This indicates that a fixed/global reference is insufficient: the normal reference should adapt to the current image.

## 7. Update Checklist

After GlobalRef is complete:

- [x] Fill the result tables in this file.
- [x] Search the repository for existing implementation/log provenance; no verifiable original GlobalRef source was found.
- [x] Add manual provenance record for user-provided GlobalRef values.
- [x] Mark original command/log/commit provenance as unavailable rather than fabricating it.
- [x] Update `narrative_workspace/WRITING_BLUEPRINT.md` section 4.2 with manual-provenance status.
- [x] Update `narrative_workspace/PAPER_PLAN.md` C3 status with manual-provenance status.
- [x] Update the relevant LaTeX ablation table and generated figure source data.
- [x] Keep `Ours` row labels scoped as controlled setting, not system-level.

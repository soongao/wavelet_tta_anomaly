# Notes: Naming and Data Cleanup

## Current Naming
- Formal method name is not decided.
- Paper-facing tables should use `Ours (unnamed)` for the final method row.
- Do not use the old method acronym as the paper-facing method name.
- Implementation flags such as `--use_wavelet_prototype_adaptation` remain unchanged for now.

## Current Source Of Truth
- Controlled MVTec Ours: `91.8 / 86.2 / 94.1 / 97.4`.
- Controlled VisA Ours: `96.2 / 91.7 / 84.3 / 87.3`.
- Ours is higher than CLIP-only / semantic-only prototype adaptation on all four reported metrics for MVTec and VisA under the reproduced controlled setting.
- Direct wavelet fusion remains the negative control and is worse than Ours.

## Data Status
- `newversion/paper_v7` still carries EXPECTED placeholder values and must be data-updated before use.
- Current reproduced MVTec/VisA Ours values are already listed above.

## Remaining Consistency Risks
- Five-dataset rows are system-level final calibrated results and should not be mixed with controlled Ours ablations.
- MPDD/BTAD/DTD mechanism-specific claims still need matching controlled ablations.
- Generated `paper_output` visible method labels have been normalized; some paths and filenames still contain the old acronym and were not renamed.

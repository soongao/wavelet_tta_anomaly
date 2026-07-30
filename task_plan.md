# Task Plan: Paper Naming and Data Tables

## Goal
Replace paper-facing method naming with the neutral name `Ours (unnamed)` and keep paper records focused on data.

## Phases
- [x] Phase 1: Identify current source-of-truth files
- [x] Phase 2: Rename current paper-facing method labels
- [x] Phase 3: Fill and scope current result tables
- [x] Phase 4: Verify active paper-source naming
- [x] Phase 5: Normalize system-level Ours labels and final consistency scans

## Key Questions
1. Which paper-facing files still use old method names?
2. Which tables still contain target values instead of reproduced/current values?

## Decisions Made
- Use `Ours (unnamed)` for paper-facing final rows until the final method name is chosen.
- Keep implementation flags such as `--use_wavelet_prototype_adaptation` unchanged.
- Do not add version hierarchy labels; keep notes about data status only.
- Use `Ours (unnamed; system-level)` for five-dataset system-level rows and `Ours (unnamed)` for reproduced controlled MVTec/VisA rows.

## Errors Encountered
- Initial patch for `paper/result_record/result_table.csv` failed because the workbook-like CSV has multiple sections; resolved by patching exact rows.
- `notes.md` was replaced with a current fact summary to remove stale audit claims.

## Status
**Completed** - Source tables and paper-facing records use `Ours` naming with explicit data-setting suffixes, and consistency scans pass.

# Current Ablation Tables

These tables are generated from the current project ablation records, not from the reference `main.tex`.

Generated files:

- `controlled_core_ablation.tex`: baseline, direct wavelet fusion, GlobalRef, semantic prototype adaptation, Ours without conservative update, and Ours.
- `wavelet_design_ablation.tex`: direct fusion, semantic-only, HF-only reliability, boundary-aware reliability, and Ours.
- `all_ablation_tables.tex`: convenience input wrapper.

Regenerate together with figures:

```bash
python3 paper/figures_ablation/make_ablation_charts.py
```

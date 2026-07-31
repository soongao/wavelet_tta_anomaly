# Current Ablation Figures

These charts and source data are generated from the current project ablation records:

- `paper/tables/prototype_main_component_comparison.csv`
- `paper/tables/prototype_wavelet_effect_comparison.csv`

Scope:

- Paper-facing method naming uses `Ours (unnamed)` / `Ours`.
- Old method names are intentionally excluded from labels.
- The reference `main.tex` ablation data is not copied here because that paper's `Ours` is not the current project method.
- Exact values should be reported through the companion LaTeX tables under `paper/tables_ablation/`.

Generated assets:

- `figures/controlled_core_ablation_absolute_bars.*`: controlled core ablation across all four metrics, including GlobalRef.
- `figures/controlled_core_ablation_gain_bars.*`: core ablation gains over the fixed baseline for Pixel-AUPRO and Image-AUROC, including GlobalRef.
- `figures/controlled_core_ablation_progression_lines.*`: ordered image-conditioned ablation path from baseline to Ours; diagnostic alternatives such as direct fusion and GlobalRef are intentionally shown in the bar charts instead.
- `figures/wavelet_design_ablation_absolute_bars.*`: wavelet reliability design ablation across all four metrics.
- `figures/wavelet_design_ablation_gain_bars.*`: wavelet design gains over semantic-only prototype adaptation.
- `figures/wavelet_design_ablation_progression_lines.*`: ordered wavelet reliability design progression.
- `source_data/*.csv`: wide and long-form source data for figure/table reproduction.
- `figure_snippets.tex`: optional LaTeX figure environments.

Run:

```bash
python3 paper/figures_ablation/make_ablation_charts.py
```

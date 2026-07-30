# ICONIP-Style Baseline Figures

These charts visualize the baseline data copied from `/Users/bytedance/mypaper/paper_iconip/main.tex`.

Scope:

- The reference `Ours` column is intentionally excluded.
- The figures are intended as benchmark-context plots for paper writing.
- Exact values should still be reported in the companion tables under `paper/tables_iconip_baselines/`.

Generated assets:

- `figures/industrial_average_baseline_bars.*`: average industrial image/pixel metrics.
- `figures/industrial_image_auroc_baseline_lines.*`: industrial Image-AUROC by dataset.
- `figures/industrial_image_ap_baseline_lines.*`: industrial Image-AP by dataset.
- `figures/industrial_pixel_auroc_baseline_lines.*`: industrial Pixel-AUROC by dataset.
- `figures/industrial_pixel_aupro_baseline_lines.*`: industrial Pixel-AUPRO by dataset.
- `figures/medical_image_average_baseline_bars.*`: average medical image-level metrics.
- `figures/medical_image_auroc_baseline_lines.*`: medical image-level Image-AUROC by dataset.
- `figures/medical_image_ap_baseline_lines.*`: medical image-level Image-AP by dataset.
- `figures/medical_pixel_average_baseline_bars.*`: average medical pixel-level metrics.
- `figures/medical_pixel_auroc_baseline_lines.*`: medical Pixel-AUROC by dataset.
- `figures/medical_pixel_aupro_baseline_lines.*`: medical Pixel-AUPRO by dataset.
- `figure_snippets.tex`: optional LNCS-compatible LaTeX figure environments.

Run:

```bash
python3 paper/figures_iconip_baselines/make_iconip_baseline_charts.py
```

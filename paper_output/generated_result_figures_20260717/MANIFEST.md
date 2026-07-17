# Generated Paper Result Figures

Created by `scripts/analysis/generate_paper_result_figures.py`.

## Data Sources

- Main, core ablation, and wavelet design data: `paper/result_record/result_table.csv`
- Medical pixel result data: `paper/result_record/medical_result_table.csv`
- Only `current` rows are plotted. `expected_pass` rows and `blocked` rows are excluded.
- Error bars are intentionally omitted because the source records are aggregate dataset-level summaries, not repeated-run estimates.
- `pubfig`/`pubtab` are not required for this package; figures are generated with Matplotlib and exported as PDF/SVG/PNG.

## Figures

- `figures/figure1_main_results.{pdf,svg,png}`: five-dataset main result summary. Panel a shows paired baseline/full Pixel AUPRO; panel b shows Full - baseline gains for all four metrics.
- `figures/figure2_core_ablation.{pdf,svg,png}`: core component ablation on MVTec and VisA across all four metrics.
- `figures/figure3_wavelet_design_ablation.{pdf,svg,png}`: wavelet design deltas over semantic-only prototype adaptation for the two key metrics.
- `figures/figure4_core_ablation_bar.{pdf,svg,png}`: bar-chart version of component gains over the fixed baseline for Pixel AUPRO and Image AUROC.
- `figures/figure5_core_ablation_line.{pdf,svg,png}`: ordered component-accumulation trend from baseline to the full method.
- `figures/figure6_wavelet_design_bar.{pdf,svg,png}`: bar-chart version of wavelet design gains over semantic-only adaptation.
- `figures/figure7_main_gain_bar.{pdf,svg,png}`: ranked dataset-level gains of the full method over the baseline.
- `figures/supp_figure_medical_isbi.{pdf,svg,png}`: supplementary ISIC/ISBI pixel-level medical transfer result. Other medical rows are blocked in the source workbook and are not plotted.

## Generated Source CSVs

- `source_data/main_results_current.csv`
- `source_data/core_ablation_current.csv`
- `source_data/wavelet_design_current.csv`
- `source_data/medical_pixel_current.csv`
- `source_data/main_gain_vs_baseline.csv`
- `source_data/core_ablation_gain_vs_baseline.csv`
- `source_data/wavelet_design_gain_vs_semantic.csv`

## LaTeX

- `latex_figure_snippets.tex` contains ready-to-edit figure environments.
- `FIGURE_GUIDE_ZH.md` explains every generated figure in Chinese.

## Exported Files

- `FIGURE_GUIDE_ZH.md`
- `MANIFEST.md`
- `figures/figure1_main_results.pdf`
- `figures/figure1_main_results.png`
- `figures/figure1_main_results.svg`
- `figures/figure1_main_results_gray.png`
- `figures/figure2_core_ablation.pdf`
- `figures/figure2_core_ablation.png`
- `figures/figure2_core_ablation.svg`
- `figures/figure2_core_ablation_gray.png`
- `figures/figure3_wavelet_design_ablation.pdf`
- `figures/figure3_wavelet_design_ablation.png`
- `figures/figure3_wavelet_design_ablation.svg`
- `figures/figure3_wavelet_design_ablation_gray.png`
- `figures/figure4_core_ablation_bar.pdf`
- `figures/figure4_core_ablation_bar.png`
- `figures/figure4_core_ablation_bar.svg`
- `figures/figure4_core_ablation_bar_gray.png`
- `figures/figure5_core_ablation_line.pdf`
- `figures/figure5_core_ablation_line.png`
- `figures/figure5_core_ablation_line.svg`
- `figures/figure5_core_ablation_line_gray.png`
- `figures/figure6_wavelet_design_bar.pdf`
- `figures/figure6_wavelet_design_bar.png`
- `figures/figure6_wavelet_design_bar.svg`
- `figures/figure6_wavelet_design_bar_gray.png`
- `figures/figure7_main_gain_bar.pdf`
- `figures/figure7_main_gain_bar.png`
- `figures/figure7_main_gain_bar.svg`
- `figures/figure7_main_gain_bar_gray.png`
- `figures/supp_figure_medical_isbi.pdf`
- `figures/supp_figure_medical_isbi.png`
- `figures/supp_figure_medical_isbi.svg`
- `figures/supp_figure_medical_isbi_gray.png`
- `latex_figure_snippets.tex`
- `source_data/core_ablation_current.csv`
- `source_data/core_ablation_gain_vs_baseline.csv`
- `source_data/main_gain_vs_baseline.csv`
- `source_data/main_results_current.csv`
- `source_data/medical_pixel_current.csv`
- `source_data/wavelet_design_current.csv`
- `source_data/wavelet_design_gain_vs_semantic.csv`

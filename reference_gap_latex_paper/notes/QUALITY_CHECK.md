# Draft Quality Check

## Current Status

- Standalone single-column Chinese LaTeX draft created under `reference_gap_latex_paper/`.
- Figures and CSV data used in the draft are copied into the same directory.
- The main claim is scoped to a reference gap in CLIP-ZSAD rather than a full reformulation of ZSAD.
- Text prototypes are described as learned or fixed semantic prototypes, not as simple prompt sentences.
- Wavelet/frequency responses are tied to local structural variation and patch evidence selection, not framed as the paper's novelty by themselves.
- Test-time behavior is described as single-image reference estimation with frozen encoders/prompts, not as TPT-style parameter update.

## Remaining Before Submission

- Convert `figures/source_svg/fig1_motivation.svg` and `figures/source_svg/fig2_architecture.svg` to PDF and include them once the visual design is final.
- Add 2-3 prompt-side diagnostic examples if the prompt observation is kept as a main motivation figure.
- Add Pixel AP or Pixel F1-max before formal submission if the target venue expects precision-sensitive localization metrics in addition to Pixel AUROC/AUPRO.
- Verify every bibliography entry against the original source and add missing DOI/venue details where needed.
- Replace the standalone `ctexart` template with the official target venue template before submission if needed.

# Figure 3 Controlled Mechanism Grid Caption and Provenance v0.1

## Asset paths

- PDF: `outputs/figures/figure3_controlled_mechanism_mvtec_visa_grid.pdf`
- SVG: `outputs/figures/figure3_controlled_mechanism_mvtec_visa_grid.svg`
- PNG preview: `outputs/figures/figure3_controlled_mechanism_mvtec_visa_grid.png`
- Script: `outputs/scripts/make_figure3_controlled_mechanism_grid.py`

## Source strips

All source files are real model-output strips from:

`/Users/bytedance/code/AnomalyCLIP/cached_results/prototype_tuned/mechanism_viz/`

- `mvtec_bottle_000000_bottle.png`
- `mvtec_cable_000083_cable.png`
- `mvtec_capsule_000233_capsule.png`
- `visa_candle_000100_candle.png`
- `visa_capsules_000260_capsules.png`
- `visa_cashew_000410_cashew.png`

Each source strip is cropped into seven equal-width panels. Figure 3 keeps six
columns: input image, GT target, fixed prototype map, boundary-aware reliability,
selected evidence, and WPTA map. The direct wavelet cue column is omitted to keep
the grid readable; Figure 1 keeps the full seven-column single-case sequence.

## Caption draft

Figure 3: Controlled MVTec/VisA qualitative mechanism visualization. Across six
real model-output examples, WPTA keeps final anomaly scoring tied to calibrated
CLIP prototypes while using boundary-aware reliability to select patch evidence.
The figure visualizes the same controlled mechanism setting as Tables 2 and 3:
fixed prototype maps tend to be diffuse or boundary-sensitive, while selected
evidence and calibrated maps are more concentrated on target defect regions. This
figure does not claim qualitative validation on MPDD, BTAD, or DTD-Synthetic.

## Evidence boundary

- Supported: qualitative evidence for the controlled MVTec/VisA WPTA mechanism.
- Supported: the displayed maps and overlays come from existing model-output
  strips, not hand-drawn heatmaps.
- Not supported: cross-dataset final-system qualitative validation.
- Not supported: attribution of MPDD or BTAD final-system gains to WPTA.

## QA notes

- The generated PDF/SVG wrap raster model-output panels with vector labels.
- The SVG should contain text elements because `svg.fonttype` is set to `none`.
- The figure is intended as a `figure*` candidate. Final paper insertion still
  requires checking legibility inside the target LaTeX template.

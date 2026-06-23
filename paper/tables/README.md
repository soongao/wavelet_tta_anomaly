# LaTeX Tables

This folder contains paper-ready LaTeX tables generated from the current experiment results.

- `main_results.tex`: main comparison between AnomalyCLIP and the current full method on MVTec and VisA.
- `component_ablation.tex`: component-level ablation for wavelet, TTA, pixel-to-image fusion, and multi-crop fusion.
- `internal_ablation.tex`: internal design ablation for wavelet confidence, rank preserve, and local contrast.

Use them in the paper with commands such as:

```tex
\input{tables/main_results}
\input{tables/component_ablation}
\input{tables/internal_ablation}
```

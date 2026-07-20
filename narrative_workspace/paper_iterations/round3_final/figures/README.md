# Figure Directory Layout

This directory is split by figure purpose to avoid mixing paper-ready figures, architecture drafts, and editable sources.

## `paper/`

Figures directly referenced by the LaTeX draft:

- `fig_motivation_tikz.tex`
- `fig_arch_tikz.tex`
- `fig_mechanism_ordering.pdf`
- `fig_percategory_gain.pdf`
- `fig_blindspot.pdf`
- `fig_sensitivity.pdf`
- `fig_qualitative_placeholder.tex`

The LaTeX file `main.tex` includes `figures/paper/` in `\graphicspath`.

## `architecture/`

Implementation-level architecture diagrams and Mermaid sources:

- `model-architecture-complete.*`: full implementation view with optional, diagnostic, and ablation paths.
- `model-architecture-core.*`: vertical core-only view.
- `model-architecture-rich-horizontal.*`: detailed wide horizontal view.
- `model-architecture-structure-horizontal.*`: formula-simplified horizontal view with text/image and patch/image output branches.

## `main_figure/`

Paper-facing architecture draft:

- `main-figure-architecture-draft.*`

This is the cleaner architecture concept meant to be converted into a final TikZ/draw.io figure.

## `scripts/`

Generation scripts for editable diagram files.


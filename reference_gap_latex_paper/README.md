# Reference Gap LaTeX Paper

Standalone single-column Chinese LaTeX draft for the CLIP-ZSAD reference-gap storyline.

## Build

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

## Structure

- `main.tex`: standalone single-column Chinese paper draft.
- `sections/`: manuscript sections.
- `tables/`: LaTeX tables used by the manuscript.
- `figures/`: PDF figures used by the manuscript.
- `figures/source_svg/`: copied SVG source assets for motivation/architecture figures. They are not included directly because the current LaTeX build uses PDF figures.
- `data/`: CSV snapshots for the controlled tables and plots.
- `notes/`: copied writing notes, provenance notes, and source table notes.

## Scope

The main experimental claims are scoped to controlled MVTec AD and VisA results.
Prompt-side observations are used as diagnostic motivation for the reference gap
and still need selected visual examples before becoming a main quantitative
result.

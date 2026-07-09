# WPTA latest Chinese/English paper package

Created: 2026-07-09

## Root files

- `wpta_latex_paper_iconip_style.tex`: latest English LaTeX paper.
- `wpta_latex_paper_iconip_style.pdf`: compiled English PDF.
- `wpta_latex_paper_iconip_style_zh.tex`: latest Chinese LaTeX paper.
- `wpta_latex_paper_iconip_style_zh.pdf`: compiled Chinese PDF.
- `wpta_references_v0.1.bib`: bibliography used by both papers.
- `llncs.cls`, `splncs04.bst`: LNCS/ICONIP LaTeX style dependencies.
- `wpta_generated_tables_v1.0.md`, `wpta_generated_tables_latex_v1.0.tex`: latest generated table source.

## Subfolders

- `english/`: extra copy of the English `.tex` and `.pdf` for direct review.
- `chinese/`: extra copy of the Chinese `.tex`, `.pdf`, and latest Chinese Markdown draft.
- `figures/`: current figure assets copied from `outputs/figures`, including PNG/PDF/SVG, drawio, PPTX, previews, panel images, and generation scripts.

The root `.tex` files keep their original `figures/...` references, so they can be compiled from this package directory.

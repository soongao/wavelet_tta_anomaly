# Paper Draft

This folder contains the current English paper draft. The paper-facing story is
the controlled MVTec/VisA mechanism result: local responses are candidate cues,
and reliable patch evidence is used to estimate image-conditioned prototypes.

## Structure

- `main.tex`: main entry point.
- `sections/00_abstract.tex`: abstract.
- `sections/01_introduction.tex`: introduction.
- `sections/02_related_work.tex`: related work.
- `sections/03_method.tex`: method.
- `sections/04_experiments.tex`: experiments.
- `sections/05_ablation_study.tex`: ablation study.
- `sections/06_analysis.tex`: analysis.
- `sections/07_conclusion.tex`: conclusion.
- `sections/08_limitations.tex`: limitations.
- `results/`: experiment result records and paper-facing result summaries.
- `references.bib`: verified bibliography entries used by the current draft.

Replace the lightweight local preamble with the official venue template before
submission.

## Data Scope

- Current paper-facing tables use `tables_ablation/` and the reproduced
  controlled Ours results.
- `tables/component_ablation.tex` and `tables/internal_ablation.tex` are
  historical system-level experiment logs and should not be mixed with the
  controlled mechanism claim.
- The temporary method name is `Ours (unnamed)` until a final name is chosen.

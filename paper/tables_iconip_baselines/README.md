# ICONIP-Style Baseline Tables

These tables follow the table style and baseline set in:

`/Users/bytedance/mypaper/paper_iconip/main.tex`

Scope:

- The reference `Ours` column is intentionally omitted because it does not belong to the current project.
- Baseline values are copied from the reference `main.tex`.
- Best and second-best markings are recomputed only among the baseline methods retained in each table.
- The files are standalone LaTeX table snippets and assume the paper preamble already loads `graphicx`, `booktabs`, and `multirow`.

Files:

- `industrial_baselines.tex`: five industrial benchmarks, image-level and pixel-level blocks.
- `medical_image_baselines.tex`: three medical image-level benchmarks.
- `medical_pixel_baselines.tex`: four medical pixel-level benchmarks.
- `all_baseline_tables.tex`: inputs all three tables in sequence.

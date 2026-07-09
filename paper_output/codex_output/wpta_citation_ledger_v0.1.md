# WPTA Citation Ledger v0.1

本文件记录当前中文稿 v0.6 使用或准备使用的引用状态。原则：能由 arXiv、CrossRef、DOI 或本地已有 `.bib` 追溯的条目才进入正文 citation key；无法核验的条目保留为 `CITATION-PENDING`，不生成伪 BibTeX。

## Verified metadata

| Key | Work | Verification source | Status | Notes |
|---|---|---|---|---|
| `radford2021clip` | Learning Transferable Visual Models From Natural Language Supervision | arXiv `2103.00020` | verified metadata | arXiv title, authors and year verified. Local bib also exists. |
| `zhou2024anomalyclip` | AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection | arXiv `2310.18961`, comment accepted by ICLR 2024 | verified metadata | Use for fixed normal/abnormal prompt baseline and ZSAD setting. |
| `jeong2023winclip` | WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | arXiv `2303.14814`; CrossRef DOI `10.1109/CVPR52729.2023.01878` | verified metadata | Use CrossRef author list in final BibTeX, not the older local bib if names differ. |
| `cao2024adaclip` | AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection | arXiv `2407.15795`; DOI `10.1007/978-3-031-72761-0_4` | verified metadata | Accepted by ECCV 2024 per arXiv metadata. |
| `shu2022tpt` | Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models | arXiv `2209.07511`; comment NeurIPS 2022 | verified metadata | Use for general VLM test-time prompt adaptation context. |
| `bergmann2019mvtec` | MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection | CrossRef DOI `10.1109/CVPR.2019.00982` | verified metadata | CVPR 2019 proceedings record verified. |
| `bergmann2021mvtec` | The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection | CrossRef DOI `10.1007/s11263-020-01400-4` | verified metadata | IJCV extension verified; use if the dataset description needs journal version. |
| `chen2024clipad` | CLIP-AD: A Language-Guided Staged Dual-Path Model for Zero-shot Anomaly Detection | arXiv `2311.00453` | verified metadata | Venue not verified from current metadata; use as arXiv work unless final venue is confirmed. |

## Local-bib metadata, final verification still needed

| Key | Work | Local source | Status | Action before submission |
|---|---|---|---|---|
| `zou2022visa` | SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation | `/Users/bytedance/mypaper/paper_iconip/references.bib` and `/Users/bytedance/mypaper/Titlelabel1__1_/ref.bib` | local metadata only | Verify official venue metadata and whether this is the canonical VisA dataset citation. |
| `jezek2021deep` | Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions | local bib with DOI `10.1109/ICUMT54235.2021.9631567` | local metadata only | Fetch CrossRef or IEEE metadata. |
| `mishra2021vt` | VT-ADL: A vision transformer network for image anomaly detection and localization | local bib, arXiv preprint | local metadata only | Verify arXiv or publisher metadata and canonical BTAD citation. |
| `aota2023zero` | Zero-shot versus many-shot: Unsupervised texture anomaly detection | local bib, WACV 2023 | local metadata only | Verify CrossRef/IEEE metadata. |
| `zhou2022coop` | Learning to Prompt for Vision-Language Models | local bib | local metadata only | Verify DOI or publisher metadata if cited. |

## Citation gaps

| Need | Current treatment | Required action |
|---|---|---|
| Wavelet / texture inspection background | `CITATION-PENDING: wavelet texture inspection` | Verify a canonical wavelet texture or surface inspection reference. Do not cite from memory. |
| Broader test-time adaptation survey | `shu2022tpt` covers VLM prompt TTA but not a full survey | Add a verified TTA survey or core TTA paper if Related Work keeps survey wording. |
| Full external comparison methods | Appendix only, protocol-reference | Verify split, backbone, input size, post-processing and evaluation script before using as main comparison. |

## Draft citation policy for v0.6

- Use verified keys directly in the Chinese manuscript when the statement is covered by the metadata.
- Use local-bib keys for dataset naming only, with an explicit manuscript note that final BibTeX verification remains required.
- Keep non-verified technical-background statements as `CITATION-PENDING`.
- Do not claim all citations are final until every key has a canonical BibTeX source.

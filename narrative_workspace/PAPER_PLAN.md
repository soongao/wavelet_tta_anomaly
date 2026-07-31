# Paper Plan

**Working title**: *Image-Conditioned Normal Reference Estimation for CLIP-Based Zero-Shot Anomaly Localization*
**One-sentence contribution**: CLIP-ZSAD uses fixed text prototypes as a generic semantic reference, but lacks the normal-appearance reference of the current test image; we estimate that image-conditioned normal reference from the test image itself, using wavelet/frequency response as a reliability signal for evidence selection rather than as a fused anomaly score.
**Venue**: CVPR (CV / industrial anomaly detection). Alt: ICCV / WACV.
**Type**: Method paper (with a diagnostic/mechanism core).
**Date**: 2026-07-19
**Page budget**: 8 pages main body (CVPR; references/appendix not counted).
**Section count**: 6.

> 数值口径更新（2026-07-30）：`EXPERIMENT_PLAN_PAPER.md` 是目标占位数值。当前 MVTec/VisA 受控真实结果以 `paper/tables/*.csv`、`paper/prototype_main_result_table.md` 和 `EXPERIMENT_TARGETS.md` 为准。正式方法名未定，表格先用 `Ours (unnamed)`。
> 叙事依据 `NARRATIVE.md`、`WRITING_BLUEPRINT.md`；查新边界见 `NOVELTY_CHECK.md`（频率本身非卖点，卖点是"图像条件化正常参照 + 频域可靠性选证据"）。

---

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Section |
|---|-------|----------|--------|---------|
| C1 | CLIP-ZSAD has a reference gap: fixed text prototypes provide generic semantic reference but not the current image's normal-appearance reference | Problem formulation; Fig 1 motivation; qualitative examples showing context-dependent local responses | Core framing; qualitative evidence still required | §1, §3.1 |
| C2 | Using HF directly (score/feature fusion) without a reference degrades the anomaly map | Controlled direct fusion vs Ours: MVTec P-AUPRO 80.4 vs 86.2; VisA 85.1 vs 91.7 | Supported (reproduced) | §5.1 |
| C3 | A fixed/global reference is insufficient; the normal reference should be conditioned on the current image | GlobalRef vs Ours: MVTec `91.2/84.3/93.1/96.9` vs `91.8/86.2/94.1/97.4`; VisA `95.8/89.2/83.2/86.4` vs `96.2/91.7/84.3/87.3`. Ours - GlobalRef pAUPRO = `+1.9` / `+2.5`. Manual provenance is recorded in `GLOBALREF_EXPERIMENT_RECORD.md`; original command/log unavailable. | Supported as manual diagnostic record | §5.1 |
| C4 | Pure semantic self-reference is limited because evidence selection and correction both rely on the initial CLIP semantic score; frequency reliability gives an additional evidence-selection cue | Controlled Ours vs semantic-only improves all four metrics: MVTec `91.6/85.2/93.7/97.1` to `91.8/86.2/94.1/97.4`; VisA `96.0/90.4/83.7/86.9` to `96.2/91.7/84.3/87.3`. Blind-spot recall still pending for stronger attribution. | Supported with scoped attribution | §5.1, §5.3 |
| C5 | Gains should concentrate on texture/micro-defect classes if the reference-gap mechanism is correct | Per-category + bootstrap analysis still pending | Pending | §5.2 |
| C6 | Method improves over AnomalyCLIP across 5 datasets without auxiliary training | MVTec/VisA controlled Ours reproduced; MPDD/BTAD/DTD remain system-level rows that need global-vs-tuned labeling | Supported with scope | §4 |
| C7 | Per-image reference does not increase false positives on normal images; overhead ≤ 25% | Stability + runtime summaries | Supported (reproduced) | §5.4 |

**Known weaknesses (be honest in paper):**
- vs CLIP-only / semantic-only prototype adaptation, the reproduced controlled gain is small but positive across all four metrics and should be scoped to MVTec/VisA. Blind-spot recall + per-category analysis would strengthen mechanism attribution.
- MPDD/BTAD/DTD use dataset-tuned settings; must separate global vs tuned (Table 6).
- Frequency usefulness for ZSAD is prior consensus (FE-CLIP/WMoE/HarmoniAD) → do NOT claim it; claim the reference gap and evidence-selection role.

---

## Structure (6 sections, 8 pages)

### §0 Abstract (~200 words)
- **What we achieve**: training-free image-conditioned normal reference estimation for CLIP-ZSAD.
- **Why hard**: fixed text prototypes provide generic normal/abnormal semantics, but the meaning of a local response depends on the current image's material and structure.
- **How**: use the initial CLIP semantic score together with wavelet-derived local reliability to select reference evidence, estimate an image-conditioned normal reference, and recompute the anomaly map in the CLIP semantic space. No training, no auxiliary normal bank.
- **Evidence**: controlled MVTec/VisA results; DirectFusion negative control; SemanticProto strong control; GlobalRef diagnostic control; normal-image stability. Blind-spot recall, per-category bootstrap, and real qualitative figures remain pending.
- **Most important caution**: do not write pending mechanism targets as observed results in the abstract.

### §1 Introduction (1.5 pages)
- **Opening hook**: ZSAD with CLIP is attractive because fixed normal/abnormal text prototypes transfer across categories, but anomaly localization is reference-dependent: the same local response can be normal texture in one image and a defect in another.
- **Gap**: prior CLIP-ZSAD works improve semantic/text/feature representations, and frequency-based works inject frequency as an additional feature or branch. The missing point here is more specific: fixed text prototypes lack the current test image's normal-appearance reference.
- **One-sentence contribution**: (see top).
- **Key questions**: (Q1) What reference is missing from fixed text-prototype ZSAD? (Q2) Can a current-image reference be estimated without target normal images or training? (Q3) What evidence-selection signal avoids simply reusing the initial semantic score?
- **Contributions** (numbered, falsifiable):
  1. A problem framing: identify a reference gap in fixed text-prototype CLIP-ZSAD, namely the absence of the current image's normal-appearance reference.
  2. A training-free method: estimate an image-conditioned normal reference from reliable patch evidence, with wavelet/frequency response used for evidence reliability rather than direct score fusion.
  3. Controlled mechanism evidence: DirectFusion fails as a direct anomaly score, while Ours improves over SemanticProto under reproduced MVTec/VisA controlled settings without increasing normal-image false positives.
- **Results preview**: controlled MVTec/VisA gains; DirectFusion negative control; SemanticProto strong control; GlobalRef diagnostic control; blind-spot recall marked pending unless completed.
- **Hero figure = Fig 1** (motivation): see Figure Plan. MUST show "same local/high-frequency response can mean defect or normal texture depending on image context → fixed text prototypes lack the current-image normal reference".
- **Key citations**: CLIP; WinCLIP; AnomalyCLIP; FE-CLIP/WMoE-CLIP (frequency prior); one TTA-for-ZSAD.
- **Front-loading check**: contribution + mechanism ordering visible before §3.

### §2 Related Work (1 page, ≥3 paragraphs, synthesized not listed)
- **(a) CLIP-based ZSAD**: WinCLIP, AnomalyCLIP, VCP-CLIP, AA-CLIP, AdaCLIP — improve semantic/text/feature representations; position: fixed or learned prototypes remain generic with respect to the current image.
- **(b) Frequency for anomaly detection**: FE-CLIP, WMoE-CLIP, HarmoniAD, frequency discriminators — frequency usefulness is prior context; position: our frequency signal is not a direct anomaly branch but an evidence reliability cue.
- **(c) Test-time / reference-based adaptation**: WinCLIP+ (needs normal images), PILOT/Dual-Image/MRAD (pseudo-labels, synthesis, retrieval); position: our setting uses no target normal bank and no parameter training, and estimates the reference from the test image itself.

### §3 Method (2 pages)
- **§3.1 Fixed prototypes and reference gap**: notation (patch features F∈R^{H×W×C}, semantic score S0); show that fixed text prototypes lack current-image normal appearance.
- **§3.2 Wavelet reliability, not wavelet scoring**: Haar DWT on patch-feature grid; LL=structure, HF=|LH|+|HL|+|HH|; boundary-aware W = HF·(1−LF_edge). Emphasize: W is an evidence reliability signal, not the anomaly map.
- **§3.3 Evidence selection**: trustworthy-normal = low S0 & low W; abnormal-evidence = high S0 & high W. Explain this as avoiding pure semantic self-reference, not as proving "CLIP blind spot" unless the blind-spot experiment is present.
- **§3.4 Image-conditioned normal reference**: estimate v_normal / v_abn from selected patches; conservative update (α0=0, small β on normal side); no backprop, no parameter update. Recompute anomaly map in CLIP semantic space.
- **Note**: multi-crop / pixel-to-image are standard aggregation add-ons, orthogonal to the mechanism → appendix only.

### §4 Experiments — Main Results (1.5 pages)
- **§4.1 Setup**: datasets (MVTec, VisA, MPDD, BTAD, DTD-Synth), metrics (pAUROC/pAUPRO/iAUROC/iAP), CLIP backbone, no-training statement.
- **§4.2 Main results**: prioritize reproduced controlled MVTec/VisA for paper-facing causal claims. Five-dataset rows can be shown separately as system-level results with global vs dataset-tuned labels. Table 2 SOTA numbers remain external `*` until verified.

### §5 Analysis & Ablation (1.5 pages) — the mechanism core
- **§5.1 What the reference must not be**: DirectFusion negative control; SemanticProto strong control; GlobalRef diagnostic control. This is the main mechanism section.
- **§5.2 Where the gain comes from**: Fig `fig_percategory_gain` (texture vs object) → C5, pending until real per-category + bootstrap evidence exists.
- **§5.3 Semantic self-reference diagnostic**: blind-spot recall only if completed; otherwise keep as future diagnostic, not main evidence.
- **§5.4 Stability, runtime, sensitivity**: Table 5 (FP area, runtime) + Fig `fig_sensitivity` (the wavelet-mix subplot doubles as a mechanism curve) → C7.
- **§5.5 Qualitative**: heatmap grid (Input/GT/Baseline/SelfRef/HF map/Ours), see QUALITATIVE_SPEC.
- **§5.6 Design ablation**: Table 4 (boundary-aware, conservative update).

### §6 Conclusion (0.5 pages)
- **Restatement**: fixed CLIP text prototypes lack the current image's normal reference; our method estimates that reference training-free and uses frequency only for evidence reliability.
- **Limitations**: small but consistent gain over CLIP-only prototype adaptation; dataset-tuned settings on 3 datasets; logical anomalies still hard.
- **Future work**: per-image reference for other frozen encoders; combining with lightweight adaptation.

---

## Figure Plan

| ID | Type | Description | Source | Priority |
|----|------|-------------|--------|----------|
| **Fig 1 (Hero)** | Motivation | Same local/high-frequency response can mean defect in one image and normal texture/boundary in another; fixed text prototypes lack the current-image normal reference. Caption must state the reference gap, not "HF is anomaly." | `figures/fig1_motivation.svg` | HIGH |
| **Fig 2** | Architecture | Test image → frozen CLIP → patch features → {semantic S0 ; Haar DWT → HF/W} → evidence select → **per-image normal reference** → anomaly map. | `figures/fig2_architecture.svg` | HIGH |
| **Fig 3** | Bar+line | Mechanism controls: DirectFusion, SemanticProto, GlobalRef, and Ours. | `result_charts/fig_mechanism_ordering` | HIGH |
| **Fig 4** | H-bar | Per-category gain Ours vs SelfRef, texture vs object split. Pending until real per-category + bootstrap evidence exists. | `result_charts/fig_percategory_gain` | MEDIUM |
| **Fig 5** | Heatmap grid | Qualitative: Input/GT/Baseline/SelfRef/HF map/Ours. **Real results required.** | `QUALITATIVE_SPEC.md` | HIGH |
| **Fig 6** | Line ×3 | Sensitivity: top-k / wavelet-mix (mechanism curve) / update β. | `result_charts/fig_sensitivity` | MEDIUM |
| **Fig 7** | Bar | CLIP blind-spot recall (compact). Can merge into §5.3. | `result_charts/fig_blindspot` | MEDIUM |
| Table 1 | Comparison | Main results, 5 datasets × 4 metrics. | `TABLES.md` T1 | HIGH |
| Table 2 | Comparison | SOTA on MVTec (external `*` to verify). | `TABLES.md` T2 | HIGH |
| Table 3 | Ablation | Core mechanism ablation, full metrics. | `TABLES.md` T3 | HIGH |
| Table 4 | Ablation | Design ablation. | `TABLES.md` T4 | MEDIUM |
| Table 5 | Stats | Normal stability + runtime. | `TABLES.md` T5 | MEDIUM |
| Table 6 | Stats | Global vs dataset-tuned. | `TABLES.md` T6 | LOW/appendix |

**Hero figure (Fig 1) caption draft**: "A strong local response is not self-explanatory: it may indicate a defect on a smooth object, but the same kind of response may be normal on a rough texture or boundary. Fixed text prototypes provide generic normal/abnormal semantics but not the current image's normal reference. Our method estimates that image-conditioned reference from reliable patch evidence."

---

## Citation Plan

- **§1 Intro**: CLIP [Radford 2021]; WinCLIP [Jeong CVPR23]; AnomalyCLIP [Zhou ICLR24]; FE-CLIP [ICCV25] `[VERIFY]`; one ZSAD-TTA.
- **§2 Related**:
  - (a) WinCLIP, AnomalyCLIP, VCP-CLIP [ECCV24], AA-CLIP [CVPR25] `[VERIFY]`, AdaCLIP `[VERIFY]`.
  - (b) FE-CLIP, WMoE-CLIP `[VERIFY arXiv]`, HarmoniAD `[VERIFY]`, DFD frequency discriminators.
  - (c) WinCLIP+, PILOT `[VERIFY]`, Dual-Image Enhanced CLIP `[VERIFY]`, MRAD `[VERIFY]`.
- **§3 Method**: Haar wavelet reference; AnomalyCLIP (baseline we build on).
- **§4 Experiments**: MVTec-AD [Bergmann CVPR19], VisA [Zou ECCV22], MPDD, BTAD, DTD-Synthetic.

**Citation rules**: verify every entry (authors/year/venue); do NOT generate BibTeX from memory; flag `[VERIFY]`; prefer published over arXiv. External SOTA numbers in Table 2 are `*`=to-verify until read from source papers.

---

## Reviewer Feedback
- Cross-review with GPT-5.4 (Codex MCP) **not run** in this environment (no OpenAI review configured). Recommended before freezing: run Step 6 of `paper-plan` on this file, focusing on (i) whether §5 mechanism evidence is strong enough to carry the paper given the small mean gain over SelfRef, and (ii) related-work positioning vs FE-CLIP/WMoE.

## Next Steps
- [ ] Replace historical target numbers with reproduced tables where available; GlobalRef values now have manual provenance in `narrative_workspace/GLOBALREF_EXPERIMENT_RECORD.md`, with original command/log unavailable. Run remaining blind-spot recall and per-category mechanism checks.
- [ ] Verify external SOTA numbers (Table 2) and all `[VERIFY]` citations from source papers.
- [ ] Generate Fig 5 qualitative heatmaps from real inference (per QUALITATIVE_SPEC.md).
- [ ] /paper-write to draft LaTeX section by section from this plan.
- [ ] /paper-compile to build PDF.

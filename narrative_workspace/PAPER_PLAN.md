# Paper Plan

**Working title**: *Reading, Not Fusing: Per-Image Normal Reference Estimation for CLIP's High-Frequency Anomaly Cues in Zero-Shot Anomaly Detection*
**One-sentence contribution**: CLIP already encodes anomaly-related high-frequency signal in its patch features, but that signal is uninterpretable without a per-image normal reference; we read it out and estimate that reference on each test image — training-free — turning an otherwise-collapsing cue into reliable zero-shot localization.
**Venue**: CVPR (CV / industrial anomaly detection). Alt: ICCV / WACV.
**Type**: Method paper (with a diagnostic/mechanism core).
**Date**: 2026-07-19
**Page budget**: 8 pages main body (CVPR; references/appendix not counted).
**Section count**: 6.

> 数值来自 `EXPERIMENT_PLAN_PAPER.md` 的 EXPECTED 目标值，成稿前用真实 log 替换。
> 叙事依据 `NARRATIVE.md`；查新边界见 `NOVELTY_CHECK.md`（频率本身非卖点，卖点是"逐图正常参照"）。

---

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Section |
|---|-------|----------|--------|---------|
| C1 | CLIP patch features already respond to anomaly-induced high-frequency change, but normal rough materials respond just as strongly → HF alone is ambiguous | Fig 1 motivation; qualitative HF map column; per-category analysis | Core diagnostic | §1, §3.1, §5 |
| C2 | Using HF directly (score/feature fusion) without a reference collapses the anomaly map | Table 3 DirectHF 82.5/73.0 vs Ours 92.4/85.8 | Supported (target) | §5.1 |
| C3 | A single global HF threshold is insufficient; the normal reference must be estimated per image | Table 3 GlobalRef 83.2 vs Ours 85.8 (pAUPRO); monotone chain | Supported (target) | §5.1 |
| C4 | The signal used to pick trustworthy-normal patches must be external to CLIP semantics (HF), else it inherits CLIP's blind spot | Table 3 SelfRef 84.6 vs Ours 85.8; Table 5 blind-spot recall 22%/18% | Supported (target) | §5.1, §5.3 |
| C5 | Gains concentrate on texture/micro-defect classes (where CLIP is frequency-blind), not object classes | Fig per-category (texture +3.1 avg, object +0.1) | Supported (target) | §5.2 |
| C6 | Method improves over AnomalyCLIP across 5 datasets without auxiliary training | Table 1 main results | Supported (target) | §4 |
| C7 | Per-image reference does not increase false positives on normal images; overhead ≤ 25% | Table 5 stability + runtime | Supported (target) | §5.4 |

**Known weaknesses (be honest in paper):**
- vs CLIP-only prototype adaptation (SelfRef), the total-mean gain is small; mechanism is carried by blind-spot recall + per-category + collapse controls, not by mean domination. → State as limitation.
- MPDD/BTAD/DTD use dataset-tuned settings; must separate global vs tuned (Table 6).
- Frequency usefulness for ZSAD is prior consensus (FE-CLIP/WMoE) → do NOT claim it; claim the reference.

---

## Structure (6 sections, 8 pages)

### §0 Abstract (~200 words)
- **What we achieve**: training-free zero-shot anomaly detection that reads CLIP's high-frequency cues and calibrates them against a per-image normal reference.
- **Why hard**: CLIP is trained for global semantic alignment; its patch features are low-pass, and although anomalies do perturb high frequencies, normal rough materials do too — so the raw HF cue is uninterpretable.
- **How**: read HF from CLIP patch features via Haar DWT; select trustworthy-normal vs abnormal evidence using HF (a signal external to CLIP semantics); estimate a per-image normal reference and recompute the anomaly map. No training, no auxiliary data.
- **Evidence**: 5 datasets; mechanism ablation showing direct fusion collapses (pAUROC −10) and a fixed global reference is insufficient; recovers 18–22% of anomalies CLIP misses.
- **Most remarkable**: gains concentrate exactly on the texture/micro-defect classes where CLIP is frequency-blind.

### §1 Introduction (1.5 pages)
- **Opening hook**: ZSAD with CLIP is attractive (open-vocabulary, no target data) but CLIP is optimized for "what object", not "what is locally wrong".
- **Gap**: prior CLIP-ZSAD works fix this in the semantic/text/feature space (WinCLIP multi-scale windows, AnomalyCLIP object-agnostic prompts, VCP/AA-CLIP feature adaptation) or fuse frequency as an extra feature via trained adapters (FE-CLIP, WMoE-CLIP). None asks whether the anomaly cue is *already present* in CLIP and merely *unusable without a reference*.
- **One-sentence contribution**: (see top).
- **Key questions**: (Q1) Is anomaly-related high-frequency signal already in CLIP features? (Q2) Why can't we use it directly? (Q3) What minimal, training-free step makes it usable?
- **Contributions** (numbered, falsifiable):
  1. A diagnostic: CLIP patch features already carry anomaly-related HF signal, but it is ambiguous because normal materials are high-frequency too — quantified by a controlled ablation where direct use collapses.
  2. A training-free method: read HF, select evidence with this CLIP-external signal, and estimate a **per-image normal reference**; anomaly = deviation from that reference.
  3. Evidence that the reference must be (a) per-image not global, and (b) driven by a signal external to CLIP — including recovering 18–22% of anomalies CLIP alone misses, concentrated on frequency-blind texture classes.
- **Results preview**: consistent gains on 5 datasets; the mechanism ablation ordering DirectHF < Baseline < GlobalRef < SelfRef < Ours.
- **Hero figure = Fig 1** (motivation): see Figure Plan. MUST show "same HF magnitude = anomaly in (a), normal in (b) → fixed threshold fails → per-image reference fixes it".
- **Key citations**: CLIP; WinCLIP; AnomalyCLIP; FE-CLIP/WMoE-CLIP (frequency prior); one TTA-for-ZSAD.
- **Front-loading check**: contribution + mechanism ordering visible before §3.

### §2 Related Work (1 page, ≥3 paragraphs, synthesized not listed)
- **(a) CLIP-based ZSAD**: WinCLIP, AnomalyCLIP, VCP-CLIP, AA-CLIP, AdaCLIP — all adapt semantics/text/features; position: they add capability, we argue capability partly exists and needs calibration.
- **(b) Frequency for anomaly detection**: FE-CLIP, WMoE-CLIP, HarmoniAD, frequency discriminators — all *fuse* frequency, mostly with training; position: fuse-in vs read-and-reference; direct fusion is our negative control.
- **(c) Test-time / reference-based adaptation**: WinCLIP+ (needs normal images), PILOT/Dual-Image/MRAD (pseudo-labels, synthesis, retrieval); position: none estimates a per-image normal reference from a CLIP-external signal.

### §3 Method (2 pages)
- **§3.1 Observation & problem formulation**: notation (patch features F∈R^{H×W×C}, semantic score S0); state the ambiguity formally — HF response mixes anomaly and material.
- **§3.2 Reading high frequency**: Haar DWT on patch-feature grid; LL=structure, HF=|LH|+|HL|+|HH|; boundary-aware W = HF·(1−LF_edge). Emphasize: this reads existing info, adds nothing.
- **§3.3 Evidence selection**: trustworthy-normal = low S0 & low W; abnormal-evidence = high S0 & high W. Why HF (CLIP-external) is required here.
- **§3.4 Per-image normal reference**: estimate v_normal / v_abn from selected patches; conservative update (α0=0, small β on normal side); no backprop, no parameter update. Recompute anomaly map as deviation from reference.
- **Note**: multi-crop / pixel-to-image are standard aggregation add-ons, orthogonal to the mechanism → appendix only.

### §4 Experiments — Main Results (1.5 pages)
- **§4.1 Setup**: datasets (MVTec, VisA, MPDD, BTAD, DTD-Synth), metrics (pAUROC/pAUPRO/iAUROC/iAP), CLIP backbone, no-training statement.
- **§4.2 Main results**: Table 1 (5 datasets × 4 metrics). Table 2 (SOTA comparison, external `*` to verify). Global vs dataset-tuned → Table 6.

### §5 Analysis & Ablation (1.5 pages) — the mechanism core
- **§5.1 Where the reference comes from**: Table 3 + Fig `fig_mechanism_ordering`. The monotone chain is the main mechanism evidence (C2, C3, C4).
- **§5.2 Where the gain comes from**: Fig `fig_percategory_gain` (texture vs object) → C5.
- **§5.3 Recovering CLIP's blind spot**: Table 5 blind-spot recall + Fig `fig_blindspot` → C4.
- **§5.4 Stability, runtime, sensitivity**: Table 5 (FP area, runtime) + Fig `fig_sensitivity` (the wavelet-mix subplot doubles as a mechanism curve) → C7.
- **§5.5 Qualitative**: heatmap grid (Input/GT/Baseline/SelfRef/HF map/Ours), see QUALITATIVE_SPEC.
- **§5.6 Design ablation**: Table 4 (boundary-aware, conservative update).

### §6 Conclusion (0.5 pages)
- **Restatement**: the anomaly cue is already in CLIP; the missing piece is a per-image reference, supplied training-free.
- **Limitations**: small mean gain over CLIP-only prototype adaptation; dataset-tuned settings on 3 datasets; logical anomalies still hard.
- **Future work**: per-image reference for other frozen encoders; combining with lightweight adaptation.

---

## Figure Plan

| ID | Type | Description | Source | Priority |
|----|------|-------------|--------|----------|
| **Fig 1 (Hero)** | Motivation | (a) smooth object+defect: HF isolates defect ✓; (b) rough normal texture: whole region HF-bright ✗; middle: "same HF magnitude, opposite meaning → fixed threshold fails"; solution: per-image reference. Caption must state the comparison. | `figures/fig1_motivation.svg` | HIGH |
| **Fig 2** | Architecture | Test image → frozen CLIP → patch features → {semantic S0 ; Haar DWT → HF/W} → evidence select → **per-image normal reference** → anomaly map. | `figures/fig2_architecture.svg` | HIGH |
| **Fig 3** | Bar+line | Mechanism ordering DirectHF<Baseline<GlobalRef<SelfRef<Ours (MVTec pAUPRO). | `result_charts/fig_mechanism_ordering` | HIGH |
| **Fig 4** | H-bar | Per-category gain Ours vs SelfRef, texture vs object split. | `result_charts/fig_percategory_gain` | HIGH |
| **Fig 5** | Heatmap grid | Qualitative: Input/GT/Baseline/SelfRef/HF map/Ours. **Real results required.** | `QUALITATIVE_SPEC.md` | HIGH |
| **Fig 6** | Line ×3 | Sensitivity: top-k / wavelet-mix (mechanism curve) / update β. | `result_charts/fig_sensitivity` | MEDIUM |
| **Fig 7** | Bar | CLIP blind-spot recall (compact). Can merge into §5.3. | `result_charts/fig_blindspot` | MEDIUM |
| Table 1 | Comparison | Main results, 5 datasets × 4 metrics. | `TABLES.md` T1 | HIGH |
| Table 2 | Comparison | SOTA on MVTec (external `*` to verify). | `TABLES.md` T2 | HIGH |
| Table 3 | Ablation | Core mechanism ablation, full metrics. | `TABLES.md` T3 | HIGH |
| Table 4 | Ablation | Design ablation. | `TABLES.md` T4 | MEDIUM |
| Table 5 | Stats | Normal stability + runtime. | `TABLES.md` T5 | MEDIUM |
| Table 6 | Stats | Global vs dataset-tuned. | `TABLES.md` T6 | LOW/appendix |

**Hero figure (Fig 1) caption draft**: "The high-frequency response of CLIP patch features reacts to a defect (a) but reacts equally to a normal rough texture (b). A single fixed threshold therefore cannot separate anomaly from material. Our method estimates each image's own normal high-frequency level and flags only what exceeds it."

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
- [ ] Run experiments; replace all EXPECTED numbers with real logs (start with DirectHF collapse — easiest, is the mechanism keystone).
- [ ] Verify external SOTA numbers (Table 2) and all `[VERIFY]` citations from source papers.
- [ ] Generate Fig 5 qualitative heatmaps from real inference (per QUALITATIVE_SPEC.md).
- [ ] /paper-write to draft LaTeX section by section from this plan.
- [ ] /paper-compile to build PDF.

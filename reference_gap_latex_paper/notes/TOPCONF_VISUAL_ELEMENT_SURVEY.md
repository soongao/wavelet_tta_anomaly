# Top-Conference Figure Element Survey for ICNR

This note summarizes how common visual elements are represented in CVPR/ICCV/ECCV/ICLR-style vision papers, and maps those conventions to the ICNR method figure and qualitative localization figure.

The current Mermaid files are useful as logic sketches, but they should not be used as final paper figures. A top-conference method figure should use visual objects, short labels, spatial grouping, and consistent color semantics rather than long text inside plain boxes.

## Survey Scope

Relevant figure families:

- Vision-language overview figures such as CLIP-style dual encoder diagrams: image encoder, text encoder, embedding vectors, similarity matrix, prompt/text tokens.
- Promptable segmentation overview figures such as SAM-style diagrams: image encoder, prompt encoder, mask decoder, output masks.
- Vision Transformer / self-supervised figures such as DINO/MAE-style diagrams: image crops, patch tokens, transformer block stacks, teacher-student or masked-token paths.
- CLIP anomaly detection figures such as WinCLIP/AnomalyCLIP/AdaCLIP/AA-CLIP-style diagrams: image-text matching, learned prompt/prototype tokens, patch anomaly maps, mask/heatmap overlays.
- Frequency / wavelet method figures: multi-scale pyramids, LL/LH/HL/HH subbands, edge maps, high-frequency response maps.
- Segmentation qualitative figures: input image, GT mask, baseline heatmap, ours heatmap, shared color scale, zoom-in regions.

Local assets worth comparing before drawing:

- `narrative_workspace/figures/fig2_architecture.svg`
- `paper/figures/wavelet_prototype_overview_layout.svg`
- `paper_output/codex_output/figures/figure2_wpta_method_overview.svg`
- `paper/figures/figure8_qualitative_overlay_comparison.pdf`
- `cached_results/prototype_tuned/mechanism_viz/*.png`

## General Pattern

Top-conference vision figures usually use a hybrid visual language:

- Real images or small heatmaps for visual evidence.
- Geometric icons for abstract tensors, tokens, encoders, decoders, prototypes, gates, and losses.
- Very short labels, typically 1-4 words.
- Panel titles for major phases instead of long explanatory text inside modules.
- Color semantics kept consistent across the whole paper.
- Formulas minimized inside the figure; symbols are shown only when they help locate the corresponding equation.

For ICNR, the final architecture figure should communicate this sentence visually:

> Frozen CLIP prototypes provide a semantic axis; ICNR selects reliable patches from the current image and estimates an image-conditioned normal reference for rescoring.

## Element-Level Conventions

| ICNR concept | Top-paper visual convention | Recommended ICNR representation | Avoid |
|---|---|---|---|
| Input image | Real image thumbnail, usually square or 4:3, thin border | Use one MVTec/VisA defect crop, preferably tile crack, leather glue, wood scratch, or zipper defect | Camera icon, generic image placeholder |
| Frozen CLIP / AnomalyCLIP encoder | Stacked transformer blocks, CNN/ViT tower, or chip-like module; lock icon for frozen | Draw 3 stacked blocks with a small lock; label `Frozen Encoder` | Long block text such as "Frozen CLIP / AnomalyCLIP visual encoder outputs multi-layer features" |
| Patch tokens / feature grid | Tiled patch grid, token sequence, or feature-map stack | Draw a 10x10 or 14x14 grid with subtle cells; optionally stack 2-3 grids for multi-layer features | Plain rectangle named `Patch features` |
| Text prototypes | Colored text tokens, embedding vectors, capsules, or anchors | Two capsule tokens: green `normal`, red/orange `abnormal`; optionally connected to a semantic axis | Paragraph about prompt learning |
| Semantic axis | Embedding line, dot-product matrix, or similarity heatmap | Horizontal line from `normal` to `abnormal`, with a few patch dots projected onto it; output small `q_i` map | Full softmax formula |
| Semantic response `q_i` | Heatmap over image/patch grid | Small blue-framed heatmap with sparse red/yellow high-response regions; label `q_i` | A box saying "semantic response on normal/abnormal axis" |
| Haar decomposition | 2x2 subband icon, pyramid, or frequency blocks | Tiny 2x2 tile labelled `LL/LH/HL/HH`, feeding a purple structure map | Writing the Haar equations in the figure |
| High-frequency magnitude `h_i` | Texture/edge response map, often purple/orange | If shown, draw as a faint intermediate mini-map beside the 2x2 subband icon | Separate large module unless it is a main claim |
| Low-frequency boundary `e_i` | Edge contour or low-frequency map | Use a thin grey boundary map overlay, not a large branch | Making it look like another anomaly map |
| Structure response `W_i` | Reliability/attention map | Purple heatmap labelled `W_i`; use boundary-suppressed high-response pattern | "Wavelet map" as final anomaly score |
| Evidence selection | Highlighted patches, lasso selection, top-k mask | Show one patch grid with green selected normal patches and red outlined suspicious patches | Separate text boxes for `N_x`, `A_x`, `omega_i^n`, `omega_i^a` |
| Normal evidence | Green selected cells or points | 6-10 green filled cells in stable background regions | Only color with no legend/shape |
| Suspicious evidence | Red/orange outlined cells or triangles | 3-5 red outlined cells near defect candidates; make it secondary if `v_a` is not central | Making suspicious evidence look like final prediction |
| Image-conditioned normal reference `r_n(x)` | Prototype anchor, centroid, or vector | Green anchor/diamond formed by arrows from selected normal patches; label `r_n(x)` | Another plain rectangle |
| Prototype calibration | Mixer, gate, switch, or small interpolation node | Small circular mixer with lock/gate glyph; label `Conservative update` | Large module listing `c_n`, `c_a`, `m_x`, `g_x`, `alpha`, `beta` |
| Rescoring | Similarity operation or output map generation | Two prototype arrows into a small similarity operator, then final anomaly heatmap | Full softmax equation |
| Final anomaly map | Heatmap overlay on input or standalone heatmap | Red-yellow heatmap thumbnail aligned with input image; label `Anomaly map` | Abstract output box |
| Multi-layer fusion | Stacked heatmaps or small aggregation icon | Optional: 2-3 faint heatmap sheets before final map | A separate long branch if space is tight |

## Recommended Method Figure Layout

Use a wide three-panel layout, suitable for a two-column figure.

### Panel A: Frozen CLIP Semantics

Elements:

- Input defect crop.
- Locked encoder stack.
- Patch feature grid.
- Two text prototype capsules: `normal`, `abnormal`.
- Semantic axis or mini similarity map.
- Output: small `q_i` heatmap.

Message:

- CLIP/AnomalyCLIP provides the frozen semantic axis.
- Patch responses are initially interpreted by text prototypes.

Visual detail:

- Put image and encoder on the top-left.
- Put text prototypes below the encoder or beneath the patch grid.
- Use blue for semantic branch.
- Do not mention reference gap here; this panel is only the baseline semantic path.

### Panel B: Boundary-Aware Evidence Selection

Elements:

- Same patch feature grid feeding a tiny Haar icon.
- 2x2 subband icon: `LL`, `LH`, `HL`, `HH`.
- Purple `W_i` structure map.
- Blue `q_i` semantic map and purple `W_i` map merging into a patch selection grid.
- Patch selection grid with green normal cells and red/orange suspicious cells.

Message:

- Frequency/structure is not the anomaly score.
- It is a reliability signal for selecting patch evidence.

Visual detail:

- Use an explicit merge symbol such as a small `x` or `+` node between `q_i` and `W_i`.
- The selected-patch grid should be the largest object in this panel because it is the core mechanism.
- Label only `q_i`, `W_i`, and `Reliable patches`.

### Panel C: Image-Conditioned Rescoring

Elements:

- Green normal selected patches aggregated into a green anchor `r_n(x)`.
- Existing normal prototype `t_n` connected to a small conservative mixer.
- Existing abnormal prototype `t_a` connected directly or through a faint dotted gate.
- Rescoring operator.
- Final anomaly heatmap.

Message:

- ICNR estimates the current image's normal reference and rescales the local anomaly response.

Visual detail:

- Make `r_n(x)` visually distinct: diamond or anchor, not a rectangle.
- Use a dotted red/orange path for suspicious direction if `v_a` is mentioned.
- Final anomaly map should mirror the input crop size.

## Recommended Shape Library

Use a limited shape vocabulary:

- Image thumbnail: rounded rectangle with real crop.
- Encoder: stacked blocks or chip, with lock icon.
- Patch field: square grid.
- Text prototype: capsule / pill shape.
- Semantic axis: line with two anchors and projected dots.
- Heatmap: small raster-like tile with a color bar only if necessary.
- Haar: four-cell square labelled `LL/LH/HL/HH`.
- Evidence: colored cells on a patch grid.
- Reference: green diamond/anchor/vector centroid.
- Gate: small circle with switch/slider icon.
- Output: heatmap overlay.

Do not use more than 7-8 major visual objects. If every formula variable becomes a separate object, the figure will read as a flowchart rather than a method overview.

## Color Semantics

Recommended palette:

- Frozen / shared CLIP components: neutral grey, `#6B7280`.
- Semantic branch: blue, `#2563EB`.
- Structure / wavelet branch: purple, `#7C3AED`.
- Normal evidence/reference: green, `#2E8B57`.
- Suspicious/abnormal evidence: vermilion/orange, `#D94841` or `#E68613`.
- Final anomaly heatmap: perceptual heatmap such as magma/inferno, with yellow as high response.

Use shape plus color:

- Normal evidence: green filled squares.
- Suspicious evidence: red/orange outlined squares or triangles.
- Final heatmap: continuous colormap.

This avoids relying on red-green color contrast alone.

## Typography and Label Rules

Use short labels:

- `Input`
- `Frozen Encoder`
- `Patch Tokens`
- `Text Prototypes`
- `q_i`
- `W_i`
- `Reliable Patches`
- `r_n(x)`
- `Conservative Update`
- `Rescoring`
- `Anomaly Map`

Avoid labels like:

- `CLIP patch features are matched with normal and abnormal text prototypes`
- `Boundary-aware local structure response suppresses low-frequency edges`
- `Evidence confidence controls update gate and prototype interpolation`

Put detailed explanations in the caption and method text.

## What to Change From the Current Mermaid Figure

Current issue:

- Too many nodes.
- Too much text per node.
- Most objects are plain rectangles.
- Formula variables appear as separate modules.
- `v_a`, `c_n`, `c_a`, gate variables, layer fusion, and final softmax all compete for visual attention.

Required simplification:

- Merge `HF`, `LF`, and `Haar` into one visual subband icon plus one `W_i` map.
- Merge `omega_i^n`, `omega_i^a`, `N_x`, `A_x` into one evidence-selection grid.
- Merge confidence and update gates into one small `Conservative Update` mixer.
- Make `r_n(x)` the central green anchor in Panel C.
- Use visual maps for `q_i`, `W_i`, and final anomaly map.
- Keep only three symbols in the figure: `q_i`, `W_i`, `r_n(x)`.

## Qualitative Localization Figure Elements

The qualitative figure should not be drawn as a pure vector diagram. It should be a hybrid PDF:

- Raster: input images, masks, heatmaps.
- Vector: labels, row names, borders, zoom boxes, arrows, mask contours, captions.

Column order:

1. `Input`
2. `GT`
3. `AnomalyCLIP`
4. `Semantic-only`
5. `ICNR`

Row categories:

- Texture false positive: carpet / wood / tile.
- Small local defect: tile crack / leather glue / pill imprint.
- Boundary confusion: zipper / cable / object contour.
- Limited gain or failure: large-area anomaly or logic defect.

Visual rules:

- Same crop size in every cell.
- Same heatmap colormap for every method.
- Same normalization within each row.
- GT mask as red contour, not opaque fill.
- Zoom box repeated in every column for tiny defects.
- Failure row labelled honestly, e.g. `limited gain`.

## Caption Templates

Method figure:

> Overview of ICNR. The frozen CLIP/AnomalyCLIP encoder and text prototypes provide a semantic response map, while a boundary-aware structure response is used only to select reliable patch evidence. ICNR aggregates stable normal patches into an image-conditioned normal reference and rescales the patch anomaly response without training or test-time parameter updates.

Qualitative figure:

> Qualitative localization examples. ICNR suppresses normal texture and structure-boundary responses while concentrating high responses on annotated defects. The last row shows a limited-gain case, where the current-image normal reference is less reliable.

## Production Recommendation

For the method figure:

- Use Figma or draw.io for the final vector figure.
- Use the Mermaid source only as a layout guide.
- Export as `figures/fig_method_architecture_icnr.pdf`.
- Keep all text editable/vector, not rasterized.

For the qualitative figure:

- Use a Python/Matplotlib script.
- Save as `figures/fig_qualitative_localization.pdf`.
- Keep images raster but labels/contours/vector overlays in PDF.

## Final Checklist

- The figure can be understood in 10 seconds.
- No element contains more than 4 words except panel titles.
- `q_i`, `W_i`, and `r_n(x)` are visually traceable.
- Frequency branch does not look like the final anomaly score.
- Normal evidence and suspicious evidence are distinct by both color and shape.
- Frozen components are visually marked as frozen.
- Ours is highlighted, but baselines are not unfairly degraded.
- Export is PDF/SVG vector.
- Font size is at least 8 pt after LaTeX scaling.
- Caption states the finding in the first sentence.

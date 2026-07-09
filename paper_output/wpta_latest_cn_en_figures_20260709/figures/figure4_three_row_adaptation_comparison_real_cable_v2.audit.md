# Figure 4 Real Cable V2 Audit

- Canvas: 2200 x 1180, three horizontal rows with compact Input -> Method space -> Evaluation view flow.
- Real imagery: the MVTec cable test image is embedded as the left input in every row; GT/fixed/direct/WPTA result panels use local real assets.
- Encoder representation: no diamond/parallelogram encoder blocks. The CLIP visual branch is represented as a spatial token field with the real image faintly embedded under patch grids.
- Difference placement: the fixed, direct-evidence, and WPTA rows differ inside the method space itself through patch selection, evidence routing, and prototype update behavior.
- Result layout: the right side has only GT and one method-specific result panel, plus a short interpretation note; no four-column serial output chain remains.
- Draw.io editability: text, boxes, arrows, patch grids, and prototype nodes are native Draw.io elements; real dataset images are embedded raster cells.
- Remaining manual review point: Draw.io may substitute fonts depending on the local diagrams.net environment.

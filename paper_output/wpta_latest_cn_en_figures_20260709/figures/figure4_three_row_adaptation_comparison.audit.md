# Figure 4 Audit

- Canvas: 1800 x 1050, white background, three horizontal comparison rows.
- Purpose: compare fixed CLIP prototype scoring, naive test-time prototype adaptation, and WPTA wavelet-gated adaptation.
- Visible elements: title/subtitle, column headers, three tinted row bands, input image thumbnails, frozen CLIP patch-feature blocks, evidence/scoring blocks, prototype-state blocks, anomaly-map heatmaps, arrows, and row-level explanatory notes.
- Medium: native Draw.io rectangles, ellipses, text, and connectors; no raster crops embedded in the `.drawio`.
- Visual proof target: row 1 shows instance mismatch under fixed text prototypes; row 2 shows drift from unreliable high-score evidence; row 3 shows semantic prior and wavelet reliability intersection producing conservative calibration and a cleaner map.
- Known limitation: PNG preview is rendered from the same coordinate specification with Pillow because Draw.io CLI is unavailable in this environment.

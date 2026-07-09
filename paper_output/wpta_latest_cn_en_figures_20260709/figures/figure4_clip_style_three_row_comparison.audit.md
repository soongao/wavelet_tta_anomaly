# Figure 4 CLIP-style three-row comparison audit

- Canvas: 2600 x 1500 px, three stacked rows on a light paper background.
- Reference style target: CLIP-like dual tower architecture, with image encoder and text encoder separated before patch-text similarity.
- Real raster inputs: MVTec cable image and three anomaly-map outputs are embedded as PNG images.
- Native/editable elements: row titles, prompt labels, encoders, token grids, prototypes, adaptation blocks, similarity matrices, arrows, and legend.
- Row (a): frozen CLIP towers, fixed normal/abnormal prototypes, no update path.
- Row (b): direct test-time prototype adaptation, red dashed contamination path from abnormal patch tokens to prototype drift.
- Row (c): wavelet-guided prototype adaptation, DWT split into LL semantic and LH/HL/HH detail bands, reliability gate before updating pN.
- Prompt coverage: Normal: "normal cable" and Abnormal: "damaged cable" are visible in every row.
- Visual QA: figure is nonblank, high-contrast, and uses distinct color roles for image/text/drift/wavelet evidence.

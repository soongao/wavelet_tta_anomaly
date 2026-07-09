# Figure 4 CLIP Architecture Wavelet Audit

- Canvas: 2400 x 1360, three horizontal rows in CLIP dual-tower architecture style.
- Real imagery: each row uses the same real MVTec cable input image; right-side anomaly maps use local cable result assets.
- Text prompts: every row includes explicit normal and abnormal prompts.
- Row (a): frozen CLIP image/text encoders, fixed text prototypes, direct cosine similarity.
- Row (b): direct test-time prototype adaptation, with abnormal patch contamination and prototype drift highlighted in red.
- Row (c): wavelet-guided prototype adaptation, with DWT, LL semantic branch, LH/HL/HH detail branch, and reliability gate highlighted in green/orange.
- Draw.io editability: architecture boxes, text, arrows, prompts, patch tokens, prototypes, and similarity blocks are native Draw.io cells; only real input/output images are embedded rasters.
- Manual review point: the PNG preview is rendered by the paired PIL generator because diagrams.net CLI is not available in this environment.

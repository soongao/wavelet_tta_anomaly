# Figure 4 enlarged CLIP dual-tower v2 audit

- Canvas: 2600 x 1580 px, three stacked rows with a larger CLIP dual-tower container in every row.
- Visual intent: make the image tower and text tower the dominant structure, then place the architectural difference inside their update paths.
- Real raster inputs: one MVTec cable image and three anomaly-map outputs are embedded as PNG images.
- Native/editable elements: row titles, prompt labels, tower lanes, encoders, tokens, prototypes, adaptation modules, similarity matrices, arrows, and legend.
- Row (a): both towers are frozen; normal/abnormal text prototypes stay locked and no cross-tower adaptation bridge exists.
- Row (b): abnormal visual tokens are marked in the image tower and a red dashed cross-tower update shows direct contamination of p_N into p_N'.
- Row (c): a wavelet adapter is inserted inside the image tower; DWT separates LL pass and high-frequency suppression before a gated update reaches p_N in the text tower.
- Prompt coverage: Normal: "normal cable" and Abnormal: "damaged cable" are visible inside every text tower.
- Visual QA target: enlarged towers, readable labels, no intentional text overlap, and clear distinction between fixed, drift, and wavelet-gated structures.

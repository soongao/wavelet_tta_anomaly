# Figure 4 enlarged CLIP dual-tower v3 audit

- Canvas: 2600 x 1580 px, three stacked rows with a larger CLIP dual-tower container in every row.
- Visual intent: make the image tower and text tower the dominant structure, then place the architectural difference in a cross-tower feedback branch rather than as a post-processing chain.
- Real raster inputs: one MVTec cable image and three anomaly-map outputs are embedded as PNG images.
- Native/editable elements: row titles, prompt labels, tower lanes, encoders, tokens, prototypes, feedback branch, similarity matrices, arrows, and legend.
- Row (a): both towers are frozen; fixed visual features and fixed text prototypes go directly to the CLIP similarity head with no feedback branch.
- Row (b): raw visual tokens branch out from the image tower and feed back to the text prototype side without filtering, so defect evidence can drift p_N into p_N'.
- Row (c): wavelet feedback branches from image patch embeddings; DWT separates LL pass and high-frequency suppression before a gated update reaches p_N in the text tower.
- Prompt coverage: Normal: "normal cable" and Abnormal: "damaged cable" are visible inside every text tower.
- Visual QA target: enlarged towers, readable labels, no intentional text overlap, and clear distinction between fixed matching, raw TTA drift, and wavelet-gated prototype adaptation.

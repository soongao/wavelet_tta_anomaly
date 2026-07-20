# Layout Fingerprint

- Reference: `/Users/bytedance/Downloads/image_1784538888.jpeg`
- Canvas: 6784 x 2496 px, landscape scientific workflow figure.
- Major regions:
  - Top-left Frozen CLIP text inputs: x=0, y=0, w=2350, h=1220.
  - Bottom-left Frozen CLIP visual inputs: x=0, y=1275, w=2350, h=1065.
  - Center per-image evidence construction: x=2345, y=0, w=2315, h=2340.
  - Prototype calibration: header x=4670, y=0, w=2114, h=210; main panel x=4410, y=1030, w=1470, h=1310.
  - Output heads: patch score panel x=6160, y=245, w=615, h=1035; image score panel x=6160, y=1430, w=615, h=905.
- Reading order: text tokens and visual patches feed into a central evidence gate; selected evidence creates visual prototypes; a mixer calibrates text prototypes; outputs feed patch and image score heads.
- Editable element count: approximately 260 draw.io cells, mostly native rectangles, labels, connectors, and small grid cells.
- Connector directions: left-to-right primary flow with dashed evidence links and a bottom dashed route to the image-score head.
- Known uncertainty: small heatmap values and the input texture are visually approximated; exact pixel heatmaps from the source are not embedded.

# Wavelet Feature Visualization

This folder contains cache-based analysis tools for comparing AnomalyCLIP patch
features and anomaly scores before and after wavelet calibration.

Important: the current wavelet module is a post-processing calibration on the
anomaly map. It does not re-encode the image or create a new CLIP patch
embedding. Therefore the raw feature projection is mainly a reference view; the
wavelet effect should be judged from score distributions, score deltas, gates,
and qualitative maps.

The main entry point is:

```bash
python3 tools/wavelet_feature_viz/visualize_wavelet_feature_distribution.py \
  --cache_dir cache/mvtec_anomalyclip_features \
  --save_dir outputs/wavelet_feature_viz/mvtec_demo \
  --classes bottle \
  --max_samples 24
```

Expected outputs:

- `score_distribution.png`: normal/anomaly patch score distributions before
  and after wavelet calibration.
- `feature_embedding.png`: PCA or t-SNE projection of raw CLIP patch tokens
  and of the score/gate context, colored by ground-truth patch label.
- `calibration_effect.png`: baseline-vs-wavelet score scatter, score delta
  distribution, and gate-vs-delta scatter.
- `qualitative_examples/*.png`: image-level panels with input image, GT mask,
  baseline map, wavelet/texture gate, calibrated map, and score delta.
- `summary.json`: numeric summary for score separation and gate/GT overlap.

For result-backed figures, use:

```bash
python3 tools/wavelet_feature_viz/visualize_experiment_effect.py \
  --save_dir outputs/wavelet_feature_viz/mvtec_experiment_effect
```

This script reads the baseline and method logs, ranks classes by pixel AUPRO
improvement, then recomputes baseline/method maps for the selected class. The
default MVTec logs select `zipper`, where the full method improves pixel AUPRO
from 73.1 to 83.4. Its outputs are:

- `class_ranking.tsv`: per-class metric deltas used to pick the class.
- `experiment_metric_comparison.png`: baseline vs method metrics from logs.
- `pixel_score_distribution.png`: normal/anomaly pixel score distributions.
- `localization_examples/*.png`: image, GT mask, baseline map, method map, and
  map delta for samples where localization improves.
- `summary.json`: selected class, log metrics, and recomputed metrics.

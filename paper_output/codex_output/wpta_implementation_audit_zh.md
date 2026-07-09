# WPTA 实现细节与叙事一致性审计

审计来源：

- `/Users/bytedance/code/AnomalyCLIP/FIVE_DATASET_RESULTS_AND_ABLATIONS.md`
- `/Users/bytedance/code/AnomalyCLIP/conf/run_ablation_experiments_conf.yaml`
- `/Users/bytedance/code/AnomalyCLIP/scripts/evaluate/eval_cached_calibration.py`
- `/Users/bytedance/code/AnomalyCLIP/src/anomalyclip/wavelet_calibration.py`

## 1. 主结果 setting 审计

| Dataset | Final result | Enabled modules in final command | Narrative risk |
|---|---:|---|---|
| MVTec | 91.8 / 85.6 / 94.5 / 97.6 | wavelet, wavelet confidence, TTA rectification, multicrop, pixel-to-image fusion | Can be discussed as full system with WPTA components. |
| VisA | 96.2 / 91.3 / 84.6 / 87.4 | wavelet, wavelet confidence, TTA rectification, multicrop, pixel-to-image fusion | Can be discussed as full system with WPTA components. |
| MPDD | 97.3 / 89.9 / 77.8 / 82.3 | multicrop, pixel-to-image fusion; no wavelet/TTA flags in final command | Cannot attribute gain to wavelet-guided prototype adaptation without extra ablation. |
| BTAD | 96.3 / 78.2 / 93.9 / 94.9 | multicrop, pixel-to-image fusion; no wavelet/TTA flags in final command | Cannot attribute gain to wavelet-guided prototype adaptation without extra ablation. |
| DTD-Synthetic | 97.9 / 91.8 / 96.9 / 98.7 | wavelet, wavelet confidence, multicrop, pixel-to-image fusion; no TTA rectification flag | Can support wavelet reliability in final system, but not test-time prototype adaptation. |

## 2. Controlled WPTA setting on MVTec/VisA

The controlled component ablation uses:

- feature layers: 1, 2, 3
- sigma: 5
- layer weighting: sum, temperature 1.0
- wavelet mode: dual_route
- wavelet beta: 0.2
- wavelet condition power: 2.0
- wavelet suppress beta: 0.0
- wavelet fusion: mean
- wavelet levels: 2
- wavelet level fusion: mean
- texture edge power: 1.0
- texture max delta ratio: 0.05
- texture local contrast kernel: 17
- texture local contrast weight: 0.5
- rank preserve top-k ratio: 0.35
- rank gate mode: hard
- rank gate temperature: 0.05
- use wavelet confidence: true
- wavelet confidence power: 1.0
- TTA mode: wavelet_guided
- TTA alpha: 0.01
- TTA top-k ratio: 0.02
- TTA min confidence: 0.2
- TTA anchor layers: mean
- TTA repulsion weight: 0.1
- TTA abnormal alpha scale: 0.75
- multicrop weight: 0.5
- pixel-to-image weight: 0.1
- pixel-to-image top-k ratio: 0.01

This setting supports the WPTA method claim on MVTec and VisA.

## 3. Required narrative correction

The paper must distinguish two claims:

1. **WPTA method claim**: On MVTec and VisA controlled ablations, wavelet-guided prototype adaptation improves over semantic-only adaptation and direct wavelet fusion.
2. **Final system result claim**: A final calibrated system improves over AnomalyCLIP baseline across five industrial datasets.

It is not currently valid to claim:

- “WPTA prototype adaptation improves all five datasets,” because MPDD and BTAD final commands do not enable wavelet/TTA.
- “Every five-dataset gain comes from wavelet-supervised prototype adaptation,” because several final settings include multicrop and pixel-to-image fusion as main active modules.

## 4. Recommended v0.4 wording

Use:

> The full calibrated system improves the AnomalyCLIP baseline on five industrial benchmarks. The WPTA mechanism itself is validated through controlled MVTec/VisA ablations, where direct wavelet fusion degrades performance while wavelet-guided evidence selection and conservative prototype calibration improve AUPRO.

Avoid:

> WPTA improves all five datasets through wavelet-supervised prototype adaptation.

## 5. Impact on score

If v0.3 is submitted as-is, a careful reviewer can challenge the causal attribution of five-dataset improvements. This is a MAJOR issue, not merely a wording issue. v0.4 must fix the claim boundary.

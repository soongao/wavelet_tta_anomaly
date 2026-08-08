# NCMA MVTec AD Final Results

Protocol: target-category zero-shot anomaly detection on MVTec AD.

Metrics: pixel AUROC / pixel AUPRO / image AUROC / image AP.

## AnomalyCLIP Baseline

| objects    | pixel_auroc | pixel_aupro | image_auroc | image_ap |
|:-----------|------------:|------------:|------------:|---------:|
| carpet     | 98.8 | 90.0 | 100.0 | 100.0 |
| bottle     | 90.4 | 80.8 | 88.7 | 96.8 |
| hazelnut   | 97.2 | 92.5 | 97.2 | 98.5 |
| leather    | 98.6 | 92.2 | 99.8 | 99.9 |
| cable      | 78.9 | 64.0 | 70.3 | 81.7 |
| capsule    | 95.8 | 87.6 | 89.5 | 97.8 |
| grid       | 97.3 | 75.4 | 97.8 | 99.3 |
| pill       | 91.8 | 88.1 | 81.1 | 95.3 |
| transistor | 70.8 | 58.2 | 93.9 | 92.1 |
| metal_nut  | 74.6 | 71.1 | 92.4 | 98.2 |
| screw      | 97.5 | 88.0 | 82.1 | 92.9 |
| toothbrush | 91.9 | 88.5 | 85.3 | 93.9 |
| zipper     | 91.3 | 65.4 | 98.4 | 99.5 |
| tile       | 94.7 | 87.4 | 100.0 | 100.0 |
| wood       | 96.4 | 91.5 | 96.9 | 99.2 |
| mean       | 91.1 | 81.4 | 91.6 | 96.4 |

## NCMA Final

| objects    | pixel_auroc | pixel_aupro | image_auroc | image_ap |
|:-----------|------------:|------------:|------------:|---------:|
| carpet     | 99.0 | 91.9 | 100.0 | 100.0 |
| bottle     | 92.3 | 84.2 | 90.4 | 97.3 |
| hazelnut   | 97.7 | 93.5 | 97.6 | 98.7 |
| leather    | 98.8 | 93.6 | 99.8 | 99.9 |
| cable      | 83.6 | 70.7 | 74.5 | 84.3 |
| capsule    | 96.4 | 89.6 | 89.5 | 97.8 |
| grid       | 97.6 | 80.1 | 97.9 | 99.3 |
| pill       | 93.4 | 90.3 | 83.3 | 95.9 |
| transistor | 76.8 | 64.1 | 94.2 | 92.6 |
| metal_nut  | 80.5 | 77.3 | 93.5 | 98.5 |
| screw      | 97.8 | 89.3 | 82.1 | 92.9 |
| toothbrush | 93.8 | 89.9 | 87.5 | 94.8 |
| zipper     | 93.0 | 72.5 | 98.5 | 99.5 |
| tile       | 95.5 | 89.0 | 100.0 | 100.0 |
| wood       | 96.6 | 92.4 | 97.3 | 99.3 |
| mean       | 92.9 | 84.6 | 92.4 | 96.7 |

## NCMA - AnomalyCLIP

| objects    | pixel_auroc | pixel_aupro | image_auroc | image_ap |
|:-----------|------------:|------------:|------------:|---------:|
| carpet     | +0.2 | +1.9 | +0.0 | +0.0 |
| bottle     | +1.9 | +3.4 | +1.7 | +0.5 |
| hazelnut   | +0.5 | +1.0 | +0.4 | +0.2 |
| leather    | +0.2 | +1.4 | +0.0 | +0.0 |
| cable      | +4.7 | +6.7 | +4.2 | +2.6 |
| capsule    | +0.6 | +2.0 | +0.0 | +0.0 |
| grid       | +0.3 | +4.7 | +0.1 | +0.0 |
| pill       | +1.6 | +2.2 | +2.2 | +0.6 |
| transistor | +6.0 | +5.9 | +0.3 | +0.5 |
| metal_nut  | +5.9 | +6.2 | +1.1 | +0.3 |
| screw      | +0.3 | +1.3 | +0.0 | +0.0 |
| toothbrush | +1.9 | +1.4 | +2.2 | +0.9 |
| zipper     | +1.7 | +7.1 | +0.1 | +0.0 |
| tile       | +0.8 | +1.6 | +0.0 | +0.0 |
| wood       | +0.2 | +0.9 | +0.4 | +0.1 |
| mean       | +1.8 | +3.2 | +0.8 | +0.3 |

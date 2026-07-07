# Closest Prior Work Table

| Method family | CLIP-based ZSAD | Fixed text prototypes | Test-time adaptation | Frequency / wavelet cue | Patch-level reliability selection | Boundary-aware spectral cue | Prototype calibration | Training-free |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| AnomalyCLIP | yes | yes | no | no | no | no | no | no |
| WinCLIP | yes | yes | no | no | window-level scoring | no | no | yes |
| CLIP-AD / PromptAD / AdaCLIP style methods | yes | partly | usually training or prompt adaptation | no | sometimes | no | prompt/prototype related | mixed |
| Test-time prompt tuning / prototype adaptation | often | no | yes | no | semantic confidence | no | yes | mixed |
| Frequency / wavelet anomaly detection | no or hybrid | no | no | yes | frequency saliency | sometimes | no | often |
| Ours | yes | calibrated at test time | yes | yes | semantic-spectral patch evidence | yes | yes | yes |

Positioning: the proposed method combines patch-level semantic-spectral reliability, boundary-aware wavelet evidence, and training-free prototype calibration. Unlike direct map fusion, the wavelet cue changes the patch evidence that builds image-specific visual prototypes, and the final anomaly map is recomputed with calibrated normal/abnormal text prototypes.

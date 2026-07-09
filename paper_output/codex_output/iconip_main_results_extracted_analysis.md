# 从 `paper_iconip/main.tex` 提取的其他方法主结果与可用性分析

来源文件：`/Users/bytedance/mypaper/paper_iconip/main.tex`

提取位置：`Table~\ref{tab:industry_result}`，约第 229-260 行。

## 1. 原文协议说明

`main.tex` 中的工业主结果表覆盖 5 个工业基准：MVTec AD、VisA、MPDD、BTAD、DTD-Synthetic。

原文实验协议写法：

- 使用 ZSAD auxiliary-target split。
- MVTec AD 和 VisA 类别与样本不重叠。
- 在 VisA 上 fine-tune 的模型评估 MVTec AD。
- 在 MVTec AD 上 fine-tune 的模型评估 VisA 和其他数据集。
- 使用 frozen OpenCLIP ViT-L/14，输入 resize 到 `518 x 518`。
- baseline 结果来自相同 backbone 和 dataset-split protocol 下的公开结果；不可直接比较时，原文作者 rerun 官方实现。

这和当前 WPTA 草稿里的 `training-free / no backpropagation / no target training data` 叙事不完全相同。因此，下面结果可以作为“候选外部方法对比数据源”，但正式写入 WPTA 论文前必须确认协议一致性。

## 2. 原文工业主结果：Image-level

指标顺序：`Image-AUROC / Image-AP`。

| Dataset | CLIP | WinCLIP | VAND | CoOp | AdaCLIP | AnomalyCLIP | AA-CLIP† | Source Ours (TAAP/INPC) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MVTec AD | 74.1 / 87.6 | 91.8 / 96.5 | 86.1 / 93.5 | 88.8 / 94.8 | 89.2 / 96.4 | 91.5 / 96.2 | 90.5 / 94.9 | **92.8 / 96.7** |
| VisA | 66.4 / 71.5 | 78.1 / 81.2 | 78.0 / 81.4 | 62.8 / 68.1 | **85.8** / 84.9 | 82.1 / 85.4 | 84.6 / 82.2 | 83.3 / **85.9** |
| MPDD | 54.3 / 65.4 | 63.6 / 69.9 | 73.0 / 80.2 | 55.1 / 64.2 | 76.0 / 80.4 | 77.0 / 82.0 | 75.1 / 80.1 | **81.8 / 84.8** |
| BTAD | 34.5 / 52.5 | 68.2 / 70.9 | 73.6 / 68.6 | 66.8 / 77.4 | 88.6 / 92.4 | 88.3 / 87.3 | **94.8 / 97.9** | 93.3 / 93.1 |
| DTD-Synthetic | 71.6 / 85.7 | 93.2 / 92.6 | 86.4 / 95.0 | - / - | 95.5 / 97.0 | 93.5 / 97.0 | 93.3 / 97.8 | **97.5 / 99.0** |
| Average | 60.2 / 72.5 | 79.0 / 82.2 | 79.4 / 83.7 | 68.4 / 76.1 | 87.0 / 90.2 | 86.5 / 89.6 | 87.7 / 90.6 | **89.7 / 91.9** |

## 3. 原文工业主结果：Pixel-level

指标顺序：`Pixel-AUROC / Pixel-AUPRO`。

| Dataset | CLIP | WinCLIP | VAND | CoOp | AdaCLIP | AnomalyCLIP | AA-CLIP† | Source Ours (TAAP/INPC) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MVTec AD | 38.4 / 11.3 | 85.1 / 64.6 | 87.6 / 44.0 | 33.3 / 6.7 | 88.7 / 37.8 | 91.1 / 81.4 | **91.9** / 84.6 | 91.5 / **85.5** |
| VisA | 46.6 / 14.8 | 79.6 / 56.8 | 94.2 / 86.8 | 24.2 / 3.8 | 95.5 / 56.8 | 95.4 / 87.0 | 95.5 / 83.0 | **95.6 / 88.3** |
| MPDD | 62.1 / 33.0 | 76.4 / 48.9 | 94.1 / 83.2 | 15.4 / 2.3 | 96.1 / 60.3 | 96.5 / 88.7 | **96.7** / 76.5 | **96.7 / 89.9** |
| BTAD | 30.6 / 4.4 | 72.7 / 27.3 | 60.8 / 25.0 | 28.6 / 3.8 | 92.1 / 32.5 | 94.2 / 74.8 | **97.0** / 69.0 | 94.1 / **77.1** |
| DTD-Synthetic | 33.9 / 12.5 | 83.9 / 57.8 | 95.3 / 86.9 | - / - | 97.7 / 75.0 | **97.9 / 92.3** | 96.4 / 85.9 | 97.4 / 91.5 |
| Average | 42.3 / 15.2 | 79.5 / 51.1 | 86.4 / 65.2 | 25.4 / 4.2 | 94.0 / 52.5 | 95.0 / 84.8 | **95.5** / 79.8 | 95.1 / **86.5** |

## 4. 与当前 WPTA 数据的直接相关子集

这里把 `main.tex` 的 MVTec AD / VisA 外部方法结果与当前 WPTA `current` 结果放在一起，方便判断是否可生成外部对比表。

指标顺序：`P-AUROC / P-AUPRO / I-AUROC / I-AP`。

| Dataset | Method | P-AUROC ↑ | P-AUPRO ↑ | I-AUROC ↑ | I-AP ↑ | Source |
|---|---|---:|---:|---:|---:|---|
| MVTec | CLIP | 38.4 | 11.3 | 74.1 | 87.6 | `main.tex` |
| MVTec | WinCLIP | 85.1 | 64.6 | 91.8 | 96.5 | `main.tex` |
| MVTec | VAND | 87.6 | 44.0 | 86.1 | 93.5 | `main.tex` |
| MVTec | CoOp | 33.3 | 6.7 | 88.8 | 94.8 | `main.tex` |
| MVTec | AdaCLIP | 88.7 | 37.8 | 89.2 | 96.4 | `main.tex` |
| MVTec | AnomalyCLIP | 91.1 | 81.4 | 91.5 | 96.2 | `main.tex` |
| MVTec | AA-CLIP† | **91.9** | 84.6 | 90.5 | 94.9 | `main.tex` |
| MVTec | Source Ours (TAAP/INPC) | 91.5 | 85.5 | 92.8 | 96.7 | `main.tex` |
| MVTec | WPTA current | 91.8 | **86.2** | **94.1** | **97.4** | current CSV |
| VisA | CLIP | 46.6 | 14.8 | 66.4 | 71.5 | `main.tex` |
| VisA | WinCLIP | 79.6 | 56.8 | 78.1 | 81.2 | `main.tex` |
| VisA | VAND | 94.2 | 86.8 | 78.0 | 81.4 | `main.tex` |
| VisA | CoOp | 24.2 | 3.8 | 62.8 | 68.1 | `main.tex` |
| VisA | AdaCLIP | 95.5 | 56.8 | **85.8** | 84.9 | `main.tex` |
| VisA | AnomalyCLIP | 95.4 | 87.0 | 82.1 | 85.4 | `main.tex` |
| VisA | AA-CLIP† | 95.5 | 83.0 | 84.6 | 82.2 | `main.tex` |
| VisA | Source Ours (TAAP/INPC) | 95.6 | 88.3 | 83.3 | 85.9 | `main.tex` |
| VisA | WPTA current | **96.2** | **91.7** | 84.3 | **87.3** | current CSV |

## 5. 关键分析

### 5.1 这些数据确实提供了其他 method 的主结果

`main.tex` 中可提取的外部方法包括：CLIP、WinCLIP、VAND、CoOp、AdaCLIP、AnomalyCLIP、AA-CLIP，以及该源论文自己的 Ours (TAAP/INPC)。对于 WPTA 当前关注的 MVTec 和 VisA，这些方法都有 image-level 与 pixel-level 主结果。

### 5.2 WPTA 在 MVTec/VisA 上的相对位置较强

若暂时把 `main.tex` 的外部结果与当前 WPTA 结果放在同一候选表中，WPTA 的表现如下：

- MVTec：WPTA 在 P-AUPRO、I-AUROC、I-AP 上最高；P-AUROC 为 91.8，仅比 `main.tex` 中 AA-CLIP 的 91.9 低 0.1。
- VisA：WPTA 在 P-AUROC、P-AUPRO、I-AP 上最高；I-AUROC 为 84.3，低于 AdaCLIP 的 85.8 和 AA-CLIP 的 84.6。
- 最强叙事仍然是 AUPRO：WPTA 在 MVTec 上 P-AUPRO 为 86.2，高于 Source Ours 的 85.5 和 AA-CLIP 的 84.6；在 VisA 上 P-AUPRO 为 91.7，高于 Source Ours 的 88.3 和 AnomalyCLIP 的 87.0。

### 5.3 当前 WPTA baseline 与 `main.tex` 的 AnomalyCLIP baseline 不完全一致

当前 CSV 与 `main.tex` 中 AnomalyCLIP 数值接近，但不是完全相同：

| Dataset | Source | P-AUROC | P-AUPRO | I-AUROC | I-AP |
|---|---|---:|---:|---:|---:|
| MVTec | current CSV AnomalyCLIP | 91.2 | 83.2 | 91.6 | 96.4 |
| MVTec | `main.tex` AnomalyCLIP | 91.1 | 81.4 | 91.5 | 96.2 |
| VisA | current CSV AnomalyCLIP | 95.5 | 86.7 | 82.0 | 85.3 |
| VisA | `main.tex` AnomalyCLIP | 95.4 | 87.0 | 82.1 | 85.4 |

MVTec 的 P-AUPRO 差异达到 1.8，这说明两份结果可能在 split、post-processing、implementation 或 evaluation script 上有差别。正式论文里如果使用 `main.tex` 的外部方法主结果，就不应该继续把 current CSV 的 baseline delta 与 `main.tex` 的外部方法混在同一列里解释。

### 5.4 是否能写 SOTA

目前还不能直接写 SOTA。原因不是数值不强，而是协议需要核验：

- `main.tex` 的协议是 auxiliary-target ZSAD，并涉及 fine-tuning on auxiliary dataset。
- 当前 WPTA 草稿强调 training-free、test-time、no backpropagation。
- 如果 WPTA current 结果是在完全不同的 training-free 协议下跑出来的，直接和 `main.tex` 的 fine-tuned ZSAD 表比较会被审稿人质疑。

更稳妥写法：

> Compared with representative CLIP-based ZSAD results reported under a similar OpenCLIP ViT-L/14 industrial protocol, WPTA is competitive on MVTec and VisA and achieves the strongest Pixel-AUPRO among the collected methods. We report this comparison as a protocol-aligned reference and keep the main claim on the fixed AnomalyCLIP baseline.

中文审稿版：

> 与 `main.tex` 中相近 OpenCLIP ViT-L/14 工业协议下的代表性 CLIP-based ZSAD 结果相比，WPTA 在 MVTec 和 VisA 上具有竞争力，并在 Pixel-AUPRO 上达到收集方法中的最高值。但由于当前 WPTA 的 training-free 协议与该表的 auxiliary-target fine-tuning 协议仍需核验，主 claim 应继续写成超过固定 AnomalyCLIP baseline。

## 6. 建议怎么用

建议生成一个“候选外部方法对比表”，但表注必须写清楚：

- external method numbers are extracted from `main.tex`;
- WPTA numbers are from current CSV;
- protocols are similar but not yet fully verified;
- do not use the table alone to claim SOTA before confirming identical split, backbone, input size, post-processing, and evaluation script.

如果要放进当前中文稿，推荐作为补充表或 “Comparison with representative reported methods”，而不是替换原来的 main result 表。

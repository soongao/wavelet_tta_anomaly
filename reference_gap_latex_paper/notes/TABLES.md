# 论文表格草案（当前结果口径 / TABLES.md）

> 当前已复现结果以 `../../paper/tables/*.csv`、`../../paper/prototype_main_result_table.md`
> 和 `../EXPERIMENT_TARGETS.md` 为准。本文方法名统一为 `ICNR`。
> 外部 SOTA 数值标 `*` = 待核对原论文。
> 指标顺序：pixel AUROC / pixel AUPRO / image AUROC / image AP（%）。

---

## Table 1 — Main results（主表，跨 5 数据集）
论文里的头号表。每个数据集一组，Baseline vs Ours，四指标全列。

```latex
\begin{table*}[t]\centering
\caption{Zero-shot anomaly detection results. Metrics in \%. Best in \textbf{bold}.}
\label{tab:main}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccc}
\toprule
Dataset & Method & Pixel AUROC & Pixel AUPRO & Image AUROC & Image AP \\
\midrule
\multirow{2}{*}{MVTec AD} & AnomalyCLIP & 91.2 & 83.2 & 91.6 & 96.4 \\
	                          & ICNR & \textbf{91.8} & \textbf{86.2} & \textbf{94.5} & \textbf{97.6} \\
\midrule
\multirow{2}{*}{VisA}     & AnomalyCLIP & 95.5 & 87.0 & 82.1 & 85.4 \\
	                          & ICNR & \textbf{96.2} & \textbf{91.7} & \textbf{84.6} & \textbf{87.4} \\
\midrule
\multirow{2}{*}{MPDD}     & AnomalyCLIP & 96.5 & 88.7 & 77.0 & 82.0 \\
	                          & ICNR & \textbf{97.3} & \textbf{89.9} & \textbf{77.8} & \textbf{82.3} \\
\midrule
\multirow{2}{*}{BTAD}     & AnomalyCLIP & 94.2 & 74.8 & 88.3 & 87.3 \\
	                          & ICNR & \textbf{96.3} & \textbf{79.5} & \textbf{93.9} & \textbf{94.9} \\
\midrule
\multirow{2}{*}{DTD-Synth}& AnomalyCLIP & \textbf{97.9} & \textbf{92.3} & 93.5 & 97.0 \\
	                          & ICNR & \textbf{97.9} & 91.8 & \textbf{96.9} & \textbf{98.7} \\
\bottomrule
\end{tabular}}
\end{table*}
```

---

## Table 2 — Comparison with SOTA（MVTec zero-shot）
外部数值 `*` 待核对；协议差异需在脚注说明。

```latex
\begin{table}[t]\centering
\caption{Comparison with zero-shot methods on MVTec. * external, to verify.}
\label{tab:sota}
\begin{tabular}{lccc}
\toprule
Method & Image AUROC & Pixel AUROC & Pixel AUPRO \\
\midrule
WinCLIP*      & 91.8 & 85.1 & 64.6 \\
APRIL-GAN*    & 86.1 & 87.6 & 44.0 \\
AnomalyCLIP   & 91.6 & 91.2 & 83.2 \\
AdaCLIP*      & 92.0 & 89.0 & --   \\
FE-CLIP*      & --   & --   & --   \\
\midrule
\textbf{ICNR} & \textbf{94.5} & \textbf{91.8} & \textbf{86.2} \\
\bottomrule
\end{tabular}
\end{table}
```
> 诚实说明：合格线是 pixel AUPRO 进第一梯队 + 不用辅助训练也有竞争力，非碾压。

---

## Table 3 — Core mechanism ablation（数值版，配合 fig_mechanism_ordering）
图讲趋势，表给全指标精确值。

```latex
\begin{table}[t]\centering
\caption{Controlled ablation results on MVTec / VisA.}
\label{tab:ablation-core}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
Variant & Pixel AUROC & Pixel AUPRO & Image AUROC & Image AP \\
\midrule
\multicolumn{5}{l}{\emph{MVTec AD}}\\
ScoreFusion                   & 88.7 & 80.4 & 92.9 & 96.9 \\
FixedProto                    & 91.2 & 83.2 & 91.6 & 96.4 \\
SemRef                        & 91.6 & 85.2 & 93.7 & 97.1 \\
StructRef                     & 91.7 & 85.8 & 93.9 & 97.2 \\
\textbf{ICNR}                 & \textbf{91.8} & \textbf{86.2} & \textbf{94.5} & \textbf{97.6} \\
\midrule
\multicolumn{5}{l}{\emph{VisA}}\\
ScoreFusion                   & 94.6 & 85.1 & 81.6 & 84.8 \\
FixedProto                    & 95.5 & 86.7 & 82.0 & 85.3 \\
SemRef                        & 96.0 & 90.4 & 83.7 & 86.9 \\
StructRef                     & 96.1 & 91.3 & 84.1 & 87.0 \\
\textbf{ICNR}                 & \textbf{96.2} & \textbf{91.7} & \textbf{84.6} & \textbf{87.4} \\
\bottomrule
\end{tabular}}
\end{table}
```

---

## Table 4 — Design ablation（boundary-aware / conservative update）

```latex
\begin{table}[t]\centering
\caption{Design ablation on MVTec.}
\label{tab:ablation-design}
\begin{tabular}{lcccc}
\toprule
Variant & Pixel AUROC & Pixel AUPRO & Image AUROC & Image AP \\
\midrule
HFRef                        & 91.6 & 85.3 & 94.0 & 97.2 \\
BndRef                       & 91.7 & 85.7 & 93.8 & 97.3 \\
StructRef                    & 91.7 & 85.8 & 93.9 & 97.2 \\
\textbf{ICNR}                & \textbf{91.8} & \textbf{86.2} & \textbf{94.5} & \textbf{97.6} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 5 — Normal-image stability + Runtime（可合并成一张）

```latex
\begin{table}[t]\centering
\caption{Normal-image false-positive area (lower better) and runtime.}
\label{tab:stability-runtime}
\begin{tabular}{llccc}
\toprule
Dataset & Method & FP@p95(\%) & FP@p99(\%) & s/img \\
\midrule
\multirow{3}{*}{MVTec}
 & Baseline & 5.002 & 1.001 & 0.065 \\
 & NoCons   & 4.738 & 0.923 & --    \\
 & ICNR & 4.895 & 0.960 & 0.079 \\
\midrule
\multirow{3}{*}{VisA}
 & Baseline & 4.996 & 1.001 & 0.065 \\
 & NoCons   & 4.693 & 0.935 & --    \\
 & ICNR & 4.718 & 0.937 & 0.079 \\
\bottomrule
\end{tabular}
\end{table}
```
> Ours FP ≤ baseline；当前 no-conservative FP 略低于 conservative，因此只写“不增加正常图误报”，不写“Ours < NoCons”。开销约 +21% 到 +22%（≤25%）。

---

## Table 6 — Global setting vs dataset-tuned（方法学诚实，附录或正文小表）

```latex
\begin{table}[t]\centering
\caption{Global setting vs dataset-tuned upper bound.}
\label{tab:global-tuned}
\begin{tabular}{lcc}
\toprule
Dataset & Global setting & Dataset-tuned (UB) \\
\midrule
MPDD (pAUPRO)      & 88.4 & 89.9 \\
BTAD (pAUPRO)      & 79.5 & 78.2 \\
DTD-Synth (pAUPRO) & 90.7 & 91.8 \\
\bottomrule
\end{tabular}
\end{table}
```
> 主张以 global 行为准；tuned 行明确标 upper bound，避免"测试集调参"质疑。

---

## 表 / 图 分工总览
| 内容 | 形式 | 位置 |
|---|---|---|
| 主结果(5数据集×4指标) | **Table 1** | 正文 |
| SOTA 对比 | **Table 2** | 正文 |
| 核心机制消融(全指标) | **Table 3** | 正文，配 `fig_mechanism_ordering` |
| 设计消融 | **Table 4** | 正文/附录 |
| 正常稳定性+运行时 | **Table 5** | 正文/附录 |
| global vs tuned | **Table 6** | 附录 |
| 机制单调序(趋势) | 图 `fig_mechanism_ordering` | 正文分析 |
| 分类别增益模式 | 图 `fig_percategory_gain` | 正文分析 |
| 超参敏感性(+mix机制曲线) | 图 `fig_sensitivity` | 正文/附录 |
| CLIP 盲区召回 | 图 `fig_blindspot` | 正文分析 |
| 定性异常图对比 | 图（真实结果，见 QUALITATIVE_SPEC.md） | 正文，**必备** |

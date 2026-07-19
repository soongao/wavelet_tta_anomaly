# 论文表格骨架（TABLES.md）

> 这些结果在论文里应当用**表格**呈现（不是图）。数值均为 EXPERIMENT_PLAN_PAPER 的
> **目标值（EXPECTED）**，做完实验用真实 log 替换。外部 SOTA 数值标 `*` = 待核对原论文。
> 指标顺序：pixel AUROC / pixel AUPRO / image AUROC / image AP（%）。

---

## Table 1 — Main results（主表，跨 5 数据集）
论文里的头号表。每个数据集一组，Baseline vs Ours，四指标全列。

```latex
\begin{table*}[t]\centering
\caption{Zero-shot anomaly detection results. Metrics in \%. Best in \textbf{bold}. (EXPECTED)}
\label{tab:main}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccc}
\toprule
Dataset & Method & Pixel AUROC & Pixel AUPRO & Image AUROC & Image AP \\
\midrule
\multirow{2}{*}{MVTec AD} & AnomalyCLIP & 91.1 & 81.4 & 91.5 & 96.2 \\
                          & Ours        & \textbf{92.4} & \textbf{85.8} & \textbf{94.2} & \textbf{97.5} \\
\midrule
\multirow{2}{*}{VisA}     & AnomalyCLIP & 95.5 & 87.0 & 82.1 & 85.4 \\
                          & Ours        & \textbf{96.5} & \textbf{91.0} & \textbf{85.3} & \textbf{88.2} \\
\midrule
\multirow{2}{*}{MPDD}     & AnomalyCLIP & 96.5 & 88.7 & 77.0 & 80.2 \\
                          & Ours        & \textbf{97.2} & \textbf{90.5} & \textbf{80.0} & \textbf{83.5} \\
\midrule
\multirow{2}{*}{BTAD}     & AnomalyCLIP & 94.2 & 74.8 & 89.5 & 91.5 \\
                          & Ours        & \textbf{95.8} & \textbf{78.5} & \textbf{92.0} & \textbf{93.5} \\
\midrule
\multirow{2}{*}{DTD-Synth}& AnomalyCLIP & 97.0 & 89.5 & 94.0 & 97.2 \\
                          & Ours        & \textbf{97.8} & \textbf{91.5} & \textbf{96.0} & \textbf{98.3} \\
\bottomrule
\end{tabular}}
\end{table*}
```

---

## Table 2 — Comparison with SOTA（MVTec zero-shot）
外部数值 `*` 待核对；协议差异需在脚注说明。

```latex
\begin{table}[t]\centering
\caption{Comparison with zero-shot methods on MVTec. * external, to verify. (Ours EXPECTED)}
\label{tab:sota}
\begin{tabular}{lccc}
\toprule
Method & Image AUROC & Pixel AUROC & Pixel AUPRO \\
\midrule
WinCLIP*      & 91.8 & 85.1 & 64.6 \\
APRIL-GAN*    & 86.1 & 87.6 & 44.0 \\
AnomalyCLIP   & 91.5 & 91.1 & 81.4 \\
AdaCLIP*      & 92.0 & 89.0 & --   \\
FE-CLIP*      & --   & --   & --   \\
\midrule
\textbf{Ours} & \textbf{94.2} & \textbf{92.4} & \textbf{85.8} \\
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
\caption{Mechanism ablation: source of the normal reference. MVTec / VisA. (EXPECTED)}
\label{tab:ablation-core}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
Variant & Pixel AUROC & Pixel AUPRO & Image AUROC & Image AP \\
\midrule
\multicolumn{5}{l}{\emph{MVTec AD}}\\
DirectHF (read HF, no ref) & 82.5 & 73.0 & 93.4 & 97.1 \\
Baseline                   & 91.1 & 81.4 & 91.5 & 96.2 \\
GlobalRef (fixed)          & 91.5 & 83.2 & 93.0 & 96.9 \\
SelfRef (CLIP-only)        & 91.9 & 84.6 & 93.9 & 97.3 \\
\textbf{Ours}              & \textbf{92.4} & \textbf{85.8} & \textbf{94.2} & \textbf{97.5} \\
\midrule
\multicolumn{5}{l}{\emph{VisA}}\\
DirectHF & 90.8 & 82.5 & 84.0 & 86.5 \\
Baseline & 95.5 & 87.0 & 82.1 & 85.4 \\
GlobalRef& 95.8 & 88.2 & 83.5 & 86.6 \\
SelfRef  & 96.1 & 89.8 & 84.4 & 87.3 \\
\textbf{Ours} & \textbf{96.5} & \textbf{91.0} & \textbf{85.3} & \textbf{88.2} \\
\bottomrule
\end{tabular}}
\end{table}
```

---

## Table 4 — Design ablation（boundary-aware / conservative update）

```latex
\begin{table}[t]\centering
\caption{Design ablation on MVTec. (EXPECTED)}
\label{tab:ablation-design}
\begin{tabular}{lcccc}
\toprule
Variant & Pixel AUROC & Pixel AUPRO & Image AUROC & Image AP \\
\midrule
HFonly (raw HF)          & 92.0 & 85.0 & 94.1 & 97.4 \\
NoCons (no conserv.)     & 92.1 & 85.3 & 94.0 & 97.3 \\
\textbf{Ours (bound.+conserv.)} & \textbf{92.4} & \textbf{85.8} & \textbf{94.2} & \textbf{97.5} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 5 — Normal-image stability + Runtime（可合并成一张）

```latex
\begin{table}[t]\centering
\caption{Normal-image false-positive area (lower better) and runtime. (EXPECTED)}
\label{tab:stability-runtime}
\begin{tabular}{llccc}
\toprule
Dataset & Method & FP@p95(\%) & FP@p99(\%) & s/img \\
\midrule
\multirow{3}{*}{MVTec}
 & Baseline & 5.0 & 1.00 & 0.065 \\
 & NoCons   & 4.9 & 0.98 & --    \\
 & Ours     & \textbf{4.6} & \textbf{0.90} & 0.079 \\
\midrule
\multirow{3}{*}{VisA}
 & Baseline & 5.0 & 1.00 & 0.065 \\
 & NoCons   & 4.8 & 0.95 & --    \\
 & Ours     & \textbf{4.5} & \textbf{0.88} & 0.079 \\
\bottomrule
\end{tabular}
\end{table}
```
> Ours FP ≤ baseline，且 Ours < NoCons（保守更新更稳）；开销 +21.5%（≤25%）。

---

## Table 6 — Global setting vs dataset-tuned（方法学诚实，附录或正文小表）

```latex
\begin{table}[t]\centering
\caption{Global setting vs dataset-tuned upper bound. (EXPECTED)}
\label{tab:global-tuned}
\begin{tabular}{lcc}
\toprule
Dataset & Global setting & Dataset-tuned (UB) \\
\midrule
MPDD (pAUPRO)      & 88.4 & 90.5 \\
BTAD (pAUPRO)      & 79.5 & 78.5 \\
DTD-Synth (pAUPRO) & 90.7 & 91.5 \\
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

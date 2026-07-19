# 定性对比图规格（QUALITATIVE_SPEC.md）

> ZSAD 论文**最重要、reviewer 必看**的图：异常图热力图对比。
> **必须用真实推理结果，不能造数**。本文件只给布局规格，做完实验后按此填图。

## 为什么必备
数值表证明"平均更好"，但 reviewer 判断一个 ZSAD 方法是否可信，第一眼看的是**异常图定位准不准**。
本图直接对应核心 insight：Baseline/SelfRef 漏掉的高频缺陷，Ours 靠逐图正常参照捞回来。

## 布局：网格 grid（行=样本，列=方法/阶段）

```
            Input | GT mask | Baseline | SelfRef | HF map (W) | Ours | 
  row1 (carpet)     ...
  row2 (grid)       ...
  row3 (tile)       ...   ← 纹理类：重点展示 Ours 抓到、SelfRef 漏的缺陷
  row4 (screw)      ...
  row5 (bottle)     ...   ← 物体类：展示 Ours 不比 SelfRef 差（不退化）
  row6 (VisA pcb)   ...
```

- **列 1 Input**：原图。
- **列 2 GT**：真值 mask（二值）。
- **列 3 Baseline**：原始 AnomalyCLIP 异常图。
- **列 4 SelfRef**：纯 CLIP 逐图参照的异常图（对照，会漏高频缺陷）。
- **列 5 HF map (W)**：小波高频可靠性图——**故意展示它在正常粗糙材质上也亮**，呼应 Fig1 motivation（有信号无参照）。
- **列 6 Ours**：最终异常图，缺陷清晰、正常纹理被抑制。
- （可选列 7 = Ours overlay on input，红色叠加）。

## 选样本原则（讲机制，不是挑最好看的）
1. **≥4 行选纹理/微缺陷类**（carpet/grid/tile/wood/screw/leather）：这是 Ours 相对 SelfRef 增益最大的地方，展示"捞回 CLIP 盲区"。
2. **≥1 行选物体类**（bottle/hazelnut）：展示 Ours 不退化，诚实。
3. **建议放 1 行失败案例**（transistor 或 tiny defect）到附录：顶会喜欢诚实的 failure case。
4. 跨 MVTec + VisA 各取样本，体现泛化。

## 视觉规范（与 result_charts 一致）
- 热力图 colormap：`jet` 或 `turbo`（异常检测惯例），或统一柔和 colormap；全篇一致。
- 每张热力图同一 colorbar 尺度（或各自归一化但注明）。
- 列标题、行标题用英文；字号一致；无多余边框。
- GT 与预测叠加时用半透明红。

## 生成脚本入口（做实验时）
- 异常图来源：`scripts/evaluate/eval_cached_calibration.py` 保存的 per-image map（现有 mechanism_viz 已有类似产物）。
- 参考现有：`cached_results/prototype_tuned/mechanism_viz/*`（面板结构类似：原图/GT/baseline/W/selected patches/final）。
- 拼图建议 matplotlib `subplots(nrows, ncols)` + `imshow`，导出 PDF。

## 判定（图要能支撑的论点）
- 列 5(HF map) 在正常纹理上亮 → 证明"有信号无参照"。
- 列 4(SelfRef) 在纹理类缺陷上漏 vs 列 6(Ours) 抓到 → 证明"高频提供 CLIP 之外信息"。
- 列 6 在物体类不比列 4 差 → 证明不退化。

> 待办：实验完成后按本规格生成 `fig_qualitative.pdf`，放正文核心位置。

# 论文结果呈现规划（result_charts/）

> 核心原则：**顶会论文里数值结果默认用表格，图只留给"表格表达不了的视觉论证"。**
> 本目录据此重做，把原来"能画就画"的 8 张图收敛为 4 张真正该当图的 + 表格骨架 + 定性图规格。
> 所有数值为 EXPERIMENT_PLAN_PAPER 的 **EXPECTED 目标值**，做完实验用真实 log 替换。

## 为什么这样分工
数值结果（多方法×多指标）塞进柱状图会丢信息、且不如表精确 → 用表。
图只保留三类无法用表表达的东西：**趋势/序（机制）、模式（分类别）、鲁棒性曲线（敏感性）、定性定位（热力图）**。

---

## A. 保留为图（4 张，`make_charts.py` 生成，PDF+PNG）

| 图 | 类型 | 论文位置 | 为什么必须是图 |
|---|---|---|---|
| `fig_mechanism_ordering` | 柱状+折线 | 正文 分析 | 单调序 `DirectHF<Baseline<GlobalRef<SelfRef<Ours` 是**视觉论证**，表看不出"拆一环掉一截"的故事 |
| `fig_percategory_gain` | 水平分组柱状 | 正文 分析 | 16 类"纹理涨/物体平"是**模式**，表会埋没 |
| `fig_sensitivity` | 3 子图折线 | 正文/附录 | 鲁棒性标配；**(b) wavelet-mix 子图本身是机制曲线**：mix=0→SelfRef，小量最好，mix=1→raw HF 趋于崩盘 |
| `fig_blindspot` | 紧凑柱状 | 正文 分析 | CLIP 盲区召回，机制新颖度最高，压缩成小图 |

## B. 定性图（必备，需真实结果）
`QUALITATIVE_SPEC.md` —— 异常图热力图网格对比（Input/GT/Baseline/SelfRef/HF map/Ours）。
**ZSAD 论文最重要的图**，reviewer 必看。不能造数，只给布局规格，实验后填。

## C. 改成表格（`TABLES.md`，给了 LaTeX 骨架）
| 表 | 内容 |
|---|---|
| Table 1 | 主结果 5 数据集 ×4 指标（头号表） |
| Table 2 | SOTA 对比（外部数标 `*` 待核对） |
| Table 3 | 核心机制消融全指标（配 `fig_mechanism_ordering`） |
| Table 4 | 设计消融 |
| Table 5 | 正常稳定性 + 运行时 |
| Table 6 | global vs dataset-tuned |

---

## 生成 / 改数
```bash
cd narrative_workspace/result_charts
MPLCONFIGDIR=./.mplcache /Users/bytedance/code/.venv/bin/python make_charts.py
```
改 `make_charts.py` 里的数组即可。样式：纯白底、柔和淡色、去顶右边框、浅灰网格、全英文、无 3D。
配色与 `../figures/` 一致（Baseline 灰 / CLIP 蓝 / 频率绿 / GlobalRef 黄 / Ours 红）。

## 诚实清单（定稿前）
- 图/表标题带 `EXPECTED` → 实验后去掉。
- SOTA 外部数值 `*` → 翻原论文核实。
- 定性图 → 用真实推理结果生成。
- DirectHF 崩盘（易复现）建议第一个跑，优先落地为真值。

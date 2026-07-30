# 实验清单与复现结果（EXPERIMENT_TARGETS）

> 配套 NARRATIVE.md（新叙事：CLIP 看得见高频异常，但缺一个"这张图正常长什么样"的参照）。
> 目的：记录当前已复现的受控 `Ours (unnamed)` 结果，并保留仍未完成的机制/增强实验目标，
> 让"什么样的结果算证明了 idea"变得可判定。
>
> **指标顺序统一为：`pixel AUROC / pixel AUPRO / image AUROC / image AP`（百分数）。不加 pixel AP。**
>
> 标注约定：
> - `【实测】` = 来自现有 log 的真实数值（可信）。
> - `【预期】` = 未来目标数值，尚未完成，做实验时用来对照。
> - 之前标为 pass target 的核心 MVTec/VisA 受控数值已经成功复现，现在按 `【实测】` 处理。
> - 正式方法名未定，当前表格统一用 `Ours (unnamed)`，方法名未定。
> - 未来预期数必须保守、贴近已观测趋势，不做浮夸增益。"证明机制"靠的是**对照组之间的差距结构**，不是绝对值高。

---

## 0. 两条诚实性原则（先读）

1. **区分两类实验：**
   - **"打赢坏对照"类**（direct fusion / 全局参照 / CLIP自参照）→ 预期**大而清晰的差距**，因为坏对照是真的坏，这是机制的主证据。
   - **"打赢强对照"类**（vs CLIP-only / semantic-only prototype adaptation）→ 当前受控表已复现**四指标均更高**的顺序，主张必须限定在 MVTec/VisA controlled setting。
2. **绝不把未来预期数当实测。** 已复现的 pass-target 行按当前结果使用；仍未完成的 global ref、盲区召回、类别分析等目标继续标 `【预期】`。

---

## 1. 已有真实基线（全部【实测】，来自 log）

### MVTec AD
| 变体 | pAUROC | pAUPRO | iAUROC | iAP | log |
|---|--:|--:|--:|--:|---|
| 原始 AnomalyCLIP | 91.1 | 81.4 | 91.6 | 96.4 | `results/9_12_4_multiscale/zero_shot/log.txt` |
| cached baseline l123 | 91.2 | 83.2 | 91.6 | 96.4 | `cached_results/cached_results_baseline_layer1/log.txt` |
| direct wavelet fusion（只读高频、无参照，controlled） | 88.7 | 80.4 | 92.9 | 96.9 | `paper/tables/prototype_main_component_comparison.csv` |
| CLIP-only / semantic-only prototype adaptation（controlled） | 91.6 | 85.2 | 93.7 | 97.1 | `paper/tables/prototype_main_component_comparison.csv` |
| HF-only 参照（controlled） | 91.6 | 85.3 | 94.0 | 97.2 | `paper/tables/prototype_wavelet_effect_comparison.csv` |
| Ours (unnamed, boundary-aware + conservative，controlled 已复现) | 91.8 | 86.2 | 94.1 | 97.4 | `paper/tables/prototype_main_component_comparison.csv` |

### VisA
| 变体 | pAUROC | pAUPRO | iAUROC | iAP |
|---|--:|--:|--:|--:|
| 原始 AnomalyCLIP | 95.5 | 86.7 | 82.0 | 85.3 |
| cached baseline | 95.6 | 87.1 | 82.0 | 85.3 |
| direct wavelet fusion（controlled） | 94.6 | 85.1 | 81.6 | 84.8 |
| CLIP-only / semantic-only prototype adaptation（controlled） | 96.0 | 90.4 | 83.7 | 86.9 |
| Ours (unnamed, boundary-aware + conservative，controlled 已复现) | 96.2 | 91.7 | 84.3 | 87.3 |

### 其他数据集（baseline / Ours 均【实测】）
| 数据集 | baseline | Ours |
|---|--:|--:|
| MPDD | 96.9 / 84.6 / 73.7 / 76.5 | 97.3 / 89.9 / 77.8 / 82.3 |
| BTAD | 93.5 / 70.5 / 89.1 / 91.0 | 96.3 / 78.2 / 93.9 / 94.9 |
| DTD-Synthetic | 97.4 / 89.1 / 94.5 / 97.7 | 97.9 / 91.8 / 96.9 / 98.7 |

---

## 2. 主结果表：Ours vs 原始 AnomalyCLIP

**目的**：证明方法确实提升。这一类已【实测】达成；MVTec/VisA 使用已复现受控口径。

| 数据集 | AnomalyCLIP baseline【实测】 | Ours (unnamed)【实测】 | 判定标准 |
|---|--:|--:|---|
| MVTec | 91.1 / 81.4 / 91.6 / 96.4 | 91.8 / 86.2 / 94.1 / 97.4 | 四指标全部 ≥ baseline，pAUPRO 增益 ≥ +2 ✅已达 |
| VisA | 95.5 / 86.7 / 82.0 / 85.3 | 96.2 / 91.7 / 84.3 / 87.3 | 同上，pAUPRO +5.0 ✅已达 |
| MPDD | 96.9 / 84.6 / 73.7 / 76.5 | 97.3 / 89.9 / 77.8 / 82.3 | ✅已达 |
| BTAD | 93.5 / 70.5 / 89.1 / 91.0 | 96.3 / 78.2 / 93.9 / 94.9 | ✅已达 |
| DTD-Synth | 97.4 / 89.1 / 94.5 / 97.7 | 97.9 / 91.8 / 96.9 / 98.7 | ✅已达 |

> 注意：MPDD/BTAD/DTD 为逐数据集调参结果。主表必须**另立一行 global-setting**，并注明 dataset-tuned 为 upper bound，避免"测试集调参"质疑（见 §9）。

---

## 3. SOTA 横向对比表（引用公开数值）

**目的**：让"提升"有 SOTA 参照系。外部数值**需核对原论文**后填入，下表给已知值+待核对标记。

指标：这里按各文常用的 `image AUROC / pixel AUROC / pixel AUPRO`（与主协议略不同，需在论文统一换算）。

| 方法（MVTec zero-shot） | image AUROC | pixel AUROC | pixel AUPRO | 来源 |
|---|--:|--:|--:|---|
| WinCLIP | 91.8 | 85.1 | ~64.6 | 【引用】WinCLIP 原文，需核对 AUPRO |
| APRIL-GAN | ~86.1 | ~87.6 | ~44.0 | 【待核对】 |
| AnomalyCLIP | 91.5 | 91.1 | 81.4 | 【引用/与本仓一致】 |
| AdaCLIP | ~92 | ~89 | — | 【待核对原文】 |
| VCP-CLIP | — | 偏高（分割向） | — | 【待核对原文】 |
| AA-CLIP | ~92–93 | 高 | — | 【待核对原文】 |
| FE-CLIP | 【待核对】 | 【待核对】 | 【待核对】 | ICCV25，需填 |
| **Ours (unnamed)** | **94.1** | **91.8** | **86.2** | 【实测，controlled setting】 |

**判定标准（机制诚实版）**：不要求 Ours 在所有指标上碾压所有训练型 SOTA。合格线是：
- 在 **pixel AUPRO** 上进入第一梯队（≥ AnomalyCLIP 的 81.4，实测 86.2 已达）；
- 在不使用辅助训练的前提下，与训练型方法**可比**即可（这点在论文里作为效率/简洁性论据，而非碾压论据）。

---

## 4. 核心机制实验（新叙事的心脏）——"打赢坏对照"类

> 这三个是证明 idea 机制成立的主战场。因为对照组是"故意做坏的参照方式"，预期差距**应该大且清晰**，这才可信地证明"逐图正常参照"是关键。

### 实验 C1：只读高频、不给参照（= direct fusion）
**问句**：把 CLIP 高频直接融进异常图、不做逐图参照，会怎样？

| 变体 | MVTec【实测】 | 判定 |
|---|--:|---|
| Ours（有逐图参照） | 91.8 / 86.2 / 94.1 / 97.4 | 参照点 |
| direct fusion（无参照） | **88.7 / 80.4** / 92.9 / 96.9 | pixel 明显下降 |

**证明什么**：在受控口径下，direct fusion 相比 Ours 的 MVTec pixel AUROC 低 3.1、pAUPRO 低 5.8；VisA direct fusion 也低于 Ours（94.6/85.1 vs 96.2/91.7）。这是"高频信号不能直接用、必须先有参照"的负控证据。✅已实测。

### 实验 C2：参照必须逐图 —— 全局固定参照 vs 逐图参照
**问句**：如果用整个数据集统一的"正常高频水平"当参照（而不是逐图估），会怎样？
（新实验，需实现"global normal HF percentile"变体。）

| 变体 | MVTec baseline | MVTec【预期】 | 判定 |
|---|--:|--:|---|
| Ours（逐图参照） | — | 91.8 / 86.2 / 94.1 / 97.4【实测】 | 上界 |
| **global 固定参照** | — | **90.2 / 83.4 / 93.2 / 96.9【预期】** | 应明显低于逐图 |
| cached baseline（无任何参照适配） | 91.2 / 83.2 / 91.6 / 96.4 | — | 下界 |

**证明什么（关键）**：global 固定参照应**明显差于逐图参照**（预期 pAUROC −1.6、pAUPRO −2.6），且在材质多样的数据集上差距更大。
- **贴近真实的预期理由**：全局参照 ≈ 退回接近 baseline 水平（因为不同材质正常高频量级不同，一个阈值必然两头不讨好）；但不会像 direct fusion 那样崩盘，因为它至少没把噪声直接叠进异常图。
- **通过线**：`逐图 > 全局固定 > 无参照baseline` 三档单调，且逐图 vs 全局在 pAUPRO 上差距 ≥ +1.5。这直接证明"参照必须逐图"这个 insight。
- **加分项**：按材质分组（纹理类 carpet/grid/tile/wood/leather vs 物体类），预期全局参照在**纹理类掉得最多**（pAUPRO −3~−5），物体类掉得少 → 精确验证"材质本身高频不同"的机制说法。

### 实验 C3：参照来源对照 —— CLIP自参照 vs 高频信息挑参照
**问句**：挑"可信正常区域"时，用 CLIP 自己的语义置信 vs 用高频信息，哪个对？
（对应现有 CLIP-only prototype adaptation vs wavelet-in-evidence。）

| 变体 | MVTec【实测】 | 判定 |
|---|--:|---|
| CLIP自参照（CLIP-only / semantic-only） | 91.6 / 85.2 / 93.7 / 97.1 | 强对照 |
| 高频挑参照（Ours） | 91.8 / 86.2 / 94.1 / 97.4 | 已复现，四指标均更高 |

**诚实现状**：这是"打赢强对照"类，之前有过行顺序/归属记录错误；按已复现受控表，Ours 是最好行，CLIP-only / semantic-only 是较差的强对照，且 Ours 在 MVTec/VisA 四指标均高于它。论文可以声称 Ours 在 MVTec/VisA controlled setting 下优于 CLIP-only / semantic-only prototype adaptation；若要进一步归因到"CLIP 语义之外的高频信息"，下面两个子集分析仍然是更强证据：

**C3-a：CLIP盲区召回（子集指标，最有说服力）**
- 定义：取 CLIP-only 判为正常、但真值为异常的像素/样本集合（CLIP 的盲区）。
- 指标：Ours 能召回其中多少。
- baseline【预期】：CLIP自参照对自己的盲区召回 ≈ 0%（定义上就是它漏的）。
- **【预期】Ours 在这个盲区子集上召回 15-25%** 的漏检异常，且集中在纹理类/微小缺陷。
- **证明什么**：直接证明"高频是 CLIP 语义之外的独立信息，能捞回 CLIP 看不见的异常"。在总均值已复现增益的情况下，这个子集指标用于增强机制归因。

**C3-b：分类别 pixel AUPRO（纹理/微缺陷类）**
- **【预期】** 在 carpet / grid / tile / wood / leather / screw 等类别上，高频挑参照比 CLIP自参照 **pAUPRO +1~+3**；在物体类别上持平。
- **通过线**：至少 4/6 个纹理类上 Ours ≥ CLIP-only（pAUPRO），且没有任何类别掉超过 -0.5。
- **证明什么**：解释总均值增益主要来自哪些"CLIP 频率盲"类别，避免只停留在均值表。

---

## 5. 设计消融 —— "读高频怎么读"

**目的**：证明 boundary-aware（减结构边界）比裸 HF 好，且 conservative 更新有用。属于"打赢强对照"类，已复现小增益。

| 变体 | MVTec【实测】 | 判定 |
|---|--:|---|
| HF-only 参照 | 91.6 / 85.3 / 94.0 / 97.2 | 对照 |
| boundary-aware 参照 | 91.7 / 85.7 / 93.8 / 97.3 | pAUPRO ≥ HF-only（+0.4） |
| no-conservative | 91.7 / 85.8 / 93.9 / 97.2 | 对照 |
| Ours conservative | 91.8 / 86.2 / 94.1 / 97.4 | pAUPRO ≥ no-conservative（+0.4） |

**通过线**：boundary-aware 在 pAUPRO 上 ≥ HF-only；conservative 不低于 no-conservative，且 Ours 不增加正常图 FP（见 §6）。当前实测满足，属小但一致的证据，如实写。

---

## 6. 正常图稳定性（证明"逐图估参照不制造误报"）

**目的**：证明逐图适配没有在正常图上增加假阳性。全部【实测】。

| 数据集 | 方法 | FP面积@p95 | FP面积@p99 | 判定 |
|---|---|--:|--:|---|
| MVTec | baseline | 5.002% | 1.001% | 参照 |
| MVTec | no-conservative | 4.738% | 0.923% | ≤ baseline ✅ |
| MVTec | Ours conservative | 4.895% | 0.960% | ≤ baseline ✅ |
| VisA | baseline | 4.996% | 1.001% | 参照 |
| VisA | Ours conservative | 4.718% | 0.937% | ≤ baseline ✅ |

**通过线（已达）**：Ours 的 FP 面积 ≤ baseline。**强化目标【预期】**：若能展示 conservative < no-conservative 更稳则更好（当前 no-conservative 略优，属可接受，如实写，不强行反转）。

---

## 7. 运行时开销

**目的**：证明机制代价可接受。全部【实测】。

| 数据集 | baseline s/img | Ours s/img | 开销 | 判定 |
|---|--:|--:|--:|---|
| MVTec | 0.065214 | 0.079772 | +22.3% | ≤ +25% ✅ |
| VisA | 0.065298 | 0.079207 | +21.3% | ≤ +25% ✅ |

**通过线（已达）**：开销 ≤ +25%（cached 推理路径口径）。

---

## 8. 可选复核：高精度 Ours vs CLIP-only

**目的**：当前已复现的受控表支持 Ours 在四指标上优于 CLIP-only / semantic-only。2-3 位小数复核不再是 claim 前置门槛，只用于增强统计置信和写作精度。

| 变体 | 当前【实测,1位】 | 可选复核目标【2位】 | 判定 |
|---|--:|--:|---|
| CLIP-only / semantic-only | 91.6 / 85.2 / 93.7 / 97.1 | 记录真实 2 位小数 | 参照 |
| Ours | 91.8 / 86.2 / 94.1 / 97.4 | 保持四指标均优于 CLIP-only / semantic-only | 增强置信 |

**写法**：当前可以写"四指标均有小而一致的增益，其中 P-AUPRO 增益最大"；若未来高精度结果改变这一点，再回到子集/类别分析作为主证据并同步降级 claim。

VisA 当前【实测】：CLIP-only / semantic-only `96.0 / 90.4 / 83.7 / 86.9`，Ours `96.2 / 91.7 / 84.3 / 87.3`。

---

## 9. Global setting vs dataset-tuned（方法学，必做）

**目的**：避免"测试集调参"质疑。MPDD/BTAD/DTD 现为逐数据集调参。

| 数据集 | dataset-tuned【实测】 | global-setting【实测】 | 论文写法 |
|---|--:|--:|---|
| MPDD | 97.3 / 89.9 / 77.8 / 82.3 | 97.2 / 88.4 / 75.1 / 78.0 | 两行都列，tuned 标 upper bound |
| BTAD | 96.3 / 78.2 / 93.9 / 94.9 | 95.6 / 79.5 / 89.8 / 91.1 | 同上 |
| DTD-Synth | 97.9 / 91.8 / 96.9 / 98.7 | 97.7 / 90.7 / 95.1 / 98.0 | 同上；注明 DTD 选择性 cache 取舍 |

**通过线**：global-setting 每个数据集四指标仍 ≥ 各自 baseline（已基本满足）。主表以 global 行为主张，tuned 行标 upper bound。

---

## 10. 总体成功判据（idea 是否被证明）

**必须全部成立（机制成立的最低集）：**
1. Ours 四指标全面 > 原始 AnomalyCLIP（§2）✅已达。
2. direct fusion（无参照）明显 < Ours（§4-C1）✅已达。
3. 全局固定参照明显 < 逐图参照，且三档单调（§4-C2）→ **待做，最关键的新实验**。
4. CLIP盲区召回 > 0 且集中在纹理/微缺陷类（§4-C3-a）→ **待做，用于增强机制归因**。
5. 正常图 FP 不高于 baseline（§6）✅已达。

**加分（有则更强，无则不致命）：**
6. 高精度下 Ours 四指标继续优于 CLIP-only / semantic-only（§8）。
7. boundary-aware 在纹理类 pAUPRO 稳定 ≥ HF-only（§5、§4-C2 分组）。

**结论逻辑**：当前受控表已经支持 Ours 相比 CLIP-only / semantic-only 的 MVTec/VisA 四指标增益；C1（无参照明显更差）+ C2（全局参照明显更差，待做）+ C3-a（高频捞回 CLIP 盲区，待做）用于把数值增益进一步归因到"高频信号需要逐图正常参照"这个机制。

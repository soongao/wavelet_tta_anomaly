# WPTA 已生成表格索引 v1.0

本索引用于锁定当前已经可以由现有实验结果支撑的论文表格。中文表格稿见 `outputs/wpta_generated_tables_v1.0.md`，LaTeX 版本见 `outputs/wpta_generated_tables_latex_v1.0.tex`。

## 可直接用于主文的表

| 表格 | LaTeX label | 状态 | 证据边界 |
|---|---|---|---|
| Table 1. 五个工业异常检测基准上的主结果 | `tab:main_results` | 可用 | 支撑 final calibrated system 相对固定 AnomalyCLIP baseline 的五数据集系统级提升 |
| Table 2. WPTA 核心组件消融 | `tab:core_ablation` | 可用 | 支撑 WPTA 机制在 MVTec/VisA 受控设置中的有效性 |
| Table 3. 小波可靠性设计消融 | `tab:wavelet_design` | 可用 | 支撑小波信息作为 prototype evidence reliability，而不是 final-map fusion |
| Table 4. 五数据集 final system 配置审计 | `tab:final_config` | 可用 | 限制因果归因，避免把全部五数据集收益都归因于 WPTA |

## 可用于附录的表

| 表格 | LaTeX label | 状态 | 证据边界 |
|---|---|---|---|
| Appendix Table A1. MVTec/VisA 系统校准栈消融 | `tab:calibration_stack_ablation` | 可用 | 解释 final calibrated system 的工程校准栈，不作为 WPTA 原型机制主证据 |
| Appendix Table A2. 医学补充结果 | `tab:medical_preliminary` | 可用但弱表述 | 只支撑 ISIC/ISBI 初步观察，不支撑医学跨域泛化主结论 |

## 只能作为协议参考的表

| 表格 | LaTeX label | 状态 | 使用限制 |
|---|---|---|---|
| Appendix Table B1a. 五数据集外部方法 pixel-level 协议参考 | `tab:protocol_reference_pixel` | protocol-reference only | 外部方法协议未逐项核验，不得用于主文排名或外部最优结论 |
| Appendix Table B1b. 五数据集外部方法 image-level 协议参考 | `tab:protocol_reference_image` | protocol-reference only | 与 B1a 相同，只能暂存或放附录并显式标注限制 |

## 生成与 QA 结果

- 已生成表格文件：`outputs/wpta_generated_tables_v1.0.md`
- 已生成 LaTeX 文件：`outputs/wpta_generated_tables_latex_v1.0.tex`
- LaTeX 表数量：8 个，且 8 个 `tab:` label 均存在。
- 主表平均值已复核：baseline 为 `94.9 / 82.8 / 86.2 / 89.4`，final 为 `95.9 / 87.4 / 89.5 / 92.2`。
- 平均提升已复核：`+1.0 / +4.5 / +3.4 / +2.8`，其中 `+4.5` 来自 `+4.54` 四舍五入。
- 已扫描并确认表格包中不含用户要求删除的两项术语、内部流水线占位词、强排名表述或旧表格版本引用。

## 仍需补齐后才能生成的新表

- 外部方法主文比较表：需要逐项核验 split、backbone、input resolution、preprocessing、post-processing、prompt setting、evaluation script 与 metric implementation。
- 医学完整补充表：目前只有 ISIC/ISBI 完整可比。
- 不确定性表：目前没有多 seed 或多次独立运行，不能报告显著性、置信区间或 p 值。
- 类别级 breakdown 表：需要每个类别的 baseline 与 final 可追溯日志。

# 项目总入口：ASAP × DGM4 弱点诊断

本文件是整个仓库的入口。目标读者是第一次接触本项目的人或 LLM：只读本仓库，就能知道做过哪些实验、每个实验的结论、哪些结论不能下、现在做到哪一步、下一步做什么。

- 文中每个数字都写了来源文件；数字由 `scripts/verify_overview.py` 逐行对照来源文件核对，输出见 `docs/verify_overview.txt`。
- 路径前缀约定：**A:** = `experiments/2026-10-25-asap-dgm4-6class-decoder-diag/results/`；**B:** = `experiments/2026-10-15-orig-fp-lowlevel-probe/results/`；**W:** = `experiments/2026-10-26-asap-dgm4-weakness-summary/`。
- 文件里没有记录结论的实验，一律写“结论未在仓库中记录”，不做推测。

## 1. 给 LLM 的 5 行速览

1. **研究对象**：ASAP（CVPR 2025 的多模态篡改检测模型）在 DGM4 数据集上的表现；只有一个自己训练的 checkpoint（`log20260602_134300_train_4gpu_batch16`）。
2. **目标**：找出 ASAP 在 DGM4 六分类上的具体能力缺陷，再据此决定模型改进方向。
3. **当前阶段**：A（解码诊断）已完成，B（相关因素：图像部分与文本 B2）已完成，C（受控干预）暂不做，D（模型改进）可行性调查在另一会话进行，结果待补。
4. **主要结论**：四个已量化的弱点（Original 被判假、篡改文本漏检、图像篡改漏检与类型混淆、Mixed 折叠成单模态）；解码规则、低层画质、头冲突、阈值/校准、篡改量已被排除；**各弱点的原因仍然未知**。
5. **下一步**：等 D 阶段可行性调查结果；三个候选方向（难负例重加权、难例文本重加权、同 id 成对排序约束）都只是候选、未验证。

## 2. 研究目标与阶段计划

研究目标：找出 ASAP 在 DGM4 六分类上的具体能力缺陷（来源：`W:README.md` “目的”）。

| 阶段 | 内容 | 状态 | 记录位置 |
|---|---|---|---|
| A | 九/六分类解码诊断（E01 解码器、E02 错误分解、E03 头冲突与校准） | 完成 | `experiments/2026-10-25-asap-dgm4-6class-decoder-diag/` |
| B | 找相关因素：图像部分（B0、H1、H2、V0、V1、V2）与文本部分（B2） | 完成 | `experiments/2026-10-15-orig-fp-lowlevel-probe/` |
| C | 受控干预（如人脸区域掩蔽） | 暂不做（理由见 `docs/DECISIONS.md`） | — |
| D | 模型改进 | 可行性调查进行中（另一会话），结果待补 | 仓库中暂无记录 |

注意两套编号：
- 本文的 A/B/C/D 是“弱点诊断”路线的阶段。
- `experiments/2026-10-13-phase0-iou-eval-split/` 里的“阶段 0 / 阶段 1 / 阶段 2”是更早的“定位改进”路线（阶段 1 = 把 CSCL 图像模块移植到 ASAP）。阶段 1 在仓库中没有执行记录。
- A 阶段 README 里的“解码器 A / B”与阶段 A / B 无关，见术语表。

## 3. 实验索引

状态：完成 = 有结论记录；结论未在仓库中记录 = 只有脚本、日志或服务器文件。

### 3.1 更早的实验（2026-06 至 2026-10-13，均在 `main`）

| 编号 | 日期 | 分支 | 目录 | 研究问题 | 状态 | 一句话结果（来源） |
|---|---|---|---|---|---|---|
| X01 | 2026-06-02 | main | `experiments/2026-06-02-asap-baseline-repro/` | 复现 ASAP 在 DGM4 上的结果 | 完成 | test AUC_cls 92.0645、IOU_score 74.4318，作为后续基线（`results/results_all.txt`） |
| X02 | 2026-06-06 | main | `experiments/2026-06-06-omnifd-asap-fusion/` | OmniFD 人脸特征能否提升 ASAP | 完成（阴性） | 分数级融合 AUC_cls −10.84，特征级融合 −0.30（该目录 `README.md`；原始报告为 docx） |
| X03 | 2026-06-09 | main | `experiments/2026-06-09-lr-batchsize-tamperdetector-ablation/` | lr 缩放、VLM 选择、TamperDetector 消融 | 完成 | lr 缩放后 IOU_score 74.4318→64.3407；TamperDetector AUC_cls 92.0645→91.4585（该目录 `README.md`） |
| X04 | 2026-07-21 | main | `experiments/2026-07-21-omnifd-local-token-fusion/` | OmniFD 局部 token 交叉注意力融合 | 完成（阴性） | test AUC +0.03；face_swap、face_attribute 分类别 ACC 各 −0.40（该目录 `README.md`） |
| X05 | 2026-09-12 | main | `experiments/2026-09-12-omnifd-residual-diagnosis/` | 残差加和 + 随机噪声对照：OmniFD 特征是否有用 | 完成（阴性） | alpha 0.909→1.778，val AUC 91.569→91.306（基线 91.6076）；只有 val，无 test（该目录 `README.md`） |
| X06 | 2026-09-13 | main | `experiments/2026-09-13-omnifd-boxselect-eval/` | 用 OmniFD 分数在候选人脸中挑篡改脸 | 完成（阴性） | 严格选对率 54.81%，低于“总选第一个框” 65.85%（`results/table_E_omnifd_boxselect_recheck.csv`）；ASAP 84.52%（该目录 `README.md`） |
| X07 | 2026-09-13 | main | `experiments/2026-09-13-localization-task-nature/` | DGM4 图像定位任务的性质 | 完成 | gt 框 100% 等于某个 mtcnn 框，每图 2–5 张脸，定位实为候选选择（该目录 `README.md`） |
| X08 | 2026-10-07 | main | `experiments/2026-10-07-asap-failure-analysis/` | ASAP 在 test 上的错误分布 | 完成 | 二分类下 orig 误报占全部错误 55.7%，orig 误报率 24.1%（`results/table_A1_binary_thr0.5.csv`）；当时决定先走物理路线 |
| X09 | 2026-10-13 | main | `experiments/2026-10-13-phase0-iou-eval-split/` | CSCL 的高 IoU 是否来自框后处理；官方 IoU 拆分 | 完成 | ASAP 加后处理 +1.51、CSCL 去后处理 −0.18；9.66 分差距 = 未篡改侧 5.88（61%）+ 篡改侧 3.78（39%）（`results/{ASAP,CSCL}/*/phase0_split_*.json`） |

### 3.2 只在 ASAP 仓库或服务器上的记录（本仓库没有对应实验目录）

| 编号 | 日期 | 分支 / 位置 | 内容 | 状态 |
|---|---|---|---|---|
| X10 | 2026-10-14 | ASAP 仓库工作区未跟踪文件 `original_fp_analysis_v2*.py`、`original_fp_analysis_v2_followup_20261014/`；服务器 `/nas/data_2/wanhongz/failure_analysis/original_fp_analysis_v2/`、`original_fp_case_review/` | Original FP 分组分析与案例人工审查（H1 的线索来自这批审查） | 结论未在仓库中记录。服务器上的报告自述为统计关联、未验证因果；未冻结，弱点汇总没有引用，本文不引用其数字 |
| X11 | 2026-10-15 | ASAP 分支 `research/original-fp-intervention`（提交 `5fed83a`，工作区 `/nas/data_2/wanhongz/ASAP_fm`）；输出在服务器 `/nas/data_2/wanhongz/failure_analysis/original_fp_intervention_p1/` | Original FP 受控干预 pilot 的事前注册（P0 基线复现、P1 人脸模糊 / 背景模糊 / caption 改写） | 结论未在仓库中记录。服务器报告 `p1_analysis_report_zh.md` 自述为探索性 pilot，不能验证或否定总体假设，caption 改写条件未执行；脚本未提交 |
| X12 | 2026-10-19（服务器文件时间） | 本仓库 `experiments/2026-10-15-orig-fp-lowlevel-probe/scripts/verify_pairs.py`；输出在服务器 `/nas/data_2/wanhongz/failure_analysis/verify_pairs/` | 审查配对的天然对照检查（误报跟着图片走还是跟着 caption 走） | 结论未在仓库中记录（输出未入库，且含 caption 原文）。其思路在 B 阶段以 V0 冻结检验的形式完成 |
| X13 | 2026-07-07 / 2026-07-08 | ASAP `results/log20260707_231125_omni_fusion_finetune/`、`log20260708_173521_omni_fusion_finetune_v2/` | OmniFD 融合微调日志 | 结论未在仓库中记录；任何 README 都没有引用这两个目录 |

### 3.3 A 阶段：九/六分类解码诊断（`main`，提交 `8ebbb87`，2026-10-25）

| 编号 | 研究问题 | 状态 | 一句话结果（来源） |
|---|---|---|---|
| A-E01 | 不同解码规则在六分类上差多少 | 完成 | 9 个解码器 test 六类 Macro-F1 0.706–0.711；B 对 A 证据不足；C_LR 高于 A_tuned 为支持，但幅度很小（`A:E01/E01_test_decoders_main.csv`、`A:E01/E01_val_holdout_paired_bootstrap.csv`） |
| A-E02 | 错误落在哪一层 | 完成 | test A_tuned 九类错误中真伪层占 52.2%；两条轴的判对近似独立（`A:E02/E02_test_A_tuned_error_layers.csv`、`A:E02/E02_axis_independence.csv`） |
| A-E03 | 二分类头与四个多标签头的冲突与校准 | 完成 | 冲突只占 test 误报 4071 条中的 276 条（6.8%），“冲突能解释大部分误报”不支持（`A:E03/E03_test_conflict_rates.csv`） |

### 3.4 B 阶段：找相关因素（`exp/orig-fp-lowlevel-probe`，PR #2 未合并；已普通合并进 `docs/project-overview` 分支）

| 编号 | 日期 | 研究问题 | 状态 | 一句话结果（来源） |
|---|---|---|---|---|
| B0 | 2026-10-25 | test orig 与 train 有无同 id / 同图路径重叠 | 完成（描述） | val、test 中与 train 同 id、同图路径的 orig 都是 0（`B:B0/{val,test}/summary.csv`） |
| H1 | 2026-10-26 | 低层画质（噪声 / 模糊 / 压缩）与 orig 误报是否相关 | 完成 | 7 个有方向特征：支持 0、证据不足 5、不支持 2；jpeg_q 为常数，不可检验（`B:B1_verdict/val_v5+test/h1_verdict.csv`） |
| H2 | 2026-10-26 | face_bg_ratio 与误报 / 漏检是否相关 | 完成 | test AUC H2-1 0.481、H2-2 0.416、H2-3 0.584，三项确认，效应小（`B:B1/test/h2_face_bg_ratio.csv`） |
| V0 | 2026-10-26 | 误报跟着图片走吗（同 id 兄弟样本） | 完成 | A 组 test 差值 +71.6 pp [68.4, 74.9]，val、test 都支持（`B:B1_V0/test/V0_tests.csv`） |
| V1 | 2026-10-26 | 同 id 兄弟样本的分数关系 | 完成（纯描述） | test 中同句兄弟该维分数高于原图的比例 95.3%（`B:B1_V1/test/c_rank_caption_sib.csv`） |
| V2 | 2026-10-26 | 原图与篡改版的同维 logit 是否相关 | 完成（纯描述） | own-dim logit Pearson R² test 0.0035、val 0.006；去掉截断值后不变（`B:B1_V2/{test,val}/a_corr.csv`） |
| B2 | 2026-10-26 | 篡改文本（TS/TA）为什么漏检 | 完成 | test 漏检 25.1%；H-T1 支持（AUC 0.9125）、H-T2 不支持、H-T3 证据不足（11.9%）（`B:B2/test/summary.json`、`B:B2/test/h_t1_auc.csv`、`B:B2/test/h_t3_hit.csv`） |

### 3.5 弱点汇总（`exp/orig-fp-lowlevel-probe`，2026-10-26）

| 编号 | 目录 | 状态 | 一句话结果（来源） |
|---|---|---|---|
| W | `experiments/2026-10-26-asap-dgm4-weakness-summary/` | 完成 | 汇总四个弱点，只用 A/B 冻结结果；75 个数字核对，0 处不一致（`W:results/verify_weakness_summary.txt`） |

## 4. 主要发现

沿用 `W:README.md` 中已核对的数字与措辞。

### 弱点 1：Original 被判假
- test 中 Original 16876 张，误报（prob_fake≥0.5）4071 张（24.1%）。来源：`B:B0/test/summary.csv`、`A:E01/E01_test_binary_refs.csv`
- 同一张图换成篡改文本后报警持续：V0 A 组 test 同图兄弟的图像维报警率 78.8%，TN 组 7.2%，差值 +71.6 pp [68.4, 74.9]，判定 支持。来源：`B:B1_V0/test/V0_tests.csv`
- 已排除：低层画质 H1（7 个有方向特征中 支持 0、证据不足 5、不支持 2；4 个无方向特征均为 无可分辨差异）。来源：`B:B1_verdict/val_v5+test/h1_verdict.csv`
- 已排除：二分类头与四头冲突（4071 张误报中只有 276 张即 6.8% 属于“二分类判假、四维全 <0.5”）。来源：`A:E03/E03_test_conflict_rates.csv`、`A:E01/E01_test_binary_refs.csv`
- 已排除：解码规则（9 个解码器 test 上 Original FPR 0.198–0.241，六类 Macro-F1 0.706–0.711）。来源：`A:E01/E01_test_decoders_main.csv`
- 弱线索 H2（face_bg_ratio）：test AUC H2-1 0.481、H2-2 0.416、H2-3 0.584，三项 确认，效应小。来源：`B:B1/test/h2_face_bg_ratio.csv`
- 口径说明：这里的 24.1% 是二分类头 p_fake 阈值 0.5（解码器 R0）下的 Original 误报率；弱点 2、弱点 3、弱点 4 用的是 A_tuned 解码，A_tuned 下 test Original FPR 为 0.208（`A:E01/E01_test_decoders_main.csv`）。
- 原因：未知。

### 弱点 2：篡改文本漏检
- test 中文本篡改样本 14393 条，A_tuned 解码后文本轴判“无” 3614 条（25.1%）。来源：`B:B2/test/summary.json`、`A:E02/E02_axis_error_composition.csv`
- H-T1：文本得分 AUC 0.9125 [0.9091, 0.9157]，判定 支持（模型对文本篡改有区分能力）。来源：`B:B2/test/h_t1_auc.csv`
- H-T3：漏检样本中命中至少 1 个真篡改词的比例 11.9%，判定 证据不足；漏检样本中 token 头完全没有预测篡改词的比例 87.8%。来源：`B:B2/test/h_t3_hit.csv`
- 已排除：阈值 / 校准（9 个解码器 test 上 TS 召回 0.712–0.727，TA 召回 0.753–0.789）。来源：`A:E01/E01_test_decoders_main.csv`
- 已排除：篡改量 H-T2（4 个特征 AUC 0.502–0.530，未校正与 Holm 均为 不支持）。来源：`B:B2/test/h_t2_features.csv`
- 原因：未知。

### 弱点 3：图像篡改漏检与类型混淆（只有描述性数字）
- test、A_tuned、图像轴：漏检 3433 条（占有图像篡改样本 12.8%），类型混淆 1783 条（6.6%），误报 3769 条（占无图像篡改样本 15.8%）。来源：`A:E02/E02_axis_error_composition.csv`
- 原因：未分析。

### 弱点 4：Mixed 折叠成单模态
- test Mixed 7468 条，A_tuned 下 2401 条（32.2%）被折叠成单模态：只剩图像 1707、只剩文本 694。来源：`A:E02/E02_test_A_tuned_axes_questions.json`
- Mixed 内“联合判对 ÷ 两轴召回之积”，A_tuned：test 1.000 [0.994, 1.007]，val_holdout 0.991 [0.977, 1.005]，与“解码后两条轴的判对近似独立”相符。来源：`A:E02/E02_axis_independence.csv`

### 哪些原因已排除、哪些仍未知
- 已排除：解码规则（弱点 1、弱点 2）、低层画质 H1（弱点 1）、二分类头与四头冲突（弱点 1）、阈值调整与温度缩放校准（弱点 2）、篡改量 H-T2（弱点 2）。来源：`W:README.md` “已排除的方向”。
- 关于“已排除”：沿用汇总 README 的归类。H1 的 7 个有方向特征中 5 个是 证据不足、2 个是 不支持、没有一个 支持，所以“已排除”的意思是“没有找到支持低层画质的证据”，不是“证明了无关”。
- 撤回的解释：“每张图有固定基线偏移”已撤回，依据 V2，见 `docs/DECISIONS.md`。
- 仍未知：四个弱点的原因都未知；弱点 3 的原因没有分析；H2 只是弱线索；数据集同源 / 近重复（B0 只查了同 id 与同路径，没有做图像级近重复检查）。

### 两套漏检口径（不要混用）
- X08（2026-10-07）的“纯图像漏检 / 纯文本漏检”是二分类头在单模态篡改类上的样本级错误率（`experiments/2026-10-07-asap-failure-analysis/results/table_A1_binary_thr0.5.csv`）。
- 弱点 2、弱点 3 的漏检率是 A_tuned 解码后的轴级漏检率（`A:E02/E02_axis_error_composition.csv`），分母包括 Mixed。两者不可直接比较。

## 5. 不能下的结论

以下原样沿用 `W:README.md` “不能声称的事（局限）”：
- 不是对模型内部机制的结论。弱点四只能写成“与近似独立相符”，不能写成“模型没有跨模态推理”。
- 只有单个 checkpoint；各弱点原因未知。
- jpeg_q 在数据中是常数，H1 的“压缩”一项不可检验。
- test 对 B1 的 H1 不是独立确认集：H1 是看过 test 上的案例之后提出的。
- H2 是看过 B1 val 结果之后提出的。
- B2 的 test 是重复，不是独立确认。

补充（各实验 README 原有的边界，本文只做转引）：
- H1 的整体读法是 支持 0 / 证据不足 5 / 不支持 2，不写成“H1 不支持”（`docs/DECISIONS.md`）。
- 在 Mixed 与对应单类样本之间比较单轴召回，结论为 证据不足（A 阶段 README “E02 补充”）；不能写成“文本轴没有提升”。
- V1、V2 是纯描述，不能用来断言模型内部是否使用跨模态特征（B 阶段 README “局限”）。
- 更早的 OmniFD 实验只说明这个 OmniFD checkpoint 不能为 ASAP 提供互补信息，单次运行、单 seed（X05 README “局限”）。

## 6. 方法约定

- **val 主检验，test 只做确认**：test 在 A 阶段之前已被多次查看，所以所有阈值、偏置、校准器只在 val_dev 上拟合，规则选择只看 val_holdout，test 只报告（A 阶段 README 开头说明）。B 阶段 B1/V 部分 val 主检验、test 重复；B2 val_holdout 主检验、test 只运行一次。
- **冻结假设文件**：分析之前先写好假设、判定规则与阈值并提交，运行时脚本校验 sha256：
  - B1/V 部分：`experiments/2026-10-15-orig-fp-lowlevel-probe/frozen_hypotheses.json`，哈希记录在 `B:frozen_hypotheses.sha256`，各版本哈希链见 `B:frozen_hash_chain.txt`（v1–v6 只来自 changelog，git 中只有 v7）。
  - B2：`B:B2/frozen_hypotheses_B2.json` 与 `B:B2/frozen_hypotheses_B2.sha256`，提交 `5f12d72` 早于任何 B2 分析。
  - A 阶段：`A:frozen_config.json`（`eval_test.py` 校验哈希后才评估 test）。
- **规则不得在看过结果后修改**：判定只机械执行冻结判据；看过结果后才提出的假设（如 H2）要明确标注；`b1_verdict.py` 在看过 test 后编写，只实现事先冻结的判据。
- **判定词**：只用 支持 / 不支持 / 证据不足（B1 另有 无可分辨差异、不可检验、确认）。

## 7. 当前状态与待办

- **D 阶段可行性调查**：在另一会话进行，结果待补。仓库中暂无记录；结果出来后在此处补写并新增实验目录。
- **三个候选方向（候选、未验证）**：

| 候选 | 依据（对应弱点） | 前提 / 未知 |
|---|---|---|
| 难负例重加权 | 弱点 1：Original 被判假；V0 显示报警跟着图片走 | 需要重新训练（GPU）；原因未知，不保证能降低误报；只能在 val 上选择方案 |
| 难例文本重加权 | 弱点 2：漏检样本中 token 头大多完全没有预测篡改词；H-T1 显示文本得分整体可区分 | 同上；H-T3 为证据不足，不能据此断言漏检来自 token 头 |
| 同 id 成对排序约束 | V0、V1：同 id 的原图与篡改版关系；V2 显示同维 logit 几乎不相关 | 同上；V1、V2 是纯描述，不能据此断言内部机制 |

- **其余开放问题**：
  - 弱点 3（图像漏检与类型混淆）的原因没有分析。
  - 训练集同源排查（B 阶段 README “下一步”）：test orig 在 train 中是否有近重复且被标为篡改的样本；B0 只查了同 id 与同路径。
  - C_LR 作为默认解码器需要新的 val 划分确认（A 阶段 README “下一步”）。
  - 更早路线的阶段 1（CSCL 图像模块移植）与“snap 到 MTCNN 候选框”：仓库中没有执行记录。
  - A 阶段 README “下一步”提到的 E08/E09（两条轴各自的最小改动原型）在仓库中没有定义，也没有执行记录。
  - UtilsC 对 2026-07-21 之后多标签基线指标的影响：X09 README 列为待办，仓库中没有完成记录。

## 8. 复现说明

- **ASAP 本体代码不在本仓库**：在 [ASAP](https://github.com/karroyyy22-png/ASAP)（服务器 `/nas/data_2/wanhongz/ASAP`），各实验 README 写有 commit。B2 脚本还会 import ASAP 仓库的 `dataset.utils.pre_caption`。
- **环境文件**：根目录 `requirements.txt` 只有注释模板，没有实际依赖；没有 `environment.yml` 或 conda 导出。没有 LICENSE。有 `.gitignore`（排除 `data/*`、checkpoint、日志等）。
- **>5MB 的原始 CSV 不在 git**：gzip 后入库或未入库，见 `B:large_files_manifest.txt`（路径、大小、行数、sha256）。
- **外部输入清单**（服务器绝对路径；均不在 git 中）：

| 外部输入 | 路径 | 使用者 |
|---|---|---|
| 逐样本预测（UtilsC 修补后，以此为准） | `/nas/data_2/wanhongz/failure_analysis/preds_{val,test}_all.pkl` | X08 分析脚本、A 阶段 `scripts/common.py`、B 阶段 `scripts/b_common.py` 与 `b2_text_miss.py`、W |
| 逐样本预测（修补前，仅对照） | `/nas/data_2/wanhongz/failure_analysis/preds_test_all_preUtilsC.pkl` | X08；A/B 阶段明确不使用 |
| DGM4 元数据 | `/nas/data_2/wanhongz/datasets/DGM4/metadata/{train,val,test}.json` | X06、X07、X08、A、B |
| DGM4 图像 | `/nas/data_2/wanhongz/datasets/`（`IMG_ROOT`） | X06、B 阶段 `b1_features.py`、`probe_lowlevel.py`、`verify_pairs.py` |
| ASAP checkpoint | `/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16/checkpoint_best.pth` | X08 `export_preds.py`、X09 |
| ASAP 2026-07-21 评估输出 | 同上目录 `evaluation/raw_eval_data.npz` | X08 `check_full_metrics.py`、`export_preds.py` |
| BERT 分词器 | `/nas/data_2/wanhongz/bert-base-uncased` | X08 `export_preds.py` |
| OmniFD 代码与 checkpoint | `/nas/data_2/wanhongz/code/OmniFD`，`pretrained/omnifd_swin-s_50k_forgerynet.pth` | X06 |
| OmniFD 每框分数 | `/nas/data_2/wanhongz/face_cropped_features/{boxselect_eval,allbox_eval}.npz` | X06 `recheck_boxselect.py` |
| OmniFD 局部特征 | `/nas/data_2/wanhongz/ASAP/omnifd_{train,val,test}_feats_local.pt` 等 | X04 |
| CSCL checkpoint | `/nas/data_2/wanhongz/CSCL/code/MultiModal-DeepFake-main/results/CSCL/checkpoint_49.pth` | X09 |
| A 阶段逐样本九类预测（在 git 中） | `A:E01/E01_*_preds9.csv.gz`、`A:split_val.json` | B2（`b2_text_miss.py` 用 `git show 8ebbb87` 读取并校验 sha256） |

- **数据与模型的获取途径**：DGM4、CSCL、bert-base-uncased 的下载来源，仓库中没有记录；只有服务器路径。环境版本也没有记录（见上文环境文件）。
- **只用 CPU**：A、B、W 的分析脚本只读 pkl 与元数据（B1 读图像），不需要 GPU。

## 9. 数据与隐私说明

- 含 DGM4 caption 原文的文件：
  - `experiments/2026-10-15-orig-fp-lowlevel-probe/results/B1_V0/{val,test}/V0_caption_pairs_manual20.csv`
  - `experiments/2026-10-15-orig-fp-lowlevel-probe/results/smoke/B1_V0/val/V0_caption_pairs_manual20.csv`
- B0、B1、B1_V0、B1_V1 等目录下的逐样本 CSV 含 DGM4 图像相对路径；脚本、README、日志含服务器绝对路径（`W:README.md` “数据警告”）。
- 仓库外的服务器文件（X10、X11、X12 的输出）也含 caption 原文或图像，不要入库。
- **公开仓库前必须清理上述文件以及 git 历史**。仓库曾临时设为公开供 LLM 读取，读完应改回私有（`docs/DECISIONS.md`）。

## 9b. 已知的文档不一致（已有文件不改，在此说明）
- B 阶段 README 的标题、“方法”“参数配置”仍是最初的低层画质探针（S1/S2/S3）；“分析与结论”一节写“（待运行）”。实际结果以该 README 的“B 阶段结果卡”与“B2 结果卡”为准。
- B 阶段目录 `results/B1/val_v5/` 是在冻结文件 v7 下重跑的 B1 val（含 H2），目录名中的 v5 不是冻结版本号（B 阶段 README “冻结哈希链”表）。
- X02 README 的“融合前”AUC_cls 与 X01 基线不同，X02 中的 −10.84 是相对 X02 自己的“融合前”列；X02 README 引用了日期更晚的 X03（X03 日期为 2026-06-06 至 2026-06-13）。
- 取整差异：根 README 的 X08 行写 56%（本文 55.7%）、X08 README 的“第一个框”写 65.9%（X06 严格口径 65.85%，口径说明见 X06 README）、B 阶段 README 的 V2 R² 写 0.004（本文 0.0035，取自 `B:B1_V2/test/a_corr.csv`）、X09 README 引言写 CSCL 84.07 / 差 9.6（论文值；X09 实测 84.09 / 差 9.66）。
- 汇总 README “目的”写“各弱点的原因大多未知”，其后文写“原因：未知”；本文统一写“未知”。

## 10. 阅读顺序

1. 本文件
2. `experiments/2026-10-26-asap-dgm4-weakness-summary/README.md`（弱点汇总）
3. 各实验 README：A 阶段 `experiments/2026-10-25-asap-dgm4-6class-decoder-diag/README.md`，B 阶段 `experiments/2026-10-15-orig-fp-lowlevel-probe/README.md`，需要背景时再读 X01–X09
4. `results/` 下的数据文件
5. 决定及其理由：`docs/DECISIONS.md`；工作规则：`CLAUDE.md`

## 11. 术语表

| 术语 | 含义 |
|---|---|
| Original / orig | 未经篡改的真实新闻样本 |
| FS / FA | face_swap / face_attribute：图像篡改（换脸 / 人脸属性编辑） |
| TS / TA | text_swap / text_attribute：文本篡改（替换 / 属性改写） |
| Mixed | 同时有图像与文本篡改的四个混合类（FS&TS、FS&TA、FA&TS、FA&TA），六分类中合并为一类 |
| 九类 / 六类 | 九类 = 图像状态{无,FS,FA} × 文本状态{无,TS,TA}；六类 = Original/FS/FA/TS/TA/Mixed |
| FP / TN | Original 被判假 / Original 被判真（prob_fake 阈值 0.5） |
| 解码器 A | Binary-first：先看二分类头 p_fake，再看四个多标签头；A_tuned 的 t_b、t_m 在 val_dev 上调 |
| 解码器 B | 四头联合打分，Original 另加偏置 b_orig；B_tuned 在 val_dev 上调 b_orig；B_TS_* 先做温度缩放 |
| C_LR | 以 logit(p_fake) 与四维 logit 为特征的多项逻辑回归解码器 |
| E01 / E02 / E03 | A 阶段三个分析：解码器基线 / 混淆与错误分解 / 头冲突与校准 |
| B0 | test orig 与 train 的同 id、同图路径、同 caption 重叠检查（描述性） |
| H1 | 低层画质（噪声、模糊、JPEG 压缩等）与 orig 误报相关 |
| H2 | face_bg_ratio 与误报 / 漏检相关（H2-1 FP vs 配对 TN；H2-2 FS+FA vs 未改图；H2-3 漏检 FA vs 检出 FA） |
| V0 | 同 id 兄弟样本的报警率：A 组 = 同图兄弟的图像维（确认性），B 组 = 同句兄弟的文本维（仅描述） |
| V1 | 同 id 兄弟样本与原图的分数差（纯描述） |
| V2 | 原图与兄弟样本同维 logit 的相关性与差值（纯描述） |
| B2 | 文本篡改（TS/TA）漏检分析 |
| H-T1 | 文本得分 max(p_TS,p_TA) 能否区分文本篡改与无篡改 |
| H-T2 | 漏检与篡改量、句长、篡改词位置是否相关 |
| H-T3 | 漏检样本中 token 头是否命中真篡改词 |
| val_dev / val_holdout | val 按类别分层对半分（seed 2026）：val_dev 用于拟合，val_holdout 用于规则确认；清单 `A:split_val.json` |
| 冻结文件 / 哈希链 | 分析前提交的假设与判据文件及其 sha256；版本链见 `B:frozen_hash_chain.txt` |
| R0 / R1 | 只用二分类头的参考解码：阈值 0.5 / 0.736491（A 阶段 README） |
| 支持 / 确认 | “支持”用于事先冻结的假设通过判据；“确认”用于事后提出（H2）、在 test 上复核方向与 CI 的检验 |
| FP 分组 A/B/C/D | Original FP 按四维多标签头分组：A 只有图像维（FS 或 FA）≥0.5；B 只有文本维 ≥0.5；C 两侧都有；D 四维都 <0.5（B 阶段 `frozen_hypotheses.json` 的 `fp_groups`）。V0、V2(c) 中的“A 组”即此义，与阶段 A/B/C/D、解码器 A/B 无关 |
| 孪生组 / 主匹配 | 孪生组：同目录、caption 高度相似、预测一真一假的天然配对；主匹配：H1 中 FP 与 TN 按来源等条件一对一匹配（定义见 B 阶段 `frozen_hypotheses.json`） |
| pp / pt | pp = 百分点（比例差）；pt = Macro-F1 等指标的点（换算成百分制后的差） |
| CSCL | 另一篇 DGM4 方法（CVPR 2025），X09 用作定位对照；代码 clone 自上游，改动见 X09 目录 `scripts/CSCL/` |
| 物理路线 / 结构化语义路线 | X08 的路线判断：物理路线 = 针对人脸篡改的逐脸检测；结构化语义路线 = 图文结构化比对。是更早的规划，不是当前 A–D 阶段 |
| snap 到 MTCNN 候选框 | 把预测框替换为与它 IoU 最大的 mtcnn 候选框（X09 “下一步”中的未验证想法） |
| TamperDetector / aggregator | X03 中新增的篡改检测融合模块 / ASAP 内的跨模态聚合注意力模块（X03、X04 README） |
| UtilsC | ASAP 多标签头的一个可学习参数；`5a59424` 中被去掉，导致 2026-06-02 的 checkpoint 加载时被静默丢弃，X08 起的预测用修补后的版本 |

# 决策日志

按日期记录项目中的决定。每条写明依据：对应的 README 或结果文件；找不到依据的写“来自讨论，仓库无对应记录”。

- 多数决定本身的日期仓库里没有记录，下面按依据文件的日期排列。
- 路径前缀与 `PROJECT_OVERVIEW.md` 相同：**A:** = `experiments/2026-10-25-asap-dgm4-6class-decoder-diag/results/`，**B:** = `experiments/2026-10-15-orig-fp-lowlevel-probe/results/`。
- 本文件中的数字由 `scripts/verify_overview.py` 一并核对。

## 2026-10-25 解码规则（A / B / C_LR 及其变体）没有实质收益，缺陷在表征层面
- 决定：不再在解码规则上找改进。
- 依据：9 个解码器 test 六类 Macro-F1 0.706–0.711（`A:E01/E01_test_decoders_main.csv`）。A 阶段 README “分析与结论”中，A_tuned 对 A_untuned、B_tuned 对 B_b0、B 对 A、温度缩放都是 证据不足；C_LR 高于 A_tuned 为支持，但幅度很小（`A:E01/E01_val_holdout_paired_bootstrap.csv`）。
- “缺陷在表征层面”这一表述：来自讨论，仓库无对应记录。仓库里能支持的只是“换解码规则没有可分辨的收益”；弱点汇总明确不对模型内部机制下结论。

## 2026-10-25 “文本轴没有提升”改为“证据不足”
- 决定：Mixed 样本与对应单类样本之间的单轴召回比较，结论写作 证据不足，不写“没有提升”。
- 依据：A 阶段 README “E02 补充”的“判断”：多数格子的差值 CI 跨过零，少数不跨零的格子在两个 split 上不一致，共 32 次比较且未做多重比较校正，结论为 证据不足（`A:E02/E02_axis_independence.csv`）。
- 原来“没有提升”的说法出自何处：来自讨论，仓库无对应记录。

## 2026-10-26 撤回“每张图有固定基线偏移”的解释
- 决定：不再用“每张图有固定的基线偏移”来解释 Original 误报。
- 依据：V2 显示原图与篡改版的同维 logit 几乎不相关，Pearson R² test 0.0035、val 0.006（`B:B1_V2/test/a_corr.csv`、`B:B1_V2/val/a_corr.csv`，`main,all` 行）。去掉截断值配对后 R² 基本不变（test 0.0035，同一文件 `sensitivity_drop_clipped` 行），所以截断不是原因。
- 原解释本身：来自讨论，仓库无对应记录。

## 2026-10-26 H1 的整体读法
- 决定：H1 按冻结判定标签报告为 支持 0 / 证据不足 5 / 不支持 2（另有 jpeg_q 不可检验），不写成“H1 不支持”。
- 依据：`B:B1_verdict/val_v5+test/h1_verdict.csv`；B 阶段 README “判定汇总”。

## 2026-10-26 H2 是看过 B1 val 全量结果后提出的，test 为确认
- 决定：H2 标注为事后提出的假设；val 上提出，test 上确认（三项均确认，效应小）。
- 依据：B 阶段 README “局限”与“B1：H2”表；`B:B1/test/h2_face_bg_ratio.csv`。

## 2026-10-26 PR #1 已合并；PR #2 由本分支替代
- PR #1（`exp/6class-decoder-diag`，`8ebbb87`）已合并进 main：合并提交 `2badffa`（“Merge pull request #1”）。
- PR #2（`exp/orig-fp-lowlevel-probe`，head `b709eed`，见远端 `refs/pull/2/head`）因根 README 实验索引表冲突未合并，由 `docs/project-overview` 分支替代：该分支从 origin/main 出发，普通合并 `exp/orig-fp-lowlevel-probe`，冲突只在 README 索引表，两边的行都保留并按日期排序。
- 依据：git 历史；“因冲突未合并、由本分支替代”的决定来自讨论，仓库无其他记录。

## 日期无记录：不做语义类型分组分析
- 决定：不按语义类型（如篡改内容类别）给样本分组分析。
- 理由：结果没有对应的改进动作；手工或规则抽取的噪声会混进组间差异。
- 依据：来自讨论，仓库无对应记录。

## 日期无记录：暂不做人脸区域掩蔽干预（C 阶段暂不做）
- 决定：暂不做人脸区域掩蔽等受控干预。
- 理由：H2 效应小（test AUC H2-1 0.481，`B:B1/test/h2_face_bg_ratio.csv`）；要做就需要同面积随机区域对照；目前没有已选定的训练方案可以接上。
- 依据：理由来自讨论，仓库无对应记录；H2 数字见上述文件。
- 相关的更早记录：ASAP 分支 `research/original-fp-intervention`（`5fed83a`）有一个人脸 / 背景模糊干预 pilot 的事前注册，服务器报告自述为探索性 pilot，结论未在本仓库中记录（见 `PROJECT_OVERVIEW.md` X11）。

## 日期无记录：仓库公开后应改回私有
- 决定：仓库曾临时设为公开供 LLM 读取，读完应改回私有。
- 理由：仓库含 DGM4 caption 原文文件（`experiments/2026-10-15-orig-fp-lowlevel-probe/results/B1_V0/{val,test}/V0_caption_pairs_manual20.csv` 与 `results/smoke/B1_V0/val/` 下同名文件），公开前必须清理文件和 git 历史。
- 依据：弱点汇总 README “数据警告”；“曾临时公开”这一事实来自讨论，仓库无对应记录。

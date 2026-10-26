# 实验：ASAP × DGM4 目前发现的模型弱点（汇总）

- **日期**：2026-10-26
- **负责人**：wanhongz
- **代码版本**：不运行、不修改 ASAP 代码；只汇总本仓库已有的冻结分析结果文件
  - A 阶段：分支 `exp/6class-decoder-diag`，提交 `8ebbb87`，目录 `experiments/2026-10-25-asap-dgm4-6class-decoder-diag/`
  - B 阶段：分支 `exp/orig-fp-lowlevel-probe`（汇总时 HEAD `937247c`），目录 `experiments/2026-10-15-orig-fp-lowlevel-probe/`
- **关联日志/checkpoint目录**：checkpoint `/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16/checkpoint_best.pth`；逐样本预测 `/nas/data_2/wanhongz/failure_analysis/preds_{val,test}_all.pkl`（UtilsC 修补后，由 `experiments/2026-10-07-asap-failure-analysis/scripts/export_preds.py` 导出）

## 目的
找出 ASAP 在 DGM4 六分类上的具体能力缺陷。

结论的总体边界：
- 只有一个 checkpoint。
- 各弱点的原因大多未知。
- 没有读模型代码去验证内部机制，不对内部机制下结论。

## 方法
- 只读汇总，不做新的模型推理或统计计算。
- 每个数字后面注明来源文件，路径前缀如下：
  - **A:** = 分支 `exp/6class-decoder-diag` 提交 `8ebbb87` 的 `experiments/2026-10-25-asap-dgm4-6class-decoder-diag/results/`
  - **B:** = 分支 `exp/orig-fp-lowlevel-probe` 的 `experiments/2026-10-15-orig-fp-lowlevel-probe/results/`
- 判定词沿用各阶段冻结文件的标签。
- 没有引用 ASAP 仓库里未冻结的旧分析（如 `original_fp_analysis_v2`）。
- 所有数字由 `scripts/verify_weakness_summary.py` 逐个对照来源文件核对，输出见 `results/verify_weakness_summary.txt`。

## 参数配置
无训练或推理参数。来源文件清单见 `config.yaml`。

## 结果

### 弱点一：Original 被判假
- test 中 Original 16876 张，误报（prob_fake≥0.5）4071 张（24.1%）。来源：`B:B0/test/summary.csv`、`A:E01/E01_test_binary_refs.csv`
- 同一张图换成篡改文本后报警持续。V0 A 组 test：同图兄弟的图像维报警率 78.8%，TN 组 7.2%；差值 +71.6 pp [68.4, 74.9]，判定 支持。来源：`B:B1_V0/test/V0_tests.csv`
- 已排除的方向：
  - 低层画质（H1）：7 个有方向特征中 支持 0、证据不足 5、不支持 2；4 个无方向特征均为 无可分辨差异；jpeg_q 不可检验。来源：`B:B1_verdict/val_v5+test/h1_verdict.csv`
  - 二分类头与四头冲突：test 4071 张误报中只有 276 张（6.8%）属于“二分类判假、四维全 <0.5”，“冲突能解释大部分误报”判定 不支持。来源：`A:E03/E03_test_conflict_rates.csv`、`A:E01/E01_test_binary_refs.csv`
  - 解码规则：9 个解码器在 test 上的 Original FPR 为 0.198–0.241，六类 Macro-F1 为 0.706–0.711。来源：`A:E01/E01_test_decoders_main.csv`
- 弱线索 H2（face_bg_ratio）：test AUC H2-1 0.481、H2-2 0.416、H2-3 0.584，三项 确认，效应小。来源：`B:B1/test/h2_face_bg_ratio.csv`
- 原因：未知。

### 弱点二：篡改文本漏检
- test 中文本篡改样本 14393 条，A_tuned 解码后文本轴判“无” 3614 条（25.1%）。来源：`B:B2/test/summary.json`、`A:E02/E02_axis_error_composition.csv`
- H-T1：文本得分 AUC 0.9125 [0.9091, 0.9157]，判定 支持。来源：`B:B2/test/h_t1_auc.csv`
- H-T2：4 个特征的 AUC 为 0.502–0.530，未校正与 Holm 均为 不支持。来源：`B:B2/test/h_t2_features.csv`
- H-T3：漏检样本中命中至少 1 个真篡改词的比例 11.9%，判定 证据不足；漏检样本中 token 头完全没有预测篡改词的比例 87.8%。来源：`B:B2/test/h_t3_hit.csv`
- 已排除的方向：
  - 阈值/校准：9 个解码器（含阈值网格与四维温度缩放）在 test 上的 TS 召回为 0.712–0.727，TA 召回为 0.753–0.789。来源：`A:E01/E01_test_decoders_main.csv`
  - 阈值/校准（续）：文本得分在 FPR 10% 时的 TPR 为 80.0%。来源：`B:B2/test/h_t1_tpr_at_fpr.csv`
  - 篡改量：篡改词数 AUC 0.523、篡改词占比 AUC 0.530，均为 不支持。来源：`B:B2/test/h_t2_features.csv`
- 原因：未知。

### 弱点三：图像篡改漏检与图像轴类型混淆（只有描述性数字）
- test、A_tuned、图像轴：
  - 漏检 3433 条，占有图像篡改样本的 12.8%
  - 类型混淆 1783 条，占 6.6%
  - 误报 3769 条，占无图像篡改样本的 15.8%
- 来源：`A:E02/E02_axis_error_composition.csv`

### 弱点四：Mixed 样本被折叠成单模态
- test Mixed 7468 条：
  - A_tuned 下 2401 条（32.2%）被折叠成单模态：只剩图像 1707、只剩文本 694。来源：`A:E02/E02_test_A_tuned_axes_questions.json`
  - B_tuned 下 2169 条（29.0%）：只剩图像 1593、只剩文本 576。来源：`A:E02/E02_test_B_tuned_axes_questions.json`
- Mixed 内“联合判对 ÷ 两轴召回之积”，A_tuned：test 1.000 [0.994, 1.007]，val_holdout 0.991 [0.977, 1.005]。这与“解码后两条轴的判对近似独立”相符。来源：`A:E02/E02_axis_independence.csv`

### 错误分层（test，A_tuned，九类）
- 九类错误 14150 个：真伪 7380（52.2%）、模态 4995（35.3%）、子类型 1775（12.5%）。来源：`A:E02/E02_test_A_tuned_error_layers.csv`

## 分析与结论

### 已排除的方向（数字见上文）
- 解码规则（弱点一、弱点二）
- 低层画质（弱点一，H1）
- 二分类头与四头冲突（弱点一）
- 阈值调整与温度缩放校准（弱点二）
- 篡改量（弱点二，H-T2）

### 不能声称的事（局限）
- 不是对模型内部机制的结论。弱点四只能写成“与近似独立相符”，不能写成“模型没有跨模态推理”。
- 只有单个 checkpoint；各弱点原因未知。
- jpeg_q 在数据中是常数，H1 的“压缩”一项不可检验。
- test 对 B1 的 H1 不是独立确认集：H1 是看过 test 上的案例之后提出的。
- H2 是看过 B1 val 结果之后提出的。
- B2 的 test 是重复，不是独立确认。

### 数据警告
本仓库（目前为私有）包含以下内容，公开仓库前必须清理这些文件以及 git 历史：
- DGM4 caption 原文：`experiments/2026-10-15-orig-fp-lowlevel-probe/results/B1_V0/{val,test}/V0_caption_pairs_manual20.csv` 与 `results/smoke/B1_V0/val/V0_caption_pairs_manual20.csv`。
- DGM4 图像相对路径：B0、B1、B1_V0、B1_V1 等目录下的逐样本 CSV。
- 服务器绝对路径：脚本、README、日志。

## 下一步
由负责人决定。本文件只汇总已有结果，不提出新的假设。

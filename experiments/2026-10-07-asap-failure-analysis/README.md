# 实验：ASAP 在 DGM4 test 上的 failure mode 分析

- **日期**：2026-10-07
- **负责人**：wanhongz
- **代码版本**：[ASAP](https://github.com/karroyyy22-png/ASAP) commit `5a59424`（含 TamperDetector、lr 消融配置等，见实验 2026-06-09），**外加一处未提交的 UtilsC 修补**（见下文"方法"和 `scripts/utilsC_patch.diff`）。修补在服务器 worktree `/nas/data_2/wanhongz/ASAP_fm`（detached HEAD）里完成，没有推送到 ASAP 仓库。
- **关联日志/checkpoint目录**：
  - checkpoint：`/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16/checkpoint_best.pth`（即 baseline，见实验 2026-06-02）
  - 7/21 评估结果：同目录下的 `evaluation/results_all.txt` 和 `evaluation/raw_eval_data.npz`
  - 逐样本预测 pkl（不进仓库，在服务器 s225 上）：
    - `/nas/data_2/wanhongz/failure_analysis/preds_test_all.pkl`（test，修补后，以此为准）
    - `/nas/data_2/wanhongz/failure_analysis/preds_val_all.pkl`（val，修补后）
    - `/nas/data_2/wanhongz/failure_analysis/preds_test_all_preUtilsC.pkl`（test，修补前，仅作对照）
  - 导出日志：`/nas/data_2/wanhongz/failure_analysis/export_test_val_all_postUtilsC.log`

## 目的
在 DGM4 test（50705 条）上分析 ASAP 的错误分布，决定下一步走"物理痕迹检测"还是"图文结构化语义比对"路线。判断规则如下：
- 错误集中在 FS/FA、小脸、图像定位 → 物理路线；
- 错误集中在 TS/TA 或混合类 → 结构化语义路线；
- 两边都有 → 先走物理路线，再把结构化分数并入同一个逐脸打分框架。

## 方法
1. **UtilsC 修补**：`5a59424` 重构 forward 时，把多标签头的系数从 `UtilsC` 改成了 `UtilsB`。checkpoint 里 UtilsC=0.3785、UtilsB=0.3973，说明训练时用的是 UtilsC。推理时恢复 UtilsC（`models/ASAP.py`），改动两处：
   ```python
   # models/ASAP.py:130  恢复参数定义
   self.UtilsC = nn.Parameter(torch.tensor(0.5))
   # models/ASAP.py:504  推理分支多标签头
   logits_multicls = self.cls_head(text_feat + self.UtilsC * vision_feat)
   ```
   训练分支（:415）仍是 UtilsB，没有改，不影响推理。修补后只有多标签分数 `prob_mc` 变化（|Δ| 最大 0.11），二分类、bbox、token 结果逐位不变。
2. **导出逐样本预测**（`scripts/export_preds.py`，单卡 GPU）：每个样本导出二分类分数、多标签分数、预测框、官方 IoU、token 级预测和标签。
3. **对照 7/21 评估**（`scripts/check_full_metrics.py`、`scripts/check_multilabel_metrics.py`，CPU）。
4. **failure 分析**（`scripts/analyze_failures.py`，CPU）：
   - A：按篡改类型统计二分类、多标签头、文本定位；
   - B：图像定位和候选框选择，按人脸尺寸、面积比、人脸数分桶；
   - C："总选最大脸"诊断基线。
5. **补充诊断**（`scripts/diagnose_extra.py`，CPU）：
   - orig 误报的来源；
   - "框落在空处"和检测结果的交叉表；
   - 零重叠样本的预测框大小；
   - "总选第一个框"基线。

复现顺序：在 ASAP worktree 里打上 `utilsC_patch.diff`，运行 `export_preds.py --split test --n 0` 和 `--split val --n 0`；然后运行 `analyze_failures.py`，再运行 `diagnose_extra.py`。脚本里的路径是服务器上的绝对路径。

## 参数配置
这次是分析，没有训练参数，所以不写 config.yaml。分析口径如下：

| 项目 | 取值 |
|---|---|
| 二分类阈值 | 0.5；val 上 ACC 最大的阈值 0.7365（val ACC 85.06） |
| 多标签判定 | `prob_mc >= 0.5`，维度顺序 [FS, FA, TS, TA] |
| 候选框选择 | 预测框和各 mtcnn 框求 IoU 取最大者，等于 gt 即为选对；和所有框 IoU 都为 0 时算选错 |
| 人脸尺寸分桶 | sqrt(w·h)：<32 / 32–64 / 64–128 / 128–256 / ≥256 |
| 面积比分桶 | 在全部 26904 个带框样本上算五分位 |
| 人脸数分桶 | 2 / 3 / 4 / 5+（DGM4 的人脸篡改图至少有 2 张 mtcnn 人脸） |

## 结果
完整报告：[`results/ASAP_failure_analysis_20261007.md`](results/ASAP_failure_analysis_20261007.md)；全部表格在 `results/*.csv`。

和 7/21 评估对照：ACC 85.5813、IoU 74.4318、IoU_ACC_50/75 81.6862/74.0400、F1_tok 70.3940，都完全一致；AUC 92.0645，7/21 为 92.0633。

| 错误构成（二分类，阈值 0.5） | 占全部错误 | 该组错误率 |
|---|---|---|
| orig 误报 | 55.7% | 24.1% |
| 纯图像（FS+FA）漏检 | 26.3% | 9.9% |
| 纯文本（TS+TA）漏检 | 15.8% | 16.7% |
| 混合（4 类）漏检 | 2.2% | 2.1% |

| 候选框选择正确率 | 全部带框 | FS | FA |
|---|---|---|---|
| ASAP | 84.5% | 82.0% | 86.6% |
| 总选第一个 mtcnn 框 | 65.9% | 59.4% | 73.4% |
| 总选最大脸 | 66.2% | 59.9% | 73.4% |

## 分析与结论
- **orig 误报是最大的错误来源，而且主要由图像侧触发。** 误报的 orig 中，多标签头只有图像维 ≥ 0.5 的占 62.8%，两侧都有的占 6.6%。误报率随最大脸尺寸增大而下降（26.5% → 11.7%），也和图片来源相关（washington_post 35.2%，guardian 20.1%）。
- **纯文本类单样本错误率最高**（16.7%），TS 维的多标签漏判率为 26–28%。
- **定位失败主要是检测失败的连带结果。** ASAP 的候选框选择明显好于两个基线。选错的样本中 82.6% 是框落在空处，其中 76.5% 的多标签对应维没判出。和任何人脸都不重叠的 3115 条里，94.2% 的预测框面积不到全图的 1%，即近零框。
- **FS 漏检和人脸尺寸没有单调关系**（9.9–16.4%）；FA 漏检随尺寸增大而下降（16.2% → 4.6%）。
- **路线判断**：属于"两边都有"。物理侧约占全部错误的 6 成（61–65%），文本侧约 29%。结论是先走物理路线，再把结构化语义分数并入同一个逐脸打分框架。
- **注意事项**（详见报告）：
  - 只评估了单一 checkpoint 和单一 seed；
  - metadata 里没有扰动字段；
  - 官方 `box_iou(test=True)` 有 19 条样本被错记 IoU=1 的 bug；
  - 以前记录的 74.43% 是官方 IoU 均值，包括非人脸样本；
  - "FS 换掉的是最大的脸"不成立，gt 是最大脸的比例只有 60%；
  - 和 7/21 npz 有逐样本小差异，但聚合指标一致。

## 下一步
先走物理路线：在 DGM4 上微调一个逐脸分类器，把 orig 图里的真脸作为负样本，对候选 mtcnn 框逐个打分；之后再把图文结构化语义分数并入同一个逐脸打分框架。具体方案待定。

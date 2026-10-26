# 实验：orig 误报机制·低层图像质量探针（未运行）

- **日期**：2026-10-15
- **负责人**：wanhongz
- **关联 commit / tag**：依赖 [2026-10-07-asap-failure-analysis](../2026-10-07-asap-failure-analysis/) 的 `preds_test_all.pkl`（UtilsC 修补后）与其 `analyze_failures.py` 加载函数
- **状态**：脚本已写好、只在本地假数据上跑通流程，**尚未在服务器原图上运行，没有结果**

## 目的
检验假设 H1：低层图像质量（噪声 / 模糊 / JPEG 压缩）是否和 ASAP 在未经干预的真实新闻（orig）上的误报相关。
线索来自 32 对 Original FP/TN 配对审查中的配对05（同场景、caption 只差 "Monday"，FP 版有明显颗粒噪声）。
注意：在审查页面嵌入的 64 张图（尺寸与原图一致，但不确定是否被重新压缩过）上，噪声指标并未区分 FP/TN，所以 H1 目前只是待检验的假设。

## 方法
只用原样输入，不改动图像或文本（`scripts/probe_lowlevel.py`，只用 CPU）：
- S1：orig 内 FP vs TN 在各指标上的 AUC，按来源分层；误报率按来源内噪声五分位统计。
- S2：face_swap / face_attribute 图像与 orig 在同一指标上的分布差异（检查数据集捷径）。
- S3：数据中天然存在的孪生对（同目录、caption Jaccard ≥ 0.8、预测一真一假）的符号检验。

指标：Immerkaer 噪声 σ（全图、最大脸）、Laplacian 方差、由量化表反推的 JPEG 质量、bytes/px。

另有 `scripts/verify_pairs.py`（只用 CPU、不改输入）：针对审查页面的配对 5 / 9 / 22 / 29，
利用 DGM4 的数据结构做天然对照——text_swap/text_attribute 样本沿用原图、face_swap/face_attribute 样本沿用原 caption，
检查误报"跟着图片走"还是"跟着 caption 走"；判定标准写在脚本开头，跑之前定好。

## 参数配置
分析实验，无训练参数。`--n_fake 5000 --workers 8 --twin_jaccard 0.8`；图像根目录 `IMG_ROOT`（默认 `/nas/data_2/wanhongz/datasets`）。

## 结果
（B 阶段进行中：val 部分已运行，test 未运行；完整结果卡待 test 重复后补写。冻结文件 `frozen_hypotheses.json` v7，sha256 `490a02f4…`）

B0 test（只读 metadata 与 pkl，描述性）在冻结文件 v1 下于 10-25 23:25 运行；B1（含 H2）、V0、V1、V2 的 test 尚未运行；冻结文件后续各版的修改均不由 B0 test 结果引起。

### V1(c) 与 V2(a) 数字并列（val，只列数字）
| 指标 | 子集 | 对数 | 值 [95% CI，id 簇] |
|---|---|---|---|
| V1(c) 同句兄弟该维分数 > 原图该维分数的比例 | 全部 | 8787 | 95.4% [95.0, 95.8] |
| V1(c) 同上 | 原图为 FP | 2251 | 88.6% [87.3, 90.0] |
| V1(c) 同上 | 原图为 TN | 6536 | 97.7% [97.4, 98.1] |
| V2(a) 原图与兄弟 own-dim logit 的 Spearman | 全部 | 8787 | −0.019 [−0.040, 0.002] |
| V2(a) 原图与兄弟 own-dim logit 的 Pearson R² | 全部 | 8787 | 0.006 [0.003, 0.009] |
| V2(a) Spearman / R² | FS 兄弟 | 4788 | 0.041 [0.015, 0.069] / 0.015 [0.009, 0.020] |
| V2(a) Spearman / R² | FA 兄弟 | 3999 | −0.063 [−0.093, −0.030] / 0.000 [0.000, 0.001] |

### 表格说明
- `results/B1/val_v5/h2_face_bg_ratio.csv` 中两行"(描述) FS/FA vs 未改图"的 `dir_as_predicted` 列：无预测方向，该列无意义。

## 分析与结论
（待运行）

## 下一步
训练集同源排查：test orig 在 train 中是否存在同 id / 近重复且被标为篡改的样本，比较 FP 与 TN 的比例。

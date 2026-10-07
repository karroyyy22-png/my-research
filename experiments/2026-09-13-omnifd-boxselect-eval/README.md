# 实验：用 OmniFD 做候选框选择的评估

- **日期**：2026-09-13（原始评估）；2026-10-07 按严格口径重算
- **负责人**：wanhongz
- **代码版本**：评估脚本不属于 ASAP 仓库，副本放在本目录 `scripts/`：
  - `eval_omnifd_boxselect.py`：只给伪造样本的候选框打分；
  - `eval_omnifd_allbox.py`：orig 的候选框也打分，其余与上一个相同；
  - `recheck_boxselect.py`：2026-10-07 只用 CPU 重算，不重跑 OmniFD。
  - OmniFD 代码：`/nas/data_2/wanhongz/code/OmniFD`，checkpoint `pretrained/omnifd_swin-s_50k_forgerynet.pth`，配置 `configs_omnifd/single/omnifd_swin-s_50k_forgerynet_image.py`。
- **关联日志/结果目录**（服务器）：
  - 每个候选框的 OmniFD 分数：`/nas/data_2/wanhongz/face_cropped_features/boxselect_eval.npz`（71513 个框）、`allbox_eval.npz`（135140 个框，含 orig）。两份在伪造样本上的分数一致，最大差 5.7e-7。
  - 原始脚本：`/nas/data_2/wanhongz/eval_omnifd_boxselect.py`、`/nas/data_2/wanhongz/eval_omnifd_allbox.py`

## 目的
DGM4 的图像定位本质上是"在 2–5 个 mtcnn 候选框里选出被篡改的那张脸"（见实验 2026-09-13-localization-task-nature）。这次评估的问题是：只用 OmniFD 的人脸伪造分数挑脸，能挑对多少？以此判断 OmniFD 是否有逐脸判别能力。

## 方法
1. DGM4 test 集中，每个带 `fake_image_box` 的伪造样本共 26904 条。把每个 mtcnn 框按原样裁剪（不外扩），resize 到 224，送进 OmniFD 图像头，取 softmax 后的伪造概率。
2. 每个样本选 OmniFD 分数最高的候选框。所有样本中都没有出现分数并列。
3. **两种口径：**
   - **旧口径（9/13 报告）**：被选框和 gt 的 IoU 均值；
   - **今天的口径（严格选对率）**：被选框等于 gt 框才算选对。gt 框总是恰好等于某一个 mtcnn 框，所以这个定义没有歧义。
4. **基线**："总选第一个 mtcnn 框"，两种口径都算；ASAP 的数字取自实验 2026-10-07。

## 参数配置
没有训练参数，不写 config.yaml。裁剪外扩比例 MARGIN=0，输入 224×224，ImageNet 均值和方差归一化，batch 64。

## 结果
2026-10-07 重算，完整表格见 `results/table_E_omnifd_boxselect_recheck.csv`，日志见 `results/recheck_boxselect_log.txt`。

| 组 | n | OmniFD 选对率 | OmniFD IoU 均值（旧口径） | 第一个框选对率 | 第一个框 IoU 均值（旧口径） | ASAP 选对率 |
|---|---|---|---|---|---|---|
| 全部带框 | 26904 | **54.81** | 54.90 | **65.85** | 65.90 | **84.52** |
| face_swap | 10515 | 58.05 | 58.13 | 59.35 | 59.43 | 82.04 |
| face_attribute | 8921 | 51.16 | 51.25 | 73.37 | 73.40 | 86.57 |
| FS 含混合 | 14568 | 57.93 | 58.02 | 59.76 | 59.83 | 82.55 |
| FA 含混合 | 12336 | 51.13 | 51.22 | 73.04 | 73.07 | 86.85 |

（单位 %。）

- **以前记录的 54.90% 是旧口径**，即 OmniFD 被选框和 gt 的 IoU 均值，不是选对率。严格选对率是 54.81%。
- **以前记录的 65.90% 也是旧口径**，即第一个框的 IoU 均值。严格选对率是 65.85%。两者差 0.053 个百分点，全部来自 278 个样本：它们的第一个框不是 gt，但和 gt 有部分重叠（IoU>0），这部分 IoU 之和除以 N 正好等于 0.053。所以 65.90 和 65.85 之间的差距完全由口径解释。
- **以前作为 ASAP 对照的 74.43%，实际上是官方 IoU 均值**，而且包括了没有人脸框的样本。按今天的口径，ASAP 的严格选对率是 84.52%。

## 分析与结论
- **OmniFD 挑脸不如"总选第一个框"。** 全体为 54.81% 对 65.85%；face_attribute 上差距最大，51.16% 对 73.37%。平均每图 2.66 张脸，随机选框的期望选对率是 mean(1/n)=41.3%；OmniFD 在 face_attribute 上只比随机高约 10 个百分点。
- **face_swap 上 OmniFD 和第一个框基线接近**（58.05% 对 59.35%），也没有超过。
- 结合 OmniFD 单独判别 AUC 只有 66.56%（实验 2026-06-06）来看，这个在 ForgeryNet 上训练的 OmniFD checkpoint，在 DGM4 的人脸上**不具备可靠的逐脸判别能力**。这也解释了为什么把它的特征融合进 ASAP（实验 2026-07-21 和 2026-09-12）没有带来提升。

## 下一步
不直接用这个 OmniFD checkpoint 做逐脸判别。按 2026-10-07 的结论，在 DGM4 上微调逐脸分类器，把 orig 里的真脸作为负样本，再用同样的严格选对率评估它的挑脸能力，和 ASAP 的 84.52%、第一个框的 65.85% 对比。

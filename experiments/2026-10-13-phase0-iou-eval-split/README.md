# 实验：阶段 0 —— 官方 IoU 的拆分评测与 CSCL 框后处理的影响

- **日期**：2026-10-12 ~ 2026-10-13（ASAP 全量评测 10-12，CSCL 全量评测 10-13）
- **负责人**：wanhongz
- **代码版本**：
  - [ASAP](https://github.com/karroyyy22-png/ASAP) 分支 `exp/phase0-box-postproc`，commit `f9d2e97`（从 `exp/omnifd-residual-gate` 拉出，已 push）。`test.py` 新增 `--box_postproc {none,cscl,zero}` / `--max_samples` / `--eval_dirname` 和拆分统计；`models/ASAP.py` 恢复 `UtilsC`。完整 diff：[`scripts/ASAP/phase0_f9d2e97.diff`](scripts/ASAP/phase0_f9d2e97.diff)
  - CSCL：clone 自 [liyih/CSCL](https://github.com/liyih/CSCL)（上游 main `6467274`），本地分支 `phase0`，commit `4b49db1`。**没有 push 权限，改动只在服务器上**，本仓库的 [`scripts/CSCL/phase0_4b49db1.patch`](scripts/CSCL/phase0_4b49db1.patch) 是唯一的线上备份（在上游 `6467274` 上 `git am` 即可还原）。改动：`models/CSCL.py`（后处理开关 + 修复第 11 行导入）、`test.py`（同样的拆分统计）、新建 `configs/test_phase0.yaml`
- **关联日志/checkpoint目录**：
  - ASAP checkpoint：`/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16/checkpoint_best.pth`（即 baseline，见实验 2026-06-02）
  - ASAP 评测输出：同目录下 `eval_phase0_full_{none,cscl,zero}/`（逐样本 `raw_eval_data.npz` 每个 9.3M，不进仓库）
  - CSCL checkpoint：`/nas/data_2/wanhongz/CSCL/code/MultiModal-DeepFake-main/results/CSCL/checkpoint_49.pth`
  - CSCL 评测输出：同目录下 `eval_phase0_full_{cscl,none}/`、`eval_phase0_full_zero_ABORTED/`
  - 修改前备份（`*.bak_20261012`）、patch 脚本、运行日志：`/nas/data_2/wanhongz/backup_phase0/`（CSCL 部分在 `cscl/` 下）

## 目的
导师要求把 CSCL（CVPR25）的图像模块移植到 ASAP（阶段 1）。CSCL 的 IoUmean 是 84.07，我们复现的 ASAP 是 74.43，差 9.6 分。移植前先确认这个差距是否真实。

**怀疑的原因**：
- CSCL 推理时有一个框后处理：多标签头的 FS、FA 两个 logit 都 < 0.5 时，把预测框置零。
- DGM4 的官方 IoU 规则是：真值为空框时，预测框四个坐标都 < 1e-4 就记 IoU = 1。
- test 集里 46.94% 的样本图像未被篡改、真值为空框。

因此，CSCL 的高分有可能主要来自"在未篡改样本上输出空框"这一后处理，而不是定位能力。

**判定**：
- 给 ASAP 加上同样的后处理后差距大幅缩小，或者 CSCL 去掉后处理后大幅下降 → 后处理是主因，需要重新考虑阶段 1；
- 否则 → CSCL 的优势是真实的，进入阶段 1。

同时把官方 IoU 拆成"篡改样本的定位"和"未篡改样本是否输出空框"两部分，看清差距具体来自哪里。

## 方法
### 1. 框后处理的三种模式
两个模型都加了同样的开关，阈值作用在**原始 logit** 上（`img_logit_min/max` 有负值，确认没有经过 sigmoid）：

| 模式 | 规则 | 等价的概率阈值 |
|---|---|---|
| `none` | 不置零（ASAP 原始行为） | — |
| `cscl` | FS、FA 的 logit 都 < 0.5 时置零（和 CSCL 官方代码完全一致，CSCL 默认） | sigmoid(0.5) ≈ 0.62 |
| `zero` | FS、FA 的 logit 都 < 0 时置零 | 0.5 |

ASAP 侧（`test.py`，在 `box_cxcywh_to_xyxy` 之前）：
```python
if args.box_postproc != 'none':
    _thr = 0.5 if args.box_postproc == 'cscl' else 0.0
    _pp_mask = (logits_multicls[:, 0] < _thr) & (logits_multicls[:, 1] < _thr)
    output_coord = output_coord.clone()
    output_coord[_pp_mask] = 0
```
CSCL 侧（`models/CSCL.py:338` 附近，原有后处理改为可切换，默认 `cscl` = 原始行为）：
```python
_pp = getattr(self, 'phase0_box_postproc', 'cscl')
if _pp != 'none':
    _thr = 0.5 if _pp == 'cscl' else 0.0
    logits_multicls_mask = ((logits_multicls[:, 0]<_thr)&(logits_multicls[:, 1]<_thr))
    output_coord[logits_multicls_mask] = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(output_coord.device)
```

### 2. 拆分指标
IoU 本身直接沿用官方 `box_ops.box_iou(..., test=True)` 的逐样本结果，只是按样本分组再求均值：
- **未篡改样本**：真值框 `fake_image_box` 四个值之和为 0（23801 条，包括 orig 和纯文本篡改）；
- **篡改样本**：其余 26904 条（图像有 FS/FA 篡改）；
- **预测为空框**：转成 xyxy 后四个坐标都 < 1e-4（和官方规则一致）。

| 指标 | 定义 |
|---|---|
| 篡改 IoUmean / IoU50 / IoU75 | 篡改样本上官方 IoU 的均值，以及 > 0.5 / > 0.75 的比例 |
| 篡改 空框率 | 篡改样本中预测为空框的比例（这些样本 IoU 必为 0，属于"该框的没框"） |
| 未篡改 空框率 | 未篡改样本中预测为空框的比例 |
| 未篡改 IoUmean | 官方规则下未篡改样本的 IoU 只能是 0 或 1，所以理论上 **= 未篡改空框率**（实测完全相等，作为自检） |
| 候选框选择准确率 | 只看篡改样本：预测框和各 `mtcnn_boxes` 求 IoU，取最大者；它等于真值框，且 IoU > 0，记为正确。预测空框必然记为错 |

`mtcnn_boxes` 和真值框按图片路径从 `test.json` 查表，再除以 (W, H) 归一化。两个自检：
- `gt_norm_mismatch`：查表得到的真值框和 dataloader 给的真值框差 > 1e-3 的条数；
- `cand_meta_missing`：查不到 metadata 的条数。

两者全量都是 0。拆分统计的完整代码见 `scripts/` 下的 diff/patch，两边逻辑相同。

### 3. 官方 IoU 的分解
记 p_t = 26904 / 50705 = 53.06%（篡改占比），p_u = 23801 / 50705 = 46.94%，则

```
官方 IoUmean = p_t × 篡改IoUmean + p_u × 未篡改空框率
             = 53.06% × 篡改IoUmean + 46.94% × 未篡改空框率
```

两个模型（或两种设置）之间的差距可以精确拆成两项：

```
Δ官方IoU = 53.06% × Δ篡改IoUmean  +  46.94% × Δ未篡改空框率
           （定位，含"该框没框"）      （判断图像是否被篡改）
```

### 4. 运行流程
1. 先跑试跑，确认代码和自检正常：
   - ASAP：`--max_samples 200`，batch 64，实际取整为前 256 条；
   - CSCL：前 256 条。
2. 再跑全量：
   - ASAP：`scripts/ASAP/run_phase0_full.sh`，三种模式顺序执行；
   - CSCL：`scripts/CSCL/run_cscl_full.sh`，三种模式每隔 30 秒在同一张 GPU 上并行启动。
3. 两个运行脚本都用 marker 文件 + `find -newer` 记录了运行期间被修改的文件，确认只写入了新建的 `eval_phase0_*` 目录。

## 参数配置
评测实验，没有训练参数。关键设置也写进了 [`config.yaml`](config.yaml)。

| 项目 | ASAP | CSCL |
|---|---|---|
| checkpoint | `log20260602_134300_train_4gpu_batch16/checkpoint_best.pth` | `CSCL/checkpoint_49.pth` |
| 代码 | ASAP `f9d2e97` | CSCL 上游 `6467274` + 本地 `4b49db1` |
| 训练数据 | DGM4 train | DGM4 train + val（CSCL 官方设置） |
| 评测数据 | DGM4 test 全量 50705 条 | 同左 |
| 服务器 / GPU | s225，RTX A4000（`CUDA_VISIBLE_DEVICES=2`） | prmigreen001，TITAN V 12GB |
| batch_size_val | 64 | 32（为适配 12GB 显存；batch 大小不影响评测结果） |
| text encoder | bert-base-uncased | roberta-base |
| config | `configs/test.yaml` | `configs/test_phase0.yaml`（新建，见"问题 4"） |
| 运行时间 | 2026-10-12 16:47–17:29（每种模式约 14 分钟） | 2026-10-13 14:08–14:37 |

## 结果
各模式的原始输出：`results/{ASAP,CSCL}/<模式>/phase0_split_<模式>.json`、`results_all.txt` 等。

### 主表（test 全量 50705 条：篡改 26904，未篡改 23801）

| 指标 | ASAP-none | ASAP-cscl | ASAP-zero | CSCL-none | **CSCL-cscl（官方）** |
|---|---|---|---|---|---|
| 全部 IoUmean | 74.43 | 75.94 | 75.98 | 83.90 | **84.09** |
| 全部 IoU50 | 81.69 | 82.94 | 83.06 | 90.36 | 90.48 |
| 全部 IoU75 | 74.04 | 75.74 | 75.73 | 86.93 | 87.14 |
| 篡改 IoUmean | 68.90 | 66.92 | 67.62 | 76.66 | 76.02 |
| 篡改 IoU50 | 82.57 | 80.09 | 80.96 | 88.83 | 88.06 |
| 篡改 IoU75 | 68.16 | 66.53 | 67.15 | 82.36 | 81.76 |
| 篡改 空框率（误输出空框） | 9.18 | 13.46 | 12.31 | 8.18 | 9.38 |
| 未篡改 空框率（= 未篡改 IoUmean） | 80.69 | 86.15 | 85.42 | 92.09 | 93.21 |
| 候选框选择准确率（篡改样本） | 84.52 | 81.82 | 82.73 | 89.30 | 88.47 |

CSCL-zero 没有结果（见"问题 7"）。

### 后处理的效果（按分解公式）

| 变化 | Δ官方IoU | 篡改侧 53.06%×Δ | 未篡改侧 46.94%×Δ |
|---|---|---|---|
| ASAP：none → cscl | **+1.51** | −1.05（篡改IoU −1.98） | +2.56（空框率 +5.46） |
| ASAP：none → zero | +1.54 | −0.68（篡改IoU −1.28） | +2.22（空框率 +4.74） |
| CSCL：cscl → none | **−0.18** | +0.34（篡改IoU +0.64） | −0.52（空框率 −1.12） |

后处理的作用是用篡改样本的漏框，换取未篡改样本的空框。两个模型上这笔交换的净收益都只有 1.5 分左右或更小。

### 差距拆解

| 对比 | 总差距 | 篡改侧（定位） | 未篡改侧（输出空框） |
|---|---|---|---|
| CSCL-cscl（官方）vs ASAP-none（原样） | **9.66** | **3.78（39%）** | **5.88（61%）** |
| CSCL-none vs ASAP-none（都不加后处理） | 9.47 | 4.12（43%） | 5.35（57%） |

- 两边篡改样本的空框率相近：ASAP 9.18%，CSCL-none 8.18%，CSCL-cscl 9.38%。所以篡改侧 3.78 分的差距不是"少漏框"造成的，而是真正的定位精度差距。
- 篡改 IoU50 只差 5.5（82.57 vs 88.06），IoU75 却差 13.6（68.16 vs 81.76），说明 CSCL 的框更紧。
- 即使两边都不加后处理，CSCL 的未篡改空框率也有 92.09%，ASAP 是 80.69%。所以 CSCL 输出空框主要靠框回归头本身，后处理只多贡献了约 1 个百分点。

### CSCL 复现核对

| | AUC | mAP | IoUmean | F1_tok |
|---|---|---|---|---|
| CSCL 论文 | 96.34 | 92.48 | 84.07 | 76.62 |
| 本次（CSCL-cscl） | 96.34 | 92.49 | 84.09 | 76.60 |

差距都在 0.02 以内，复现成功。

### 其他指标（与后处理模式无关）
后处理只改预测框，二分类、多标签、文本 token 的指标在各模式下完全相同（已逐项核对 `results_all.txt`）。

| | AUC | ACC | EER | mAP | OF1 | CF1 | ACC_tok | F1_tok |
|---|---|---|---|---|---|---|---|---|
| ASAP | 92.0645 | 85.5813 | 15.2975 | 84.1607 | 78.8387 | 78.2864 | 91.3561 | 70.3940 |
| CSCL | 96.3413 | 90.3303 | 9.8898 | 92.4861 | 86.9323 | 86.2139 | 92.6356 | 76.5982 |

ASAP 的 `results_all.txt` 是从 `exp/omnifd-residual-gate` 继承的输出格式，里面有 OmniFD 分数级融合的 Before/After 对比（alpha=0.50），这和阶段 0 无关。**ASAP 自身的指标看 Before 列和 JSON 行即可。** JSON 行里的 `val_AUC_cls` 是融合后的 80.3265，不要误用。

### 自检（全量，两个模型的所有模式都通过）
- `gt_norm_mismatch = 0`：坐标换算正确；
- `cand_meta_missing = 0`：按路径查表全部命中（CSCL 去掉了路径前缀，见"问题 5"）；
- 未篡改 IoUmean 与未篡改空框率完全相等；
- `img_logit_min < 0`（ASAP −15.47，CSCL −13.99）：取到的是原始 logit。

### 试跑（同一批前 256 条：篡改 142，未篡改 114），仅作参考

| | 全部 IoUmean | 篡改 IoUmean | 未篡改 空框率 | 候选框选择准确率 |
|---|---|---|---|---|
| ASAP-none（UtilsC 修复后） | 71.49 | 69.03 | 74.56 | 84.51 |
| CSCL-none | 84.24 | 78.64 | 91.23 | 91.55 |
| CSCL-cscl | 84.18 | 77.11 | 92.98 | 89.44 |

试跑结果在 `results/smoke/`。ASAP 修复 UtilsC 之前跑的 `eval_phase0_smoke200_{none,cscl,zero}` 已作废，没有放进本仓库。

## 实验中发现的问题与处理
1. **ASAP 代码和 checkpoint 不一致（UtilsC）**
   - checkpoint 是 2026-06-02 训练的，当时 MLC 头用 `UtilsC`。checkpoint 里 `UtilsC = 0.378522`，已被训练过。
   - 2026-07-10 的提交 `5a59424` 把它改成了 `UtilsB`，并注释掉了参数定义，所以加载时出现 `unexpected_keys=['UtilsC']`，`UtilsC` 被静默丢弃。
   - `f9d2e97` 在 `models/ASAP.py` 里恢复了 3 行：参数定义（:136）、**训练分支**（:449）、推理分支（:565）。
     ```python
     self.UtilsC = nn.Parameter(torch.tensor(0.5))
     output_cls = self.cls_head(text_feat + self.UtilsC * vision_feat)        # 训练分支
     logits_multicls = self.cls_head(text_feat + self.UtilsC * vision_feat)   # 推理分支
     ```
   - 和 10-07 那次的修补（只改推理分支，没有提交）不同，这次连训练分支也改了。**以后在这个分支上训练，MLC 头用的是 UtilsC**，这和 0602 checkpoint 的训练方式一致。
   - 修复前后 256 条试跑对比：AUC、IoU、文本指标不变，mAP 86.3997 → 86.3999，影响很小。
   - **对历史结果的影响**：2026-07-21 之后用新代码评测这个 checkpoint 得到的多标签指标（mAP、F1_FS/FA/TS/TA），以及 OmniFD 各实验里作为对照的基线多标签指标，可能有轻微偏差。AUC、IoU、文本指标不受影响。
   - `5a59424` 前后的 ASAP.py 对比存档在 `scripts/ASAP/diff_5a59424_ASAP.txt`。
2. **CSCL 官方代码的导入 bug**：`models/CSCL.py` 第 11 行 `from consist_modeling import ...` 和第 15 行的相对导入冲突，导致无法导入。改成 `from .consist_modeling import ...` 后正常。
3. **CSCL `test.py` 的原地二值化陷阱**：第 184–187 行对 `logits_multicls` 做了原地二值化（`cls_pred` 是它的视图）。拆分统计需要原始 logit，所以在模型输出后立刻 `.clone()` 保存一份。
4. **CSCL `test.yaml` 引用了不存在的 `trainval.json`**：`create_dataset` 会同时构造训练集，所以新建了 `configs/test_phase0.yaml`，把 `train_file` 改成 `train.json`（训练集不参与评测），并设 `batch_size_val: 32`。
5. **CSCL dataset 返回的路径带 `../../datasets/` 前缀**：查 metadata 前用 `p.split('datasets/', 1)[1]` 去掉前缀。
6. **评测条件差异**：
   - CSCL 用 train+val 训练，ASAP 只用 train；
   - ASAP 论文报告 IoUm 77.35，我们自己训练的 checkpoint 是 74.43；
   - 两边的 GPU 和 batch 不同（见"参数配置"），batch 大小不影响评测结果。

   所以 9.66 分里有一部分可能来自训练数据量和 checkpoint 本身。本次没有区分这一部分。
7. **CSCL zero 模式没有完成**：三种模式在同一张 TITAN V（12GB）上并行启动，显存不足，zero 模式在 14:12 被终止（`Terminated`，exit=143）。目录已改名为 `eval_phase0_full_zero_ABORTED`，只留下 `shell_all.txt`。ASAP 的 zero 结果和 cscl 只差 0.03，所以缺这一组不影响结论。
8. **服务器状况**：
   - s247 上 `nvidia-smi` 报 `Unable to determine the device handle for GPU0000:C1:00.0: Unknown Error`，随后整台机器无法连接，需要报告给管理员；
   - CSCL 环境（torch 1.9.0+cu111）不支持 RTX 6000 Ada（s50、irielab-Super-Server），所以 CSCL 改在 prmigreen001 上评测。
9. **沿用官方 `box_iou` 的已知问题**：10-07 的报告发现官方 `box_iou(test=True)` 有 19 条样本被错记为 IoU=1。本次的拆分统计直接用官方 IoU，没有修正这一点。它对全部 IoUmean 的影响最多 19/50705 ≈ 0.04 分，不影响结论。

## 分析与结论
- **后处理不是差距的来源。**
  - 给 ASAP 加上 CSCL 的后处理，只 +1.51（zero 模式 +1.54）；
  - 给 CSCL 去掉后处理，只 −0.18；
  - 对比 9.66 分的总差距，后处理的贡献很小。
- **官方 IoU 混合了两件事。** 官方 IoU = 53.06% × 篡改样本 IoU + 46.94% × 未篡改样本空框率，也就是把"判断图像是否被篡改"和"定位"混在了一起。只看官方 IoU，会把检测能力的差别误读成定位能力的差别。
- **9.66 分的差距**（CSCL 官方设置 vs ASAP 原样）：
  - 61%（5.88）来自未篡改样本，即会不会输出空框；
  - 39%（3.78）来自篡改样本的定位精度。
- **篡改侧的差距是真实的定位差距。** 两边篡改样本的空框率相近，差距集中在 IoU75（68.16 vs 82.36），说明 CSCL 的框更紧。
- **CSCL 输出空框靠的是框回归头本身。** 不加后处理时，它的未篡改空框率已经是 92.09%（ASAP 80.69%）。
- **结论**：CSCL 的优势是真实的，可以进入阶段 1，把 CSCL 的图像模块移植到 ASAP。但阶段 1 的评估必须同时报告拆分指标（篡改 IoUmean/IoU50/IoU75、篡改空框率、未篡改空框率、候选框选择准确率），否则无法判断提升来自定位还是来自检测。
- **局限**：
  - 只评估了单一 checkpoint；
  - 训练数据量不同（train vs train+val）带来的影响没有分离；
  - CSCL-zero 缺失。

## 下一步
1. **阶段 1**：把 CSCL 的图像模块移植到 ASAP。评估时用 `f9d2e97` 的 `--box_postproc` 和 `phase0_split_*.json`，在原有指标之外报告上面的拆分指标。考虑到训练数据差异，必要时补一组只用 train 训练的对照。
2. **snap 到 MTCNN 候选框（未验证）**：
   - ASAP 的候选框选择准确率是 84.52%。如果把预测框替换成和它 IoU 最大的 MTCNN 框（snap），选对的样本 IoU 都会变成 1。
   - 粗略估计：篡改 IoUmean 可能从 68.9 升到 85 左右，官方 IoU 上涨约 8 分，而且不需要训练。
   - 这和阶段 2（对 MTCNN 候选框打分）的思路一致，可以用 ASAP 的 `raw_eval_data.npz` 或逐样本预测离线验证。
   - **需要和导师确认**：测试时使用 metadata 里的 `mtcnn_boxes` 是否合理。"真值框一定是某个 MTCNN 框"是 DGM4 数据集构造带来的特点（见实验 2026-09-13 localization-task-nature），审稿人可能会质疑。
3. **UtilsC 对历史结果的影响**：在相关实验（2026-07-21 之后用 `5a59424` 及其后代码评测 0602 checkpoint 的多标签指标，包括 OmniFD 各实验的基线）的 README 里加注说明，必要时用 `f9d2e97` 重新评测基线的多标签指标。
4. 把 s247 的 GPU 故障报告给管理员（如果还没报告）。

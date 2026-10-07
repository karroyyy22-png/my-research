# 实验：OmniFD 残差加和诊断（含随机噪声对照）

- **日期**：2026-09-12 ~ 2026-09-13
- **负责人**：wanhongz
- **代码版本**：[ASAP](https://github.com/karroyyy22-png/ASAP) 分支 `exp/omnifd-residual-gate`，commit `6e51a95`（基于 main 的 `484a230`）。
  - 注意：这个 commit 里 `train.py` 的 `UNFROZEN_KEYWORDS` 改动是无条件生效的。在该分支上跑局部 token 模式时，`omni_proj` 和 `aggregator` 会被冻结，和 484a230 不同；要复现局部 token 实验，请用 main 上的 `484a230`。
- **关联日志/checkpoint目录**（服务器 s50）：
  - 真实特征组：`/nas/data_2/wanhongz/ASAP/results/logomni_residual_exp01/`
  - 噪声对照组：`/nas/data_2/wanhongz/ASAP/results/logomni_noise_control_exp01/`
  - 第一次启动在 epoch 0 崩溃：`/nas/data_2/wanhongz/ASAP/results/logomni_residual_exp01_crashed_ep0/`（不计入结果）
  - 调试运行：`results/logdebug_alpha_lr_fix/`、`logdebug_alpha_grad/`、`logdebug_residual_check/`
  - 初始化 checkpoint：baseline `log20260602_134300_train_4gpu_batch16/checkpoint_best.pth`
  - 两组的逐 epoch 日志副本：`results/log_residual_exp01.txt`、`results/log_noise_control_exp01.txt`

## 目的
局部 token 融合（实验 2026-07-21）没有带来提升，但无法判断原因：是模型没用上 OmniFD 特征，还是 OmniFD 特征本身没有判别信息？这次改成结构最简单的残差加和，用一个可学习标量 alpha 直接观察模型对 OmniFD 通路的"态度"：
- alpha 增大，说明模型在主动使用这条通路；
- alpha 被压到 0，说明模型在主动关掉它。

同时设一个随机噪声对照组，排除"alpha 只是在拟合噪声"的可能。

## 方法
在 `vision_feat` 上加一条门控残差，只对有脸的样本生效（`models/ASAP.py`，6e51a95）：

```python
omni_proj_out = self.omni_residual_proj(omni_feat_global)   # LayerNorm(512) + Linear(512 -> 768)
mask = omni_face_mask.to(vision_feat.dtype).unsqueeze(-1)   # 有脸=1
vision_feat = vision_feat + mask * self.omni_residual_alpha * omni_proj_out   # alpha 初值 0.1
```

- **只解冻** `omni_residual*`、`cls_head`、`itm_head`，不解冻 aggregator。这样 AUC 的变化只可能来自新增的 OmniFD 通路。
- **alpha 单独一个 param group**：lr 1e-3，weight_decay 0，每次 step 前强制恢复 lr，不受 cosine 调度影响。原因有两个：
  - 用主干的 lr（1.25e-6），alpha 在 50 个 epoch 里也只能移动约 0.27，看不出趋势；
  - weight_decay 会把 alpha 持续拉向 0，和"模型主动关掉通路"的信号混在一起。
- **噪声对照组**（`--omni_noise_control`）：把每张图的 OmniFD 全局特征换成固定的随机向量。每张图的向量由 img_dir 做种子生成，std 0.255，和真实 OmniFD 全局特征的 std 一致。其他设置完全相同。

## 参数配置
不另写 config.yaml。实际运行用的是 `configs/train.yaml`，其中优化器和调度器参数与仓库里的 `configs/train_residual_debug.yaml` 相同。

| 参数 | 真实特征组 | 噪声对照组 |
|---|---|---|
| 开关 | `--use_omni_fusion_residual` | 同左，加 `--omni_noise_control --omni_noise_std 0.255` |
| 主干优化器 | adamW，lr 1.25e-6，lr_img 6.25e-6，weight_decay 0.02 | 同左 |
| alpha | lr 1e-3，weight_decay 0，初值 0.1 | 同左 |
| 调度 | cosine，50 epoch，warmup 10 | 同左 |
| 实际跑完的 epoch | 14（epoch 0–13） | **4（epoch 0–3）** |
| 硬件 | s50 单卡，batch 16 | 同左 |

## 结果
**alpha 轨迹和 val AUC**（训练时在 val 上评估；同一 baseline checkpoint 的 val AUC 为 **91.6076**）：

| epoch | 0 | 1 | 2 | 3 | 6 | 9 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|
| 真实特征组 alpha | 0.909 | 1.238 | 1.361 | 1.304 | 1.485 | 1.596 | 1.787 | 1.778 |
| 真实特征组 val AUC | 91.569 | 91.524 | 91.493 | 91.476 | 91.402 | 91.386 | 91.339 | 91.306 |
| 噪声组 alpha | −0.002 | −0.028 | 0.102 | −0.018 | — | — | — | — |
| 噪声组 val AUC | 91.555 | 91.504 | 91.516 | 91.496 | — | — | — | — |

- **真实特征组**：alpha 从 0.909 持续增大到 1.778；val AUC 从 91.57 降到 91.31，相对 val 基线从 −0.04 一路降到 −0.30。val ACC 从 85.08 降到 84.97，基线是 85.00。IoU 各 epoch 不变（73.86），因为 bbox 头被冻结。
- **噪声对照组**：只跑了 4 个 epoch。alpha 在 0 附近（−0.03 ~ 0.10），val AUC 和真实特征组同 epoch 相比相差不超过 0.025（依次为 −0.015、−0.020、+0.023、+0.021）。
- 这次没有做 test 集评估。

**OmniFD 单独的判别力**（出自实验 2026-06-06 和 2026-09-13，用作背景）：
- OmniFD 单独判别真伪：AUC 66.56%；
- 在 ASAP 判错的 2346 条样本上，OmniFD 判对的只有 46.16%；
- 用 OmniFD 分数在候选框里挑脸：严格选对率 54.81%（FS 58.05%，FA 51.16%），低于"总选第一个框"的基线 65.85%，远低于 ASAP 的 84.52%。以前写的 54.90% 是被选框和 gt 的 IoU 均值，不是选对率，详见实验 2026-09-13-omnifd-boxselect-eval。
- 以前作为 ASAP 对照的 74.43%，实际上是官方 IoU 均值（包括非人脸样本），不是候选框选择率。今天的口径是：ASAP 选对率 84.5%，第一个框 65.85%。

## 分析与结论
- **alpha 变大，但 AUC 下降。** 模型确实越来越"用"OmniFD 通路，但用得越多 val AUC 越低。这说明加进来的信号对真伪判别没有正向贡献，alpha 的增长更像是在拟合训练集。
- **噪声对照组的 AUC 和真实特征组几乎一样。** 至少在前 4 个 epoch，换成随机向量对 AUC 的影响和真实 OmniFD 特征没有区别；不同的只是 alpha：噪声组的 alpha 停在 0 附近，真实特征组的 alpha 持续增大。
- 结合 OmniFD 单独的判别力（AUC 66.56%；在 ASAP 错的样本上不如随机；挑脸不如"总选第一个框"）来看，**这个 OmniFD checkpoint 的全局特征不能为 ASAP 提供互补信息。** 问题出在特征本身，不在融合方式。

**局限：**
- 噪声组只有 4 个 epoch，真实特征组有 14 个，对照不完整。后 10 个 epoch 两组是否仍然相同，没有数据。
- 两组都只有单次运行和单一 seed。
- AUC 的差异都在 0.3 以内，没有方差估计。
- 只有 val 结果，没有 test 结果。
- 噪声向量的种子是 `abs(hash(img_dir))`。Python 的字符串哈希每个进程随机化（除非设置 PYTHONHASHSEED），所以噪声在一次运行内对每张图固定，但重新运行会得到不同的噪声，无法逐位复现。

## 下一步
不再继续往 ASAP 里融合这个 OmniFD checkpoint 的特征。按 2026-10-07 的 failure 分析，改为在 DGM4 上直接微调逐脸分类器，对候选框逐个打分。

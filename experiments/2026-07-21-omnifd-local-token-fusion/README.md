# 实验：OmniFD 局部 token 交叉注意力融合

- **日期**：2026-07-21 ~ 2026-07-22（训练），代码于 2026-09-08 提交
- **负责人**：wanhongz
- **代码版本**：[ASAP](https://github.com/karroyyy22-png/ASAP) main 分支 commit `484a230`（experiment: OmniFD fusion via local-token cross-attention with valid mask）。训练在提交之前完成，提交的就是这次实验用的代码。
- **关联日志/checkpoint目录**（服务器 irielab-Super-Server）：
  - 训练日志和 checkpoint：`/nas/data_2/wanhongz/ASAP/results/loglocal_token_fusion_exp01/`
  - test 评估：同目录下的 `evaluation/`（`results_all.txt`、`results_per_category_orig.json`、`raw_eval_data.npz`）
  - 初始化 checkpoint：`/nas/data_2/wanhongz/ASAP/results/log20260602_134300_train_4gpu_batch16/checkpoint_best.pth`（baseline，见实验 2026-06-02）
  - OmniFD 局部特征：`/nas/data_2/wanhongz/ASAP/omnifd_{train,val,test}_feats_local.pt`、`omnifd_*_hasface_local.pt`（几十 GB，不进仓库）

## 目的
实验 2026-06-06 已经说明，分数级融合 OmniFD 是有害的。这次换成特征级融合：把 OmniFD 的局部人脸 token 作为额外的 key/value 送进 ASAP 的 aggregator 交叉注意力，看能否让 face_swap / face_attribute 的检测超过 baseline。

## 方法
每张图取 OmniFD 的 64 个局部 token（512 维），用 `valid_mask` 标记无效位置（例如图里没有脸）。投影到 text_width 以后，和 ASAP 原有的局部特征拼接成 key/value，cls token 作为 query 做交叉注意力（`models/ASAP.py`，484a230）：

```python
omni_tokens_proj = self.omni_proj(omni_feat)            # [B, 64, 512] -> [B, 64, text_width]
existing_kv = self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:])
fused_kv = torch.cat([existing_kv, omni_tokens_proj], dim=1)
fused_padding_mask = torch.cat([existing_mask, ~omni_valid_mask], dim=1)
local_feat_aggr_omni = self.aggregator(
    query=self.norm_layer_aggr(cls_tokens_local),
    key=fused_kv, value=fused_kv, key_padding_mask=fused_padding_mask)[0]
vision_feat = local_feat_aggr_omni.squeeze(1)
```

从 baseline checkpoint 开始微调，只解冻 `omni_proj`、`aggregator`、`cls_head`、`itm_head`，单卡训练。

## 参数配置
不另写 config.yaml。实际运行用的是 ASAP 仓库的 `configs/train.yaml`，启动参数如下：

| 参数 | 值 |
|---|---|
| 开关 | `--use_omni_fusion_local` |
| 优化器 | adamW，lr 1.25e-6，lr_img 6.25e-6，weight_decay 0.02 |
| 调度 | cosine，50 epoch（实际跑了 22 个，epoch 0–21），warmup 10 epoch |
| batch | train 16 / val 64，单卡（world_size=1） |
| 解冻参数 | omni_proj, aggregator, cls_head, itm_head |

## 结果
- **val AUC**：epoch 0 为 91.50，epoch 5 达到最高 91.6455，epoch 21 为 91.42。同一个 baseline checkpoint 的 val AUC 是 91.6076。
- **test 评估用的是 checkpoint_best**，也就是 val AUC 最高的 epoch 5，不是第 22 个 epoch。

| test 指标 | baseline | 局部 token | 差值 |
|---|---|---|---|
| AUC | 92.0645 | 92.0933 | +0.03 |
| ACC | 85.58 | 85.62 | +0.04 |
| face_swap 分类别 ACC | 88.32 | 87.92 | −0.40 |
| face_attribute 分类别 ACC | 92.20 | 91.79 | −0.40 |
| text_swap 分类别 ACC | 82.77 | 82.35 | −0.42 |
| text_attribute 分类别 ACC | 84.54 | 84.49 | −0.05 |
| orig 分类别 ACC | 75.88 | 76.61 | +0.73 |
| IoU（官方，全体均值） | 74.43 | 74.26 | −0.17 |

- baseline 的 AUC 取自 2026-10-07 重新导出的结果（见实验 2026-10-07）；7/21 原始评估是 92.0633。
- 各类别 ACC 的 baseline 也都用 2026-10-07 的导出（阈值 0.5）。face_attribute 的精确差值是 −0.4035。如果 baseline 改用 7/21 原始评估的 `results_per_category_orig.json`，face_swap 和 face_attribute 的差值是 −0.38 和 −0.39；如果按 7/21 报告的 F1 口径（单一类子集上的二分类 F1），差值是 −0.22 和 −0.21。
- 在这个模型上再叠加分数级融合（alpha=0.5），AUC 降到 80.62，和实验 2026-06-06 的结论一致。

## 分析与结论
- **阴性结果**：AUC 只提高了 0.03，在单次运行的波动范围内。目标类别 face_swap 和 face_attribute 的 ACC 反而下降约 0.4，没有恢复到 baseline。
- **orig 的 ACC 上升、伪造类下降**，说明模型整体略偏向判真，而不是更好地利用了人脸伪造痕迹。
- **val AUC 在 epoch 5 以后没有继续上升**，加入 OmniFD 局部 token 并没有带来可学习的增益。

## 下一步
改成更容易诊断的残差加和方案，用一个可学习的 alpha 直接观察模型是否在使用 OmniFD 特征，见实验 2026-09-12。

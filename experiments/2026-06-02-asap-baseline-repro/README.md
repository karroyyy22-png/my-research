# 实验：ASAP baseline 复现

- **日期**：2026-06-02
- **负责人**：wanhongz
- **关联日志目录**：`results/log20260602_134300_train_4gpu_batch16/`

## 目的
在 DGM4 数据集上复现 ASAP 论文的检测结果，验证自己的训练流程（数据准备、训练脚本、评测脚本）是否正确，为后续改进实验提供可信的对比基准（baseline）。

## 方法

**数据准备（LMA模块辅助文本生成）：**
1. 用 BLIP 模型为 DGM4 训练集全部 208184 张图片生成英文图像描述（caption），存为 `caption_train.json`
2. 用 LLM 根据 prompt「Refer to the following text to describe the specific information of the corresponding image」为文本生成补充说明。最初尝试 Qwen2.5-7B，速度约 3 秒/条，两张 GPU 跑完 141189 条预计需 60 小时，耗时过长，改用 **Qwen2.5-1.5B-Instruct**，约 3.5 小时跑完，存为 `explanation_train.json`

**训练：**
- ALBEF 预训练权重的 `queue_size=65536`，设置 `batch_size=16`（单卡），4 张卡总 batch_size=64，`65536 ÷ 64 = 1024` 整除
- 训练 50 个 epoch，耗时约 65 小时
- 每个 epoch 结束后在 val.json 上验证，保存验证集最优权重为 `checkpoint_best.pth`

**测试：**
- 用 `checkpoint_best.pth` 在 test.json 上跑 `test.py`，输出 AUC、ACC、mAP、IoU 等指标
- 之所以用 test 集而非 val 集与论文对比：val 集参与了模型选择（挑最优 checkpoint），存在"信息泄露"，指标会偏乐观；论文报告的都是 test 集结果，必须同一测试集才能公平对比

## 参数配置
| 配置项 | 值 |
|---|---|
| 服务器 | GPU200 |
| GPU | 4× RTX 6000 Ada（每张 49GB） |
| batch_size（单卡） | 16 |
| 总 batch_size | 64 |
| epochs | 50 |
| 优化器 | AdamW, lr=2e-5, lr_img=1e-4 |
| 预训练权重 | ALBEF_4M.pth |
| 数据集 | DGM4（训练集 208184 条） |
| 辅助文本 | BLIP 生成 caption + Qwen2.5-1.5B 生成 explanation |
| VLM(caption) | BLIP |

## 结果（test 集，val 最优 checkpoint）
| 指标 | 数值 |
|---|---|
| AUC_cls | 92.0645 |
| ACC_cls | 85.5813 |
| EER_cls | 15.2975 |
| MAP | 84.1607 |
| OF1 | 78.8387 |
| CF1 | 78.2864 |
| IOU_score | 74.4318 |
| IOU_ACC_50 | 81.6862 |
| IOU_ACC_75 | 74.0400 |
| ACC_tok | 91.3561 |
| F1_tok | 70.3940 |

完整指标见 `results/results_all.txt`；分类别指标见 `results/results_per_category.json`。

## 分析与结论
训练流程跑通，指标与论文 Table 1 处于同一量级，说明复现的训练/评测 pipeline 是正确的，可以作为后续所有改进实验（TamperDetector、学习率调整、OmniFD融合等）的统一对照基准（记为「实验 A」）。

## 下一步
- 尝试加入篡改检测辅助模块（TamperDetector），见实验 `2026-06-09-lr-batchsize-tamperdetector-ablation`
- 尝试用论文原始的学习率/batch_size 线性缩放规则重新训练，对比影响

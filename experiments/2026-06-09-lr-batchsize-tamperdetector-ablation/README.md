# 实验：学习率/batch_size 缩放 + TamperDetector 消融实验

- **日期**：2026-06-06 ~ 2026-06-13
- **负责人**：wanhongz
- **关联日志目录**：
  - A: `results/log20260602_134300_train_4gpu_batch16/`（即 baseline，见实验 2026-06-02）
  - B: `results/log20260606_215506_train_4gpu_batch16/`
  - C: `results/log20260609_232652_train_4gpu_batch16/`
  - D: `results/log20260613_125731_train_4gpu_batch16/`

## 目的
在 baseline（实验A）基础上，做两组独立的消融对比，分析三个因素对 ASAP 检测效果的真实影响：
1. **学习率/batch_size 缩放**：是否应该按论文原始设置（8卡×batch128=1024）对 lr 做线性缩放
2. **VLM(caption) 选择**：BLIP 与论文原始使用的 VisCPM 差异有多大
3. **TamperDetector 模块**：新增的动态篡改检测融合模块是否带来提升

## 方法

**TamperDetector 模块设计（实验B）：**
基于 EfficientNet-B0 新增子模块，与 ASAP 联合端到端训练，根据输入图像动态输出篡改置信度 `s`，动态调节 text_feat 与 vision_feat 的融合权重：

```python
class TamperDetector(nn.Module):
    def __init__(self):
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.fc = nn.Linear(1280, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, image):
        feat = self.backbone(image)
        s = torch.sigmoid(self.fc(feat))  # s ∈ (0,1)
        return s
```

原论文融合方式（固定权重）：
```python
output_cls = self.cls_head(text_feat + self.UtilsB * vision_feat)
```

改进后融合方式（动态权重）：
```python
s = self.tamper_detector(image)
final_feat = (1 - s) * text_feat + s * vision_feat
output_cls = self.cls_head(final_feat)
```
设计动机：DGM4 数据集包含两类主要篡改——脸部强篡改（face_swap/infoswap，视觉异常明显，依赖视觉特征）和语义不一致（文本图像不符，屏幕正常但文本错误，依赖文本-图像一致性）。固定权重融合无法区分这两种情况，动态权重理论上可以让模型按需侧重不同分支。

**lr 缩放实验（实验C/D）：**
发现 baseline 用的 batch_size(64) 远小于论文原始设置（8卡×128=1024），按线性缩放规则重新计算：
```
lr(C/D) = 2e-5 × (64/1024) = 1.25e-6
lr_img(C/D) = 1e-4 × (64/1024) = 6.25e-6
```

**VisCPM 对比（实验D）：**
在实验C的基础上，只把 caption 生成模型从 BLIP 换成论文原始使用的 VisCPM-Chat，其余配置不变。

## 参数配置
| 实验 | 模型 | GPU数 | batch/GPU | 总batch | lr | lr_img | epochs | VLM(caption) |
|---|---|---|---|---|---|---|---|---|
| A（baseline） | 原始ASAP | 4 | 16 | 64 | 2e-5 | 1e-4 | 50 | BLIP |
| B | 原始ASAP + TamperDetector | 4 | 16 | 64 | 2e-5 | 1e-4 | 50 | BLIP |
| C | 原始ASAP | 4 | 16 | 64 | 1.25e-6 | 6.25e-6 | 50 | BLIP |
| D | 原始ASAP | 4 | 16 | 64 | 1.25e-6 | 6.25e-6 | 50 | VisCPM |

完整配置见 `config.yaml`。

## 结果

**全体指标对比（test集）：**
| 指标 | A（baseline） | B（+Tamper） | C（缩放lr） | D（+VisCPM） |
|---|---|---|---|---|
| AUC_cls | 92.0645 | 91.4585 | 92.5712 | 92.5482 |
| ACC_cls | 85.5813 | 85.1494 | 85.5793 | 85.4571 |
| OF1 | 78.8387 | 78.0678 | 78.0410 | 77.9438 |
| CF1 | 78.2864 | 77.4801 | 76.7464 | 76.5400 |
| IOU_score | 74.4318 | 73.4312 | **64.3407** | 63.9043 |
| IOU_ACC_50 | 81.6862 | 80.7258 | **70.7386** | 70.1765 |
| IOU_ACC_75 | 74.0400 | 72.3814 | **50.1588** | 49.3581 |

完整全体指标和分类别指标（face_swap / face_attribute / text_swap 等 9 个类别的 IoU、F1、ACC）见 `results/`。原始 Excel 汇总表见 `results/ASAP_実験結果比較.xlsx`。

**关键发现（按类别IoU@50下降幅度）：**
- 人脸相关类别下降幅度大（约15-20个百分点）：face_swap 79.82%→61.76%，face_attribute 84.90%→67.28%
- 纯文本类别下降幅度小（约3-4个百分点）：text_swap 80.11%→76.23%，text_attribute 80.49%→77.50%

## 分析与结论

**1. lr / batch_size 影响最大（A vs C）：**
- AUC_cls 提升 +0.51%，分类性能小幅改善
- 但 IOU_score 大幅下降 -10.09%，IOU_ACC_75 更是暴跌 -23.88 个百分点，bbox 定位能力显著退化
- 训练曲线上，大 lr 模型在 epoch 1 时 IoU@50 就从 9% 冲到 59%；小 lr 模型 epoch 4 都还不到 40%，说明 bbox 回归对学习率极度敏感
- **推测原因**：bbox 回归损失（L1+GIoU）的梯度尺度天然大于分类交叉熵，需要更大的学习率才能有效优化；将线性缩放规则无差别应用到所有 loss（包括分类和回归）是不合理的

**2. VLM(caption) 选择影响很小（C vs D）：**
BLIP 与 VisCPM 在所有指标上差异都在 ±0.6% 以内，说明 caption 生成模型的选择对最终检测性能影响可以忽略，不是复现效果差异的主因

**3. TamperDetector 未带来提升，反而小幅下降（A vs B）：**
AUC_cls -0.60%，OF1 -0.77%，IOU_score -1.00%，只有 text_swap 的 F1 有 +0.44% 的小幅改善。**判定为负面结果**：新增模块未能提升整体性能

**综合结论：**性能差异的主因是 lr/batch_size 设置，而非 VLM 选择。bbox 回归任务对学习率极度敏感，线性缩放规则不能一律套用到所有损失项。

## 下一步
1. 尝试为 bbox 回归头（loss_bbox、loss_giou）单独设置更大的学习率，不与分类损失共用同一套缩放规则
2. 缩短 warmup epoch，让 bbox 回归更快进入正常收敛区间
3. 后续不再依赖 VLM 的隐式跨模态语义对齐去做纯物理层面的换脸定位，改为引入显式的微观物理痕迹提取器 —— 见实验 `2026-06-06-omnifd-asap-fusion`

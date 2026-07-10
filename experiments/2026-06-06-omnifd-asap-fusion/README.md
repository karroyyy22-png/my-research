# 实验：OmniFD 人脸特征提取 + 与 ASAP 融合

- **日期**：2026-06-06
- **负责人**：wanhongz
- **关联checkpoint**：`results/log20260606_215506_train_4gpu_batch16/checkpoint_best.pth`
- **代码版本**：
  - [ASAP](https://github.com/karroyyy22-png/ASAP)（fork自 [CriliasMiller/ASAP](https://github.com/CriliasMiller/ASAP)），commit `5a59424`
  - [OmniFD](https://github.com/karroyyy22-png/OmniFD)（fork自 [haotianll/OmniFD](https://github.com/haotianll/OmniFD)），commit `7401d1f`（含人脸特征提取脚本 `extract_face_features.py` 等）

## 目的
上一组消融实验（见 `2026-06-09-lr-batchsize-tamperdetector-ablation`）发现，ASAP 依赖 VLM 隐式跨模态语义对齐来做换脸定位，鲁棒性很脆弱（lr 缩小后 face_swap 的 IoU@50 从 79.82% 暴跌到 61.76%）。因此本实验尝试引入显式的微观物理痕迹提取器 **OmniFD**（人脸伪造检测模型），探索能否用它的判别信息提升 ASAP 在人脸篡改类别上的检测效果。

OmniFD 采用统一框架，直接以原始图像/视频为输入，端到端完成分类、空间定位、时间定位任务。

## 方法

### 第一步：OmniFD 特征提取
先只跑 OmniFD 的人脸定位部分，验证它对 DGM4 数据（face_swap / face_attribute 子集）本身是否有区分度，再决定是否接入 ASAP。

- **第一次尝试**：直接输入整张新闻大图（未裁剪人脸），real/fake 分数区分度极差（仅 0.02）
- **问题定位**：发现 `__getitem__` 没有使用 `mtcnn_boxes`，导致模型看到的是整张大图而非人脸区域
- **修复**：根据 `mtcnn_boxes` 裁剪出人脸区域再输入模型：
  ```python
  boxes = item.get('mtcnn_boxes', [])
  if boxes:
      x1, y1, x2, y2 = boxes[0]
      img = img.crop((x1, y1, x2, y2))
  ```
- **数据过滤**：只保留包含人脸修改的图像（orig / face_swap / face_attribute），排除纯文本篡改图像。裁剪后共 43,780 条样本（真实 24,344 条，face_swap 10,515 条，face_attribute 8,921 条；约 6,925 张图像因未检测到人脸被排除）
- **重新提取特征**：`local_feat: (43780, 768)`，`global_feat: (43780, 512)`，`score`（伪造概率）

**特征提取结果验证：**
- real/fake 区分度从 0.02 提升到 0.17（约 8 倍改善）
- 但 face_swap 和 face_attribute 两个子类之间无法区分（分数都在 0.485-0.486 附近）
- 全局 ROC-AUC：**0.6656**；平均精度 AP：**0.7643**
- face_swap vs face_attribute 的 Mann-Whitney U 显著性检验：P = 0.9366 ≥ 0.05，两组分数分布无统计学显著差异

特征、标签、分析结果保存在 `/nas/data_2/wanhongz/face_cropped_features/`（local_feat.npy / global_feat.npy / score.npy / labels.npy / meta.json / analysis.json），ROC 曲线、PR 曲线、分数直方图见同目录 `plots_output/`。

### 第二步：接入 ASAP —— 两种融合方案

**排除的方案**：在模型内部结构层面融合。因为 ASAP 的 `cls_head` 是 4 个类别共享的同一个 MLP，无法只给人脸相关通道加信息而不影响文本通道，需要改模型结构重新训练，风险和工作量都大，未采用。

**实验一：分数级融合**
在 `test.py` 里改造：
1. ASAP 对测试集正常推理一遍，得到自己的判断（`logits_real_fake` → fake 概率）
2. 对每张图检查：①标签是否属于 face_swap/face_attribute（含混合类别）；②能否在 OmniFD 分数表里查到分数（即是否检测到人脸）
3. 两条件都满足则按 `0.5×ASAP + 0.5×OmniFD` 加权融合；不满足则完全保留 ASAP 原值
4. 重新计算 AUC_cls / ACC_cls / F1_FS / F1_FA 等指标，输出融合前后对比

**实验二：特征级融合**
不再简单加权分数，而是将 OmniFD 的 512 维 `global_feat` 通过投影层对齐维度，与 ASAP 的视觉特征拼接后压缩，替换原始视觉特征喂给下游分类头；仅解冻新增融合层+分类头做小规模微调（5 个 epoch），其余 ASAP 参数冻结。

融合架构：
```
ASAP路： 完整图像 (256x256) → 特征768维
OmniFD路： 人脸区域 (224x224) → 特征512维
融合： 768 + 512 → 分类
```

**对应代码**（`models/ASAP.py`，commit `5a59424`，`ASAP.__init__` 内）：
```python
self.omni_proj = nn.Linear(512, text_width)                  # OmniFD 512维特征投影对齐维度
self.omni_fusion_fc = nn.Linear(text_width * 2, text_width)  # 拼接后压缩回原维度
```
`forward()` 接口也相应加了 `omni_feat` 参数用于传入 OmniFD 特征：
```python
def forward(self, image, label, text, fake_image_box, fake_text_pos, cap_input, prom_input,
            res_fake_pos, res_fake_pos_patch, alpha=0, is_train=True, omni_feat=None):
```
实际融合发生在 `forward()` 内部（训练分支和推理分支各出现一次，逻辑相同）：
```python
if omni_feat is not None:
    omni_feat_proj = self.omni_proj(omni_feat)
    vision_feat = self.omni_fusion_fc(torch.cat([vision_feat, omni_feat_proj], dim=-1))
```
即：OmniFD 的 512 维特征投影到 text_width 维，与原始 `vision_feat` 拼接后再压缩回 text_width 维，**直接替换**原来的 `vision_feat`，后续分类头等下游逻辑不变。`omni_feat` 为 `None` 时（即没有对应人脸的样本）该逻辑跳过，等价于原始 ASAP。

> 注：同一份代码里还留有一段被注释掉的 `TamperDetector` 类定义和调用（对应实验 `2026-06-09-lr-batchsize-tamperdetector-ablation` 里的实验B），是切换实验思路时的中间状态，未删除，如需恢复可直接取消注释。

## 参数配置

| 项目 | 值 |
|---|---|
| ASAP 代码位置 | /nas/data_2/wanhongz/ASAP |
| ASAP checkpoint | results/log20260606_215506_train_4gpu_batch16/checkpoint_best.pth |
| ASAP 特征维度 | 768（vision_width） |
| ASAP 输入尺寸 | 256×256 |
| OmniFD 代码位置 | /nas/data_2/wanhongz/OmniFD |
| OmniFD 权重 | omnifd_swin-s_50k_forgerynet.pth（MMPretrain, Swin-S backbone） |
| OmniFD 特征维度 | 512 |
| OmniFD 输入尺寸 | 224×224 |
| 数据集 | DGM4（/nas/data_2/wanhongz/datasets/DGM4/） |
| Python | 3.13 |
| PyTorch | 2.7.1 |
| GPU | 4× RTX 6000 Ada |

## 结果（负面结论，如实记录）

**实验一：分数级融合（test集，26904/50705 样本参与融合）**
| 指标 | 融合前 | 融合后 | 变化 |
|---|---|---|---|
| AUC_cls | 91.46% | 80.61% | **-10.84%** |
| ACC_cls | 85.15% | 85.04% | -0.11% |
| F1_FS | 92.72% | 92.64% | -0.08% |
| F1_FA | 95.51% | 95.33% | -0.18% |

**实验二：特征级融合（test集，同一起点checkpoint，严格同口径对比）**
| 指标 | 融合前 | 融合后 | 变化 |
|---|---|---|---|
| AUC_cls | 91.46% | 91.16% | -0.30% |
| ACC_cls | 85.21% | 85.06% | -0.15% |

Bbox 定位、多标签分类等不受此次融合影响的指标保持不变。

**诊断分析**：在 ASAP 判断错误、且能查到 OmniFD 分数的 2,346 个样本中，OmniFD 判断正确的比例仅 **46.16%**（低于随机猜测的 50%）。

## 分析与结论

两种融合方式（分数级、特征级）均未能提升 ASAP 的检测性能：

- **分数级融合明显有害**：OmniFD 自身判别力（AUC=0.6656）远弱于 ASAP（91.46%），简单加权会拖累整体表现
- **特征级融合负面影响小很多**：说明模型在训练中学会了不过度依赖新特征（未造成灾难性破坏），但依然没有学到有效的正向增益
- **根本原因**：在 ASAP 判断错误的样本上，OmniFD 同样大概率判断错误（正确率仅46.16%，低于随机水平），说明两个模型的错误高度重合，OmniFD 未能提供真正互补的判别信息

综合两次实验的证据，指向同一个结论：**OmniFD 与 ASAP 在 DGM4 任务上捕捉到的判别信息高度重合，无论在分数层面还是特征层面融合，均无法提升整体检测效果。**

这是一个方法执行正确、但验证结果为阴性的实验。完整逐样本数据已保存（`raw_eval_data.npz`），供后续需要时复查或尝试其他融合策略。

## 下一步
- 探索其他跟 ASAP 判别信息互补性更强的辅助模型，而不是同样基于深度网络隐式特征的方案
- 如果继续用 OmniFD，需要先分析它和 ASAP 犯错样本的具体差异，找到真正不重合的判别维度，而不是直接做分数/特征融合

# 研究项目名称

## 研究目标
简要描述这个研究项目要解决的问题、核心假设。

## 目录结构
```
├── experiments/   # 每次实验一个子目录，记录方法、参数、结果、结论
├── data/          # 数据（大文件已在 .gitignore 中排除）
├── src/           # 可复用代码（模型、工具函数等）
├── notebooks/     # 探索性分析、Jupyter notebook
└── docs/          # 长篇笔记、文献综述、阶段性总结
```

## 实验索引
按时间顺序记录所有实验，方便快速回顾。

| 日期 | 实验目录 | 目的 | 关键结果 |
|------|---------|------|---------|
| 2026-06-02 | [asap-baseline-repro](experiments/2026-06-02-asap-baseline-repro/) | 复现ASAP在DGM4上的检测效果 | AUC=92.06%，作为后续baseline |
| 2026-06-06 | [omnifd-asap-fusion](experiments/2026-06-06-omnifd-asap-fusion/) | 探索OmniFD人脸特征能否提升ASAP | 阴性结果，分数级融合AUC反降10.84% |
| 2026-06-09 | [lr-batchsize-tamperdetector-ablation](experiments/2026-06-09-lr-batchsize-tamperdetector-ablation/) | lr缩放/VLM选择/TamperDetector消融 | lr缩放显著损害bbox定位(IoU -10~24%) |
| 2026-07-21 | [omnifd-local-token-fusion](experiments/2026-07-21-omnifd-local-token-fusion/) | OmniFD局部token作为aggregator的额外key/value融合 | 阴性：test AUC +0.03，face_swap/face_attribute ACC 各−0.40 |
| 2026-09-12 | [omnifd-residual-diagnosis](experiments/2026-09-12-omnifd-residual-diagnosis/) | 可学习alpha残差加和+随机噪声对照，诊断OmniFD特征是否有用 | alpha 0.91→1.78但val AUC −0.04→−0.30；噪声组AUC几乎相同 |
| 2026-09-13 | [omnifd-boxselect-eval](experiments/2026-09-13-omnifd-boxselect-eval/) | 用OmniFD分数在候选人脸中挑篡改脸 | 严格选对率54.81%，低于“总选第一个框”65.85%（ASAP 84.52%） |
| 2026-09-13 | [localization-task-nature](experiments/2026-09-13-localization-task-nature/) | 分析DGM4图像定位任务的性质 | gt 100%等于某个mtcnn框，每图2–5张脸，定位实为候选选择 |
| 2026-10-07 | [asap-failure-analysis](experiments/2026-10-07-asap-failure-analysis/) | 分析ASAP在DGM4 test上的错误分布，决定后续路线 | orig误报占错误56%、纯图像漏检26%，物理侧约6成，先走物理路线 |
| 2026-10-13 | [phase0-iou-eval-split](experiments/2026-10-13-phase0-iou-eval-split/) | 检验CSCL高IoU是否来自框后处理，并把官方IoU拆成篡改/未篡改两部分 | 后处理只±1.5；9.66分差距中未篡改侧5.88(61%)、篡改侧定位3.78(39%)，可进入阶段1 |

## 实验记录整理规范
新实验记录（不管是自己写还是用LLM整理）请遵循 [EXPERIMENT_LOGGING_GUIDE.md](EXPERIMENT_LOGGING_GUIDE.md) 里的格式规范。

## 关联代码仓库
本仓库只存实验记录（方法、参数、结果、结论），不包含大型代码库本身。实际代码单独仓库管理，各实验 README 会注明使用的精确 commit：
- [ASAP](https://github.com/karroyyy22-png/ASAP)（fork自 [CriliasMiller/ASAP](https://github.com/CriliasMiller/ASAP)）
- [OmniFD](https://github.com/karroyyy22-png/OmniFD)（fork自 [haotianll/OmniFD](https://github.com/haotianll/OmniFD)）

## 环境依赖
```bash
pip install -r requirements.txt
```

## 使用方法
1. 新建实验：复制 `experiments/_template/` 为新目录，命名格式 `YYYY-MM-DD-实验简述`
2. 填写该实验的 README.md
3. 跑完后把结果（图表/日志）放进对应的 `results/`
4. commit，message 格式建议：`exp: <简述> (关键指标)`
5. 更新本文件的"实验索引"表格

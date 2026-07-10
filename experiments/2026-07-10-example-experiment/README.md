# 实验：baseline 模型（示例）

- **日期**：2026-07-10
- **负责人**：
- **关联 commit / tag**：v0.1-baseline

## 目的
验证最简单的baseline方法在数据集上的表现，作为后续改进的对照基准。

## 方法
使用逻辑回归模型，不做任何特征工程，直接在原始特征上训练。

## 参数配置
见 `config.yaml`：
- learning_rate: 0.001
- batch_size: 32
- epochs: 50

## 结果
- 关键指标：accuracy = 0.82
- 图表/日志见 `results/`

## 分析与结论
baseline 效果符合预期，作为后续对比基准。观察到验证集loss在epoch 30后略有上升，可能存在轻微过拟合。

## 下一步
尝试加入正则化（L2 / dropout），并做特征标准化。

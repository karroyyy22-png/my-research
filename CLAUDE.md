# CLAUDE.md

先读 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)（项目总入口），再按其中“阅读顺序”读其他文件。

## 规则
- 冻结假设文件（`frozen_hypotheses*.json`、`frozen_config.json`）及其 sha256 在看过结果后不得修改。
- test 只用于确认，不用来选规则；规则与阈值只在 val（val_dev / val_holdout）上定。
- 分析一律用 CPU：`nice` 运行，最多 4 个 worker，`OMP_NUM_THREADS=4`。
- 需要 GPU 时，只能使用完全空闲、没有任何他人进程的 GPU；没有空闲 GPU 就停止。运行前后都要用 `nvidia-smi` 检查。
- 不要修改已有实验目录里的文件（结果、脚本、冻结文件、README）；新内容新建目录或文件。
- 没被要求不要 push。不要 force push、不要改写历史。
- 提交说明格式：`exp: <摘要>`。
- 文档里的每个数字都写出来源文件；没有记录结论的实验写“结论未在仓库中记录”。
- 每完成一步汇报：做了什么、为什么、一句话结果、下一步、整体位置（在 A/B/C/D 阶段中的位置）。

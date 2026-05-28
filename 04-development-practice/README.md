# 04 Development Practice

本部分专门讲具体开发相关知识，放在 vLLM Ascend 基础之后、关键特性之前。这部分先理解系统如何工作，再学习如何改、如何测、如何观察性能、如何排障，最后进入关键特性。

## 文档清单

- [00-development-workflow.md](00-development-workflow.md)：开发流程、改动分层、patch/upstream 判断。
- [01-testing-ci-and-nightly.md](01-testing-ci-and-nightly.md)：测试、CI、nightly workflow。
- [02-observability-and-performance-tuning.md](02-observability-and-performance-tuning.md)：性能观测、benchmark、profile、调优。
- [03-debugging-playbook.md](03-debugging-playbook.md)：故障排查手册。

## 本阶段目标

- 能根据改动范围选择单测、e2e、nightly 或 benchmark。
- 能读懂 vLLM Ascend CI workflow 中关键变量和执行路径。
- 能建立一次性能实验的 baseline、指标和结论。
- 能按固定路径排查启动、模型加载、HCCL、OOM、精度和性能问题。


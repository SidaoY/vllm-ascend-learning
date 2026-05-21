# 实验记录模板

这份模板适合性能实验、精度对齐、CI/nightly 失败复盘和特性验证。可以按需删减，但建议保留环境、workload、指标和结论四部分，否则后续很难复现。

## 背景

- 目标：
- 相关 issue/PR：
- 相关代码：
- 实验负责人：
- 日期：

## 环境

- vLLM commit：
- vLLM Ascend commit：
- CANN：
- torch / torch-npu：
- Python：
- 镜像：
- 硬件和卡数：
- SOC_VERSION：
- 模型：
- checkpoint / revision：
- 启动参数：
- 关键环境变量：
- 是否使用 graph：
- 并行策略：
- 是否开启 prefix cache / spec decode / CP / PD：

## Workload

- 数据集：
- 输入长度分布：
- 输出长度分布：
- 并发：
- request rate：
- 请求数：
- warmup：
- sampling 参数：
- 随机种子：
- 是否流式输出：

## 指标

- TTFT：
- TPOT：
- ITL：
- end-to-end latency：
- request throughput：
- token throughput：
- goodput：
- NPU memory：
- KV cache utilization：
- prefix cache hit rate：
- spec decode acceptance rate：
- 通信耗时：
- 其他：

## 结果

- baseline：
- 实验组：
- 差异：
- 初步结论：
- 是否达到目标：
- 是否存在回归：

## Profile 和日志

- profiling 工具：
- 关键时间线：
- 最耗时算子或阶段：
- 关键日志：
- 异常栈：

## 问题与后续

- 异常现象：
- 已排除原因：
- 仍需确认：
- 下一步：
- 需要补充的测试：
- 需要更新的文档：

## 复盘结论

- 根因：
- 修复方式：
- 风险：
- 后续观察指标：

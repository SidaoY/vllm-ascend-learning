# 03 vLLM Ascend Foundation

本部分解释 vLLM Ascend 如何在上游 vLLM 之上工作。核心思路是：尽量复用 vLLM 的入口层、engine、scheduler、model executor 等通用能力，在 platform、worker/model runner、attention backend、ops、distributed、patch 等位置提供 Ascend NPU 后端适配。

读这一章时要带着上一章的架构图来看：vLLM Ascend 不是把 vLLM 重新实现一遍，而是在关键边界上接入 NPU 能力。

## 文档清单

1. [00-architecture-overview.md](00-architecture-overview.md)：整体架构和与 vLLM 的关系。
2. [01-platform-registration.md](01-platform-registration.md)：NPU platform、环境变量、能力声明。
3. [02-patch-mechanism.md](02-patch-mechanism.md)：platform patch、worker patch、patch 生命周期。
4. [03-worker-and-model-runner.md](03-worker-and-model-runner.md)：NPU worker/model runner、input batch、block table。
5. [04-attention-and-kernels.md](04-attention-and-kernels.md)：Ascend attention、custom ops、Triton/CANN 路径。
6. [05-distributed-and-kv-transfer.md](05-distributed-and-kv-transfer.md)：分布式、HCCL、KV transfer、PD 分离。

## 本阶段目标

- 能解释 vLLM Ascend 如何被 vLLM 发现和使用。
- 能区分平台适配、执行适配、算子适配、分布式适配和 patch。
- 能判断一个问题更可能发生在 platform、worker、attention、ops、distributed 还是 patch 层。
- 能找到 NPU 执行路径中 model runner、attention、block table、KV transfer 的目录级入口。
- 能理解后续开发实践章节中测试、性能观测和故障排查的代码背景。

## 阅读建议

先读 `00` 和 `01`，理解 vLLM Ascend 的接入方式；再读 `02`，理解为什么会有 patch；之后读 `03` 和 `04`，进入设备侧执行和 attention；最后读 `05`，理解多卡、多机和 KV transfer。

本章仍然遵循一个原则：本地代码用于校准当前实现，但文档不绑定具体行号，也不过度依赖某个方法名。优先理解职责边界和数据流。

# Platform 注册与环境适配

Platform 层负责让 vLLM 认识 Ascend NPU。它回答的问题是：当前设备是什么、支持哪些能力、默认配置如何修正、启动前需要应用哪些全局适配。

## NPUPlatform 的职责

NPU platform 通常负责：

- 声明设备类型、设备名称和设备能力。
- 处理 dtype、attention backend、graph、pin memory 等平台能力。
- 读取和规范化 Ascend 相关环境变量。
- 根据硬件形态选择特化路径。
- 注册 custom op 或 lazy 初始化底层算子。
- 在 worker 启动前应用 platform-level patch。
- 调整部分 vLLM config，使其符合 NPU 约束。

Platform 层尽量不承载具体模型执行逻辑。模型执行、attention、KV cache、sampling 等应放在 worker/model runner 或 backend 层。

## 启动阶段的作用

```mermaid
flowchart LR
    A[Create vLLM config] --> B[Select platform]
    B --> C[NPU platform updates config]
    C --> D[Apply platform patches]
    D --> E[Create engine / executor]
    E --> F[Create NPU worker]
```

这意味着 platform patch 的时机早于 worker 初始化。凡是影响 config 校验、模型注册、全局分布式行为、上游类定义或 schema 的改动，通常需要在 platform 阶段完成。

## 环境变量和附加配置

vLLM Ascend 会通过环境变量和 additional config 控制很多行为，例如：

- 是否启用某些 graph 或 compile 路径。
- 是否启用特定 attention、MoE、通信或 KV 功能。
- HCCL、CANN、torch-npu 相关运行时设置。
- profiling、debug、性能调优开关。
- 特定模型或硬件的兼容配置。

新人不需要一次记住所有变量。更好的方式是按问题类型查：

- 启动失败：查 installation、env vars、CANN/torch-npu/HCCL。
- graph 问题：查 graph mode / ACL graph / npugraph_ex。
- attention 问题：查 feature guide 和 attention backend。
- 性能问题：查 performance and debug 文档。
- KV transfer 问题：查 KV pool、PD disaggregation、connector 文档。

## 硬件差异

Ascend 不同硬件形态可能有不同算子支持、内存布局、block size、通信能力和性能建议。Platform 层会做一部分硬件判断，但业务代码也可能有硬件特化路径。

阅读代码时遇到硬件分支，不要只问“为什么这里 if/else 很多”，而要先确认：

- 这个硬件是否支持对应算子。
- dtype 和 layout 是否一致。
- graph capture 是否支持。
- 通信能力是否不同。
- 测试覆盖是否按硬件拆分。

## 与 torch-npu / CANN / HCCL 的关系

- torch-npu：PyTorch 到 Ascend NPU 的运行时和算子接口。
- CANN：Ascend 底层计算框架和算子生态。
- HCCL：Ascend 集合通信库，对应多卡、多机通信。
- vLLM Ascend：在 vLLM 框架中组织这些能力，并适配推理服务的调度、KV cache、attention、graph 和分布式需求。

排查问题时要分清层次：Python 配置错误、vLLM Ascend 适配错误、torch-npu 行为、CANN 算子问题、HCCL 通信问题，现象可能相似，但定位路径不同。

## 代码入口

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/platform.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/envs.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ascend_config.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/utils.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/_310p`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/configuration/env_vars.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/configuration/additional_config.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/support_matrix`

建议搜索关键词：`NPUPlatform`、`envs`、`additional_config`、`attention backend`、`graph`、`HCCL`、`CANN`。

## 新增平台能力时的检查点

- 是否需要新增环境变量或 additional config。
- 默认值是否会改变现有用户行为。
- 是否只在特定硬件或模型上可用。
- 是否需要 platform patch 早于 worker 生效。
- 是否需要更新支持矩阵和用户文档。
- 是否有最小单测、e2e 或 nightly 覆盖。

## 思考与探索

1. 为什么 config 校验相关 patch 通常要在 platform 阶段生效？
2. 如果一个功能只在某类 NPU 上可用，文档、代码和测试分别应该如何体现？
3. 遇到启动阶段报错时，你会如何区分 platform 配置问题和 worker 执行问题？

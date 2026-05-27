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

| 步骤 | 阶段 | 说明 |
| --- | --- | --- |
| 1 | Create vLLM config | 创建 vLLM 配置 |
| 2 | Select platform | 选择 platform（NPUPlatform） |
| 3 | Apply platform patches | 应用 platform-level patches（`pre_register_and_update`） |
| 4 | NPU platform updates config | NPU platform 调整 config（`check_and_update_config`） |
| 5 | Create engine / executor | 创建 engine / executor |
| 6 | Create NPU worker | 创建 NPU worker |

这意味着 platform patch 的时机早于 config 校验和 worker 初始化。凡是影响 config 校验、模型注册、全局分布式行为、上游类定义或 schema 的改动，通常需要在 platform 阶段完成。

## 环境变量和附加配置

vLLM Ascend 会通过环境变量和 additional config 控制很多行为，例如：

- 是否启用某些 graph 或 compile 路径。
- 是否启用特定 attention、MoE、通信或 KV 功能。
- HCCL、CANN、torch-npu 相关运行时设置。
- profiling、debug、性能调优开关。
- 特定模型或硬件的兼容配置。

不需要一次记住所有变量。更好的方式是按问题类型查：

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

## Platform 与 torch-npu / CANN / HCCL 的关系

Platform 层之所以要讨论 torch-npu、CANN、HCCL，是因为 **platform.py 正是连接 vLLM 与这些底层 Ascend 组件的桥接层**。NPUPlatform 中的大量方法直接调用了这些底层库的能力：

```
NPUPlatform.set_device()            →  torch.npu.set_device()
NPUPlatform.get_device_name()       →  torch.npu.get_device_name()
NPUPlatform.num_compute_units()     →  torch.npu.get_device_properties()
NPUPlatform.get_current_memory()    →  torch.npu.max_memory_allocated()
NPUPlatform.import_kernels()        →  加载 CANN 自定义算子
NPUPlatform.get_device_communicator_cls() →  NPUCommunicator（封装 HCCL）
NPUPlatform.set_additional_forward_context() →  管理 HCCL 通信上下文
```

换句话说，vLLM 框架不知道 Ascend 的底层细节——它只知道通过 `current_platform` 调用方法，由 platform 层把调用翻译成 torch-npu / CANN / HCCL 的 API。三者分工如下：

| 组件 | 职责 | platform.py 的交互方式 |
|---|---|---|
| **torch-npu** | PyTorch 到 Ascend NPU 的运行时适配，提供 `torch.npu.*` API 和设备管理 | `set_device`、`get_device_name`、`num_compute_units`、`get_current_memory_usage` 等直接调用 |
| **CANN** | Ascend 底层计算框架和算子生态（ACLNN、TBE、AscendC 等） | `import_kernels` 加载 custom op、`get_static_graph_wrapper_cls` 返回 ACL Graph wrapper |
| **HCCL** | Ascend 集合通信库，对应多卡多机通信 | `get_device_communicator_cls` 返回封装 HCCL 的 NPUCommunicator、`set_additional_forward_context` 管理通信上下文 |

排查问题时要分清层次：Python 配置错误、vLLM Ascend 的 platform 适配错误、torch-npu API 调用异常、CANN 算子问题、HCCL 通信问题，现象可能相似，但定位路径不同。platform 层是最先被调用的，如果 platform 选择或初始化就错了，后面的 Worker、Attention、Distributed 都不会正常工作。

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

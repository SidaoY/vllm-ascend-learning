# 代码库概览

## 写作目标

把 vLLM 距离"推理框架"最近的目录和概念讲清楚，让读者在读具体章节前知道：代码在哪里、怎么分层、不同目录之间是什么关系。

## vLLM 顶层的"推理相关"目录

```text
vllm/
├── entrypoints/          # API 入口（OpenAI compatible、LLM class）
├── v1/                   # V1 架构（engine、scheduler、worker、KV cache 等核心）
├── attention/            # Attention backend 抽象和通用 metadata
├── model_executor/       # 模型定义、layers、attention 接口
├── distributed/          # 并行状态、通信器、KV transfer
├── compilation/          # torch compile、graph mode
├── transformers_utils/   # 模型加载、配置检测
├── platforms/            # Platform 抽象基类
├── config/               # 各类配置（SchedulerConfig 等）
└── spec_decode/          # Speculative decoding 相关
```

## 分层模型

vLLM 可以大致按以下层次理解：

| 层级 | 组件 | 职责 |
| --- | --- | --- |
| API 层 | Entrypoints | OpenAI compatible API、LLM class 离线推理 |
| 引擎层 | Engine | 管理请求生命周期、串联 scheduler 和 worker、output processing |
| 调度层 | Scheduler | token budget 分配、KV block 分配、prefill/decode 决策、chunked prefill、prefix caching |
| 执行层 | Worker / Model Runner | 将 scheduler output 转为设备输入、管理输入 batch 和 attention metadata、调用 attention backend 和 model |
| 计算层 | Model / Attention / Ops | 模型前向、attention 计算（FlashAttention 等）、自定义算子 |
| 设备层 | Devices / Communication | GPU/NPU 内存管理、集合通信（NCCL/HCCL）、KV transfer |

## 各目录注解

### entrypoints

提供对外 API。当前主要入口有两个：
- OpenAI compatible API（`vllm/entrypoints/openai/api_server.py`）
- `LLM` 类离线推理（`vllm/entrypoints/llm.py`）

`entrypoints` 不关心底层是 GPU 还是 NPU。它只负责：
- 解析请求参数（messages、SamplingParams 等）
- 选择 engine 实现（sync/async）
- 格式化输出

### v1/

V1 是 vLLM 当前的主架构。核心包括：

- `v1/engine/`: LLMEngine、AsyncLLM。管理 engine core 和输出。
- `v1/core/`: Scheduler、KV cache manager。不与具体设备绑定。
- `v1/core/sched/`: 调度器主逻辑。
- `v1/worker/`: 抽象 worker 和 GPU worker 实现。
- `v1/attention/`: V1 的 attention backend。
- `v1/kv_cache_interface.py`: 给 platform 自定义 KV cache 的接口。
- `v1/kv_offload/`: KV cache offload 相关。

### attention

attention 层的抽象，包括：
- Attention metadata 定义（描述本轮 attention 需要哪些 KV block、哪些 query）。
- 通用 attention backend 接口。
- GPU attention backend 实现。

Ascend 在这里的集成方式是通过 platform 提供自己的 attention backend 类。

### model_executor

模型定义和加载。当 `LLM(model='some-model')` 被调用时：
1. `transformers_utils` 检测模型类型。
2. `model_executor` 加载对应模型实现。
3. 模型内部的 attention layers 调用 attention backend。

多模态模型（如 Qwen2-VL）也在这里。

### distributed

并行执行相关：
- `parallel_state.py`: 管理 TP/PP/DP worker group。
- `device_communicators/`: 自定义通信器。
- `kv_transfer/`: KV transfer 抽象接口（用于 PD 分离）。

### compilation

图模式相关。vLLM 支持两种图优化：
- CUDA graph（`cuda_graphs.md`）
- torch.compile（`torch_compile.md`）

Ascend 在此基础上提供 ACL graph 的捕获和 replay。

### platforms

平台抽象基类。`Platform` 定义了平台必须提供的接口：
- 设备检测
- attention backend 选择
- worker 创建
- 设备属性查询

`NPUPlatform`（在 vLLM Ascend 中实现）继承这个基类，提供 NPU 后端。

## 目录间的数据流

一个请求的完整流动（简化）：

```
entrypoints
  → engine.add_request()
    → scheduler.schedule()        [v1/core/sched/]
      → scheduler_output           [v1/core/sched/]
        → worker.execute_model()   [v1/worker/]
          → attention_metadata
          → model.forward()
            → attn_backend.forward()
            → sampling
  ← engine.step() → output_processor → 返回给客户端
```

## 与 vLLM Ascend 的边界

vLLM Ascend 主要替换以下部分：
- `entrypoints/`: 不替换，完全复用。
- `engine/`: 不替换（逻辑层面），可能通过 patch 调整调度参数。
- `scheduler/`: 不替换核心逻辑，可能通过 patch 调整 token budget 或 block 分配。
- `worker/`: 替换为 NPU worker 和 model runner。
- `attention/`: 替换为 Ascend attention backend（MLA/SFA/FA3/CP）。
- `distributed/`: 替换为 HCCL 实现。
- `compilation/`: 替换为 ACL graph。
- `model_executor/`: 大部分复用，attention layer 和部分 ops 走 Ascend 路径。

## 参考入口

- `$PATH_TO_VLLM/docs/design/arch_overview.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/patch.md`

## 思考与探索

1. scheduler 代码在 `v1/core/sched/`，scheduler output 定义在 `v1/core/sched/output.py`。为什么 scheduler 和 output 不放在同一个文件里？
2. `entrypoints` 完全不依赖具体设备，这是刻意设计还是巧合？
3. 如果 Ascend 想在 scheduler 里加一个 NPU 专用的调度策略，从分层角度看应该放在哪里？为什么？
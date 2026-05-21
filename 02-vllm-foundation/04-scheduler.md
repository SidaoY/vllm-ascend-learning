# 调度算法

Scheduler 是 vLLM 推理服务的核心。它每个 engine step 都要回答一个问题：在有限 token budget、KV cache、并发请求和服务约束下，本轮应该推进哪些请求、推进多少 token？

## Scheduler 不是简单的 batch 拼接器

普通直觉里，batching 是“把多个请求拼起来跑一次模型”。vLLM 的 scheduler 更复杂，因为每个请求的状态不同：

- 有的请求刚进入，需要 prefill prompt。
- 有的请求已经在 decode，每轮只需要少量 token。
- 有的请求 prompt 很长，需要 chunked prefill。
- 有的请求命中 prefix cache，可以跳过一部分 prefill。
- 有的请求需要 structured output、spec decode、多模态 encoder 输入。
- 有的请求因为 KV cache block 不足，需要等待或被抢占。

所以 scheduler 的目标不是单纯凑大 batch，而是在吞吐、TTFT、TPOT、公平性和内存之间做取舍。

## 一个更稳定的理解方式

可以把每个请求看成有两个进度：

- 已经计算到哪里：当前已经完成多少 token 的模型计算。
- 目标推进到哪里：prompt、已生成 token、speculative tokens 等共同决定还需要计算到哪里。

每个 step，scheduler 尝试让请求的“已经计算进度”追上“目标进度”。如果一次追不上，就分多轮推进。这种理解可以同时覆盖 prefill、decode、chunked prefill、prefix caching 和投机推理。

## Scheduler 每轮考虑什么

主要输入包括：

- waiting 请求队列。
- running 请求列表。
- 每个请求的 prompt 长度、已生成长度、已计算 token 数。
- sampling params 中的 max tokens、stop 条件等。
- token budget，例如最大 batched tokens。
- sequence budget，例如最大并发请求数。
- KV cache 可用 block。
- prefix cache 命中情况。
- encoder / 多模态输入预算。
- structured output 或 grammar 约束。
- spec decode 的 draft token。
- KV transfer / offload 等外部依赖。

主要输出包括：

- 本轮新进入 worker 的请求。
- 本轮继续执行的已有请求。
- 每个请求本轮调度的 token 数。
- 本轮需要分配或复用的 KV block。
- 本轮需要处理的 encoder / 多模态输入。
- 本轮完成、释放、抢占或等待的请求。
- 传给 worker 的 KV connector / structured output / spec decode metadata。

## Prefill 与 Decode 的调度差异

Prefill 通常一次推进多个 prompt token，计算密集，适合较大的矩阵计算。Decode 通常每个请求每轮推进少量 token，读历史 KV cache 多，常受内存带宽和 batch 形态影响。

Scheduler 需要在两者之间平衡：

- 如果长 prompt prefill 一次占满 token budget，decode 请求的 TPOT 会变差。
- 如果过度偏向 decode，长 prompt 的 TTFT 会变差。
- Chunked prefill 通过把长 prompt 拆成多轮，为 decode 插队留出空间。

这也是为什么 `max_num_batched_tokens`、`max_num_seqs`、chunked prefill、long prefill threshold 这类配置会强烈影响服务表现。

## KV Cache 对调度的约束

每推进一个 token，都需要对应的 KV cache slot。Scheduler 在决定调度前要确认 KV cache manager 能否提供足够 block。

如果 block 不足，可能发生：

- 新请求继续等待。
- 运行中的请求暂停或被抢占。
- 已完成请求释放 block 后再调度。
- 在某些配置下，使用 recompute、swap、offload 或外部 KV connector。

从服务现象看，KV block 不足会表现为吞吐下降、队列增长、TTFT 变差，甚至 OOM 或调度停滞。

## Prefix Cache 的影响

Prefix cache 命中后，请求的一部分 prompt KV 已经可复用。Scheduler 可以减少实际需要 prefill 的 token 数，并复用已有 KV block。

但 prefix cache 不是简单字符串缓存。它通常依赖 token ids、block hash、模型相关信息、多模态内容标识等。命中逻辑越复杂，scheduler 和 KV cache manager 的交互也越重要。

## SchedulerOutput 的意义

SchedulerOutput 是 scheduler 和 worker 的边界对象。它不是“模型输入”本身，而是告诉 worker 如何构造本轮模型输入：

- 哪些请求是新请求，哪些请求已经在 worker 本地缓存过状态。
- 每个请求本轮要计算多少 token。
- 哪些请求已经完成，需要释放本地状态。
- 哪些 KV block、encoder 输入、spec decode token 或 connector metadata 需要参与本轮执行。

具体字段可能随版本变化，但这个边界非常稳定：scheduler 做决策，worker 按决策执行。

## Ascend 相关铺垫

vLLM Ascend 后续会有 balance scheduling、动态 batch、PD 分离、KV transfer 等适配或优化。它们的共同背景是：NPU 上的性能不仅取决于单个 kernel，还取决于 scheduler 生成的 batch 形态、shape 稳定性、KV cache 压力和跨设备通信。

本章先理解 vLLM 原生 scheduler；Ascend 侧改动放到后续章节展开。

## 代码入口

- `$PATH_TO_VLLM/vllm/v1/core/sched`
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_utils.py`
- `$PATH_TO_VLLM/vllm/v1/request.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_balance_schedule.py`
- `$PATH_TO_VLLM_ASCEND/tests/ut/core/test_scheduler_dynamic_batch.py`

建议搜索关键词：`SchedulerOutput`、`token budget`、`running`、`waiting`、`prefix cache`、`chunked prefill`、`preempt`。

## 思考与探索

1. 为什么 scheduler 要优先考虑 running 请求，而不是每轮都从 waiting 队列重新排序所有请求？
2. 一个 32K prompt 请求如果被拆成多轮 prefill，对 TTFT 和 TPOT 分别有什么影响？
3. 如果线上 TTFT 突然变差，你会从 scheduler 的哪些输入信息开始排查？

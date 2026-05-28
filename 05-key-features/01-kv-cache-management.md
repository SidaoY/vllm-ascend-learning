# KV Cache 管理

KV cache 管理是 vLLM 和 vLLM Ascend 的核心能力之一。它决定一个服务能承载多长的上下文、多少并发请求、是否能复用 prefix、是否能做 PD 分离，以及 decode 阶段能否稳定地读到历史 K/V。

## KV Cache 管理管的不是一块普通内存

每个请求生成 token 时，模型每层都会产生 K/V。Decode 需要反复读取历史 K/V，所以 KV cache 既是显存大户，也是性能关键路径。

KV cache 管理至少要解决这些问题：

- 容量：显存里能放多少层、多少 token、多少请求的 K/V。
- 分配：每个请求新增 token 时如何拿到 slot。
- 索引：attention kernel 如何从 block table 找到历史 K/V。
- 释放：请求结束、取消、抢占后如何归还 block。
- 复用：prefix cache 命中后如何复用已有 K/V。
- 传输：PD 分离或 KV pool 中如何跨实例移动 K/V。
- 一致性：scheduler、worker、attention backend、connector 对同一份 K/V 的理解必须一致。

## vLLM 的基础抽象

vLLM 使用 block/page 管理 KV cache。KV payload 本身没有减少，但 block 化管理带来几个好处：

- 按需分配，避免为每个请求按最大上下文预留完整空间。
- 减少动态请求长度带来的外部碎片。
- 支持 prefix cache、copy-on-write、block 复用等能力。
- 让 scheduler 可以把 KV block 作为显式资源参与调度。
- 让 attention backend 通过 block table 找到非连续的历史 K/V。

学习入口：

- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/kv_cache_interface.py`
- `$PATH_TO_VLLM/docs/design/paged_attention.md`
- `$PATH_TO_VLLM/docs/design/prefix_caching.md`
- `$PATH_TO_VLLM/docs/design/hybrid_kv_cache_manager.md`

建议先理解 `block`、`block table`、`slot mapping`、`cache spec` 这些词，再看具体代码。

## KV Cache 的分配流程

KV cache 的分配不是简单的"有多少显存就分多少"。vLLM 在启动时经历一个完整的 warmup → profile → 计算 → 分配的流程。

### 第一步：模型加载与 Warmup

引擎启动时，首先加载模型权重到 NPU 显存。权重加载完成后，vLLM 会执行一次或多次 dummy forward（warmup），目的是：

- 触发 CUDA/NPU graph capture（如果开启了图模式）。
- 让 PyTorch 的 caching allocator 稳定下来，避免后续运行时出现不可预期的显存碎片。
- 为后续的显存 profiling 提供稳定的基准线。

### 第二步：Profiling 可用显存

Warmup 完成后，vLLM 调用 `torch.npu.mem_get_info()`（或 CUDA 侧的 `torch.cuda.mem_get_info()`）获取当前设备的空闲显存和总显存。这一步得到的是"模型权重 + graph buffer + 其他固定开销"之外的剩余空间。

关键参数 `gpu_memory_utilization`（默认 0.90）决定了这些剩余空间中有多少比例可以用于 KV cache：

```text
available_memory = free_memory * gpu_memory_utilization
```

这个参数不是越大越好。设得太高（比如 0.98）可能导致：
- 没有足够空间应对运行时的临时分配（如 attention workspace、通信 buffer）。
- OOM 发生在 decode 中途而非启动时，更难排查。

### 第三步：计算 Block 数量

拿到 `available_memory` 后，vLLM 通过 `get_num_blocks()` 计算可以分配多少个 KV cache block：

```python
# kv_cache_utils.py: get_num_blocks()
num_blocks = int(available_memory // page_size // num_layers)
```

其中：
- `page_size`：每个 block 的字节数，由 `block_size * num_kv_heads * head_dim * dtype_size * 2(K+V)` 决定。
- `num_layers`：模型的 attention 层数。

这个公式的含义是：**所有层共享同一份 block 池**。每层有自己独立的 K buffer 和 V buffer（大小都是 `num_blocks * page_size`），但 block table 只有一张——scheduler 为请求分配了 block `[5, 7, 12]`，这个结果对 32 层同时生效。

### 第四步：多 Worker 一致性处理

在 TP/PP 等多卡场景中，不同 worker 的可用显存可能不同。`get_kv_cache_configs()` 会：

1. 合并所有 worker 的 KV cache spec，得到全局的 layer 视图。
2. 为每个 worker 生成 projected groups（考虑 PP 切分后各 worker 实际持有的层）。
3. 取所有 worker 中最小的 `num_blocks` 作为统一值，确保 scheduler 的分配决策在所有 worker 上都有效。
4. 如果设置了 `num_gpu_blocks_override`，则跳过 profiling 结果，直接使用覆盖值。

### 第五步：自动适配 max_model_len

如果用户没有显式设置 `max_model_len`（即 `original_max_model_len == -1`），vLLM 会通过二分搜索自动估算一个能放入可用显存的最大模型长度：

```python
# kv_cache_utils.py: estimate_max_model_len()
# 二分搜索：left=1, right=original_max_model_len
# 每次检查 fits_in_memory(mid)，找到最大的能放入显存的 model_len
```

这个机制让用户不需要手动计算"我的显存能支持多长的上下文"，但也意味着如果 `gpu_memory_utilization` 设得太低，自动估算的 `max_model_len` 也会偏小。

### 完整流程图

```text
Startup
  │
  ├─ 1. Load model weights → NPU memory
  ├─ 2. Warmup (dummy forward, graph capture)
  ├─ 3. Profiling: torch.npu.mem_get_info()
  │     └─ available_memory = free_memory * gpu_memory_utilization
  ├─ 4. get_kv_cache_configs()
  │     ├─ Merge KV cache specs from all workers
  │     ├─ page_size = block_size * kv_heads * head_dim * dtype * 2
  │     ├─ num_blocks = available_memory // page_size // num_layers
  │     ├─ Take min num_blocks across workers for consistency
  │     └─ If max_model_len == -1, binary search auto-estimate
  ├─ 5. Allocate KV cache buffer (per-layer K/V tensors)
  └─ 6. Initialize KVCacheManager (block pool, free queue, prefix cache)
```

## Ascend 侧 Block Table

在 vLLM Ascend 中，scheduler 做出的 KV block 分配结果，需要被 worker/model runner 转换成 NPU attention backend 能消费的结构。这里的关键就是 block table 和 slot mapping。

| 阶段 | 组件 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1 | Scheduler | —  | SchedulerOutput（KV allocation） |
| 2 | Ascend worker | SchedulerOutput | NPU input batch |
| 3 | NPU input batch | —  | Block table / slot mapping |
| 4 | Ascend attention backend | Block table / slot mapping | 读写 KV cache buffer |

需要特别注意：

- block table 描述的是逻辑 token 到物理 KV block 的映射。
- slot mapping 描述本轮新 token 的 K/V 应该写到哪里。
- seq lens、query lens、position ids、block table 必须一起对齐。
- Ascend 当前设备亲和的 block size 通常是 128，一般不建议随意修改。
- graph、CP、KV transfer、量化 KV cache 都会让 metadata 更复杂。

### Ascend 侧 Block Size 的特殊处理

vLLM Ascend 对 block size 有额外的 patch（`patch_kv_cache_utils.py`），主要涉及 CP 场景下的对齐：

- 当开启 PCP 或 DCP 时，`scheduler_block_size` 需要乘以 CP 的并行度：`block_size * dcp * pcp`。
- 对于 hybrid KV cache（如 MLA + SWA-MLA 混合），社区 vLLM 在 CP 场景下会直接报错，但 Ascend 通过 LCM（最小公倍数）计算统一的 `scheduler_block_size` 来支持这种组合。
- CP + KV transfer 场景中，`cp_kv_cache_interleave_size` 需要与 `block_size` 保持一致（通常都是 128），否则 KV 切分和传输会错位。

代码入口：

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/block_table.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/npu_input_batch.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/model_runner_v1.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_kv_cache_utils.py`

## Prefix Caching

Prefix caching 的目标是复用公共前缀已经计算好的 KV cache。它对系统的价值很直接：公共 system prompt、RAG 模板、agent 历史上下文等重复前缀越多，prefill 的重复计算就越值得省掉。

但 prefix cache 不是字符串缓存。它通常和 token ids、block hash、模型身份、LoRA、多模态输入、cache salt 等信息有关。两个文本看起来相同，不代表 token 序列和 cache key 一定相同。

排查 prefix cache 时先确认：

- token ids 是否完全一致。
- 前缀长度是否足够覆盖完整 block。
- 相关模型和 adapter 信息是否一致。
- 是否因为内存压力被驱逐。
- 是否与 CP、PD、KV transfer 组合使用。

### Prefix Cache 与 Chunked Prefill 的交互

Chunked prefill 把长 prompt 拆成多个 chunk 分步执行。这给 prefix cache 带来两个影响：

**1. Cache 命中粒度变细**

```text
Without chunked prefill:
  prompt = [sys_prompt(1000 tokens) + user_input(500 tokens)]
  → Entire prompt prefill in one pass
  → prefix cache only hits sys_prompt (if previously cached)

With chunked prefill (chunk_size=256):
  chunk 0: tokens [0,    255]   → may hit cache
  chunk 1: tokens [256,  511]   → may hit cache
  ...
  chunk 3: tokens [768,  999]   → last segment of sys_prompt
  chunk 4: tokens [1000, 1255]  → user_input begins, cache miss
```

chunked prefill 让 prefix cache 的命中检查更细粒度，但也意味着每个 chunk 都需要做一次 cache 查询。

**2. 部分命中后的处理**

如果 sys_prompt 的前 800 个 token 命中 cache，但后 200 个 token 没有（比如 sys_prompt 被修改了），chunked prefill 可以：

- Chunk 0-2（前 768 tokens）：直接从 cache 复用 KV。
- Chunk 3（tokens 768-999）：前 32 个 token 命中 cache，后 224 个 token 需要计算。
- Chunk 4+：全部需要计算。

这种"部分命中"在不开 chunked prefill 时很难处理（因为整个 prompt 是一次性 prefill），但 chunked prefill 天然支持。

**3. 排查要点**

- chunked prefill 下 prefix cache 命中率可能看起来更低（因为查询次数更多），但实际节省的计算量可能更大。
- 确认 chunk 边界是否与 block 边界对齐——如果不对齐，cache 命中后仍可能需要重新计算部分 token。
- CP + chunked prefill + prefix cache 的组合需要确认分布式 KV 的 cache key 是否正确。

## KV Offload

KV offload 把部分 KV cache 从 NPU 显存转移到其他介质，常见是 CPU/NPU 之间移动。它的目标是缓解显存压力，但代价是引入传输延迟和同步复杂度。

适合先问：

- offload 的对象是哪些层、哪些请求、哪些 block？
- 什么时候搬出，什么时候搬回？
- decode 需要读取时是否会阻塞？
- 失败或超时如何处理？
- profiler 里传输时间是否超过节省的显存收益？

代码入口：

- `$PATH_TO_VLLM/vllm/v1/kv_offload`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/kv_offload`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload`

## KV Transfer 和 KV Pool

PD 分离后，prefill 实例生成 KV，decode 实例继续使用 KV。KV transfer 就是把这份 KV 从 producer 交给 consumer。

| 步骤 | 发起方 | 接收方 | 动作 |
| --- | --- | --- | --- |
| 1 | Scheduler | Prefill worker | schedule prefill |
| 2 | Prefill worker | —  | write KV cache |
| 3 | Prefill worker | Connector / KV pool | publish or transfer KV |
| 4 | Connector / KV pool | Decode worker | load KV for decode |
| 5 | Decode worker | Scheduler | KV ready / continue decode |

KV transfer 的正确性依赖：

- producer/consumer role 配置一致。
- layer 顺序、block size、dtype、layout 一致。
- 每个请求的 KV metadata 能被正确传递。
- 传输完成信号和 scheduler 状态一致。
- 网络失败、超时、重试、recompute 策略清楚。

KV pool / Ascend store 更进一步，把 KV cache 放入共享存储或池化系统，支持更多实例之间复用。它会引入 pool scheduler、pool worker、backend、lease、eviction 等问题。

代码入口：

- `$PATH_TO_VLLM/vllm/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer/kv_p2p`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer/kv_pool`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/kv_pool.md`

## Hybrid KV Cache

Hybrid KV cache 是 vLLM 社区引入的一种混合管理策略，用于处理模型中同时存在多种 attention 类型的情况。典型场景是 DeepSeek-V3 等模型：既有 MLA（Multi-head Latent Attention）层，又有 SWA-MLA（Sliding Window Attention）层。

### 为什么需要 Hybrid

不同 attention 类型的 KV cache 需求不同：

| Attention 类型 | KV cache 特点 | block 需求 |
| --- | --- | --- |
| Full attention (MLA) | 需要完整历史 KV | 按完整上下文分配 block |
| Sliding window (SWA-MLA) | 只需要最近 W 个 token 的 KV | 按窗口大小分配 block |

如果统一按 full attention 分配，SWA 层会浪费大量显存。Hybrid KV cache manager 为不同类型的 attention 层分别管理 block pool，让 SWA 层只分配窗口大小的 block。

### Ascend 侧的适配

vLLM Ascend 在 `patch_kv_cache_utils.py` 中对 hybrid KV cache 做了额外处理：

- CP 场景下，社区 vLLM 对 hybrid KV cache + CP 的组合会直接报错，但 Ascend 通过 LCM（最小公倍数）计算统一的 `scheduler_block_size` 来支持这种组合。
- 具体做法：取 full attention 和 SWA attention 各自 block size 的 LCM 作为调度 block size，确保 scheduler 的分配决策对两种 attention 类型都有效。

### 排查要点

- 确认 `num_blocks` 是否按 attention 类型分别计算。
- CP + hybrid 场景下，确认 `scheduler_block_size` 的 LCM 计算是否正确。
- SWA 层的 block 是否被错误地按 full attention 分配。

## Block 生命周期

理解一个 KV block 从分配到释放的完整生命周期，有助于排查 block 泄漏、不足和 prefix cache 问题。

```text
                    ┌──────────────────────────────┐
                    │        Free Block Pool        │
                    └──────────────┬───────────────┘
                                   │ allocate
                                   ▼
                    ┌──────────────────────────────┐
                    │       Allocated (in use)      │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  Normal finish   │  │   Preempted      │  │    Cancelled     │
   │  (request completed) │  │  (preempted by higher │  │   (client disconnected)│
   │                      │  │   priority request)   │  │                        │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │ free / evict
                                  ▼
                    ┌──────────────────────────────┐
                    │        Free Block Pool        │
                    └──────────────────────────────┘
```

**Prefix cache 的特殊路径：**

当 prefix cache 命中时，block 不是从 free pool 分配，而是直接从 prefix cache pool 引用（copy-on-write）。被引用的 block 只有在所有引用者都释放后才会回到 free pool。

**常见泄漏场景：**

- 请求异常退出但 block 未归还（scheduler 状态与 KV cache manager 不一致）。
- Prefix cache 引用计数错误导致 block 永远不被释放。
- PD 分离中 consumer 端 block 未正确释放。

## KV Cache 监控指标

排查 KV cache 问题时，以下指标可以帮助快速定位：

| 指标 | 含义 | 排查方向 |
| --- | --- | --- |
| `vllm:gpu_cache_usage_perc` | KV cache 使用率 | > 90% 时关注是否频繁 preempt |
| `vllm:num_requests_waiting` | 等待调度的请求数 | 持续 > 0 说明 KV cache 或 slot 不足 |
| `vllm:num_requests_running` | 正在运行的请求数 | 与 max_num_seqs 对比 |
| `vllm:num_preemptions_total` | 累计抢占次数 | 频繁抢占说明 KV cache 容量不足 |
| `vllm:prefix_cache_hit_rate` | prefix cache 命中率 | 低命中率检查 token 序列和 cache salt |
| `vllm:prefix_cache_queries_total` | prefix cache 查询次数 | 与命中率配合分析 |

此外，Ascend 侧还可以关注：

- `torch.npu.memory_allocated()` / `torch.npu.memory_reserved()`：NPU 显存使用。
- block table 中 `num_free_blocks` 的变化趋势。
- KV transfer 的传输延迟和失败次数（PD 分离场景）。

## KV Cache Quantization

KV cache quantization 会改变 KV cache 的 dtype、scale/offset metadata 和 attention backend 读写路径。vLLM Ascend 中可以关注 C8 KV cache 这类路径，但学习时先把它当作"KV cache layout 的一种变化"来理解。

代码入口：

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/quantization/methods/kv_c8.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`
- `$PATH_TO_VLLM_ASCEND/tests/ut/quantization`

## 常见问题

OOM：先看模型权重、max model len、max_num_seqs、max_num_batched_tokens、graph buffer、spec decode 和 KV dtype。

KV block 不足：先看并发、输出长度、请求是否及时释放、prefix cache 是否占用过多 block。

Decode 精度异常：先看 block table、slot mapping、position ids、seq lens、dtype 和 layer 顺序。

KV transfer 失败：先看 producer/consumer role、connector 配置、端口、网络、block size、timeout。

Prefix cache miss：先看 token ids、block hash、模型/LoRA/多模态信息和驱逐。

## 参考入口

- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/kv_cache_interface.py`
- `$PATH_TO_VLLM/vllm/v1/kv_offload`
- `$PATH_TO_VLLM/vllm/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/block_table.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/kv_offload`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/quantization/methods/kv_c8.py`
- `$PATH_TO_VLLM_ASCEND/tests/ut/kv_connector`
- `$PATH_TO_VLLM_ASCEND/tests/ut/distributed/ascend_store`

## 思考与探索

1. PagedAttention 没有减少每个 token 的 K/V payload，为什么仍然能显著提升服务容量？
2. PD 分离中，如果 prefill 和 decode 的 block size 不一致，会出现什么问题？
3. Prefix cache 命中率高但 TTFT 没明显下降，你会从哪些方向排查？
4. 如果 `gpu_memory_utilization` 从 0.90 调到 0.95，KV cache 容量增加了约 5.5%，但为什么可能导致运行时 OOM？

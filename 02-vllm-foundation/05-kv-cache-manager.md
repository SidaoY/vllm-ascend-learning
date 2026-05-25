# vLLM KV Cache Manager

KV cache manager 负责把逻辑上的请求上下文映射到物理 KV cache block。它是 scheduler 和 attention backend 之间的桥：scheduler 需要它判断能不能调度更多 token，attention backend 需要它提供 block table 之类的信息来读取历史 K/V。

## 为什么需要单独的 KV Cache Manager

如果每个请求都独占一段最大长度 KV cache，内存利用率会很低；如果每次动态扩展连续内存，又会带来碎片和搬迁成本。vLLM 使用 block/page 的方式管理 KV cache，让请求逻辑上连续，物理上按 block 分配和复用。

KV cache manager 要解决的问题包括：

- 哪些 block 空闲，哪些 block 已分配。
- 一个请求当前持有哪些 block。
- 新 token 到来时需要追加多少 block。
- 请求结束时如何释放 block。
- prefix cache 命中时如何复用已计算 block。
- 多种 KV cache spec 共存时如何分组管理。
- KV offload / KV transfer 场景下如何和外部存储或远端实例协作。

## 核心概念

Block：固定数量 token 的 KV cache 存储单位。block size 会影响内部碎片、metadata 开销和 backend 亲和性。

Block pool：物理 block 的池子，负责空闲 block 管理、引用计数、复用和释放。

Block table：请求逻辑 block 到物理 block 的映射。attention backend 根据它找到历史 K/V。

KV cache group：一组共享相同管理方式的 KV cache。普通 full attention、sliding window、MLA、Mamba、cross attention 等可能需要不同 spec。

Prefix cache：把已经计算过的 prefix block 缓存起来，后续请求命中后复用，减少 prefill 计算。

## Scheduler 如何使用 KV Cache Manager

每个 step 调度前，scheduler 会问 KV cache manager：如果给某个请求推进 N 个 token，需要多少新 block？当前有没有足够 block？是否命中 prefix cache？

调度成功后，KV cache manager 会返回本轮需要使用的 block 信息。Scheduler 再把这些信息放进 scheduler output，worker/model runner 用它构造 block table 和 attention metadata。

请求完成或被取消时，scheduler 会通知 KV cache manager 释放对应 block。Prefix cache 命中的 block 可能不会立即物理释放，而是保留给后续请求复用。

## Prefix Caching 的直观流程

| 步骤 | 判断 | 动作 |
| --- | --- | --- |
| New Request Tokens | —  | Split into blocks |
| Split into blocks | —  | Compute block hash |
| Compute block hash | Prefix block hit? | —  |
| → Yes | —  | Reuse cached KV block |
| → No | —  | Allocate new block → Run prefill and write KV → Cache completed block |
| After reuse or cache | —  | Schedule remaining tokens |

Prefix cache 对 TTFT 很有帮助，但也引入了额外复杂度：

- 需要保证两个 prefix 真的等价。
- 多模态输入不能只看 placeholder token。
- block hash、cache salt、LoRA、模型配置等都可能影响命中安全性。
- 命中后 scheduler 仍要正确处理剩余 token 和 block table。

## Hybrid KV Cache

不是所有 layer 都使用相同的 KV cache 形态。某些模型可能混合 full attention、sliding window、chunked local attention、MLA、Mamba 或 cross attention。Hybrid KV cache manager 的作用是让不同 spec 的层可以在统一调度框架下共存。

对初学者来说，先记住两点：

- KV cache manager 不是只管理一个简单数组。
- attention backend、模型结构和 KV cache spec 会共同决定 block 的形状和生命周期。

## KV Offload 与 KV Transfer 的位置

KV offload：把暂时不用的 KV cache 移到 CPU 或其他介质，以换取设备内存空间。代价是传输延迟和调度复杂度。

KV transfer：在 disaggregated prefill、remote prefill、KV connector 等场景中，把 KV cache 在实例之间传输。Scheduler 需要知道哪些 KV 已经在远端、哪些正在传输、失败时是否需要 recompute。

这些机制不是 KV cache manager 的简单附属功能，而是会影响请求能否被调度、何时调度、是否需要等待外部依赖。

## 和 Attention Backend 的关系

KV cache manager 管理“哪些 block 属于哪个请求”。Attention backend 关心“本轮 kernel 如何根据 metadata 读写这些 block”。

中间的桥通常包括：

- block table。
- slot mapping。
- sequence length。
- query length。
- cache dtype 和 layout。
- 每个 backend 需要的特殊 metadata。

这条链路一旦错位，就会出现 shape mismatch、读错 KV、输出精度异常或 kernel 不支持等问题。

## 代码入口

- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/core/single_type_kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_coordinator.py`
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_utils.py`
- `$PATH_TO_VLLM/vllm/v1/kv_cache_interface.py`
- `$PATH_TO_VLLM/vllm/v1/kv_offload`
- `$PATH_TO_VLLM/vllm/distributed/kv_transfer`
- `$PATH_TO_VLLM/docs/design/hybrid_kv_cache_manager.md`
- `$PATH_TO_VLLM/docs/features/automatic_prefix_caching.md`

建议搜索关键词：`KVCacheConfig`、`KVCacheSpec`、`block pool`、`prefix cache`、`block hash`、`KV connector`。

## 常见误区

- 误区一：KV cache manager 只负责显存申请。它还影响 scheduler、prefix caching、attention metadata 和分布式。
- 误区二：Prefix cache 命中只是字符串相同。真实系统要考虑 token、模型、salt、多模态内容和 block 边界。
- 误区三：Block table 是纯 Python 状态。它最终会影响设备 kernel 如何访问 KV cache。
- 误区四：KV offload 只会降低显存。它也可能增加延迟、带宽压力和调度复杂度。

## 思考与探索

1. 为什么 block size 太大和太小都会带来问题？
2. 如果一个请求命中 prefix cache，scheduler 和 attention backend 分别需要知道什么？
3. 线上出现 KV cache usage 很高但吞吐不高时，你会怀疑哪些可能原因？

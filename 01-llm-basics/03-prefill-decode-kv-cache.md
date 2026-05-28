# Prefill、Decode 与 KV Cache

Prefill、decode 和 KV cache 是推理系统最核心的三件事。只要理解这三者，就能理解为什么 vLLM 需要 scheduler、PagedAttention、block table、prefix caching、chunked prefill，也能理解 vLLM Ascend 为什么会有大量围绕 KV cache 和 attention 的适配。

## Autoregressive Generation

生成式 LLM 通常是 autoregressive 的：每一步根据已有上下文预测下一个 token。

```text
输入:  The capital of France is
预测:  Paris
下一步输入上下文: The capital of France is Paris
预测:  .
```

如果每生成一个 token 都重新计算整个上下文，成本会非常高。KV cache 的目的就是保存历史 token 在每层 attention 中的 K/V，避免重复计算。

## Prefill

Prefill 处理 prompt 中已有的 token。例如 prompt 长度是 1024，模型会一次性处理这 1024 个 token。

Prefill 会做三件关键事情：

- 计算每一层每个 prompt token 的 hidden states。
- 为每一层 attention 写入 prompt token 的 K/V cache。
- 取最后位置的 logits，用来采样第一个输出 token。

Prefill 的输入长度通常较大，所以更像“大矩阵计算”。长 prompt、长上下文、多模态输入都会增加 prefill 成本。

## Decode

Decode 每次只处理新生成的 token。假设已经生成到第 1025 个 token，下一步只需要输入这个新 token，同时读取前 1024 个 token 的 KV cache。

Decode 会做三件关键事情：

- 计算新 token 的 Q/K/V。
- 用 Q 读取历史 KV cache，完成 attention。
- 把新 token 的 K/V 追加到 KV cache。

Decode 的单步计算量小，但需要执行很多轮。每轮都要读历史 KV cache，所以常常受内存带宽、KV cache 布局和 batch 大小影响。

## KV Cache 保存什么

每一层 attention 都会产生 K 和 V。KV cache 保存的是历史 token 的 K/V，不是 token 文本，也不是 hidden states 全量。

一个简化的 KV cache 维度可以理解为（暂不考虑page attention）：

```text
layers x tokens x kv_heads x head_dim x 2(K,V)
```

真实代码中会因为 backend、dtype、block size、MLA、量化、内存布局而变化。先记住：上下文越长、层数越多、KV heads 越多、head dim 越大，KV cache 越大。

## KV Cache 对显存的影响

KV cache 会随“活跃请求数 x 上下文长度”增长。服务端同时跑很多长上下文请求时，KV cache 很容易成为主要内存占用。

具体来说。一个模型通常有很多层，每层都要为每个 token 保存 K/V。即使权重不变，用户并发越高、输出越长，KV cache 也会持续增长。OOM 问题经常不是模型权重放不下，而是运行中的 KV cache 放不下。

## Block、Page、Block Table

vLLM 的核心思想之一是把 KV cache 切成固定大小的 block/page。请求的逻辑上下文仍然是连续的，但底层物理 KV block 可以按需分配、追加和复用，把浪费控制在 block 粒度。

PagedAttention 不是让同样 token 的 K/V 数据本身变少，而是让 KV cache 更适合动态服务场景。请求长度不同、结束时间不同，如果按最大上下文长度为每个请求预留一整段 KV cache，会浪费未使用位置；如果频繁申请可增长的连续空间，又容易产生碎片和搬迁成本。

block size 会影响内部碎片、metadata 开销和 attention kernel 的硬件亲和性。在当前 vLLM Ascend 的主流 NPU 路径中，Ascend 设备更亲和的 block size 通常是 128，开发中一般不会随意修改这个值。

- block/page：一小段固定数量 token 的 KV cache 存储单位。
- block table：记录一个请求的逻辑 token block 对应哪些物理 cache block。
- slot mapping：告诉 kernel 每个 token 应该写入或读取哪个 cache slot。

可以把 block table 想象成操作系统里的页表：请求看到的是连续上下文，底层物理内存可以是分散的 block。

## Chunked Prefill

长 prompt 的 prefill 可能一次占用太多 token budget，阻塞 decode 请求。Chunked prefill 会把长 prompt 分块处理，让 scheduler 在多个 engine step 中逐步完成 prefill，并有机会穿插 decode。

它解决的是调度公平性和延迟问题，但也增加了 scheduler、attention metadata 和 KV cache 管理复杂度。

## Prefix Caching

很多请求有相同前缀，例如同一个 system prompt、同一份文档、同一轮多用户共享上下文。Prefix caching 会缓存已经计算过的前缀 KV cache，后续请求命中后可以跳过一部分 prefill。

它能显著降低 TTFT，但需要准确判断“前缀是否真的相同”。对于多模态输入，不能只看 placeholder token，还需要区分真实图片或媒体内容。

## KV Offload 和 KV Transfer

KV offload：把暂时不用的 KV cache 放到 CPU 或其他介质，缓解设备内存压力，但会引入搬运开销。

KV transfer：在 disaggregated prefill 或多实例场景中，把 prefill 侧生成的 KV cache 传给 decode 侧，让两类工作分离部署。

这些内容会在后续 vLLM Ascend 的 KV cache 管理和关键特性章节深入。

## 和代码的连接

- PagedAttention 设计：`$PATH_TO_VLLM/docs/design/paged_attention.md`
- Prefix caching 设计：`$PATH_TO_VLLM/docs/design/prefix_caching.md`
- vLLM KV manager：`$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- KV cache 工具：`$PATH_TO_VLLM/vllm/v1/core/kv_cache_utils.py`
- KV cache 接口：`$PATH_TO_VLLM/vllm/v1/kv_cache_interface.py`
- Ascend block table：`$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/block_table.py`

## 常见误区

- 误区一：KV cache 是可选优化。对现代长上下文 decode 来说，没有 KV cache 基本不可接受。
- 误区二：KV cache 只影响内存。它还影响 attention kernel、scheduler、prefix caching、并行和通信。
- 误区三：prefill 完成后请求就快结束了。很多长输出请求的大部分时间花在 decode 循环。
- 误区四：block table 只是 Python 状态。它最终会影响设备 kernel 如何读写 KV cache。

## 思考与探索

1. 假设 block size 是 16，一个请求当前有 33 个 token。思考 PagedAttention 需要分配几个 KV block？其中有多少 token 位置暂时未使用？
2. 思考一个 32K prompt 的请求为什么可能需要 chunked prefill。
3. 在 `kv_cache_manager.py` 里找到分配和释放 block 的相关函数名，记录你不理解的字段，后续 vLLM 章节再回来看。

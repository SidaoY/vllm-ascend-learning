# Attention Backend 与 PagedAttention

Attention backend 是 vLLM 高吞吐推理的关键抽象。它连接了三件事：模型里的 attention layer、scheduler/KV cache manager 管理的 block、设备侧 kernel 对内存布局和 metadata 的要求。

## 为什么需要 Attention Backend

不同模型、硬件和场景对 attention 的要求不一样：

- Dense attention、sliding window、local attention、MLA、Mamba、cross attention 的 KV cache 形态不同。
- Prefill 和 decode 的计算形态不同。
- GPU、NPU、CPU、XPU 等平台支持的 kernel 不同。
- dtype、量化、block size、head dim、KV heads 都会影响可用路径。
- graph capture 要求 shape 和内存地址更稳定。

如果模型代码直接调用某个固定 attention kernel，系统很难支持这些差异。Attention backend 抽象就是为了把“模型结构”和“具体 kernel 实现”解耦。

## PagedAttention 的核心思想

PagedAttention 不是一种让 K/V 数据变少的魔法，而是一套 KV cache 分页管理和访问机制。

请求看到的是连续上下文：

```text
token 0, token 1, token 2, ..., token N
```

底层物理 KV cache 可以分散在多个 block 中：

```text
logical block 0 -> physical block 17
logical block 1 -> physical block 03
logical block 2 -> physical block 42
```

attention backend 通过 block table 找到每个请求的历史 K/V。这样 vLLM 可以按需分配、释放和复用 block，避免为每个请求预留最大上下文长度。

## Metadata 连接了 scheduler 和 kernel

Scheduler 决定本轮哪些 token 要算，KV cache manager 决定这些 token 对应哪些 block。Attention backend 还需要一组 runtime metadata，常见包括：

- block table：逻辑 block 到物理 block 的映射。
- slot mapping：当前 token 的 K/V 应写入哪个 cache slot。
- sequence length：每个请求的历史长度。
- query length：本轮每个请求要计算多少 query token。
- query start location：把多个请求拼接后，每个请求在 batch 中的起始位置。
- common prefix blocks：某些优化路径会利用 batch 内共同前缀。
- backend-specific metadata：某些 kernel 或模型结构需要额外字段。

具体字段名会随 vLLM 版本变化，但这些概念非常稳定。

## Prefill 和 Decode 的 Attention 差异

Prefill：

- query 数多。
- key/value 主要来自当前 prompt。
- 计算更像大矩阵 attention。
- 更关注计算吞吐和中间矩阵访存。

Decode：

- 每个请求 query 数少。
- key/value 主要来自历史 KV cache。
- 更关注 KV cache 读取、block table、metadata 和 batch 形态。
- 很容易受内存带宽和 kernel launch overhead 影响。

同一个 backend 可能同时处理 prefill 和 decode，也可能内部拆成不同路径。

## Backend 选择

Backend 选择通常受这些因素影响：

- 平台能力：GPU/NPU/CPU/XPU。
- 用户指定的 attention backend。
- 模型结构：MHA/GQA/MLA/sliding window/local attention。
- dtype 和量化。
- 是否启用 graph/compile。
- 是否支持当前 block size、head dim、sequence length。
- 是否需要训练推理一致性或特殊精度路径。

vLLM Ascend 后续会在 platform 和 attention backend 层做大量适配。理解 vLLM 原生 backend 抽象后，再看 Ascend 的 attention 路径会顺很多。

## 与 KV Cache Manager 的分工

| 模块 | 负责什么 |
| --- | --- |
| Scheduler | 决定本轮推进哪些 token |
| KV Cache Manager | 分配、复用、释放 KV block |
| Model Runner | 构造 input batch、block table、slot mapping、attention metadata |
| Attention Backend | 使用 metadata 调用合适 kernel，读写 KV cache |
| Model Layer | 提供 Q/K/V/O projection 和模型结构参数 |

这几个模块任何一个出错，都可能表现为 attention 输出异常。

## FlashAttention 和 PagedAttention 的关系

FlashAttention 主要优化 attention 计算过程，减少中间矩阵落地和显存访问。PagedAttention 主要优化 KV cache 的服务化管理和访问。它们不是互斥概念，也不是同一层抽象。

可以这样理解：

- FlashAttention 更关注“attention 怎么算得快”。
- PagedAttention 更关注“长上下文、多请求的历史 KV 怎么管理和读取”。

在真实 backend 中，两类思想可能一起出现。

## 代码入口

- `$PATH_TO_VLLM/docs/design/paged_attention.md`
- `$PATH_TO_VLLM/docs/design/attention_backends.md`
- `$PATH_TO_VLLM/vllm/v1/attention`
- `$PATH_TO_VLLM/vllm/model_executor/layers/attention`
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/kv_cache_interface.py`
- `$PATH_TO_VLLM/vllm/v1/worker`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`

建议搜索关键词：`AttentionBackend`、`AttentionMetadata`、`block table`、`slot mapping`、`query_start_loc`、`seq_lens`、`PagedAttention`。

## 常见问题定位

- block table 错：可能读到错误历史 KV，输出精度异常或直接 shape 错。
- slot mapping 错：新 token 的 K/V 写入位置错误，后续 decode 会持续受影响。
- seq lens 错：mask、causal attention 或 kernel 边界可能错误。
- backend 选择错：可能走慢路径、不支持路径或精度不同路径。
- block size 不匹配：kernel 可能不支持或性能大幅下降。

## 思考与探索

1. 用一句话分别解释 block table 和 slot mapping。
2. 为什么 decode 比 prefill 更依赖 KV cache layout？
3. 如果一个 backend 在 prefill 正常、decode 异常，你会重点检查哪些 metadata？

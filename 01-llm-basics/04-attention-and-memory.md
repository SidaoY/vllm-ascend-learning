# Attention 与内存视角

Attention 是 Transformer 的核心，也是推理优化最集中的地方之一。初次学习不需要一开始推导完整公式，但需要理解 Q/K/V、causal mask、prefill attention、decode attention、PagedAttention 和内存布局之间的关系。

## Attention 的直观含义

每个 token 生成 query，去和上下文 token 的 key 做匹配，再根据匹配权重读取 value。简化写法是：

```text
Q, K, V = q_proj(hidden), k_proj(hidden), v_proj(hidden)
scores = Q x K^T / sqrt(head_dim)
weights = softmax(scores + mask)
context = weights x V
output = o_proj(context)
```

其中 `o_proj` 是 output projection，负责把 attention heads 的结果重新投影回模型 hidden size。代码里讨论 KV cache 时主要关注 K/V，是因为历史 token 的 K/V 可以复用；Q 通常来自当前要计算的 token，不会像 K/V 那样作为历史缓存长期保存。

如果模型使用 RoPE，位置相关变换通常会作用在 Q/K 上，再进入 `Q x K^T` 这一步。也就是说，attention 不只在比较内容相似度，还会通过 Q/K 中的位置编码感知 token 的顺序和距离。

在生成式模型里，mask 是 causal 的：当前位置只能看过去和自己，不能看未来。

## Prefill Attention

Prefill 阶段一次处理整个 prompt。假设 prompt 长度为 N，attention 需要处理 N 个 query 和 N 个 key/value。因为 causal mask 的存在，第 i 个 token 只能看前 i 个 token。

Prefill attention 的特点：

- 序列长度可能很长。
- 计算量随 prompt 长度增长很快。
- 更适合使用高效矩阵计算和 FlashAttention 类算法。
- 会同时生成用于后续 decode 的 KV cache。

## Decode Attention

Decode 阶段通常每个请求每步只新增一个 query，但需要读取这个请求全部历史 K/V。

Decode attention 的特点：

- query 很少，历史 KV 很长。
- 大量时间花在读 KV cache。
- KV cache 布局、block table、内存带宽非常重要。
- batch 中每个请求历史长度可能不同，metadata 更复杂。

这就是为什么 decode 经常被认为是 memory bandwidth bound。

## FlashAttention 与 PagedAttention

FlashAttention 主要解决 attention 计算中的显存读写效率问题，通过分块计算减少中间矩阵落地。它对 prefill 这类大矩阵 attention 很重要。

PagedAttention 主要解决 KV cache 管理问题。它把 KV cache 切成 block/page，让不同请求按需分配物理 block，通过 block table 让 kernel 找到对应的历史 K/V。

二者关注点不同：

| 技术 | 主要解决的问题 | 典型影响 |
| --- | --- | --- |
| FlashAttention | attention 计算和中间显存访问效率 | prefill 性能、长序列计算 |
| PagedAttention | KV cache 分页管理和高并发内存利用 | decode、并发、长上下文、内存碎片 |

## Attention Backend

推理框架需要根据模型、硬件、dtype、序列形态选择不同 attention 实现。vLLM 中 attention backend 抽象用于屏蔽这些差异。

backend 需要处理：

- 输入 Q/K/V。
- KV cache 读写。
- block table 或 slot mapping。
- sequence length 和 query length。
- prefill/decode 的不同 metadata。
- 特定模型结构，例如 MLA、sliding window、ALiBi。

后续 vLLM 基础章节会单独深入 attention backend 和 PagedAttention。

## MLA、SFA 和 Ascend 路径

在 vLLM Ascend 中，attention 适配会更复杂。原因包括：

- NPU kernel 对内存布局、对齐、shape 可能有特定要求。
- MLA 这类模型结构的 KV cache 表示与普通 MHA/GQA 不同。
- 长上下文 CP 并行会改变 attention 的通信和分块方式。
- 图模式要求 shape 和内存地址更稳定。

当前阶段只需要知道这些名字的定位：

- MLA：一种特殊 attention/KV 表示，常见于 DeepSeek 类模型。
- FA/FA3：这里的 FA3 指 Flash Attention 3，vLLM Ascend 中对应 `flash_attn_npu_v3.flash_attn_with_kvcache` backend，主要用于让推理侧 attention 实现和训练侧 FlashAttention 更一致；它和 KV cache 量化不是同一个概念。
- fa_quant / C8 KV cache quantization：KV cache 量化相关路径，容易和 FA3 的名字混淆，当前只需要知道它属于量化和 KV cache 管理交叉的问题。
- SFA：Sparse Flash Attention，是 vLLM Ascend 中面向 DSA / sparse attention 模型的 backend 路径，典型关联 DeepSeek-V3.2、GLM5 等模型；不要把它理解成泛指所有 Ascend attention kernel。
- CP attention：为 context parallelism 设计的 attention 路径。

这里先建立概念边界即可。SFA/DSA 的具体代码路径、模型适配和 CP 结合方式，会在 vLLM Ascend 章节里展开。

## 内存布局为什么重要

设备 kernel 不只是“拿到张量就能快”。张量的连续性、stride、对齐、dtype、block 大小都会影响性能。KV cache 尤其敏感，因为 decode 每步都要反复读取历史 KV。

常见影响因素：

- block size 过小，metadata 和调度开销增加，block size 过大，内部碎片和内存浪费增加。在当前 vLLM Ascend 的主流 NPU 路径中，Ascend 设备更亲和的 block size 通常是 128，开发中一般不会随意修改这个值。
- 非连续布局可能导致 kernel 读写效率下降。
- 对齐不满足硬件要求可能触发额外拷贝或不支持路径。

## 和代码的连接

- vLLM attention backend 文档：`$PATH_TO_VLLM/docs/design/attention_backends.md`
- PagedAttention 文档：`$PATH_TO_VLLM/docs/design/paged_attention.md`
- vLLM V1 attention：`$PATH_TO_VLLM/vllm/v1/attention`
- attention layer：`$PATH_TO_VLLM/vllm/model_executor/layers/attention`
- vLLM Ascend attention：`$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`

## 思考与探索

1. 用一句话区分 FlashAttention 和 PagedAttention。
2. 解释为什么 decode attention 更依赖 KV cache 读性能。
3. 在 vLLM Ascend 的 `attention` 目录里看文件名，列出你认为和 MLA、SFA、CP 相关的文件。

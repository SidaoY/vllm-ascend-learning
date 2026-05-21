# 01 LLM Basics

本章介绍阅读 vLLM 和 vLLM Ascend 代码前必须掌握的大模型推理基础。这里不追求训练理论完整性，也不会深入某个模型论文，而是围绕推理服务中的真实问题展开：请求如何变成 token，模型如何生成下一个 token，KV cache 为什么重要，scheduler 为什么要区分 prefill 和 decode，MoE 为什么会带来负载均衡问题。

本章旨在帮助读者用统一语言讨论请求、模型、内存、并行和性能指标。进入后续 vLLM 章节时，看到 `scheduler`、`KVCacheManager`、`attention backend`、`SamplingParams`、`parallel_state`、`fused_moe` 等名字不会完全陌生。

## 学习顺序

1. [00-inference-lifecycle.md](00-inference-lifecycle.md)：从 prompt 到 token 输出的完整生命周期。
2. [01-tokenization-and-chat-template.md](01-tokenization-and-chat-template.md)：tokenization、chat template、messages、tools。
3. [02-transformer-and-model-families.md](02-transformer-and-model-families.md)：Transformer 和主流大模型结构。
4. [03-prefill-decode-kv-cache.md](03-prefill-decode-kv-cache.md)：prefill、decode、KV cache、block。
5. [04-attention-and-memory.md](04-attention-and-memory.md)：attention 计算、PagedAttention、显存/NPU 内存视角。
6. [05-sampling-and-output.md](05-sampling-and-output.md)：采样、logits processor、detokenize、streaming、输出后处理。
7. [06-parallelism-basics.md](06-parallelism-basics.md)：TP、PP、DP、EP、CP、SP 的基础关系。
8. [07-moe-basics.md](07-moe-basics.md)：MoE、router、expert、token dispatch、负载不均衡基础。
9. [08-performance-metrics.md](08-performance-metrics.md)：TTFT、TPOT、吞吐、并发、队列延迟。
10. [09-quantization-brief.md](09-quantization-brief.md)：量化基础简述。

## 本阶段目标

- 能解释一次请求从 OpenAI API 入参到输出 token 的端到端过程。
- 能判断一个问题是否发生在 tokenizer/chat template 阶段。
- 能解释 prefill 和 decode 的计算差异，以及为什么 decode 常常受内存带宽影响。
- 能解释为什么 KV cache 决定长上下文和高并发能力。
- 能区分 dense 模型、MoE 模型、MHA/MQA/GQA/MLA 等常见结构差异。
- 能理解主要并行策略和 MoE 负载不均衡的基本来源。
- 能读懂 vLLM 调度、benchmark、性能文档里的核心术语。

## 推荐学习方式

先按顺序读完概念，再回到每篇文档末尾的“思考与探索”。如果有 NPU 环境，可以在读完 `00` 到 `05` 后跑一次最小 offline inference，把 prompt、token ids、sampling params 和输出打印出来。读完 `08` 后，再看一次 benchmark 输出，会更容易理解 TTFT、TPOT、吞吐之间的取舍。

## 阶段验收

完成本章后，建议用下面几个问题自测：

- 一个 chat completion 请求里的 `messages` 怎样变成模型输入 token ids？
- 为什么同一个 prompt 在两个模型上可能需要不同 chat template？
- Prefill 和 decode 的输入输出分别是什么？为什么它们适合不同的 batching 策略？
- KV cache 的 K/V 保存了什么？block table 解决了什么问题？
- Greedy、temperature、top-p 各自如何影响生成结果？
- TP、PP、DP、EP、CP 分别切分了什么？
- MoE 为什么可能出现 expert 热点？这和后续 EPLB 有什么关系？
- TTFT 和 TPOT 分别受哪些因素影响？

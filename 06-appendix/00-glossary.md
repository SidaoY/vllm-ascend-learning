# 术语表

这份术语表不是为了替代前面的章节，而是方便读代码和看日志时快速确认概念。每个术语只保留最小解释，想深入时可以跳到“首次阅读”对应章节。

| 术语 | 中文含义 | 简短解释 | 首次阅读 |
| --- | --- | --- | --- |
| Tokenization | 分词 | 把文本转换为 token ids 的过程。模型实际消费的是 token ids，不是原始字符串。 | [01-01](../01-llm-basics/01-tokenization-and-chat-template.md) |
| Chat template | 对话模板 | 把 messages、role、system prompt、工具调用等拼成模型期望输入格式的模板。 | [01-01](../01-llm-basics/01-tokenization-and-chat-template.md) |
| Checkpoint | 模型检查点 | 一组可加载的模型权重、配置、tokenizer 等文件。模型和 tokenizer 通常应来自同一 checkpoint。 | [01-01](../01-llm-basics/01-tokenization-and-chat-template.md) |
| Prefill | 预填充 | 对 prompt token 做一次或多次 forward，生成初始 KV cache，并得到首 token 所需 logits。 | [01-03](../01-llm-basics/03-prefill-decode-kv-cache.md) |
| Decode | 解码生成 | 在已有 KV cache 基础上增量生成后续 token。每轮通常只为每个请求推进少量 token。 | [01-03](../01-llm-basics/03-prefill-decode-kv-cache.md) |
| KV cache | K/V 缓存 | Transformer attention 中历史 token 的 Key/Value 缓存，用于避免 decode 时重复计算历史上下文。 | [01-03](../01-llm-basics/03-prefill-decode-kv-cache.md) |
| PagedAttention | 分页注意力 | vLLM 将 KV cache 按 block/page 管理，并通过 block table 让 attention 访问非连续 KV 的机制。 | [02-07](../02-vllm-foundation/07-attention-backends-and-pagedattention.md) |
| Block | KV block | KV cache 分配和管理的基本单位，包含固定数量 token 的 K/V 空间。 | [01-03](../01-llm-basics/03-prefill-decode-kv-cache.md) |
| Block table | 块表 | 记录请求的逻辑 token block 到物理 KV block 的映射。 | [05-01](../05-key-features/01-kv-cache-management.md) |
| Slot mapping | 槽位映射 | 描述本轮新 token 的 K/V 应该写入 KV cache 哪些位置。 | [05-01](../05-key-features/01-kv-cache-management.md) |
| Chunked prefill | 分块预填充 | 把长 prompt 的 prefill 拆成多轮执行，以平衡 TTFT、TPOT 和 token budget。 | [02-04](../02-vllm-foundation/04-scheduler.md) |
| Prefix caching | 前缀缓存 | 复用相同前缀已经计算好的 KV cache，减少重复 prefill。 | [05-01](../05-key-features/01-kv-cache-management.md) |
| Scheduler | 调度器 | 每个 engine step 决定推进哪些请求、推进多少 token、分配哪些 KV block。 | [02-04](../02-vllm-foundation/04-scheduler.md) |
| SchedulerOutput | 调度输出 | scheduler 和 worker 的边界对象，告诉 worker 本轮要执行哪些请求和相关 metadata。 | [02-04](../02-vllm-foundation/04-scheduler.md) |
| Attention backend | 注意力后端 | vLLM 中具体执行 attention 的后端实现。Ascend 侧包含 dense、MLA、SFA、FA3、CP 等路径。 | [02-07](../02-vllm-foundation/07-attention-backends-and-pagedattention.md) |
| MLA | Multi-head Latent Attention | 一类特殊 attention/KV 表示，常见于 DeepSeek 类模型。 | [01-04](../01-llm-basics/04-attention-and-memory.md) |
| SFA | Sparse Flash Attention | 面向 DSA/sparse attention 模型的稀疏 attention 路径，不是普通 Ascend attention 的统称。 | [03-04](../03-vllm-ascend-foundation/04-attention-and-kernels.md) |
| FA3 | Flash Attention 3 | vLLM Ascend 中一类 Flash Attention 3 路径，主要用于 attention 实现一致性，不等同于 KV cache 量化。 | [03-04](../03-vllm-ascend-foundation/04-attention-and-kernels.md) |
| Graph mode | 图模式 | 通过捕获和复用计算图降低动态执行开销。Ascend 侧通常关注 ACL/NPU graph。 | [02-08](../02-vllm-foundation/08-graph-mode.md) |
| ACL graph / NPU graph | Ascend 图模式 | Ascend NPU 上的图捕获和 replay 机制，关注 shape、metadata、unsupported op 和同步点。 | [02-08](../02-vllm-foundation/08-graph-mode.md) |
| Speculative decoding | 投机推理 | 用 proposer 提前提出候选 token，再由 target model 验证，接受率高时提升 decode 吞吐。 | [02-09](../02-vllm-foundation/09-speculative-decoding.md) |
| Proposer | 候选生成器 | 投机推理中产生 draft tokens 的模块，可以是 n-gram、EAGLE、MTP、suffix 等。 | [05-02](../05-key-features/02-speculative-decoding.md) |
| Draft token | 草稿 token | proposer 生成、等待目标模型验证的候选 token。 | [05-02](../05-key-features/02-speculative-decoding.md) |
| Rejection sampler | 拒绝采样器 | 投机推理中根据目标模型分布决定 draft tokens 接受或拒绝的组件。 | [05-02](../05-key-features/02-speculative-decoding.md) |
| EAGLE | EAGLE 投机方法 | 基于辅助模型或隐藏状态预测后续 token 的投机推理方法。 | [05-02](../05-key-features/02-speculative-decoding.md) |
| MTP | Multi Token Prediction | 模型自身一次预测多个后续 token 的能力，可用于投机推理。 | [05-02](../05-key-features/02-speculative-decoding.md) |
| TP | Tensor Parallelism | 按张量/权重维度切分模型计算。 | [01-06](../01-llm-basics/06-parallelism-basics.md) |
| PP | Pipeline Parallelism | 按模型层切分，把不同层放到不同设备。 | [01-06](../01-llm-basics/06-parallelism-basics.md) |
| DP | Data Parallelism | 多个副本处理不同请求，提高服务吞吐。 | [01-06](../01-llm-basics/06-parallelism-basics.md) |
| EP | Expert Parallelism | MoE 模型中按 expert 维度切分或放置专家。 | [01-07](../01-llm-basics/07-moe-basics.md) |
| CP | Context Parallelism | 按 sequence/context 维度切分长上下文，分担 attention 和 KV 压力。 | [05-03](../05-key-features/03-context-parallelism.md) |
| PCP | Prefill Context Parallelism | prefill 阶段的 CP，用于降低长 prompt 的 prefill 压力。 | [05-03](../05-key-features/03-context-parallelism.md) |
| DCP | Decode Context Parallelism | decode 阶段的 CP，用于分担长历史 KV 访问和容量压力。 | [05-03](../05-key-features/03-context-parallelism.md) |
| SP | Sequence Parallelism | 通常指按序列维度切分中间激活或计算的并行方式，具体含义要看上下文。 | [01-06](../01-llm-basics/06-parallelism-basics.md) |
| MoE | Mixture of Experts | 每层包含多个 expert，router 为 token 选择部分 expert 计算。 | [01-07](../01-llm-basics/07-moe-basics.md) |
| Expert | 专家 | MoE 中被 router 选择的子网络，通常是 FFN 类结构。 | [01-07](../01-llm-basics/07-moe-basics.md) |
| EPLB | Expert Parallel Load Balancing | MoE expert 负载均衡，调整 expert 放置、冗余或映射以缓解热点。 | [05-04](../05-key-features/04-load-balancing.md) |
| PD disaggregation | 预填充/解码分离 | 将 prefill 和 decode 放到不同实例或角色中，提高资源利用率。 | [03-05](../03-vllm-ascend-foundation/05-distributed-and-kv-transfer.md) |
| KV transfer | KV 传输 | PD 分离或 KV pool 场景中，把 prefill 产生的 KV 交给 decode 或共享存储。 | [03-05](../03-vllm-ascend-foundation/05-distributed-and-kv-transfer.md) |
| KV pool | KV 池 | 用共享存储或池化机制管理跨实例 KV cache。 | [05-01](../05-key-features/01-kv-cache-management.md) |
| HCCL | Huawei Collective Communication Library | Ascend 多卡/多机通信使用的重要 collective communication 后端。 | [03-05](../03-vllm-ascend-foundation/05-distributed-and-kv-transfer.md) |
| TTFT | Time To First Token | 从请求进入到首 token 输出的时间，常用于衡量 prefill 和排队体验。 | [01-08](../01-llm-basics/08-performance-metrics.md) |
| TPOT | Time Per Output Token | 输出 token 平均耗时，常用于衡量 decode 性能。 | [01-08](../01-llm-basics/08-performance-metrics.md) |
| ITL | Inter-token Latency | 相邻输出 token 的时间间隔，常用于观察流式输出抖动。 | [01-08](../01-llm-basics/08-performance-metrics.md) |

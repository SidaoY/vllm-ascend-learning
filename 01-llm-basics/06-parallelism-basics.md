# 并行基础

大模型推理经常需要多卡甚至多机。并行策略决定了模型权重如何切分、请求如何分发、attention 和 KV cache 如何组织、通信发生在哪里。初次学习先建立坐标系，后续看 CP、EPLB、PD 分离和多机部署时会轻松很多。

## 为什么需要并行

单卡推理会遇到几类限制：

- 模型权重太大，单卡放不下。
- KV cache 太大，单卡可服务并发有限。
- 单卡算力不足，吞吐不够。
- 单卡内存带宽不足，decode 慢。
- 多租户或大规模服务需要横向扩展。

不同并行方式解决不同瓶颈，没有一种策略能包打天下。

## Tensor Parallelism

Tensor Parallelism，简称 TP，把单层中的大矩阵计算切到多张卡上。每张卡保存权重的一部分，共同完成同一层计算。

直观理解：

- 切的是模型层内部的张量。
- 每个请求会同时使用 TP 组内所有设备。
- 层与层之间通常需要通信同步。
- 常用于单卡放不下模型或单卡算力不足的场景。

TP 的代价是通信。TP size 越大，不一定越快，因为通信可能抵消计算收益。

## Pipeline Parallelism

Pipeline Parallelism，简称 PP，把模型的不同层放到不同设备上。例如前 20 层在卡 0，后 20 层在卡 1。

直观理解：

- 切的是模型深度方向。
- 一个请求要依次经过 pipeline stages。
- 可以降低单卡权重占用。
- 需要处理 pipeline bubble 和 stage 间通信。

PP 对单请求延迟不一定友好，但能帮助部署更大的模型。

## Data Parallelism

Data Parallelism，简称 DP，复制多份完整模型或模型分片组，不同请求分配到不同 replica。

直观理解：

- 切的是请求流量。
- 每个 DP replica 处理不同请求。
- 横向扩展吞吐最直接。
- 需要负载均衡和全局路由。

在 serving 场景中，DP 常常和外部负载均衡、内部 coordinator、PD 分离一起出现。

## Expert Parallelism

Expert Parallelism，简称 EP，主要用于 MoE 模型。不同 experts 放在不同设备上，token 根据 router 结果被 dispatch 到对应 expert。

直观理解：

- 切的是 MoE experts。
- token 会跨设备发送到目标 expert。
- expert 热点会造成负载不均。
- 后续 EPLB/Swift Balancer 就是围绕它做优化。

EP 的核心挑战不是“能不能切”，而是“切完之后负载是否均衡、通信是否可控”。

## Context Parallelism

Context Parallelism，简称 CP，把长上下文序列维度切到多张卡上。它主要解决长序列 attention 和 KV cache 压力。

直观理解：

- 切的是上下文长度。
- 每张卡负责一段 context 或一部分 attention 工作。
- 对长上下文有价值。
- 需要 attention 通信和更复杂的 block table/metadata。

vLLM Ascend 的 CP 是团队关键特性之一，后续会专门讲 PCP、DCP、长序列和 CP attention。

## PD分离——Prefill/Decode Disaggregation

PD disaggregation 把 prefill 和 decode 分到不同实例或设备组。原因是二者资源特征不同：

- Prefill 更偏计算密集。
- Decode 更偏持续读 KV cache 和低延迟循环。

分离后，prefill 侧需要把生成的 KV cache transfer 给 decode 侧。这会引入 KV transfer、connector、load balance 等问题。

## 如何选择并行策略

选择策略时要看瓶颈：

| 目标或瓶颈 | 常见策略 |
| --- | --- |
| 模型权重单卡放不下 | TP、PP |
| 请求吞吐不够 | DP |
| MoE expert 太多或太大 | EP |
| 长上下文 KV/attention 压力大 | CP |
| prefill 和 decode 负载形态差异大 | PD disaggregation |
| 单卡 decode 利用率低 | batching、graph、DP、spec decode |

实际部署常常组合多种策略。

## Rank、World、Group

分布式代码中经常出现：

- rank：当前进程在某个通信组里的编号。
- world size：通信组大小。
- process group：一组需要互相通信的进程。
- local rank：同一节点内的设备编号。
- global rank：全局进程编号。

读并行代码时，第一件事是确认“这个 rank 属于哪个 group”。TP rank、DP rank、EP rank、CP rank 不是同一个概念。

## 和代码的连接

- vLLM 并行部署文档：`$PATH_TO_VLLM/docs/serving/parallelism_scaling.md`
- vLLM DP 文档：`$PATH_TO_VLLM/docs/serving/data_parallel_deployment.md`
- vLLM CP 文档：`$PATH_TO_VLLM/docs/serving/context_parallel_deployment.md`
- vLLM parallel state：`$PATH_TO_VLLM/vllm/distributed/parallel_state.py`
- Ascend parallel state：`$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/parallel_state.py`
- Ascend CP 文档：`$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/context_parallel.md`

## 章节边界

本章只讲并行策略坐标系。CP 并行会在关键特性章节展开，EPLB/负载均衡会在 MoE 和关键特性章节展开，具体多机部署和 CI 验证会放到开发实践章节。

## 思考与探索

1. 用一句话分别解释 TP、PP、DP、EP、CP 切分了什么。
2. 为什么 DP 需要负载均衡？
3. 一个长上下文模型遇到 KV cache OOM 时，除了减小并发，还可以考虑哪些并行或系统策略？

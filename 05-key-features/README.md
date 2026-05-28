# 05 Key Features

本部分深入关键特性。每个专题建议先讲业务/性能动机，再讲 vLLM 基础机制，最后讲 vLLM Ascend 实现和测试。具体测试、CI、性能观测和故障排查方法在 `../04-development-practice/README.md` 中先统一学习。

## 文档清单

- [00-feature-map.md](00-feature-map.md)：关键特性总览和相互关系。
- [01-kv-cache-management.md](01-kv-cache-management.md)：KV cache 管理、KV pool、offload、transfer。
- [02-speculative-decoding.md](02-speculative-decoding.md)：投机推理在 vLLM Ascend 中的实现。
- [03-context-parallelism.md](03-context-parallelism.md)：CP 并行、PCP/DCP、长序列。
- [04-load-balancing.md](04-load-balancing.md)：负载均衡、EPLB、Swift Balancer、外部 DP 负载均衡。

## 本阶段目标

- 能讲清每个关键特性解决的问题和主要收益。
- 能指出每个特性修改或扩展了 vLLM 的哪些层。
- 能找到核心代码、测试、用户文档、设计文档。
- 能独立定位常见问题。

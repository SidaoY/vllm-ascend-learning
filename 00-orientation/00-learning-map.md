# 学习路线图

## 写作目标

把整套学习资料拆成 5 个阶段：基础概念、vLLM 核心、vLLM Ascend 适配、开发实践、团队关键特性。每个阶段都要有明确的学习结果和探索任务。

## 建议结构

### 阶段一：LLM 推理基础

- 学会 tokenization/chat template、prefill/decode、KV cache、batching、sampling、并行策略、MoE、常见模型结构。
- 输出：能解释一次请求从文本到 token 输出的生命周期。
- 建议耗时：2-3 天。
- 必读：
  - [00-inference-lifecycle.md](../01-llm-basics/00-inference-lifecycle.md)
  - [01-tokenization-and-chat-template.md](../01-llm-basics/01-tokenization-and-chat-template.md)
  - [03-prefill-decode-kv-cache.md](../01-llm-basics/03-prefill-decode-kv-cache.md)
- 选读：
  - [02-transformer-and-model-families.md](../01-llm-basics/02-transformer-and-model-families.md)
  - [06-parallelism-basics.md](../01-llm-basics/06-parallelism-basics.md)
- 验收题：
  1. 用自己的话解释 prefill 和 decode 在计算形态上的核心差异。
  2. 为什么 KV cache 对长上下文推理至关重要？
  3. 同一个 `messages` 在两个模型上可能生成不同的 token ids，原因在哪里？

### 阶段二：vLLM 核心机制

- 学会 entrypoints、engine、scheduler、KV cache manager、attention backend、PagedAttention、graph mode、model runner 的关系。
- 输出：能沿代码追踪一个 OpenAI compatible request。
- 建议耗时：3-4 天。
- 必读：
  - [00-codebase-overview.md](../02-vllm-foundation/00-codebase-overview.md)
  - [01-entrypoints-and-api.md](../02-vllm-foundation/01-entrypoints-and-api.md)
  - [02-request-flow-v1.md](../02-vllm-foundation/02-request-flow-v1.md)
  - [03-engine-and-worker.md](../02-vllm-foundation/03-engine-and-worker.md)
  - [04-scheduler.md](../02-vllm-foundation/04-scheduler.md)
- 选读：
  - [05-kv-cache-manager.md](../02-vllm-foundation/05-kv-cache-manager.md)
  - [07-attention-backends-and-pagedattention.md](../02-vllm-foundation/07-attention-backends-and-pagedattention.md)
- 验收题：
  1. 画出"API 入口 → engine → scheduler → worker → model runner → output processor"的数据流。
  2. scheduler output 为什么不是"模型输入"，而只是"告诉 worker 本轮算什么"？
  3. 如果线上 TTFT 突然变差，你会从 scheduler 的哪些输入开始排查？

### 阶段三：vLLM Ascend 适配

- 学会 NPU platform 注册、patch 机制、worker/model runner、Ascend attention、分布式通信。
- 输出：能判断一个功能应该改上游 vLLM、vLLM Ascend 原生代码，还是临时 patch。
- 建议耗时：3-4 天。
- 必读：
  - [00-architecture-overview.md](../03-vllm-ascend-foundation/00-architecture-overview.md)
  - [01-platform-registration.md](../03-vllm-ascend-foundation/01-platform-registration.md)
  - [02-patch-mechanism.md](../03-vllm-ascend-foundation/02-patch-mechanism.md)
  - [03-worker-and-model-runner.md](../03-vllm-ascend-foundation/03-worker-and-model-runner.md)
- 选读：
  - [04-operator-system.md](../03-vllm-ascend-foundation/04-operator-system.md)
  - [05-attention-backend.md](../03-vllm-ascend-foundation/05-attention-backend.md)
  - [06-distributed-and-kv-transfer.md](../03-vllm-ascend-foundation/06-distributed-and-kv-transfer.md)
- 验收题：
  1. vLLM Ascend 可以复用 OpenAI entrypoint 和 engine 的核心原因是什么？
  2. 一个改动需要同时改 platform 和 worker 时，patch 应该放在哪个阶段？为什么？
  3. 为什么 block table 错一次，可能影响后续多轮 decode？

### 阶段四：开发实践

- 学会测试、CI/nightly workflow、性能观测和调优、故障排查。
- 输出：能为一次代码改动选择合适验证路径，并能定位常见失败。
- 建议耗时：2-3 天。
- 必读：
  - [00-development-workflow.md](../04-development-practice/00-development-workflow.md)
  - [01-testing-ci-and-nightly.md](../04-development-practice/01-testing-ci-and-nightly.md)
- 选读：
  - [02-observability-and-performance-tuning.md](../04-development-practice/02-observability-and-performance-tuning.md)
  - [03-debugging-playbook.md](../04-development-practice/03-debugging-playbook.md)
- 验收题：
  1. 一个 patch 改了 scheduler 行为，至少需要跑哪些测试验证？
  2. Nightly 里同一个 case 在 A2 通过、A3 失败，优先检查什么？
  3. 如何设计一个 benchmark 来验证 spec decode 的提升，同时排除其他变量干扰？

### 阶段五：团队关键特性

- 深入 KV cache 管理、投机推理、CP 并行、负载均衡。
- 输出：能独立阅读相关代码、补测试、定位常见问题。
- 建议耗时：3-5 天。
- 必读：
  - [00-feature-map.md](../05-key-features/00-feature-map.md)
  - [01-kv-cache-management.md](../05-key-features/01-kv-cache-management.md)
- 选读（按关注方向选择）：
  - 长上下文方向：[03-context-parallelism.md](../05-key-features/03-context-parallelism.md)
  - Decode 性能方向：[02-speculative-decoding.md](../05-key-features/02-speculative-decoding.md)
  - MoE 方向：[04-load-balancing.md](../05-key-features/04-load-balancing.md)
- 验收题：
  1. PagedAttention 没有减少每个 token 的 K/V payload，为什么能显著提升服务容量？
  2. PD 分离中，如果 prefill 和 decode 的 block size 不一致，会出现什么问题？
  3. CP、投机推理、PD 分离同时开启时，最容易出错的共享边界在哪里？

## 新人 Onboarding Checklist

建议新人在前两周按顺序完成以下事项：

**第一周：环境和概念**

- [ ] 阅读 [00-orientation](../00-orientation/README.md)，理解学习路线和仓库关系。
- [ ] 完成环境搭建，跑通最小离线推理（参考 [02-environment-and-workflow.md](02-environment-and-workflow.md)）。
- [ ] 完成阶段一（LLM 推理基础），能用自己的话解释 prefill/decode/KV cache。
- [ ] 读完阶段二的必读文档，建立 vLLM 架构地图。

**第二周：代码和 Ascend 适配**

- [ ] 完成阶段二余下内容，能在代码中追踪一条请求。
- [ ] 完成阶段三的必读文档，理解 vLLM Ascend 如何接入 vLLM。
- [ ] 阅读至少 2 个已有 patch，理解它们为什么存在。
- [ ] 阅读阶段四的开发工作流文档。

**第三周及之后：深入和实践**

- [ ] 完成阶段四余下内容，参与一次 PR review 或 CI 失败排查。
- [ ] 选择一个关键特性方向深入学习（KV cache / spec decode / CP / EPLB）。
- [ ] 完成至少一次 benchmark 实验，使用 [实验记录模板](../06-appendix/03-experiment-template.md) 记录。
- [ ] 完成一次完整的改动流程：定位问题 → 写代码 → 补测试 → 跑 CI → 记录结果。

## 常用搜索关键词

日常阅读代码时，这些关键词可以帮助快速定位：

| 主题 | 关键词 |
| --- | --- |
| 请求入口 | `OpenAIServing`, `AsyncLLM`, `EngineCoreRequest`, `SamplingParams` |
| 调度 | `SchedulerOutput`, `token budget`, `waiting`, `running`, `chunked prefill` |
| KV Cache | `KVCacheManager`, `block table`, `slot mapping`, `prefix cache`, `KVConnector` |
| Worker | `execute_model`, `model runner`, `input batch`, `attention metadata` |
| Patch | `adapt_patch`, `platform patch`, `worker patch`, `monkey patch` |
| Graph | `graph capture`, `replay`, `cudagraph_capture_sizes`, `ACL graph` |
| Attention | `MLA`, `SFA`, `FA3`, `block_size`, `slot_mapping`, `aclnn` |
| Spec Decode | `proposer`, `draft token`, `acceptance rate`, `rejection sampler` |
| CP | `prefill_context_parallel_size`, `decode_context_parallel_size`, `PCP`, `DCP` |
| EPLB | `expert_map`, `expert_heat`, `DYNAMIC_EPLB`, `Swift Balancer` |
| CI/Nightly | `VLLM_COMMIT`, `nightly`, `smart e2e`, `profile`, `msprobe` |
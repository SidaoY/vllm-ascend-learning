# 学习路线图

## 写作目标

把整套学习资料拆成 5 个阶段：基础概念、vLLM 核心、vLLM Ascend 适配、开发实践、团队关键特性。每个阶段都要有明确的学习结果和探索任务。

## 建议结构

1. 阶段一：LLM 推理基础
   - 学会 tokenization/chat template、prefill/decode、KV cache、batching、sampling、并行策略、MoE、常见模型结构。
   - 输出：能解释一次请求从文本到 token 输出的生命周期。
2. 阶段二：vLLM 核心机制
   - 学会 entrypoints、engine、scheduler、KV cache manager、attention backend、PagedAttention、graph mode、model runner 的关系。
   - 输出：能沿代码追踪一个 OpenAI compatible request。
3. 阶段三：vLLM Ascend 适配
   - 学会 NPU platform 注册、patch 机制、worker/model runner、Ascend attention、分布式通信。
   - 输出：能判断一个功能应该改上游 vLLM、vLLM Ascend 原生代码，还是临时 patch。
4. 阶段四：开发实践
   - 学会测试、CI/nightly workflow、性能观测和调优、故障排查。
   - 输出：能为一次代码改动选择合适验证路径，并能定位常见失败。
5. 阶段五：团队关键特性
   - 深入 KV cache 管理、投机推理、CP 并行、负载均衡。
   - 输出：能独立阅读相关代码、补测试、定位常见问题。

## 待补内容

- 每阶段建议耗时。
- 每阶段必读文档和选读文档。
- 每阶段验收题。
- 新人 onboarding checklist。

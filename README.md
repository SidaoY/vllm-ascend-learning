# vLLM Ascend 新员工学习路线

这套资料面向刚接触大模型推理、vLLM 和 vLLM Ascend 的同学，目标是从基础概念逐步过渡到真实代码，再进入团队关键特性的设计与开发实践。

当前阶段只规划目录结构和每章写作范围，不展开正文。后续建议按编号顺序补全文档，每篇控制在一个明确主题内，并在文末保留“代码阅读入口”和“思考与探索”。

## 学习顺序

1. [00-orientation](00-orientation/README.md)
   - 学习路线、仓库关系、环境和阅读方法。
2. [01-llm-basics](01-llm-basics/README.md)
   - LLM 推理基础、tokenization、chat template、Transformer/MoE、prefill/decode、KV cache、sampling、并行、性能指标、量化基础。
3. [02-vllm-foundation](02-vllm-foundation/README.md)
   - vLLM 代码结构、请求处理流程、调度算法、KV cache 管理、模型执行、attention backend、PagedAttention、图模式。
4. [03-vllm-ascend-foundation](03-vllm-ascend-foundation/README.md)
   - vLLM Ascend 的插件式运作方式、NPU 平台适配、worker/model runner、patch 机制、算子、分布式和 KV transfer。
5. [04-development-practice](04-development-practice/README.md)
   - 具体开发实践：开发流程、测试、CI/nightly workflow、性能观测和调优、故障排查。
6. [05-key-features](05-key-features/README.md)
   - 团队关键特性：KV cache 管理、投机推理、CP 并行、负载均衡。
7. [06-appendix](06-appendix/README.md)
   - 术语表、代码导航、实验记录模板、资料索引。

## 已纳入大纲的补充主题

以下主题已经被纳入相应章节，后续写正文时按这些位置展开。

- Tokenization 和 chat template：放在 `01-llm-basics/01-tokenization-and-chat-template.md`。
- Sampling 和输出后处理：放在 `01-llm-basics/05-sampling-and-output.md`。
- 并行策略：放在 `01-llm-basics/06-parallelism-basics.md`。
- MoE 基础：放在 `01-llm-basics/07-moe-basics.md`，为后续负载均衡/EPLB 铺垫。
- 量化：放在 `01-llm-basics/09-quantization-brief.md` 做基础简述。
- Attention backend 和 PagedAttention：放在 `02-vllm-foundation/07-attention-backends-and-pagedattention.md`。
- 图模式：放在 `02-vllm-foundation/08-graph-mode.md`，后续再衔接 Ascend ACL/NPU graph。
- 测试、CI 和 nightly workflow：放在 `04-development-practice/01-testing-ci-and-nightly.md`。
- 性能观测和调优：放在 `04-development-practice/02-observability-and-performance-tuning.md`。
- 故障排查手册：放在 `04-development-practice/03-debugging-playbook.md`。

## 仍建议后续关注的横向主题

这些主题不一定单独成章，但写正文时应在相关章节中穿插。

- Prefix caching 和 disaggregated prefill：它们和 KV 复用、PD 分离、KV transfer、负载均衡经常一起出现。
- Patch 生命周期和上游同步策略：新人需要知道什么时候该 patch、什么时候该推动上游修复、patch 如何退出。
- 多模型、多模态、tool call、reasoning output 对请求入口和输出后处理的影响。
- 上游 vLLM 版本变化对 vLLM Ascend patch 和 CI 的影响。

## 路径约定

为了让文档在不同学习者的机器上都能使用，本文档中的代码路径不假设固定目录结构，而是使用以下占位变量：

- `$PATH_TO_VLLM`：本地 vLLM 仓库根目录，例如 `/home/yht/code/ascend/vllm`。
- `$PATH_TO_VLLM_ASCEND`：本地 vLLM Ascend 仓库根目录，例如 `/home/yht/code/ascend/vllm-ascend`。
- `$PATH_TO_LEARNING`：本学习资料根目录，例如 `/home/yht/code/ascend/vllm-ascend-learning`。

这些变量只是文档里的路径占位符，不要求学习者真的在 shell 中 `export`。例如：

- `$PATH_TO_VLLM/vllm/entrypoints/openai`
- `$PATH_TO_VLLM/vllm/v1/core/sched/scheduler.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/model_runner_v1.py`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/patch.md`

学习资料内部跳转仍然使用相对 Markdown 链接，例如 [01-llm-basics](01-llm-basics/README.md)。

## 仓库关系

- `$PATH_TO_VLLM`：上游 vLLM 代码和文档，重点关注 `vllm/v1`、`vllm/entrypoints`、`vllm/model_executor`、`docs/design`、`docs/features`。
- `$PATH_TO_VLLM_ASCEND`：Ascend 后端适配代码和文档，重点关注 `vllm_ascend/platform.py`、`vllm_ascend/worker`、`vllm_ascend/patch`、`vllm_ascend/attention`、`vllm_ascend/distributed`、`vllm_ascend/eplb`、`docs/source`。

## 编写约定

- 每篇文档尽量遵循“为什么需要、核心概念、代码入口、关键流程、常见坑、思考与探索”的结构。
- 指向 vLLM 或 vLLM Ascend 的代码路径使用 `$PATH_TO_VLLM` / `$PATH_TO_VLLM_ASCEND` 占位符。
- 学习资料内部链接使用相对 Markdown 链接，方便在文档站点或 IDE 中点击跳转。
- 对复杂流程优先画 Mermaid 图，文字只解释关键分叉。
- 每个专题至少给出一个可运行命令、一个推荐断点或日志位置、一个常见问题。
- 面向新人时先解释稳定抽象，再提醒版本差异；避免一上来陷入历史包袱。

# 02 vLLM Foundation

本部分开始进入 vLLM 代码。重点不是记住某个函数当前叫什么，而是建立一张稳定的架构地图：请求从入口层进入 engine，scheduler 决定每个 step 计算哪些 token，KV cache manager 管理 block，worker/model runner 在设备侧执行模型，attention backend 根据 metadata 读写 KV cache。

vLLM 发展很快，具体类名、方法名和文件组织会变化。本章会尽量使用目录级入口、核心对象和数据流来讲解；本地代码用于校准当前实现，不把某个方法或行号作为长期依赖。

## 文档清单

1. [00-codebase-overview.md](00-codebase-overview.md)：vLLM 代码结构和推荐阅读顺序。
2. [01-entrypoints-and-api.md](01-entrypoints-and-api.md)：OpenAI API、offline API、CLI 入口。
3. [02-request-flow-v1.md](02-request-flow-v1.md)：V1 请求处理主流程。
4. [03-engine-and-worker.md](03-engine-and-worker.md)：engine、executor、worker、model runner 的职责边界。
5. [04-scheduler.md](04-scheduler.md)：调度算法、token budget、prefill/decode/chunked prefill。
6. [05-kv-cache-manager.md](05-kv-cache-manager.md)：KV cache manager、block pool、prefix caching。
7. [06-model-executor-and-attention.md](06-model-executor-and-attention.md)：模型加载、模型执行、attention layer。
8. [07-attention-backends-and-pagedattention.md](07-attention-backends-and-pagedattention.md)：attention backend、PagedAttention、attention metadata。
9. [08-graph-mode.md](08-graph-mode.md)：graph capture 的收益、约束和代码位置。

## 本阶段目标

- 能从 API 入口追到 engine、scheduler 和 worker。
- 能解释 scheduler 每轮到底决定了什么。
- 能解释 KV cache manager、block table 和 attention metadata 的关系。
- 能解释 attention backend / PagedAttention 为什么是 vLLM 高吞吐推理的关键。
- 能理解 graph mode 的收益和约束。
- 能判断一个问题更可能发生在 entrypoint、engine、scheduler、worker、model executor、attention backend 还是 output processor。

## 阅读方式

建议采用“先主干，后细节”的方式：

1. 先读 `00` 到 `03`，只建立请求主流程。
2. 再读 `04` 和 `05`，理解 scheduler 与 KV cache 的耦合。
3. 再读 `06` 和 `07`，理解模型执行和 attention backend。
4. 最后读 `08`，理解 graph mode 这种横向优化。

读本地代码时，优先使用 `rg` 查目录、类名和对象名。不要一开始就追进每个分支、每个模型文件、每个 backend 特化；vLLM 的代码量很大，先保持主线清楚，比记住某个局部实现更重要。

## 推荐观察点

- 请求入口：OpenAI server 或 offline `LLM` API 如何构造请求。
- Engine 层：请求如何进入 engine core，输出如何回到 output processor。
- Scheduler 层：每个 step 调度了哪些请求、每个请求调度了多少 token。
- KV cache 层：请求持有哪些 block，哪些 block 被命中、分配、释放。
- Worker 层：调度结果如何变成模型输入、attention metadata 和采样输出。
- Output 层：token ids 如何 detokenize、stream、转换成 API schema。

## 和 Ascend 的关系

本章只讲 vLLM 主仓库的基础机制。后续 vLLM Ascend 章节会解释：Ascend 如何注册 platform、替换或扩展 worker/model runner、实现 NPU attention backend、处理 patch、适配 graph、KV transfer 和关键特性。

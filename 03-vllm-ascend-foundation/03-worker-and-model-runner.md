# Worker 与 Model Runner

vLLM Ascend 的 worker/model runner 是 scheduler output 和 NPU 执行之间的桥。它负责把 vLLM 通用调度结果转换成 NPU 可执行的 input batch、block table、attention metadata，并调用模型 forward 和 sampler。

## NPU Worker 的职责

NPU worker 通常负责：

- 初始化 NPU 设备和分布式环境。
- 加载模型权重。
- profile 可用内存，并初始化 KV cache。
- 管理 model runner 生命周期。
- 接收 executor 传来的 scheduler output。
- 处理请求完成、状态释放、KV connector 等 worker 侧状态。
- 返回 model runner output。

Worker 更关注“设备和执行环境”，model runner 更关注“每个 step 的输入组织和模型执行”。

## Model Runner 的职责

Model runner 负责：

- 缓存新请求和已有请求的本地状态。
- 组织 input ids、positions、seq lens、query lens。
- 维护 NPU input batch。
- 管理 block table 和 slot mapping。
- 构造 attention metadata。
- 处理 graph/eager 路径选择。
- 调用模型 forward。
- 执行 sampler，生成 token 和 logprobs。
- 处理 KV transfer、spec decode、多模态、LoRA 等扩展信息。

如果 scheduler output 是“本轮要做什么”，model runner 就是“把它变成 NPU 上能做的事情”。

## 数据流

| 阶段 | 组件 | 输入来源 | 产物 |
| --- | --- | --- | --- |
| 1 | NPU Model Runner | SchedulerOutput | Request states, NPU input batch, Block table, Attention metadata |
| 2 | Model forward | NPU input batch, Block table, Attention metadata | Logits / Hidden States |
| 3 | Sampler | Model forward 输出 | ModelRunnerOutput |

这条链路里，最常见的 bug 来自状态不一致：scheduler 认为请求推进到了某个 token，worker 本地状态、block table 或 attention metadata 却没有同步到同一视图。

## Input Batch

Input batch 是 model runner 对当前 step 的执行描述。它通常包含：

- 请求 id 和 batch 内索引。
- 本轮 token ids。
- positions。
- 每个请求本轮 scheduled token 数。
- sequence length。
- 多模态或 encoder 输入索引。
- sampling、structured output、spec decode 相关信息。

Input batch 的目标是让后续模型 forward 和 attention backend 不再关心上层请求队列细节。

## Block Table

Block table 记录请求逻辑 KV block 和物理 KV cache block 的映射。Ascend 侧 block table 不只是 Python 状态，它最终会影响 NPU attention kernel 如何读写 KV cache。

排查 block table 问题时，要同时看：

- scheduler 是否分配了正确 block。
- worker 是否应用了 staged update。
- slot mapping 是否对应本轮写入 token。
- attention backend 是否按相同 layout 理解 KV cache。
- 请求结束或 abort 后是否释放本地状态。

## Graph 对 Model Runner 的影响

Graph mode 要求 shape、buffer 和 metadata 更新方式更稳定。因此 model runner 需要处理：

- dummy input 或 padding。
- capture size 和实际 batch 的映射。
- graph replay 前后 metadata 更新。
- 不支持 graph 的路径回退 eager。
- attention backend、sampling、KV cache 和 graph 的约束一致性。

Graph 问题不一定发生在 graph 代码本身，也可能是 model runner 准备的 metadata 不能满足 graph replay 的约束。

## 硬件特化路径

部分硬件、模型或算子组合会有特化路径。阅读这些路径时先判断：

- 它解决的是硬件能力差异、性能问题还是兼容问题。
- 它是否只影响特定模型或 dtype。
- 它是否有对应测试和文档。
- 它是否需要和 platform、attention backend 或 patch 配合。

## 代码入口

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/worker.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/model_runner_v1.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/npu_input_batch.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/block_table.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/pcp_utils.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/_310p`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md`

建议搜索关键词：`NPUWorker`、`model runner`、`input batch`、`block table`、`slot mapping`、`attention metadata`。

## 常见问题定位

- shape mismatch：先看 input batch、positions、seq lens、attention metadata。
- 输出持续错误：看 block table 和 slot mapping 是否在某步后错位。
- graph 失败：看 batch shape、dummy input、metadata 更新和不支持 op。
- 多卡行为异常：看 worker 分布式初始化和 rank group。
- KV transfer 异常：看 scheduler/worker 两侧 connector metadata 是否一致。

## 思考与探索

1. 为什么 worker 需要缓存请求状态，而不是每个 step 都从 engine 传完整请求？
2. Block table 错一次，为什么可能影响后续多轮 decode？
3. 如果一个 bug 只在 graph mode 出现，你会如何比较 eager 和 graph 路径的 input batch？

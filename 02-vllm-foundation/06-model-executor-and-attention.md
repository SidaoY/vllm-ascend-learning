# Model Executor 与 Attention Layer

前面几章讲的是请求如何被调度。本章进入模型执行视角：vLLM 如何根据模型配置加载模型，如何组织 layer，attention layer 如何接入 backend，以及模型 forward 的输出如何进入 sampler。

本章只讲 model executor 的位置和职责。PagedAttention、block table 和 attention metadata 的细节放到下一章。

## Model Executor 负责什么

Model executor 可以理解为“把 Hugging Face 风格模型变成高吞吐推理模型”的执行基础设施。它负责：

- 根据模型配置找到对应的模型类。
- 加载权重，并处理 sharded weight、量化权重、LoRA 等变体。
- 初始化 attention layer、MLP、MoE、norm、embedding、lm head 等模块。
- 在 forward 中接收 input ids、positions、KV cache、attention metadata。
- 输出 hidden states 或 logits，供后续 sampler 使用。
- 与 torch compile、custom op、graph mode 等执行优化对接。

初次学习时，先不要从某一个模型文件里迷路。模型文件很多，但它们大多遵循类似结构：embedding -> 多层 decoder layer -> norm -> lm head。

## 从模型配置到模型类

模型加载大致经历三步：

1. 读取模型配置，判断 architecture、model type、任务类型、dtype、max model length 等。
2. 在 model registry 中找到 vLLM 支持的模型类。
3. 使用 model loader 加载权重，并把权重映射到 vLLM 模型结构中。

这里最容易出问题的是“checkpoint 的结构”和“vLLM 模型实现”不匹配。例如模型家族新增字段、权重命名变化、量化格式变化、tie embedding 行为变化，都可能让加载阶段失败或精度异常。

## Attention Layer 的位置

在模型文件里，attention layer 通常是 decoder layer 的核心组件之一。它不是直接写死某个 kernel，而是通过 vLLM 的 attention 抽象接入 backend。

可以把 attention layer 分成三层理解：

- 模型层：负责 q/k/v/o projection、RoPE、head 数量、模型结构参数。
- vLLM attention 抽象层：负责选择 backend、管理 KV cache spec、统一 forward 接口。
- backend/kernel 层：负责真正执行 attention，处理 block table、slot mapping、paged KV cache 和平台特化。

这种分层让 vLLM 可以在不同硬件和模型结构上复用上层模型代码，同时替换底层 attention 实现。

## Model Runner 和 Model Executor 的关系

Model runner 负责“本轮要算什么”和“输入张量怎么准备”；model executor 负责“模型结构怎么执行”。

| 阶段 | 组件 | 输入来源 | 产物 |
| --- | --- | --- | --- |
| 1 | Model Runner | SchedulerOutput | Input Batch, Attention Metadata |
| 2 | Model Executor | Input Batch, Attention Metadata | Logits / Hidden States |
| 3 | Sampler | Logits / Hidden States | ModelRunnerOutput |

如果 input ids、positions、block table 或 attention metadata 错了，模型 executor 可能只是“按错误输入正确执行”。所以排查模型输出异常时，要同时看 model runner 准备的数据和 model executor 的 forward。

## Sampling 在执行链路中的位置

模型 forward 的直接输出通常是 logits。Sampler 根据 sampling params、logits processor、structured output 约束等选择下一 token。

vLLM 会尽量把高频 sampling 工作放在设备侧或高效路径中，但输出后处理、detokenization、stop string 等仍在 engine/output processor 层发挥作用。

因此要区分：

- logits 是否正确：更靠近 model executor / attention / weights。
- token 采样是否符合参数：更靠近 sampler / logits processor。
- 文本输出是否符合 API：更靠近 detokenizer / output processor / entrypoint。

## Custom Op、Compile 和 Graph 的位置

vLLM 的执行优化可能出现在多个层次：

- model executor layer 使用 custom op 或 fused op。
- attention backend 使用特定 kernel。
- torch compile 优化模型片段。
- graph capture 降低重复 decode 的启动开销。

这些优化都可能影响性能和可调试性。新同学读代码时先理解 eager 路径，再看 compile/graph/custom op，否则很容易被优化路径绕晕。

## 代码入口

- `$PATH_TO_VLLM/vllm/model_executor/models`
- `$PATH_TO_VLLM/vllm/model_executor/models/registry.py`
- `$PATH_TO_VLLM/vllm/model_executor/model_loader`
- `$PATH_TO_VLLM/vllm/model_executor/layers`
- `$PATH_TO_VLLM/vllm/model_executor/layers/attention`
- `$PATH_TO_VLLM/vllm/v1/worker`
- `$PATH_TO_VLLM/vllm/v1/attention`
- `$PATH_TO_VLLM/docs/design/attention_backends.md`
- `$PATH_TO_VLLM/docs/design/custom_op.md`
- `$PATH_TO_VLLM/docs/design/torch_compile.md`

建议搜索关键词：`ModelRegistry`、`model_loader`、`Attention`、`get_attn_backend`、`logits processor`、`sampler`、`custom op`。

## 常见问题定位

- 模型加载时报 missing key / unexpected key：优先看 model loader、模型类和 checkpoint 权重命名。
- 只有某个模型家族输出异常：优先看该模型文件、RoPE、attention 结构、lm head、特殊 patch。
- 所有模型都慢：优先看 backend、graph、batch 形态、scheduler，而不是某个模型文件。
- logits 正常但文本异常：优先看 sampler、detokenizer、output processor。

## 思考与探索

1. 为什么 attention layer 不应该直接绑定某个硬件 kernel？
2. 如果模型加载成功但输出完全不对，你会按什么顺序排查 tokenizer、weights、attention、sampler？
3. 在 `$PATH_TO_VLLM/vllm/model_executor/models` 中选择一个 familiar 模型，只看它的 decoder layer 结构，标出 attention 和 MLP 的位置。

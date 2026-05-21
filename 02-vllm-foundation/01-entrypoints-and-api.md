# Entrypoints 与 API

vLLM 有多种入口：OpenAI compatible server、offline `LLM` API、CLI、batch serving 等。入口层的职责不是执行模型，而是把用户输入转换成 engine 能理解的请求，并在输出回来后按协议组装响应。

## 入口层做什么

入口层通常负责这些事情：

- 解析 HTTP / CLI / Python API 参数。
- 校验模型名、请求字段、streaming 配置和 sampling 参数。
- 对 chat 请求应用 chat template。
- 调 tokenizer，把文本、多模态占位和工具调用信息转换成模型输入。
- 构造 sampling params、structured output 约束、tool parser 配置。
- 把请求交给 engine。
- 将 engine 输出转换成 OpenAI compatible schema 或 offline API 返回值。

入口层越复杂，越容易出现“模型其实没错，但请求转换错了”的问题。排查 API 行为时，要先区分是模型生成问题、tokenization/chat template 问题，还是 response schema 组装问题。

## Server 与 Offline 的差异

OpenAI server 面向在线服务：

- 请求来自 HTTP。
- 需要处理 streaming、取消请求、并发、超时、鉴权、metrics。
- 输出要符合 OpenAI compatible schema。
- chat completion、completion、responses、embedding、rerank 等接口会走不同 serving 逻辑。

Offline `LLM` API 面向本地批处理：

- 请求来自 Python 调用。
- 通常没有 HTTP streaming 和 OpenAI schema 的压力。
- 更适合做最小复现、单机调试、benchmark 前的功能验证。

很多问题可以先用 offline API 复现。如果 offline 正常、server 异常，优先怀疑入口层、schema、streaming、chat template 或输出后处理。

## 字段如何进入内部请求

| 外部字段 | 入口层处理 | 后续影响 |
| --- | --- | --- |
| `model` | 校验 served model name | 路由到对应 engine / 模型配置 |
| `messages` | chat template + tokenizer | prompt token ids、tool call 格式 |
| `prompt` | tokenizer 或直接使用 token ids | prompt token ids |
| `temperature/top_p/top_k` | 构造 sampling params | sampler 行为 |
| `max_tokens` | 构造 sampling params | stop condition、scheduler 目标长度 |
| `stop` / `stop_token_ids` | 构造停止条件 | output processor 和 detokenizer |
| `stream` | 选择 streaming 输出路径 | chunk 生成和 finish reason |
| `response_format` / grammar | structured output 配置 | logits processor / grammar mask |
| `tools` / `tool_choice` | tool parser / template 参数 | 输入 prompt 和输出解析 |

## Streaming 为什么更复杂

Non-streaming 可以等请求完成后一次性组装响应。Streaming 需要边生成边返回，因此入口层和输出层要维护更多状态：

- 已发送到客户端的文本片段。
- stop string 是否跨 token 边界。
- tool call arguments 是否是合法的增量 JSON 片段。
- finish reason 应该在哪个 chunk 返回。
- usage 是否需要在最后一个 chunk 返回。

这也是为什么 streaming bug 经常看起来像模型问题，实际却发生在 detokenization、stop string 或 API schema 组装阶段。

## 代码入口

- `$PATH_TO_VLLM/vllm/entrypoints/openai`
- `$PATH_TO_VLLM/vllm/entrypoints/cli`
- `$PATH_TO_VLLM/vllm/entrypoints/serve`
- `$PATH_TO_VLLM/vllm/v1/engine`
- `$PATH_TO_VLLM/docs/serving/openai_compatible_server.md`
- `$PATH_TO_VLLM/docs/serving/offline_inference.md`

这些路径用于建立入口层地图。具体类名和函数名可能会随 vLLM 版本调整，建议使用 `rg "ChatCompletion"`、`rg "SamplingParams"`、`rg "stream"` 这类关键词定位当前实现。

## 常见问题定位

- prompt token 数和预期不同：先看 tokenizer、chat template、工具调用模板和多模态占位。
- sampling 参数没有生效：先确认入口层构造的 sampling params，而不是直接怀疑 sampler。
- streaming 内容多发或少发：先看 stop string、detokenizer、streaming chunk 组装。
- usage 不正确：先区分 prompt tokens、completion tokens、reasoning tokens 的统计来源。
- OpenAI client 报 schema 错：先看 response object 的字段，而不是模型输出文本。

## 思考与探索

1. 为什么同一个 `messages` 在两个模型上可能被渲染成不同 prompt？
2. 如果 offline API 输出正常，但 OpenAI server streaming 输出异常，你会先看哪一层？
3. 用 `rg "chat_template"` 在 `$PATH_TO_VLLM/vllm/entrypoints` 下搜索，观察 chat template 配置可能从哪些地方进入。

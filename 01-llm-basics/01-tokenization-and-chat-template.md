# Tokenization 与 Chat Template

LLM 并不直接理解 Python 字符串，也不天然知道 OpenAI API 里的 `system`、`user`、`assistant` 是什么。模型真正接收的是一串 token ids。Tokenization 和 chat template 就是把“人类可读的请求”变成“模型训练时见过的输入格式”的过程。

## Tokenizer 做什么

Tokenizer 负责文本和 token ids 之间的转换。

```text
文本:   Hello world
tokens: [Hello, world]       只是示意，真实切分取决于 tokenizer
ids:    [9906, 1917]         只是示意，不同模型词表不同
```

常见 tokenizer 会把词、子词、符号、空格、中文字符、特殊控制符切成 token。一个 token 不一定等于一个词，也不一定等于一个字符。中文、代码、标点、空格密集文本的 token 数都可能和直觉不同。

推理系统最关心 tokenizer 的几个结果：

- `input_ids`：模型输入 token ids。
- `prompt_token_count`：prompt token 数，影响 prefill 成本。
- `special tokens`：BOS、EOS、PAD、role tokens、tool call tokens。
- `decode` 行为：token ids 转回文本时如何处理空格、特殊 token、非法字节。

## Chat Template 做什么

Chat API 通常传入 messages：

```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Explain KV cache in one sentence."}
]
```

但模型训练时不一定见过这种 JSON。Chat template 会把 messages 拼成某个模型约定的 prompt，例如：

```text
<system>
You are a helpful assistant.
</system>
<user>
Explain KV cache in one sentence.
</user>
<assistant>
```

上面只是示意。不同模型的真实 template 差异很大。有的使用 `<|im_start|>` 和 `<|im_end|>`，有的使用 `[INST]`，有的把 system prompt 合并到第一轮 user，有的需要额外 assistant generation prompt。

## 为什么 Template 很重要

模型是在某种对话格式上训练或对齐的。template 不匹配时，模型可能出现以下问题：

- 回答带出 role 标记。
- 不遵守 system prompt。
- 工具调用 JSON 格式错。
- 多轮对话上下文混乱。
- 模型提前输出 EOS。
- reasoning content 和 final answer 混在一起。

## Special Tokens

Special tokens 是词表中有特殊含义的 token。常见类型包括：

- BOS：begin of sequence，序列开始。
- EOS：end of sequence，序列结束。
- PAD：padding，用于对齐 batch。
- role token：标识 system/user/assistant/tool。
- image/audio placeholder：多模态输入占位。
- tool call token：标识工具调用开始、参数、结束。

EOS 和 stop condition 特别容易出问题。EOS 配错会导致请求停不下来，或者一生成就结束。多个 EOS token 并存时，还要确认 tokenizer、模型 config 和 serving 参数是否一致。

## Tool Call 和 Structured Output

工具调用和结构化输出让输入输出格式更复杂。

对于 tool call，chat template 通常需要把可用工具描述放进 prompt，模型输出再由 parser 解析成函数名和参数。streaming tool call 还要求 parser 能处理增量 JSON 片段。

对于 structured output，系统可能在 decode 过程中约束 token 选择，使输出符合 JSON schema、正则表达式或 grammar。它和普通 sampling 不同，错误可能来自 tokenizer、schema、parser 或 logits processor。

## 多模态输入

多模态模型会把图片、音频、视频等输入转换成 embedding，再和文本 token 对齐。文本 prompt 中通常会出现占位 token，例如 `[IMG]` 或模型特定的 image placeholder。

- 多模态 placeholder 会占用 token 或位置，对 max model len 和 prefix caching 有影响。
- 真实图像内容不是普通文本 token，而是在 prefill 时以 embedding 形式进入模型。

## 和 vLLM 代码的连接

- OpenAI 请求入口：`$PATH_TO_VLLM/vllm/entrypoints/openai`
- tokenizer 工具：`$PATH_TO_VLLM/vllm/tokenizers`
- chat template 相关：`$PATH_TO_VLLM/vllm/transformers_utils/chat_templates`
- transformers 适配：`$PATH_TO_VLLM/vllm/transformers_utils`
- tool parser：`$PATH_TO_VLLM/vllm/tool_parsers`
- tool calling 文档：`$PATH_TO_VLLM/docs/features/tool_calling.md`
- structured outputs 文档：`$PATH_TO_VLLM/docs/features/structured_outputs.md`

## 常见排查顺序

1. 确认模型权重、config、tokenizer、chat template 是否来自同一个模型目录或同一个模型发布版本。
2. 打印或保存 apply chat template 后的 prompt。
3. 检查 BOS/EOS/PAD token id。
4. 检查 `add_generation_prompt` 或等价选项是否符合模型要求。
5. 检查 tool schema 是否过长、是否符合模型支持格式。
6. 检查 max model len 是否被 prompt token 数耗尽。

## 思考与探索

1. 找一个 Qwen 或 GLM 的 tokenizer，观察同一句中文和英文分别会被切成多少 token。
2. 手写一个两轮对话，尝试把它转换成“模型可能看到的 prompt 字符串”。
3. 思考：如果用户说“模型总是输出工具调用格式错误”，你会先看哪三个地方？


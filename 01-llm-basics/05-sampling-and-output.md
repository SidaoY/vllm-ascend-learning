# Sampling 与输出处理

模型 forward 的直接结果不是文本，而是 logits。Sampling 与输出处理负责把 logits 变成 token，把 token 变成文本，再按 API 协议返回给用户。这个阶段看起来离 kernel 很远，但它直接决定接口行为、稳定性和用户体验。

## Logits 是什么

Logits 是模型对词表中每个 token 的打分。词表大小是 V，那么每个需要采样的位置都会得到一个长度为 V 的向量。

```text
logits -> logits processors -> probability distribution -> next token
```

如果直接选分数最高的 token，就是 greedy decoding。如果引入随机性，就会用 temperature、top-p、top-k 等采样策略。

## Greedy 和随机采样

Greedy：每次选择 logit 最大的 token。输出稳定，可复现性好，但可能更死板。

Temperature：控制分布平滑程度。temperature 越低，越接近 greedy；temperature 越高，随机性越强。

Top-k：只在概率最高的 k 个 token 中采样。

Top-p：选择累计概率达到 p 的最小 token 集合，再在其中采样，也叫 nucleus sampling。

Min-p：过滤掉相对最高概率过低的 token，常用于控制低质量尾部采样。

这些参数不是越多越好。服务排障时，要先确认请求实际使用的 sampling params。

## Logits Processor

Logits processor 会在采样前修改 logits。常见用途：

- repetition penalty：降低重复 token 的概率。
- presence/frequency penalty：按出现情况惩罚 token。
- bad words：禁止某些 token 序列。
- guided decoding：约束输出符合 JSON schema、regex 或 grammar。
- force EOS：在某些条件下强制结束。

它们可能改变模型输出，也可能影响性能。某些 processor 对 greedy sampling 可以跳过，某些不能跳过。

## Stop Condition

生成何时停止由多种条件共同决定：

- 生成到 EOS token。
- 生成到 stop token ids。
- 生成文本匹配 stop strings。
- 达到 max tokens。
- 请求被取消或超时。
- structured output 认为结果完整。

stop string 比 stop token 更难处理，因为字符串可能跨越多个 token。streaming 时还要避免把应该截断的内容提前发给用户。

## Detokenization

Detokenization 把 token ids 转回文本。它不是简单的逐 token 拼接，因为 tokenizer 可能用特殊规则表示空格、字节、Unicode 字符或 special token。

常见问题包括：

- 中文或 emoji 被错误切分后显示异常。
- special token 泄漏到输出。
- stop string 被截断或残留。
- streaming 增量文本和最终文本不一致。

## Streaming 输出

Streaming 会把生成过程拆成多个 chunk 返回。用户更快看到首字，但服务端需要维护更多状态：

- 哪些 token 已经 detokenize 并发送。
- 当前 chunk 是否包含 stop string 的一部分。
- tool call arguments 是否是合法的增量片段。
- finish reason 应该在哪个 chunk 返回。
- usage 是否在最后返回。

Tool call streaming 尤其容易出错，因为函数名、参数 JSON、finish reason 可能分布在多个 chunk 中。

## Usage Accounting

OpenAI 兼容 API 通常会返回 usage：

- prompt tokens：输入 token 数。
- completion tokens：输出 token 数。
- total tokens：两者之和。
- reasoning tokens：某些 reasoning 模型会额外统计思考 token。

usage 看似只是统计，但对计费、限流、监控和用户感知都很重要。vLLM Ascend 中也有针对特定模型 usage accounting 的 patch。

## 输出后处理和 API Schema

输出最终要符合接口 schema。以 chat completion 为例，常见字段包括：

- `choices`
- `message` 或 `delta`
- `finish_reason`
- `logprobs`
- `usage`
- tool call 相关字段

排查接口兼容性问题时，要区分“模型生成了什么”和“server 按协议返回了什么”。

## 和代码的连接

- vLLM sampling：`$PATH_TO_VLLM/vllm/v1/sample`
- output processor：`$PATH_TO_VLLM/vllm/v1/engine/output_processor.py`
- detokenizer：`$PATH_TO_VLLM/vllm/v1/engine/detokenizer.py`
- OpenAI entrypoints：`$PATH_TO_VLLM/vllm/entrypoints/openai`
- MiniMax usage patch：`$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_minimax_usage_accounting.py`
- GLM tool call patch：`$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_glm_tool_call_parser.py`
- reasoning outputs 文档：`$PATH_TO_VLLM/docs/features/reasoning_outputs.md`

## 思考与探索

1. 分别用一句话解释 temperature、top-k、top-p。
2. 为什么 stop string 在 streaming 场景下比 stop token 更复杂？
3. 如果用户报告 usage 中 completion tokens 不对，你会看 tokenizer、output processor 还是 model runner？为什么？

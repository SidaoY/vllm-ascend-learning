# Transformer 与主流模型结构

vLLM 和 vLLM Ascend 支持很多模型。不需要一开始理解每个模型的论文细节，但需要掌握推理代码中反复出现的结构概念：decoder-only Transformer、attention、MLP、RoPE、GQA、MLA、MoE、多模态模块。这些结构会直接影响 KV cache shape、attention backend、并行策略和 patch 路径。

## Decoder-only Transformer

当前主流生成式大模型大多是 decoder-only Transformer。一次 forward 可以粗略理解为：

```text
token ids
  -> embedding
  -> transformer block 1
  -> transformer block 2
  -> ...
  -> transformer block N
  -> final norm
  -> lm head
  -> logits
```

每个 transformer block 通常包含：

- Norm：RMSNorm 或 LayerNorm。
- Attention：让当前 token 读取上下文信息。
- MLP/FFN：逐 token 的非线性变换。
- Residual connection：残差连接，帮助深层网络稳定。

推理代码里常见的 `hidden_states` 就是在这些层之间流动的张量。

## Embedding 和 LM Head

Embedding 把 token id 映射成向量。假设 hidden size 是 4096，那么每个 token 会变成一个长度为 4096 的向量。

LM head 把最后的 hidden state 映射回词表大小的 logits。词表有多少 token，logits 就有多少维。sampling 阶段会从 logits 中选择下一个 token。

很多模型会共享 embedding 和 lm head 权重，但这不是必须的。并行和量化实现中，经常会对 embedding/lm head 做特殊处理。

## Attention 结构

Attention 中最常见的三个张量是 Q、K、V：

- Q：query，当前 token 想查询什么。
- K：key，历史 token 提供哪些可匹配的信息。
- V：value，匹配后要读取的内容。

生成式模型使用 causal attention，当前 token 只能看自己和过去 token，不能看未来 token。

一次 attention 的简化计算过程可以写成：

```text
hidden_states
  -> q_proj / k_proj / v_proj
  -> Q, K, V
  -> scores = Q x K^T / sqrt(head_dim)
  -> weights = softmax(scores + causal_mask)
  -> context = weights x V
  -> o_proj
  -> attention output
```

这里的 `o_proj` 也常被叫做 output projection。它负责把多个 attention heads 的结果重新映射回模型的 hidden size，交给后续 residual、norm 和 MLP。

推理阶段最关键的优化是缓存历史 token 的 K/V。这样 decode 时不需要重新计算所有历史 token 的 K/V，只需要计算新 token 的 Q/K/V，并读取已有 KV cache。

## MHA、MQA、GQA、MLA

不同模型的 attention head 组织方式不同，会影响 KV cache 大小和 attention kernel。

| 结构 | 直观解释 | 对推理的影响 |
| --- | --- | --- |
| MHA | 每个 Q head 都有自己对应的 K/V head | KV cache 最大，表达能力强，成本高 |
| MQA | 所有 Q heads 共享同一组 K/V head | KV cache 最小，decode 成本低 |
| GQA | 把 Q heads 分成多组，每组 Q heads 共享一组 K/V head | MHA 和 MQA 的折中 |
| MLA | 使用低秩 latent 表示压缩 KV，并结合 RoPE 部分信息 | KV cache 形态特殊，backend 和 cache 管理更复杂 |

可以用一个 32 个 Q heads 的模型来直观理解：

```text
MHA: 32 个 Q heads，对应 32 个 K heads 和 32 个 V heads
MQA: 32 个 Q heads，共享 1 个 K head 和 1 个 V head
GQA: 32 个 Q heads，分成 8 组，每组 4 个 Q heads 共享 1 个 K/V head
```

所以 GQA 常见配置是 `num_attention_heads > num_key_value_heads`，并且多个 Q heads 会映射到同一个 KV head。这样既减少 KV cache，也保留比 MQA 更细的表达能力。

读 vLLM/vLLM Ascend 代码时，看到 `num_attention_heads`、`num_key_value_heads`、`kv_lora_rank`、`qk_rope_head_dim` 等字段，就要意识到它们会影响 KV cache 形状。一个粗略判断是：普通 MHA/GQA/MQA 中，`num_key_value_heads` 越小，每层每个 token 需要缓存的 K/V 越少。

## 位置编码

Transformer 的 attention 本身只看 token 之间的内容匹配，并不天然知道 token 的顺序。如果没有额外位置信息，模型很难区分“我喜欢你”和“你喜欢我”这种 token 集合相似但顺序不同的输入。位置编码的作用，就是告诉模型每个 token 在序列中的位置，以及 token 之间的大致距离。

在推理代码中，位置相关信息通常会体现为 `positions`、`position_ids`、`rope_theta`、`rope_scaling`、`max_position_embeddings`、`max_model_len` 等字段。

### 绝对位置编码

早期 Transformer 常见绝对位置编码，例如 learned position embedding 或 sinusoidal position embedding。它们会为第 0、1、2、... 个位置提供一个位置向量，再和 token embedding 结合。

直观理解是：

```text
token embedding + position embedding -> hidden states
```

这种方式容易理解，但在现代主流 decoder-only LLM 中，RoPE 更常见。

### RoPE

RoPE 是 Rotary Position Embedding，旋转位置编码。LLaMA、Qwen、DeepSeek 等模型都大量使用 RoPE。

RoPE 的直观理解是：它不直接给 token embedding 加一个位置向量，而是在 attention 里对 Q 和 K 做与位置相关的旋转变换。这样 attention 在计算 `Q x K^T` 时，就能感知 token 之间的位置和距离。

简化流程是：

```text
hidden_states -> Q, K, V
apply RoPE to Q and K
attention(Q, K, V)
```

注意 RoPE 通常作用在 Q/K 上，而不是 V 上。这个细节和 KV cache 有关系：decode 阶段会缓存历史 token 的 K/V，其中 K 已经携带了对应位置的信息；后续新 token 的 Q 也必须使用正确的当前位置进行 RoPE 变换。

### Decode 中的位置递增

Prefill 阶段会为 prompt 中每个 token 准备位置。例如 prompt 有 4 个 token，可以理解为位置 `0, 1, 2, 3`。

Prefill 结束后采样出第一个输出 token。下一轮 decode 输入这个 token 时，它的位置应该接在 prompt 后面，也就是位置 `4`。再下一轮 decode 的输入 token 位置是 `5`，以此类推。

```text
prompt tokens:        p0  p1  p2  p3
positions:             0   1   2   3
generated token #1:                 position 4
generated token #2:                 position 5
```

所以 decode 不是每轮从 position 0 重新开始。位置必须沿着已有上下文持续递增，否则 attention 看到的相对距离会错。

### RoPE Scaling 和长上下文

模型训练时通常有一个最大上下文长度，例如 4K、8K、32K。推理时如果想支持更长上下文，可能会使用 RoPE scaling 方法调整位置编码。

常见名字包括：

- linear scaling
- dynamic scaling
- NTK scaling
- YaRN

这些方法的目标是让模型在超过训练长度的位置上仍然能比较稳定地工作。但它们不是无损扩容，可能影响精度、稳定性和性能。阅读模型 config 时，如果看到 `rope_scaling`、`rope_theta` 或类似字段，就要意识到它们会影响长上下文行为。

### mRoPE

mRoPE 可以理解为 multi-dimensional RoPE，多模态模型中比较常见。文本只有一维顺序，但图像、视频等输入可能同时有高度、宽度、时间等维度。

例如视觉语言模型可能需要同时表达：

```text
文本 token 的位置
图像 patch 的行位置
图像 patch 的列位置
视频帧的时间位置
```

这时位置编码不再只是简单的 `0, 1, 2, ...`，而可能需要多维 position 信息。读多模态模型代码时，`mrope_position_ids`、`image_grid_thw` 这类字段通常就和这个方向有关。

### 和推理系统的关系

位置编码看起来是模型结构细节，但它会影响推理系统的很多地方：

- Tokenization 后生成的 token 需要分配 position。
- Prefill 要为整段 prompt 准备连续 position。
- Decode 每轮要给新 token 使用正确递增的 position。
- Chunked prefill 不能因为 prompt 被拆块而重置 position。
- Prefix caching 复用 KV cache 时，必须保证 position 语义一致。
- CP 并行和长上下文场景下，不同 rank 处理的 token 位置需要对齐。
- 多模态模型需要处理文本 token 和视觉 token 的 position 对齐。

所以在 vLLM/vLLM Ascend 代码中，看到 `positions`、`position_ids`、`rope`、`mrope`、`cos_sin_cache` 等名字时，应该意识到它们都和位置编码有关。

## MLP 与激活函数

Transformer block 中的 MLP 通常占用大量参数和计算。常见结构是 gated MLP，例如 SwiGLU：

```text
hidden -> gate_proj / up_proj -> activation and multiply -> down_proj
```

Dense 模型里，每个 token 都经过同一套 MLP 权重。MoE 模型里，token 会被 router 分配到部分 experts，每个 expert 可以理解为一套独立 MLP。

## Dense、MoE 和多模态模型

Dense 模型：每个 token 激活全部层的同一套参数。结构相对规整，负载更稳定。

MoE 模型：每个 token 只激活部分 experts。总参数量可以很大，但每个 token 的激活参数有限。问题是不同 expert 的 token 数可能不均衡，需要 expert parallelism 和负载均衡。

多模态模型：除了文本模型主体，还会有视觉/音频 encoder、projector、特殊 processor。它们会改变输入处理和 prefill 路径。

## 模型家族中的常见特殊点

下面是阅读代码时常见的模型差异，具体实现以后再深入：

- Qwen 系列：chat template、GQA、部分模型有 MoE、多模态和 MTP 变体。
- DeepSeek 系列：MLA、MoE、长上下文、推理优化路径较多。
- GLM 系列：tool call parser、chat template、特殊输出格式可能需要适配。
- Kimi/MiniMax 等模型：可能有模型结构、reasoning output、spec decode 或量化加载方面的特殊路径。
- 多模态模型：processor、placeholder、mRoPE、图像 embedding 对齐是重点。

这些差异解释了为什么 vLLM Ascend 中会有许多 worker patch 和模型特化逻辑。

## 和代码的连接

- 模型定义：`$PATH_TO_VLLM/vllm/model_executor/models`
- layer 实现：`$PATH_TO_VLLM/vllm/model_executor/layers`
- attention layer：`$PATH_TO_VLLM/vllm/model_executor/layers/attention`
- vLLM Ascend worker patch：`$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker`
- vLLM Ascend 模型教程：`$PATH_TO_VLLM_ASCEND/docs/source/tutorials/models`

## 思考与探索

1. 找一个模型 config，观察 hidden size、num layers、attention heads、KV heads、max position embeddings。
2. 解释 GQA 为什么能减少 KV cache 大小。
3. 在 `vllm_ascend/patch/worker` 中挑一个模型 patch 文件，只看文件名和 import，猜它可能在适配什么模型差异。

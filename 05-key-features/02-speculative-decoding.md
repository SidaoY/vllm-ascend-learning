# 投机推理

投机推理的目标是优化 decode：用较便宜的方式先提出多个候选 token，再让目标模型一次性验证。接受率足够高时，目标模型不必严格一轮只推进一个 token，从而降低 TPOT、提升 decode 吞吐。

基础概念已在 [vLLM 核心概念章节](../01-llm-basics/03-prefill-decode-kv-cache.md) 中介绍 decode 瓶颈，本章在此基础上展开 vLLM Ascend 中投机推理的适配边界。

## 数据流

| 步骤 | 发起方 | 接收方 | 动作 |
| --- | --- | --- | --- |
| 1 | Request state | Proposer | current tokens / hidden states |
| 2 | Proposer | Ascend model runner | draft tokens |
| 3 | Ascend model runner | Target model verify | verify draft tokens |
| 4 | Target model verify | Rejection sampler | logits / probabilities |
| 5 | Rejection sampler | Output processor | accepted tokens and corrected token |
| 6 | Output processor | Request state | update request state |

这条链路有两个关键点：

- Proposer 和 target model 不一定是同一个模型，也不一定走同一种执行路径。
- 接受/拒绝之后，request state、KV cache、position、output token、sampling state 必须保持一致。

## vLLM Ascend 主要适配什么

vLLM Ascend 复用 vLLM 的投机推理框架，同时在 NPU 侧适配：

- proposer：n-gram、NPU n-gram、EAGLE、draft model、suffix、extract hidden states 等。
- NPU 执行：把 draft tokens、verify tokens 和 attention metadata 接到 Ascend model runner。
- rejection sampler：适配 NPU sampling/rejection 路径和必要 patch。
- 模型特殊路径：MTP、DeepSeek/Qwen3 Next 等模型的特殊 proposer 或 head。
- graph：投机推理下 verify token 数可能变成 `(K + 1) * batch`，graph capture size 需要配合。
- KV cache：为 lookahead tokens 预留空间，并在拒绝时保持状态一致。

代码入口：

- `$PATH_TO_VLLM/vllm/v1/spec_decode`
- `$PATH_TO_VLLM/docs/features/speculative_decoding`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/spec_decode`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/triton/spec_decode`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker/patch_rejection_sampler.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker/patch_qwen3_next_mtp.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker/patch_deepseek_mtp.py`

## 常见 proposer

N-gram proposer：从 prompt 或上下文中找重复片段。优点是成本低，不需要额外模型；缺点是适用场景比较依赖文本重复。

EAGLE proposer：基于轻量 draft 模型或辅助结构预测后续 token。优点是通用性更好；缺点是需要额外模型状态、配置和图模式适配。

MTP proposer：模型本身带 Multi Token Prediction 能力时，可以直接产出多个候选 token。它经常和特定模型结构、特定 head、特定 patch 绑定。

Suffix proposer：利用 suffix 匹配提出候选，对代码编辑、重复上下文、agent loop 等场景可能更友好。

Extract hidden states：它不是真正为了加速推理，而是为了抽取目标模型中间 hidden states，通常用于 EAGLE 类 draft 模型的数据准备。

### MTP 模型专项说明

Multi Token Prediction 是模型原生支持一次预测多个 future token 的能力。与 n-gram 或 EAGLE 不同，MTP 不需要额外的 draft 模型——proposer 本身就是目标模型的一部分。

**DeepSeek-V3 / DeepSeek-R1 的 MTP：**

DeepSeek 系列模型的 MTP 通过额外的 MTP module 实现。每个 MTP module 包含独立的 embedding layer、output head 和共享的 transformer block。关键适配点：

- `patch_deepseek_mtp.py`：Ascend 侧对 DeepSeek MTP 的 patch，主要处理 MTP module 的权重加载和前向传播。
- MTP 的 hidden states 来自目标模型的中间层，需要正确截取和传递。
- MTP 的 embedding 和 output head 与主模型共享 tokenizer，但权重独立。

**Qwen3-Next 的 MTP：**

Qwen3-Next 的 MTP 实现与 DeepSeek 不同，通过 `patch_qwen3_next_mtp.py` 适配：

- Qwen3-Next 的 MTP head 结构与主模型的 lm_head 共享部分权重。
- 需要正确处理 MTP head 的 TP 切分（与主模型 TP 配置一致）。

**MTP 的通用排查点：**

- MTP module 的权重是否正确加载（检查 checkpoint 路径和 key mapping）。
- MTP 的 hidden states 截取位置是否正确（不同模型可能从不同层截取）。
- MTP 的 TP 配置是否与主模型一致。
- MTP 输出的 token 是否需要经过与主模型相同的 logits processors。

## 投机推理与其他特性的组合

### 投机推理 + PD 分离

PD 分离场景下，prefill 和 decode 在不同实例上运行。投机推理的 proposer 和 target verify 都在 decode 实例上执行，但需要注意：

- **KV 预留**：prefill 实例产生的 KV 需要为 draft token 预留 lookahead 位置。如果 prefill 时不知道 decode 会使用投机推理，KV block 数量可能不足。
- **Metadata 传递**：connector 传递的 KV metadata 需要包含"哪些位置是 draft token 的 KV"，以便 decode 实例的 rejection sampler 正确处理。
- **Proposer 状态**：如果 proposer 依赖 prefill 阶段产生的 hidden states（如 EAGLE），这些 hidden states 也需要通过 connector 传递。

### 投机推理 + CP 并行

CP 与投机推理组合时，主要挑战在 DCP 侧：

- **Verify shape 变化**：DCP 下每轮 decode 的 token 数从 1 变为 `K+1`（K 个 draft + 1 个 bonus 位置）。DCP metadata 中的 seq_len、position、block table 都需要适配这个变化。
- **AllGather Q 的 shape**：DCP 的 all_gather Q 操作需要处理 `[B*(K+1), num_heads, head_dim]` 的 shape，而非通常的 `[B, num_heads, head_dim]`。
- **FAUpdate 合并**：多个 rank 的 attention 输出合并时，需要知道哪些位置是 draft token（可能被拒绝），哪些是 bonus token（一定被接受）。

### 投机推理 + Graph Mode

投机推理与 graph mode 的组合是最容易出问题的场景之一：

- **Capture sizes**：graph 需要预先 capture 固定的 batch size 和 token 数。投机推理的 verify token 数是 `B * (K+1)`，需要确保 graph capture 覆盖了这个 shape。
- **动态控制流**：rejection sampler 中的接受/拒绝逻辑是动态的（取决于概率比较结果），graph 无法 capture 这部分。通常 rejection sampler 在 graph 之外执行。
- **Proposer 是否支持 graph**：n-gram proposer 通常不支持 graph（涉及 CPU 侧的文本匹配），EAGLE 和 MTP proposer 可能支持 graph。

排查 graph + 投机推理问题时，建议先在 eager 模式下验证正确性，再逐步开启 graph。

## 接受/拒绝机制详解

投机推理的核心是 rejection sampler。它的任务不是"判断 draft token 对不对"，而是"保证输出分布与目标模型一致"。理解这一点是理解整个投机推理的关键。

### 三种 token 角色

一轮投机推理中涉及三种 token：

| 角色 | 含义 | 来源 |
| --- | --- | --- |
| accepted token | 被接受的 draft token | proposer 提出，验证通过 |
| recovered token | 拒绝后重新采样的修正 token | 从 `max(0, target_prob - draft_prob)` 分布中采样 |
| bonus token | 全部 draft 被接受后的额外 token | 仅从 target 分布采样 |

输出序列 = accepted tokens + recovered token（如果有拒绝）+ bonus token（如果全部接受）。

### Greedy 模式下的接受/拒绝

Greedy 模式最简单：draft token 被接受，当且仅当它等于 target model 的 argmax。

**示例 1：部分接受**

假设 proposer 提出 3 个 draft token，target model 验证结果如下：

| Position | Draft | Target argmax | Result |
| --- | --- | --- | --- |
| pos 0 | "的" | "的" | ✅ 接受 |
| pos 1 | "是" | "在" | ❌ 拒绝，输出 target argmax "在" |
| pos 2 | "一" | "个" | — 不再验证 |

- pos 0：draft "的" == target argmax "的" → 接受
- pos 1：draft "是" != target argmax "在" → 拒绝，输出 target argmax "在"
- pos 2：因为 pos 1 已被拒绝，pos 2 不再验证

最终输出：`["的", "在"]`，本轮前进了 2 个 token。

**示例 2：全部接受**

| Position | Draft | Target argmax | Result |
| --- | --- | --- | --- |
| pos 0 | "的" | "的" | ✅ 接受 |
| pos 1 | "是" | "是" | ✅ 接受 |
| pos 2 | "一" | "一" | ✅ 接受 |

全部接受后，还会追加一个 bonus token（从 target 分布单独采样）：

```text
bonus: target sample → "种"
```

最终输出：`["的", "是", "一", "种"]`，本轮前进了 4 个 token。

### 随机采样模式下的接受/拒绝

随机采样模式更复杂。对于每个 draft token，接受条件为：

```text
target_prob(draft_token) / draft_prob(draft_token) >= uniform_random
```

其中 `uniform_random` 是 [0, 1) 之间的均匀随机数。

**示例 3：随机模式下的接受与拒绝**

假设词表为 {A, B, C}，proposer 提出 draft token "A"：

```text
draft_probs:  [A: 0.7, B: 0.2, C: 0.1]
target_probs: [A: 0.5, B: 0.3, C: 0.2]
uniform_random = 0.6
```

计算接受条件：`target_prob(A) / draft_prob(A) = 0.5 / 0.7 ≈ 0.714`

`0.714 >= 0.6` → 接受！输出 "A"。

如果 `uniform_random = 0.8`，则 `0.714 < 0.8` → 拒绝。此时需要从修正分布中采样 recovered token：

```text
Corrected distribution = max(0, target_probs - draft_probs)
                       = max(0, [0.5-0.7, 0.3-0.2, 0.2-0.1])
                       = [0, 0.1, 0.1]
After normalization     = [0, 0.5, 0.5]
```

从 {B: 0.5, C: 0.5} 中采样 recovered token。注意 "A" 的概率被置为 0——这正是"拒绝"的含义：draft token 被排除后，从剩余概率中重新采样。

**为什么这个算法保证分布正确？**

直觉上：如果 draft 模型对某个 token 过于自信（draft_prob 很高），而 target 认为它没那么好（target_prob 较低），则 `target_prob / draft_prob` 变小，更容易被拒绝。被拒绝后从 `max(0, target - draft)` 中重采样，恰好补回了 target 分布中"被 draft 高估"的部分。数学上这保证了输出分布严格等于 target model 的分布。

### 完整流程示例

假设 batch 中有 2 个请求，proposer 分别提出 3 个和 2 个 draft token：

```text
Request 0: draft = [t0, t1, t2], num_draft=3
Request 1: draft = [u0, u1],     num_draft=2

target model forward:
  logits shape = [3+2+2, vocab_size] = [7, vocab_size]
  （3 个 draft + 2 个 draft + 2 个 bonus 位置）
```

Rejection sampler 的处理顺序：

1. 从 bonus 位置采样 bonus token（每个请求一个）。
2. 对 draft 位置应用 logits processors（temperature、top-k、top-p、penalties）。
3. 对 greedy 请求：逐位置比较 draft_token_id == target_argmax。
4. 对随机请求：逐位置计算 `target_prob / draft_prob >= uniform_random`。
5. 被拒绝的位置，从修正分布采样 recovered token。
6. 全部接受时，追加 bonus token。
7. `parse_output` 过滤掉 PLACEHOLDER_TOKEN_ID，输出最终 token 列表。

### 关键实现细节

从代码中可以看到几个值得注意的设计：

- **float64 随机数**：`uniform_probs` 使用 float64 而非 float32，因为 float32 有非零概率采样到精确的 0.0（PyTorch issue #16706），会导致接受条件异常。
- **in-place 更新 logits**：temperature scaling 和 top-k/top-p 直接修改 logits tensor 以节省显存。
- **Greedy 快速路径**：全部请求都是 greedy 时，跳过 target_probs 的 softmax 计算和 recovered token 采样，直接比较 argmax。
- **synthetic 模式**：用于测试和调试，用预设的接受率替代真实概率比较，不依赖 draft_probs。

## Graph 和 shape

投机推理和 graph 的关系比较敏感。假设每次 proposer 提出 `K` 个 speculative tokens，target verify 往往需要处理 `K + 1` 个 token，因为还要包含目标模型自身修正位置。batch size 为 `B` 时，verify shape 可能接近 `B * (K + 1)`。

因此需要注意：

- graph capture sizes 是否覆盖目标 batch 和 speculative token 数。
- proposer 和 target verify 是否都能在 graph 下运行。
- 接受 token 数是动态的，不能简单等同于 verify token 数。
- rejection sampler 是否包含 graph 不友好的动态控制流。

Graph 下的问题常表现为 eager 正常、开启 graph 后 shape mismatch、capture 失败或性能没有收益。

## KV Cache 和状态一致性

投机推理会提前为 draft tokens 准备位置。如果 token 被接受，状态前进多个 token；如果某个 token 被拒绝，需要从拒绝位置开始修正。

需要特别关注：

- lookahead KV block 是否足够。
- draft token 的 position ids 是否正确。
- 被拒绝 token 对应的 KV 是否会污染后续状态。
- output processor 是否只输出被接受和修正后的 token。
- scheduler 是否知道请求实际前进了几个 token。

这类 bug 很少只停留在 spec decode 模块内部，经常会牵涉 scheduler、KV manager、model runner 和 sampler。

## 指标

投机推理调优时至少看：

- acceptance rate：接受率。
- num speculative tokens：每轮提出多少 token。
- TPOT / ITL：decode 阶段是否真的变快。
- request throughput / token throughput：整体吞吐是否提升。
- TTFT：额外 proposer 或图捕获是否影响首 token。
- NPU memory：draft model、lookahead KV cache、graph buffer 是否带来额外压力。
- fallback 次数：是否走到预期 NPU proposer 和 sampler 路径。

接受率低时，不要只调大 `num_speculative_tokens`。提出更多 token 可能让验证成本和 KV 预留变大，反而更慢。

## 测试建议

- 单测：proposer 输出、rejection sampler、特殊模型 patch、hidden states 提取。
- 单卡 e2e：n-gram、EAGLE、MTP、suffix 的基本生成路径。
- 多卡 e2e：TP/DP/EP 下 proposer 和 target 是否一致。
- 精度：低随机性配置下对齐 token ids 或 logprobs。
- 性能：对比不开投机推理的 baseline，看接受率、TPOT 和吞吐。

测试入口：

- `$PATH_TO_VLLM_ASCEND/tests/ut/spec_decode`
- `$PATH_TO_VLLM_ASCEND/tests/ut/sample/test_rejection_sampler.py`
- `$PATH_TO_VLLM_ASCEND/tests/e2e/singlecard/spec_decode`
- `$PATH_TO_VLLM_ASCEND/tests/e2e/multicard/2-cards/spec_decode`

## 常见问题

Draft 模型加载失败：看模型路径、TP 配置、draft_tensor_parallel_size、checkpoint 和 tokenizer。

接受率低：看 proposer 质量、采样参数、prompt 类型、draft length 和模型匹配程度。

输出不一致：看 rejection sampler、随机性、token ids、position ids、状态回滚。

Graph capture 失败：看 capture sizes、verify shape、proposer 是否支持 graph。

OOM：看 draft model、lookahead KV cache、graph buffer 和 batch size。

## 参考入口

- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/speculative_decoding.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/Multi_Token_Prediction.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/tutorials/features/suffix_speculative_decoding.md`

## 思考与探索

1. 为什么投机推理提升的是 decode 侧能力，而不是 prefill 的核心优化？
2. 如果接受率很高但吞吐没有提升，可能是哪几个环节抵消了收益？
3. 为什么 rejection sampler 的语义错误会导致“看起来能生成文本，但分布不对”？

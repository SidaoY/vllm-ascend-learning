# CP 并行

Context Parallelism，简称 CP，主要面向长上下文推理。长序列会让 prefill attention 计算变重，也会让 decode 阶段每个请求携带大量历史 KV。CP 的思路是沿 sequence/context 维度切分，让多个 rank 分担长上下文压力。

## CP 解决的问题

长上下文下有两个不同压力：

- Prefill：prompt token 多，attention 计算量大，TTFT 容易变差。
- Decode：每轮新 token 少，但要读取很长的历史 KV，KV cache 容量和带宽压力大。

所以 vLLM Ascend 把 CP 分成：

- PCP：Prefill Context Parallel，用于 prefill 长序列切分。
- DCP：Decode Context Parallel，用于 decode 阶段分担长历史 KV。

PCP 和 DCP 可以一起使用，也可以根据场景分别使用。

## 和其他并行的区别

| 并行 | 切分维度 | 主要目的 |
| --- | --- | --- |
| TP | 模型权重和计算张量 | 放下大模型、提升单层计算并行度 |
| PP | 模型层 | 放下深模型、流水化执行 |
| DP | 请求副本 | 提升服务吞吐和容量 |
| EP | MoE expert | 分散 expert 权重和 token 负载 |
| CP | sequence/context | 分担长上下文 attention 和 KV 压力 |

CP 不是 TP 的替代品。很多场景会同时配置 TP 和 CP，world size 也要按组合关系计算。

## PCP 与 DCP

PCP 更关注 prefill 的 query/context 切分。它希望多个 rank 分担长 prompt 的 attention 计算，从而降低长 prompt 的 TTFT。

DCP 更关注 decode 的历史 KV 切分。它希望每个 rank 只持有或处理部分 context，从而缓解 decode 阶段 KV duplication 和容量压力。

| 阶段 | 输入 | 处理 | 输出 |
| --- | --- | --- | --- |
| 1 | Long prompt | PCP: split prefill context | Distributed KV cache |
| 2 | Distributed KV cache | DCP: split decode context | Next token |

从实现上看，PCP/DCP 都要求 attention metadata、block table、position、mask 和通信逻辑对切分方式有一致理解。

## Context 如何切分

CP 的核心操作是把一个长序列沿 context 维度切到多个 rank 上。切分不是简单的"前一半给 rank 0，后一半给 rank 1"，而是需要考虑 causal attention 的特性。

### Prefill 阶段的切分（PCP）

假设一个长度为 8192 的 prompt，PCP size = 4：

```text
Original sequence: [t0, t1, t2, ..., t8191]

After PCP split, each rank's context range:
  rank 0: tokens [0,    2047]
  rank 1: tokens [2048, 4095]
  rank 2: tokens [4096, 6143]
  rank 3: tokens [6144, 8191]
```

但 attention 计算不是各算各的。对于 rank 1 上的 token，它需要 attend 到 rank 0 的 context（历史 token），这部分不需要 mask；而 attend 到 rank 1 自身及之后的 context 时，需要 causal mask。因此 PCP 把每个 rank 的 attention 分成两部分：

- **nomask attention**：当前 rank 的 Q attend 到前面 rank 的 KV（历史 context，不需要 causal mask）。
- **mask attention**：当前 rank 的 Q attend 到自己 rank 的 KV（需要 causal mask）。

以 rank 1 为例：

```text
rank 1's Q needs to attend to:
  ┌─────────────────────┬──────────────────────┐
  │ rank 0's KV         │ rank 1's KV          │
  │ (history context)   │ (current context)    │
  │ → nomask attention  │ → mask attention     │
  └─────────────────────┴──────────────────────┘
```

代码中通过 `AscendPCPMetadata` 中的索引张量来路由这些计算：

- `q_head_idx` / `q_tail_idx`：区分 Q 的 head 部分和 tail 部分。
- `kv_with_q_head_nomask_idx` / `kv_with_q_head_mask_idx`：head Q 对应的 nomask/mask KV 索引。
- `kv_with_q_tail_nomask_idx` / `kv_with_q_tail_mask_idx`：tail Q 对应的 nomask/mask KV 索引。

### Decode 阶段的切分（DCP）

Decode 阶段每轮只产生 1 个新 token，但需要读取全部历史 KV。DCP 的思路是让每个 rank 只持有部分历史 KV：

```text
Total KV cache length = 8192, DCP size = 4

KV held by each rank:
  rank 0: KV positions [0,    2047]
  rank 1: KV positions [2048, 4095]
  rank 2: KV positions [4096, 6143]
  rank 3: KV positions [6144, 8191]
```

新 token 的 Q 需要 attend 到所有 rank 的 KV。DCP 的做法是：

1. 每个 rank 用本地 KV 计算 attention，得到本地的 output 和 lse（log-sum-exp）。
2. 通过 all_gather 收集所有 rank 的 output 和 lse。
3. 用 `npu_attention_update` 合并所有 rank 的结果。

## 数据通信：AllGather + FAUpdate

CP attention 的通信模式可以概括为"本地计算 + 全局合并"。attention 算子本身不跨 rank 通信，通信发生在算子之外。

### Decode 路径的通信流程

以 DCP 为例，一次 decode step 的完整流程：

```text
Step 1: DCP all_gather Q
  Per-rank Q shape: [num_tokens, num_heads, head_dim]
  After all_gather:  [num_tokens, num_heads * dcp_size, head_dim]
  Purpose: let each rank's Q query all ranks' KV simultaneously

Step 2: Local FIA computation
  Each rank uses local KV cache + all_gathered Q to call
  npu_fused_infer_attention_score
  → produces local attn_out and softmax_lse

Step 3: Merge out and lse
  _process_attn_out_lse():
    - Concatenate attn_out and softmax_lse on last dim
    - If dcp_size > 1: all_to_all exchange (split back by head dim)
    - If pcp_size > 1: all_gather (collect along batch dim)

Step 4: FAUpdate merge
  _npu_attention_update():
    - Reshape [PCP*S, DCP*H, D+1] to [N, S*H, D] and [N, S*H]
    - Call torch_npu.npu_attention_update(lse_list, out_list)
    - Equivalent to: O = sum(exp(lse_i - lse_final) * O_i)
```

### FAUpdate 的数学原理

`npu_attention_update` 解决的核心问题是：多个 attention 输出如何正确合并。

假设 rank 0 和 rank 1 分别计算了 attention，得到 `(O0, lse0)` 和 `(O1, lse1)`。合并后的结果应该是：

```text
lse_final = log(exp(lse0) + exp(lse1))
O_final = (exp(lse0 - lse_final) * O0 + exp(lse1 - lse_final) * O1)
```

这等价于把所有 rank 的 KV 拼在一起做一次完整的 attention——但避免了实际拼接 KV 的显存和通信开销。

### Prefill 路径的通信流程

PCP prefill 的通信更复杂，因为涉及 nomask 和 mask 两路 attention 的合并：

```text
Step 1: Each rank computes locally
  - nomask attention: Q attends to KV of previous ranks
  - mask attention:   Q attends to KV of own rank (with causal mask)

Step 2: Merge nomask and mask results
  _npu_attn_out_lse_update():
    Use npu_attention_update to merge output and lse from both attention paths

Step 3: Restore global order
  Use q_full_idx to concatenate head and tail merged results back to original Q order
```

### Chunked Prefill + CP 的通信

当 chunked prefill 和 CP 同时开启时，情况更复杂。每个 chunk 的 prefill 需要：

1. 从 KV cache 中加载之前 chunk 累积的 KV（`_load_kv_for_chunk`）。
2. 计算当前 chunk 的 attention。
3. 通过 `_update_chunk_attn_out_lse_with_current_attn_out_lse` 将当前 chunk 的结果与历史累积结果合并。
4. 如果 PCP > 1，还需要 all_gather Q 并通过 `cp_kv_recover_idx_for_chunk` 恢复正确的 token 顺序。

## CP Attention 路径

vLLM Ascend 中 CP attention 不是一个单独 kernel 名字，而是一组围绕不同 attention 类型的实现：

- `common_cp`：CP 通用工具和通信逻辑（`_process_attn_out_lse`、`_npu_attention_update`、`_update_out_and_lse` 等）。
- `attention_cp`：普通 dense/GQA attention 的 CP 路径。
- `mla_cp`：MLA 模型的 CP 路径。
- `sfa_cp`：Sparse Flash Attention / DSA 相关模型的 CP 路径。

代码入口：

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention/context_parallel`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/pcp_utils.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/passes/sequence_parallelism.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/passes/sequence_parallelism_moe.py`

## CP 和 KV Cache

CP 会改变 KV cache 的组织方式。尤其在 KV transfer、KV pool、PD 分离等场景里，producer 和 consumer 必须知道 KV 是如何按 CP rank 切分或交错存放的。

需要特别注意：

- block table 是否按 CP 切分后仍然正确。
- slot mapping 是否写入正确 rank 的 KV cache。
- position ids 和 mask 是否与全局序列位置一致。
- CP KV cache interleave 是否与 block size 对齐。
- 当前 Ascend 设备亲和的 block size 通常是 128，CP + KV transfer 场景中经常需要让 interleave size 与 block size 保持一致。

如果 CP 场景中出现 decode 精度异常，优先看 KV cache 切分、block table、position 和 attention metadata。

## CP 和其他特性

Chunked prefill：长 prompt 被拆成多轮后，PCP 需要处理每个 chunk 的 context 切分和累积状态。

Prefix caching：cache 命中后，CP 需要正确复用分布式 KV，而不是只复用单 rank 的局部状态。

Spec decode：投机推理的 verify tokens 和 lookahead KV 会改变每轮 decode 的 token 数，DCP metadata 需要一起适配。

Graph mode：CP 下 shape 和通信模式更复杂，graph capture 要确认固定 shape、metadata 更新和通信调用都可支持。

PD/KV transfer：prefill 和 decode 可能处在不同实例，CP 切分方式必须被 connector 正确传递。

## 配置理解

用户通常通过 `prefill_context_parallel_size` 和 `decode_context_parallel_size` 开启 PCP/DCP。实际部署时要同时考虑：

- TP、DP、EP、CP 的 world size 关系。
- 模型是 MLA、GQA、SFA 还是普通 dense attention。
- 硬件数量、单机/多机网络。
- 是否和 PD、KV pool、prefix cache、spec decode 组合。
- 当前模型教程或 feature guide 中的约束。

不要把 CP size 当成越大越好。CP 会降低某些计算或 KV 压力，但也会引入通信和同步成本。

### World Size 计算公式

CP 与其他并行方式组合时，world size 的计算需要特别注意：

```text
Total world size = DP_size × TP_size × PP_size × EP_size × max(PCP_size, DCP_size)

Where:
  - PCP_size and DCP_size take max because they share the same set of ranks
  - If only PCP (no DCP): world_size = ... × PCP_size
  - If only DCP (no PCP): world_size = ... × DCP_size
  - If both PCP and DCP: world_size = ... × max(PCP_size, DCP_size)
```

**Example calculation:**

```text
Scenario: DeepSeek-V3, 8-card single node
  TP=8, EP=8, PCP=2, DCP=2

  world_size = 1(DP) × 8(TP) × 1(PP) × 8(EP) × max(2, 2) = 128 cards

  With only 8 cards, this config cannot run!
  Adjust: TP=4, EP=4, PCP=2, DCP=2
  world_size = 1 × 4 × 1 × 4 × 2 = 32 cards (still not enough)

  Further adjust: TP=2, EP=2, PCP=2, DCP=2
  world_size = 1 × 2 × 1 × 2 × 2 = 8 cards ✅
```

**常见错误：**

- 把 PCP 和 DCP 当成独立的并行维度相乘（实际取 max）。
- 忘记 EP 也占用 world size（MoE 模型）。
- 单机 8 卡配置了需要 16 卡的并行参数。

### CP 显存收益量化分析

CP 对显存的影响可以从两个维度量化：

**1. KV Cache 显存节省（DCP 的主要收益）**

```text
Without DCP: each rank holds full KV cache
  per_rank_kv_bytes = num_layers × max_model_len × num_kv_heads × head_dim × dtype_size × 2

With DCP (size=N): each rank holds 1/N of KV cache
  per_rank_kv_bytes = num_layers × (max_model_len / N) × num_kv_heads × head_dim × dtype_size × 2

Savings = (N-1)/N × original KV cache memory
```

Using DeepSeek-V3 as example (MLA mode, kv_lora_rank=512, 128K context):

```text
Without DCP: per_rank_kv ≈ 58 layers × 128K × 512 × 2 bytes × 2(K+V) ≈ 14.5 GB
With DCP=4:  per_rank_kv ≈ 58 layers × 32K  × 512 × 2 bytes × 2(K+V) ≈ 3.6 GB
Savings ≈ 10.9 GB/card
```

**2. Attention 计算量分担（PCP 的主要收益）**

PCP 不减少总计算量，但把计算分散到多个 rank：

```text
Without PCP: single rank computes all attention
  compute_per_rank = O(S² × H × D)  # S=seq_len, H=num_heads, D=head_dim

With PCP (size=N): each rank computes 1/N of attention
  compute_per_rank ≈ O((S/N)² × H × D) + O(S²/N × H × D)  # mask + nomask
```

注意 PCP 的通信开销（all_gather Q、FAUpdate 合并）会抵消部分计算收益。实际 TTFT 降低比例通常小于 PCP size。

**3. 通信开销估算**

CP 的通信开销主要来自：

| 操作 | 通信量（per step） | 通信模式 |
| --- | --- | --- |
| DCP all_gather Q | `B × H × D × dtype_size` | all_gather |
| DCP all_to_all out/lse | `B × H × (D+1) × dtype_size` | all_to_all |
| PCP all_gather Q | `S/N × H × D × dtype_size` | all_gather |
| PCP FAUpdate | `S/N × H × (D+1) × dtype_size` | all_gather |

在低带宽设备（如 Atlas A2）上，通信开销可能成为瓶颈。建议通过 profiler 确认通信时间占比，如果超过 20%，考虑减小 CP size 或优化通信路径。

## 测试建议

- 短序列 CP：先确认基础正确性。
- 长序列 CP：验证 TTFT、TPOT、显存和吞吐变化。
- MLA/GQA/SFA 分模型路径验证。
- CP + graph、CP + prefix cache、CP + PD/KV transfer 的组合验证。
- 多卡 rank 配置和失败路径验证。

测试入口：

- `$PATH_TO_VLLM_ASCEND/tests/ut/attention/test_attention_cp.py`
- `$PATH_TO_VLLM_ASCEND/tests/ut/attention/test_common_cp.py`
- `$PATH_TO_VLLM_ASCEND/tests/ut/attention/test_mla_cp.py`
- `$PATH_TO_VLLM_ASCEND/tests/ut/attention/test_sfa_cp.py`
- `$PATH_TO_VLLM_ASCEND/tests/e2e/multicard/4-cards/long_sequence`

## 常见问题

Rank 配置错误：表现为启动失败、通信 timeout 或某些 rank shape 不一致。

精度异常：先看 position ids、mask、block table、seq lens 和 CP gather/reduce。

性能不达预期：看通信占比、CP size 是否过大、chunked prefill 配置、batch shape。

KV transfer 异常：看 interleave size、block size、producer/consumer CP 配置是否一致。

Graph 异常：看 capture sizes、metadata 更新和 CP 通信是否被正确处理。

## 参考入口

- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/context_parallel.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/context_parallel.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/tutorials/features/long_sequence_context_parallel_single_node.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/tutorials/features/long_sequence_context_parallel_multi_node.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/assets/cp`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_kv_cache_utils.py`

## 思考与探索

1. PCP 降低 TTFT 的直觉是什么？DCP 缓解 KV 压力的直觉是什么？
2. 为什么 CP + KV transfer 场景需要格外关注 block size 和 interleave size？
3. 一个 CP size 更大的配置性能更差，可能有哪些合理原因？

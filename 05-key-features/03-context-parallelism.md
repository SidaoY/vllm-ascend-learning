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

PCP 使用 **DualChunkSwap** 策略进行切分，核心思想是让每个 rank 同时拿到序列头部和尾部的 token，从而均衡各 rank 的 attention 计算量。

具体做法：将每个请求的序列 pad 到 `2 × pcp_size` 的倍数，然后切成 `2 × pcp_size` 个 chunk。rank r 拿到第 r 个 chunk（head）和第 `2×pcp_size - r - 1` 个 chunk（tail）。

以 PCP size = 2，一个 8 token 的序列为例：

```text
Original sequence: [t0, t1, t2, t3, t4, t5, t6, t7]

Split into 2*2 = 4 chunks:
  chunk 0: [t0, t1]    chunk 1: [t2, t3]
  chunk 2: [t4, t5]    chunk 3: [t6, t7]

DualChunkSwap assignment:
  rank 0: head=chunk 0 [t0, t1] + tail=chunk 3 [t6, t7]  → [t0, t1, t6, t7]
  rank 1: head=chunk 1 [t2, t3] + tail=chunk 2 [t4, t5]  → [t2, t3, t4, t5]
```

这样 rank 0 拿到的是序列两端，rank 1 拿到的是序列中间，因为每个 rank 都同时有 head 和 tail，整体 attention 计算量是均衡的。

以 PCP size = 4，8192 token 的序列为例：

```text
Padded to 2*4=8 multiple: 8192 is already aligned
Split into 8 chunks of 1024 tokens each

DualChunkSwap assignment:
  rank 0: head=chunk 0 [0,    1023] + tail=chunk 7 [7168, 8191]
  rank 1: head=chunk 1 [1024, 2047] + tail=chunk 6 [6144, 7167]
  rank 2: head=chunk 2 [2048, 3071] + tail=chunk 5 [5120, 6143]
  rank 3: head=chunk 3 [3072, 4095] + tail=chunk 4 [4096, 5119]
```

如果按简单的连续切分（rank 0 拿 [0,2047]，rank 3 拿 [6144,8191]），rank 0 的 token 需要 attend 到很少的历史（计算量小），而 rank 3 的 token 需要 attend 到大量历史（计算量大），负载不均衡。DualChunkSwap 通过交错分配解决了这个问题。

attention 计算时，每个 rank 的 Q 需要 attend 到两类 KV：

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

## 数据通信

CP 的通信模式在 PCP 和 DCP 中不同，但核心思路都是"本地计算 + 通信合并"。attention 算子本身不跨 rank 通信，通信发生在算子之外。

### PCP 通信流程：AllGather KV + 本地 Attention

PCP 的通信策略是：先把所有 rank 的 KV 通过 all_gather 汇聚到每个 rank，然后每个 rank 在本地完成完整的 attention 计算。

```text
Step 1: AllGather KV across PCP ranks
  Each rank produces its local KV (from DualChunkSwap split)
  all_gather KV on dim 0 → each rank gets the full KV
  Restore original token order via pcp_allgather_restore_idx

Step 2: Split Q and KV into head/tail
  Use precomputed indices (q_head_idx, q_tail_idx, etc.) to split
  Q and KV into head part and tail part for load-balanced computation

Step 3: Head attention (nomask + mask)
  head Q attends to:
    - Previous ranks' KV → nomask attention (no causal mask)
    - Own rank's KV      → mask attention (with causal mask)
  Merge nomask and mask results via _npu_attn_out_lse_update

Step 4: Tail attention (nomask + mask)
  Same as head, but for the tail part of Q

Step 5: Merge head and tail results
  Use q_full_idx to interleave head and tail outputs back to original Q order
  For hybrid attention: all_gather output and scatter back via pcp_exit_fa_scatter_idx
```

关键点：PCP 的 attention 计算完全是**本地的**——每个 rank 拿到完整 KV 后，在自己的 NPU 上完成所有 attention 计算，不需要跨 rank 的 attention 算子通信。通信只发生在 attention 之前的 KV all_gather。

### DCP 通信流程：AllGather Q → 本地 Attention → All2All + AllGather → FAUpdate

DCP 的通信策略与 PCP 相反：不汇聚 KV，而是汇聚 Q（head 维度），然后合并各 rank 的部分 attention 结果。

```text
Step 1: DCP all_gather Q (before attention)
  Per-rank Q shape: [num_tokens, num_heads, head_dim]
  After all_gather on dim=1: [num_tokens, num_heads * dcp_size, head_dim]
  Purpose: aggregate TP-split heads so each rank can compute full-head attention

Step 2: Local FIA computation
  Each rank uses local KV cache + all_gathered Q to call
  npu_fused_infer_attention_score
  → produces local attn_out and softmax_lse

Step 3: All2All (DCP) + AllGather (PCP) merge
  _process_attn_out_lse():
    - Concatenate attn_out and softmax_lse on last dim
    - If dcp_size > 1: all_to_all exchange (split heads back to each DCP rank)
    - If pcp_size > 1: all_gather (collect along batch dim across PCP ranks)
  Note: when both DCP and PCP are enabled, all2all and allgather both execute
  before FAUpdate, producing [PCP*S, DCP*H, D+1] shaped tensor

Step 4: FAUpdate merge
  _npu_attention_update():
    - Reshape [PCP*S, DCP*H, D+1] to [N, S*H, D] and [N, S*H]
    - Call torch_npu.npu_attention_update(lse_list, out_list)
    - Equivalent to: O = sum(exp(lse_i - lse_final) * O_i)
```

### PCP vs DCP 通信对比

| | PCP | DCP |
| --- | --- | --- |
| 通信内容 | all_gather KV | all_gather Q（head 维度） |
| 通信时机 | attention 之前 | attention 之前 |
| 合并方式 | 本地直接算完整 attention | FAUpdate 合并各 rank 的部分 attention 结果 |
| 设计动机 | GQA/MLA 下 KV 体积远小于 Q；KV cache 需要完整 KV | TP 风格的 head 切分聚合，各 rank 输入 token 相同 |

PCP 选择 all_gather KV 而非 all_gather Q 的原因：在 GQA 中 `num_heads >> num_kv_heads`，在 MLA 中 KV 被压缩到 `kv_lora_rank`，KV 体积远小于 Q，通信 KV 的代价更低。同时 KV cache 需要存放完整 KV，gather KV 后可以直接写入。

DCP 选择 all_gather Q 的原因：decode 阶段每个 rank 的输入 token 是相同的，DCP 将 attention head 切分到不同 rank（类似 TP 的做法）。all_gather Q 沿 head 维度聚合后，每个 rank 拿到完整 head 数的 Q，用本地 KV 计算 attention，再通过 all_to_all 把 head 维度拆分回各 rank，最后 FAUpdate 合并结果。

### FAUpdate 的数学原理

`npu_attention_update` 解决的核心问题是：多个部分 attention 输出如何正确合并为完整 attention 的结果。

假设 rank 0 和 rank 1 分别计算了 attention，得到 `(O0, lse0)` 和 `(O1, lse1)`。合并后的结果应该是：

```text
lse_final = log(exp(lse0) + exp(lse1))
O_final = (exp(lse0 - lse_final) * O0 + exp(lse1 - lse_final) * O1)
```

这等价于把所有 rank 的 KV 拼在一起做一次完整的 attention——但避免了实际拼接 KV 的显存和通信开销。FAUpdate 仅在 DCP 路径中使用，PCP 路径因为已经拿到了完整 KV，不需要 FAUpdate。

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

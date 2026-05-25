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

## CP Attention 路径

vLLM Ascend 中 CP attention 不是一个单独 kernel 名字，而是一组围绕不同 attention 类型的实现：

- `common_cp`：CP 通用工具和通信逻辑。
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

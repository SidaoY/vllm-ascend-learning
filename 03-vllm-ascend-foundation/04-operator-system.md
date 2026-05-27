# 算子体系

vLLM Ascend 的算子层是所有模型计算（attention、FFN、MoE、norm、RoPE、sampling）共享的基础设施。每一层模型计算，最终都要落在某条算子路径上——torch-npu 调用、Triton NPU kernel、CANN/aclnn custom op，或是 compile pass 中融合后的等价实现。

与社区原生（CUDA/FlashInfer/Triton GPU）相比，vLLM Ascend 在 attention 层有 `npu_fused_infer_attention_score`（FIA）和 `flash_attn_npu_v3` 等专有算子，在 FFN/MoE 层有 MC2 dispatch/combine、fused activation 等专有算子，在 norm/RoPE/linear/sampling 层也各有 Ascend 专有实现。理解算子体系，才能理解"为什么这个模型在这个 NPU 上需要这样跑"。

## 算子三层路径

vLLM Ascend 的算子层可以理解为一条从"通用"到"定制"的递进路径：

- **torch-npu**：PyTorch 官方 Ascend 适配，提供 `torch.npu.*` API 和常见算子的 NPU 实现。大部分标准运算（matmul、add、norm 等）直接走这条路径。它是覆盖率最广的一层，但不一定性能最优。
- **Triton NPU**：在 Ascend 上运行的 Triton kernel，用于对性能敏感的算子（如 MoE attention 片段、fused activation 等），提供比通用路径更优的 NPU 利用率。
- **CANN / ACLNN custom op**：通过 AscendC 或 ACLNN API 开发的自定义算子，用于 Triton 或通用路径无法高效表达的算子（如 MC2 dispatch、fused dispatch_ffn_combine 等）。这是最底层的路径，性能最高但开发成本也最高。

三条路径不是互斥的——同一个模型的不同层，可能同时走 torch-npu 的 matmul、Triton 的 fused activation、CANN 的 MoE dispatch。选择哪条路径取决于性能需求、开发成本和算子成熟度。

## 各层算子的 Ascend 差异

### MoE（Mixture of Experts）

MoE 层是 vLLM Ascend 与社区实现差异最大的部分之一。上游 vLLM 的 MoE 依赖 CUDA fused_moe kernel 和标准的 all-to-all 通信，而 vLLM Ascend 有自己完整的 MoE 算子体系。

**路由与 Expert 选择**

`AscendUnquantizedFusedMoEMethod.apply()` 完全覆盖了上游的 MoE 前向逻辑。路由阶段由 `select_experts()` 完成，支持 grouped topk、renormalize、e_score_correction_bias 和 zero expert 等 Ascend 扩展。

**Dispatch / Combine**

这是差异最大的部分。上游 vLLM 使用通用的 all-to-all 通信来做 token dispatch/combine，而 vLLM Ascend 有四套不同的 dispatch/combine 实现：

| 通信方式 | 实现类 | 适用场景 |
|---|---|---|
| MC2（fused dispatch+ffn+combine） | `TokenDispatcherWithMC2` + `FusedMC2CommImpl` | 小 batch，dispatch/ffn/combine 融合为一个 NPU 算子，延迟最低 |
| MC2（分离 dispatch+ffn+combine） | `TokenDispatcherWithMC2` + `MC2CommImpl` | 中等 batch，dispatch、ffn、combine 分步执行 |
| All2All | `TokenDispatcherWithAll2AllV` + `AlltoAllCommImpl` | 大 batch（超过 MC2 512 token/rank 限制） |
| AllGather | `TokenDispatcherWithAllGather` + `AllGatherCommImpl` | 小 expert 数或特殊拓扑 |

MC2 dispatch 底层调用的是 `torch_npu.npu_moe_distribute_dispatch_v2`——这是 Ascend NPU 的专有算子，在 CUDA 上没有对应物。FusedMC2 更是将 dispatch、FFN 计算和 combine 融合为一个 `dispatch_ffn_combine` 算子。

**Prepare / Finalize**

`PrepareAndFinalize` 负责 MoE 计算前后的张量准备和收尾：
- `PrepareAndFinalizeWithMC2`：pad 到 TP size、按 TP rank 切片、生成 mc2_mask，用于 MC2 通信路径。
- `PrepareAndFinalizeWithAll2All`：pad 到 TP size、按 TP rank 切片，用于 All2All 通信路径。

finalize 阶段负责 all-gather 切片、unpad 到原始 token 数、可选 all-reduce。

**权重格式**

Ascend MoE 权重需要转换为 Fractal NZ 格式（`ACL_FORMAT_FRACTAL_NZ`）。这是因为 `dispatch_ffn_combine` 融合算子只支持 NZ 格式。一旦底层算子更新为支持 ND 格式，这个转换逻辑就可以移除。

### FFN / Linear

标准 FFN 层的 matmul 大多走 torch-npu 路径，但 vLLM Ascend 提供了自己的 `linear.py` 和 `linear_op.py`，用于处理 TP 切分、weight prefetch 和特殊 layout。`layer_shard_linear.py` 处理层间切分的 linear 操作。

### Activation

`activation.py` 提供 Ascend 专有的 fused activation 实现（如 SiLU、GELU 的 fused 版本），避免多次 kernel launch 的开销。

### Norm

`layernorm.py` 提供 Ascend 上的 RMSNorm 和 LayerNorm 实现。这些看似简单的算子，在 graph mode 和动态 batch 下需要特殊处理。

### RoPE

`rotary_embedding.py` 和 `rope_dsv4.py` 提供 Ascend 上的 rotary position embedding 实现。DeepSeek-V4 的 RoPE 有特殊的 multi-dimensional 需求，`rope_dsv4.py` 专门处理这一场景。

### Sampling

Sampling 层（greedy、top-k、top-p、rejection sampler）在 Ascend 上也需要适配。虽然大部分走 torch-npu，但某些高性能采样路径需要 CANN custom op。

### 其他 Custom Op

vLLM Ascend 还有一系列模型专用的 custom op：

| 文件 | 用途 |
|---|---|
| `mla.py` | Multi-head Latent Attention 的 Ascend 算子 |
| `dsa.py` | Dynamic Sparse Attention 的 Ascend 算子 |
| `mhc.py` | Multi-head Cross Attention 或类似结构 |
| `gdn.py` | Generalized Divisive Normalization |
| `bailing_moe_linear_attn.py` | Bailing 模型的 MoE Linear Attention |
| `mm_encoder_attention.py` | 多模态 encoder attention |
| `qwen2_decoder.py` | Qwen2 模型的 NPU 优化 decoder |
| `conv.py` | 卷积算子 |

## 新增或修改算子的关注点

新增或修改算子时，要同时考虑：

- eager 路径是否正确。
- graph/compile 路径是否支持。
- dtype、layout、shape 约束是否清楚。
- 是否需要 workspace 或特殊内存对齐（如 Fractal NZ）。
- 单卡、多卡、不同硬件（A2/A3/A5）是否都适用。
- fallback 路径是什么。
- 如何做精度对齐。
- 如何做性能 benchmark。
- 是否需要用户文档或 feature matrix 更新。

不要只把 op 跑通。推理服务里的 op 还要经受动态 batch、长上下文、多卡、graph、量化、异常输入和 CI/nightly 的考验。

## 代码入口

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops`：算子主目录
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/fused_moe`：MoE dispatch/combine/routing
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/triton`：Triton NPU kernel
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/_cann_ops_custom`：CANN custom op
- `$PATH_TO_VLLM_ASCEND/csrc`：C++ 扩展
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation`：图编译相关
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md`：custom op 开发文档

建议搜索关键词：`fused_moe`、`dispatch`、`combine`、`MC2`、`All2All`、`custom op`、`aclnn`、`Triton`、`CANN`、`Fractal NZ`。

## 常见问题定位

- MoE 输出异常：先确认走的是哪个通信路径（MC2/All2All/AllGather），再检查 prepare/finalize 的 pad/slice/group 是否正确。
- FusedMC2 异常但分离 MC2 正常：重点看 `dispatch_ffn_combine` 融合算子的输入格式（权重是否为 NZ、scale 是否为空 tensor）。
- eager 正常、graph 异常：重点看 metadata 更新、shape padding、unsupported op。
- 单卡正常、多卡异常：重点看 MC2/ep group、TP 切分、通信和 all-reduce/all-gather。
- 只有某个模型异常：重点看特殊 RoPE、特殊 expert 配置、特殊 cache spec。
- 性能异常：确认是否走到预期路径，是否发生 fallback，shape 是否命中高效路径，是否有不必要的格式转换。

## 思考与探索

1. 为什么 MC2 路径限制 512 token/rank？超过这个限制时应该 fallback 到哪个路径？
2. `PrepareAndFinalizeWithMC2` 和 `PrepareAndFinalizeWithAll2All` 的核心区别是什么？为什么 MC2 需要 mc2_mask 而 All2All 不需要？
3. 如果一个 custom op 在 eager 下正确但 graph 下失败，你会先检查哪些约束？
4. Fractal NZ 格式转换是永久性的还是可以撤销的？什么条件下可以移除这个转换？
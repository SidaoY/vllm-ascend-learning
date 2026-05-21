# Attention 与算子适配

Attention 和算子适配是 vLLM Ascend 性能和正确性的核心。vLLM 提供 attention backend 抽象，vLLM Ascend 在这个抽象下实现 NPU 侧 dense attention、MLA、SFA、FA3、CP attention，并通过 torch-npu、Triton NPU、CANN/custom op 等路径接入底层算子。

## Attention 路径的定位

常见 attention 路径可以先这样理解：

- Dense attention：普通 MHA/GQA 模型的主路径。
- MLA：面向 Multi-head Latent Attention 等特殊 KV 表示的路径，常见于 DeepSeek 类模型。
- SFA：Sparse Flash Attention，面向 DSA / sparse attention 模型，例如 DeepSeek-V3.2、GLM5 等。
- FA3：Flash Attention 3 路径，主要用于训练/推理 attention 实现一致性，不等同于 KV cache 量化。
- CP attention：为 context parallelism 设计的 attention 路径，包括 dense、MLA、SFA 等组合形态。

这些名字不要混成一类。它们解决的问题、适用模型、metadata 和 kernel 约束都不同。

## Attention Backend 消费什么

Ascend attention backend 通常需要：

- Q/K/V 或 MLA 相关输入。
- KV cache buffer。
- block table。
- slot mapping。
- seq lens、query lens、query start loc。
- mask 或 causal 信息。
- dtype、layout、block size。
- graph 或 CP 场景下的额外 metadata。

如果这些 metadata 和 kernel 预期不一致，可能出现精度错误、shape mismatch、非法内存访问、kernel fallback 或性能退化。

## Prefill 与 Decode

Prefill 和 decode 的 NPU 路径可能差异很大：

- Prefill token 数多，更关注大矩阵计算和长序列处理。
- Decode token 数少，更关注 KV cache 读取、block table、graph replay 和小 batch overhead。
- Chunked prefill 让 prefill 也进入多 step、动态 shape 的调度场景。
- CP 会改变 prefill/decode 的通信和切分方式。

因此 attention bug 需要先确认发生在 prefill、decode、chunked prefill 还是混合 batch。

## Ops 体系

vLLM Ascend 的 ops 层包含多类实现：

- Python 封装的 torch-npu / torch 调用。
- Triton NPU kernel。
- CANN / ACLNN custom op。
- MoE、linear、norm、RoPE、sampling、rejection sampler 等高频算子。
- 图融合或 compile pass 中使用的 pattern 和 replacement。

新增或修改算子时，要同时考虑：

- eager 路径是否正确。
- graph/compile 路径是否支持。
- dtype、layout、shape 约束是否清楚。
- 单卡、多卡、不同硬件是否都适用。
- 精度测试和性能测试是否覆盖。

## Custom Op 开发关注点

新增 custom op 时建议先回答：

- 为什么现有 torch-npu / Triton / CANN 路径不够用？
- 输入输出 shape、dtype、layout 是什么？
- 是否需要 workspace 或特殊内存对齐？
- 是否支持 graph capture？
- fallback 路径是什么？
- 如何做精度对齐？
- 如何做性能 benchmark？
- 是否需要用户文档或 feature matrix 更新？

不要只把 op 跑通。推理服务里的 op 还要经受动态 batch、长上下文、多卡、graph、量化、异常输入和 CI/nightly。

## Attention 和量化

量化会影响 attention 路径，尤其是 KV cache quantization、C8、sparse C8 等场景。这里不展开量化算法，只要记住：

- cache dtype 变化会影响 KV cache spec 和 backend。
- scale/offset metadata 可能进入 attention 或 KV transfer。
- 量化模型的慢路径、fallback 或精度差异需要单独验证。

## 代码入口

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention/context_parallel`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/triton`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/_cann_ops_custom`
- `$PATH_TO_VLLM_ASCEND/csrc`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/add_custom_aclnn_op.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/flash_attention.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/graph_mode.md`

建议搜索关键词：`AscendAttention`、`MLA`、`SFA`、`FA3`、`context_parallel`、`custom op`、`aclnn`、`Triton`、`CANN`。

## 常见问题定位

- prefill 正常、decode 异常：重点看 KV cache layout、block table、slot mapping、seq lens。
- eager 正常、graph 异常：重点看 metadata 更新、shape padding、unsupported op。
- 单卡正常、多卡异常：重点看 rank group、CP/TP 切分、通信和 all-reduce/all-gather。
- 只有某个模型异常：重点看 MLA/SFA/特殊 RoPE/特殊 cache spec。
- 性能异常：确认是否走到预期 backend，是否发生 fallback，shape 是否命中高效路径。

## 思考与探索

1. 为什么 SFA 不应被理解为普通 Ascend attention kernel 的统称？
2. 新增 attention backend 时，model runner 需要配合提供哪些 metadata？
3. 如果一个 custom op 在 eager 下正确但 graph 下失败，你会先检查哪些约束？

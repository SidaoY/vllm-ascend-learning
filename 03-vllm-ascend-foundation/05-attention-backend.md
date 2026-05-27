# Attention 后端

Attention 后端是 vLLM 中具体执行 attention 计算的抽象层。vLLM Ascend 在这个抽象下实现了 Ascend NPU 专属的多种 attention 路径。这些路径消费上一章描述的算子基础设施（torch-npu、Triton NPU、CANN custom op）来完成实际计算。

Attention 后端是 vLLM Ascend 中值得重点关注的模块。当前模型结构的主要变化集中在 attention 层（MLA、DSA、sliding window 等），FFN 层基本不再有新的结构出现；同时 Ascend 侧的 attention 算子（FIA、flash_attn_npu_v3）与社区实现差异较大，metadata 约定、kernel 约束和调度逻辑都需要专门适配。

## Attention 路径概览

- **Dense attention**：普通 MHA/GQA 模型的主路径，对应 `AscendAttentionBackend`。这是最通用的路径，覆盖大部分开源模型。
- **MLA**：Multi-head Latent Attention，面向特殊 KV 表示（低秩压缩的 KV）的路径，常见于 DeepSeek-V2/V3 类模型，对应 `AscendMLABackend`。
- **SFA / DSA**：两者都是基于 indexer 的稀疏 attention 结构。SFA（Sparse Flash Attention）使用 `npu_lightning_indexer` + `npu_sparse_flash_attention`，对应 `AscendSFABackend`，常见于 DeepSeek-V3.2、GLM5 等模型。DSA（Dynamic Sparse Attention）使用 `npu_lightning_indexer` + `npu_sparse_attn_sharedkv`，对应 `AscendDSABackend`，常见于 DeepSeek-V4 等模型。两者都是模型结构层面的稀疏 attention，不是单纯的加速手段。
- **FA3**：Ascend 侧的 Flash Attention 路径（基于 `flash_attn_npu_v3`），对应用户可选择启用的 `AscendFABackend`。不等同于 KV cache 量化，FA3 关注的是 attention 计算本身的一致性和性能。
- **CP attention**：为 context parallelism 设计的 attention 路径，包括 dense、MLA、SFA 等组合形态。CP 下先 all_gather Q，各 rank 本地计算 attention，再 all_gather output 和 lse 后通过 `npu_attention_update` 合并——attention 算子本身不跨 rank 通信。

这些名字不要混成一类。它们解决的问题、适用模型、metadata 和 kernel 约束都不同。

## Attention Backend 与算子的关系

Attention backend 不直接实现矩阵乘法或 softmax。它负责：

1. 接收 model runner 传来的 Q/K/V、KV cache、block table、slot mapping、seq lens 等 metadata。
2. 根据 attention 类型（dense/MLA/SFA/DSA/FA3/CP）选择合适的计算路径。
3. 调用算子层的 torch-npu / Triton NPU / CANN custom op 来实际完成 attention score、softmax、KV cache store/load 等计算。

如果 metadata 和 kernel 预期不一致，可能出现精度错误、shape mismatch、非法内存访问、kernel fallback 或性能退化。

## Attention Backend 消费的 Metadata

Ascend attention backend 通常需要以下输入：

- Q/K/V 或 MLA 相关输入。
- KV cache buffer。
- block table：逻辑 block 到物理 block 的映射。
- slot mapping：新 token 的 K/V 写入 KV cache 的位置。
- seq lens、query lens、query start loc：各序列的长度和对齐信息。
- mask 或 causal 信息。
- dtype、layout、block size。
- graph 或 CP 场景下的额外 metadata。

不同 attention 路径对这些 metadata 的依赖程度不同。例如 MLA 需要知道 latent KV 的维度信息，CP attention 需要知道 rank group 和通信拓扑。

## Prefill 与 Decode

Prefill 和 decode 的 NPU attention 路径可能差异很大：

- Prefill token 数多，更关注大矩阵计算和长序列处理。dense attention 在 prefill 时通常使用 FlashAttention 类的高效实现。
- Decode token 数少（通常每个请求只有 1 个 token），更关注 KV cache 读取、block table 查询、graph replay 和小 batch overhead。
- Chunked prefill 让 prefill 也进入多 step、动态 shape 的调度场景，metadata 在每个 chunk 之间需要正确更新。
- CP 会改变 prefill/decode 的通信和切分方式——prefill CP 需要跨 rank 交换 Q/K/V，decode CP 需要访问其他 rank 的 KV cache。

因此 attention bug 需要先确认发生在 prefill、decode、chunked prefill 还是混合 batch。

## Attention 和量化

量化会影响 attention 路径，尤其是 KV cache quantization、C8、sparse C8 等场景：

- cache dtype 变化会影响 KV cache spec 和 backend 选择。
- scale/offset metadata 可能进入 attention 计算或 KV transfer 流程。
- 量化模型的慢路径、fallback 或精度差异需要单独验证。

## 代码入口

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`：attention backend 主目录
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention/context_parallel`：CP attention
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/mla.py`：MLA 算子
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/dsa.py`：DSA 算子
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/flash_attention.md`：FA3 用户文档
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/graph_mode.md`：图模式文档

建议搜索关键词：`AscendAttention`、`MLA`、`SFA`、`DSA`、`FA3`、`context_parallel`。

## 常见问题定位

- prefill 正常、decode 异常：重点看 KV cache layout、block table、slot mapping、seq lens。
- eager 正常、graph 异常：重点看 metadata 更新、shape padding、unsupported op。
- 单卡正常、多卡异常：重点看 rank group、CP/TP 切分、通信和 all-reduce/all-gather。
- 只有某个模型异常：重点看 MLA/SFA/DSA/特殊 RoPE/特殊 cache spec。
- 性能异常：确认是否走到预期 backend，是否发生 fallback，shape 是否命中高效路径。

## 思考与探索

1. 为什么 SFA 不应被理解为普通 Ascend attention kernel 的统称？
2. 新增 attention backend 时，model runner 需要配合提供哪些 metadata？
3. MLA 的 KV 表示和标准 MHA 的 KV 表示有什么本质区别？这对 attention backend 的实现意味着什么？
4. CP attention 在 prefill 和 decode 阶段的通信模式有什么不同？
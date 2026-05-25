# 资料索引

资料索引分成三类：本地仓库文档、vLLM Ascend 文档、外部背景资料。外部资料用于建立概念，不要求新人一开始全部读完；实际开发时仍以当前仓库代码和文档为准。

## vLLM 文档

- `$PATH_TO_VLLM/docs/design/arch_overview.md`
- `$PATH_TO_VLLM/docs/design/paged_attention.md`
- `$PATH_TO_VLLM/docs/design/prefix_caching.md`
- `$PATH_TO_VLLM/docs/design/attention_backends.md`
- `$PATH_TO_VLLM/docs/design/cuda_graphs.md`
- `$PATH_TO_VLLM/docs/design/torch_compile.md`
- `$PATH_TO_VLLM/docs/design/hybrid_kv_cache_manager.md`
- `$PATH_TO_VLLM/docs/features/speculative_decoding/README.md`
- `$PATH_TO_VLLM/docs/features/tool_calling.md`
- `$PATH_TO_VLLM/docs/features/structured_outputs.md`
- `$PATH_TO_VLLM/docs/serving/parallelism_scaling.md`
- `$PATH_TO_VLLM/docs/benchmarking`
- `$PATH_TO_VLLM/docs/usage/metrics.md`
- `$PATH_TO_VLLM/docs/usage/troubleshooting.md`

## vLLM Ascend 文档

- `$PATH_TO_VLLM_ASCEND/docs/source/quick_start.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/installation.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/faqs.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/support_matrix`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/configuration/env_vars.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/configuration/additional_config.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/context_parallel.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/speculative_decoding.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/kv_pool.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/eplb_swift_balancer.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/flash_attention.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/graph_mode.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/external_dp.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/epd_disaggregation.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/Multi_Token_Prediction.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/patch.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/context_parallel.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/ACL_Graph.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/eplb_swift_balancer.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/testing.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/e2e_ci_test.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/nightly_ci_test.md`

## 外部背景资料

- Transformer：`Attention Is All You Need`，理解 self-attention、multi-head attention、position encoding 的基础论文。
- vLLM / PagedAttention：vLLM 论文和项目文档，理解 continuous batching、PagedAttention、KV cache block 管理。
- Speculative decoding：原始 speculative decoding 论文、EAGLE、Medusa、MTP、Suffix Decoding 等资料，帮助理解 proposer/verify/accept-reject 思路。
- MoE：Switch Transformer、GShard、Mixtral、DeepSeekMoE 等资料，帮助理解 router、expert、EP 和 expert load balance。
- Long context / context parallel：Ring Attention、sequence parallel、context parallel 相关资料，帮助理解为什么长序列需要按 context 切分。
- Ascend / CANN / torch-npu：Ascend 官方 CANN、torch-npu、HCCL、ACL graph、custom op 文档，开发 NPU backend、custom op 和通信问题时需要查阅。

## 阅读建议

基础概念先读本学习资料和 vLLM 文档；遇到具体 Ascend 行为，再读 vLLM Ascend user guide 和 developer guide；遇到底层算子、通信或图模式问题，再查 Ascend 官方资料。这样能避免一开始被大量硬件和系统细节淹没。

## 本仓库内部索引

- [学习路线图](../00-orientation/00-learning-map.md)：按阶段规划学习路径。
- [仓库地图](../00-orientation/01-repo-map.md)：仓库关系、目录速览、新人读代码推荐路径。
- [环境与工作流](../00-orientation/02-environment-and-workflow.md)：环境搭建、命令、日志、常见错误。
- [代码阅读索引](01-code-reading-index.md)：按主题列出代码入口和搜索关键词。
- [术语表](00-glossary.md)：快速确认术语含义。
- [实验记录模板](03-experiment-template.md)：性能实验和复盘的实验模板。

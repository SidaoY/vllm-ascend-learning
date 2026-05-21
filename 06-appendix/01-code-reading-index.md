# 代码阅读索引

本索引按主题列出相对稳定的代码和文档入口。vLLM 和 vLLM Ascend 都迭代很快，阅读时优先抓住目录边界和数据流，不要把某个函数名当成长期不变的接口。

## 阅读方法

建议按“入口 -> 状态 -> 执行 -> 输出”的顺序读：

1. 先找用户请求或配置从哪里进入。
2. 再找请求状态、调度输出和 metadata 在哪里构造。
3. 然后看 worker/model runner 如何把 metadata 变成设备侧输入。
4. 最后看 attention、ops、sampling、KV transfer 等具体执行路径。

遇到版本差异时，用 `rg` 搜索关键词比沿着旧行号跳转更可靠。

## 请求入口

- `$PATH_TO_VLLM/vllm/entrypoints/openai`
- `$PATH_TO_VLLM/vllm/entrypoints/llm.py`
- `$PATH_TO_VLLM/vllm/v1/engine/llm_engine.py`
- `$PATH_TO_VLLM/vllm/v1/engine/async_llm.py`
- `$PATH_TO_VLLM/vllm/v1/engine/input_processor.py`
- `$PATH_TO_VLLM/vllm/v1/engine/output_processor.py`

建议搜索关键词：`OpenAIServing`、`AsyncLLM`、`EngineCoreRequest`、`SamplingParams`。

## 调度

- `$PATH_TO_VLLM/vllm/v1/core/sched/scheduler.py`
- `$PATH_TO_VLLM/vllm/v1/core/sched/output.py`
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/request.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_balance_schedule.py`

建议搜索关键词：`SchedulerOutput`、`token budget`、`waiting`、`running`、`chunked prefill`、`prefix cache`。

## KV Cache

- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`
- `$PATH_TO_VLLM/vllm/v1/kv_cache_interface.py`
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_utils.py`
- `$PATH_TO_VLLM/vllm/v1/kv_offload`
- `$PATH_TO_VLLM/vllm/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/block_table.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/kv_offload`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/quantization/methods/kv_c8.py`

建议搜索关键词：`KVCacheManager`、`KVCacheSpec`、`block table`、`slot mapping`、`KVConnector`、`AscendStore`。

## Worker / Model Runner

- `$PATH_TO_VLLM/vllm/v1/worker`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/worker.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/model_runner_v1.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/npu_input_batch.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/pcp_utils.py`

建议搜索关键词：`execute_model`、`model runner`、`input batch`、`attention metadata`。

## Platform / Config

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/platform.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/envs.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ascend_config.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/utils.py`

建议搜索关键词：`NPUPlatform`、`additional_config`、`SOC_VERSION`、`environment variable`。

## Patch

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/__init__.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/patch.md`

建议搜索关键词：`patch`、`monkey patch`、`platform patch`、`worker patch`。

## Attention

- `$PATH_TO_VLLM/vllm/v1/attention`
- `$PATH_TO_VLLM/vllm/model_executor/layers/attention`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention/context_parallel`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/triton`
- `$PATH_TO_VLLM_ASCEND/csrc`

建议搜索关键词：`MLA`、`SFA`、`FA3`、`block_size`、`slot_mapping`、`actual_seq_lengths`、`aclnn`。

## Graph Mode

- `$PATH_TO_VLLM/docs/design/cuda_graphs.md`
- `$PATH_TO_VLLM/docs/design/torch_compile.md`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/acl_graph.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/graph_fusion_pass_manager.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/passes`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/ACL_Graph.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/graph_mode.md`

建议搜索关键词：`graph capture`、`replay`、`cudagraph_capture_sizes`、`ACL graph`、`full graph`。

## Spec Decode

- `$PATH_TO_VLLM/vllm/v1/spec_decode`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/spec_decode`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/triton/spec_decode`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker/patch_rejection_sampler.py`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/speculative_decoding.md`
- `$PATH_TO_VLLM_ASCEND/tests/ut/spec_decode`

建议搜索关键词：`proposer`、`draft token`、`acceptance rate`、`rejection sampler`、`lookahead`。

## CP

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention/context_parallel`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/pcp_utils.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/passes/sequence_parallelism.py`
- `$PATH_TO_VLLM_ASCEND/docs/source/assets/cp`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/context_parallel.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/context_parallel.md`

建议搜索关键词：`prefill_context_parallel_size`、`decode_context_parallel_size`、`PCP`、`DCP`、`cp_kv_cache_interleave_size`。

## EPLB / Load Balancing

- `$PATH_TO_VLLM/vllm/distributed/eplb`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb/core/policy`
- `$PATH_TO_VLLM_ASCEND/examples/external_online_dp`
- `$PATH_TO_VLLM_ASCEND/examples/disaggregated_prefill_v1`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_balance_schedule.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_multiproc_executor.py`

建议搜索关键词：`expert_map`、`expert_heat`、`DYNAMIC_EPLB`、`Swift Balancer`、`load balance`。

## Distributed / KV Transfer

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/parallel_state.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/device_communicators`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/examples/disaggregated_prefill_v1`
- `$PATH_TO_VLLM_ASCEND/examples/epd_disaggregated`

建议搜索关键词：`HCCL`、`parallel_state`、`Mooncake`、`AscendStore`、`kv_producer`、`kv_consumer`。

## Development Practice

- `$PATH_TO_VLLM_ASCEND/tests`
- `$PATH_TO_VLLM_ASCEND/.github/workflows`
- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/config.yaml`
- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/run_suite.py`
- `$PATH_TO_VLLM/docs/benchmarking`
- `$PATH_TO_VLLM_ASCEND/benchmarks`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug`

建议搜索关键词：`VLLM_COMMIT`、`nightly`、`smart e2e`、`profile`、`msprobe`、`benchmark`。

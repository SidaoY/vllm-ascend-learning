# 仓库地图

## 写作目标

解释 `/home/yht/code/ascend` 下两份仓库的配套关系，以及新人读代码时应优先关注的目录。

## vLLM 重点目录

- `$PATH_TO_VLLM/vllm/entrypoints`：OpenAI API、CLI、离线推理等入口。
- `$PATH_TO_VLLM/vllm/v1/engine`：V1 engine、async client、core client、output processor。
- `$PATH_TO_VLLM/vllm/v1/core/sched`：调度器和调度输出。
- `$PATH_TO_VLLM/vllm/v1/core/kv_cache_manager.py`：KV cache block 管理。
- `$PATH_TO_VLLM/vllm/v1/attention`：V1 attention backend。
- `$PATH_TO_VLLM/vllm/v1/spec_decode`：投机推理核心模块。
- `$PATH_TO_VLLM/vllm/v1/worker`：worker 和模型执行相关逻辑。
- `$PATH_TO_VLLM/vllm/model_executor`：模型定义、layer、attention、loader、sampling 相关基础设施。
- `$PATH_TO_VLLM/docs/design`：架构设计文档。
- `$PATH_TO_VLLM/docs/features`：功能使用文档。
- `$PATH_TO_VLLM/docs/benchmarking`：benchmark 和性能观测入口。

## vLLM Ascend 重点目录

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/platform.py`：NPU platform 注册与能力声明。
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/utils.py`：patch 入口、版本判断、环境辅助函数。
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch`：platform patch 和 worker patch。
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker`：NPU worker、model runner、block table、input batch。
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`：Ascend attention backend、MLA、SFA、CP。
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed`：parallel state、KV transfer、通信适配。
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb`：专家并行负载均衡。
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/kv_offload`：KV offload。
- `$PATH_TO_VLLM_ASCEND/csrc`：自定义算子和底层实现。
- `$PATH_TO_VLLM_ASCEND/tests`：单测、e2e、nightly。
- `$PATH_TO_VLLM_ASCEND/.github/workflows`：CI 和 nightly workflow。
- `$PATH_TO_VLLM_ASCEND/docs/source`：已有用户文档和设计文档。

## 待补内容

- 一张仓库关系图。
- 新人第一次读代码推荐路径。
- 常用 `rg` 搜索关键词。
- vLLM 版本和 vLLM Ascend 分支/commit 的对应关系。

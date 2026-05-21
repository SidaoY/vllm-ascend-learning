# 开发工作流

vLLM Ascend 的开发不是“看到问题就找一个文件改掉”。它同时连接上游 vLLM、Ascend 后端适配、CANN/torch-npu 能力、CI/nightly 环境和模型特性。一个可靠的改动，通常要先判断它属于哪一层，再决定是改 vLLM Ascend 原生代码、做 patch，还是推动上游 vLLM 提供更稳定的扩展点。

## 先问三个问题

动手前建议先把问题拆成三句话：

1. 这个问题发生在哪个阶段：启动、模型加载、prefill、decode、sampling、KV transfer、graph capture、通信、输出后处理？
2. 这个问题影响哪类用户：单卡、多卡、长上下文、MoE、PD 分离、某个模型家族、某个硬件版本、某个 nightly workflow？
3. 这个问题的边界在哪里：只影响 NPU backend，还是 vLLM 通用逻辑也应该变化？

这三个问题能帮助你避免把一个局部适配写成全局行为，也能避免把一个上游通用问题长期留在 vLLM Ascend patch 里。

## 改动分层

可以按下面这张表定位改动层次：

| 层级 | 典型内容 | 常见代码入口 |
| --- | --- | --- |
| API / entrypoint | OpenAI API、offline LLM、输入输出格式 | `$PATH_TO_VLLM/vllm/entrypoints` |
| Engine / processor | 请求生命周期、输入处理、输出处理 | `$PATH_TO_VLLM/vllm/v1/engine` |
| Scheduler / KV manager | 调度、token budget、KV block、prefix cache | `$PATH_TO_VLLM/vllm/v1/core` |
| Executor | 多进程、分布式执行、worker 创建 | `$PATH_TO_VLLM/vllm/v1/executor` |
| Ascend platform | NPU 能力声明、环境变量、配置扩展 | `$PATH_TO_VLLM_ASCEND/vllm_ascend/platform.py`、`$PATH_TO_VLLM_ASCEND/vllm_ascend/envs.py` |
| Ascend worker | NPU input batch、block table、model runner | `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker` |
| Attention / ops | attention backend、custom op、Triton/CANN 路径 | `$PATH_TO_VLLM_ASCEND/vllm_ascend/attention`、`$PATH_TO_VLLM_ASCEND/vllm_ascend/ops` |
| Distributed | HCCL、rank group、KV transfer、PD 分离 | `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed` |
| Patch | 临时兼容、上游扩展点不足、模型特殊适配 | `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch` |
| Tests / docs | 单测、e2e、nightly、用户文档、设计文档 | `$PATH_TO_VLLM_ASCEND/tests`、`$PATH_TO_VLLM_ASCEND/docs` |

真正的改动可能跨多层。比如 CP 并行会涉及 attention metadata、worker input、distributed group、测试和文档；投机推理会涉及 proposer、verify、sampling、graph 和模型特殊路径。

## 改哪里

可以用下面的判断方式：

- 如果问题在所有设备上都成立，并且属于 vLLM 的通用抽象，优先考虑上游 vLLM。
- 如果问题只与 Ascend NPU 的能力、layout、kernel、通信或环境相关，优先放在 vLLM Ascend 原生模块。
- 如果上游暂时没有扩展点，但 vLLM Ascend 必须跟进某个版本或模型，可以使用 patch。
- 如果改动只是绕过某个具体失败，需要先确认它不会改变默认语义，也不会隐藏更底层的问题。

Patch 不是低质量代码的同义词。好的 patch 应该有清晰原因、作用时机、版本边界、测试覆盖和退出计划。随着上游提供扩展点或问题修复，patch 应该被收敛或删除。

## Patch 生命周期

新增 patch 时建议记录：

- 为什么不能直接改 vLLM Ascend 原生代码。
- 为什么当前不能等待上游 vLLM。
- patch 在 platform 阶段还是 worker 阶段生效。
- patch 依赖哪些 vLLM 行为或配置。
- 如何测试它没有影响无关场景。
- 未来上游合入后如何删除它。

阅读入口：

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/__init__.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/patch.md`

## 一次改动的推荐节奏

1. 复现问题：尽量得到一个最小命令、最小模型或最小测试。
2. 定位层级：判断问题在 API、scheduler、worker、attention、ops、distributed 还是 patch。
3. 读相邻代码：先看同目录已有实现、测试和文档，尽量沿用现有风格。
4. 缩小改动：先让核心路径正确，再补兼容分支。
5. 补测试：单测覆盖判断逻辑，e2e 覆盖用户可见行为，必要时加 nightly 配置。
6. 做性能确认：涉及 hot path、kernel、graph、通信、调度时，要保留 baseline 和实验组结果。
7. 补文档：用户可见配置、限制、环境变量和故障处理都应该写清楚。
8. 准备 review：说明问题、方案、验证范围、风险和后续计划。

## Review 自检清单

提交前可以快速过一遍：

- 默认行为是否变化，变化是否必要。
- 是否影响非 Ascend 平台或不相关模型。
- 是否引入新的环境变量、配置项或 patch 生效条件。
- 错误日志是否能帮助定位问题，而不是只抛出底层异常。
- shape、dtype、device、rank、block size 的约束是否明确。
- eager、graph、单卡、多卡、长上下文是否需要分别验证。
- 是否覆盖了失败路径，而不只是 happy path。
- 是否有性能风险，是否需要 benchmark 或 profile 数据。
- 是否需要更新 feature matrix、用户文档或设计文档。

## 实验记录和问题复盘

性能、精度、nightly 失败这类问题不要只留下“修好了”的结论。建议至少记录：

- 环境：vLLM commit、vLLM Ascend commit、CANN、torch-npu、硬件、镜像。
- Workload：模型、输入长度、输出长度、并发、request rate、采样参数。
- 指标：TTFT、TPOT、ITL、吞吐、NPU 显存、KV cache 使用率、错误率。
- 对照：baseline、实验组、差异、可疑变量。
- 结论：确认原因、修复方式、残留风险、后续观察项。

可以直接使用 [实验记录模板](../06-appendix/03-experiment-template.md)。

## 思考与探索

1. 一个 scheduler patch 如果让某个长上下文场景更快，可能会让哪些普通场景变慢？
2. 一个 custom op 在本地单卡通过后，为什么仍然需要考虑 graph 和多卡？
3. 如果一个 patch 已经存在三个月，它更像临时兼容，还是已经暴露了上游扩展点缺失？

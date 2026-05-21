# 性能观测与调优

性能调优的第一步不是改参数，而是把问题说清楚：是在追求更低 TTFT、更低 TPOT、更高吞吐、更稳定延迟，还是更高资源利用率？这些目标经常互相牵制。比如更大的 batch 可能提高吞吐，但让个别请求等待更久；更激进的 graph 捕获可能降低 kernel overhead，但对动态 shape 更敏感。

## 常用指标

| 指标 | 含义 | 常见用途 |
| --- | --- | --- |
| TTFT | Time To First Token，从请求进入到首 token 输出 | 衡量 prefill、排队、模型加载后首轮执行 |
| TPOT | Time Per Output Token，输出 token 平均耗时 | 衡量 decode 阶段体验 |
| ITL | Inter-token Latency，相邻输出 token 间隔 | 观察流式输出抖动 |
| E2E latency | 请求端到端耗时 | 用户整体体验 |
| Request throughput | 单位时间完成请求数 | 服务吞吐 |
| Token throughput | 单位时间处理 token 数 | 模型执行吞吐 |
| Goodput | 满足延迟 SLO 的有效吞吐 | 线上容量评估 |
| KV cache utilization | KV cache block 使用率和浪费情况 | 判断内存瓶颈和调度压力 |
| NPU memory | 权重、KV cache、workspace、graph 等显存占用 | 判断 OOM 和容量 |

看指标时要区分 prefill 和 decode。长 prompt 场景主要看 TTFT、prefill throughput 和 KV cache 压力；高并发短输出场景更容易暴露 scheduler、decode batch、graph replay 和 sampling overhead。

## Workload 设计

一次有意义的 benchmark 至少要固定：

- 模型和 checkpoint。
- tokenizer/chat template。
- 输入长度分布，而不是只写平均长度。
- 输出长度分布和 stop 条件。
- 并发、request rate、请求数、warmup。
- sampling 参数，例如 temperature、top_p、max_tokens。
- 并行策略，例如 TP、DP、PP、CP、EP。
- graph、prefix caching、chunked prefill、spec decode 等开关。
- vLLM commit、vLLM Ascend commit、CANN、torch-npu、硬件和镜像。

如果 workload 不稳定，结论也会不稳定。特别是线上 trace、随机 prompt 和开放式生成，最好保留数据集版本和随机种子。

## Benchmark 常用入口

vLLM 和 vLLM Ascend 都有 benchmark 入口：

- `$PATH_TO_VLLM/benchmarks`
- `$PATH_TO_VLLM/docs/benchmarking`
- `$PATH_TO_VLLM/docs/usage/metrics.md`
- `$PATH_TO_VLLM_ASCEND/benchmarks`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug/performance_benchmark.md`

命令形态通常是两类：

```bash
# 离线推理 benchmark：更容易隔离模型执行和调度影响
python benchmarks/benchmark_offline.py ...

# 服务模式 benchmark：更接近线上请求、排队和网络行为
python benchmarks/benchmark_serving.py ...
```

具体参数要以当前 vLLM 版本为准。vLLM benchmark 脚本也会随社区演进，文档里记录命令时建议写清楚对应 commit。

## 服务侧观测

服务侧先看：

- 请求是否排队，waiting/running 数量是否异常。
- scheduler 每轮 token budget 是否被用满。
- prefill/decode 是否互相挤压。
- KV cache block 是否频繁不足、释放、抢占或等待。
- prefix cache 命中率是否符合预期。
- spec decode 接受率是否足够高。
- graph 是否成功捕获和 replay。
- 输出处理或 sampling 是否成为瓶颈。

vLLM metrics、日志和 benchmark 输出通常能给出第一层线索。不要只看平均值，p90/p99 和抖动往往更接近线上真实问题。

## NPU 侧观测

NPU 侧主要关注：

- 算子耗时：attention、linear、MoE、sampling、all-reduce/all-gather。
- kernel 是否走到预期 backend，有没有 fallback。
- graph capture/replay 是否成功，shape 是否稳定。
- NPU 显存：权重、KV cache、临时 workspace、graph buffer。
- 通信耗时：HCCL、CP 通信、TP/EP/MoE 通信、KV transfer。
- host overhead：Python 调度、数据准备、CPU/NPU 同步、日志过多。

相关入口：

- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug/msprobe_guide.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/profiler`
- `$PATH_TO_VLLM_ASCEND/tests/ut/profiler`

## 常见调优方向

| 方向 | 主要影响 | 风险 |
| --- | --- | --- |
| `max_num_batched_tokens` | prefill/decode 混合、TTFT、吞吐 | 过大可能挤压 decode 或增加显存压力 |
| `max_num_seqs` | 并发请求数、decode batch | 过大可能导致 KV block 不足 |
| chunked prefill | 长 prompt TTFT/TPOT 平衡 | chunk 太碎会增加调度和 kernel overhead |
| prefix caching | 复用公共前缀，降低 prefill | 命中率低时收益有限，还会增加管理成本 |
| graph mode | 降低动态执行 overhead | 对 shape、metadata 和 fallback 敏感 |
| attention backend | 影响 prefill/decode 核心耗时 | 不同模型和 dtype 支持差异大 |
| KV transfer/offload | 缓解 PD 或显存压力 | 引入网络、CPU/NPU 拷贝和一致性问题 |
| spec decode | 提升 decode token 产出 | 接受率低时可能负收益 |
| CP/TP/EP/DP | 扩展长序列或大模型容量 | 通信和负载不均衡可能吃掉收益 |

调参时一次只改少量变量。多个变量一起改，很容易得到一个看似变快但无法解释的结果。

## 性能回归定位

性能回归建议按这个顺序：

1. 固定 workload 和环境，确认回归可以重复出现。
2. 对比 vLLM commit、vLLM Ascend commit、镜像、CANN、torch-npu。
3. 对比启动参数和环境变量，尤其是 graph、attention、KV cache、并行策略。
4. 看服务指标，判断问题更像排队、prefill、decode、KV block、sampling 还是通信。
5. 用 profiler 找到耗时变化最大的算子或阶段。
6. 如果是上游同步后出现，使用 bisect workflow 或本地二分缩小范围。
7. 记录 baseline、实验组和结论，避免下次重新排查同一类问题。

## 结果解读示例

- TTFT 变差、TPOT 基本不变：优先看 prefill、排队、chunked prefill、prefix cache、模型加载后首轮 graph。
- TPOT 变差、TTFT 基本不变：优先看 decode attention、KV cache layout、graph replay、sampling、spec decode 接受率。
- 平均吞吐上升但 p99 变差：优先看调度公平性、长请求插队、batch shape 抖动。
- 单卡正常、多卡变慢：优先看通信、rank 分组、CP/TP/EP 切分、负载均衡。
- 显存接近上限且吞吐下降：优先看 KV cache block、max_num_seqs、prefix cache、offload 和 graph buffer。

## 思考与探索

1. 为什么只报告 token throughput，不足以判断一个在线服务变快了？
2. 如果 spec decode 的接受率很低，TPOT 可能如何变化？
3. 一个长上下文 benchmark 中，CP 提升吞吐但增加 TTFT，这个结果是否一定是坏事？

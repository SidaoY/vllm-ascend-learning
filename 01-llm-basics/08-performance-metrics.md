# 性能指标

推理服务优化不能只说“快”或“慢”。需要用统一指标描述首 token 延迟、每 token 延迟、吞吐、并发、队列等待、资源占用和 SLO。本章建立基础语言，后续开发实践章节会讲如何 benchmark 和 profile。

## Latency 与 Throughput

Latency 是单个请求的耗时，用户直接感知。Throughput 是单位时间处理多少请求或 token，服务成本和容量更关心。

二者经常有冲突。更大的 batch 可能提升吞吐，但让某些请求等待更久。更激进的低延迟策略可能牺牲设备利用率。

## TTFT

TTFT 是 time to first token，表示请求到达后多久返回第一个输出 token。

TTFT 通常包括：

- 排队等待。
- tokenization 和输入处理。
- prefill 执行。
- 第一次 sampling。
- streaming 首 chunk 返回。

TTFT 对交互式体验非常重要。长 prompt、长队列、prefix cache miss、prefill 被 decode 阻塞都会影响 TTFT。

## TPOT 和 ITL

TPOT 是 time per output token，表示输出阶段平均每个 token 的耗时。

ITL 是 inter-token latency，表示相邻输出 token 之间的间隔。很多文档会把 decode 阶段的 inter-token latency 近似看作 TPOT。

TPOT/ITL 主要受 decode 影响：

- KV cache 读性能。
- decode batch 大小。
- attention kernel。
- sampling 和 detokenization。
- graph mode。
- 多卡通信。

## End-to-End Latency

端到端延迟是从请求到达服务端到请求完成的总耗时。

```text
E2E latency = waiting + input processing + prefill + all decode steps + output processing
```

短输出请求更关注 TTFT，长输出请求更关注 TPOT。评估时要看请求的输入/输出长度分布。

## Throughput

常见吞吐指标：

- requests/s：每秒完成多少请求。
- input tokens/s：每秒处理多少输入 token。
- output tokens/s：每秒生成多少输出 token。
- total tokens/s：输入和输出合并统计。

只看 requests/s 容易误导，因为不同请求的 token 数差异很大。LLM serving 通常更关注 token throughput，同时结合 latency/SLO。

## Goodput 和 SLO

Goodput 是满足 SLO 的有效吞吐。例如要求 95% 请求 TTFT 小于 2 秒、TPOT 小于 50ms，那么超出 SLO 的请求即使完成了，也可能不算有效吞吐。

线上服务不能只优化平均值。p90、p95、p99 延迟更能反映用户体验和尾延迟问题。

## Prefill-heavy、Decode-heavy、Mixed

不同场景的瓶颈不同。

| 场景 | 特点 | 常见瓶颈 |
| --- | --- | --- |
| Prefill-heavy | 输入长、输出短 | prefill attention、token budget、prefix cache |
| Decode-heavy | 输入短、输出长 | KV cache 读、decode batch、sampling |
| Mixed | 长短请求混合 | scheduler、公平性、队列、chunked prefill |

评估性能时必须说明输入长度、输出长度、并发、request rate，否则指标很难比较。

## 内存和 KV Cache 指标

内存指标同样重要：

- 权重占用。
- KV cache 已用 block 和剩余 block。
- NPU/显存峰值。
- CPU offload 占用。
- prefix cache 命中率。
- KV transfer 带宽和耗时。

很多性能退化实际上来自内存压力。例如 KV cache block 不足会导致请求等待或抢占，最终表现为 latency 增大。

## 指标之间的取舍

一些常见取舍：

- 增大 batch：吞吐可能提升，排队延迟可能上升。
- 开启 prefix caching：命中时 TTFT 降低，cache 管理开销增加。
- 开启 chunked prefill：decode 延迟更平滑，prefill 完成时间可能变化。
- 使用更多并行：模型能跑更大，通信开销增加。
- 使用 graph mode：decode 开销降低，但 shape 灵活性受限。
- 使用 spec decode：decode token throughput 可能提升，但接受率和额外模型成本决定收益。

## 和代码/文档的连接

- vLLM benchmark 文档：`$PATH_TO_VLLM/docs/benchmarking`
- vLLM benchmark 脚本：`$PATH_TO_VLLM/benchmarks`
- vLLM metrics 设计：`$PATH_TO_VLLM/docs/design/metrics.md`
- vLLM Ascend benchmark：`$PATH_TO_VLLM_ASCEND/benchmarks`
- Ascend performance benchmark：`$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug/performance_benchmark.md`
- Ascend optimization guide：`$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug/optimization_and_tuning.md`

## 思考与探索

1. 对一个"长输入短输出"的场景，猜测 TTFT 和 TPOT 哪个更容易成为痛点。
2. 为什么 benchmark 结果必须同时记录输入长度和输出长度？
3. 如果吞吐上升但 p99 latency 变差，你会如何向业务方解释这个结果？


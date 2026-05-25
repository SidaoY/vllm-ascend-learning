# 负载均衡

负载均衡不是单一功能。vLLM Ascend 中至少有三类负载均衡问题：请求如何分到不同实例，PD 分离中 prefill/decode 角色如何分流，MoE expert 如何在设备之间保持负载均衡。本章把这些层次放在一起看。

## 负载不均衡从哪里来

常见来源包括：

- 请求长度差异：长 prompt 和短 prompt 的 prefill 成本差异很大。
- 输出长度差异：有的请求很快结束，有的 decode 很久。
- Prefill/decode 混合：prefill 计算密集，decode 读 KV 多，资源压力不同。
- DP replica：不同副本收到的请求量或请求长度不同。
- PD 分离：prefill 实例和 decode 实例的压力不一致。
- MoE expert 热点：router 把大量 token 分到少数 expert。
- 多机网络差异：同样的计算负载，通信成本可能不同。

所以负载均衡要先说清楚“均衡的对象是什么”：请求、token、KV、rank、expert，还是实例。

## 请求级和 Expert 级不是一回事

| 层级 | 组件 | 流向 | 目标 |
| --- | --- | --- | --- |
| 请求级 | Client → Request / DP / PD load balancer | → | Instance A / Instance B |
| Expert 级 | Instance A → MoE routing and EPLB | → | Experts on devices |
| Expert 级 | Instance B → MoE routing and EPLB | → | Experts on devices |

请求级负载均衡决定请求进入哪个实例或哪个角色。Expert 级负载均衡发生在 MoE 模型内部，决定 expert 副本、放置、迁移和 token load 如何在设备间分布。

这两层会互相影响，但不能混为一谈。请求分得很均匀，不代表 expert 就均匀；expert 均匀，也不代表 prefill/decode 实例压力均匀。

## DP / External Load Balancing

DP 场景中，多个 replica 可以承接请求。负载均衡可以在 vLLM 内部做，也可以由外部 proxy 或网关做。

关注点：

- 路由指标：队列长度、running 请求数、KV cache 压力、TTFT/TPOT、健康状态。
- 会话保持：多轮对话是否需要进入同一实例。
- Prefix cache：路由变化可能降低 cache 命中率。
- 失败处理：实例退出、超时、重试时是否会重复生成或丢请求。

入口：

- `$PATH_TO_VLLM/docs/deployment/frameworks`
- `$PATH_TO_VLLM/docs/assets/deployment/dp_external_lb.png`
- `$PATH_TO_VLLM_ASCEND/examples/external_online_dp`

## PD 分离场景的负载均衡

PD 分离中，prefill 和 decode 是不同角色。Prefill 主要吃长 prompt 计算，decode 主要吃持续生成和 KV 读取。负载均衡需要同时考虑：

- prefill 节点是否排队。
- decode 节点是否有足够 KV cache 和 decode capacity。
- KV transfer 是否成为瓶颈。
- 请求路由是否能保持 producer/consumer metadata 一致。
- prefix cache 或 KV pool 是否改变路由策略。

入口：

- `$PATH_TO_VLLM_ASCEND/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`
- `$PATH_TO_VLLM_ASCEND/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py`
- `$PATH_TO_VLLM_ASCEND/examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer`

## MoE / EPLB

MoE 模型中，每个 token 会被 router 分配到少数 expert。真实 workload 下，expert 负载经常不均衡：某些 expert 很热，某些 expert 很闲。EPLB，Expert Parallel Load Balancing，目标是让 expert 放置和冗余更适合当前流量。

EPLB 关注：

- expert heat：每个 expert 被选中的频率或 token 量。
- expert map：expert 到设备的映射。
- redundant experts：为热点 expert 提供冗余副本。
- device transfer：expert 迁移或加载带来的开销。
- update policy：何时更新映射，如何避免过度抖动。

vLLM Ascend 里 EPLB 有 adaptor、core、policy、worker、updator 等部分。Swift Balancer 是其中一种更强调异步和平滑迁移的策略。

代码入口：

- `$PATH_TO_VLLM/vllm/distributed/eplb`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb/core/policy`
- `$PATH_TO_VLLM_ASCEND/tests/ut/eplb`

## Dynamic EPLB 和 Static EPLB

Dynamic EPLB 会在服务运行中收集 expert heat，并按策略动态调整 expert map。它适合流量分布会变化的场景，但要控制更新频率，避免策略本身带来抖动。

Static EPLB 更像先记录或生成一个 expert map，再在后续部署中加载。它适合流量相对稳定、希望启动后行为更可控的场景。

常见配置会涉及：

- `DYNAMIC_EPLB`
- `EXPERT_MAP_RECORD`
- `expert_heat_collection_interval`
- `algorithm_execution_interval`
- `num_redundant_experts`
- `expert_map_path`
- `expert_map_record_path`

具体字段以当前文档和代码为准：

- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/eplb_swift_balancer.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/eplb_swift_balancer.md`

## Scheduler 和 Executor Patch

负载均衡有时需要影响 vLLM 的调度或多进程执行方式。vLLM Ascend 中相关 patch 包括：

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_balance_schedule.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform/patch_multiproc_executor.py`

阅读这类 patch 时要问：

- 它改变的是请求选择、batch 形态，还是 worker 分配？
- 是否只在特定配置下生效？
- 对普通单卡或非 MoE 场景有没有影响？
- 是否有明确测试覆盖和退出计划？

## 指标和调参

请求级负载均衡常看：

- 每个实例 waiting/running 请求数。
- 每个实例 TTFT、TPOT、吞吐、错误率。
- KV cache 使用率和 block 不足次数。
- 请求长度分布和路由分布。

EPLB 常看：

- expert load 分布。
- load balance ratio。
- expert map 更新次数。
- 迁移耗时和迁移期间延迟抖动。
- MoE dispatch/combine 耗时。
- token throughput 和 p99 latency。

调参时要避免“看见不均衡就频繁迁移”。迁移有成本，统计窗口太短也可能让策略追着噪声跑。

## 常见问题

请求级负载均衡仍然不均：看路由指标是否只看请求数而忽略请求长度和 KV 压力。

PD 场景 decode 等不到 KV：看 prefill/decode 路由、connector metadata 和 KV transfer 状态。

EPLB 后性能波动：看 expert map 更新频率、迁移成本、统计窗口和 warmup。

热点 expert 没缓解：看 redundant expert 数量、policy 是否生效、router token load 是否真的变化。

单卡/普通模型被影响：看 patch 生效条件是否过宽。

## 参考入口

- `$PATH_TO_VLLM/docs/deployment/frameworks`
- `$PATH_TO_VLLM/docs/assets/deployment/dp_external_lb.png`
- `$PATH_TO_VLLM/vllm/distributed/eplb`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb`
- `$PATH_TO_VLLM_ASCEND/examples/external_online_dp`
- `$PATH_TO_VLLM_ASCEND/examples/disaggregated_prefill_v1`
- `$PATH_TO_VLLM_ASCEND/examples/epd_disaggregated`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/eplb_swift_balancer.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/eplb_swift_balancer.md`
- `$PATH_TO_VLLM_ASCEND/tests/ut/eplb`

## 思考与探索

1. 为什么按请求数均匀分发，不一定能让实例负载均匀？
2. MoE expert 热点和 DP replica 负载不均有什么区别？
3. EPLB 的统计窗口太短或太长，分别可能带来什么问题？

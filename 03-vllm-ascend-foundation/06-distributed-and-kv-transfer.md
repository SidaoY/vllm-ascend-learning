# 分布式与 KV Transfer

vLLM Ascend 的分布式能力覆盖多卡、多机、MoE、CP、PD 分离、KV transfer、KV pool 等场景。它建立在 vLLM 分布式抽象之上，并适配 Ascend 的 HCCL、NPU communicator 和各类 KV connector。

## 分布式先看什么

排查分布式问题时，先确认四件事：

- 并行策略：TP、PP、DP、EP、CP 哪些启用了。
- rank 拓扑：world size、local rank、各并行组如何划分。
- 通信后端：HCCL、torch distributed、外部 connector 是否正常。
- 数据流：权重、activation、KV cache、expert token 或请求流量到底在哪里传。

很多“模型输出异常”或“性能异常”其实是 rank group、通信或 KV transfer 的问题。

## Parallel State

Parallel state 管理不同并行维度的 group。常见 group 包括：

- Tensor parallel group。
- Pipeline parallel group。
- Data parallel group。
- Expert parallel group。
- Context parallel group。
- PD 分离或 KV transfer 场景中的 producer/consumer group。

## HCCL 与通信

Ascend 多卡通信通常依赖 HCCL。vLLM Ascend 需要把 vLLM 的分布式抽象映射到 NPU 通信能力上，包括：

- collective communication，例如 all-reduce、all-gather、broadcast。
- MoE dispatch/combine。
- CP 下的 KV 或 attention 通信。
- PD/KV transfer 的跨实例数据传输。
- 多机网络配置和超时重试。

通信问题常见症状包括 hang、timeout、rank 不一致、性能突然下降、单机正常多机失败。

## PD 分离和 KV Transfer

Disaggregated prefill 把 prefill 和 decode 拆到不同实例或不同角色中。核心挑战是：prefill 侧生成的 KV cache 必须被 decode 侧正确拿到。

| 步骤 | 阶段 | 说明 |
| --- | --- | --- |
| 1 | Request | 请求进入 |
| 2 | Prefill instance | Prefill 实例生成 KV cache |
| 3 | KV cache produced | KV cache 产出 |
| 4 | KV connector / KV pool | 通过 connector 或 pool 传输 KV |
| 5 | Decode instance | Decode 实例接收 KV 并继续生成 |
| 6 | Generated tokens | 产出 token |

KV transfer 需要协调：

- scheduler 何时认为远端 KV 可用。
- worker 如何发送或接收每层 KV。
- KV cache layout、dtype、block size 是否一致。
- 网络失败时是否 recompute、retry 或 fail。
- prefix cache、KV pool、PD role 和请求路由如何配合。

## Connector 形态

vLLM Ascend 中可能遇到多种 KV connector 或存储形态：

- P2P connector：实例之间直接传输 KV。
- Mooncake / layerwise connector：面向 PD 分离和分层传输场景。
- Ascend store / KV pool：把 KV cache 放入统一存储池。
- LMCache / UCM：更偏外部缓存或统一缓存管理。
- CPU/NPU offload：在本地不同介质之间搬运 KV。

学习时先理解“生产者、消费者、存储介质、传输时机、失败策略”五个问题，不要一开始陷入某个 connector 的所有参数。

## CP、MoE 和 KV Transfer 的交叉

分布式特性经常叠加：

- CP 会改变长上下文 attention 和 KV cache 的切分。
- MoE 会引入 expert dispatch/combine 和负载均衡。
- PD 分离会让 KV cache 在实例间移动。
- Graph mode 会要求通信和 shape 更稳定。

这些组合场景最容易出复杂 bug。排查时建议先关掉非必要特性，建立最小复现，再逐个打开。

## 代码入口

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/parallel_state.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/device_communicators`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/distributed/kv_transfer`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/kv_offload`
- `$PATH_TO_VLLM_ASCEND/examples/disaggregated_prefill_v1`
- `$PATH_TO_VLLM_ASCEND/examples/epd_disaggregated`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/kv_pool.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md`
- `$PATH_TO_VLLM_ASCEND/tests/ut/kv_connector`
- `$PATH_TO_VLLM_ASCEND/tests/ut/distributed`

建议搜索关键词：`parallel_state`、`HCCL`、`KVConnector`、`Mooncake`、`AscendStore`、`PD disaggregation`、`kv_transfer`。

## 常见问题定位

- 多卡 hang：先看 rank group、world size、环境变量和每个 rank 是否进入同一通信。
- 单机正常多机失败：重点看网络、HCCL、RDMA、hostfile、端口和超时。
- Decode 读不到 KV：看 producer/consumer role、connector metadata、block layout 和传输完成信号。
- KV transfer 精度异常：看 dtype、layout、block size、layer 顺序和 cache spec。
- 性能差：看传输粒度、同步点、网络带宽、KV pool 命中率和 batch 形态。

## 思考与探索

1. PD 分离为什么一定要解决 KV cache 传输问题？
2. 多卡通信问题为什么经常表现为 hang 而不是明确异常？
3. 如果 KV transfer 失败后选择 recompute，会影响哪些性能指标？

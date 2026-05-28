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

所以负载均衡要先说清楚"均衡的对象是什么"：请求、token、KV、rank、expert，还是实例。

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

### External DP 的典型架构

External DP 的核心思路是：vLLM 实例本身不感知 DP，每个实例以 `dp_size=1` 运行，由外部组件管理多实例的路由和负载均衡。

```text
                    ┌─────────────────────────────┐
                    │     Load Balancer / Proxy    │
                    │  (Nginx, HAProxy, Envoy,     │
                    │   custom proxy server)       │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │  vLLM Instance │ │  vLLM Instance │ │  vLLM Instance │
     │   (dp_size=1)  │ │   (dp_size=1)  │ │   (dp_size=1)  │
     │   TP=4, EP=8   │ │   TP=4, EP=8   │ │   TP=4, EP=8   │
     └────────────────┘ └────────────────┘ └────────────────┘
```

**与内置 DP 的对比：**

| 维度 | 内置 DP | External DP |
| --- | --- | --- |
| 实例感知 | 实例间通过 NCCL/HCCL 通信 | 实例间完全独立 |
| 负载均衡 | vLLM scheduler 内部处理 | 外部 proxy 控制 |
| 扩缩容 | 需要重启所有实例 | 可以独立增减实例 |
| 故障隔离 | 一个实例 OOM 可能影响全组 | 实例故障不影响其他 |
| Prefix cache | 跨实例不可共享 | 跨实例不可共享（需 sticky routing） |
| 适用场景 | 同构硬件、固定规模 | 异构硬件、弹性扩缩容 |

### External DP 的路由策略

vLLM Ascend 的 `external_online_dp` 示例展示了三种常见的路由策略：

**1. 轮询（Round Robin）**

最简单的策略，请求依次分发到各实例。适合请求长度和处理时间均匀的场景。

```text
Request 1 → Instance 0
Request 2 → Instance 1
Request 3 → Instance 2
Request 4 → Instance 0
...
```

**2. 最少请求数（Least Connections）**

将请求发送到当前 running 请求数最少的实例。适合请求处理时间差异大的场景。

```text
Instance 0: 3 running → skip
Instance 1: 1 running → selected ✅
Instance 2: 5 running → skip
```

**3. 加权轮询 / 加权最少连接**

为不同性能的实例设置不同权重。适合异构硬件场景（如部分实例使用 A3、部分使用 A2）。

```text
Instance 0 (A3, weight=3): receives 3/6 = 50% of requests
Instance 1 (A2, weight=2): receives 2/6 ≈ 33% of requests
Instance 2 (A2, weight=1): receives 1/6 ≈ 17% of requests
```

### External DP 的会话保持（Sticky Session）

多轮对话场景中，同一会话的请求需要路由到同一实例，否则 prefix cache 无法命中。常见的会话保持方式：

- **基于 session_id 的哈希路由**：`instance = hash(session_id) % num_instances`。简单但扩缩容时会话会重新分布。
- **一致性哈希**：扩缩容时只影响部分会话，减少 cache miss。
- **Proxy 层维护会话映射表**：灵活但需要额外存储。

### External DP 的健康检查

Proxy 需要持续监控后端实例的健康状态：

```text
Health check endpoint: GET /health
  → 200: instance healthy, can accept requests
  → 503: instance unhealthy (OOM, model load failure, etc.), remove from routing table

Key checks:
  - Is the process alive?
  - Is the model loaded?
  - Is KV cache usage above threshold (e.g. 95%)?
  - Is the waiting queue too long?
```

### External DP 的常见问题

- **实例退出导致请求丢失**：需要 proxy 层重试机制，但要注意幂等性（同一请求可能被多个实例处理）。
- **Prefix cache 命中率下降**：非 sticky 路由或扩缩容后会话重新分布。
- **负载不均**：轮询策略下，长请求和短请求混合时可能出现"长请求堆积"。
- **健康检查延迟**：实例已经 OOM 但健康检查还没发现，导致请求失败。

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

MoE 模型中，每个 token 会被 router 分配到少数 expert。真实负载下，expert 负载经常不均衡：某些 expert 很热，某些 expert 很闲。EPLB，Expert Parallel Load Balancing，目标是让 expert 放置和冗余更适合当前流量。

### 核心概念

在深入 EPLB 实现之前，先理清几个关键概念：

| 概念 | 含义 | 示例（DeepSeek-R1，256 routed experts + 32 redundant，32 EP ranks） |
| --- | --- | --- |
| Logical Expert | 模型定义中的 expert，有独立权重 | 256 个 |
| Physical Expert | 实际实例化在设备上的 expert 副本 | 256 + 32 = 288 个 |
| Redundant Expert | 为热点 logical expert 创建的额外副本 | 32 个 |
| Local Physical Expert | 当前设备上的 physical expert | 288 / 32 = 9 个/卡 |
| Expert Map | expert 到设备的映射关系 | `[layer, rank, slot] → expert_id` |
| Expert Heat | 每个 expert 被选中的 token 量 | 滑动窗口内的累计 token 数 |

### EPLB 的完整生命周期

EPLB 不是一次性操作，而是一个持续运行的循环。以 Dynamic EPLB 为例，完整流程如下：

```text
┌─────────────────────────────────────────────────────────────────┐
│                     EPLB Lifecycle (per round)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Collect Expert Heat (expert_heat_collection_interval steps)│
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Each forward pass:                                        │  │
│  │   expert_load_pass[layer, physical_expert] += token_count │  │
│  │                                                           │  │
│  │ At end of each step:                                      │  │
│  │   expert_load_window[step % window_size] = expert_load_pass│  │
│  │   expert_load_pass.zero_()                                │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  Phase 2: Wake EPLB Worker, execute algorithm (algorithm_execution_interval steps)│
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. all_gather moe_load from each rank → global expert heat│  │
│  │ 2. Call policy.rebalance_experts() to compute new expert map│  │
│  │ 3. Generate expert migration plan (sender, receiver, expert)│  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  Phase 3: Migrate Expert Weights layer by layer (num_moe_layers steps, 1 layer/step)│
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Parse send/recv info → P2P communication tasks         │  │
│  │ 2. Async D2D transfer of expert weights                   │  │
│  │ 3. Update expert_map and log2phy_map                      │  │
│  │ 4. Update expert weight tensors                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1：Expert Heat 如何收集

Expert heat 的收集不是简单的"数 token"。vLLM 使用滑动窗口机制来平滑负载波动：

```text
expert_load_window: [window_size, num_moe_layers, num_physical_experts]

Example: window_size=10, num_moe_layers=58, num_physical_experts=288:

  step 0: window[0] = expert_load_pass  ← record step 0 load
  step 1: window[1] = expert_load_pass  ← record step 1 load
  ...
  step 9: window[9] = expert_load_pass  ← record step 9 load
  step 10: window[0] = expert_load_pass ← overwrite step 0 (circular)
```

收集到的 `expert_load_window` 需要从 physical expert 视角转换为 logical expert 视角，因为算法关心的是"每个 logical expert 有多热"，而不是"每个 physical expert 副本有多热"：

```text
logical_expert_load = scatter_add(
    physical_expert_load,
    index=physical_to_logical_map  # sum loads of all replicas of the same logical expert
)
```

然后通过 all_reduce 汇总所有 EP rank 的负载，得到全局的 `global_expert_load_window`。

### Phase 2：Rebalance 算法详解

vLLM 社区的 DefaultEplbPolicy 实现了经典的 EPLB 算法（源自 DeepSeek EPLB 论文），分为三个步骤：

**Step 1：Pack groups to nodes（组到节点的打包）**

MoE 模型中，experts 被组织成 group（如 DeepSeek-V3 有 8 个 group，每个 group 32 个 expert）。同一 group 内的 expert 之间有更多通信，因此尽量把同一 group 放在同一节点上可以减少跨节点流量。

```text
Input: tokens_per_group [num_layers, num_groups]
      e.g. layer 0: [group0: 1000, group1: 800, group2: 1200, ...]

Algorithm: balanced_packing(weight, num_nodes)
      → assign groups to nodes, balancing total load per node

Output: group_pack_index [num_layers, num_groups]
      e.g. layer 0: [node0, node1, node0, node1, ...]
```

`balanced_packing` 的核心逻辑是贪心：每次把当前最重的 group 放到当前最轻的 node 上。这保证了 node 间的负载均衡。

**Step 2：Replicate experts within nodes（节点内复制 expert）**

在每个 node 内部，为热点 logical expert 创建冗余副本：

```text
Input: tokens_per_mlog [num_layers, num_logical_experts // num_nodes]
      num_physical_experts // num_nodes  ← physical experts per node

Algorithm: replicate_experts(weight, num_phy)
      → greedy replication: each time replicate the expert with max (load / replica_count)

Example (simplified, 1 layer, 4 logical experts, 6 physical experts):
  logical expert load: [A: 100, B: 50, C: 30, D: 20]
  Initial: phy2log = [A, B, C, D, ?, ?], logcnt = [1, 1, 1, 1]

  Replication 1: max(load/logcnt) = max(100/1, 50/1, 30/1, 20/1) = A
              phy2log = [A, B, C, D, A, ?], logcnt = [2, 1, 1, 1]

  Replication 2: max(100/2, 50/1, 30/1, 20/1) = B
              phy2log = [A, B, C, D, A, B], logcnt = [2, 2, 1, 1]

Result: A has 2 replicas, B has 2 replicas, C and D each have 1 replica
```

**Step 3：Pack physical experts to GPUs（物理 expert 到 GPU 的打包）**

复制完成后，把 physical expert 分配到各 GPU：

```text
Input: tokens_per_phy [num_layers, num_physical_experts // num_nodes]
      (effective load per physical expert = logical_load / replica_count)

Algorithm: balanced_packing(weight, num_gpus // num_nodes)
      → assign physical experts to GPUs, balancing total load per GPU

Output: phy2pphy → final mapping is phy2log [num_layers, num_physical_experts]
```

**Hierarchical vs Global 策略**

- **Hierarchical（分层）**：当 `num_groups % num_nodes == 0` 时使用。先在 node 间均衡 group，再在 node 内做 expert 复制和 GPU 打包。适合 prefill 阶段（EP size 较小）。
- **Global（全局）**：当 group 数不能被 node 数整除时使用。跳过 group→node 打包，直接在全局做 expert 复制和 GPU 打包。适合 decode 阶段（EP size 较大）。

### Swift Balancer 的增强

vLLM Ascend 在社区 DefaultEplbPolicy 基础上实现了 Swift Balancer（`policy_swift_balancer.py`），主要针对低带宽设备（如 Atlas A2）做了优化：

**1. 不均衡度计算**

Swift Balancer 用 `max_load / avg_load` 来衡量每层的不均衡程度：

```text
For each layer:
  for each rank:
    rank_load = sum(expert_load[expert_id] / num_replicas[expert_id])
  max_load = max(rank_load)
  avg_load = total_load / num_ranks
  imbalance = max_load / avg_load  ← closer to 1.0 means more balanced
```

**2. 冗余 Expert 的智能分配**

Swift Balancer 不是简单地把冗余 expert 平均分配，而是根据当前负载动态决定哪些 expert 需要更多副本：

```text
compute_redundant_assignments():
  For each redundant slot:
    Find the expert with max (load / replica_count)
    Add one replica for it
    Update effective load = original_load * (replica+1) / (replica+2)
```

Example:
  expert A: load=100, replica=1 → effective load=100
  expert B: load=80,  replica=1 → effective load=80
  expert C: load=60,  replica=1 → effective load=60

  Slot 1 to A: A effective load = 100 * 2/3 ≈ 66.7
  Slot 2: max(66.7, 80, 60) = B → B effective load = 80 * 2/3 ≈ 53.3
  Slot 3: max(66.7, 53.3, 60) = A → A effective load = 100 * 3/4 = 75
  ...
```

**3. 约束本地交换**

`constraint_expert_local_exchange` 确保 expert 迁移时，同一 GPU 上保持不变（不需要移动）的 expert 尽量留在原位，减少不必要的权重拷贝。

### FlashLB 策略

FlashLB（`policy_flashlb.py`）是一种轻量级的负载均衡策略，与 EPLB 的"全局重排"思路不同，FlashLB 采用"阈值触发 + 局部调整"的方式：

**核心思路：**

```text
Instead of periodically recomputing the globally optimal expert map,
only make minimal adjustments when imbalance exceeds a threshold.
```

**与 EPLB 的对比：**

| 维度 | EPLB (Default/Swift) | FlashLB |
| --- | --- | --- |
| 触发方式 | 定期（按步数间隔） | 阈值触发（不均衡度超过阈值） |
| 调整范围 | 全局重排所有 expert | 局部调整热点 expert |
| 迁移成本 | 较高（可能大量 expert 移动） | 较低（只移动少量 expert） |
| 适用场景 | 流量模式变化大 | 流量模式相对稳定 |
| 收敛速度 | 慢（需要多轮才能稳定） | 快（单轮即可缓解热点） |

**FlashLB 的典型工作流程：**

```text
1. Collect expert heat each step (same as EPLB)
2. Compute per-layer imbalance = max_load / avg_load
3. If imbalance < threshold (e.g. 1.2): skip, no adjustment
4. If imbalance >= threshold:
   a. Find the hottest expert and the coldest expert
   b. Migrate one replica of the hot expert to the GPU of the cold expert
   c. Update expert_map
5. Wait for cooldown steps before checking again (avoid frequent migrations)
```

**适用建议：**

- 流量模式稳定（如固定 system prompt、固定任务类型）时，FlashLB 的迁移成本更低。
- 流量模式波动大（如不同时间段处理不同类型的请求）时，EPLB 的全局重排更合适。
- 可以先用 FlashLB 快速消除热点，再切换到 EPLB 做全局优化。

### Phase 3：Expert 权重如何迁移

计算出新的 expert map 后，需要把 expert 权重从旧位置搬到新位置。这个过程涉及 P2P 通信和权重 tensor 更新。

**社区 vLLM 的迁移流程（`rebalance_execute.py`）**

```text
move_to_buffer():
  1. 本地移动：同一 GPU 内，expert 位置变化 → 直接 copy_ 到 buffer
  2. 发送：确定哪些 expert 需要发给哪些 rank → P2P isend
  3. 接收：确定从哪些 rank 接收哪些 expert → P2P irecv
  4. 执行通信：communicator.execute() → 所有 P2P 操作同时进行

move_from_buffer():
  1. 从 buffer 拷贝回 expert_weights（本地接收 + 远程接收的 primary）
  2. 复制远程接收的 primary expert 到同一 GPU 上的 duplicate 位置
```

**vLLM Ascend 的 D2D 迁移（`eplb_device_transfer_loader.py`）**

Ascend 侧使用 `D2DExpertWeightLoader` 管理设备间 expert 权重传输，通过状态机控制迁移流程：

```text
状态机: WAITING → READY → TRANSFERRING → WAITING

WAITING:       等待 EplbWorker 计算出新的 expert_map
READY:         解析 send/recv 信息，生成 P2P 通信任务列表
TRANSFERRING:  异步执行 batch_isend_irecv，等待完成后更新 expert_map 和权重
```

迁移完成后，还需要更新 `log2phy_map`（logical expert → physical expert 的映射），确保后续的 token dispatch 能把 token 送到正确的 physical expert。

### Async EPLB Worker

社区 vLLM 支持异步 EPLB（`async_worker.py`），通过后台线程执行 expert 迁移，避免阻塞主推理线程：

```text
Main Thread                      Async Worker Thread
  │                                │
  │  Collect expert_load_window    │
  │  rearrange_event.set()  ──────→ Woken up
  │                                │
  │                                ├─ Compute new_physical_to_logical_map
  │                                ├─ transfer_layer() → move weights to buffer
  │                                ├─ cuda_stream.synchronize()
  │                                ├─ Set pending_result
  │                                │
  │  move_to_workspace()  ←─────── │  consumed_event.wait()
  │  Move from buffer to           │
  │  expert_weights                │
  │  consumed_event.record() ─────→ Continue next layer
  │                                │
```

异步模式的优势是 expert 迁移和推理可以 overlap，减少"stop-the-world"的延迟影响。

### Dynamic EPLB 和 Static EPLB

**Dynamic EPLB** 在服务运行中持续收集 expert heat 并动态调整 expert map。关键参数：

| 参数 | 含义 | 建议值 |
| --- | --- | --- |
| `expert_heat_collection_interval` | 收集 expert heat 的步数 | 稳定流量 400+，波动流量 100-200 |
| `algorithm_execution_interval` | 算法执行间隔（heat 收集完成后等待的步数） | ≥ 30，避免启动阶段过早触发 |
| `num_redundant_experts` | 冗余 expert 数量 | 通常等于 EP size |
| `window_size` | 滑动窗口大小 | 控制负载平滑程度 |

**Static EPLB** 先记录 expert map 到文件，后续部署时直接加载。适合流量模式稳定的场景：

```shell
# 第一步：记录 expert map
export EXPERT_MAP_RECORD="true"
vllm serve ... --additional-config '{
  "eplb_config": {
    "expert_map_record_path": "/path/to/eplb.json",
    "num_redundant_experts": 16
  }}'

# 第二步：加载已记录的 expert map
vllm serve ... --additional-config '{
  "eplb_config": {"expert_map_path": "/path/to/eplb.json"}
}'
```

### EPLB 架构总览

```text
vllm_ascend/eplb/
├── adaptor/
│   └── vllm_adaptor.py          # Model adapter: get expert weights, moe_load, global_expert_map
├── core/
│   ├── policy/
│   │   ├── policy_abstract.py        # Policy abstract interface
│   │   ├── policy_default_eplb.py    # Community EPLB algorithm
│   │   ├── policy_swift_balancer.py  # Swift Balancer (Ascend enhanced)
│   │   ├── policy_flashlb.py         # Threshold-triggered adjustment
│   │   └── policy_factory.py         # Policy factory
│   ├── eplb_device_transfer_loader.py # D2D weight transfer
│   ├── eplb_utils.py                 # Expert map init/mapping utilities
│   └── eplb_worker.py                # Async Worker process
├── eplb_updator.py                   # Central coordinator: manages heat collection→algorithm→migration
└── utils.py                          # EPLB interface registration

vllm/distributed/eplb/
├── async_worker.py              # Community async Worker thread
├── eplb_state.py                # EPLB state management (EplbState, EplbModelState, EplbStats)
├── rebalance_execute.py         # Weight rearrangement execution (move_to_buffer, move_from_buffer, transfer_layer)
├── eplb_communicator.py         # P2P communication abstraction
└── policy/
    ├── abstract.py              # AbstractEplbPolicy
    └── default.py               # DefaultEplbPolicy (balanced_packing, replicate_experts, rebalance_experts_hierarchical)
```

### 关键设计决策

**为什么用滑动窗口而不是瞬时值？**

单步的 expert load 波动很大（某个 batch 可能恰好有很多 token 路由到 expert A，下一个 batch 可能没有）。滑动窗口平滑了这种波动，让算法基于"近期趋势"而非"瞬时噪声"做决策。

**为什么逐层迁移而不是一次性全部迁移？**

一次性迁移所有层的 expert 权重会导致：
- 通信峰值过高，可能阻塞推理。
- 显存峰值过高（需要同时 buffer 所有层的权重）。
- 逐层迁移把成本分摊到多个 step，对推理延迟的影响更平滑。

**为什么需要 `constraint_expert_local_exchange`？**

如果 expert A 在 GPU 0 的 slot 3，新 map 中它仍然在 GPU 0 但变成了 slot 5，直接按新 map 重排会导致不必要的权重拷贝。`constraint_expert_local_exchange` 尽量保持不变的 expert 在原位，只移动真正需要跨 GPU 迁移的 expert。

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

- expert load 分布（`avg_tokens`, `max_tokens`, `balancedness = avg/max`）。
- load balance ratio（越接近 1.0 越均衡）。
- expert map 更新次数和迁移耗时。
- 迁移期间延迟抖动。
- MoE dispatch/combine 耗时。
- token throughput 和 p99 latency。

调参时要避免"看见不均衡就频繁迁移"。迁移有成本，统计窗口太短也可能让策略追着噪声跑。

### EPLB 指标告警建议

在生产环境中，建议对以下指标设置告警：

| 告警项 | 条件 | 严重程度 | 处理建议 |
| --- | --- | --- | --- |
| 不均衡度过高 | `max_load / avg_load > 2.0` 持续 5 分钟 | Warning | 检查 window_size 是否过小，考虑增大 num_redundant_experts |
| 不均衡度持续恶化 | `max_load / avg_load` 持续上升 10 分钟 | Critical | 检查流量模式是否发生根本变化，考虑切换 policy |
| 迁移过于频繁 | 每分钟 expert map 更新 > 5 次 | Warning | 增大 algorithm_execution_interval 或 window_size |
| 迁移耗时过长 | 单次迁移 > 500ms | Warning | 检查 P2P 带宽，考虑减小 num_redundant_experts |
| 迁移期间延迟抖动 | p99 latency 在迁移时 > 2x baseline | Warning | 检查异步迁移是否生效，考虑逐层迁移的步间隔 |
| Expert 负载为 0 | 某 expert 连续 100 步负载为 0 | Info | 正常现象（某些 expert 确实很少被选中），但需确认不是 bug |

**调参决策树：**

```text
Imbalance > 2.0?
  ├─ Yes → window_size < 50?
  │        ├─ Yes → Increase window_size (smooth noise)
  │        └─ No  → num_redundant_experts < EP size?
  │                 ├─ Yes → Increase num_redundant_experts
  │                 └─ No  → Consider switching policy (Default → Swift Balancer)
  └─ No  → Migration latency > 200ms?
           ├─ Yes → Check P2P bandwidth / consider FlashLB
           └─ No  → Current config is reasonable, keep monitoring
```

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
4. 如果 `num_groups % num_nodes != 0`，为什么 hierarchical 策略会退化为 global 策略？
5. 逐层迁移 expert 权重时，如果某一层迁移失败（P2P 通信异常），应该如何恢复？

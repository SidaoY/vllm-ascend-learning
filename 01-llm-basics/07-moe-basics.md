# MoE 基础

MoE 是 Mixture of Experts 的缩写。它是当前大模型中非常重要的一类结构，尤其在大参数量、高吞吐服务中常见。vLLM Ascend 的负载均衡、EPLB、token dispatcher、expert map 等内容，都需要先理解 MoE 的基本工作方式。

## Dense FFN 与 MoE FFN

普通 dense Transformer block 里的 MLP/FFN 对每个 token 都使用同一套参数：

```text
token hidden state -> shared FFN -> output
```

MoE 把 FFN 替换成多个 experts。每个 expert 可以理解成一套独立 FFN。router 会为每个 token 选择少数几个 experts：

```text
token hidden state -> router -> selected experts -> weighted combine -> output
```

这样模型总参数量可以很大，但每个 token 实际只激活一部分参数。

## Router 和 Top-k

Router，也叫 gate，会为每个 token 计算 expert 分数。Top-k routing 会选择分数最高的 k 个 experts。例如 top-2 表示每个 token 发送给两个 experts，再按权重合并结果。

关键字段通常包括：

- `num_experts`：总 expert 数。
- `top_k`：每个 token 选择几个 experts。
- `expert weights`：选中 experts 的合并权重。
- `expert ids`：每个 token 被路由到哪些 experts。

Router 输出决定 token 后续流向，所以它直接影响负载均衡。

## Shared Expert 和 Routed Expert

一些 MoE 模型同时有 shared experts 和 routed experts。

- Shared expert：每个 token 都会经过，类似公共 FFN。
- Routed expert：由 router 选择，只有部分 token 经过。

Shared expert 可以提升稳定性，但也增加计算和调度复杂度。某些优化会把 shared expert 和 routed expert 的执行重叠起来。

## Token Dispatch 和 Combine

如果 experts 分布在多张卡上，token 需要被发送到对应 expert 所在设备。

流程可以概括为：

1. Router 计算每个 token 的 expert ids 和 weights。
2. Dispatch：把 token hidden states 发送到目标 experts。
3. Expert compute：每个 expert 处理收到的 token。
4. Combine：把 expert 输出按权重合并回原 token 顺序。

这就是为什么 MoE 经常伴随 all-to-all 通信。通信实现和 token 排列格式会显著影响性能。

## Expert Parallelism

Expert Parallelism 把不同 experts 放到不同设备上。它能支持更多 experts 或更大模型，但带来两个核心问题：

- 通信：token 要跨设备 dispatch/combine。
- 负载：不同 experts 收到的 token 数可能差异很大。

如果一个热门 expert 收到大量 token，它所在设备会成为瓶颈，即使其他设备很空闲，整个 batch 也要等慢设备完成。

## 负载不均衡

MoE 负载不均衡来自 router 的选择。某些输入分布会让部分 experts 频繁被选中，形成 expert 热点。

负载不均衡会造成：

- 某些设备计算时间更长。
- all-to-all 通信等待增加。
- batch 尾部延迟变高。
- 整体吞吐下降。
- 多机部署中跨节点通信放大。

这就是 EPLB 和 Swift Balancer 的背景。它们试图通过 expert 放置、冗余 expert、动态策略等方式，让 token load 更均衡。

## MoE 与量化

MoE 模型常常很大，因此量化在 MoE 中也常见。MoE 的 expert 权重、router、dispatch/combine、activation format 都可能影响量化 kernel 选择和性能。

## 和代码的连接

- vLLM fused MoE：`$PATH_TO_VLLM/vllm/model_executor/layers/fused_moe`
- vLLM EPLB：`$PATH_TO_VLLM/vllm/distributed/eplb`
- vLLM MoE kernel 文档：`$PATH_TO_VLLM/docs/design/fused_moe_modular_kernel.md`
- Ascend fused MoE：`$PATH_TO_VLLM_ASCEND/vllm_ascend/ops/fused_moe`
- Ascend EPLB：`$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb`
- Ascend EPLB 用户文档：`$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/eplb_swift_balancer.md`
- Ascend EPLB 设计文档：`$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/eplb_swift_balancer.md`

## 常见误区

- 误区一：MoE 参数多，所以每个 token 计算一定多。实际每个 token 只激活部分 experts。
- 误区二：expert 数越多越好。expert 多会增加路由、通信和负载均衡难度。
- 误区三：EP 只是把 expert 平均放到卡上。真实负载取决于 token 路由，不只取决于 expert 数。
- 误区四：MoE 性能只看 GEMM。dispatch/combine 和跨卡通信同样关键。

## 思考与探索

1. 用一个 top-2 routing 例子说明 token 如何被分配到 experts。
2. 思考为什么两个请求 batch 的 expert 负载可能完全不同。
3. 打开 `vllm_ascend/eplb`，只看文件名，猜哪些文件负责策略，哪些文件负责和 vLLM 对接。

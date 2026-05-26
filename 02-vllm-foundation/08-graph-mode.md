# 图模式

Graph mode 的目标是减少重复推理中的 CPU 调度和 kernel launch 开销。对 decode 这类“小计算、多轮次、重复执行”的 workload 来说，graph capture 往往很有吸引力。

本章只讲 vLLM 中 graph mode 的基础概念。Ascend 侧 ACL graph / NPU graph 的具体实现和限制，会在 vLLM Ascend 章节和开发实践中继续展开。

## Eager、Compile 和 Graph

Eager：每次执行都由 Python / runtime 动态调度算子。灵活，容易调试，但重复 launch 开销较高。

Compile：通过编译框架优化一段模型执行图，减少 Python overhead，融合或重排部分操作。收益和约束取决于模型结构、动态 shape 和编译后端。

Graph capture：提前捕获一段稳定执行过程，后续 replay。它通常要求 shape、内存地址、控制流更稳定。

三者不是简单替代关系。实际系统可能在不同阶段、不同 batch size、不同模型路径中混合使用。

## 为什么 Decode 更常受益

Decode 每轮通常只生成少量 token，但要重复很多轮。单轮计算不一定大，CPU 调度、kernel launch、metadata 构造等开销占比会变高。

Graph mode 可以把稳定的 decode 路径捕获下来，后续反复 replay，降低每步 overhead。

Prefill 通常 token 数更大、shape 更动态，虽然也可能有优化空间，但捕获和复用难度通常更高。

## Graph Capture 的约束

常见约束包括：

- shape 稳定：batch size、token 数、sequence length 不能任意变化，或需要 padding 到可捕获 shape。
- 内存地址稳定：graph replay 依赖捕获时的 buffer 地址。
- 控制流稳定：不能每次走完全不同的 Python 分支。
- kernel 支持：某些 op 或 backend 不支持 graph capture。
- metadata 稳定：attention metadata、sampling metadata、KV cache 指针等需要可复用或可安全更新。
- 随机性处理：sampling 相关状态要保证正确。

这些约束解释了为什么 graph mode 经常和 padding、dummy input、capture size list、batch descriptor 这类概念一起出现。

## Shape 与 Padding

在线服务的 batch 形态是动态的：有时 1 个请求，有时 37 个请求；有时每个请求 decode 1 token，有时混入 chunked prefill。Graph capture 不喜欢这种无限动态。

常见做法是选择一组可捕获 shape，例如若干 batch size 或 token count，把实际请求 padding 到其中某个 shape。这样可以复用 graph，但会带来额外计算和内存占用。

所以 graph mode 的收益取决于：

- 请求形态是否稳定。
- padding 浪费是否可接受。
- capture shape 是否覆盖主要 workload。
- backend 是否真正支持该路径。

## Graph 和 Attention Backend

Attention backend 是 graph mode 的关键参与者，因为 decode 高度依赖 KV cache 和 attention metadata。Graph replay 时，backend 需要安全地使用更新后的 block table、seq lens、slot mapping 或其他 metadata。

如果 attention backend 无法满足 graph capture 的约束，系统可能回退到 eager，或者只捕获部分路径。

## Graph 和 Sampling

Sampling 也可能影响 graph：

- Greedy 和随机采样路径不同。
- top-k/top-p/min-p 等处理可能引入动态 shape 或排序操作。
- structured output / grammar mask 会改变 logits processor。
- logprobs 会增加输出和计算。

因此 graph mode 的问题不一定只发生在模型 forward，也可能发生在 sampling 或 logits processor 附近。

## Ascend 相关铺垫

Ascend 侧会涉及 ACL graph、图融合和 shape 约束。当前只需要先理解：

- graph 是为了减少重复执行开销。
- graph 依赖稳定 shape、稳定 buffer 和 backend 支持。
- graph 失败时常见处理是回退 eager、调整 capture size、关闭某些不支持路径或修改 backend metadata 更新方式。

## 代码入口

- `$PATH_TO_VLLM/docs/design/cuda_graphs.md`
- `$PATH_TO_VLLM/docs/design/cuda_graphs_multimodal.md`
- `$PATH_TO_VLLM/docs/design/torch_compile.md`
- `$PATH_TO_VLLM/docs/design/optimization_levels.md`
- `$PATH_TO_VLLM/vllm/v1/worker`
- `$PATH_TO_VLLM/vllm/v1/cudagraph_dispatcher.py`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/ACL_Graph.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/npugraph_ex.md`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/acl_graph.py`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/compilation/graph_fusion_pass_manager.py`

建议搜索关键词：`cudagraph`、`capture size`、`dummy input`、`graph replay`、`compile`、`acl graph`。

## 常见问题定位

- graph capture 失败：先看是否有不支持 op、动态 shape、动态控制流或 metadata 更新问题。
- graph 开启后反而慢：看 padding 浪费、capture shape 覆盖、batch 形态和 fallback 比例。
- 某些请求正常某些请求失败：看这些请求是否触发不同 sampling、structured output、多模态、LoRA 或 attention backend 路径。
- graph 与精度问题相关：确认 eager 和 graph 路径是否使用相同 backend、dtype、mask 和 sampling 逻辑。

## 思考与探索

1. 为什么 decode 比 prefill 更适合 graph capture？
2. Graph mode 为什么经常需要 padding？
3. 如果开启 graph 后吞吐没有提升，你会先观察哪些指标或日志？

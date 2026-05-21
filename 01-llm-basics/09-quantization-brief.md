# 量化基础简述

阅读模型加载、显存优化、MoE、KV cache、benchmark 文档时，经常会看到 INT8、INT4、FP8、W8A8、W4A8、KV cache quantization 等词。本章只做量化相关知识的基础简述，帮助读者先看懂常见术语和它们对推理系统的影响。

## 量化解决什么问题

大模型推理主要消耗两类资源：

- 存储：模型权重和 KV cache 占用大量内存。
- 带宽：decode 阶段反复读取权重和 KV cache。

量化用更低 bit 的数据格式表示权重、activation 或 KV cache，目标是降低内存占用、降低带宽压力，有时也能提升吞吐。

## 常见量化对象

Weight-only quantization：只量化权重，activation 仍使用 FP16/BF16 等格式。实现相对简单，常用于降低模型权重占用。

Weight-activation quantization：权重和 activation 都量化，例如 W8A8、W4A8。对 kernel 和精度要求更高。

KV cache quantization：量化 KV cache，降低长上下文和高并发下的 cache 占用。它会影响 attention 读取路径和精度。

## 常见格式

| 名称 | 直观含义 |
| --- | --- |
| INT8 | 8-bit integer |
| INT4 | 4-bit integer |
| FP8 | 8-bit floating point |
| W8A8 | weight 8-bit，activation 8-bit |
| W4A8 | weight 4-bit，activation 8-bit |
| W4A16 | weight 4-bit，activation 16-bit |

不同硬件、kernel、模型结构支持的格式不同。不能只看 checkpoint 标了某种量化格式，还要确认 backend 是否支持。

## Scale 和 Granularity

低 bit 数值范围有限，所以量化通常需要 scale 把浮点数映射到整数范围。

常见粒度：

- per-tensor：整个张量共享 scale。
- per-channel：每个输出/输入 channel 有自己的 scale。
- per-group：按 group 使用 scale。
- per-token：每个 token 动态计算 scale。

粒度越细，精度可能越好，但 metadata 和计算开销也可能增加。

## Static 与 Dynamic

Static quantization：scale 等参数提前计算好，推理时直接使用。通常需要校准数据或离线处理。

Dynamic quantization：推理时根据当前 activation 动态计算 scale。适应性更强，但运行时开销更高。

## 量化的风险

量化不是免费午餐，常见风险包括：

- 精度下降，尤其是数学、代码、长上下文、reasoning 任务。
- 某些 layer 不适合量化，需要保留高精度。
- kernel 支持不完整，可能回退到慢路径。
- 分布式切分和权重打包格式更复杂。
- 不同硬件对 FP8/INT4/INT8 的支持差异很大。

## 排查量化模型时先看什么

遇到量化模型相关问题时，建议先确认：

- 模型 checkpoint 的量化格式。
- vLLM/vLLM Ascend 是否支持该格式。
- 是否需要 `--quantization` 或额外配置。
- 是否存在模型家族特定 patch。
- 是否有对应精度和性能测试。

## 参考入口

- vLLM quantization 文档：`$PATH_TO_VLLM/docs/features/quantization/README.md`
- vLLM quantized KV cache：`$PATH_TO_VLLM/docs/features/quantization/quantized_kvcache.md`
- vLLM Ascend quantization 文档：`$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/quantization.md`

## 思考与探索

1. 用一句话解释 W8A8 和 W4A16 的区别。
2. 为什么 KV cache quantization 对长上下文场景有吸引力？
3. 如果一个量化模型速度比 BF16 还慢，你会先怀疑哪些原因？

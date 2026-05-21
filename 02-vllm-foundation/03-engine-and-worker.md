# Engine、Executor、Worker 与 Model Runner

vLLM 把推理服务拆成控制面和执行面。控制面负责请求状态、调度和输出；执行面负责在设备上跑模型。理解这条边界，是后续阅读 scheduler、KV cache、attention backend 和 vLLM Ascend 适配的基础。

## 四个角色

Engine：面向上层 API，管理请求生命周期。它接收请求、驱动 engine step、处理输出、处理 abort、记录统计。

Scheduler：engine 内部的资源分配器。它根据请求状态、token budget、KV cache、prefix cache、structured output 等信息，决定每个 step 推进哪些 token。

Executor：engine 和 worker 之间的执行通道。它屏蔽单进程、多进程、Ray、外部 launcher 等部署形态差异。

Worker / Model Runner：设备侧执行者。worker 管理设备、分布式环境、模型加载、KV cache 初始化；model runner 负责把 scheduler output 转换成模型输入、attention metadata，并执行 forward 和采样。

## 控制面和执行面

| 层级 | 主要职责 | 不该承担的职责 |
| --- | --- | --- |
| Engine | 请求生命周期、输出处理、统计、abort | 直接操作设备 tensor |
| Scheduler | 每 step 调度、KV block 分配、请求状态推进 | 执行模型 forward |
| Executor | 选择执行形态、跨进程/跨节点调用 worker | 理解具体模型结构 |
| Worker | 设备初始化、模型加载、KV cache 初始化 | 决定服务端 API schema |
| Model Runner | 输入准备、模型 forward、采样、attention metadata | HTTP 参数解析 |

这个边界在 vLLM Ascend 中也很重要。Ascend 适配通常发生在 platform、worker/model runner、attention backend、custom op、patch 等位置，而不是重写整个 OpenAI server。

## Worker 生命周期

一次服务启动时，worker 通常会经历这些阶段：

1. 初始化设备和分布式环境。
2. 加载模型权重。
3. 通过 profile 或配置确定可用于 KV cache 的内存。
4. 初始化 KV cache buffer。
5. 进入执行循环，接收 scheduler output。
6. 每个 step 更新本地请求状态，准备 input batch 和 attention metadata。
7. 执行模型 forward、采样，并返回 model runner output。
8. 请求完成或被取消时，释放本地状态。

不同设备后端会在这些阶段做自己的适配，例如内存布局、通信、graph capture、custom op、attention backend 选择。

## Model Runner 的位置

Model runner 是“调度结果”和“模型 forward”之间的桥。它关心的不是 HTTP 请求，而是当前 step 的执行批次。

它通常要处理：

- 新请求和已有请求的本地状态。
- input ids、positions、sequence length。
- block table、slot mapping、attention metadata。
- 多模态或 encoder 输入。
- LoRA、prompt adapter、structured output、spec decode 等扩展状态。
- graph/eager 执行路径选择。
- sampler 输入和输出。

当你排查设备侧 shape、metadata、KV cache 或 attention kernel 问题时，model runner 往往是最重要的观察位置。

## Executor 形态

vLLM 支持多种 executor 形态：

- 单进程：适合最小调试和单卡。
- 多进程：常见于多卡 tensor parallel / pipeline parallel。
- Ray：用于更复杂的分布式部署。
- 外部 launcher：用于和外部调度系统集成。

这些形态的目标是让 engine 看到统一的执行接口，而把进程、设备、通信和失败处理细节封装起来。

## 代码入口

- `$PATH_TO_VLLM/vllm/v1/engine`
- `$PATH_TO_VLLM/vllm/v1/executor`
- `$PATH_TO_VLLM/vllm/v1/worker`
- `$PATH_TO_VLLM/vllm/model_executor`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker`

本章只需要理解层级边界。具体到某个版本时，可以搜索 `Executor`、`Worker`、`ModelRunner`、`SchedulerOutput`、`ModelRunnerOutput` 来定位当前实现。

## 常见问题定位

- 请求排队时间长：优先看 engine/scheduler，而不是 model runner。
- 模型加载失败：优先看 worker、model loader、分布式初始化。
- KV cache OOM：scheduler 和 worker 都要看，前者决定 block 分配，后者实际持有 cache buffer。
- attention shape 不匹配：优先看 model runner 准备的 attention metadata 和 backend 要求。
- 输出文本异常：不一定是 worker 问题，也可能是 output processor 或 detokenizer。

## 思考与探索

1. 为什么 scheduler 不直接调用模型 forward？
2. 如果一个请求已经结束，engine、scheduler、worker 分别需要清理什么状态？
3. vLLM Ascend 为什么可以在 worker/model runner 和 attention backend 层做大量适配，而不必重写 OpenAI entrypoint？

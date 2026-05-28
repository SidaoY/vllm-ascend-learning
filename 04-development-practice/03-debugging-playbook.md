# 调试手册

调试时先把问题归类，再进入对应路径。很多 vLLM Ascend 问题表面上是 kernel 报错，实际原因可能在模型配置、tokenizer、scheduler、KV cache、graph、通信或 CI 环境。下面的手册按“现象、先看、常见原因、验证”组织，方便快速建立排查顺序。

## 通用排查顺序

1. 记录环境：vLLM commit、vLLM Ascend commit、CANN、torch-npu、Python、硬件、镜像。
2. 确认启动参数：模型路径、dtype、并行策略、max model len、graph、prefix cache、spec decode、CP/EP/DP。
3. 判断阶段：启动、模型加载、prefill、decode、sampling、通信、退出清理。
4. 找首个错误：优先看第一个异常栈和 worker 日志，不要只看最后的 timeout。
5. 缩小复现：减少并发、缩短 prompt、关闭可选特性、单卡复现、多卡复现。
6. 对照已知路径：同模型 CPU/CUDA 是否正常，同配置 eager 是否正常，同硬件其他模型是否正常。
7. 补最小测试或复现脚本：让问题能被 review 和 CI 捕捉。

## 启动失败

现象：服务进程启动即退出，或卡在初始化阶段。

先看：

- vLLM 和 vLLM Ascend 版本是否匹配。
- `vllm_ascend` 是否被正确 import。
- NPU 是否可见，驱动、CANN、torch-npu 是否可用。
- 环境变量和 additional config 是否拼写正确。
- patch 是否在预期阶段生效。

常见原因：

- vLLM commit 与 vLLM Ascend 当前适配范围不一致。
- Python 环境里装到了旧 wheel 或多个 vLLM 版本。
- CANN/torch-npu 与镜像或硬件不匹配。
- 平台检测失败，导致没有进入 NPU platform。

验证：

- 用最小模型、最少参数启动。
- 打印 `python -c "import vllm, vllm_ascend"` 确认包路径。
- 查看 `$PATH_TO_VLLM_ASCEND/docs/source/installation.md` 和 `$PATH_TO_VLLM_ASCEND/docs/source/faqs.md`。

## 模型加载失败

现象：权重加载、config 解析、模型类选择或 tokenizer 初始化失败。

先看：

- 模型 checkpoint 是否完整，模型名和 tokenizer 是否来自同一 checkpoint。
- 模型是否在 feature matrix 或模型教程中声明支持。
- dtype、量化配置、trust remote code、revision 是否正确。
- TP/PP/EP 等并行配置是否符合模型结构。

常见原因：

- 模型配置使用了 vLLM Ascend 尚未适配的 architecture。
- 权重格式、量化格式或 safetensors 文件不完整。
- tokenizer/chat template 与模型不匹配。
- 模型特殊 patch 没有生效或只覆盖了部分路径。

验证：

- 用同一 checkpoint 跑最小离线推理。
- 对比 `$PATH_TO_VLLM_ASCEND/docs/source/tutorials/models` 中相近模型配置。
- 查看 `$PATH_TO_VLLM_ASCEND/vllm_ascend/model_loader` 和相关 patch。

## Tokenizer / Chat Template / Tool Call 异常

现象：输出格式异常、首 token 不对、工具调用 JSON 不稳定、同一 prompt 与预期不一致。

先看：

- chat template 是否来自模型 checkpoint。
- request 传入的是 raw prompt、messages，还是 prompt token ids。
- sampling params、stop、stop token ids、structured output 是否一致。
- OpenAI entrypoint 和 offline LLM 路径是否有差异。

常见原因：

- tokenizer 和模型权重来自不同 checkpoint。
- template 里 special tokens、role 名称或 generation prompt 不匹配。
- 输出后处理、stop 条件或 tool parser 触发了截断。

验证：

- 打印 token ids，而不只看文本。
- 用 vLLM 上游同版本路径对照。
- 先关闭 structured output/tool call，再逐步打开。

## CANN / torch-npu / HCCL 环境问题

现象：NPU 不可见、算子找不到、HCCL 初始化失败、通信超时。

先看：

- CANN、torch-npu、PyTorch 版本组合。
- `SOC_VERSION`、visible devices、rank/world size、master addr/port。
- HCCL 相关日志中第一个错误。
- CI runner 或本地机器是否有残留进程占卡。

常见原因：

- 镜像和宿主机驱动/CANN 不匹配。
- 多卡 rank 配置错误。
- 网络、端口或 HCCL 环境变量不一致。
- 某个 rank 先失败，其他 rank 只表现为 timeout。

验证：

- 先跑 torch-npu 最小通信样例或项目内 distributed 单测。
- 单卡能跑后再扩到多卡。
- 多卡失败时分别查看每个 rank 的首个异常。

## 单卡 OOM 或 KV Block 不足

现象：启动时 OOM、prefill OOM、decode 中途 OOM、日志提示 KV cache block 不足。

先看：

- 模型权重大小、dtype、max model len、max_num_seqs、max_num_batched_tokens。
- KV cache dtype、block size、可用 NPU memory。
- graph buffer、workspace、prefix cache、spec decode 额外状态。
- 是否存在长 prompt 或输出长度超预期。

常见原因：

- max model len 配得过大，KV cache 预留压力过高。
- 并发过高导致 KV block 被占满。
- graph、spec decode、CP 或量化路径引入额外 buffer。
- 请求没有及时完成或释放。

验证：

- 降低 `max_num_seqs` 或 `max_num_batched_tokens` 观察是否恢复。
- 缩短 prompt 和 max_tokens 区分权重 OOM 与 KV OOM。
- 查看 `$PATH_TO_VLLM_ASCEND/vllm_ascend/worker/block_table.py` 和 KV 相关日志。

## 多卡通信失败

现象：单卡正常，多卡 hang、timeout、精度异常或吞吐异常。

先看：

- TP/DP/PP/EP/CP 配置是否符合模型和硬件。
- rank group 是否正确建立。
- HCCL 日志是否显示某个 rank 先失败。
- 是否只有某个 attention backend、MoE 或 CP 场景失败。

常见原因：

- rank 到设备映射错误。
- 某个 rank 的输入 shape 或 metadata 不一致。
- collective 调用顺序不一致。
- 通信和 compute overlap 引入同步问题。

验证：

- 从 2 卡最小配置开始。
- 关闭可选特性，例如 graph、CP、spec decode，逐个加回。
- 跑 `$PATH_TO_VLLM_ASCEND/tests/ut/distributed` 相关测试。

## Prefix Cache 命中率异常

现象：公共前缀明显相同，但 TTFT 没下降，或 cache hit 低于预期。

先看：

- prompt token ids 是否真的相同。
- chat template 是否引入动态内容。
- block hash、model id、LoRA、多模态输入等 cache key 相关信息。
- prefix caching 是否与当前调度、KV transfer 或 CP 配置兼容。

常见原因：

- 文本相同但 token ids 不同。
- 请求前缀没有对齐到 block 边界，收益有限。
- 多模态或特殊 metadata 进入 hash 后导致 miss。
- cache 被内存压力驱逐。

验证：

- 记录 token ids 和 prefix cache 相关日志。
- 用固定 prompt 和固定 sampling params 做 A/B。
- 降低并发，排除 cache 被快速驱逐。

## Spec Decode 接受率低或输出不一致

现象：开启投机推理后吞吐不升反降，接受率低，或输出和普通 decode 不一致。

先看：

- draft/proposer 模型是否与 target 模型匹配。
- sampling params 是否适合 spec decode。
- proposer、verify、rejection sampler 是否走到预期路径。
- graph capture 是否影响 shape 或 metadata。

常见原因：

- draft 模型质量不足，接受率天然偏低。
- temperature/top_p 等采样参数放大了拒绝率。
- 特殊模型 MTP/EAGLE 路径适配不完整。
- rejection sampler patch 或 NPU op 与上游语义不一致。

验证：

- 先用 greedy 或低随机性配置对齐。
- 记录 propose token 数、accept token 数、接受率和 TPOT。
- 关闭 graph 对照 eager 路径。

## CP 长序列异常

现象：长序列开启 CP 后精度不一致、通信异常、性能不达预期。

先看：

- CP size、rank group、TP/DP/CP 组合是否正确。
- attention backend 是否支持当前模型的 CP 路径。
- block table、seq lens、position ids、mask 是否按 CP 切分。
- prefill 和 decode 是否走了不同 CP 逻辑。

常见原因：

- PCP/DCP 配置与测试条件不匹配。
- 某个 rank 的序列切分或 padding 不一致。
- CP 通信开销超过 attention 收益。
- MLA/SFA/特殊 RoPE 与 CP metadata 不匹配。

验证：

- 先跑短序列 CP，再逐步拉长。
- 对比不开 CP、开 CP 的 token ids 和 logprobs。
- 查看 `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/feature_guide/context_parallel.md`。

## 负载均衡波动

现象：MoE 场景中某些卡很忙，expert 迁移后延迟抖动，吞吐周期性波动。

先看：

- token load 是否集中在少数 expert。
- expert map 是否更新，迁移是否频繁。
- DP/EP/PD 负载均衡是否同时生效。
- 迁移成本是否超过收益。

常见原因：

- 负载分布变化太快，平衡策略滞后。
- expert 热点明显，但迁移粒度或周期不合适。
- 外部负载均衡和 EPLB 目标不一致。
- 统计窗口太短导致决策抖动。

验证：

- 固定输入分布，观察 expert load 和迁移记录。
- 关闭动态平衡对照 baseline。
- 查看 `$PATH_TO_VLLM_ASCEND/vllm_ascend/eplb` 和 EPLB 文档。

## CI / Nightly 失败

现象：本地通过，CI 失败；或 nightly 某个模型突然失败。

先看：

- workflow 名称、runner、镜像、硬件。
- `VLLM_COMMIT`、`VLLM_ASCEND_COMMIT`、CANN、torch-npu。
- 失败阶段是安装、模型下载、启动、测试执行还是清理。
- 是否只有某个模型、某个硬件或某个节点失败。

常见原因：

- 上游 vLLM commit 变化导致适配失效。
- CI 镜像或模型缓存变化。
- 测试依赖外部网络或资源。
- 多卡测试中某个 rank 的首个错误被 timeout 掩盖。

验证：

- 使用同一 commit 和镜像复现。
- 先跑失败文件，再缩到失败 case。
- 参考 [测试、CI 与 Nightly Workflow](01-testing-ci-and-nightly.md)。

## 性能退化

现象：正确性不变，但 TTFT、TPOT、吞吐或 p99 变差。

先看：

- benchmark 测试条件是否完全一致。
- graph、attention backend、并行策略、KV cache 配置是否变化。
- 是否发生 fallback、cache miss、block 不足或通信等待。
- profiler 中哪个阶段耗时变长。

常见原因：

- 上游调度变化改变 batch shape。
- 某个 kernel 没命中高效路径。
- graph capture 失败后回到 eager。
- 新增日志、同步或 CPU/NPU 拷贝。

验证：

- 先复现 baseline，再二分配置和 commit。
- 对比 profiler 时间线和服务指标。
- 参考 [性能观测与调优](02-observability-and-performance-tuning.md)。

## 参考入口

- `$PATH_TO_VLLM/docs/usage/troubleshooting.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/faqs.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/performance_and_debug`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/configuration/env_vars.md`
- [测试、CI 与 Nightly Workflow](01-testing-ci-and-nightly.md)
- [性能观测与调优](02-observability-and-performance-tuning.md)

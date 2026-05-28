# 仓库地图

## 写作目标

把 vLLM、vLLM Ascend 和当前学习资料的关系讲清楚：哪个目录解决什么问题、目录间怎么跳、第一次该从哪里读起。

## 仓库关系

vLLM Ascend 开发通常涉及三个仓库：

```text
上游仓库：vLLM（推理框架主体）
  ├── entrypoints、engine、scheduler、worker、attention 等核心逻辑
  ├── docs（设计文档和 benchmark）
  └── benchmarks（离线和服务 benchmark 工具）

下游仓库：vLLM Ascend（NPU 适配插件）
  ├── patch（临时 monkey patch，最终应合入上游或本仓库）
  ├── attention（Ascend attention 后端、CP）
  ├── compilation（ACL graph、graph fusion）
  ├── distributed（HCCL 通信、KV transfer）
  ├── eplb（Ascend 侧 expert load balance）
  ├── ops（custom op、triton op）
  ├── quantization（KV cache 量化等）
  ├── spec_decode（Ascend 侧 speculative decoding）
  ├── worker（platform worker、model runner、block table）
  └── tests（ut + e2e）

学习资料仓库：vllm-ascend-learning（本仓库）
  └── 按 topic 整理的概念讲解、代码入口、常见问题和思考题
```

vLLM Ascend 通过 `vllm_ascend.platform` 把自己注册到 vLLM 的 platform 机制中，作为 NPU 后端运行。启动时 vLLM 根据实际硬件加载 Ascend platform，此后该 platform 接管 worker、attention、分布式通信等后端选择。

## vLLM 目录速览

- `vllm/entrypoints`: OpenAI compatible API、LLM class、input/output processor。
- `vllm/v1`: vLLM V1 架构（engine、scheduler、worker、KV cache、spec decode 等）。
- `vllm/model_executor`: 模型定义、layers、attention 接口、量化。
- `vllm/distributed`: 并行状态、通信器、KV transfer 接口。
- `vllm/attention`: 通用 attention backend 和 metadata。
- `vllm/transformers_utils`: 模型加载、配置检测、tokenizer 包装。
- `vllm/compilation`: torch compile、CUDA graph。
- `vllm/platforms`: platform 抽象基类。

## vLLM Ascend 目录速览

- `vllm_ascend/__init__.py`: 自动注册 Ascend platform。
- `vllm_ascend/platform.py`: platform 注册入口和硬件检测。
- `vllm_ascend/envs.py`: 环境变量定义。
- `vllm_ascend/ascend_config.py`: Ascend 特有配置。
- `vllm_ascend/patch`: 使用 monkey patch 临时修改 vLLM 行为。
- `vllm_ascend/worker`: NPU worker、model runner、block table、PCP utils。
- `vllm_ascend/attention`: Ascend attention 后端、CP attention。
- `vllm_ascend/compilation`: ACL graph 捕获和 replay、graph fusion。
- `vllm_ascend/ops`: custom ops、triton kernel、layers。
- `vllm_ascend/distributed`: HCCL 并行状态、device communicator、KV transfer。
- `vllm_ascend/eplb`: Ascend 侧 expert load balance。
- `vllm_ascend/spec_decode`: Ascend 侧 speculative decoding。
- `vllm_ascend/quantization`: KV cache 量化。
- `vllm_ascend/utils.py`: 工具函数。

## 第一次读代码推荐路径

不要从头到尾顺序读，按以下路线建立"代码地图"：

**第一步：找到注入点（10 分钟）**

```bash
cd $PATH_TO_VLLM_ASCEND
rg "platform" vllm_ascend/__init__.py
rg "class NPUPlatform" vllm_ascend/platform.py
```

看懂 `vllm_ascend` 是如何被 vLLM 发现的：`__init__.py` 中注册 platform → `platform.py` 检测 NPU → 返回 platform 实例 → vLLM 用这个 platform 选择 worker 和 attention 后端。

**第二步：跟一条请求（30 分钟）**

沿"入口 → engine → scheduler → worker → model runner"读一遍。先不用深究每个函数的细节，建立整体流动感。

- 起点：`vllm/v1/engine/async_llm.py` 的 `generate()`
- 调度：`vllm/v1/core/sched/scheduler.py` 的 `schedule()`
- 执行：`vllm_ascend/worker/model_runner_v1.py` 的 `execute_model()`
- 输出：`vllm/v1/engine/output_processor.py`

**第三步：读一个 patch（20 分钟）**

选一个简单的 patch，例如 scheduler 或 worker 相关的小 patch，理解：
1. 上游的原始行为是什么
2. patch 做了什么修改
3. 这个修改为什么不能直接改上游或本仓库原生代码

**第四步：读模型加载和执行（30 分钟）**

从 `platform.py` 的 `get_attn_backend()` 和 `get_worker()` 进入，看 Ascend 如何提供 attention 后端、如何创建 NPU worker、worker 如何加载模型和多模态输入。

**第五步：深入一个特性（按需）**

根据当前工作方向，选择 KV cache、spec decode、CP 或 EPLB 中的一个深入阅读。使用 [代码阅读索引](../06-appendix/01-code-reading-index.md) 找到代码入口。

## vLLM 版本和 vLLM Ascend 分支/commit 的对应关系

vLLM Ascend 不是独立分支，而是以 Python 包形式安装在 vLLM 之上。但两者的版本有严格对应关系：

- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/config.yaml` 中记录了 CI nightly 测试时使用的 `VLLM_COMMIT`。
- 本地开发时，vLLM 应该 checkout 到与当前 vLLM Ascend 版本匹配的 commit，否则可能出现：scheduler/worker 接口不匹配、metadata 字段缺失、attention backend 注册方式变化等问题。
- 判断版本是否匹配的快速方法：
  1. 看 `$PATH_TO_VLLM_ASCEND` 的 setup.py/pyproject.toml 中对 vLLM 的版本约束。
  2. 看 CI config 中的 `VLLM_COMMIT`。
  3. 启动时看是否有 import error 或 attribute error。

## 本仓库 vs 源码仓库的对应关系

本仓库（vllm-ascend-learning）是对 vLLM / vLLM Ascend 源码的二次解释，不替代阅读源码。使用时：

1. 先读本仓库对应章节建立概念。
2. 带着概念和"常见问题"去读源码。
3. 读源码时对照本仓库的代码入口索引。
4. 发现源码版本与本仓库描述不一致时，以源码为准，并反馈给本仓库维护者更新。

## 参考入口

- `$PATH_TO_VLLM/docs/design/arch_overview.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/patch.md`
- [学习路线图](00-learning-map.md)
- [代码阅读索引](../06-appendix/01-code-reading-index.md)

## 思考与探索

1. vLLM Ascend 在 `__init__.py` 中注册 platform 后，vLLM 是怎么"发现"并加载它的？
2. 如果上游 vLLM 升级了 worker 接口，vLLM Ascend 的 patch 最容易在哪个阶段暴露问题？
3. 本仓库的代码入口索引里列了 `$PATH_TO_VLLM` 和 `$PATH_TO_VLLM_ASCEND`，为什么不用绝对行号？
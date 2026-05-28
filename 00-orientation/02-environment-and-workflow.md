# 环境与工作流

本章帮助读者建立基本开发环境，回答三个问题：需要安装什么、怎么验证安装成功、日常开发怎么用命令。

本章只做环境入口说明。测试策略、CI 细节、性能观测和故障排查，在 [04-development-practice](../04-development-practice/README.md) 中深入展开。

## 版本关系

vLLM Ascend 同时依赖上游 vLLM、PyTorch、torch-npu 和 CANN。这些组件不是任意版本都能组合。安装前应先确认当前支持的版本组合，入口在：

- `$PATH_TO_VLLM_ASCEND/docs/source/installation.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/support_matrix`

版本不匹配最常见的症状是：import 失败、NPU 不可见、算子找不到、HCCL 初始化失败、启动阶段 segmentation fault。遇到这些问题，建议先对照 support matrix 确认版本组合，而不是直接怀疑代码。

典型依赖链：

```text
CANN（底层计算框架）
  ├── torch-npu（PyTorch NPU 运行时）
  │     └── vLLM（推理框架）
  │           └── vLLM Ascend（NPU 后端适配）
  └── HCCL（集合通信库）
        └── vLLM Ascend distributed
```

## 安装方式

vLLM Ascend 一般以 editable install 方式使用，方便本地改代码后立即生效：

```bash
cd $PATH_TO_VLLM_ASCEND
pip install -e .
```

vLLM 主仓库同理：

```bash
cd $PATH_TO_VLLM
pip install -e .
```

安装前确认 Python 环境干净，避免多个 vLLM 版本共存。可以用 `pip list | grep vllm` 检查已安装的 vLLM 相关包。

环境变量通常需要在启动前设置好。核心变量包括：

- NPU 可见设备：`ASCEND_RT_VISIBLE_DEVICES` 或类似变量，控制哪些 NPU 对进程可见。
- CANN 路径：`ASCEND_HOME`、`ASCEND_TOOLKIT_HOME` 等，指向 CANN 安装目录。
- SOC_VERSION：目标 Ascend 硬件型号，例如 `Ascend910B2`、`Ascend910C`。
- HCCL 相关：多卡通信时需要配置 master addr、world size、rank 等。

示例（具体变量名以当前文档和镜像为准）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export SOC_VERSION=Ascend910B2
```

## 最小验证

安装完成后，先做最小验证，确认 NPU 可见且 vLLM Ascend 被正确导入：

```bash
python -c "import torch; import torch_npu; print(torch_npu.npu.is_available())"
python -c "import vllm; import vllm_ascend; print('OK')"
```

如果第二步报错，先检查 vLLM 和 vLLM Ascend 版本是否匹配、`pip list` 是否有多余 vLLM 包。

## 最小离线推理

用一个较小模型跑离线推理，是验证环境最直接的方式：

```bash
cd $PATH_TO_VLLM
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='your-model-path', max_model_len=2048)
outputs = llm.generate(['Hello, world!'], SamplingParams(max_tokens=16))
print(outputs[0].outputs[0].text)
"
```

如果模型路径、tokenizer、权重布局、dtype 或 CANN 有问题，这一步会直接暴露。能用离线推理跑通后，再进入 server 模式。

## OpenAI Compatible Server 启动和请求

启动服务：

```bash
cd $PATH_TO_VLLM
python -m vllm.entrypoints.openai.api_server \
  --model your-model-path \
  --max-model-len 2048 \
  --port 8000
```

用 curl 或 Python `openai` 客户端发请求：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hi!"}],
    "max_tokens": 16
  }'
```

如果能收到正常 JSON 响应，说明从 API 入口到模型执行到输出处理的链路是通的。

## 常用环境变量速查

这里列出关键环境变量，详细说明见正式文档：

| 类别 | 变量示例 | 作用 |
| --- | --- | --- |
| NPU 可见性 | `ASCEND_RT_VISIBLE_DEVICES` | 控制进程可见的 NPU |
| 硬件型号 | `SOC_VERSION` | 指定目标 Ascend 硬件 |
| Graph | `VLLM_ASCEND_GRAPH_MODE` | 启用/关闭 NPU graph |
| Attention | `VLLM_ASCEND_ATTENTION_BACKEND` | 选择 attention 后端路径 |
| Patch | 各类 patch 开关 | 控制特定 patch 是否生效 |
| HCCL | `HCCL_*` 系列 | 多卡通信配置 |
| CANN | `ASCEND_HOME`、`ASCEND_TOOLKIT_HOME` | CANN 安装路径 |
| Profiling | `VLLM_ASCEND_PROFILING_*` | 性能采集开关 |
| Debug | `VLLM_ASCEND_DEBUG_*` | 调试和日志开关 |

具体变量名以当前文档为准：
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/configuration/env_vars.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/configuration/additional_config.md`

## 日志

vLLM 和 vLLM Ascend 的日志输出主要受 Python logging 级别和 `VLLM_LOGGING_LEVEL` 等变量控制。排查问题时建议：

- 启动阶段：看 stdout/stderr 的第一个异常，不要只看最后一行。
- 服务阶段：看 engine log、worker log、scheduler log。
- 多卡场景：每个 rank 的日志都要看，超时往往是某个 rank 先失败。
- 增加日志级别：`VLLM_LOGGING_LEVEL=DEBUG` 可以看到更多调度和 KV cache 信息。

日志里出现 `fallback`、`patch`、`unsupported`、`timeout`、`OOM`、`shape mismatch` 等关键词时，可以先在 [调试手册](../04-development-practice/03-debugging-playbook.md) 中搜索对应章节。

## Benchmark 基础命令

离线推理 benchmark：

```bash
cd $PATH_TO_VLLM
python benchmarks/benchmark_offline.py \
  --model your-model-path \
  --input-len 1024 \
  --output-len 128 \
  --num-prompts 100
```

服务模式 benchmark：

```bash
cd $PATH_TO_VLLM
python benchmarks/benchmark_serving.py \
  --model your-model-name \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 100
```

命令的具体参数随 vLLM 版本变化，建议运行时先 `--help` 确认。

性能分析和深入调优放在 [性能观测与调优](../04-development-practice/02-observability-and-performance-tuning.md)。

## 测试选择策略

开发时的测试选择可以按这个原则：

- 改配置/platform/patch：先跑相关单测和一个最小启动 e2e。
- 改 worker/model runner：跑 worker 单测 + 单卡 e2e。
- 改 attention/ops：跑 attention/ops 精度单测 + 对应模型 e2e。
- 改 distributed/KV transfer：跑 kv_connector 单测 + 多卡 e2e。
- 改 scheduler：跑 core 单测 + 高并发或长 prompt e2e。

详细分层和 CI 流程见 [测试、CI 与 Nightly Workflow](../04-development-practice/01-testing-ci-and-nightly.md)。

本地常用命令形态：

```bash
cd $PATH_TO_VLLM_ASCEND

# 单测
pytest -sv tests/ut/worker/test_block_table.py

# 单卡 e2e
pytest -sv tests/e2e/singlecard/test_attention_fa3.py

# 多卡 e2e（需要多卡环境）
pytest -sv tests/e2e/multicard/2-cards/spec_decode
```

## 常见环境错误排查表

| 现象 | 先查什么 | 常见原因 |
| --- | --- | --- |
| `import vllm_ascend` 失败 | `pip list`、Python 路径 | 未安装、多版本冲突、路径错误 |
| NPU 不可见 | `ASCEND_RT_VISIBLE_DEVICES`、`npu-smi` | 设备未挂载、权限、驱动问题 |
| CANN 相关报错 | `SOC_VERSION`、`ASCEND_HOME` | CANN 版本与硬件不匹配 |
| torch-npu 算子找不到 | torch-npu 版本、CANN 版本 | 版本组合不支持 |
| HCCL init 失败 | rank 配置、网络、端口 | 多卡配置错误、网络不通 |
| 模型下载失败 | 网络、镜像缓存、权限 | 离线环境、代理、认证 |
| OOM 启动即退 | max_model_len、dtype | 权重 + KV cache 预留超过显存 |
| 单卡正常多卡失败 | HCCL 日志、rank 配置 | 某个 rank 先失败、通信不同步 |

## 参考入口

- `$PATH_TO_VLLM/docs/getting_started/quickstart.md`
- `$PATH_TO_VLLM/docs/configuration/env_vars.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/installation.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/quick_start.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/user_guide/support_matrix`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/testing.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/faqs.md`
- [测试、CI 与 Nightly Workflow](../04-development-practice/01-testing-ci-and-nightly.md)
- [调试手册](../04-development-practice/03-debugging-playbook.md)

## 思考与探索

1. 如果 `import vllm_ascend` 成功但 `torch_npu.npu.is_available()` 返回 False，问题更可能在哪一层？
2. 为什么建议先跑通离线推理再启动 server？
3. 多卡环境下，一个 rank 的日志显示 timeout，为什么不能只看这个 rank 的日志？
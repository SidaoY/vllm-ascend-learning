# 测试、CI 与 Nightly Workflow

vLLM Ascend 的测试目标不是“把所有测试都跑一遍”。更实际的目标是：根据改动影响面选择足够小、足够相关、足够能暴露风险的验证集合。单测负责快速定位逻辑问题，e2e 负责确认用户行为，nightly 负责覆盖成本较高的模型、硬件和多卡组合。

## 测试分层

| 层级 | 适合验证什么 | 典型入口 |
| --- | --- | --- |
| Python 单测 | 配置解析、patch 条件、metadata 构造、block table、采样逻辑 | `$PATH_TO_VLLM_ASCEND/tests/ut` |
| Attention / ops 单测 | shape、dtype、layout、精度、kernel 选择 | `$PATH_TO_VLLM_ASCEND/tests/ut/attention`、`$PATH_TO_VLLM_ASCEND/tests/ut/ops` |
| 单卡 e2e | OpenAI API、模型加载、prefill/decode、graph、sampling、LoRA 等用户路径 | `$PATH_TO_VLLM_ASCEND/tests/e2e/singlecard` |
| 多卡 e2e | TP、DP、CP、通信、长序列、spec decode 多卡路径 | `$PATH_TO_VLLM_ASCEND/tests/e2e/multicard` |
| light e2e | PR 阶段较轻量的冒烟覆盖 | `$PATH_TO_VLLM_ASCEND/tests/e2e/light` |
| nightly / weekly | 重点模型、长耗时、多节点、回归风险 | `$PATH_TO_VLLM_ASCEND/tests/e2e/nightly`、`$PATH_TO_VLLM_ASCEND/tests/e2e/weekly` |
| 文档和静态检查 | doc link、lint、类型和格式问题 | `$PATH_TO_VLLM_ASCEND/.github/workflows` |

新增特性时，通常至少需要一个单测说明核心逻辑，一个 e2e 说明用户路径能跑通。涉及性能或长上下文时，还需要 benchmark 或 nightly 覆盖。

## 本地常用命令形态

下面这些命令是形态示例，实际运行前要根据本地环境、模型路径和硬件资源调整：

```bash
cd $PATH_TO_VLLM_ASCEND

# 跑某个单测文件
pytest -sv tests/ut/worker/test_block_table.py

# 跑某个具体测试用例
pytest -sv tests/ut/attention/test_attention_cp.py::test_attention_cp

# 跑某个特性目录
pytest -sv tests/ut/spec_decode

# 跑单卡 e2e 中的某个文件
pytest -sv tests/e2e/singlecard/test_attention_fa3.py
```

如果测试依赖真实 NPU、模型下载、私有镜像或 CI 环境变量，本地失败不一定等于代码错误。先确认测试是否本来就要求特定环境。

## 选择测试范围

可以按改动类型选择：

- 改 platform/env/config：跑配置相关单测、启动类 e2e、受影响 feature 的 smoke test。
- 改 patch：跑 patch 单测、被 patch 影响的用户路径、一个无关路径防止误伤。
- 改 worker/model runner：跑 worker 单测、单卡 e2e、必要时跑 graph 和多卡。
- 改 attention backend：跑 attention 精度单测、对应模型 e2e、长上下文或 decode 场景。
- 改 custom op：跑 op 单测、精度对齐、性能 benchmark、graph capture 场景。
- 改 distributed/KV transfer：跑 kv_connector 单测、PD 或多卡 e2e、相关 nightly。
- 改 scheduler：跑 core 单测、动态 batch、长 prompt、高并发、prefix cache 或 spec decode 相关用例。

测试范围不是越大越好。好的测试组合应该能直接回答：“这次改动最可能破坏什么？”

## CI Workflow 地图

常见 workflow 可以先这样理解：

- `pr_test_light.yaml`：PR 阶段的轻量测试入口，会结合 smart scope 选择部分测试。
- `pr_test_full.yaml`：更完整的 PR 验证入口，成本更高。
- `_e2e_test.yaml`：可复用的 e2e workflow 模板。
- `_optional_smart_e2e.yaml`：按变更范围选择 e2e 子集。
- `schedule_nightly_test_a2.yaml`、`schedule_nightly_test_a3.yaml`：定时 nightly 入口。
- `_e2e_nightly_single_node.yaml`、`_e2e_nightly_multi_node.yaml`：nightly 中单节点和多节点执行模板。
- `_e2e_nightly_single_node_models.yaml`：模型类 nightly 测试执行模板。
- `schedule_weekly_test_a3.yaml`：weekly 测试入口。
- `dispatch_main2main_bisect.yaml`：用于定位上游 vLLM main 到 vLLM Ascend main 之间的回归。

脚本和配置入口：

- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/config.yaml`
- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/upstream_config.yaml`
- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/ut_config.yaml`
- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/run_suite.py`
- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/determine_smart_e2e_scope.py`
- `$PATH_TO_VLLM_ASCEND/.github/workflows/scripts/ci_log_summary.py`

## Nightly 里重点看什么

Nightly 失败时，先不要直接跳到业务代码。建议按下面顺序看：

1. 触发信息：是定时、手动、PR 触发，还是上游同步触发。
2. 环境信息：镜像、硬件、CANN、torch-npu、Python、SOC_VERSION。
3. 代码版本：`VLLM_COMMIT` 和 `VLLM_ASCEND_COMMIT` 是否符合预期。
4. 安装阶段：vLLM、vLLM Ascend、依赖和 custom kernels 是否安装成功。
5. 模型阶段：模型是否下载成功，是否命中离线缓存，权限是否正常。
6. 测试阶段：失败是单个 case、整组 case、某张卡、某个 runner，还是超时。
7. 日志摘要：先看 CI summary，再看 pytest 原始日志和 worker 日志。

`VLLM_COMMIT` 很关键。vLLM 社区变化快，同一份 vLLM Ascend 代码配不同 vLLM commit，失败含义可能完全不同。

## 精度测试注意点

精度失败要先区分三类问题：

- 输入不同：tokenizer、chat template、sampling params、随机种子或 prompt 不一致。
- 路径不同：eager/graph、attention backend、dtype、量化、rank 切分不一致。
- 数值不同：kernel 精度、容差、累积误差、非确定性。

对于生成文本，不要只看最终字符串。更可靠的方式是同时关注 token ids、logprobs、top-k 候选、接受率、stop reason 和中间 metadata。对于 attention/op 单测，要明确 tolerance，并说明为什么这个容差合理。

## 新增测试的基本原则

- 测试名体现行为，不只是体现实现细节。
- 尽量构造最小输入，让失败能快速定位。
- 对 feature 开关、环境变量和异常路径也要覆盖。
- 不要把重模型、大下载、多卡长耗时测试塞进普通单测。
- 对 flaky 场景要先定位随机性来源，再决定是否放入 nightly。
- 如果测试依赖外部模型或特定硬件，要在注释或文档里说明。

## CI 失败分析模板

可以在 issue 或 PR comment 中按这个格式记录：

```text
现象：
影响范围：
触发 workflow：
VLLM_COMMIT：
VLLM_ASCEND_COMMIT：
CANN / torch-npu：
失败阶段：
首个错误日志：
本地是否可复现：
已排除项：
下一步：
```

## 参考入口

- `$PATH_TO_VLLM_ASCEND/tests`
- `$PATH_TO_VLLM_ASCEND/.github/workflows`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/testing.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/e2e_ci_test.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/nightly_ci_test.md`
- `$PATH_TO_VLLM/docs/usage/troubleshooting.md`

## 思考与探索

1. 一个 patch 只改了模型加载失败，为什么仍然可能需要跑一个无关模型的启动测试？
2. Nightly 里同一个 case 在 A2 通过、A3 失败，优先检查代码还是环境和 kernel 支持？
3. 如何设计一个测试，让它既能覆盖 KV transfer 生命周期，又不会把模型下载和网络波动变成主要变量？

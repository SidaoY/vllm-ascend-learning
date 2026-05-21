# 环境与工作流

## 写作目标

说明新人如何准备环境、运行基本推理、执行测试、定位日志，并理解 CI/nightly 与本地开发的关系。

## 建议覆盖内容

- Python、CANN、torch-npu、vLLM、vLLM Ascend 的版本关系。
- 本地安装方式：editable install、依赖安装、环境变量。
- 最小离线推理命令。
- OpenAI compatible server 启动和请求示例。
- 单测、e2e、nightly 的选择策略。
- 重要环境变量：NPU 可见设备、patch 开关、调度开关、graph 开关、HCCL 相关变量。
- 日志位置和日志级别。
- benchmark 基础命令。
- 具体测试、CI、性能观测、故障排查放在 `../04-development-practice/README.md`，本章只做环境入口说明。

## 参考入口

- `$PATH_TO_VLLM/docs/getting_started/quickstart.md`
- `$PATH_TO_VLLM/docs/configuration/env_vars.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/installation.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/quick_start.md`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/contribution/testing.md`
- `$PATH_TO_VLLM_ASCEND/.github/workflows`

## 待补内容

- 本团队推荐的标准开发环境。
- 常用启动脚本。
- 常见环境错误排查表。

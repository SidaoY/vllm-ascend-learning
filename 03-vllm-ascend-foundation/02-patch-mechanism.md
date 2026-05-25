# Patch 机制

vLLM Ascend 通过 patch 机制补齐上游 vLLM 与 Ascend 后端之间的临时缺口。Patch 的存在不是为了绕开设计，而是为了在上游快速演进、扩展点尚未稳定、NPU 有特殊约束时保持功能可用。

## 为什么需要 patch

常见原因包括：

- 上游 vLLM 还没有提供 NPU 需要的扩展点。
- 上游实现假设了 GPU 行为，NPU 需要不同处理。
- 某个模型新特性需要先在 Ascend 上验证。
- 上游 bug 尚未合入当前支持的 vLLM 版本。
- 配置、schema、模型注册、分布式、sampling、graph 等路径需要临时兼容。

Patch 应该是有生命周期的：引入时说明原因，使用中有测试保护，条件成熟后推动上游或删除。

## 两类 patch

Platform patch：在 worker 启动前应用。适合影响全局行为、config 校验、模型注册、分布式基础类、schema 或 engine 初始化的改动。

Worker patch：在 worker 侧应用。适合影响设备侧执行、模型 forward、算子替换、sampling、graph、权重加载等和 worker 生命周期绑定的改动。

| 步骤 | 阶段 | 说明 |
| --- | --- | --- |
| 1 | NPU platform selected | NPU platform 被选中 |
| 2 | Apply platform patches | 应用 platform-level patches |
| 3 | Engine / executor setup | 初始化 engine / executor |
| 4 | Worker starts | Worker 启动 |
| 5 | Apply worker patches | 应用 worker-level patches |
| 6 | Load model and execute | 加载模型并执行 |

判断 patch 放在哪里，核心看“它必须在什么时候生效”。如果放晚了，上游对象可能已经初始化，patch 可能无效或只影响部分进程。

## Patch 的常见形式

- 替换函数或方法。
- 包装原函数，在前后增加 NPU 逻辑。
- 注册新的 backend、connector、parser 或 alias。
- 替换某个模型子模块的实现。
- 修改默认配置或绕过不适合 NPU 的校验。
- 在特定硬件、模型、版本或环境变量下启用兼容逻辑。

无论形式如何，都要尽量缩小作用范围，避免影响无关模型和无关平台。

## Patch 风险

Patch 的风险通常来自：

- 导入顺序：patch 生效前对象已被引用或缓存。
- 多进程：父进程 patch 不一定自动影响子进程。
- schema 缓存：配置类或数据模型可能在 patch 前已经生成校验 schema。
- 版本漂移：上游方法签名、类名、文件位置变化。
- 测试隔离：一个 patch 影响另一个测试。
- 条件过宽：为了一个模型改动影响所有模型。
- 缺少退出计划：patch 长期堆积，后续维护成本越来越高。

## 新增 patch 的建议流程

1. 明确问题属于上游缺口、NPU 约束、模型临时适配还是本地 bug。
2. 先寻找上游已有扩展点，能不用 patch 就不用 patch。
3. 决定 platform patch 还是 worker patch。
4. 缩小触发条件，例如模型名、硬件、vLLM 版本、环境变量或配置。
5. 在 patch 说明中写清 Why、How、影响范围和退出条件。
6. 补最小单测或 e2e，覆盖 patch 生效和不生效两种情况。
7. 如果是上游缺口，记录后续 upstream PR 或删除计划。

## Review Checklist

- 是否真的需要 patch，而不是改 vLLM Ascend 原生实现或推动上游？
- patch 是否足够小，条件是否足够窄？
- patch 是否 idempotent，多次导入是否安全？
- 多进程或 worker 侧是否会正确生效？
- 是否有日志或注释帮助后续维护者理解原因？
- 是否有测试覆盖失败路径和正常路径？
- 是否写明何时可以删除？

## 代码入口

- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/platform`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/patch/worker`
- `$PATH_TO_VLLM_ASCEND/vllm_ascend/utils.py`
- `$PATH_TO_VLLM_ASCEND/docs/source/developer_guide/Design_Documents/patch.md`
- `$PATH_TO_VLLM_ASCEND/tests/ut/patch`

建议搜索关键词：`adapt_patch`、`platform patch`、`worker patch`、`monkey-patch`、`Remove this patch`。

## 思考与探索

1. 如果一个 patch 影响 config 校验，它应该放在 platform 还是 worker？为什么？
2. 为什么 patch 需要退出计划？
3. 阅读一个已有 patch，试着判断它解决的是上游缺口、NPU 约束还是模型特殊行为。

# Task13b Minor M1 专项检查（2026-11-12）

结论：M1 已通过局部修复关闭；独立专项 Review **Approved，Critical 0 / Important 0 / Minor 0**。不重新验收 Task13b 的生产计算，不推进 Task13c/14，不 Commit/Push。

## 根因及影响

已阅读 Task13b 实施报告、`review.md`、`review-fix1.md`、四份原始 fault 日志和正式 GREEN 日志。原始四份故障日志均包含 `_pytest/config/__init__.py:1290` 的 `PytestAssertRewriteWarning: Module already imported so cannot be rewritten; anyio`；正式 GREEN 未见此警告。

**根因是模型依赖的提前导入，不需要嵌套或重复 pytest。** 原探针虽已 `import pytest`，但尚未调用 `pytest.main()` 初始化 assertion rewriting，便导入模型栈。实际导入栈：`model_runner → nanovllm.config → transformers.utils.hub → huggingface_hub.hf_api → httpcore._synchronization → anyio`。pytest 随后为自动发现的 pytest11 插件包登记 rewriting 时发现 anyio 已存在于 `sys.modules`，产生警告。只导入 pytest 并不会安装该 hook。

三个全新进程、各一次 `pytest.main()` 的 collect-only 实验：无预导入时无警告；先导入 anyio、先导入 ModelRunner 都出现同一警告，三个进程均 exit 0。原报告“嵌套 pytest 产生”是过于笼统的归因，本次予以纠正。证据是 `m1/matrix.json`、三个对应日志及 `m1/anyio-import-stack.log`，并核对本机 pytest 的 `_consider_importhook`、`_mark_plugins_for_rewrite`、`mark_rewrite` 源码。

该警告不表示正式测试断言失效：这里涉及已经导入的 anyio 包无法补做 rewriting，不是 Task13b 测试模块的断言被关闭。原故障仍到达有效断言。但它污染故障证据，在 warnings-as-errors 环境可能阻止探针正常运行；维护者若复制“先模型、后 pytest.main”写法还会复发。当前 CI 入口使用 `python -m pytest`，在测试导入模型前初始化 pytest；现有 quick 和完整 CPU 的严格警告回归均清洁。未运行远程 CI，不对任意依赖版本或未来入口作保证。

## 修改范围

- `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13b/fault_probe.py`：将模型导入和故障注入移入 `FaultPlugin.pytest_sessionstart`；通过 `config.add_cleanup` 恢复补丁，保持补丁覆盖整个测试生命周期。故障方法、替换内容、节点及 GPU gate 保持。新 JUnit 输出至 `m1/`，保留原 fault XML。
- 同目录 `m1/test_fault_probe.py`：三个独立子进程执行真实 CPU 故障，检查预期 exit 1、具体故障断言文本及无 rewriting warning；报告目的地覆盖到 tmp_path，防止覆盖历史 XML。回归留在诊断证据目录，不使正式 quick CI 导入模型栈。
- 本报告、原 Task13b 实施报告的专项追加记录、实施计划中的专项记录、证据目录 `summary.json` 更新 M1 状态；新日志、diff、审查和审计存于 `m1/`。

没有改正式生产文件、正式测试、依赖配置、Warning 全局过滤、Oracle 或阈值；没有放宽断言、跳过测试、关闭插件或关闭 assertion rewriting。`protected-audit.json` 确认最终冻结的生产/正式测试 SHA 一致，Oracle 141 文件 SHA 一致。

## RED → GREEN 与回归

| 检查 | 结果 |
| --- | --- |
| 新回归 RED | 3 failed，均在无 Warning 断言失败；此前故障 exit 1 和预期断言文本检查已通过 |
| 新回归 GREEN | 3 passed，6.64s |
| 新回归严格 Warning GREEN | 3 passed，7.41s |
| 完整 CPU，host，严格 Warning | 324 passed、32 deselected，10.12s |
| quick CI，严格 Warning | 31 passed、325 deselected，2.02s |
| 禁止模型栈导入 quick CI | 31 passed、325 deselected，1.67s |
| 真实 GPU live_output 故障，严格 Warning | 预期 exit 1、1 failed，13.67s；命中 `live profile output would cancel activation reserve`，无 rewriting warning |
| 未注入故障的正式 GPU 显存测试，严格 Warning | 1 passed，26.82s |
| scoped Ruff / format / git diff --check | 通过 |

严格 Warning 使用 `-W error::pytest.PytestAssertRewriteWarning`；对子进程通过 `PYTEST_ADDOPTS` 继承，是加强检查，并非屏蔽。完整 CPU 首次 sandbox 为 293 passed / 31 errors，均 Gloo socket 权限错误；GPU 在 sandbox 不可见，host 识别 RTX 5060 Ti。原失败日志 `m1/cpu.log/xml` 保留，host 重跑成功日志为 `cpu-host.log/xml`，不把环境错误计成通过。回归测试编写初次出现报告路径 IndexError，保留为 `test-authoring-error.log`，不计为有效 RED；修正测试后才记录有效 `red.log`。

正式 GPU 测试固定写回原 `resources.json`，本次新资源结果已归档 `m1/resources-gpu-green.json`。原验收资源数值由保留的 `resources-first-green.json` 加最终冻结源码 SHA 重新绑定恢复；没有事前字节快照，不声明恢复后的文件与旧文件逐字节相同。原 fault/GREEN 历史日志和 XML 未覆盖；本轮资源波动不替换原验收结论。

独立 Reviewer 另以只读观察插件确认：故障在 sessionfinish 仍生效，`pytest.main()` 返回后原方法 identity 恢复。Review 见 `m1/review.md`；M1 从 deferred 改为 closed，原首轮和 fix1 审查保留原文作为历史。

## 复现

仓库根目录执行；证据根目录为 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13b`，以下 B 仅为便捷变量。完整 CPU 需要允许 Gloo socket；真实 GPU 需要 host 设备访问。故障探针预期 exit 1。

```bash
B=.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13b
export PYTHONPATH=.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export PYTEST_ADDOPTS='-W error::pytest.PytestAssertRewriteWarning'
.venv/bin/python -m pytest "$B/m1/test_fault_probe.py" -q
.venv/bin/python "$B/fault_probe.py" cancel_activation
.venv/bin/python "$B/fault_probe.py" omit_reserve
.venv/bin/python "$B/fault_probe.py" clamp_negative
.venv/bin/python "$B/fault_probe.py" live_output
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py::MemoryTests -m gpu --require-gpu -q
```

最后一条正式测试会刷新其固定路径的资源 JSON，重复运行前需另存历史资源报告。根因的最小对照：新进程中执行 `import pytest; import anyio; pytest.main(['tests/test_qwen3_vl_memory.py', '--collect-only', '-q'])`；删除 `import anyio` 则无该警告（使用本轮同一环境且未设置严格 Warning 参数）。

# Task 7：单活动请求、容量准入与 prefix cache 隔离

2026-11-02，按已批准 Plan 完成 Task7。分支 `feat/multimodal`，HEAD 仍为 `89c1e40e9aa52867ad107954496b3d342f24170d`。Task2–5 已提交，Task6 已完成但未提交；本轮保留这些成果及用户个人实验文件，未开始 Task8，未暂存、Commit 或 Push。

## 实现说明

生产代码只涉及 `nanovllm/engine/block_manager.py` 和 `nanovllm/engine/scheduler.py`。

- `BlockManager(num_blocks, block_size, *, enable_prefix_cache=True)` 保持兼容默认值。VL 传入 `False`，`can_allocate` 不查 hash、`hash_blocks` 不计算或发布 hash；禁止在关闭模式传入非零 cached-block 数。仍正常分配、追加和释放物理块，不禁用语言 KV Cache。
- `Scheduler.add` 在 VL 分支先检查 `L + max_tokens <= max_model_len`、`L <= max_num_batched_tokens`，以及 `ceil((L + max_tokens) / block_size) <= 池总块数`。不满足就抛出 `ValueError`，队列和块状态不变。这里比较总池，而非当前空闲块，因此前一个请求占块时，合法后续请求仍可排队。
- 独立的小型 VL 调度分支：存在 running 请求时只调度它的一个 Decode token；完成释放后，才准入下一个 waiting 请求。Prefill 一次调度完整 L；准入前再次检查空闲块能覆盖完整未来预算。物理块仍按需分配，不预分配所有未来块，也不进入抢占重算。
- 如果准入时未来容量不足，或 Decode 无法追加，则抛出清晰的 KV capacity invariant `RuntimeError`。请求保留在原队列，供调用方执行 `abort`；本任务没有提前接入 Task8 的上层异常捕获。
- `Scheduler.abort(seq)` 从 waiting/running 移除请求，释放其块、清空 cached/scheduled 计数并置 `FINISHED`；对 waiting、已调度 Prefill、Decode、已完成请求重复调用均安全，不影响其他请求。

旧 Qwen3 继续走原 prefix/chunked-prefill/preempt 算法。与 Task7 开始时快照比较，旧 `schedule` 主体 AST 完全相同；`postprocess`、`preempt`、既有块分配/释放及追加方法 AST 均未变。BlockManager 的格式调整修正了原有单行 E701，不改变相关方法语义。Task6 其他三个生产文件 SHA256 未变。

测试新增 `tests/test_qwen3_vl_scheduler.py`。另外只为 Task6 测试的 `SimpleNamespace` 配置补上 `max_model_len=1024`，满足新容量契约；旧断言不变。新测试依赖模型配置栈，沿用现有 `hf_reference` 标记并延迟导入，使最小依赖快速 CI 可正常收集；完整 CPU 命令仍执行全部新增测试。

## TDD、环境错误与修复

先运行完整 CPU 基线：沙箱内 **130 PASSED、0 FAILED、13 ERROR、0 SKIPPED、15 DESELECTED**。13 项全部在真实 `ProcessGroupGloo` 创建 TCP Socket 时发生 `Operation not permitted`，是 setup 环境错误，不计为功能 RED。对单条 CPU 命令申请沙箱外权限后，基线 **143 PASSED、0 FAILED、0 ERROR、0 SKIPPED、15 DESELECTED**，确认问题来自沙箱 Socket 限制。

随后先写 Task7 测试并运行：**17 FAILED、0 ERROR、0 SKIPPED**。失败确实暴露 waiting 优先策略、容量校验缺失、hash 发布、abort/新缓存开关接口缺失；其中新接口缺失的 AttributeError/TypeError 是功能过渡失败，不宣称 17 项全部为数值断言失败。实现后 Task7、Task6 和旧 Qwen3 CPU 定向组合 **56 PASSED、5 DESELECTED**。

自查另发现新测试顶层导入会破坏最小依赖 CI。诊断脚本禁止导入可选模型栈，先复现 **1 项收集 ERROR**（xxhash），再按仓库既有模式改为延迟导入及标记，得到 **31 PASSED、144 DESELECTED、0 ERROR、0 SKIPPED**。这项诊断只检查快速 CI 的依赖边界，不替代 Gloo、完整 CPU 或模型测试。修复后重新执行完整 CPU，仍为 160 项通过。未修改 Gloo fixtures、测试断言或 GPU opt-in 门槛，未用 skip、Mock Gloo 掩盖环境错误。

## 最终真实结果

| 检查 | PASSED | FAILED | ERROR | SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Task7 CPU | 17 | 0 | 0 | 0 | 0 | 1.04s |
| 最终完整 CPU（沙箱外真实 Gloo） | 160 | 0 | 0 | 0 | 15 | 8.91s |
| 最小模型导入诊断下快速 CPU | 31 | 0 | 0 | 0 | 144 | 1.65s |
| 旧 Qwen3 真实 GPU 回归 | 2 | 0 | 0 | 0 | 9 | 8.04s |

常规快速 CPU 亦通过，准确用时及上述各次机器可读统计见 `task7/summary.json`。DESELECTED 是显式 marker 选择，不是 SKIPPED；完整 CPU 中未选择的 GPU 测试不计作通过。新增调度测试检查真实 Scheduler、Sequence、BlockManager，无模型替身：

- 255/256/257 页边界、连续三个请求，Prefill 完整长度及每步物理块数量。
- 等待请求不占块，活动请求先完成 Decode；池暂被占满仍允许合法排队。
- 超池、超总长度、超 Prefill token 预算立即拒绝，free/used/ref_count/hash 不变。
- 相同长 token IDs 的连续请求零前缀命中、零 hash 发布；另植入有效旧 hash 条目独立检验不查找。
- waiting、Prefill 已调度、Decode、正常完成后的重复 abort，检查块释放和后续请求可继续。
- 使用真实额外块所有者破坏容量不变量，验证明确报错且不抢占；调用方 abort 后恢复排队处理，最终 ref_count 全零、`is_finished=True`。
- 旧 Qwen3 prefix hit/ref_count、chunked prefill、内存抢占 CPU 回归；GPU 保留既有短生成和固定 8-token 预期，未重采样或修改 fixture。

CUDA 与 Gloo 单独确认：沙箱内 CUDA `False / 0`，经独立授权的宿主探测为 `True / 1`，设备 NVIDIA GeForce RTX 5060 Ti，torch `2.9.1+cu130` / CUDA `13.0`。确认可见后才执行两项真实 GPU 回归，显式 `--require-gpu`，退出码 0。没有 sudo、全局 Full access 或 CUDA 环境修改。

Ruff check/format 检查四个相关 Python 文件通过，`git diff --check` 通过。远端 CI 未运行；最小依赖检查是本地禁止可选模型导入的诊断，没有新装依赖。

## Review 与验证边界

独立 Reviewer `/root/task7_review` 只读审查 Task7 起始快照到最终代码的差异及 Plan/Spec；独立运行定向 CPU 组合 **56 PASSED、5 DESELECTED**，最终新测试 **17 PASSED**。结论 Task7 可通过：无 Critical、无未解决 Important、无新增 Minor。Reviewer 阅读了执行者的快速 CPU/GPU 日志，没有独立复跑 GPU 或完整 CPU。

本轮验证到调度、预算、缓存策略和块所有权元数据层，以及旧模型真实生成兼容。**尚未验证**完整 VL Runner 路由、VL Decode 的物理 KV 内容、VL 端到端生成、真实不同图片及 rope_delta/DeepStack 隔离、初始化异常和进程级 CUDA 恢复；分别留给 Task8 及后续输入/视觉/生命周期任务。上层自动调用 abort 尚未实现。没有调整或使用未批准的宽 BF16 阈值，没有改变持久化只读 HF Oracle，没有新增数值正确性或性能结论。

接受 Reviewer 的上述范围划分；空队列 schedule 和复用已完成 Sequence 沿用现有 fresh-request 生命周期，不新增重用契约。Task6 已记录的非阻塞 Minor（全正温度均匀 logits 对照不能单独检测温度缩放回归）继续保留，本轮未修复、未重新归为 Task7 问题。

实施裁定：沿用当前 feature checkout，保留 Task6 和个人文件；因此将来提交时仍需按正式交付列表挑选文件。Task6 测试配置补必需字段，不在生产代码使用静默容量默认值。Review 排除的真实输入、Runner、物理 KV、异常生命周期与数值/性能范围均按原任务依赖保留；这些边界意味着本轮不能宣称 P1 或完整 VL 推理已完成。

## 复现命令与证据

仓库根目录，使用现有 venv；完整 CPU 和旧模型 GPU 运行在经批准可创建本地 Socket 的宿主环境，GPU 先独立探测。模型路径使用既有默认本地目录，无需下载或修改环境：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1

.venv/bin/python -m pytest tests/test_qwen3_vl_scheduler.py -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_scheduler.py tests/test_qwen3_vl_sampling.py tests/test_original_baseline.py -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q

.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest tests/test_original_baseline.py -m gpu --require-gpu -q

.venv/bin/python -m ruff check nanovllm/engine/block_manager.py nanovllm/engine/scheduler.py tests/test_qwen3_vl_scheduler.py tests/test_qwen3_vl_sampling.py
.venv/bin/python -m ruff format --check nanovllm/engine/block_manager.py nanovllm/engine/scheduler.py tests/test_qwen3_vl_scheduler.py tests/test_qwen3_vl_sampling.py
git diff --check
```

证据保留在 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task7/`：`baseline-sandbox.log/xml`、`baseline-host.log/xml`、`red.log/xml`、`green.log/xml`、`cpu-final.log/xml`、`task-final.xml`、`quick.log/xml`、`ci-import-red.log`、`ci-import-green.log`、`fast_without_model_imports.py`、`cuda-sandbox.log`、`cuda-host.log`、`gpu-final.log/xml`、`review.md`、`source-audit.json`、`summary.json` 和包含新文件的 `task7.diff`。Task7 最终测试日志另在上级 `task-7-tests.log`。原始 RED 及环境失败日志均保留。

正式交付：两个生产文件、新调度测试、Task6 测试配置的一行补充、本报告和实施计划 Task7 状态。拟提交信息可为 `feat: isolate Qwen3-VL scheduling and KV admission`；本轮没有暂存、提交或推送。

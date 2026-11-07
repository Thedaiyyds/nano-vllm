# Task 8b：完整文本 Prefill 与真实分页 KV 绑定

2026-11-07，Task8b 验收完成。基线及当前 HEAD 为 `5912fcc7f342b3f10150fd128da1bfb33ac917f4`，分支 `feat/multimodal`。Task8a 及此前成果已提交，本轮不重复实现；只执行 Task8b，未开始 Task8c、未暂存、Commit 或 Push。个人实验文件保留。

## 实现说明

生产代码仅修改 `nanovllm/engine/model_runner.py`，测试扩展原 `tests/test_qwen3_vl_runner.py`。

- VL `warmup_model` 使用单个完整 `max_model_len` 文本请求，即使 `max_num_batched_tokens` 更小也不缩短 profile。目标配置实际完成4096-token warmup；小型真实模型另检验 max_model_len=64、batch预算32时仍执行64-token forward。
- VL `prepare_prefill` 在创建 Tensor／修改 Context 前拒绝多请求、cached prefix 和 partial Prefill。纯文本 positions 返回同值三轴 `int64[3,N]`；物理 slot 仍按真实 block table 计算，和三轴语言位置分开。原返回值 `(input_ids, positions)` 不变。
- VL 分配只遍历 `model.language_model` 中的真实 `Attention`，分配前检查数量与 text_config 层数相同；目标模型为28层。不会绑定 visual namespace 下的 Attention。
- 分配净预算为 `int(total*gpu_memory_utilization - used - peak + current) - reserve`。其中 peak/current 来自完整文本 warmup，显式安全余量只在调用方扣一次。负净预算按零可用字节传给 Task8a 纯函数，再明确拒绝不足容量；不缩小所需池。
- 使用 Task8a 的 `select_kv_block_count`，按配置推导完整上下文容量；实际4096／256为16块。显式分配 CUDA BF16 `[2,28,16,256,8,128]`，语言 Attention 的 K/V 绑定到对应层的物理存储 view。每块28 MiB，池共448 MiB；未硬编码通用16块或28层到生产计算中。

旧 Qwen3 保留原 warmup、分配与 Prefill 分支。除上述三个既有方法及新增分配 helper 外，Runner 的其余方法 AST 不变；`__init__`、`run`、`exit`、`prepare_decode` 均未改。Task8a 原有全部测试函数／类 AST 不变，其余已跟踪 nano/tests 文件和批准过的旧模型 fixture 字节不变。

## TDD 与真实数值验收

先正常 collection：28节点，包含新 `PrefillRunnerTests`。新 GPU 测试真实 RED 为 **5 FAILED、0 ERROR、0 SKIPPED，16.73s**：

- 位置形状到达 `(257,) != (3,257)` 断言。
- partial／cached／batched Prefill 未明确拒绝。
- 实际绑定层数到达 `0 != 28` 断言。
- 小模型分配读取错误的顶层 num_key_value_heads；warmup 把一维 positions 送入 MRoPE，属于 call 失败。

完整 worker 的 RED 在 warmup 阶段被拦住，尚未到达 logits/KV 比较，不将其描述成数值断言 RED。CUDA/Gloo setup 错误不算 RED。最小实现后首次新 GPU 测试 **5 PASSED，19.29s**，原始 RED 及首次成功参考均保留。

完整测试调用实际生产 `ModelRunner.__init__`，使用真实单rank NCCL、Task8a 原生模型和严格 loader，完成4096 warmup与16块分配；加载310个语言源 key，仅跳过315个 visual key。HF 完整模型只在独立参考进程运行，先退出再启动 native worker，两份完整模型不同时驻留 GPU。

复用既有 Task5 HF FA2 reference worker，参考输出写入本任务持久化忽略目录；输入 IDs 和原始位置取自持久化只读 Oracle。没有覆盖旧参考、重新采集正式 Oracle 或使用 HF generate 执行 nano 推理。

| 用例 | tokens | 物理页 | 浮点组件数 | max_abs / mean_abs / relative L2 |
| --- | ---: | --- | ---: | --- |
| text_short | 24 | [2] | 57 | 全部0 |
| text_255 | 255 | [2] | 57 | 全部0 |
| text_256 | 256 | [2] | 57 | 全部0 |
| text_257 | 257 | [2,0] | 57 | 全部0 |

每例57项为最后位置 logits 和全部28层 K/V 的首／中／末位置、所有 KV heads 抽样。另精确检查 input IDs、三轴 positions、sample indices、slot mapping 和 cu_seqlens。浮点检查 shape、dtype、有限值及零误差门槛，未只比较 top-1 或可读文本。257-token 末位置实际写入物理页0；未使用的物理页1保持 NaN。测试还检查真实 visual Attention decoy 不被绑定、语言层数量不符和安全余量不足时不分配新池、不修改已有池。

完整 Prefill 的 Attention 使用直接 Q/K/V 计算并由实际 Triton kernel 写入分页 KV，再通过物理 slot gather 对照；**本任务未验证 Decode 从分页 KV 读取**。

## 最终测试结果

| 验证 | PASSED | FAILED | ERROR | SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Task8b 首次 GPU GREEN | 5 | 0 | 0 | 0 | 0 | 19.29s |
| 完整 CPU | 183 | 0 | 0 | 0 | 20 | 9.89s |
| 快速 CPU | 31 | 0 | 0 | 0 | 172 | 1.70s |
| 最终正式 GPU 组合（恢复后） | 7 | 0 | 0 | 0 | 32 | 26.98s |

最终7项为 Task8b 的5项加既有旧 Qwen3 的短生成／固定8-token回归2项。CPU回归在中断前已通过；恢复后对照已审查 diff 和保护哈希确认生产代码、测试、fixture、Oracle 均未变化，因此复用该结果，没有把它写成恢复后重跑。Ruff 两文件 check/format 和 `git diff --check` 通过。

沙箱内完整 CPU 为 **167 PASSED、16 ERROR、20 DESELECTED**，16项全部是 Gloo 创建 Socket 的 `Operation not permitted`：13项旧 loader/schema 加已有3项模型加载测试。授权宿主执行后即为上表183通过。未 skip、Mock Gloo 或改断言隐藏环境问题。

CUDA 独立确认：沙箱 `False / 0`；宿主 `True / 1`、RTX5060Ti、torch2.9.1+cu130、CUDA13.0。GPU显式 `--require-gpu`，全部串行，不把 Gloo Socket 当作 GPU 权限问题；没有 sudo、全局 Full access 或 CUDA 环境修改。

## 中断前组合测试失败与恢复诊断

中断前组合测试曾为 **4 PASSED、3 FAILED、0 ERROR、0 SKIPPED、32 DESELECTED，573.16s**。首个失败为 HF reference worker 超过原有120秒超时，未写出参考／native输出；后续两项旧模型在 KV 容量断言处失败。此轮不计通过，完整日志和 JUnit 已另存 `gpu-combined-failed.log/xml`，原 `gpu-final.xml` 亦保留。

恢复后只读检查：无 Python GPU计算进程、空闲显存约15 GB，CUDA正常。没有当时资源快照，**不能确定历史超时的具体外部原因，也未宣称修复了生产代码缺陷**。

随后以完全相同的正式测试、120秒 worker 超时、基准、阈值和断言重测，同时每2秒只读采样 nvidia-smi 与 MemAvailable。原命令7项全部通过；采样中最高显存占用4861 MiB、最低空闲10982 MiB，系统最低 MemAvailable约4.84 GiB。采样值不是连续峰值或性能基准。失败与成功记录分别保留，没有靠增大超时、调小输入、缩小KV或修改基准取得通过。

## 实测显存与边界

最终 native run：`task8b/prefill-aloyr1kd/`。实际4096 warmup 后的分配输入记录 peak=3,718,725,120 bytes、current=3,450,222,592 bytes，显式 reserve=1 GiB。实际池16块，余量在调用方计算时保留。

| native用例 | Prefill peak allocated MiB | peak reserved MiB | 记录时设备空闲 MiB |
| --- | ---: | ---: | ---: |
| 24 | 3739.893 | 3822 | 11303.250 |
| 255 | 3754.626 | 3828 | 11297.250 |
| 256 | 3756.689 | 3828 | 11297.250 |
| 257 | 3758.707 | 3828 | 11297.250 |

这些峰值在对应 `run_model` 周围重置并读取；不是整个启动峰值、联合图文 profile 或 Benchmark。4096仅验证了真实完整文本 warmup和容量，并未做4096-token HF logits数值对照。视觉 profile 留 Task13。

## 独立 Review 与范围裁定

独立 Reviewer `/root/task8b_review` 只读审查实现、Plan/Spec、共享 Attention 和测试；独立运行 BudgetTests **6 PASSED，3.20s**，独立加载保存的 HF/native 张量核查每例59项、总236项的名称、有限值和精确相等。读取执行者GPU5通过及CPU183通过日志，没有独立重跑 GPU。

结论 Task8b 可通过，无 Critical／Important，未提出需处理的 Minor。最终组合GPU7项是执行者之后完成的门禁，不冒称 Reviewer 独立执行。恢复后源码与被审查 diff 相同，未另做代码修复或第二次Review。

裁定及成本：沿用当前feature工作区；复用参考工具但独立生成本任务持久张量；采用与 native 相同的 HF FA2、同设备、BF16、原默认 reduced-precision=True 的零误差门槛，不能推广成跨backend正确性结论。负净预算归零后仍必须满足完整容量，不缩池；余量仅扣一次。批准 Plan 的其余范围保持：队列策略沿用Task7；输入契约／图像隔离／rope_delta／DeepStack待后续；Decode分页读、EOS生成闭环、初始化/run异常收尾和幂等exit归8c/14。当前仅正常调用一次exit及worker清理，不宣称生产异常恢复已完成。

**没有使用或批准宽 BF16 阈值；只读 Oracle 的 tolerances.json、SHA256SUMS、case/checkpoint元数据哈希保持不变。** Task6 原有延期 Minor保留。本轮不宣称 P1 或完整 VL 推理链路已完成。

## 复现命令与交付

仓库根目录、既有venv，无需下载或新依赖。完整CPU需本地Socket权限；GPU先独立确认可见性。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py --collect-only -q
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py::PrefillRunnerTests -m gpu --require-gpu -q
# 最终正式组合：5项新Prefill + 2项既有旧模型生成回归
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py tests/test_original_baseline.py -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/engine/model_runner.py tests/test_qwen3_vl_runner.py
.venv/bin/python -m ruff format --check nanovllm/engine/model_runner.py tests/test_qwen3_vl_runner.py
git diff --check
```

持久化忽略证据目录 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task8b/` 包含 collection、red/green、cpu-sandbox/cpu-final、quick、gpu-combined-failed、gpu-observed 的日志／JUnit；三组独立运行目录分别保留RED、首次GREEN和最终HF/native张量、identity、comparison、显存／slot记录；另有 `source-audit.json`、`review.md`、`summary.json`、`task8b.diff`、原源码快照、CUDA探测和 `gpu-resource-observation.jsonl`。`gpu_observed.py` 仅是调用原正式命令并记录资源的诊断脚本，不是生产交付或测试替身。最终 task-done 日志在上级 `task-8b-tests.log`。

正式交付为两个Python文件、本报告和实施计划8b状态，拟提交说明 `feat: connect Qwen3-VL text prefill and paged KV allocation`。本轮未暂存／提交／推送，止于Task8b。

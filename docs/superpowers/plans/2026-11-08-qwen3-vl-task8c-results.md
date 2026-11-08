# Task 8c：纯文本 Decode 与 Runner / Engine 生命周期

2026-11-08，Task8c完成。基线和当前 HEAD 为 `90aeebae48e644875aaf6d931e54b09f166d6117`，分支 `feat/multimodal`。延续已批准 Plan 的 Task8c，未重复 Task2–8b 开发。本轮未暂存、Commit 或 Push，未开始 Task9；个人实验脚本保持原状。

## 实现说明

生产代码仅修改 `nanovllm/engine/model_runner.py` 和 `nanovllm/engine/llm_engine.py`；测试扩展既有 `tests/test_qwen3_vl_runner.py`。

- `prepare_decode(seqs)` 保留 `(input_ids, positions)` 返回接口。VL仅接受一个活动请求；positions为同值三轴 `int64[3,1]`，纯文本逻辑位置为 `len(seq)-1`。物理 `slot_mapping`、`context_lens` 和 `block_tables` 继续按真实页表计算，独立于三轴语言位置。旧 Qwen3 的一维位置分支保留。
- Runner 初始化捕获并在 `finally` 恢复调用方实际默认 dtype/device，而非固定恢复CPU；真实加载、warmup、KV分配异常均进入清理。标记自己拥有的进程组与资源，拒绝重复初始化时保留已有外部组。
- `ModelRunner.run()` 在 prepare、forward、sample 任一处报错时仍通过 `finally` 清空全局 Context，异常继续抛给调用方。
- `LLMEngine.step()` 在可恢复异常后调用既有 `Scheduler.abort()` 释放受影响请求的块；schedule 尚未返回时处理原活动/队首请求，保留其他等待请求。没有改写 Task7 调度或块管理策略。
- Runner/Engine `exit()` 均幂等。Engine取消队列、退出Runner、注销自身atexit；初始化失败时清理自身创建的worker。Runner清空Context、释放模型/KV/graph等自身引用、销毁自身进程组，再回收参数绑定 loader 形成的循环引用并归还空闲CUDA allocator缓存。

本轮未修改共享 Loader、Attention、线性层、Sampler、Scheduler、BlockManager 或模型数学实现。原 Task8a/8b 测试函数与类 AST 保持不变，其余已跟踪 nano/tests 文件、旧Qwen3 fixture及正式Oracle保护哈希一致。

## TDD、Review 与 Important 修复

collection正常：35个节点，包含新增 `EngineCleanupTests` 和 `TextDecodeRunnerTests`。初始CPU行为RED **3 FAILED，1.95s**；真实GPU RED **4 FAILED，0 ERROR，21.85s**，分别到达错误的一维 Decode positions、未恢复默认device、残留Context及完整链路被一维位置阻断的失败。完整数值worker当时尚未到达浮点比较，不将其称为数值RED。

最小实现后CPU **3 PASSED，2.62s**。首次GPU为 **3 PASSED、1 FAILED，28.96s**，失败来自测试错误假设“缺少tokenizer文件必然报错”。改为在真实Runner已初始化后显式注入tokenizer边界ValueError；没有Mock Gloo、CUDA、模型或KV。另以基线Engine初始化方法的隔离快照、当前真实Runner做诊断，确认旧路径失败后group/model仍存活；诊断未回滚工作区。修正后的新GPU四项 **4 PASSED，31.70s**。失败、诊断与成功记录均保留。

独立 Reviewer `/root/task8c_review` 对最初实现结论为可通过，无 Critical / Important / Minor；独立CPU预算及Engine清理 **9 PASSED，1.61s**，读取执行者GPU/CPU日志、四例114组件比较汇总和32步Context记录。Reviewer未独立重跑GPU，也未重新计算全部保存浮点张量。该Review对应 `reviewed-before-resource-fix.diff`。

之后执行者的最终显存自查发现一项 **Important**：仅删除模型引用并 `empty_cache()` 不能及时释放113个Embedding/Linear参数。其 `Parameter.weight_loader` 为绑定方法，形成 `Parameter → Module → Parameter` 循环；原小模型测试测量前的GC掩盖了这一点。完整模型退出后仍 allocated **3,449,950,208 bytes**，额外GC后才降至 **8,523,776 bytes**。

新增确定性回归，在exit期间关闭自动GC，**先记录全部参数和KV弱引用及显存，再做测试诊断GC**。原生产实现 **1 FAILED，23.89s**，113参数仍存活；只在生产Runner exit删除自身引用后加入 `gc.collect()`、再 `empty_cache()`，同一回归 **1 PASSED，23.06s**。未弱化断言、修改共享loader或调整阈值。随后重跑完整CPU、快速CPU及正式GPU组合，见下表。按既有流程完成一次Important修复验证，未重复派发Review；原独立Review不能冒称覆盖后加修复。

## 实际数值与生成验证

使用固定 Qwen3-VL-2B snapshot `89644892e4d85e24eaac8bacfd4f463576704203`，Transformers `2711ad22c82105e240a2877b92d082fe2857936f`，torch2.9.1+cu130 / CUDA13.0 / flash-attn2.8.3，RTX5060Ti。BF16、TP=1、`enforce_eager=True`、`TORCHDYNAMO_DISABLE=1`，两端原默认 reduced-precision reduction=True。HF完整模型与native模型在独立进程依次运行。

正式Oracle `/home/thedai/nano-vllm-artifacts/qwen3-vl/oracle-reviewed/` 只读；仅取四个有效纯文本输入。新HF FA2参考写在本任务持久化忽略目录，显式all-ones attention mask和同值三轴位置，未覆盖原Oracle。native使用实际Runner初始化、4096-token warmup、严格语言权重加载和既有16块BF16 KV池。

每例先独立Prefill，再将HF greedy得到的32个teacher IDs逐个送入真实 `prepare_decode/run_model`。**32次Decode输入不包含Prefill**，也不把Prefill得到的首token算作一次Decode。每例比较114个浮点组件：Prefill最后logits、`[32,151936]` Decode logits、全部28层每步最后token的K/V（56项）、最终KV首/中/末抽样（56项）；另精确比较实际消费的teacher IDs和KV抽样位置。检查shape、dtype、有限值及 max_abs/mean_abs/relative-L2 全零。

| 输入 | Prefill tokens | 实际Decode输入次数 | 浮点组件 | 三项误差 | 另行Engine greedy IDs |
| --- | ---: | ---: | ---: | --- | --- |
| text_short | 24 | 32 | 114 | 全部0 | 32/32一致 |
| text_255 | 255 | 32 | 114 | 全部0 | 32/32一致 |
| text_256 | 256 | 32 | 114 | 全部0 | 32/32一致 |
| text_257 | 257 | 32 | 114 | 全部0 | 32/32一致 |

物理页使用非连续 `[2,0]`，实际执行 Triton KV写入和 `flash_attn_with_kvcache` 分页读取。255-token用例前三次Decode依次为逻辑位置255/256/257、context长度256/257/258、物理slot767/0/1；页表由 `[2]` 切换 `[2,0]`。最终保存的四例各32条Context已逐条核对，三轴位置、长度、slot和页表精确正确。

另行真实 `LLMEngine.generate()` 将四个请求排队，每次只有一个活动请求，其余等待请求不占块；`temperature=0,max_tokens=32,ignore_eos=True`，四例全部32个生成Token ID与HF一致。完成后队列为空、所有ref_count=0、used blocks/hash map为空。再独立运行 `max_tokens=1`，恰好生成一个正确ID。这里32-token固定长度验证明确忽略EOS；沿用Task6双EOS单元测试，不声称本轮目标checkpoint触发自然EOS停止。

真实CUDA小模型还验证load、warmup、allocate和Engine tokenizer初始化异常，恢复调用方FP64/CPU默认状态，释放已建模型/池/进程组；既有外部Gloo组不被误销毁。prepare、forward、sample异常保留原ValueError并清空Context。CPU使用真实Scheduler/BlockManager，仅以局部边界对象注入Runner异常；真实GPU数值链路未替换模型或内核。

## 最终测试与环境

以下均为Important修复后的结果。CPU两项在中断前完成；恢复后核对保存diff/源码、受保护文件哈希和JUnit，复用其真实结果。最终GPU组合在恢复后执行。

| 验证 | PASSED | FAILED | ERROR | SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 资源释放定向GREEN | 1 | 0 | 0 | 0 | 0 | 23.06s |
| 完整CPU | 186 | 0 | 0 | 0 | 24 | 10.04s |
| 快速CPU | 31 | 0 | 0 | 0 | 179 | 1.76s |
| 最终正式GPU组合 | 11 | 0 | 0 | 0 | 35 | 56.44s |

11项包含旧Task8b Prefill五项、新Decode/生命周期四项和既有旧Qwen3生成回归两项。快速CPU是完整CPU的子集，定向GREEN也是GPU组合的子集，不将它们相加冒称独立覆盖。Ruff三个Python文件check/format及 `git diff --check` 通过。

完整CPU需要Gloo创建本地Socket；此前已确认沙箱 `Operation not permitted` 属环境权限错误，本轮完整CPU使用针对测试命令批准的宿主执行，不将其当作测试断言失败。没有skip、Mock Gloo或修改断言掩盖权限问题。CUDA单独探测：沙箱False/0，宿主True/1、RTX5060Ti；GPU显式 `--require-gpu`，全部串行，无sudo、全局Full access或CUDA环境修改。

最终native run `task8c/decode-uwgty4q0/`：退出前allocated **3,919,988,736 bytes**；退出后所有模型参数弱引用失效、KV根Tensor失效、NCCL组不存在、第二次exit无异常。测试额外GC前/后allocated均 **8,523,776 bytes**，设备空闲 **15,442,509,824 bytes**。剩余少量worker Tensor/后端分配不称为“CUDA上下文零占用”。该进程记录的累计peak allocated **3,940,984,832 bytes**、peak reserved **4,013,948,928 bytes**；native peak从既有warmup重置后累计，HF peak从其进程启动累计，均不是单独Decode的峰值，也不是公平性能基准。未做性能比较或优化。

## 验证边界与交付

P1的有效纯文本、固定BF16/FA2、TP1/eager范围内 Prefill、分页Decode、greedy生成和生命周期回归通过。没有使用或批准宽BF16自动放行阈值；`tolerances.json`、`SHA256SUMS`、case/checkpoint元数据保持原哈希。

未接通图像、Processor输入契约、rope_delta、视觉Encoder、DeepStack或图文融合；尚未验证单图端到端生成、4096-token HF数值对照、更长Decode、多请求并发、TP>1、CUDA Graph、其他backend或性能。进程级致命CUDA错误可能使上下文不可再用，仍要求重启进程；本任务可恢复异常测试不能证明其恢复能力。Task6已有延期Minor保持原裁定，无新增未处理阻塞项。

正式交付为两个生产Python文件、一个测试文件、本报告和实施计划状态。具体代码diff、原始失败日志、JUnit及大型HF/native参考只保留在持久化忽略证据目录，不加入Git。未暂存/Commit/Push，不开始Task9。

## 复现命令

仓库根目录，使用既有venv和本地checkpoint，无新依赖或下载。完整CPU需允许本地Socket；GPU先独立确认CUDA可见性。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py --collect-only -q
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py::TextDecodeRunnerTests -m gpu --require-gpu -q
# 正式GPU组合：8b五项 + 8c四项 + 旧Qwen3两项
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py tests/test_original_baseline.py -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/engine/model_runner.py nanovllm/engine/llm_engine.py tests/test_qwen3_vl_runner.py
.venv/bin/python -m ruff format --check nanovllm/engine/model_runner.py nanovllm/engine/llm_engine.py tests/test_qwen3_vl_runner.py
git diff --check
```

证据根目录 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task8c/`：最终 `gpu-after-resource.log/xml`、`cpu-after-resource.log/xml`、`quick-after-resource.log/xml`；`resource-red/green.log/xml` 及对应 `decode-kl0ju8g2/`、`decode-2jn_2s2e/`；原始 `cpu-red`、`gpu-red`、`gpu-green`（夹具失败）、`gpu-fixed-fixture` 和tokenizer诊断；最终 `decode-uwgty4q0/` 的HF/native张量、identity/comparison/Context/资源记录；`lifecycle-ww37654a/` 的注入故障记录；另含 `source-audit.json`、`summary.json`、`review.md`、`task8c.diff` 及初始源码快照。task-done命令和执行状态保存在上级 `progress.md`、`task-8c-tests.log`，commit范围仍为 `90aeeba..90aeeba`，不代表创建了新提交。

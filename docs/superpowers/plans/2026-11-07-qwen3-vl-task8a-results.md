# Task 8a：模型选择、dtype 与 KV 预算纯函数

2026-11-07。代码实现与 Task8a 验收完成；**唯一 Important 环境门槛已按用户明确授权更新一行指纹，并经正式 GPU 回归通过后关闭。** 分支 `feat/multimodal`，HEAD `89c1e40e9aa52867ad107954496b3d342f24170d`。保留已完成且未提交的 Task6/7 和个人实验文件；未暂存、Commit、Push 或开始 Task8b。

## 实现

生产代码只修改 `nanovllm/engine/model_runner.py`，新增 `tests/test_qwen3_vl_runner.py`。

- `kv_block_bytes(text_config, block_size, dtype)`：计算全模型、TP=1 的语言 KV 每块字节数，包含 K/V、语言层数、KV heads、head_dim、block slots 和 dtype.itemsize。优先显式 head_dim；缺失或 None 时从 hidden_size / attention heads 推导。拒绝非正或非整数的块及 KV 维度。
- `select_kv_block_count(config, available_bytes)`：从 `ceil(max_model_len / block_size)` 得到完整上下文所需块数。消费调用方已扣除非 KV 常驻、峰值增量和安全余量的净预算，不再减 `kv_cache_reserve_bytes`。预算低于完整容量就 `ValueError`，更大的预算仍返回容量上限；不静默缩小池、不修改 Config 或分配 tensor。
- 提取 `ModelRunner._load_model()` 并接入现有初始化：依据固定 Config 的 `hf_config.model_type`，VL 构造原生 `Qwen3VLForConditionalGeneration(text_only=True)`，调用严格的 `load_qwen3_vl_weights(text_only=True)` 并保留 `weight_load_report`；旧 Qwen3 保持原模型和 `load_model`。
- 构造及加载时使用规范化的 `config.model_dtype`，解决 VL 顶层 dtype 为 None、实际 BF16 位于 text_config 的问题。helper 用 finally 恢复自身临时 dtype；后续初始化使用同一规范化 dtype。

预算函数尚未接入真实 KV 分配和 profile，归 Task8b；完整 Prefill 的三轴 positions、语言 Attention 绑定、实际分页 KV 同样未提前实现。Decode、整个初始化失败收尾、run finally 和幂等 exit 归 Task8c。helper 恢复自己的 dtype 不等于整个 Runner 已具备失败恢复。

已有 `allocate_kv_cache`、warmup、prepare_prefill/decode、run、exit 等函数 AST 与 Task8a 开始时快照一致；Ruff 仅格式化其原有写法。Task6/7 的七个受保护源文件 SHA256 均未变。未改模型、loader、调度器或测试框架。

## TDD 与真实 CPU 结果

先正常收集 20 项预算测试，节点包含 BudgetTests；可导入接口用 return-0 占位，接口缺失不算 RED。行为 RED 为 **20 FAILED、0 ERROR**，其中批准命令的 BudgetTests 为 **6 FAILED**：实际到达 `0 != 29360128`、`0 != 16`、缺失 ValueError 等断言。实现后 **20 PASSED**，BudgetTests **6 PASSED**。

模型加载用例复用原有真实 Gloo fixture，仅对三项加载测试启用；BudgetTests 不依赖 Gloo。用真实小型 HF text model 生成临时权重，再由实际原生构造器和 loader 加载，未用模型/loader/Gloo Mock。原 Qwen3-only scaffold 在宿主得到 **3 项 call FAILED**：旧分支 dtype 为 FP32、VL 顶层缺 vocab_size；这不是 setup 失败。分支及 dtype 实现后，本任务 **23 PASSED**。

检查包括 BF16 参数、tied embedding/head 同一 Parameter、Q projection 与 embedding 逐值映射、visual namespace 显式跳过及报告、缺失 q_proj 拒绝、成功/失败时 helper dtype 恢复。

| 最终运行 | PASSED | FAILED | ERROR | SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Task8a CPU（含真实 Gloo 加载） | 23 | 0 | 0 | 0 | 0 | 1.75s |
| 更新正式 fixture 后完整 CPU | 183 | 0 | 0 | 0 | 15 | 10.42s |
| 快速 CPU | 31 | 0 | 0 | 0 | 167 | 1.64s |
| 更新正式 fixture 后原 Qwen3 正式 GPU 回归 | 2 | 0 | 0 | 0 | 9 | 12.27s |

开发前完整 CPU 沙箱内 **147 PASSED、13 ERROR、15 DESELECTED**；13 项全部是 ProcessGroupGloo 创建 TCP Socket 的 `Operation not permitted`。授权沙箱外基线 **160 PASSED、0 ERROR**。模型加载 RED 的沙箱 3 项 setup ERROR 也单独保留，真正 RED 使用宿主结果；没有把环境错误当作功能 RED。

预算数值来自实际固定 checkpoint 配置元数据，不加载完整权重：

| 配置/净预算 | 实际结果 |
| --- | --- |
| 28 层、8 KV heads、head_dim 128、block 256、BF16 | 29,360,128 bytes/block（28 MiB） |
| L=4096，469,762,048 bytes | 16 块（448 MiB） |
| L=4096，469,762,047 bytes | ValueError，少 1 byte 也拒绝 |
| L=4096，十倍净预算 | 仍为 16 块 |
| L=257，block 256，58,720,256 bytes | 2 块 |
| L=4096，block 512，同样 448 MiB | 8 块 |

另覆盖 255/256/257、1024 的向上取整，不同显式 head_dim、BF16/FP32 itemsize、head_dim fallback、非法维度和非法预算类型、Config/default dtype/default device/CUDA 初始化状态不变。16 仅对应 4096/256，未硬编码。

## 原始 GPU 环境阻塞与用户授权后的解决

CUDA 单独探测：沙箱 `False / 0`，授权宿主 `True / 1`，RTX 5060 Ti、torch 2.9.1+cu130、CUDA 13.0。更新指纹前运行既有真实 GPU 两项回归，短生成 smoke 通过；`test_qwen3_fixed_input_regression` 在环境元数据比较时 FAILED。不是 CUDA 不可用，也不是 Gloo 错误。

唯一差异是 `environment.packages.transformers.code_dirty_sha256`：

```text
更新前正式 fixture：ed1bdf64190488630314c96aac9154ab2fc059979cb71731dcfac6122184aa14
当前工作区： ac1ed6c4cc6c3b461fce22df7132bb1d0af4a4345ecb3135d1bb76512d334542
```

Transformers HEAD 仍为固定的 `2711ad22c82105e240a2877b92d082fe2857936f`；全部三份 dirty Python 文件去除 docstring 后 AST 与固定提交相同，差异属于注释/docstring。当前 8 个 token 与正式 fixture 完全一致：`[576,3405,374,25,304,279,4771,11]`。

为排除 Runner 改动造成真实回归，另获准使用归档的原 nano-vLLM `bb823b3e06983d71485a8e1f23715ebd87d98ef8` 在当前环境运行两次，诊断结果保存到忽略目录。两次原始实现相互一致，并且与当前实现的完整环境元数据及生成 token 一致。调用现有 `assert_legacy_match(..., require_reference=True)` 比较通过。

上述诊断未被当作正式门槛通过。更新前的正式 GPU 结果为 **1 PASSED、1 FAILED**，保留在原始 log/xml 中；没有修改预期 token、断言、marker、Gloo fixture 或 CUDA 环境来放行失败。

用户随后明确批准应用候选中的一行 Transformers 环境指纹更新，并要求重跑正式旧 Qwen3 GPU 回归。执行前重新核对：当前 HF diff 指纹和三份源码 SHA 与已有 AST 证据一致；候选与正式 fixture 的原始字节只相差这一行；token 和其余 JSON 字段完全相同。旧正式基准另存为 `legacy_qwen3_fixed.before-approved-update.json`，原失败日志、GPU 输出及诊断均保留。

已将该一行更新写入正式 `tests/fixtures/legacy_qwen3_fixed.json`。随后独立确认宿主 CUDA 可见 RTX5060Ti，重新运行同一正式 GPU 命令：**2 PASSED、0 FAILED、0 ERROR、0 SKIPPED、9 DESELECTED，12.27s，退出码0**。正式比较仍执行原有全部元数据和 token 断言。接着完整 CPU 复跑 **183 PASSED、0 FAILED、0 ERROR、0 SKIPPED、15 DESELECTED，10.42s**，其中包含本任务全部23项；预算类最终复验6项也通过。

更新前后共有52份受保护源码、测试和旧证据哈希逐项相同；没有修改生产代码、测试断言、预期 token、数值阈值或只读 HF Oracle。`approved-fixture-update.diff` 证明正式基准仅1行删除/1行增加，`approved-update-protected-sha256.json` 保存保护清单。

## 独立 Review

Reviewer `/root/task8a_review` 审查实际源代码、Task8a-only diff、Plan/Spec 和日志，独立执行 **20 PASSED、3 DESELECTED，1.58s**。未独立执行 Gloo 加载或 GPU。

原始 Review 结论为 **With fixes — validation resolution only**：无 Critical、无生产代码 Important、无新增 Minor；1项 Important 是正式旧模型固定回归的环境门槛。该原始结论及失败记录保留，不改写为一次无问题的 Review。用户明确批准一行指纹更新后，执行者以正式 GPU 两项全通过和完整 CPU 回归关闭该 Important；目前无未解决的阻塞问题。未进行第二次 Review，也不将执行者的 GPU 复跑说成 Reviewer 独立验证。

接受其范围划分：实际分配/profile/绑定/Prefill 留8b；Decode、全初始化/run/exit 恢复留8c；输入验证、图像/rope_delta/DeepStack 生命周期及集成留后续；既有 Task6/7 不重做。宽 BF16 阈值未使用、未批准，持久化只读 HF Oracle 未改。Task6 原延期 Minor 保留。

实施裁定：复用 feature 工作区并保留此前未提交交付，后续提交须按列表挑选；提取模型加载 helper 以在 CPU 上验证真实构造和加载，不模拟完整 CUDA Runner；纯预算函数不提前接入8b分配。因此本轮不宣称完整 VL Runner、真实 KV、端到端数值或性能已通过。

## 复现与交付

仓库根目录使用现有 venv。前三条预算相关命令不需要 Gloo；完整本任务及全 CPU 需要允许本地 Socket 的环境：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py::BudgetTests -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py -k 'not ModelSelectionTests' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q

.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
# 用户批准更新一行环境指纹后的正式 fixture：2 passed、0 failed。
.venv/bin/python -m pytest tests/test_original_baseline.py -m gpu --require-gpu -q

.venv/bin/python -m ruff check nanovllm/engine/model_runner.py tests/test_qwen3_vl_runner.py
.venv/bin/python -m ruff format --check nanovllm/engine/model_runner.py tests/test_qwen3_vl_runner.py
git diff --check
```

持久证据目录 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task8a/`：基线沙箱/宿主 log+xml，collection、预算 RED/GREEN、routing RED 沙箱/宿主、task-green、cpu-final、quick、gpu-environment-failed、cuda 探测、`budget-report.json`、`legacy-environment-diff.json`、`hf-ast-audit.json`、原实现两次诊断及 comparison、候选 fixture/diff、`source-audit.json`、`review.md`、`summary.json`、含新测试及批准基准更新的 `task8a.diff`。最终补充 `gpu-approved.log/xml`、`cpu-approved.log/xml`、`budget-approved.xml`、`cuda-approved-host.log`、正式基准旧副本、授权diff与保护哈希清单；原先失败和基准诊断文件未覆盖。Ruff 两文件及 `git diff --check` 通过；远端 CI 未运行，未安装新依赖。

本轮正式交付为两个 Python 文件、用户明确批准的 legacy fixture 一行环境指纹更新、本报告和计划状态。拟提交信息 `feat: select Qwen3-VL runner model and calculate KV budgets` 仅供审阅；未自动提交。

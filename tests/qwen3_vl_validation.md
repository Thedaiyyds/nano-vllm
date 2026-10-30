    # Task 1b：旧功能轻量兼容检查结果

日期：2026-10-29。范围仅Task 1b；Task 1c未启动。Native执行，未安装/升级依赖、未下载权重、未修改CUDA/驱动，未暂存、Commit或Push。HEAD保持`bb823b3e06983d71485a8e1f23715ebd87d98ef8`。

## 最终状态与实际结果

| 项目 | 状态 | 结果 |
| --- | --- | --- |
| Task 1b CPU | PASSED | 9 passed，2 deselected，1.51s |
| Qwen3-0.6B真实GPU Smoke | PASSED | 与固定输入回归合并执行的两项之一 |
| 原提交固定输入回归 | PASSED | 原提交两次独立GPU采样一致；当前工作区8-token输出精确一致 |
| Task 1b GPU合并运行 | PASSED | 2 passed，9 deselected，7.86s |
| 全部已建立CPU测试 | PASSED | 26 passed，2 deselected，6.68s |
| 快速CPU | PASSED | 19 passed，9 deselected，1.54s |
| Ruff/格式/git diff检查 | PASSED | 无错误 |
| SKIPPED | 无 | 没有以skip/xfail替代所需验证；deselected是命令主动筛选，不能算测试通过 |
| 默认沙箱GPU入口 | BLOCKED（历史运行条件） | Smoke和固定输入测试均在setup报告CUDA不可用并exit1；不是模型失败 |
| 宿主GPU入口 | PASSED（最终条件） | 只读权限探测后确认真实GPU可见，同一命令在宿主GPU权限下成功 |

宿主实际硬件：RTX 5060 Ti，16311MiB，compute capability12.0；驱动595.99.02，torch2.9.1+cu130。工具默认沙箱看不到GPU，但宿主可访问，故没有改动CUDA环境或让Mock顶替。门禁中的CPU自测double属于Task 1a，不构成这里的GPU结果。

## 每个核心测试在检查什么

| 测试名称 | 通俗解释、必要性与边界 |
| --- | --- |
| `TestLegacyCPU::test_public_exports_and_qwen3_config` | 检查旧的LLM/SamplingParams入口还能用，Qwen3配置仍能读取，并把请求长度压到模型支持的范围；也拒绝不符合分页要求的块大小。防止为VL增加嵌套配置时误伤旧入口。只用临时小配置，不加载0.6B权重。 |
| `test_prefix_hit_miss_and_reference_release` | 两个请求共享同一段前缀时复用相同块；先结束其中一个不能把另一个还在用的块释放，全部结束后内存才全部归还。不相同的前缀不能命中。防止缓存串内容或提前回收。 |
| `test_chunked_prefill_does_not_generate_early` | 300-token输入分成128、128、44三段；前两段不得提前生成回答，读完才生成，达到输出数量后释放块。保护原有分块Prefill语义。 |
| `test_memory_pressure_preempts_and_requeues` | 用只有两块的小池模拟内存紧张：调度器应回收较后请求的缓存，把它排回等待队列；较前请求完成后，后者重新读入包括已生成token在内的内容并完成。没有用Mock替代Scheduler/BlockManager。 |
| `test_positive_temperature_sampling_is_seeded_and_not_greedy` | 相同seed应复现同一采样结果；相同分数的一组输入应采到不同token，防止正温度路径被错误改成永远argmax。极端分数时也应采到明确高分项。只测试小CPU张量。 |
| `test_worker_propagates_failure_and_output` | 子进程失败的exit7和输出必须传回主测试；不能把运行失败吞掉，或丢掉诊断证据。是测试助手，不是模型正确性测试。 |
| `test_fixed_regression_rejects_metadata_drift[...]`（3项） | 人为提供变过的config指纹、采样参数、原提交标识，比较器必须拒绝。只检查验证逻辑；不把这些人工记录当GPU生成结果。 |
| `test_qwen3_short_generation_smoke` | 真正加载本地0.6B权重，在真实单卡上对一条短输入生成最多8个token，验证输出结构、合法token范围和正常退出；不是完整模型数值或性能基准。 |
| `test_qwen3_fixed_input_regression` | 对同一固定输入、seed和环境，当前实现必须与原提交保存的8-token结果逐个相等；防止共享组件改动悄悄改变旧行为。仅一条短样例，不保存全层logits/KV矩阵。 |

## 固定输入与基线来源

Prompt：`Hello, nano-vLLM.`；精确输入IDs：`[9707,11,50493,8273,4086,44,13]`。

seed=2026、temperature=0.7、max_tokens=8、ignore_eos=False、BF16、TP1、batch1、关闭torch.compile/CUDA Graph；warmup/缓存配置记录在fixture中。原模型不支持greedy，本任务没有为采基线改Sampler。

从固定原提交`bb823b3e06983d71485a8e1f23715ebd87d98ef8`执行`git archive ... nanovllm`到临时目录，不切换工作区HEAD；worker在首次导入前选择原目录，并检查实际模块路径。两个独立进程生成相同输出后才写fixture；默认拒绝覆盖已有fixture。

Expected IDs：`[576,3405,374,25,304,279,4771,11]`。基线在[legacy_qwen3_fixed.json](fixtures/legacy_qwen3_fixed.json)，包含原提交、权重SHA256、config/tokenizer/template SHA256、采样设置、GPU/驱动/包版本，以及editable Transformers的git revision/dirty diff指纹。元数据不一致明确拒绝，不自动重采或放宽为“文字相似”。

Review后在新临时路径再执行原提交两次短生成，以补全指纹；替换fixture前强制检查原提交、权重SHA、输入、seed、运行选项和8个expected token与旧fixture全部相同。只是补全元数据，没有刷新输出期望。

## TDD与真实失败记录

- 5项旧组件测试第一次即通过，属于已有正确行为的兼容保护；不人为制造All Red。
- 子进程助手开始的NotImplementedError仅是过渡，不当作充分Red证据。随后真实子进程返回exit7，但`result.stdout is not None`失败（stdout=None）；补capture_output/text后通过。
- 真实GPU smoke起初可生成；新增模型词表一致性断言发现测试助手记录的是tokenizer基础词表151643，而模型输出空间是151936。先实际观察`151643 != 151936`，再改为模型config词表，GPU重跑通过。没有修改模型数值算法。
- Review指纹问题：三个元数据负例先失败（错误指纹/采样/来源被接受），再补采集/比较/来源门禁并通过。最终测试只检验有效行为，未破坏生产代码制造失败。
- 固定输入模型回归第一次在宿主真实GPU上就通过；这与旧组件首次通过一样，不算“新增计算实现先红后绿”。不宣称每个新测试助手/普通元数据采集函数都有独立Red记录。

日志目录：`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task1b/`。关键文件：`initial.log`、`worker-red.log`、`vocab-red.log`、`review-red.log`、`cpu-final.log`、`full-cpu-final.log`、`gpu-reviewed-host.log`、`quick-final.log`。该自忽略目录只存本机证据，不提交大文件。

## 可直接复制的复现命令

在有GPU权限的SSH宿主终端、仓库根目录运行；Codex默认沙箱中CUDA可能不可见，需同样的宿主GPU访问权限。这里不要求sudo，不安装任何依赖。

```bash
cd /home/thedai/nano-vllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_MODEL=/home/thedai/huggingface/Qwen3-0.6B

# 核实真实设备；不要在False时将GPU测试改成skip
nvidia-smi
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'

# Task 1b CPU：实际9 passed，2 deselected
.venv/bin/python -m pytest tests/test_original_baseline.py -m 'not gpu' -q

# Task 1b两项真实GPU：实际2 passed，9 deselected
.venv/bin/python -m pytest tests/test_original_baseline.py -m gpu --require-gpu -q --junitxml=/tmp/qwen-vl-legacy-gpu.xml

# 也可以单独选择Smoke或固定输入回归
.venv/bin/python -m pytest tests/test_original_baseline.py -k test_qwen3_short_generation_smoke -m gpu --require-gpu -q
.venv/bin/python -m pytest tests/test_original_baseline.py -k test_qwen3_fixed_input_regression -m gpu --require-gpu -q

# 全部已建立CPU测试：实际26 passed，2 deselected
.venv/bin/python -m pytest tests -m 'not gpu' -q
# 快速CPU：实际19 passed，9 deselected
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check tests nanovllm/__init__.py
.venv/bin/python -m ruff format --check tests nanovllm/__init__.py
```

若需审查基线采集流程，显式指定一个**不存在的新路径**运行，不覆盖已审阅fixture：

```bash
.venv/bin/python -m tests.support --capture-legacy-baseline "$QWEN3_MODEL" /tmp/qwen3-legacy-review-new.json
```

缺权重/无GPU/元数据漂移均为明确阻塞或非零失败，不会自动下载、生成expected或跳过。GPU运行没有遇到最终阻塞；早期沙箱BLOCKED记录保留作为运行条件证据。

## 独立Code Review与裁定

独立Reviewer实际审查新增diff并复跑CPU：无Critical，1项Important（指纹不完整）、2项Minor（来源revision未校验、缺本报告）。重要指纹问题已按RED→GREEN修复；来源revision问题重评为影响原基线来源可信度的重要门禁，同次修复；本报告是原计划应交付文件，已补齐。

修复后由主代理重新执行9项CPU、2项真实GPU及完整CPU套件，均通过；未要求Reviewer二次复审，也不将其初次“暂不可验收”改写成“初次无问题”。无延期Minor。

实施裁定：复用当前feature工作区，保留Task1a及用户实验；宿主GPU权限用于真实执行，不更改环境。`single_rank`助手因本任务CPU组件不需要分布式初始化而延期到首个实际使用任务，避免引入无用进程组；CPU没有使用分布式Mock。

Task1b技术工作完成，停止等待验收；未开始Task1c，未自动Commit或Push。

## Task 1c（单独交付）

Qwen3-VL-2B参考数据、真实Kernel、阈值依据及复现说明见[Task 1c报告](qwen3_vl_reference_results.md)。旧Qwen3-0.6B回归范围未扩大。

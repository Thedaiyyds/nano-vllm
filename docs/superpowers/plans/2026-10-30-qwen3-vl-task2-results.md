# Task 2：VL 配置规范化与预算字段

2026-10-30，仅实施已批准Plan的Task 2。起点与当前HEAD均为`f331e43406de157049028e716bca61926a38378e`，分支`feat/multimodal`。开始时跟踪文件无改动；保留用户所有未跟踪实验、图片和视频。未安装依赖、未修改参考数据、未暂存/Commit/Push，未开始Task 3。

## 实现说明

`nanovllm/config.py`保留完整`hf_config`，按`model_type`区分Qwen3与Qwen3-VL，新增派生字段`text_config`、`model_dtype`、`is_multimodal`及不可变`eos_token_ids`。这些派生字段不接受构造覆盖。读取本地GenerationConfig中的EOS，缺该文件则从模型配置解析；未改动旧`eos`标量及其tokenizer赋值路径。

固定VL checkpoint的顶层dtype为None，文字dtype为BF16；解析后hidden_size=2048、theta=5000000、mrope_section=[24,20,20]，EOS集合为{151645,151643}。不改变torch全局默认dtype。

VL明确要求TP=1、max_num_seqs=1、enforce_eager=True、BF16；上下文配置最多4096、视觉token最多256，允许降低。上下文、每批token、视觉token及KV预留预算必须为正；错误包含字段及值。最终上下文长度再与文字checkpoint能力取min。未知/不支持的model_type拒绝。Qwen3保留原TP/eager/序列数量规则，不应用VL视觉及预留预算限制。

`kv_cache_reserve_bytes=1073741824`是初始工程参数；本任务没有测量实际显存安全余量。单活动请求调度、预算实际消费、模型路由、EOS停止逻辑均由后续任务接通，不能把配置通过表述为VL推理已实现。

## TDD与测试结果

新增模块先正常收集21项，再执行原实现：21 failed。主要真实生产行为失败为VL解析错误访问顶层`max_position_embeddings`（AttributeError），以及不支持的llama配置未抛ValueError；新增字段相关项另有TypeError。这些不是ImportError或setup失败。嵌套数值断言当时尚未到达，不宣称观察到了hidden_size/dtype数值不匹配。最小实现后21项通过。

本地检查另发现测试顶层导入torch会破坏仅安装pytest的快速CPU环境；在`tests/test_package_imports.py`增加阻断torch/transformers导入的行为回归，先观察子进程被顶层torch导入阻断、测试1 failed，再将模型栈导入移至测试体，变为1 passed。没有修改CI依赖或扩大CI承诺。

| 验证 | 本次真实结果 |
| --- | --- |
| 开发前全部CPU基线 | 54 passed，6 deselected，7.39s |
| Task2配置模块 | 21 passed |
| 修正后的全部CPU | 76 passed，6 deselected，7.48s |
| 快速CPU | 20 passed，62 deselected，1.65s |
| 独立Reviewer定向CPU复跑 | 34 passed，2 deselected |
| 三个改动Python文件Ruff check/format | 全部通过 |
| git diff --check | 通过 |
| GPU | NOT RUN；Task2为CPU配置任务，没有新增GPU数值结论 |

完整CPU包含Task1a门禁、Task1b旧组件兼容及Task1c参考工具行为回归；未重新采集Oracle，未重复完成Task1。没有以skip/xfail代替验收；deselected项不计入通过数量。远端GitHub CI本次未运行。

## 独立Review与范围裁定

使用requesting-code-review工作流，由未参与实现的独立代理只读检查完整生产diff、未跟踪新测试及导入回归，并独立执行CPU测试。结论：Task2可通过，无Critical、Important或必须修复的Minor。Reviewer另构造真实本地Qwen3-0.6B配置，确认dtype、配置身份与旧eos标量保持；不等同于本次跑过旧模型GPU生成。

Reviewer明确暂不判定Plan Review Focus中的输入/grid、调度准入、跨请求cache/delta/DeepStack隔离、资源生命周期、实际greedy/EOS停止/分页，以及GPU数值与显存；这些分别属于Task6–14，本轮未改其链路。裁定接受上述边界，后续任务按各自门槛验收。本报告只完成Task2，不表示M1里程碑整体通过。

沿用已批准Native工作流：当前feature分支目标文件与用户实验无冲突，原地实施；不创建worktree、不移动用户文件。测试导入兼容修正增加一个现有测试文件，属于Task2必要回归，不新增生产范围。

## 持久化参考与复现命令

参考只读目录仍为`/home/thedai/nano-vllm-artifacts/qwen3-vl/oracle-reviewed`，本次确认目录存在且`tolerances.json` SHA256为`0a0029401ec73945930072300237f8dd45b7c14730c038619655a1de723600d9`，与恢复报告一致。未在本次重复全量Oracle审核。宽BF16包络、factor2及one-ULP提案仍未批准用于正确性自动放行；Task2没有加载这些浮点阈值。

在仓库根目录、现有venv执行，无需下载或安装：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_VL_ARTIFACTS="$HOME/nano-vllm-artifacts/qwen3-vl"
export QWEN3_VL_ORACLE="$QWEN3_VL_ARTIFACTS/oracle-reviewed"
export QWEN3_VL_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"

.venv/bin/python -m pytest tests/test_qwen3_vl_config.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_config.py -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/config.py tests/test_qwen3_vl_config.py tests/test_package_imports.py
.venv/bin/python -m ruff format --check nanovllm/config.py tests/test_qwen3_vl_config.py tests/test_package_imports.py
git diff --check
```

本机持久化日志/JUnit位于`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task2/`，包括`collection.log`、`red.log`、`green.log/xml`、`collection-red.log`、`collection-green.log`、`cpu.log/xml`、`quick.log/xml`及含新文件的`task2.diff`。该目录自忽略，不提交大产物。

最终Python文件SHA256（将报告与候选代码对应）：

```text
4d0d63fdc99e832084c4743c266879409bd715b7432b3ef99bdf740ccb25ef7d  nanovllm/config.py
cbbd37866498030b1ce7c52d418d773f93dda5f9990364b596250773aaa05c08  tests/test_qwen3_vl_config.py
1135693d78978f8830da72d8fbd6a57addeb1039ced3ccee3394d10e751d63a9  tests/test_package_imports.py
```

交付文件清单：上述三个Python文件、本Task2报告及实施计划的当前状态/Task2勾选更新。拟提交信息：`feat: normalize Qwen3-VL configuration and resource limits`。技术验证及独立Review通过；工作区保留待审阅，未获Commit批准，不暂存或提交。

# Task 3：三轴位置与 Interleaved-MRoPE

2026-10-30，Native执行，仅实施已批准Plan的Task3。分支`feat/multimodal`，HEAD仍为`f331e43406de157049028e716bca61926a38378e`。保留全部Task2未提交改动与用户实验文件；未安装依赖、未改HF源码/原一维RotaryEmbedding/Oracle，未暂存、Commit或Push，未开始Task4。

## 实现说明与边界

新增`nanovllm/layers/mrope.py`两个接口：

- `build_mrope_positions(...)`：按文字和图片段累计三轴位置，输出CPU int64 `[3,L]`及Python整数delta。图片跨度由每行grid和merge size计算；支持多行grid及相邻图片的数学分段，不硬编码只准一图。检查负/零维度、merge整除、grid不足/剩余、图片跨度和模态长度不匹配；只接受text=0/image=1。
- `Qwen3VLRotaryEmbedding(text_config)`：处理`positions[3,N]`及`Q/K[N,heads,128]`。FP32计算频率和sin/cos，再用H的`1:60:3`、W的`2:60:3`替换T频率位置，其余保留T；复制到128维、转为Q dtype，再按HF文字公式旋转。保留BF16乘法及加法的舍入次序，支持不同Q/K头数和非连续Tensor，不修改输入。

手算用例：文字2 token、合并后2×3图片、后续文字3 token，总长度L=11，三轴最大位置7，所以delta=8−11=−3。首个Decode位置为`[8,8,8]`，后续31步为9至39。物理token长度仍为11；本任务不实现KV更新或Runner调用。

保留原一维RotaryEmbedding。入口的单图限制、占位符语法及具体token合法性归Task9；本次数学函数能处理多行grid不代表引擎支持多图。没有实现视频、调度、请求生命周期或模型推理接线。

## TDD与真实测试

先写测试和可导入的最小接口stub：位置暂为普通递增序列、旋转暂为identity；未用ImportError或NotImplementedError冒充RED。正常收集26项后，CPU实跑为**20 failed、4 passed、2 deselected**。失败包含delta实际0而期望−3、交错旋转不相等、相邻图片T/H位置实际`[0,1,2]`而期望`[0,0,2]`、错误grid未拒绝；4个纯文本Oracle用例在stub下即通过，不算新实现先红后绿。随后替换stub为完整最小实现，CPU全部通过。

| 验证 | 实际结果 |
| --- | --- |
| 开发前全部CPU基线 | 76 passed，6 deselected，7.64s |
| Task3 CPU模块 | 24 passed，2 deselected，2.86s |
| 独立Reviewer CPU复跑 | 24 passed，2 deselected，2.57s |
| 10个持久Oracle用例 | positions/delta全部精确相等，包含在上述CPU结果中 |
| 最终宿主GPU | 2 passed，1.77s，无failure/error/skip |
| 全部CPU回归 | 100 passed，8 deselected（最终日志见cpu-final.log） |
| 快速CPU | 20 passed，88 deselected，1.63s |
| Ruff check/format、git diff --check | 通过 |
| 测试模块在阻断torch/transformers导入时加载 | 通过，保持轻量CI收集 |

首次默认沙箱GPU运行在setup明确报告CUDA不可见，**2 errors、exit1**，未以skip计通过。获得宿主GPU执行权限后同一GPU测试命令通过；补齐JUnit设备/显存元数据后再次通过。CPU与GPU过滤产生的deselected不计作已运行通过。远端GitHub CI未运行。

GPU实际环境为NVIDIA GeForce RTX5060Ti，torch2.9.1+cu130、CUDA13.0；关闭Dynamo编译，PyTorch eager与同设备HF组件对照。BF16测试使用16Q头、8KV头、head_dim128，stride切片产生非连续Q/K，覆盖N=1/11/255/256/257；另验证图文delta对应的32步Decode旋转。每个输出均检查finite及`torch.equal`，输入Q/K保持不变。所有被比较元素相等，因此对应max_abs/mean_abs/relative L2均为0；GPU数值摘要附在`gpu-report.json`，依据是实际通过的精确比较断言。

| GPU组件测试 | peak allocated | peak reserved | 测试后设备空闲 |
| --- | ---: | ---: | ---: |
| BF16非连续Q/K、多长度 | 12.1494 MiB | 34 MiB | 15270.8125 MiB |
| 32步Decode位置旋转 | 0.03125 MiB | 34 MiB | 15270.8125 MiB |

峰值来自各测试期间的进程PyTorch allocator统计；reserved可包含同进程前一测试留下的allocator池，设备空闲量是瞬时整卡读数。这些是组件测试数据，不是完整模型或引擎显存预算，不验收Task13的安全余量。

## 参考身份与阈值裁定

只读使用`~/nano-vllm-artifacts/qwen3-vl/oracle-reviewed/hf-sdpa/`的十个用例，逐个调用现有`validate_artifact()`检查revision、文件SHA256、必需张量、shape/dtype/finite及缓存前缀，再比较整数positions和delta。没有重新采集或覆盖参考。

本次另运行现有`provenance()`，确认Transformers提交`2711ad22c82105e240a2877b92d082fe2857936f`、相关本地源码去注释/docstring后的AST与固定提交一致，checkpoint固定文件指纹通过；模型snapshot仍为`89644892e4d85e24eaac8bacfd4f463576704203`。未加载模型权重到GPU。身份记录存`provenance.json`。

Plan原文字要求浮点使用冻结组件阈值；用户尚未批准宽BF16包络。本次裁定：对固定HF、同设备同dtype且相同基础算子路径的组件，用finite加逐元素精确相等这一更严格标准验收，**不加载或批准`tolerances.json`**。独立Reviewer认可该裁定；它不意味着后续decoder/vision或跨kernel比较都应bitwise，也不改变原阈值提案状态。

## 独立Review

独立代理只读审查两个新增Python文件、Plan/Design、RED/GREEN及宿主GPU JUnit，并自行复跑CPU。结论无Critical、Important或Minor，Task3范围可通过。补充JUnit元数据fixture后，Reviewer再次只读确认报告功能未改变输入、旋转或断言，原结论保持。

一项非阻断建议留待将来修改相应检查时采用：补充rotary配置/形状拒绝路径的直接测试，例如奇数head_dim、sections总和不符、Q/K长度或dtype不一致、positions形状错误。当前实现静态检查无明显缺陷，不为该建议扩展本轮测试范围。

Reviewer不判定Runner/调度/KV接线、Task9入口和占位符语法、视频、多图引擎支持、任意checkpoint或非default rope scaling，以及Task2既有未提交改动。本轮接受这些边界，不声称M1整体完成。

## 复现命令与证据

仓库根目录、已有venv运行；GPU命令需要宿主CUDA可见权限，无需安装或下载：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_VL_ARTIFACTS="$HOME/nano-vllm-artifacts/qwen3-vl"
export QWEN3_VL_ORACLE="$QWEN3_VL_ARTIFACTS/oracle-reviewed"
export QWEN3_VL_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"

.venv/bin/python -m pytest tests/test_qwen3_vl_mrope.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_mrope.py -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_mrope.py::MRopeGPUTests -m gpu --require-gpu -q --junitxml=/tmp/qwen3-vl-task3-repro.xml
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/layers/mrope.py tests/test_qwen3_vl_mrope.py
.venv/bin/python -m ruff format --check nanovllm/layers/mrope.py tests/test_qwen3_vl_mrope.py
git diff --check
```

本机持久化证据目录为`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task3/`：`baseline.log`、`collection.log`、`red.log`、`green.log`、`cpu-unit.xml`、`gpu-sandbox.log`、`gpu.xml`、`gpu-final.xml`、`gpu-report.json`、`provenance.json`、`cpu-final.log/xml`、`quick.log/xml`、`review.md`及包含新增文件的`task3.diff`。这些本机产物自忽略，不进入Git。复现示例中的/tmp只放新JUnit，不存正式HF Oracle。

最终实现文件SHA256：

```text
39ff40d0d32c4a1dce8eb07c0984f7ec093a3320fc69d03e8eb5e3081f6231f3  nanovllm/layers/mrope.py
be45da6b4bcaf8babf68bda0c95ea5d4f7f4127bd0ef99ff59cd44fbcb4a4237  tests/test_qwen3_vl_mrope.py
```

Task3交付文件为上述两个新Python文件、本报告及实施计划状态/Task3勾选更新。Task2三个Python文件SHA与其报告一致，未被本任务修改。拟提交信息为`feat: add Qwen3-VL positions and interleaved MRoPE`；仅供审阅，未获Commit授权，不暂存、提交或推送。

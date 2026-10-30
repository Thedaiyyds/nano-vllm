# Task 1c：Qwen3-VL-2B HF Oracle、GPU Kernel 与数值阈值依据

> 2026-10-30 恢复说明：下文保留Task1c首次执行的历史路径与结果。原`/tmp`正式数据已丢失；当前正式参考已使用原生成器重新采集到`~/nano-vllm-artifacts/qwen3-vl/oracle-reviewed/`，并完成完整性审核及校准。后续Task使用长期目录，详见[恢复记录](../docs/superpowers/plans/2026-10-30-qwen3-vl-oracle-recovery.md)。

范围：仅Task 1c。没有实施Task 2，没有扩展旧Qwen3-0.6B GPU回归，没有安装/升级依赖或修改CUDA环境，没有暂存、Commit或Push。当前新增内容均为测试、参考采集工具、小型输入/身份清单及报告。

## 实际结果与状态

| 项目 | 真实结果 | 能说明什么 |
| --- | --- | --- |
| Task1c CPU | **28 passed，4 deselected** | 比较器、数据身份/完整性、输入契约和错误实验拒绝逻辑通过 |
| Task1c真实GPU Kernel | **4 passed，28 deselected** | BF16 matmul及255/256/257长度的实际KV存储、分页Attention通过受控对照 |
| 正式HF参考 | **30个独立GPU进程成功** | 10输入×SDPA/eager/SDPA重复；没有两个模型同时驻留 |
| 数据回放审核 | **30组通过** | SHA、revision、组件清单、shape/dtype、finite、cache前缀不变检查均通过；1680对首步KV逐位相同 |
| SDPA重复确定性 | **10用例全部浮点张量max_abs=0** | 在当前固定环境、输入和后端下可重复；不是跨后端逐位等价保证 |
| 全部已建立CPU测试 | **54 passed，6 deselected** | 包含Task1a/1b已有CPU兼容检查；未重新跑旧模型GPU矩阵 |
| 阈值提案 | **PROPOSED_FROZEN_PENDING_USER_ACCEPTANCE** | 已保存、带来源哈希且拒绝覆盖，等待用户书面验收；不是已批准的自动放行门槛 |
| HF BF16跨路径argmax | **有差异，完整保留** | 960个位置中23处cached/重算argmax不同，21处至少一侧前两名同分，分歧处cached margin最大0.125 |
| Ruff/格式/Diff | **通过** | 不代表模型数值已经等价 |

正式数据仅使用`/tmp/nano-vllm-qwen3-vl-validation/oracle-reviewed/`。先导目录`hf-sdpa`、`oracle-v1`和`oracle-final`用于工具验证，**不参与正式阈值**。所有成功/失败/重采历史均保留；最终30组使用同一份归档生成器源码与输入清单。

## 参考数据如何产生

模型固定为`Qwen/Qwen3-VL-2B-Instruct` snapshot `89644892e4d85e24eaac8bacfd4f463576704203`；HF源码固定为`2711ad22c82105e240a2877b92d082fe2857936f`。本地研究注释允许保留，但三份相关源码移除docstring后的AST必须与固定提交一致。

[checkpoint.json](fixtures/qwen3_vl/checkpoint.json) 固定配置、权重、tokenizer、chat template和Processor文件SHA256。该清单取自原固定snapshot，并独立检查LFS SHA256/Git blob SHA1与缓存blob名称；不会从任意`QWEN3_VL_MODEL`路径自动接受新内容。权重SHA256为`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`。搬移相同文件允许，内容改变拒绝。

每次采集：离线加载625张量，确认无missing/unexpected/mismatched keys；BF16、eval、inference_mode、无采样/dropout、关闭compile和CUDA Graph。先使用HF SDPA生成固定32-token续写，再让所有实验沿同一续写teacher-force。因此比较的是**相同prefix下的数值**，不是两段自由生成文本碰巧相似。

32个预测位置包括Prefill的首个预测及31次Decode；逐步KV对应最后prompt位置及31个新输入token。cached路径复用KV；全序列重算路径每步从完整prefix和空历史重新计算。重算设置`use_cache=True`仅为读取本次新算出的KV，**不传入/复用上一步past**；并非靠缓存少算。固定图像的HF视觉特征可复用，语言网络全部重新计算，避免把同一图的Vision Encoder重复跑32遍。

| 用例 | 实际L | V | SDPA cached/重算argmax相同位置 |
| --- | ---: | ---: | ---: |
| text_short | 24 | 0 | 31/32 |
| text_255 | 255 | 0 | 29/32 |
| text_256 | 256 | 0 | 31/32 |
| text_257 | 257 | 0 | 32/32 |
| image_square | 83 | 64 | 32/32 |
| image_rect | 115 | 96 | 32/32 |
| image_same_layout_other_pixels | 83 | 64 | 32/32 |
| image_limit | 275 | 256 | 32/32 |
| image_candy | 83 | 64 | 31/32 |
| image_cat | 115 | 96 | 32/32 |

文本长度用重复合法token固定，主要验证长度/分页边界，不是语言质量评测。图片包含受控RGB图、同布局不同颜色、512×512上限图和仓库`candy.JPG`/`哈基米.jpg`；均由固定AutoProcessor处理。这里的最长语言序列不是4096；4096整引擎上限验收属于后续Task13，不在此冒充完成。

保存的数据包括：原始CPU输入、grid/mm类型、HF三轴positions/delta、32续写IDs；patch embed、插值索引/权重、FP32/BF16绝对位置、视觉RoPE与首层旋转Q/K、ViT0/5/11/17/23、主merger和三组DeepStack；融合embedding、文本RoPE、前三层Q/K norm及旋转结果、注入前后hidden、最终norm/logits；28层Prefill/末步选定KV及每步新增token的KV。

Hooks只观察/委托真实HF操作，不用Mock替代模型。有/无hooks的Prefill最后logits也要求逐位相等。张量及时搬CPU；布局保留HF原始维度，KV统一token-major，并记录sample positions。每组有manifest、文件哈希、环境和生成器源码哈希。正式BF16最大allocated **4.1617 GiB**、reserved **4.2598 GiB**，不是完整nano-vLLM引擎显存或性能承诺。

## 每项核心测试证明什么

| 测试组 | 用通俗中文解释 |
| --- | --- |
| 比较器9项 | 用手算`[1,3]`与`[1,2]`确认误差真是max=1、mean=0.5、relative L2=1/√5；shape/dtype错误、NaN/Inf、缺阈值、绝对或相对超差都不能通过；整数位置必须精确相等。不是只检查函数能运行。 |
| Artifact身份3项 | revision或文件SHA不对时拒绝；读取时同时检查完整张量清单、shape、dtype、finite。防止拿旧/坏数据作参考。 |
| Processor契约10项 | 每个用例的实际token数、grid、patch形状、视觉占位符数与预期一致；输入留在CPU。图片尺寸不等于视觉token数，必须依据Processor结果。 |
| Review负例6项 | 错backend、错case、同架构不同权重、KV采样位置错、未成对计算KV误差、三组同时缺组件均被拒绝。防止“错误实验也得到零差异”。 |
| GPU matmul | 实际执行BF16矩阵乘法，与禁TF32的FP32积比较；误差预算来自FP32累加与BF16输出舍入，不是凭结果现调atol。只证明本受控算例。 |
| GPU Attention：255/256/257 | 使用Qwen3-VL的16Q头/8KV头/head_dim128与256-token页；逻辑页映射到物理页2、0，写入后逐元素gather验证，检查padding slot与未用页；真实FA变长Prefill、分页Prefill和Decode与FP32 math SDPA比较。能发现块边界/布局错误，不用普通连续Tensor假装分页KV。 |
| HF逐步重算 | 同一个预测位置用cached与完整prefix两条路径计算，同时记录logits及每步每层KV。positions必须相同、cache长度必须正确、已缓存前缀不能被追加操作改动。浮点差异量化保留，不要求BF16所有路径逐位相等。 |

Kernel受控Q/K幅度0.1、V幅度0.1。初始Attention预算漏掉了`P→BF16`再做PV的舍入，产生3 failed/1 passed；独立诊断在255/256/257分别看到24/13/11个旧预算超界点，加入该计算阶段后为0。最终预算为`u_bf16*(|reference|+P@|V|)+8*L*eps_fp32*max|V|`；最后一项是受控输入的工程余量，**不是任意attention输入的通用误差定理**。matmul用`gamma_128*|A|@|B| + u_bf16*|reference|`。没有修改生产Kernel或以skip隐藏失败。

实际FA Prefill最大绝对误差分别约`5.413e-4/6.219e-4/7.019e-4`，Decode约`6.810e-5/5.874e-5/5.728e-5`；Triton写入/gather要求逐位一致。

## 阈值怎样确定，哪些尚未获批准

正式`tolerances.json`按case/组件保存三个同时需要满足的指标：max_abs、mean_abs、relative L2。整数/布尔、shape、dtype精确检查，不借浮点阈值放行。

每项浮点提案：`limit = 2 * max(observed_HF_variations) + one_outward_ULP(reference)`。观察项包括eager对SDPA、SDPA重复；Decode logits与KV另加入cached/重算的**配对**差异。ULP取该张量dtype下朝外相邻数值间隔，mean与L2预算按逐元素ULP形成。factor2是明确的工程安全系数，不是统计置信界或数学正确性证明。

140/3164个浮点组件的观测差异为零；它们仍单列one-ULP提案，避免将“此时为零”当成“任意后端必须bitwise”。**该floor及factor2尚需用户书面验收**。文件拒绝覆盖，后续候选实现失败不能直接放宽它。

例子：text_short Prefill logits的观测max_abs=0.15625，提案max_abs=0.375；image_square主视觉特征观测relative L2约0.05272，包络约0.11120。后者已经较宽，不能只凭落在包络内就声称模型正确；必须配合整数元数据、逐级对齐、Kernel、KV布局与生成门槛。

特别是text_257的HF eager第14层V：BF16 cached/重算max_abs=57.5、relative L2=0.138945，局部位置比整体更敏感。对此补做同权重FP32小用例诊断，未把这项当作普通微小误差：

| 同一text_257用例 | BF16 | FP32计算（checkpoint权重原为BF16） |
| --- | ---: | ---: |
| 第14层V max_abs | 57.5 | 0.0058823 |
| 第14层V relative L2 | 0.138945 | 1.39293e-5 |
| logits relative L2 | 原始BF16记录逐项保留 | 2.58657e-6 |
| cached/重算argmax | eager原始差异保留 | 32/32一致 |
| 峰值allocated | 正式BF16全组最大4.1617GiB | 8.0447GiB |

这支持数值路径敏感性的解释，**不证明全部BF16差异都已逐一解释**。宽BF16包络定位为诊断提案，不自动作为正确性充分条件；后续应对敏感组件保留同后端/FP32小用例定位。Task1c交付参考事实与依据，未将候选VL实现标为通过。

## 证据位置

- 小型版本控制摘要：[calibration_summary.json](fixtures/qwen3_vl/calibration_summary.json)：所有case误差、重复结果、23处分歧、Kernel证据及来源哈希。
- 正式大数据：`/tmp/nano-vllm-qwen3-vl-validation/oracle-reviewed/`，30份manifest/PT，约GiB量级，不入Git。
- 提案：同目录`tolerances.json`（约3.1MB），带各源manifest哈希；没有从nano候选误差反推。
- FP32诊断：同目录`fp32-final/diagnostic.json`、`tensors.pt`、`source.py`。
- 本机日志：`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task1c/`，含comparator-red、artifact-red、review-red、inventory-red、kernel-first/kernel-reviewed、corpus-reviewed、reference-audit和fp32-final。

## 可复制的命令

先在GPU可见的SSH宿主终端进入仓库；默认工具沙箱可能看不到设备，不能将这种BLOCKED改成skip。

```bash
cd /home/thedai/nano-vllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_VL_MODEL=/home/thedai/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203

# 实际28 passed，4 deselected
.venv/bin/python -m pytest tests/test_qwen3_vl_reference.py -m 'not gpu' -q

# 新目录保存Kernel证据，避免覆盖原记录；实际4 passed，28 deselected
QWEN3_VL_ARTIFACTS="$(mktemp -d /tmp/qwen3-vl-kernels.XXXXXX)" .venv/bin/python -m pytest tests/test_qwen3_vl_reference.py -m gpu --require-gpu -q

# 如需完整重采，必须用新目录；真实执行为30个独立GPU进程
export QWEN3_VL_ARTIFACTS="$(mktemp -d /tmp/qwen3-vl-oracle.XXXXXX)"
for run in hf-sdpa hf-eager hf-repeat; do
  backend=sdpa
  if [ "$run" = hf-eager ]; then backend=eager; fi
  for case in text_short text_255 text_256 text_257 image_square image_rect image_same_layout_other_pixels image_limit image_candy image_cat; do
    .venv/bin/python -B -m tests.reference_qwen3_vl --case "$case" --backend "$backend" --output "$QWEN3_VL_ARTIFACTS/$run" || break 2
  done
done
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B -m tests.reference_qwen3_vl --calibrate "$QWEN3_VL_ARTIFACTS"

# 对刚采集的数据复现FP32诊断；输出目录应为空
PYTHONPATH=/home/thedai/nano-vllm .venv/bin/python -B /tmp/nano-vllm-qwen3-vl-validation/oracle-reviewed/fp32-final/source.py "$QWEN3_VL_ARTIFACTS/fp32-check"

# 全套已建立CPU：实际54 passed，6 deselected
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m ruff check tests nanovllm/__init__.py
.venv/bin/python -m ruff format --check tests nanovllm/__init__.py
```

采集与阈值命令在目标已有结果时拒绝覆盖；不要删除已审阅期望以让失败消失。缺数据、模型指纹不符、GPU不可用均非零退出，无自动下载/重建环境。

## TDD与Review

比较器9个断言先失败再通过；artifact身份3项先失败再通过；Review的身份/配对5项及完整组件1项先失败再通过。测试模块先正常收集，没有将ImportError当作Red。Processor契约、HF已有正常路径和Kernel是特性验证，不人为破坏其实现制造All Red；普通采集/报告辅助函数不声称每个都有独立Red记录。

独立Review无Critical，提出3项Important，均在一次修复中处理：逐步KV成对对照；backend/case及必需组件校验；固定checkpoint内容身份。修复后重新采集正式30组并校准、回放核对，CPU/GPU全套重验；没有将初次Review改写成“无问题”，未二次复审。

1项Minor延期：Kernel JSON写入器在重复使用同一输出目录时可能覆盖。当前正式证据放独立目录并记录哈希/只读归档；复现命令使用新目录。正式Oracle和阈值本身拒绝覆盖。

实施裁定：复用当前无文件冲突的feature工作区；继续使用宿主GPU权限而不改CUDA；采集阶段以新鲜cache返回KV但不复用历史，保证可观测完整重算；将未配对/未完整诊断的先导数据排除于正式阈值；对BF16敏感用例增加单模型FP32诊断，未扩展旧模型矩阵。阈值提案等待用户验收，不执行Task2。

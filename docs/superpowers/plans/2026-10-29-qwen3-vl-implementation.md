# Qwen3-VL-2B-Instruct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在单张 RTX 5060 Ti 16GB、TP=1 上实现并验证 Qwen/Qwen3-VL-2B-Instruct 的纯文本及单图 Prefill/Decode，与固定 Transformers 中同一 Qwen3-VL-2B checkpoint 逐层及端到端数值对齐；正确性通过后，进行公平、可复现的性能比较。

**Architecture:** 自实现视觉网络、三轴位置与 Interleaved-MRoPE、图文替换和 DeepStack；复用现有语言 Attention、线性层及分页 KV。VL 使用独立 decoder 保持 HF residual 顺序；先完整 Prefill、单活动请求，禁用 VL prefix cache 与 CUDA Graph。单图和多图共用模块及按图分段的接口，多图属于下一轮验收，不创建另一套模型。

**Tech Stack:** Python 3.12、PyTorch 2.9.1+cu130、Transformers 5.21.0.dev0（固定源码提交）、flash-attn 2.8.3、Triton 3.5.1、safetensors；拟采用 uv 管理开发工具、pytest 统一运行、Ruff 增量检查。现有 CUDA 环境原地复用；unittest.TestCase 断言可由 pytest 执行。

**Spec:** [已批准设计](../specs/2026-10-29-qwen3-vl-design.md)。执行者必须同时阅读两份文档。

**Status:** 2026-11-10，Native Task12本地技术验收完成。完整625源tensor实载无skip，HF固定特征/native自编码两组GPU2项通过51.16s，全部计算对照三指标0；完整CPU313、快速CPU31、既有text/vision GPU8通过。独立Review规格/质量PASS，Critical/Important/Minor均0。见[Task12报告](2026-11-10-qwen3-vl-task12-results.md)。未暂存/Commit/Push，不开始Task13；Oracle只读，宽BF16阈值未批准，远程CI未运行。

## Global Constraints

- 硬件目标为 RTX 5060 Ti 16GB，TP=1。
- 初期限定 batch=1、单张静态 RGB 图片、无 padding 的内部输入、完整 Prefill、BF16 权重、TP=1、关闭 CUDA Graph。
- 初始验收上限为 `L+max_new_tokens<=4096`、单图 `V<=256`（P<=1024）；以 Processor 实际 grid 为准。
- 参考 Transformers 提交：`2711ad22c82105e240a2877b92d082fe2857936f`；模型 snapshot：`89644892e4d85e24eaac8bacfd4f463576704203`。
- 首轮不支持多图、视频、混合批次、chunked prefill、抢占重算、TP>1 与 CUDA Graph；必须明确拒绝未经支持的输入/配置。
- 复用 HF AutoProcessor、配置解析、PyTorch 基础算子；HF 完整模型和视觉模块只作为测试参考，不委托 HF generate 执行 nano-vLLM 推理。
- 初期 VL 禁用 prefix cache 的查找和发布，保留语言 paged KV 读写；共享组件保留针对性兼容回归，原 Qwen3-0.6B 仅保留必要加载/生成smoke，不建立全模型数值基线。
- `enforce_eager=True` 不会自动关闭 torch.compile；正确性基线额外使用 `TORCHDYNAMO_DISABLE=1`，编译优化另行验收。
- 16GB 下 HF 与 nano-vLLM 分进程顺序运行，参考张量落 CPU；所有 GPU 测试串行执行。
- 当前执行授权为Native Task12；新工具安装须先展示dry-run并获用户批准；不推进后续任务、不自动提交或推送。每任务保留验证记录；每个里程碑经过独立 Review 后，提交具体 diff 和拟提交文件清单，请用户人工批准 Commit，未获批准不得暂存、提交或推送。

## Review Focus

1. 空 prompt、孤立 image/video 占位符、伪造 grid 或占位符数量不一致：应在 KV 分配前报错，不能生成似乎正常的文本。归 Task 9/14。
2. 排队请求中前一个占用 KV，或队首请求永远无法容纳：只准入一个，容量不足直接失败而非卡住或重算。归 Task 7/14。
3. 相同 token IDs 搭配不同像素、后续纯文本请求：不得共享错误 KV、rope_delta 或 DeepStack 状态。归 Task 7/14。
4. 请求异常、重复退出、初始化中途失败：可恢复异常释放资源，进程级 CUDA 错误明确要求重启，不能宣称已恢复。归 Task 8/14。
5. HF 默认采样和双 EOS、max_tokens=1、页边界 255/256/257：生成数量、停止时机和 KV 内容应正确，不能以“输出能读”验收。归 Task 6/8/14。

## 范围、资源与执行顺序

本计划交付设计中的 P0–P3。P4 是后续路线：同一请求多图/图文交错 → 多请求并发 → chunking/抢占 → 视频 → 按需优化。P4 不混入本轮验收；按图片列表、`image_grid_thw[I,3]` 和分段 attention 保留接口，入口暂校验 I=1，不重复实现视觉模型。

| 里程碑 | 任务 | 可单独审查的产物 |
| --- | --- | --- |
| P0 | 1a→1b→1c | 测试基础、旧功能轻量兼容检查、同模型HF oracle及内核报告 |
| P1 | 2–7、8a→8b→8c | 目标 VL checkpoint 的纯文本 Prefill/Decode 与原 Qwen3 回归 |
| P2a | 9（依赖 3） | Processor 契约及三轴位置对齐 |
| P2b | 10、11a→11b→11c | 自实现视觉中间结果及所有 merger 对齐 |
| P2c | 12、13a→13b→13c | 图文替换、DeepStack、完整 Prefill 与真实 KV 写入 |
| P3 | 14a→14b→14c、15a→15b | 单图Decode/生命周期、正确性验收；随后同模型性能比较 |

建议顺序为 1a→1b→1c→2→3→4→5→6→7→8a→8b→8c→9→10→11a→11b→11c→12→13a→13b→13c→14a→14b→14c→15a→15b；带字母的子任务替代原父任务的执行步骤，后文依赖 Task 1/8/11/13/14/15 表示该父任务全部子任务已通过。任务的依赖栏是实际门槛；CPU 工作可在 GPU 暂不可用时继续，但 GPU 门槛不得标记通过。此系统的模块需要一起组成推理链，因此用一个计划保持接口一致，不拆成互不相干的子项目。

### 是否需要租算力

**本轮不要求租卡，也不实现 TP。** 这是基于预算的工程判断，尚不是实机性能保证：权重约 3.96 GiB，BF16 KV 每 token 112 KiB，每 256-token block 为 28 MiB，4096 token 的 16 个 block 为 448 MiB。首版单活动请求的 KV 池可封顶到 `ceil(max_model_len / block_size)`，无需把剩余显存全部吃满。仍须为视觉激活、Attention 工作区、logits、加载临时副本、CUDA context 和内存碎片留余量。

- HF/nano 用独立子进程顺序加载；checkpoint 按 tensor 流式复制，不同时保留整份 CPU/GPU state_dict；测试保存关键层/关键位置，不保存 GPU 上所有层全量 attention 矩阵。
- GPU 常规验收用 BF16；FP32 诊断先做单层/小输入。完整 FP32 双模型并存不是验收前提，若确实需要大规模 FP32 诊断，再评估更大显存的单卡。
- 多图、短视频、低并发本身不要求多卡；首先按总视觉 token 和语言长度控制预算。高分辨率、多帧、长上下文或高并发压测超出实测预算时，才考虑更大显存 GPU；租用容量应由峰值报告决定，不预先指定昂贵机型。
- **以后做真实 TP=2 时，需要至少两张独立 GPU** 来验收参数切分、NCCL 通信、KV 分片及故障行为。单 GPU 多进程或 CPU/Gloo mock 只能测试部分逻辑，不能作为 TP GPU 验收。优先同机两卡，并实际测通信能力；两卡显存不能简单视为一个自动合并的内存池。
- 已安装 PyTorch 的 `distributed_c10d.py` 明确指出 NCCL 进程需要独占其使用的 GPU；因此不设计“一卡伪装两卡”的验收方案。该说明与当前项目 `set_device(rank)` 的结构一致。
- 上次检查中 `nvidia-smi` 在当前会话不可访问；Task 1 必须重新区分宿主驱动/会话 GPU 暴露问题和内核架构支持问题。先定位，不能直接推断需要租卡。若现有 FlashAttention 不可用，应先提出有证据的兼容性处理方案；不得悄悄换后端或跳过真实分页 KV 测试。

### 数值对齐协议（所有 GPU 任务共用）

1. 固定模型、Processor、chat template、源码 revision、环境与输入张量；HF `eval()`、`inference_mode()`、`do_sample=False`，关闭 dropout。生成参数不继承 checkpoint 的随机采样默认值。
2. Task 1c 在实现候选模型前生成 HF eager/SDPA、重复运行、cached/无 cache 重算的参考差异，按组件冻结 `tolerances.json`。每组记录 max_abs、mean_abs、相对 L2、logit top-k 和 top-1 margin；数值阈值来自实测基线及书面理由，不在计划阶段捏造“已验证”的统一 atol。测试遇到缺失阈值、NaN/Inf、shape 不符必须失败。
3. IDs、mask、grid、positions、delta、权重映射与分块元数据必须精确相等；浮点同时检查绝对与整体相对指标。旋转/残差 dtype 舍入点必须遵循 HF，不能全靠放宽误差掩盖。
4. 对齐顺序：patch embed → 绝对位置 → 视觉 Q/K RoPE → ViT 层/merger → 融合 embedding → 前三层 DeepStack 后输出 → 最终 norm/logits → 各层物理 KV。语言 attention 可采用现有 FlashAttention，但其误差应在独立 kernel 对照中有据可查。
5. Decode 使用 HF 选定的同一续写 token 序列做 teacher forcing，比每一步 logits 和 KV；另做短固定 greedy tokens 一致性检查。低 margin 分歧记录为未通过/待定位，不隐去、不事后扩大阈值。
6. 默认仅保存最后位置 logits；诊断选定位置时手动用 LM head 做分段投影，避免 `[4096,151936]` 的完整 logits 长驻 GPU。CPU fixtures 也按 case/层拆分、读取后释放。
7. 验证报告包含命令、case、revision、dtype/backend、峰值 allocated/reserved、GPU 可用内存和退出码。GPU 不可用或 fixture 缺失返回失败/明确 blocked，不以 skip 计作通过。

## 文件职责与固定接口

| 文件 | 变化与职责 |
| --- | --- |
| `nanovllm/config.py` | 规范化 text_config/dtype/EOS；VL 初期限制和资源上限 |
| `nanovllm/models/qwen3_vl.py`（新） | 显式 residual 文本 decoder、VL wrapper、融合与 DeepStack |
| `nanovllm/models/qwen3_vl_vision.py`（新） | patch embed、插值/二维 RoPE、ViT、主/DeepStack merger |
| `nanovllm/layers/mrope.py`（新） | prompt 三轴位置、delta、Interleaved-MRoPE |
| `nanovllm/utils/multimodal.py`（新） | PreparedMultimodalPrompt、Processor 适配、输入校验 |
| `nanovllm/utils/loader.py` | 严格 VL 权重映射、packed shard 完整性与 tied alias 审计 |
| `nanovllm/engine/sequence.py` | 请求级 CPU 输入/位置/delta；不扩展 TP pickle 协议 |
| `nanovllm/engine/block_manager.py`、`scheduler.py` | 可关闭 prefix cache、单活动请求准入、EOS/释放 |
| `nanovllm/engine/model_runner.py`、`llm_engine.py` | 模型分支、生命周期、VL profile、Prefill/Decode 接入 |
| `nanovllm/sampling_params.py`、`layers/sampler.py` | 真正 greedy，保留正温度采样 |
| `tests/reference_qwen3_vl.py`、`tests/support.py`（新） | 固定 HF oracle、串行子进程及比较助手 |
| `tests/fixtures/qwen3_vl/cases.json`、各 `tests/test_qwen3_vl_*.py`（新） | 小型输入定义、CPU/GPU 定向测试；不拷贝模型权重 |
| `tests/qwen3_vl_validation.md`（新） | 环境、阈值来源、任务验收与显存结果 |
| `README.md`、`examples/qwen3_vl.py`（后期新增示例） | 最终用法、范围与限制 |

除1a有证据才进行的包入口延迟导入外，默认不改 `models/qwen3.py`、`layers/attention.py`、`layers/linear.py`、`layers/embed_head.py`、`layers/rotary_embedding.py` 和已有实验脚本；复用这些组件并回归。若内核 smoke 显示必须修改 Attention 后端，先记录事实并修订相关任务，不顺手重构。

## 工程环境核实、工具选择与迁移

本轮只读检查所得事实（实施前1a重新采集，不能把本次清单当作永久锁）：

| 项目 | 核实结果 | 影响 |
| --- | --- | --- |
| `pyproject.toml` | setuptools构建；Python >=3.10,<3.13；torch>=2.4、transformers>=4.51、triton>=3、flash-attn、xxhash；无dev/pytest/Ruff配置 | 这些下限不足以复现VL参考版本；本轮仅规划未来配置变更 |
| uv/虚拟环境 | `/home/thedai/.local/bin/uv` 0.13.0；`.venv/pyvenv.cfg` 标记由uv创建；Python3.12.15 | 复用原`.venv`；uv CLI位于环境外，不在venv metadata中属正常 |
| CUDA相关包 | torch2.9.1+cu130、torchvision0.24.1+cu130、flash-attn2.8.3、triton3.5.1 | 禁止测试工具安装触发替换、重编译或降级这些包 |
| 其他工具 | pytest/Ruff未安装；无仓库uv.toml、uv.lock、.python-version、pytest/Ruff配置或workflow；NumPy2.5.4 | 工具版本需dry-run确认，不能声称pytest/Ruff已经可用 |
| 已有测试 | 无tests目录；根目录test*.py是处理器/卷积/全模型实验，部分顶层就运行GPU或写图 | pytest只收集tests/，保留根脚本，不用其作为自动测试suite |
| HF参考 | editable到`/home/thedai/transformers`，HEAD为已批准提交；三份本地修改文件移除docstring后AST与HEAD一致 | 保存路径、git提交、dirty diff和AST；不清理参考仓库 |
| 配置/权重 | 顶层dtype=None、text BF16，28层/8KV头/128维，theta5000000，sections[24,20,20]；vision24层/DeepStack[5,11,17]/axial theta10000；EOS双值；625张量无独立lm_head | 保留原计划的nested config、tied head与命名空间审计 |
| HF接口 | apply_chat_template有processor_kwargs；vision插值align_corners=True、主/DeepStack norm顺序不同 | 测试据固定本地源码，不照搬其他HF版本示例 |
| GPU | nvidia-smi仍无法连接驱动 | GPU相关门禁目前未运行；不推断租卡或内核不兼容 |

**推荐 uv + pytest，但迁移不等于重建项目环境。** unittest无需额外依赖，但参数化、多层fixture、marker和JUnit产物需更多自建代码；pytest提供这些能力，并可直接运行原计划的unittest.TestCase方法。当前没有正式unittest测试文件需要迁移，成本主要是新增conftest/markers、明确收集范围和把GPU开关转成marker；保留原数值断言及TestCase亦可，不要求一次改写所有测试风格。pytest运行unittest类时不依赖自定义load_tests，也不向TestCase方法注入fixture参数，使用setup或autouse适配。

uv用于指定解释器下的工具安装/包清单；测试用`.venv/bin/python -m pytest`。**不执行uv sync、uv lock、uv venv、uv python install或裸uv run自动同步**，也不将本地editable HF替换为PyPI同名版本。开发工具候选先用`pytest>=8,<9`、`ruff>=0.9,<1`解析，1a将实际可安装版本精确固定到`requirements-test.txt`并记录；候选范围是计划选择，非已验证兼容性事实。约束文件精确限制当前CUDA包/HF版本；manifest另存HF editable源，工具安装前后逐项比较；若安装方案涉及受保护包，停止并修订方案，不以用户批准计划推定可替换CUDA环境。

新文件职责补充：`requirements-test.txt`只含测试工具的精确版本；`tests/conftest.py`只管GPU门禁/元数据，不加载全模型或在收集阶段导入torch；`tests/test_harness.py`测门禁自身；`tests/test_original_baseline.py`只做旧功能兼容与原模型短生成smoke（沿用文件名，不再采数值baseline）；`tests/environment_manifest.json`记录便携版本及来源（机器路径放本地artifact）；`.github/workflows/tests.yml`仅做非GPU质量门禁。未来`pyproject.toml`增加pytest/Ruff配置，`nanovllm/__init__.py`若需延迟导入见1a，不随意扩展API。

### 测试命令、TDD与证据

以下为执行命令约定；当前执行范围为Task12，工具安装单独批准，Task13及以后命令不执行。仓库根目录：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_VL_MODEL=/home/thedai/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
export QWEN3_MODEL=/home/thedai/huggingface/Qwen3-0.6B
export QWEN3_VL_ARTIFACTS="$HOME/nano-vllm-artifacts/qwen3-vl"
export QWEN3_VL_ORACLE="$QWEN3_VL_ARTIFACTS/oracle-reviewed"
```

2026-10-30 数据恢复：正式HF参考已在`$QWEN3_VL_ORACLE`重新采集、审核并校准，原`/tmp`数据已丢失；见[恢复记录与使用命令](2026-10-30-qwen3-vl-oracle-recovery.md)。该目录仅作为只读基线，后续Task的新输出和Benchmark写到`$QWEN3_VL_ARTIFACTS`下的其他独立目录。现存正式数据不得覆盖；阈值仍为原PROPOSED状态，不因重采自动批准。

测试分为`gpu`、`hf_reference`、`checkpoint`标记；需要本地模型的测试必须标记checkpoint。两种CPU入口明确如下，均不以GPU skip代表成功：

```bash
# 快速CPU（不依赖GPU、HF参考或本地checkpoint；CI同口径）
.venv/bin/python -m pytest tests -m "not gpu and not hf_reference and not checkpoint" -q
# 本地完整CPU（包含已具备本地资源的HF/配置检查；缺资源须报错）
.venv/bin/python -m pytest tests -m "not gpu" -q
```

GPU显式命令加`--require-gpu -m gpu`；此选项不是pytest内置选项，由1a定义。选中GPU测试但CUDA不可用、未选中任何GPU测试、GPU项skipped/xfail或必需fixture缺失，退出非零。CPU CI排除GPU属于**未运行**，报告不得显示“GPU通过”。

- [ ] 对每个新模块先建立最小可导入接口，执行`--collect-only`并核对节点数。ImportError、插件缺失、驱动失败、fixture setup异常只表示基础设施/脚手架未就绪，不计TDD red。
- [ ] 再运行具体节点，要求到达断言并显示数值/状态行为不符；可以使用显式NotImplementedError作为过渡，对于MRoPE/视觉位置、DeepStack注入、KV映射/隔离、Decode delta和资源释放等高风险逻辑，还必须演示受控错误扰动会使有效断言失败；普通配置/字段校验只需有效行为断言的red→green，不机械要求额外扰动。日志记录断言预期与实际。
- [ ] 最小实现后跑同一节点变绿，再跑所属模块和前置回归；已有行为通过则保留覆盖，不重复实现或故意破坏生产代码。
- [ ] 浮点测试必须带固定输入、独立HF/解析期望、dtype/backend和已冻结阈值；禁止比较对象与自身、仅比shape就宣称数值通过、随机权重同构实现互证或事后放宽阈值。
- [ ] 原文各Python片段保留为命名测试的核心断言，`self`为TestCase；pytest风格新增子任务可用原生assert。输入/模型/hook由对应setup产生，普通setup不改变契约。

`tests/support.py`保留single_rank(device)、assert_aligned、run_worker等接口；CPU组件Gloo、GPU NCCL，均保证销毁。Runner自己创建NCCL，测试不嵌套初始化；GPU worker串行、超时失败，输出CPU张量后进程退出。

## Git工作流、里程碑Review与质量门禁

当前按用户要求Native恢复执行Task11c；需要时隔离工作区，任务结束进行独立Review，禁止暂存/commit/push。每次开始开发先运行：

```bash
git status --short --branch
git branch --show-current
git rev-parse --git-dir --git-common-dir
git diff --stat
git diff --cached --stat
git ls-files --others --exclude-standard
```

基线feat/multimodal上有用户未跟踪脚本/图片/docs，不可reset/clean或直接`git add .`。未提交修改与目标文件冲突时，在using-git-worktrees流程下选独立worktree；执行前明确分支、目标路径及如何带入已批准设计/计划，显式只复制所需文档，用户实验文件按只读路径使用；linked worktree已有隔离则复用。不自动stash、搬移用户数据或清空当前目录。

**以下每个里程碑均逐项执行同一门禁，任何一项不能以另一个里程碑的Review代替：**

| 里程碑 | 范围与回归（pytest命令参数共用上文） | Review重点 |
| --- | --- | --- |
| M0a | 1a；tests/test_harness.py；Ruff仅新配置/测试 | 工具安装diff、CUDA包不变、收集/GPU门禁不会伪通过 |
| M0b | 1b；共享组件CPU定向测试 + 一个原模型短生成GPU smoke | 旧接口可用，保持原调度/采样行为，不采旧模型全量数值 |
| M0c | 1c；test_qwen3_vl_reference.py + HF采集/校准CLI | oracle独立、阈值先冻结、GPU内核真实性 |
| M1 | 2–7、8a/b/c；config/mrope/loader/text/sampling/scheduler/runner模块 + 受影响共享组件定向回归 | residual/dtype/tied权重、调度安全、cache布局 |
| M2a | 9；input模块 + M1 CPU回归 | 孤立占位符、CPU所有权、输入预算 |
| M2b | 10、11a/b/c；vision_positions/vision_block/vision_merger/vision模块 + 1b | 两套视觉位置、分段Attention、norm/DeepStack顺序 |
| M2c | 12、13a/b/c；fusion/prefill/memory模块 + M1文本/GPU回归 | 全模型加载、真实KV与上限峰值 |
| M3正确性 | 14a/b/c、15a；decode/lifecycle/acceptance + 共享组件回归 + 最终一次原模型smoke | 同模型VL精度、delta/cache、异常释放与显存 |
| M3性能 | 15b；正确性报告通过后执行benchmarks/qwen3_vl.py | 同checkpoint/输入/配置，计时边界公平、可复现；无提速门槛 |

- [ ] **Diff**：`git diff --check`、`git diff --stat`、`git diff -- <本里程碑文件>`，另逐个阅读`git ls-files --others --exclude-standard`列出的新文件（普通diff不含它们）；核对生产/测试范围，无未解释依赖变更。
- [ ] **Lint/测试**：`.venv/bin/python -m ruff check <本里程碑改动的Python文件>`；`.venv/bin/python -m ruff format --check <相同文件>`；跑表中定向pytest及对应参考对齐。既有代码lint债务先记录，不借此全仓格式化；新文件全量、修改文件若遇历史问题先作局部修复或记录带行号豁免，由Review批准，不能全文件ignore掩盖新增错误。
- [ ] **独立Code Review**：使用requesting-code-review流程，由未参与该里程碑实现的reviewer审查完整diff、未跟踪新文件、Design/Plan、JUnit、精度和显存证据；可由人或届时授权的独立代理承担。本轮不指定执行模式。Critical/High阻止交付，Medium需修复或书面解释并经用户接受；Reviewer结论不能代替实测。
- [ ] **回归闭环**：修复review问题后重跑受影响共享组件测试和必要的VL参考对齐，记录SHA/diff digest与输入digest，避免旧报告套新代码。
- [ ] **人工批准Commit**：交付具体diff、新文件、测试/Review结果、已知阻塞、拟提交文件清单及commit message；等待用户明确批准后才能按清单暂存并执行commit。每次批准只覆盖所展示里程碑，不隐含后续commit/push授权；修改后须重新提供可审查结果。用户未批准时保留工作区，标记“技术验证通过，Commit待批”。

### 基础CI（未来1a配置基础、15收口）

Create `.github/workflows/tests.yml`，PR/push触发，Python3.12，安装**精确固定测试工具**，先`ruff check`/`ruff format --check`新增文件范围，再跑`.venv/bin/python -m pytest tests/test_harness.py -m "not gpu and not hf_reference and not checkpoint" -q --junitxml=cpu-junit.xml`；随着模块完成扩充到全部可运行CPU单元测试。CI只读提交中新增文件，安装发生在独立CI环境，绝不对本地CUDA环境sync。

轻量CPU suite不得从`nanovllm/__init__.py`无条件引入flash-attn；pytest conftest在收集阶段也不调用CUDA。若CPU模型单元测试需要torch/HF，增加独立CPU CI job与`requirements-ci.txt`，解析固定torch CPU wheel/HF源码提交及其依赖、dry-run验证后锁定；这不是复制CUDA requirements快照。该job未建立前仅将harness/config-independent tests列为CI覆盖，不宣称所有CPU测试已过CI。本地CPU含checkpoint套件及GPU仍是里程碑必需验证，不能用CI绿替代。

workflow应保存JUnit/工具版本并报告GPU=NOT RUN；本地GPU单独保存报告含GPU型号、capability、驱动、dtype/backend、峰值与case。1a用配置/门禁自检测试验证失败传播，15检查CI实际运行和覆盖清单（无远端运行权限时注明未执行，不声称CI通过）。

## Task 1：测试基础设施、旧功能轻量兼容与HF Oracle（三个独立阶段）

### Task 1a：原环境审计与pytest/Ruff运行基础

**执行状态：** Native Task 1a已执行并完成本地验证，等待用户验收；用户已批准四个工具包。详见[Task 1a结果](2026-10-29-qwen3-vl-task1a-results.md)。GPU不作为CPU门禁前提，未执行后续任务。

**Files：** Create `.github/workflows/tests.yml`（只运行已存在的harness套件）；Modify `pyproject.toml`（testpaths=tests、strict-markers、markers gpu/hf_reference/checkpoint，Ruff target py310、初始E4/E7/E9/F）；Create `requirements-test.txt`、`tests/__init__.py`、`tests/conftest.py`、`tests/test_harness.py`、`tests/environment_manifest.json`；条件Modify `nanovllm/__init__.py`仅延迟LLM导入以让轻量配置/调度测试不强制import FlashAttention。先证明导入问题再做条件修改；`from nanovllm import LLM, SamplingParams`保持可用并单独测。

**Interfaces：** pytest选项`--require-gpu`；conftest在setup检查CUDA、sessionfinish检查选中/执行/skip/xfail计数，输出JUnit；metadata函数`environment_manifest() -> dict`只读版本/来源。基础工具安装操作单独批准且不触及受保护依赖。

- [x] **1. 审计/保存（命令）**：`uv --version`；`.venv/bin/python -V`；`uv pip freeze --python .venv/bin/python > /tmp/qwen-vl-env-before.txt`；`uv pip check --python .venv/bin/python`。保存torch/torchvision/triton/flash-attn版本、wheel来源，HF direct_url、commit/AST。已有pip check问题单列，不自动升级修复。
- [x] **2. 生成约束并预演（命令）**：先用现有解释器生成有效的版本约束，避免把freeze中的editable行直接当作constraints：

```bash
.venv/bin/python -B - <<'PYCONSTRAINTS'
from importlib.metadata import version
from pathlib import Path
names = ['torch', 'torchvision', 'triton', 'flash-attn', 'transformers', 'numpy']
Path('/tmp/qwen-vl-protected-constraints.txt').write_text(
    ''.join(f'{name}=={version(name)}\n' for name in names))
PYCONSTRAINTS
```

`uv pip install --python .venv/bin/python --dry-run -c /tmp/qwen-vl-protected-constraints.txt 'pytest>=8,<9' 'ruff>=0.9,<1'`。dry-run若无法访问包源或约束无法解析则记录blocked，不移除保护。检查拟变更仅测试工具及新辅助依赖；选择准确版本写requirements-test。只有用户批准该具体安装清单后，才执行`uv pip install --python .venv/bin/python -c /tmp/qwen-vl-protected-constraints.txt -r requirements-test.txt`，再次freeze并与before比较。**禁止把该步骤作为当前授权安装。**
- [x] **3. 配置/门禁自检（red→green）**：`.venv/bin/python -m pytest tests/test_harness.py --collect-only -q`须正常收集；`test_no_gpu_required_fails`用子进程模拟CUDA不可见且`--require-gpu`，断言returncode!=0；`test_gpu_skip_rejected`断言GPU项skip/xfail导致失败；`test_root_experiments_not_collected`核对nodes均在tests/。再执行`.venv/bin/python -m pytest tests/test_harness.py -q`、`.venv/bin/python -m ruff check tests/conftest.py tests/test_harness.py`。
- [x] **4. 完成条件**：工具命令可用，保护包版本/源/ABI记录未改变，非GPU测试无需驱动/FlashAttention运行；CPU CI基础依赖方案可行。若轻量import因__init__触发GPU扩展，先新增import与公开API测试再最小延迟导入，必须通过1b公开API回归。本阶段不声称任何模型数值通过。

### Task 1b：旧功能轻量兼容检查（不建立Qwen3-0.6B数值基线）

**执行状态：** Task1b已完成CPU与真实宿主GPU验证及独立Review问题修复，等待用户验收；详见[验证报告](../../../tests/qwen3_vl_validation.md)。未开始Task1c。

**前置依赖：** 1a。若改动包入口，检查原公开API仍可用；无需为原模型建立独立基准副本或全量输出档案。

**Files：** Create `tests/test_original_baseline.py`（保留文件名便于追踪，内容为兼容测试）、`tests/support.py`、`tests/qwen3_vl_validation.md`。不修改原模型计算实现。

**Interfaces：** `run_worker(module: str,args: list[str]) -> CompletedProcess`、`single_rank(device)`保留；取消全量`capture_original_baseline`。共享组件用小输入/小块池；原模型保留一个GPU smoke和一条固定输入的短确定性回归。

- [x] **1. CPU定向检查**：`.venv/bin/python -m pytest tests/test_original_baseline.py -m 'not gpu' -q`。验证公开导入/旧Config构造；BlockManager prefix hit/miss及refcount回收；Scheduler保留chunked prefill和抢占状态；Sampler正温度路径用小型固定logits/seed检查。涉及真实CUDA的Sampler断言可放GPU小张量测试，不加载0.6B来验证组件。新增VL分支不能影响这些旧分支。
- [x] **2. 原模型GPU smoke**：`.venv/bin/python -m pytest tests/test_original_baseline.py -k test_qwen3_short_generation_smoke -m gpu --require-gpu -q --junitxml=/tmp/qwen-vl-legacy-smoke.xml`。仅一条固定短文本，正温度、`max_tokens=8`，验证加载成功、返回非空且不超过8个合法token、公开输出结构正确、可正常退出。无需与HF Qwen3-0.6B或历史token序列相等，不保存旧模型logits/各层KV，不测旧模型多长度/32步续写/性能矩阵。
- [x] **3. 固定输入确定性回归**：在共享组件修改前，使用原HEAD/原计算路径、固定短prompt token IDs、固定seed、BF16/eval、固定后端、关闭compile/Graph，运行最多8步正温度采样并保存精确token IDs、环境与输入digest到小型fixture；在相同环境重跑并要求token IDs精确一致。原模型不支持greedy，不能为采基线先改Sampler。先确认原路径重复运行稳定；若不稳定，定位随机/后端来源，不放宽为“文本相似”或扩大矩阵。该fixture只覆盖这一固定输入的短输出，不保存全模型logits/KV，不扩展多长度/分页组合。共享组件修改后及最终15a重放；工具升级改变环境须单独审查，不能静默刷新expected。命令：`.venv/bin/python -m pytest tests/test_original_baseline.py -k test_qwen3_fixed_input_regression -m gpu --require-gpu -q`。该基线属于Task 1b，当前Task 1a不运行或实现它。

- [x] **4. 完成条件与重跑范围**：共享组件行为断言通过，smoke及固定输入回归实际执行且无skip。后续任务仅改动相关共享组件时重跑相应组件测试；改动loader/Runner/包入口等旧推理路径时重跑同一个smoke和固定输入回归，最终15a再跑一次，不按里程碑扩展成原模型数值套件。真实GPU不可用就明确BLOCKED，不能伪造通过；HF/VL测试基础工作可独立继续，不能让旧模型详细基线成为VL Oracle前提。

```python
self.assertGreater(len(output['token_ids']), 0)
self.assertLessEqual(len(output['token_ids']), 8)
self.assertTrue(all(0 <= t < config.hf_config.vocab_size for t in output['token_ids']))
```

### Task 1c：HF Oracle、kernel对照与阈值冻结

**执行记录：** 用户已确认Task1c完成；见[Task1c报告](../../../tests/qwen3_vl_reference_results.md)及[持久化恢复记录](2026-10-30-qwen3-vl-oracle-recovery.md)。阈值仍为PROPOSED，未自动批准宽BF16包络或one-ULP floor。Task2授权独立于数值阈值批准。

**前置依赖：** 1a测试基础就绪；数值部分依赖真实GPU可用。不依赖Qwen3-0.6B输出或数值基线，不依赖候选VL实现。1b兼容门槛独立记录。

**Files：** Create `tests/reference_qwen3_vl.py`、`tests/fixtures/qwen3_vl/cases.json`、`tests/test_qwen3_vl_reference.py`；Modify `tests/support.py`、`tests/qwen3_vl_validation.md`。

**Interfaces：** `build_case(name: str,model_path: str) -> dict[str,Tensor]`返回CPU输入；`capture_reference(case: str,backend: str,output_dir: Path) -> Path`；`assert_aligned(name: str,actual: Tensor,expected: Tensor,tolerances: dict) -> dict[str,float]`。CLI增加`--calibrate DIR`读取两组HF artifacts并输出tolerances及理由，缺组件不得自动取默认阈值。

- [x] **1. 写/运行比较器行为red→green**：`.venv/bin/python -m pytest tests/test_qwen3_vl_reference.py -m 'not gpu' -q`；比较器必须拒绝shape、NaN、缺阈值和明确超阈值张量，不能只测manifest存在。

```python
with self.assertRaises(AssertionError):
    assert_aligned('hidden', torch.tensor([float('nan')]), torch.zeros(1), tolerances)
self.assertEqual(manifest['model_revision'], '89644892e4d85e24eaac8bacfd4f463576704203')
```

- [x] **2. 内核命令**：`.venv/bin/python -m pytest tests/test_qwen3_vl_reference.py -m gpu --require-gpu -q`。测BF16 matmul、Triton KV store、FlashAttention变长Prefill/分页Decode与SDPA，长度255/256/257、非连续物理块；不能仅import扩展。按真实layout gather旋转后K和V。
- [x] **3. HF命令**：原采集接口为`.venv/bin/python -B -m tests.reference_qwen3_vl --case text_short --backend sdpa --output "$QWEN3_VL_ORACLE/hf-sdpa"`；将backend/output分别改为eager/hf-eager，另采SDPA的hf-repeat，随后`.venv/bin/python -B -m tests.reference_qwen3_vl --calibrate "$QWEN3_VL_ORACLE"`。正式数据现已存在，命令拒绝覆盖；再次采集必须另设新的目标目录。同样覆盖text_255/256/257、image_square、image_rect、同token布局不同像素和image_limit。加入仓库candy.JPG和哈基米.jpg受控输入；禁止联网下载替代fixture。
- [x] **4. 完成条件**：保存IDs/grid/position/delta、关键hidden/merger/logits、32步teacher-forced续写、各层选定KV；cached与无cache重算对照通过。按组件记录HF eager/SDPA/重复运行/kernel基线并在实现前冻结阈值；若某组件HF自差异为零，阈值仍需独立精度依据与书面批准，不能假定跨kernel bitwise。缺GPU/阈值则阻塞对应数值任务。

## Task 2：规范化 VL 配置及预算字段

**执行记录：** 本轮实现、TDD、回归及独立Review见[Task2报告](2026-10-30-qwen3-vl-task2-results.md)。仅配置阶段，不表示M1文本推理或GPU数值验收完成。

**前置依赖：** Task 1a 的 CPU 基础设施。

**Files：** Modify `nanovllm/config.py:7` 的 Config；Create `tests/test_qwen3_vl_config.py`。

**Interfaces：** Config 新增 `text_config`、`model_dtype: torch.dtype`、`is_multimodal: bool`、`eos_token_ids: frozenset[int]`、`max_visual_tokens: int=256`、`kv_cache_reserve_bytes: int=1073741824`。后者是计划采用的初始 1 GiB 安全余量，Task 13 实测不足则不能验收；原 Qwen3 不应用 VL 专用限制。`hf_config` 保留完整对象。

- [x] **1. 写 `ConfigTests.test_nested_config_and_limits`：** 本地 VL 配置 `text_config.hidden_size==2048`、`model_dtype==torch.bfloat16`、EOS 为 `{151645,151643}`；顶层 dtype 为 None 也成功；TP=2、VL CUDA Graph、非正预算、超出本期上限的配置明确 `ValueError`，未知 model_type 明确拒绝。原 Qwen3 配置仍可构造。

```python
self.assertEqual(config.text_config.hidden_size, 2048)
self.assertEqual(config.model_dtype, torch.bfloat16)
self.assertEqual(config.eos_token_ids, frozenset({151645, 151643}))
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_config.py -m "not gpu" -q`，collection通过后失败。实际RED为生产解析访问错误的顶层字段，尚未到达nested数值断言；同时捕获不支持的model_type未拒绝。偏离原预期的具体失败形式见Task2报告。
- [x] **3. 实现解析。** `__post_init__` 中按 model_type 小分支解析文字字段/GenerationConfig，本期 VL 必须 BF16、`max_num_seqs=1`、`enforce_eager=True`；上限可降低，不默许扩大至未验收范围。明确错误信息包含字段和值；不修改全局默认 dtype。
- [x] **4. green/验收：** 同一CPU pytest命令通过；断言解析到 theta=5000000、sections=[24,20,20]。外层 max_model_len 与 checkpoint 能力取 min；原 Qwen3 配置回归通过。记录初始余量为工程参数，不宣称实机已测。

## Task 3：三轴位置与 Interleaved-MRoPE

**执行记录：** 仅本任务的实现、TDD、CPU/真实GPU及独立Review见[Task3报告](2026-10-30-qwen3-vl-task3-results.md)。同设备HF组件采用有限值加逐元素精确相等检查，未采用未批准的宽BF16阈值；不表示引擎Decode接入或M1整体完成。

**前置依赖：** Task 1c CPU oracle、Task 2。

**Files：** Create `nanovllm/layers/mrope.py`、`tests/test_qwen3_vl_mrope.py`。

**Interfaces：** `build_mrope_positions(token_ids: list[int], mm_token_type_ids: Tensor, image_grid_thw: Tensor, spatial_merge_size: int) -> tuple[Tensor,int]` 输出 CPU int64 `[3,L]` 与 delta；纯文本用空 `[0,3]` grid。`Qwen3VLRotaryEmbedding(text_config).forward(positions: Tensor, q: Tensor, k: Tensor) -> tuple[Tensor,Tensor]` 消费 `[3,N]` 与 `[N,heads,head_dim]`。图像列表边界来自 grid；不在该数学函数内硬编码只准一图，入口限制由 Task 9 负责。

- [x] **1. 写 `MRopeTests.test_interleaved_axes` 和 `test_prompt_delta`。** 三轴不同的输入与固定 HF 比较；具体断言：half-dim 的 H 索引 `1:60:3`，W `2:60:3`，其余 T；纯文本 delta=0。合并后 2×3 图片网格、前2/后3文字 token 应得到 L=11、delta=-3；KV 长度仍为11。图片前后标签按实际 text token 计算。

```python
positions, delta = build_mrope_positions(ids, types, torch.tensor([[1, 4, 6]]), 2)
self.assertEqual(tuple(positions.shape), (3, 11))
self.assertEqual(delta, -3)
self.assertTrue(torch.equal(positions, hf_positions))
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_mrope.py -m "not gpu" -q`，先通过collection，再由具体数值/状态断言失败构成red。
- [x] **3. 实现两个接口。** 位置按图文段顺序累计；频率 FP32 计算，cos/sin 转为 Q dtype 后按 HF 文字公式旋转。检查负维度、grid 消费不足/剩余和模态长度错配，不修改原一维 RotaryEmbedding。
- [x] **4. green/验收：** 所属CPU pytest模块通过；`.venv/bin/python -m pytest tests/test_qwen3_vl_mrope.py::MRopeGPUTests -m gpu --require-gpu -q` 检查 BF16 Q/K、非连续输入、首次 Decode `[L+delta]*3` 与连续步数。整数结果精确一致；当前同设备同dtype HF基础算子组件浮点采用更严格的有限值加精确相等，未使用未批准阈值，不推广到跨kernel或后续组件。

## Task 4：VL 权重映射与严格加载审计

**执行记录：** 实现、TDD、625-key header审计、旧loader回归及独立Review修复见[Task4报告](2026-10-30-qwen3-vl-task4-results.md)。本任务不表示完整VL权重已经实载或M1整体完成。

**前置依赖：** Task 1；测试用小模型不依赖完整 VL 构造。

**Files：** Modify `nanovllm/utils/loader.py:12`；Create `tests/test_qwen3_vl_loader.py`。

**Interfaces：** 保留 `load_model(model, path)` 原 Qwen3 接口。新增 `load_qwen3_vl_weights(model: nn.Module, path: str, *, text_only: bool=False) -> WeightLoadReport`；report 字段 `loaded_keys`、`skipped_keys`、`tied_aliases` 为字符串列表，失败抛含 missing/unexpected/shape/shard 信息的异常。

- [x] **1. 写 `LoaderTests.test_packed_shards_and_aliases`。** 用临时 safetensors 和小 nn.Module：缺少 K shard、重复映射、shape 错误、unexpected key 必须失败；vision `attn.qkv` 不被语言 packed 规则误伤；合法 tied embedding/head 只加载一次。text_only 仅允许跳过 `model.visual.`。

```python
with self.assertRaisesRegex(ValueError, "missing.*k_proj"):
    load_qwen3_vl_weights(tiny_model, checkpoint_missing_k, text_only=True)
self.assertTrue(all(key.startswith("model.visual.") for key in report.skipped_keys))
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_loader.py -m "not gpu" -q`，接口可导入后，缺K shard未被拒绝或合法tied alias被拒绝的行为断言失败。
- [x] **3. 实现 loader/report。** 模型内部使用 checkpoint 命名空间 `model.language_model`/`model.visual`，仅语言 q/k/v、gate/up 做 packed 映射；按目标参数与每个 packed 子 shard 记录覆盖率，不能只计总 tensor 数。逐 tensor 加载释放，不缓存全量 state_dict。
- [x] **4. green/验收：** CPU pytest通过；实际 header 625 keys 全部有明确归属。Task 5/11 再用真实模型完成实载验收；此任务只声称映射/审计工具通过，不声称全模型已加载成功。原 Qwen3 loader 行为保持。

## Task 5：VL 文本 decoder 与 tied LM head

**执行记录：** 文本实现、真实语言权重实载、TDD、CPU/GPU及独立Review见[Task5报告](2026-11-01-qwen3-vl-task5-results.md)。VL投影继承原packed参数/loader，分别执行Q/K/V、gate/up GEMM以对齐HF舍入顺序；原共享线性层不改。使用独立固定HF FlashAttention2 reference进行同后端零容差验收，旧SDPA Oracle仅诊断，未批准宽阈值未使用。性能影响留Benchmark，不表示Runner/Decode/视觉接通或M1整体完成。

**前置依赖：** Tasks 2–4，GPU 部分依赖 Task 1c GPU 门槛。

**Files：** Create `nanovllm/models/qwen3_vl.py`、`tests/test_qwen3_vl_text.py`。

**Interfaces：** `Qwen3VLForConditionalGeneration(config, *, text_only: bool=True)`；内部 `model.language_model`，`lm_head` 与 embedding 绑定同一 Parameter。`forward(input_ids: Tensor, positions: Tensor, *, visual_features: tuple[Tensor,list[Tensor]]|None=None, visual_mask: Tensor|None=None) -> Tensor[N,2048]`；`compute_logits(hidden_states: Tensor) -> Tensor` 保持原末位置契约。此阶段 visual_features 非 None 显式报未接通，后续 Task 12 实现。

- [x] **1. 写 `TextTests.test_residual_order_and_tie`。** 单 rank 小配置验证 head 与 embedding 为同一 Parameter；固定输入先对齐单 decoder layer 的两个 residual/norm，再对齐整模型。额外构造可区分“延迟 residual”与“立即 BF16 add”的输入。

```python
self.assertIs(model.lm_head.weight, model.model.language_model.embed_tokens.weight)
assert_aligned("text_layer_0", actual_layer_0, hf_layer_0, tolerances)
self.assertEqual(tuple(last_logits.shape), (1, 151936))
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_text.py -m gpu --require-gpu -q`，collection通过后，模型已可构造但显式residual顺序、tied Parameter或实际hidden/logits断言失败；缺模块先补最小可导入接口，不计red。
- [x] **3. 实现 VL text attention/MLP/decoder/wrapper。** 复用 QKVParallelLinear、MergedColumnParallelLinear、RowParallelLinear、RMSNorm 和 Attention；Q/K norm 后使用 Task 3，显式 `x = residual + attn`、`x = residual + mlp`，最终 norm。不复制修改原 Qwen3 decoder，也不误用其 fused add norm。VL专用投影子类保留packed存储与加载，独立GEMM计算以保留HF舍入点；依据及性能成本见Task5报告。
- [x] **4. green/验收：** GPU pytest通过 HF 单层及完整目标文本模型对照；实载报告只跳过视觉参数，无其他漏项。Prefill 最后 logits、选定层 hidden、各层旋转后 K/V 对固定独立HF FlashAttention2 reference有限值且精确相等。此同后端门槛不代表旧SDPA阈值获批或跨后端一致；语言用同一 checkpoint，不以 Qwen3-0.6B 代替。

## Task 6：确定性 greedy 与 VL 双 EOS

**执行记录：** 从中断继续完成；实现、TDD、CPU/真实GPU及独立Review见[Task6报告](2026-11-02-qwen3-vl-task6-results.md)。本任务只处理采样/EOS语义，不表示完整VL Runner或生成链路接通。Review无Critical/Important，1项非阻塞正温度测试覆盖增强建议记录为延期Minor。

**前置依赖：** Task 2；GPU sampling 对照依赖 Task 1。

**Files：** Modify `nanovllm/sampling_params.py:5`、`nanovllm/layers/sampler.py:5`、`nanovllm/engine/scheduler.py:10,81`、`nanovllm/engine/llm_engine.py:32`；Create `tests/test_qwen3_vl_sampling.py`。

**Interfaces：** `SamplingParams(temperature=0.0, max_tokens=...)` 表示 greedy；Sampler.forward 签名不变。Scheduler 的结束条件使用 Config.eos_token_ids（VL）或原 eos（Qwen3），正温度原算法保留。

- [x] **1. 写 `SamplingTests.test_zero_temperature_and_eos`：** logits `[1,3,2]` 在 temperature=0 恒返回1；同分数取 argmax 首项；负温度/非正 max_tokens 报错。两个 EOS 分别结束，ignore_eos 忽略两个，max_tokens=1 仍在首个生成 token 后结束。

```python
sp = SamplingParams(temperature=0.0, max_tokens=1)
self.assertEqual(Sampler()(torch.tensor([[1., 3., 2.]]), torch.tensor([sp.temperature])).item(), 1)
# seq 已由 fixture 调度为 running，且下一次 postprocess 追加首个生成 token。
scheduler.postprocess([seq], [151643], True)
self.assertTrue(seq.is_finished)
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_sampling.py -m "not gpu" -q`，collection通过后，合法temperature=0被错误拒绝而行为断言失败。
- [x] **3. 实现 greedy mask 分支和 EOS 集合。** 不对零温度行做除法，不用极小温度近似；LLMEngine 不用 tokenizer 单 EOS 覆盖 VL generation_config 的集合。拒绝 NaN/Inf temperature。
- [x] **4. green/验收：** 所属CPU pytest模块和 `.venv/bin/python -m pytest tests/test_qwen3_vl_sampling.py::SamplingGPUTests -m gpu --require-gpu -q` 通过；全正温度批次固定 seed 与原实现结果一致；混合零/正温度行各自合法且无除零。无新增 top-k/top-p 功能。

## Task 7：单活动请求、容量准入与 prefix cache 隔离

**前置依赖：** Tasks 2、6。提前实现本项，避免 P1 就暴露不安全的 VL 调度入口。

**Files：** Modify `nanovllm/engine/block_manager.py:28,58,75,110`、`nanovllm/engine/scheduler.py:10,22,25,75,81`；Create `tests/test_qwen3_vl_scheduler.py`。

**Interfaces：** `BlockManager(num_blocks, block_size, *, enable_prefix_cache: bool=True)`；关闭时查找与 hash 发布均停用，其他接口不变。`Scheduler.abort(seq: Sequence) -> None` 从队列移除、释放该 seq 块、置 FINISHED；重复 abort 安全。`Scheduler.add` 在 VL 分支校验预算。

- [x] **1. 写 `SchedulerTests.test_serial_admission_and_capacity`。** 两个等待请求：第二个在第一个 Decode/完成前不得占块；第一个 Prefill 必须调度全 L。`ceil((L+max_tokens)/256)` 超池或超长度、L 超 max_num_batched_tokens 均报错且 free/used/ref_count 不变；不能死循环。`test_no_prefix_lookup_or_publish` 用相同长 token IDs 连续两请求，断言 cached_tokens=0、hash map 为空。

```python
scheduled, is_prefill = scheduler.schedule()  # 第一个请求已完成 Prefill，第二个仍等待。
self.assertEqual(scheduled, [first])
self.assertFalse(is_prefill)
self.assertEqual(second.block_table, [])
self.assertEqual(scheduler.block_manager.hash_to_block_id, {})
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_scheduler.py -m "not gpu" -q`，现有 waiting 优先策略应暴露问题。
- [x] **3. 实现 VL 小分支。** add 时以池总块数判断请求是否最终可容纳，不能因为前一请求暂占块就拒绝合法排队；真正准入时检查可用块数。有 running 就 Decode，不准入 waiting；仅一个请求时根据完整未来长度证明追加容量，仍按需分配块。不进入抢占分支；不可能追加时抛清晰 invariant 错误并由上层 abort。原 Qwen3 仍走原 chunk/prefix/preempt 路径。
- [x] **4. green/验收：** CPU pytest通过含 255/256/257、连续排队与重复 abort；加原 Qwen3 prefix hit、chunked prefill、抢占状态的 CPU 回归。断言所有完成/异常请求 ref_count 归零，空队列 is_finished=True。


**2026-11-02 执行结果：** Task7完成；CPU17/17，完整CPU160通过、15未选，旧Qwen3真实GPU2通过、9未选，最终FAILED/ERROR/SKIPPED均0。独立Review无未解决Critical/Important、无新增Minor；最小CI导入问题已修复。未暂存/提交/推送，止于Task7；详见[Task7报告](2026-11-02-qwen3-vl-task7-results.md)。

## Task 8：Runner 接通纯文本与真实 paged KV

**共同前置依赖（子任务另列）：** Tasks 1–7；GPU smoke 必须通过。

**Files：** Modify `nanovllm/engine/model_runner.py:17,91,103,129,172,195,214`、`nanovllm/engine/llm_engine.py:17,37,49`；Create `tests/test_qwen3_vl_runner.py`。

**Interfaces：** Runner 按 model_type 选择模型；此阶段内部构造 VL `text_only=True`，不增加用户可切换的半成品模式。新增 `kv_block_bytes(text_config, block_size: int, dtype: torch.dtype) -> int`、`select_kv_block_count(config: Config, available_bytes: int) -> int` 在 runner 模块；`prepare_prefill/prepare_decode` 返回形式不变，VL positions 为 `[3,N]`，纯文本 delta=0。`exit()` 幂等。

**原任务共同要求 1：写 `RunnerTests.test_text_prefill_decode_and_blocks`。** 单请求实际目标 checkpoint 比最后 logits、teacher-forced 32 步、greedy 和 KV gather；block tables 用非连续物理块来检验映射。`BudgetTests` 断言 block_bytes=29360128，4096 长度 cap=16 块，预算小于该 cap 所需空间时初始化失败而非偷偷降容量。

```python
self.assertEqual(kv_block_bytes(config.text_config, 256, torch.bfloat16), 29360128)
self.assertEqual(select_kv_block_count(config, 16 * 29360128), 16)
with self.assertRaises(ValueError):
    select_kv_block_count(config, 16 * 29360128 - 1)
```

`available_bytes` 的定义是已扣除非KV常驻、profile峰值增量和安全余量之后的净KV预算，助手不能再次重复扣减。目标VL纯文本的重复调用和失败清理在本模块提供worker；原模型smoke复用1b，不再另建旧模型对照worker。

**原任务共同要求 3：实现模型分支、dtype、KV 与异常收尾。** KV 只绑定语言 Attention，断言数量等于28，形状 `[2,28,num_blocks,256,8,128]`；VL 池封顶16块，原 Qwen3 分配方式保留。本阶段文本 warmup 使用完整最大文本长度，并保留 Config 的安全余量；Task 13 替换为图文联合 profile。`__init__` 用 try/finally 恢复默认 device/dtype，失败释放已创建资源；`run` finally reset_context，`LLMEngine.step` 可恢复异常调用 abort 后重抛。
**原任务共同要求 4：green/验收：** Runner GPU pytest类 通过 255/256/257 和追加新块，受影响共享组件测试及1b单一原模型smoke通过；所有255/256/257数值对齐只使用目标Qwen3-VL-2B。测试注入 warmup 异常后 default device/dtype 恢复；正常 exit 调用两次不报错；单请求最多16块。P1 完成记录明确视觉尚未接通。

以下子任务替代原任务执行步骤；上面的接口、有效断言及数值要求共同适用，不把父节另计完成任务。

### Task 8a：模型选择、dtype与KV预算纯函数

**前置依赖：** 1a、1b、2–7。

**Files：** Modify `nanovllm/engine/model_runner.py`；Test `tests/test_qwen3_vl_runner.py`。

**Interfaces：** 消费Config；产出父节kv_block_bytes/select_kv_block_count；available_bytes为扣除余量后的净预算。

- [x] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py --collect-only -q`，节点包含`BudgetTests`；缺模块先补可导入接口并重试，不计red。
- [x] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py::BudgetTests -m "not gpu" -q`，测试断言：

```python
self.assertEqual(kv_block_bytes(config.text_config, 256, torch.bfloat16), 29360128)
self.assertEqual(select_kv_block_count(config, 16 * 29360128), 16)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [x] **3. 最小实现**：nested dtype和model_type分支；预算不足直接ValueError；明确16只对应max_model_len4096/block256，不硬编码通用16。
- [x] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；CPU预算/异常边界断言全过，原Qwen3分支保留。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

**2026-11-07 最终状态：** Task8a验收完成。本任务CPU23项通过；用户明确批准正式fixture仅一行Transformers注释指纹更新，原始失败和基准证据保留；正式旧模型GPU2项全通过（12.27s），更新后完整CPU183项通过（10.42s），FAILED/ERROR/SKIPPED均0。Review原唯一Important由该授权更新和正式回归关闭，原始With fixes结论保留；无代码修复或第二次Review。未修改token、数值阈值或断言。详见[Task8a报告](2026-11-07-qwen3-vl-task8a-results.md)。止于8a，不开始8b，不暂存/提交/推送。

### Task 8b：纯文本完整Prefill与分页KV绑定

**前置依赖：** 8a、1cGPU门槛。

**Files：** Modify `nanovllm/engine/model_runner.py`；Test `tests/test_qwen3_vl_runner.py`。

**Interfaces：** 消费VL text_only模型和三轴相同positions；产出最后logits及28层真实KV。

- [x] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py --collect-only -q`，节点包含`PrefillRunnerTests`；缺模块先补可导入接口并重试，不计red。
- [x] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py::PrefillRunnerTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(bound_attention_layers, 28)
assert_aligned("text_prefill_logits", actual_logits, hf_logits, tolerances)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [x] **3. 最小实现**：绑定仅语言Attention；验证非连续slot/block table；新增测试使用现有源码实际Attention/store内核。
- [x] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；短prompt和255/256/257 Prefill logits与选定KV达到HF阈值。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

**2026-11-07 执行结果：** Task8b验收完成。真实Runner4096完整warmup、28层语言KV绑定、16块BF16池；24/255/256/257每例57浮点组件与独立HF FA2精确相等，未使用宽BF16阈值。CPU183通过，快速31通过；独立Review无阻塞项。中断前组合曾因HF worker超时及旧模型容量断言3失败，日志保留；恢复后不改代码/基准/断言/超时，同一正式GPU组合7项通过、FAILED/ERROR/SKIPPED均0（26.98s）。原失败历史资源原因未能确定，不冒称生产修复。详见[Task8b报告](2026-11-07-qwen3-vl-task8b-results.md)。不开始8c，不暂存/Commit/Push。

### Task 8c：纯文本Decode与Runner初始化收尾

**前置依赖：** 8b、6–7。

**Files：** Modify `nanovllm/engine/model_runner.py`、`nanovllm/engine/llm_engine.py`；Test `tests/test_qwen3_vl_runner.py`。

**Interfaces：** 消费8b缓存；产出可复现32步文本Decode和幂等exit；不预先接通图像。

- [x] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py --collect-only -q`，节点包含`TextDecodeRunnerTests`；缺模块先补可导入接口并重试，不计red。
- [x] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py::TextDecodeRunnerTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(actual_greedy_tokens, hf_greedy_tokens)
self.assertEqual(default_dtype_after_failure, default_dtype_before)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [x] **3. 最小实现**：Decode context/slot映射；init/run try/finally恢复默认状态；注入初始化和warmup异常，重复exit安全。
- [x] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；32步teacher forcing/跨页通过；重跑1b受影响共享组件与单一原模型smoke；GPU不可用不能完成P1。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

**2026-11-08 执行结果：** Task8c完成。24/255/256/257-token用例各完成独立Prefill后32次实际Decode输入，114浮点组件/例与HF FA2严格零误差；独立真实Engine排队生成每例32个greedy ID一致，max_tokens=1通过。初始化/prepare/forward/sample异常清理、默认device/dtype恢复、请求abort、外部Gloo组保留及重复exit通过。独立Review初次无阻塞项；之后自查发现Important参数循环引用问题，以完整模型弱引用测试RED1失败、生产exit回收修复、GREEN1通过关闭；未做第二次Review。修复后完整CPU186通过（10.04s）、快速31通过（1.76s）、正式GPU组合11通过（56.44s），FAILED/ERROR/SKIPPED均0。退出前后参数/KV释放在测试额外GC前验证。详见[Task8c报告](2026-11-08-qwen3-vl-task8c-results.md)。P1纯文本范围完成；图像、Processor、视觉融合及进程级CUDA错误恢复未验证，不开始Task9，不暂存/Commit/Push。


## Task 9：Processor 输入契约与请求级多模态状态

**前置依赖：** Tasks 2、3、7、8。

**Files：** Create `nanovllm/utils/multimodal.py`、`tests/test_qwen3_vl_input.py`；Modify `nanovllm/engine/sequence.py:18`、`nanovllm/engine/llm_engine.py:43,60`。

**Interfaces：** `PreparedMultimodalPrompt` 字段：`token_ids: list[int]`、`pixel_values: Tensor[P,1536]`、`image_grid_thw: Tensor[I,3]`、`mm_token_type_ids: Tensor[L]`（CPU）。`prepare_multimodal(messages: list[dict], processor, *, max_visual_tokens: int=256) -> PreparedMultimodalPrompt`；`validate_multimodal(prompt, hf_config, *, max_visual_tokens: int) -> None`。Sequence 新增 keyword-only `multimodal: PreparedMultimodalPrompt|None=None`，拥有 `prompt_positions: Tensor|None`、`rope_delta: int=0`、`visual_mask: Tensor|None`，CPU 数据复制/拥有，不引用可被调用者修改的张量。位置由 Task 3 计算。

- [x] **1. 写 `InputTests.test_processor_grid_and_validation`：** 方图/非方图与 HF 的 IDs、grid、pixels、模态类型一致；T!=1、奇数 H/W、像素数错、image 数量不符、空 prompt、孤立占位符、video、多图、padding、超256视觉 token 均在分配 KV 前报错。`test_request_owns_cpu_state` 修改原输入后 Sequence 内容不变。

```python
self.assertEqual(prepared.image_grid_thw.tolist(), [[1, 16, 24]])
self.assertEqual(tuple(prepared.pixel_values.shape), (384, 1536))
self.assertEqual(prepared.token_ids.count(hf_config.image_token_id), 96)
with self.assertRaises(ValueError):
    validate_multimodal(prompt_with_bad_grid, hf_config, max_visual_tokens=256)
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_input.py -m "not gpu" -q`，先通过collection，再由具体数值/状态断言失败构成red。
- [x] **3. 实现 CPU 预处理和校验。** 按固定 HF 版本把处理参数放入 `processor_kwargs`；像素/视觉 token 上限明确传入并二次检查，不随意猜图像 resize 后 token 数。入口支持原 str/list[int] 和 Prepared；batch 内可排队多请求，但每请求最多一图。此阶段识别合法 Prepared 后仍明确报“模型图像执行尚未接通”，Task 13 才解除，禁止丢弃图片继续走纯文本。
- [x] **4. green/验收：** CPU pytest通过；纯文本 image/video 孤立 token 也拒绝。grid 接口 I 可变但首期校验 I=1；Sequence 不新增 GPU 张量或 TP 序列化负担，原文本请求可继续使用。

**2026-11-08 执行结果：** Task9完成。真实HF Processor方图/矩形/上限图的CPU pixels、IDs、grid、模态类型与直接调用和持久化只读Oracle精确一致；Sequence复制请求数据，三轴位置/delta与HF一致，受控别名及位置扰动均触发有效断言。输入83项通过，完整CPU269通过（9.56s）、快速31通过（1.68s）、既有文本GPU11通过（52.83s），最终FAILED/ERROR/SKIPPED均0。独立Review发现1项Important：字符串/别名来源的动画可被HF静默截为首帧；新增来源矩阵RED11失败后，明确只接受canonical image字段中的已解码静态PIL对象，路径/URL/base64及别名提前报错，完整回归关闭问题，无二次Review。Sequence保持原TP序列化AST；目标merge2以外布局明确拒绝。合法图片生成仍NotImplemented、未进入Scheduler/KV；没有实现视觉网络/融合或图像Decode。详见[Task9报告](2026-11-08-qwen3-vl-task9-results.md)。不开始Task10、不暂存/Commit/Push。

## Task 10：视觉 patch embedding 与两种视觉位置编码

**前置依赖：** Tasks 1、2、4、9（CPU fixture）；GPU对照依赖 P0。

**Files：** Create `nanovllm/models/qwen3_vl_vision.py`、`tests/test_qwen3_vl_vision_positions.py`。

**Interfaces：** `Qwen3VLVisionPatchEmbed(vision_config).forward(pixel_values: Tensor) -> Tensor[P,1024]`；`vision_position_ids(grid_thw: Tensor, merge_size: int) -> Tensor[P,2]`；`interpolate_vision_positions(weight: Tensor, grid_thw: Tensor, merge_size: int) -> Tensor[P,1024]`；`Qwen3VLVisionRotaryEmbedding(vision_config).forward(position_ids: Tensor) -> tuple[Tensor,Tensor]` 输出 `[P,64]` FP32 cos/sin。

- [x] **1. 写 `VisionPositionTests.test_patch_order_interpolation_rope`：** 4×4 可辨认 patch 保持 Processor 的2×2块顺序；48×48表的原尺寸、非方网格、插值边界与 HF 对齐；theta=10000，head_dim=64，频率 `[H,W,H,W]`，不是语言三轴。单图重复时间帧只由 Processor 产生，模型不得再次复制。

```python
self.assertEqual(vision_position_ids(torch.tensor([[1, 4, 4]]), 2)[:4].tolist(),
                 [[0, 0], [0, 1], [1, 0], [1, 1]])
assert_aligned("vision_absolute_position", actual_pos_embeds, hf_pos_embeds, tolerances)
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py -m "not gpu" -q`，先通过collection，再由具体数值/状态断言失败构成red。
- [x] **3. 实现 Conv3d/插值/二维 RoPE。** Conv kernel/stride=(2,16,16)，含 bias；按四邻点 FP32 加权、align_corners=True/border clamp 和 HF cast 顺序。位置坐标与输入顺序一致，T 按 grid 重复；视觉旋转 Q/K 在 FP32 后回原 dtype。
- [x] **4. green/验收：** 所属CPU pytest模块与 `.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py::VisionPositionGPUTests -m gpu --require-gpu -q` 通过；单独加载视觉对应权重，patch 输出和加绝对位置后的 hidden 分别对齐。长宽交换的图必须覆盖，不能只测正方形。

**2026-11-09 执行结果：** 新增vision primitives及测试，未修改此前生产模块。真实checkpoint只读加载patch weight/bias与pos_embed三项权重，七个图像用例（含16×24/24×16）每例七浮点组件max_abs/mean_abs/relL2全0，位置精确一致；无宽BF16放行。CPU11通过、完整CPU280通过（10.00s）、快速31通过（1.67s）、最终正式GPU1通过（23.63s），FAILED/ERROR/SKIPPED均0。独立Review无Critical/Important/Minor，独立CPU11通过，未独立重跑GPU。流程限制：初始RED仅NotImplemented/schema过渡，三项数值/布局扰动在实现后触发有效断言，不能倒写为最初的数值RED。详见[Task10报告](2026-11-09-qwen3-vl-task10-results.md)。Task9及Oracle保护哈希未变，保留其现有Git状态；无ViT/merger/融合/图像执行/TP扩展，不开始Task11、不暂存/Commit/Push。

## Task 11：ViT blocks、主 merger 和三组 DeepStack merger

**共同前置依赖（子任务另列）：** Tasks 4、10。

**Files：** Modify `nanovllm/models/qwen3_vl_vision.py`；Create `tests/test_qwen3_vl_vision.py`。

**Interfaces：** `Qwen3VLVisionModel(vision_config).forward(pixel_values: Tensor, grid_thw: Tensor) -> tuple[Tensor,list[Tensor]]`：主特征 `[V,2048]` 与三项 `[V,2048]`。`Qwen3VLVisionPatchMerger(vision_config, *, use_postshuffle_norm: bool)`；模块名称沿用 HF 属性，避免额外权重映射表。

**原任务共同要求 1：写 `VisionTests.test_blocks_mergers_and_segment_isolation`：** 比第0/5/11/17/23层、主输出和三组 DeepStack；main norm 维度1024、DeepStack norm4096。底层视觉函数接受两个 grid 的测试输入，改变图2像素不应改变图1特征，证明 attention 分段；这不是开放产品多图入口。

```python
self.assertEqual(tuple(main.shape), (96, 2048))  # 单个384×256 fixture
self.assertEqual(len(deepstack), 3)
self.assertTrue(all(tuple(x.shape) == (96, 2048) for x in deepstack))
torch.testing.assert_close(first_image_before, first_image_after, rtol=0, atol=0)
```

**原任务共同要求 3：实现24层非因果 SDPA ViT。** QKV bias、LayerNorm eps=1e-6、GELU tanh MLP；main pre-shuffle norm、DeepStack post-shuffle norm和普通GELU；层5/11/17输出在各自独立 merger 中处理。按每图/时间组切分 attention，dropout=0，不设置 k_cache/v_cache 属性。
**原任务共同要求 4：green/验收：** GPU pytest使用方图、非方图、受控真实图和最大 P=1024 通过逐层阈值；视觉单模块峰值报告落盘。checkpoint 全部视觉 keys 消费，无伪造/随机初始化参数。输出顺序与视觉占位符顺序一致。

以下子任务替代原任务执行步骤；上面的接口、有效断言及数值要求共同适用，不把父节另计完成任务。

### Task 11a：非因果视觉Attention与单个ViT block

**前置依赖：** 10、4、1c。

**Files：** Modify `nanovllm/models/qwen3_vl_vision.py`；Create `tests/test_qwen3_vl_vision_block.py`。

**Interfaces：** Qwen3VLVisionAttention.forward(hidden_states,cu_seqlens,position_embeddings)->Tensor[P,1024]；block同输入输出。

- [x] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_vision_block.py --collect-only -q`，节点包含`VisionBlockTests`；缺模块先补可导入接口并重试，不计red。
- [x] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_vision_block.py::VisionBlockTests -m gpu --require-gpu -q`，测试断言：

```python
assert_aligned("vision_block_0", actual_block, hf_block, tolerances)
torch.testing.assert_close(first_image_before, first_image_after, rtol=0, atol=0)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [x] **3. 最小实现**：实现biased QKV、分图SDPA、LayerNorm和GELU-tanh MLP，dtype顺序遵HF；两段attention互不串图。
- [x] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；单block输出、旋转后Q/K和分段隔离均过；不是仅shape通过。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

**2026-11-09 执行结果：** 单个ViT block及分段非因果SDPA完成；严格加载block0全部12keys及patch/absolute3keys，HF/native分进程串行。真实GPU初始数值RED为block0 max_abs18.625>0，最终GPU1通过42.34s，十例九组件max_abs/mean_abs/relL2全部0，图/时间组隔离精确。最终完整CPU290通过9.84s、快速31通过1.68s、所属CPU21通过1.87s；Task10 GPU回归1通过23.21s。受控错误分段/因果/漏旋转均触发数值失败。独立Review唯一Important（新增torch测试未隔离轻量CI）经blocked-import快速RED10failed31passed→GREEN31passed修复；无遗留Critical/Important/Minor，无二次Review。见[Task11a报告](2026-11-09-qwen3-vl-task11a-results.md)。仅技术验收，未Commit/Push，不开始11b。

### Task 11b：主merger及DeepStack post-shuffle merger

**前置依赖：** 11a、4。

**Files：** Modify `nanovllm/models/qwen3_vl_vision.py`；Create `tests/test_qwen3_vl_vision_merger.py`。

**Interfaces：** 消费固定中间hidden；Qwen3VLVisionPatchMerger(...use_postshuffle_norm)->Tensor[V,2048]。

- [x] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_vision_merger.py --collect-only -q`，节点包含`VisionMergerTests`；缺模块先补可导入接口并重试，不计red。
- [x] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_vision_merger.py::VisionMergerTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(tuple(main_merger.norm.weight.shape), (1024,))
self.assertEqual(tuple(deep_merger.norm.weight.shape), (4096,))
assert_aligned("deep_merger", actual_deep, hf_deep, tolerances)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [x] **3. 最小实现**：主norm先于reshape，DeepStack norm后于reshape；merger普通GELU；同一输入显式验证两路径不混用。
- [x] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；两种merger分别加载真实权重并对齐HF；故意交换norm顺序会被断言发现。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

**2026-11-09 执行结果：** HF命名PatchMerger追加完成，主路1024维norm先于reshape，DeepStack4096维norm后于reshape，普通GELU；固定HF hidden层23/5/11/17及24真实keys严格加载。初始GPU有效数值RED主路及三路DeepStack均失败；最终正式GPU1通过13.58s，4mergers×6例的96计算输出+24输入全部三指标0。CPU2通过、完整CPU292通过9.90s、所属视觉CPU23通过1.85s、快速31通过1.69s、禁止模型栈导入CI31通过1.65s；前置Task10/11a GPU2通过64.85s。三项合法维度扰动触发有效数值失败。独立Review规格/质量PASS、无Critical/Important/Minor，独立CPU2通过并复算120保存Tensor，不独立重跑GPU。旧vision前缀及其他保护哈希未变，Oracle只读，见[Task11b报告](2026-11-09-qwen3-vl-task11b-results.md)。未暂存/Commit/Push，不开始11c。

### Task 11c：24层视觉堆叠与三路特征提取

**前置依赖：** 11a、11b。

**Files：** Modify `nanovllm/models/qwen3_vl_vision.py`；Test `tests/test_qwen3_vl_vision.py`。

**Interfaces：** 消费patch/positions/blocks/mergers；产出父节Qwen3VLVisionModel tuple(main,list[3])。

- [x] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_vision.py --collect-only -q`，节点包含`VisionTests`；缺模块先补可导入接口并重试，不计red。
- [x] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_vision.py::VisionTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(len(deepstack), 3)
self.assertTrue(all(tuple(x.shape) == (96, 2048) for x in deepstack))
assert_aligned("vision_layer_17", actual_layer_17, hf_layer_17, tolerances)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [x] **3. 最小实现**：提取零起点5/11/17完整block输出，主路取末层；逐tensor实载视觉权重并审计覆盖。
- [x] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；方图/非方图/真实图/上限P1024逐层通过，输出顺序与占位符契合，记录视觉峰值。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。


**2026-11-09 执行结果：** 完整24层视觉模型追加完成，消费已验收patch/绝对位置/轴向RoPE/分段Attention/merger，提取完整零起点5/11/17输出。全部315视觉keys实际流式加载，310语言keys明确排除，HF/native独立进程串行；12用例全部24层及四路输出348项三指标全零，图/时间组隔离精确，矩形四路均[96,2048]。初始GPU有效数值RED vision_layer_17 max_abs331.75>0，最终正式GPU1通过57.12s；完整CPU298通过10.00s、视觉CPU29通过1.93s、快速31通过1.68s，M2b前置三项视觉GPU+旧Qwen3两项GPU共5通过87.23s。三项选层/返回顺序/跨图Attention扰动触发数值失败。独立Review规格/质量PASS、Critical0/Important0/Minor0，独立CPU6通过并复算348保存Tensor，不独立重跑GPU。旧组件前缀和54项保护哈希未变，Oracle只读。见[Task11c报告](2026-11-09-qwen3-vl-task11c-results.md)。Task11及M2b本地技术门禁完成，未暂存/Commit/Push，Task12未开始。

## Task 12：主视觉融合和 DeepStack 注入

**前置依赖：** Tasks 5、9、11。

**Files：** Modify `nanovllm/models/qwen3_vl.py`；Create `tests/test_qwen3_vl_fusion.py`。最小附加范围：更新 `tests/test_qwen3_vl_text.py` 中原“视觉不可用”的过渡测试为paired输入提前校验；其余既有测试不变。

**Interfaces：** 实现 Task 5 已预留的 visual_features/visual_mask；`encode_images(pixel_values: Tensor, grid_thw: Tensor) -> tuple[Tensor,list[Tensor]]` 在 VL wrapper 调用自实现视觉模型。构造 `text_only=False` 时注册 `model.visual`；生产最终将只使用该完整形式。

- [x] **1. 写 `FusionTests.test_replacement_and_deepstack`：** 主特征替换 image_pad embedding；非视觉位置 embedding 不变。仅 decoder0/1/2完整 residual 之后向视觉位置分别加三路特征；第3层不再注入。mask/V/DeepStack 长度或隐藏维度不符均报错。对比 HF 前三层注入后的完整 hidden 与最终 norm。

```python
torch.testing.assert_close(fused[visual_mask], main, rtol=0, atol=0)
torch.testing.assert_close(fused[~visual_mask], token_embeds[~visual_mask], rtol=0, atol=0)
assert_aligned("deepstack_after_layer_2", actual_after_layer_2, hf_after_layer_2, tolerances)
```

- [x] **2. red：** `.venv/bin/python -m pytest tests/test_qwen3_vl_fusion.py -m gpu --require-gpu -q`，collection通过后由融合替换/DeepStack实际输出断言失败；单纯NotImplementedError只算过渡状态，不足以完成red证据。
- [x] **3. 实现融合。** 顺序按 mask 对应主和 DeepStack 特征；不在尚未合并的 residual 或 norm 前注入。先用 HF 特征 fixture 隔离融合错误，再换 Task 11 自实现特征重跑；两种测试均须保留清晰标记。
- [x] **4. green/验收：** GPU pytest两组对照都通过；完整 checkpoint 加载625 tensor，无视觉 skip，embedding/head共享。单图 Prefill 最后 logits 达阈值，但此任务不声称 Runner 的图像请求生命周期已接通。

**2026-11-10 实施结果（待独立Review）：** 完整wrapper加载625真实源tensor、无visual skip、embedding/head同Parameter；HF固定特征与native自编码两模式各六图，120项对照三指标全0（其中24项fixed视觉输入复制，96项native计算）。首次GPU有效RED2failed39.18s；最终CPU313passed11.69s、GPU2passed51.16s、既有text/visionGPU8passed141.88s。mask/order/placement三项进程内错误均有效数值失败，生产源码未改。快速与禁止模型栈CI各31通过，保护SHA其余54项不变。详见[Task12报告](2026-11-10-qwen3-vl-task12-results.md)。独立Review待安排；无Runner图像接通/Task13/Stage/Commit/Push。

**2026-11-10 执行结果：** model.visual及encode_images接通；主视觉mask替换和decoder0/1/2完整residual后注入，三路输入全部提前校验。625源tensor实载无skip且head/embedding共享。两具名GPU模式六图对照2通过51.16s，96项计算结果与24项固定输入区分记录、三指标全零；CPU313通过11.69s，quick31通过，旧text/visionGPU8通过141.88s。有效GPU RED为fused失配9.8125，错mask/order/placement三项真实数值扰动失败。独立Review PASS，无Critical/Important/Minor；真实GPU未由Reviewer重复运行。唯一额外旧文件为更新过时text视觉未接通断言。见[Task12报告](2026-11-10-qwen3-vl-task12-results.md)。仅模型级验收，Task13/14生命周期未接通，未暂存/Commit/Push。

## Task 13：图文 Runner、视觉 profile 与完整 Prefill

**共同前置依赖（子任务另列）：** Tasks 7–12。

**Files：** Modify `nanovllm/engine/model_runner.py` 的模型构造、warmup、KV分配、run_model/run；Modify `nanovllm/engine/llm_engine.py` 的 Prepared 接入；Create `tests/test_qwen3_vl_prefill.py`。

**Interfaces：** 完整 VL 默认 `text_only=False`，仅单元测试可显式使用 text_only。`run_model(input_ids, positions, is_prefill, *, visual_features=None, visual_mask=None)`；`profile_vl_memory(self) -> dict[str,int]` 返回 peak allocated/reserved、free_after、reserve_bytes，调用 Task 8 预算助手。视觉张量只在当前 run 局部存在，不挂在全局模型状态。

**原任务共同要求 1：写 `PrefillTests.test_real_image_prefill_and_memory`：** 合法 Prepared 经 LLMEngine 到真实模型最后 logits、融合输出及所有28层选定位置KV，与 HF 对齐；positions 与物理 slot 分离。视觉 forward spy 断言调用1次。纯文本在完整模型中仍通过 P1。超过 L/budget/V 或无图片数据时 encoder 不被调用、KV 不改变。

```python
self.assertEqual(vision_forward_spy.call_count, 1)
self.assertEqual(tuple(runner.kv_cache.shape), (2, 28, 16, 256, 8, 128))
assert_aligned("image_prefill_logits", actual_logits, hf_logits, tolerances)
```

**原任务共同要求 3：实现图文路由。** 只有已准入请求把 pixels 搬到 GPU，encode 一次后调用融合，取最后 logits；完整 Prefill 后释放局部 pixels/features。请求 positions/mask 取 `[start:end]` 对应范围，但首期 start必须0且end=L；非完整输入显式拒绝。
**原任务共同要求 4：实现联合 profile。** 分别跑最大纯文本和图文输入，P=1024、语言总预算4096、方图和极端合法长宽比；测完整流程及视觉中间峰值，释放临时张量后按可用预算减峰值增量和 Config 余量，再分配最多16块。缓存已经存在后的真实 Prefill/Decode 峰值还要再验收；不足时启动失败并给预算，不静默缩小已承诺范围。注意 profile 输出的存活张量不能让 `current` 抵消本应预留的激活。
**原任务共同要求 5：green/验收：** GPU pytest通过完整路径；V=256 可处理，V>256 拒绝。通过 profile 后 KV数满足全程预算；GPU峰值、free margin 与输入grid写报告。不得仅用文本 warmup 或视觉单模块峰值宣布整引擎适配完成。

以下子任务替代原任务执行步骤；上面的接口、有效断言及数值要求共同适用，不把父节另计完成任务。

### Task 13a：图像请求路由和完整Prefill

**前置依赖：** 12、9、8c、11c。

**Files：** Modify `nanovllm/engine/model_runner.py`、`nanovllm/engine/llm_engine.py`；Test `tests/test_qwen3_vl_prefill.py`。

**Interfaces：** 消费Prepared CPU状态；产出positions[3,L]、视觉features和完整模型最后logits；使用明确小输入预算。

- [ ] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_prefill.py --collect-only -q`，节点包含`PrefillRoutingTests`；缺模块先补可导入接口并重试，不计red。
- [ ] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_prefill.py::PrefillRoutingTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(vision_forward_spy.call_count, 1)
assert_aligned("image_prefill_logits", actual_logits, hf_logits, tolerances)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [ ] **3. 最小实现**：只在准入后搬pixels；完整模型text_only=False；拒绝chunk输入；对照融合hidden和28层选定KV。
- [ ] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；小图完整引擎Prefill真实对齐，非法输入不调用encoder；暂不宣称最大显存范围已通过。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

### Task 13b：联合profile和KV分配预算

**前置依赖：** 13a、8a。

**Files：** Modify `nanovllm/engine/model_runner.py`；Create `tests/test_qwen3_vl_memory.py`。

**Interfaces：** profile_vl_memory()->dict[str,int]报告peak/free/reserve；select_kv_block_count消费净预算。

- [ ] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py --collect-only -q`，节点包含`MemoryTests`；缺模块先补可导入接口并重试，不计red。
- [ ] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py::MemoryTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertGreaterEqual(measured_free_margin, config.kv_cache_reserve_bytes)
self.assertEqual(runner.kv_cache.shape[2], (config.max_model_len + 255) // 256)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [ ] **3. 最小实现**：覆盖最大文本/图文及合法极端长宽比；清临时张量再计预算；预KV与分配后峰值分测；负预算/不足预算拒绝。
- [ ] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；1GiB建议余量是否充分有实测依据；不足不能标成功或静默缩范围，必须报修订请求；不要求额外租卡。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

### Task 13c：上限Prefill及完整模型文本回归

**前置依赖：** 13b。

**Files：** Test `tests/test_qwen3_vl_prefill.py`；Modify `tests/qwen3_vl_validation.md`；仅必要时修正`nanovllm/engine/model_runner.py`。

**Interfaces：** 消费联合profile和完整模型；产出4096总预算/256视觉token边界证据。

- [ ] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_prefill.py --collect-only -q`，节点包含`PrefillLimitTests`；缺模块先补可导入接口并重试，不计red。
- [ ] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_prefill.py::PrefillLimitTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(accepted_visual_tokens, 256)
self.assertTrue(over_limit_rejected_before_allocation)
assert_aligned("complete_model_text_logits", actual_text_logits, hf_text_logits, tolerances)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [ ] **3. 最小实现**：构造L=4095/max_new_tokens=1的真实请求；profile内部可独立用4096 token不采样；V256成功/V超256拒绝；重跑P1。
- [ ] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；验证分配后的峰值和最后logits/KV；仅大输入做OOM测试但不测数值不能过门槛。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。


## Task 14：单图 Decode、异常释放与连续请求

**共同前置依赖（子任务另列）：** Task 13。

**Files：** Modify `nanovllm/engine/model_runner.py:172,214`、`nanovllm/engine/sequence.py`、`nanovllm/engine/llm_engine.py:49`、`nanovllm/engine/scheduler.py` 的结束/abort；Create `tests/test_qwen3_vl_decode.py`。

**Interfaces：** `Sequence.release_multimodal() -> None` 清除该请求CPU输入/positions/mask并将 delta 归零，在结束或 abort 后调用；Decode位置为 `[len(seq)-1+seq.rope_delta]*3`。`run` finally 清临时视觉引用和 context；结束前请求只保留CPU元数据，Decode不再次encode。

**原任务共同要求 1：写 `DecodeTests.test_teacher_forcing_pages_and_isolation`：** Prefill之后 teacher-force至少32步，对齐每步 logits与各层KV；准备实际 L=255/256/257 的图文prompt覆盖页边界，真实贪心短样例token一致。首次Decode位置为 L+delta、context_len=L+1，不能把delta加进cache长度。

```python
self.assertEqual(first_decode_positions[:, 0].tolist(), [prompt_len + delta] * 3)
self.assertEqual(first_decode_context_lens.item(), prompt_len + 1)
self.assertEqual(vision_forward_spy.call_count, 1)  # Prefill + 多步Decode之后
self.assertEqual(actual_greedy_tokens, hf_greedy_tokens)
```

**原任务共同要求 3：完成 Decode/生命周期。** 不传 visual_features/mask，不重新跑视觉，KV在现有物理slot更新。正常EOS、max_tokens及可恢复Python错误释放请求CPU和KV；无法恢复的device-side assert等明确终止引擎，后续验证用新进程。
**原任务共同要求 4：写恢复/交叉测试并验证。** 同token布局不同像素A→B→纯文本，比较各自独立HF结果；排队至少3请求，全程最多1个占KV；注入encode/forward异常后，在可恢复路径下一请求可成功，hash map无残留。构造首token即EOS、两种EOS、ignore_eos、max_tokens=1，并确保视觉总调用数等于完成Prefill的图像请求数。
**原任务共同要求 5：green/验收：** `.venv/bin/python -m pytest tests/test_qwen3_vl_decode.py -m gpu --require-gpu -q` 通过；对齐 HF cached 与无cache重算的关系，不能只对齐同样可能错误的两份cached结果。结束时 used blocks=0、requests无pixels引用，未出现跨请求delta泄漏。P3功能门槛完成。

以下子任务替代原任务执行步骤；上面的接口、有效断言及数值要求共同适用，不把父节另计完成任务。

### Task 14a：请求级delta与逐步图像Decode

**前置依赖：** 13c。

**Files：** Modify `nanovllm/engine/model_runner.py`；Test `tests/test_qwen3_vl_decode.py`。

**Interfaces：** 消费prompt_positions/delta/物理KV；产出每步三轴位置与logits，不再传视觉features。

- [ ] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_decode.py --collect-only -q`，节点包含`DecodePositionTests`；缺模块先补可导入接口并重试，不计red。
- [ ] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_decode.py::DecodePositionTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(first_decode_positions[:, 0].tolist(), [prompt_len + delta] * 3)
self.assertEqual(first_decode_context_lens.item(), prompt_len + 1)
self.assertEqual(vision_forward_spy.call_count, 1)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [ ] **3. 最小实现**：应用len(seq)-1+delta构造三轴位置，保持物理cache长度；32步teacher forcing，HF cached/重算互证。
- [ ] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；255/256/257跨页与短greedy都过；无第二次视觉计算。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

### Task 14b：请求结束、异常与引擎退出生命周期

**前置依赖：** 14a、7。

**Files：** Modify `nanovllm/engine/sequence.py`、`nanovllm/engine/scheduler.py`、`nanovllm/engine/model_runner.py`、`nanovllm/engine/llm_engine.py`；Create `tests/test_qwen3_vl_lifecycle.py`。

**Interfaces：** Sequence.release_multimodal()->None；Scheduler.abort(seq)->None；产出可恢复错误后的空闲资源与幂等exit。

- [ ] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_lifecycle.py --collect-only -q`，节点包含`LifecycleTests`；缺模块先补可导入接口并重试，不计red。
- [ ] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_lifecycle.py::LifecycleTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertEqual(len(block_manager.used_block_ids), 0)
self.assertIsNone(seq.multimodal)
self.assertEqual(seq.rope_delta, 0)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [ ] **3. 最小实现**：测正常EOS/两EOS/ignore_eos/max_tokens1；在encode、forward和退出前后注入Python异常，清CPU/GPU引用、finally reset_context；CUDA致命错误只允许新进程恢复。
- [ ] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；CPU故障注入加真实GPU请求结束均验证；不能仅mock释放函数被调用，必须检查块/refcount/引用和下一请求成功。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。

### Task 14c：连续图像/文本请求与排队隔离

**前置依赖：** 14b、14a。

**Files：** Test `tests/test_qwen3_vl_decode.py`；Modify `tests/qwen3_vl_validation.md`。

**Interfaces：** 消费完整生成链；产出A图/B图/文本连续与独立运行的一致结果。

- [ ] **1. Collection前置检查**：`.venv/bin/python -m pytest tests/test_qwen3_vl_decode.py --collect-only -q`，节点包含`RequestIsolationTests`；缺模块先补可导入接口并重试，不计red。
- [ ] **2. 行为red**：运行`.venv/bin/python -m pytest tests/test_qwen3_vl_decode.py::RequestIsolationTests -m gpu --require-gpu -q`，测试断言：

```python
self.assertLessEqual(max_active_kv_owners, 1)
self.assertEqual(block_manager.hash_to_block_id, {})
self.assertEqual(actual_greedy_tokens, hf_greedy_tokens)
```

记录到达的断言和不匹配值；不是setup/import错误。
- [ ] **3. 最小实现**：同token布局不同像素A→B→纯文本，至少3请求排队；和每请求独立HF结果比较，记录delta/特征引用/峰值。
- [ ] **4. Green/完成条件**：重跑上述命令并跑本父任务已完成子模块回归；所有请求终止后零占块、无视觉状态残留；异常路径后能执行新请求，P3闭环。保存JUnit/数值报告，进入所属里程碑质量门禁，不自动commit。


## Task 15：正确性验收、同模型性能Benchmark与使用示例

### Task 15a：正确性与资源范围最终验收

**前置依赖：** Tasks 1–14 全部应过门槛；15a先验正确性，15b再测同模型性能，不以总结替代缺失测试。

**Files：** Create `tests/test_qwen3_vl_acceptance.py`、`examples/qwen3_vl.py`；Modify `README.md`、`tests/qwen3_vl_validation.md`、`.github/workflows/tests.yml`；如增加CPU模型CI job，Create `requirements-ci.txt`，须先验证独立CPU依赖解析。

**Interfaces：** `REQUIRED_GATES: set[str]` 列出1a/1b/1c、2–7、8a/b/c、9/10、11a/b/c、12、13a/b/c、14a/b/c全部实际门槛；`validate_gate_reports(reports: list[dict], required_tasks: set[str]) -> None` 负责检查门槛，失败抛 ValueError；`tests.test_qwen3_vl_acceptance` 调用 support.run_worker 顺序运行目标VL各GPU任务和Task 1b的单一原模型smoke，串行队列、不重复加载两模型。示例使用 AutoProcessor → prepare_multimodal → LLM.generate，输出契约仍为含 text/token_ids 的列表，不引入服务端或新CLI框架。

- [ ] **1. 写 `AcceptanceTests.test_all_gates_and_budget`：** 缺任何门槛报告、fixture revision不符、GPU suite为skip/blocked时失败；输入上限4096/256的成功和超限错误均列入矩阵。用执行命令与case ID关联报告，防止拿旧结果覆盖新实现。

```python
with self.assertRaises(ValueError):
    validate_gate_reports(reports_missing_decode, REQUIRED_GATES)
validate_gate_reports(all_passing_reports, REQUIRED_GATES)
```

- [ ] **2. 验证 red/green。** 先删除临时报告中的一个必填门槛，断言聚合器失败；恢复后执行 `.venv/bin/python -m pytest tests/test_qwen3_vl_acceptance.py -m gpu --require-gpu -q`。定向CPU pytest suites统一跑一遍，不全仓discover。
- [ ] **3. 完成资源报告。** 记录最大纯文本/方图/非方图/长宽极值Prefill、跨页Decode、连续请求的allocated/reserved/剩余显存；确认单模型、16块KV、至少配置安全余量的预算判断与实际峰值一致。对照阶段调低输入只能用于定位，不算4096/256上限通过。精度报告列每组件阈值来源、最大误差、greedy分歧数，任何未解决分歧保留为失败。
- [ ] **4. 写用法与限制。** 新示例离线加载现有checkpoint，max_tokens小于总预算，temperature=0、enforce_eager=True、max_num_seqs=1；README说明BF16/版本/单图范围、不支持功能和复现命令，不覆盖原Qwen3示例。再运行示例一次作为公开接口smoke。
- [ ] **5. 最终验收。** 旧模型轻量兼容、目标VL文本、单图Prefill/Decode、输入拒绝、cache隔离、释放和显存矩阵全部通过；报告明确“本次未验收TP、多图、视频、并发、chunking”。保留工作区改动待用户审阅；走M3人工批准Commit门禁，未批不得git add/commit/push。

### Task 15b：同一Qwen3-VL-2B的公平性能Benchmark

**前置依赖：** 15a通过，1c数值报告与当前候选代码digest一致；真实本地GPU可用。正确性未通过时可以调试计时工具，但不得发布有效性能对比。此任务新增测量，不优化算法、不改变已批准模型/调度方案。

**Files：** Create `benchmarks/__init__.py`、`benchmarks/qwen3_vl.py`、`benchmarks/qwen3_vl_cases.json`、`tests/test_qwen3_vl_benchmark.py`；Modify `tests/qwen3_vl_validation.md`、`README.md`。原`bench.py`保留，不用其旧模型工作负载直接代表VL比较。

**Interfaces：** `run_benchmark(backend: str, case: dict, *, warmup: int, repeats: int) -> dict`（backend为hf/nano）；`summarize_benchmark(samples: list[dict]) -> dict`；CLI `--backend hf|nano|both --cases PATH --warmup 3 --repeats 20 --rounds 2 --output DIR`。both只做串行子进程编排，禁止两模型并存GPU；每个backend/case/计时阶段使用独立worker、可设置超时。repeats指两轮合计20个有效样本（每轮10次），每worker先3次不计入统计的warmup；轮次顺序HF→nano，再nano→HF，逐case匹配比较。

- [ ] **1. 先测试测量工具，不靠性能大小判定通过**：`.venv/bin/python -m pytest tests/test_qwen3_vl_benchmark.py -m 'not gpu' -q`。collection正常后，`test_timing_aggregation_and_manifest`用人工样本验证中位数/p95/单位/吞吐分母；`test_invalid_comparison_rejected`拒绝checkpoint、输入digest、生成长度、dtype不一致或未通过15a的报告；`test_unexpected_early_stop_fails`拒绝少生成token。工具行为失败为red，正确聚合后green。保留以下有效断言：

```python
self.assertEqual(summary['decode_tokens'], 31 * 20)  # 每样本生成32个token，Prefill已产出第一个
self.assertAlmostEqual(summary['decode_tokens_per_second'], 620 / total_decode_seconds)
self.assertEqual(hf_manifest['input_digest'], nano_manifest['input_digest'])
self.assertEqual(hf_manifest['model_revision'], nano_manifest['model_revision'])
```

- [ ] **2. 冻结同模型工作负载**：两边均用本计划固定的Qwen3-VL-2B模型revision/Processor、BF16、eval/inference_mode、batch1/TP1、单活动请求。默认输出G=32 token，忽略整个EOS集合，关闭随机采样及其他logits处理；HF显式设置min/max_new_tokens=G、do_sample=False、无提前EOS退出（例如显式eos_token_id=None并校验实际长度）；nano用temperature0、ignore_eos=True、max_tokens=G。正式运行前API smoke确认固定HF版本确实执行G步。除文本用例外，每次完整Prefill重新编码图片；两边每次从空语言KV开始，禁prefix/视觉跨请求缓存。

`benchmarks/qwen3_vl_cases.json`记录固定CPU输入的digest、实际L、grid、V、G和teacher-forced token digest：文本L=128/512/2048；单图覆盖V=64方图、V=96矩形图、V=256上限图，控制总L为可复现的固定长度（如512/1024/2048，预处理后按实际IDs核实，不以原始图片尺寸推测），均满足L+G<=4096。复用1c的受控图和预处理逻辑；新增性能长度只增加小型输入定义，不重新采0.6B数值基线。图像分辨率、prompt内容、tokenizer/chat template必须完全相同；缺fixture或无法构造指定长度明确失败。

HF语言/视觉主报告使用sdpa，nano语言使用实际FlashAttention、视觉SDPA；两边均关闭CUDA Graph/torch.compile（包括环境开关），记录实际后端及可用的kernel选择。实现/缓存架构差异是比较对象，不能宣称后端完全一致；额外HF FlashAttention或compile优化只作为后续独立标注实验，不混入主表。

- [ ] **3. 实现并核对四个指标的边界**：

| 指标 | 主报告定义 | 排除项/必要细节 |
| --- | --- | --- |
| Prefill延迟，ms/request | 预处理后的CPU输入就绪、开始一次请求的H2D/位置准备/视觉+语言Prefill，至最后位置logits在GPU上可用；同步后停止wall clock | 不含首token采样、不含图片解码/Processor与模型加载；图像编码必须计入，positions准备不能只给某一方免费预算 |
| Decode吞吐，token/s（batch=1单请求） | 完整Prefill后，以同一份HF参考续写teacher-force G-1=31个Decode step，包含语言forward、LM head和该端必要的KV/调度准备；总token数/总wall seconds | 不含Prefill和采样；同续写ID在计时前就绪，排除I/O/参考采集；不能把Prefill产出的首token算入Decode分子。逐步校验未重新执行Vision Encoder |
| 端到端延迟，ms/request | 同样的CPU prepared request就绪，经过请求准入/位置/H2D/视觉+语言/argmax逐token生成G=32，至最后token IDs可供调用者读取并完成请求收尾 | 明确不含Processor；排除模型加载、图片解码、文件写入和tokenizer.decode；双方包含生成loop、host同步/必要token传输与请求释放；公开路径不含聊天UI等外部服务 |
| 峰值显存，MiB | 单独memory worker，模型加载及warmup后释放历史请求状态，reset_peak_memory_stats，运行完整G-token请求，报告peak allocated与peak reserved绝对值及相对空闲模型的增量 | current常驻模型和nano预分配KV也计入绝对峰值；报告nano块数/容量及HF实际cache类型/use_cache，不能扣掉nano常驻KV或把HF无cache模式用于比较。CUDA上下文/其他进程占用另列，PyTorch峰值不冒充整卡峰值 |

报告表头和manifest明确标注：`E2E excludes Processor`、`Decode throughput: batch=1, single request`，不得将该吞吐写成并发或系统总吞吐。

主计时用perf_counter并在测量区间前后cuda synchronize，保证异步工作完成；可附CUDA event设备时间，但不替代包含调度开销的wall时间。Decode可同步整个31步区间，不强制每步额外同步；两边使用同一边界。HF Prefill以logits_to_keep=1避免无必要全序列logits分配，nano维持最后位置head契约。位置计算与请求准备在相应计时区间内完成，不使用oracle positions绕过某一方真实工作。

每个样本都重新初始化请求级KV状态/rope_delta/视觉元数据；nano可复用空的预分配池，HF可复用allocator但不复用有效缓存。计时采集不挂hidden/KV hooks、不保存诊断张量；memory采集另跑，避免hook/profile开销污染速度。初始化/profile/模型加载耗时单独记录为startup，不计入上述暖态结果。HF cache默认采用固定版本实际标准dynamic cache并记录；nano实际paged preallocated容量如实记录，两者服务相同上下文/输出长度，不强行伪装相同分配策略。

- [ ] **4. 真实硬件运行（未来命令）**：先`.venv/bin/python -m pytest tests/test_qwen3_vl_benchmark.py -m gpu --require-gpu -q`验证两个adapter的计时/计数/API小样例，再执行：

```bash
.venv/bin/python -B -m benchmarks.qwen3_vl --backend both --cases benchmarks/qwen3_vl_cases.json --warmup 3 --repeats 20 --rounds 2 --output "$QWEN3_VL_ARTIFACTS/benchmark"
```

记录每条原始样本、输出长度/digest、HF/nano代码与环境digest、GPU型号/驱动/温度和可读的时钟/功耗状态、其他GPU进程、L/V/G、运行顺序、backend/cache/compile配置。使用同一单卡和资源隔离条件；不自动改系统时钟/功耗限制。明显其他进程干扰或热状态不同则标记不具可比性，解释后完整重跑受影响配对，不挑最快样本。GPU不可用/OOM/未跑完分别报BLOCKED/OOM/FAILED，不drop case或算成功。

- [ ] **5. 完成条件/发布**：按case分别报告Prefill与端到端延迟的median/p95、Decode总token/总时间的吞吐及样本分布、allocated/reserved峰值；附20样本原始JSON/CSV和manifest，不能混合不同L/V计算一个模糊“加速比”。性能表同时显示两端数值、差值和可选HF/nano比值；无“nano必须更快/显存更低”的通过条件。若nano更慢也如实记录，区分测量结果与后续优化建议，**不在本任务顺手优化或扩展范围**。正确性状态和性能报告分开：速度不能抵消精度失败，测量失败也不能把正确性已过的结果改写成未实现。独立Review检查公平性，随后按M3人工批准Commit流程交付。

## 后续阶段与租卡触发条件（非本计划实施任务）

| 扩展 | 复用与新增内容 | 验收及算力条件 |
| --- | --- | --- |
| 同一请求多图 | 复用全部视觉网络；放开I=1并核对各图feature/mask、位置连续性与总预算 | 两图不同分辨率、文字夹在图间、交换顺序；总预算内先用本机，不要求TP |
| 多请求/不同长度混合batch | 扁平三轴positions、请求各自delta、并发容量预留 | 串行结果与batch一致、无串KV；小batch先本机 |
| chunking/抢占重算 | 三轴/DeepStack按切片、视觉特征生命周期及完整输入重建 | 切在图片内部、跨块、重算与完整Prefill一致；主要是语义复杂度，不是租卡前提 |
| 视频 | Processor时间戳、帧采样/时间组、视频mask和图文视频顺序 | 先短视频总budget内；长/高分辨率视频超实测显存才评估大显存单卡 |
| TP=2及以上 | 视觉复制/分片策略、文字权重与KV切分、输入广播/Sequence序列化 | 另做设计和计划，真实至少2GPU，与TP=1逐层/输出对齐；确需租用或借用多卡时再申请 |
| prefix cache/CUDA Graph/compile | 带视觉身份的cache key与失效、静态图输入/位置更新 | 先建立正确性基线，再单独验收；通常不要求多卡 |

## 技术假设、待测量值与计划自审

已核实事实见前文环境表；以下是建议或派生预算，不能写成GPU已验证结论：

- 1GiB `kv_cache_reserve_bytes`是计划工程起点，不是HF配置字段或测得余量。13b必须测分配后真实峰值，不能用1GiB保证零OOM；若不足，报告并提出预算/范围修订供审批，不私自改已批准范围。
- 16块是4096/256的派生上限，KV=448MiB是BF16、28层、8KV头、head_dim128的算术值，不包含激活/工作区；通用预算助手从配置计算。允许降低max_model_len时块数也相应改变。
- 4096总token/256视觉token是设计的初始验收范围，不是已证明显卡极限。13c分别验证真实请求边界、数值和峰值。
- pytest/Ruff候选版本、CI CPU依赖闭包、FlashAttention在当前GPU架构上的运行和HF阈值均待1a/1c实际验证；不为解决工具问题擅自升级torch/HF。
- Task 11保留的跨图隔离bitwise断言仅适用固定形状、确定后端、同一图输入不变的独立分段计算；跨backend/HF数值对齐仍用冻结阈值。若内核有真实非确定性，先记录HF/同实现重复基线再审查隔离断言，不随意设大atol。

自审覆盖：1a测试/CI基础，1b轻量旧功能兼容，1c同模型VL oracle；2–7保持配置/MRoPE/加载/文本/采样/调度接口；8a/b/c拆分Runner；9输入契约；10视觉位置；11a/b/c拆分block/merger/提取；12融合；13a/b/c拆分路由/profile/上限；14a/b/c拆分Decode/释放/隔离；15a正确性验收，15b公平性能Benchmark。父任务共同契约保留原数值断言与计算细节，所有子任务声明文件、依赖、输入输出、真实red、green和完成条件。

Review Focus五项均映射到测试；多图/视频/并发/TP不纳入实施范围；任务间positions为[3,N]、features为tuple(main,list[3])、delta为请求级int，物理KV长度独立。ImportError不再作为TDD red；pytest收集、依赖缺失和GPU不可见分别记基础设施状态。每个里程碑均有diff/回归/独立Review/人工Commit批准门禁。

## 审核交接

历史第三版交接止于Task1c。Task2–5现已提交为`89c1e40`，当前用户授权Task6，本轮边界以顶部Status及Task6执行报告为准；阈值提案未因任务授权而获批准。不自动暂存/Commit/Push，不推进Task7。

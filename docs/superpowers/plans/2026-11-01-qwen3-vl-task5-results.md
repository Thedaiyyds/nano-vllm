# Task 5：VL 文本 decoder、tied head 与真实语言权重实载

2026-11-01，Native执行，仅完成已批准Plan的Task5。分支`feat/multimodal`，HEAD仍为`f331e43406de157049028e716bca61926a38378e`。保留Task2–4及用户实验，未安装依赖、未改CUDA或HF源码、未重采正式Oracle，未暂存/Commit/Push，未开始Task6。

## 实现说明

新增`nanovllm/models/qwen3_vl.py`，实现文本attention、MLP、decoder、language model和`Qwen3VLForConditionalGeneration` wrapper：

- 参数位于`model.language_model`，可直接消费Task4严格加载器。`lm_head.weight`与`embed_tokens.weight`为同一Parameter。
- Q/K投影后分别RMSNorm，再调用Task3三轴Interleaved-MRoPE，复用原Attention的FlashAttention计算及Triton物理KV写入；MLP使用SiLU。
- 每层立即执行`x = residual + attn`，再norm/MLP，立即执行第二次residual add，最后独立final norm。没有使用旧decoder的延迟residual/fused add norm。对FP32输入复制norm输入，避免共享RMSNorm的原地操作破坏residual；本轮完整模型验收仍为BF16。
- `forward()`返回展平token hidden `[N,2048]`；`compute_logits()`复用ParallelLMHead的Prefill末位置契约，单请求输出`[1,151936]`。
- 当前只允许TP=1、text_only模型；`text_only=False`、非空visual_features或visual_mask明确报未接通。未接入Runner、采样、调度、视觉或生成循环。

QKV及gate/up使用VL专用的原parallel linear子类，**保留packed Parameter和原weight_loader，forward分别执行独立GEMM，再拼接输出**。这是本次有数值证据支持的局部取舍：参数命名、Task4映射和共享线性层保持不变，但每层QKV由1次变为3次GEMM，gate/up由1次变为2次，可能增加开销。当前不宣称性能提升，实际成本留Task15b测量；不为此修改旧Qwen3或提前做优化。

## TDD与数值定位

先写3项GPU测试和可构造的最小scaffold，正常collection后，在宿主真实GPU得到**3 failed**：BF16 residual witness输出不符、head/embedding不是同一Parameter、视觉特征没有拒绝。`model-red.py`保存当时scaffold；没有将ImportError、NotImplementedError或CUDA setup失败当作RED。

residual witness以常量attention输出和放大后的MLP隔离残差流程，手算比较“BF16立即add后再norm”与“FP32未舍入sum直接norm”，结果确实不同；实际decoder须与前者精确相等。该隔离测试不代替真实Attention计算，完整checkpoint测试另外执行真实FA和KV写入。

初次小配置遗漏顶层`tie_word_embeddings=True`，实现后曾出现1 passed/2 failed的fixture配置问题；已补齐顶层字段，与目标snapshot一致，没有放宽绑定断言。此失败保留在`small.log`，不当作新的正确性RED。

直接复用原packed单GEMM后，小模型3项通过，但真实模型严格对照失败：

| 阶段 | 真实结果与原因 |
| --- | --- |
| 原packed GEMM，默认BF16 reduced precision=True | 3 passed/1 failed；24与257-token有非零误差，255/256精确一致；119项浮点比较失败 |
| 两端仅诊断性关闭BF16 reduction截断 | 3 passed/1 failed；257归零，24仍有54项非零 |
| 隔离GEMM诊断 | 同输入同权重，257首层packed/独立V的max_abs=0.001953125；关闭截断后归零 |
| 进一步隔离24-token第2层 | 即便关闭截断，同输入下packed/独立K的max_abs=0.00006103515625，K norm后为0.00048828125，说明仍有归约顺序差异 |
| VL专用独立投影，恢复原默认精度设置 | 四个真实用例全部精确相等，最终GPU **4 passed/1 deselected** |

据此只调整新增VL投影forward，没有扩大容差、删除失败用例、替换权重或改变共享算子。最终未设置新的生产全局精度开关：HF与nano worker均使用原默认`allow_bf16_reduced_precision_reduction=True`；False仅存在于保留的诊断记录。初次失败数据、诊断和最终成功数据分别保存在不同目录。

## 真实实载与GPU验收

固定模型snapshot `89644892e4d85e24eaac8bacfd4f463576704203`，HF提交`2711ad22c82105e240a2877b92d082fe2857936f`。reference worker调用现有`provenance()`核对HF源码AST/revision及checkpoint文件指纹，再用HF原生完整模型执行；candidate worker运行新增native模型。两者独立进程顺序加载，参考及时落CPU，没有两份完整模型同时驻留GPU。

真实native模型已通过Task4 public loader实际复制**310个语言源key**，仅跳过**315个视觉源key**，tied alias为`lm_head.weight`，没有其他缺项。这是完整语言部分实载，未加载视觉模块。新增CPU测试还直接检查native meta结构的226个独立Parameter及其header契约，补足Task4仅用HF meta替代schema的边界。

采用持久Oracle的纯文本IDs、三轴positions和用例定义，24/255/256/257-token各运行完整Prefill。native使用物理页顺序`[2,0]`，实际Triton写入后再gather；257的末token跨到物理页0。每例比较：前三层完整输出及attention residual后的norm输入、final norm、末位置logits，以及全部28层的旋转后K与V抽样（首/中/末位置、全部KV头）。未检查每个缓存位置，不声称完整Decode复用已接通。

| 用例 | token数 | 浮点比较项 | 对独立HF FA2的max_abs/mean_abs/relative L2 |
| --- | ---: | ---: | --- |
| text_short | 24 | 64 | 全部0 |
| text_255 | 255 | 64 | 全部0 |
| text_256 | 256 | 64 | 全部0 |
| text_257 | 257 | 64 | 全部0 |

64项为6个前三层hidden/residual快照、final norm、logits、56个逐层K/V快照；整数位置和采样索引另做精确比较。浮点同时检查shape、dtype、finite及零容差误差指标，没有只比较可读文本或top-1。

实际硬件RTX5060Ti，torch2.9.1+cu130、CUDA13.0、FlashAttention2。两端BF16/eval/inference_mode，`TORCHDYNAMO_DISABLE=1`，没有CUDA Graph、sampling或HF generate。

| native用例 | Prefill peak allocated | peak reserved | 测试后设备空闲 |
| --- | ---: | ---: | ---: |
| 24 | 3389.893 MiB | 3936 MiB | 11177.875 MiB |
| 255 | 3404.340 MiB | 3942 MiB | 11171.875 MiB |
| 256 | 3404.402 MiB | 3942 MiB | 11171.875 MiB |
| 257 | 3404.467 MiB | 3942 MiB | 11171.875 MiB |

上述为语言模型已加载、测试用3物理页KV已分配后的PyTorch进程峰值，reserved包含allocator池；空闲量为当时整卡瞬时值。不是模型启动峰值、4096上限显存验收、完整VL引擎预算或Benchmark。HF完整模型也加载视觉权重，其本轮Prefill峰值4070.944–4109.382 MiB，完整记录随reference保存；不据此发布公平性能结论。

## 浮点验收边界

**未批准的宽BF16阈值文件没有加载或用于放行。** 正式`~/nano-vllm-artifacts/qwen3-vl/oracle-reviewed/`保持只读，未重采、覆盖或改阈值。

本次增加独立的、固定HF **FlashAttention2** reference，与native实际语言Attention后端匹配，采用有限值与精确相等门槛；新reference位于本机持久化Task5目录，每次pytest创建新的run目录。它们与原SDPA/eager Oracle分开，不覆盖原参考，也不宣称跨backend bitwise一致。

与原SDPA Oracle的末位置logits差异单独记录为诊断：

| 用例 | max_abs | mean_abs | relative L2 |
| --- | ---: | ---: | ---: |
| text_short | 0.296875 | 0.0671155 | 0.0309828 |
| text_255 | 0.2571411 | 0.0558264 | 0.0258983 |
| text_256 | 0.2571411 | 0.0558264 | 0.0258983 |
| text_257 | 0.203125 | 0.0340877 | 0.0162776 |

这些数值没有被包装成SDPA对齐通过，也没有被用来反推新阈值。Task5通过结论限于声明的同模型、同设备、BF16、HF FA2对照；跨后端精度政策和原宽阈值仍待独立批准，不推广到后续视觉或decoder的任意环境。

## 最终结果与独立Review

| 验证 | 真实最终结果 |
| --- | --- |
| 开发前CPU基线 | 112 passed，8 deselected，8.85s |
| Task5 GPU | **4 passed，1 deselected，13.75s** |
| 全部CPU | **113 passed，12 deselected，8.62s** |
| 快速CPU | **20 passed，105 deselected，1.65s** |
| Ruff check/format、git diff --check | 通过 |
| 测试模块阻断torch/transformers后的收集导入 | 通过 |
| worker无模式参数执行 | exit2，执行前拒绝 |
| Task2–4生产文件SHA256 | 与各任务交付一致 |

默认沙箱GPU最初在setup报3 errors（CUDA不可见）；之后获得宿主GPU权限执行上述真实测试。该基础设施失败未计入TDD RED，也未改成skip。CPU schema及已有loader回归使用真实单rank Gloo，需要宿主本地socket权限。deselected不计作已运行通过；远端CI本轮未运行。没有重做Task1参考采集或旧模型GPU矩阵。

独立Reviewer只读审查两新增文件、Plan/Spec、共享算子、失败诊断和成功数值报告：无Critical、Important或必须修改的Minor，Task5范围可通过；认可局部split GEMM与同后端零容差裁定。Reviewer没有自行重跑pytest，依据源码及落盘GPU证据核查。其审查时成功GPU记录为4 passed/14.82s；随后仅补reference后端断言、环境/显存记录及worker参数校验，主代理再跑最终GPU13.75s和全部CPU。没有将Reviewer阅读证据写成独立重跑。

本轮裁定及成本：复用无目标冲突的Native feature工作区；复用Task4模块内rank fixture，不提前扩展测试框架；独立FA2 reference用于本轮同后端验收；VL局部独立GEMM换取HF舍入路径一致，代价是更多GEMM和潜在性能下降。均不扩展用户功能范围，正式阈值和共享实现保持不变。

Reviewer留给后续的范围：Runner、Decode分页读取和teacher forcing属Task8；greedy/EOS/调度属Task6–7；视觉/融合/DeepStack属Task10–12；请求边界属Task9；TP>1、编译和性能另行验收。本轮接受上述边界，不声称M1整体完成。无已知待修复重要问题或延期Minor。

## 复现命令与交付

仓库根目录、现有venv，在GPU和本地socket可用的宿主终端执行，不安装或下载：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_VL_ARTIFACTS="$HOME/nano-vllm-artifacts/qwen3-vl"
export QWEN3_VL_ORACLE="$QWEN3_VL_ARTIFACTS/oracle-reviewed"
export QWEN3_VL_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"

.venv/bin/python -m pytest tests/test_qwen3_vl_text.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_text.py -m gpu --require-gpu -q
.venv/bin/python -m pytest tests/test_qwen3_vl_text.py -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/models/qwen3_vl.py tests/test_qwen3_vl_text.py
.venv/bin/python -m ruff format --check nanovllm/models/qwen3_vl.py tests/test_qwen3_vl_text.py
git diff --check
```

证据目录`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task5/`包含原RED、所有失败/诊断及最终JUnit、`task5-summary.json`、`review.md`、`task5.diff`。最终完整成功数据为`full-c9om_u8w/`：独立HF/nano CPU张量、HF provenance/环境、真实加载报告、逐项误差与SDPA诊断。`small-*`保留小模型独立HF参考；`gemm_diagnostic.*`、`gemm_localized.*`保留定位源码和结果。该持久化自忽略目录不进入Git。

最终源码SHA256：

```text
7c8a48b797a647165436d24ca92cc1be2b36c31fae9a32e58a94a6bb7892cf51  nanovllm/models/qwen3_vl.py
969a52b4ab2b1e5e54c6d8db1299be4e204ad6ae89f6ff46b572c15ea879b564  tests/test_qwen3_vl_text.py
```

Task5交付文件：上述两个新Python文件、本报告、实施计划状态及Task5勾选更新。拟提交信息`feat: add Qwen3-VL text decoder and tied LM head`。技术验证和独立Review完成，工作区保留待审阅；未暂存、Commit或Push，本轮止于Task5。

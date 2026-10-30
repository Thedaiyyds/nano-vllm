# nano-vLLM Qwen3-VL-2B-Instruct 适配设计草案

状态：用户已于 2026-10-29 明确批准设计；当前仅授权生成实施计划，计划仍须审核，不开始实施、不提交或推送。沿用原设计文件。下文保留的“待批准/方案建议”措辞记录设计讨论过程，以本条最新批准状态为准。

## 目标与决策

首轮范围已由用户在本轮明确确认：先纯文本推理，再单图完整 Prefill，再单图 Decode；多图、视频和并发调度后续扩展。该范围确认不等于整份设计获批。以 Transformers 为正确性参考，不做性能优化。硬件目标为 RTX 5060 Ti 16GB，TP=1。

既有草案记录用户选择路线 A，本轮沿用此历史决策，不将其视为整份设计已批准：自行实现视觉模块。复用 HF AutoProcessor、配置解析、PyTorch 基础算子；HF 完整模型和视觉模块只作为测试参考，不委托 HF generate 执行 nano-vLLM 推理。

备选路线 B 是复用 HF Vision Encoder，开发更快，但保留视觉运行时实现依赖且不能满足最终选择。整体包装 HF generate 会绕过本项目的调度和分页 KV Cache，不采用。

初期限定 batch=1、单张静态 RGB 图片、无 padding 的内部输入、完整 Prefill、BF16 权重、TP=1、关闭 CUDA Graph。不支持的多图、视频、混合批次、超长输入直接明确报错，不能静默丢弃。保留原 Qwen3 模型入口及行为；新能力首先限定到 qwen3_vl 分支。纯文本路径指目标 VL checkpoint 的文本分支，包含 Prefill 和基本逐 token Decode；后面的 Decode 阶段专指多模态。

## 现状证据

本轮检查时分支为 `feat/multimodal`，HEAD 为 `bb823b3`；已跟踪文件无工作区差异。`docs/`、`image/`、`video/`、根目录实验脚本与两份 requirements 快照均为已有未跟踪内容，保留原状，不清理、不暂存。没有 `tests/` 目录；以下“已有”表示源码存在，不代表本轮 GPU 回归已通过。

- nanovllm/models/qwen3.py：已有 Qwen3 GQA、Q/K RMSNorm、SwiGLU、embedding 和 LM head；没有视觉模块、三轴位置、DeepStack 或图文替换。
- nanovllm/config.py：直接访问顶层 max_position_embeddings；VL 的文字配置在 text_config。
- nanovllm/engine/model_runner.py：硬编码 Qwen3ForCausalLM；KV 分配读取顶层配置；Prefill/Decode positions 均为一维序列索引；只用文本 warmup 估算激活峰值。
- nanovllm/utils/loader.py：直接匹配权重名并做 packed QKV/gate-up 映射，无 VL 命名空间适配和完整加载审计。
- Sequence/LLMEngine：仅 token IDs，无 pixels/grid/视觉位置/rope_delta 等请求状态。
- scheduler.py 已支持 chunked Prefill 和抢占重算；block_manager.py 已支持只按 token IDs 和前缀 hash 做缓存复用。图片占位符相同不代表图片相同，此路径不可直接用于 VL。
- Attention 是因果语言注意力并写分页 KV，不能直接用于非因果视觉注意力。
- SamplingParams 禁止零温度；Sampler 只做随机采样。需要可用的 greedy 分支进行确定性验收。
- LLMEngine 仅读取 tokenizer 的单个 EOS，Scheduler 用标量相等判断结束；目标 checkpoint 的 generation_config 指定 `[151645, 151643]`，需对齐结束条件，不能只改模型计算。
- ParallelLMHead 在 Prefill 仅投影每请求最后一个 hidden state；完整 logits 对齐需要独立测试路径，避免改变生产输出契约。
- 现有 test-Qwen3VL.py 等属于 HF 实验脚本，不是 nano-vLLM VL 集成测试；其末尾把 forward logits 当生成 token 解码，需要独立替换为正确的参考验证工具。

## 参考版本与环境

项目 .venv：Python 3.12 环境、torch 2.9.1+cu130、flash-attn 2.8.3、Transformers 5.21.0.dev0，后者 editable 指向 /home/thedai/transformers。

参考 Transformers 提交：2711ad22c82105e240a2877b92d082fe2857936f。modeling_qwen3_vl.py、processing_qwen3_vl.py、vision_utils.py 有本地研究注释；对比 HEAD，忽略注释/docstring 后 AST 一致。requirements 快照仍是 transformers 4.57.6，不能混作当前版本。后续验证固定上述提交和依赖，不跟随 main。

本地模型 snapshot：89644892e4d85e24eaac8bacfd4f463576704203。safetensors 文件已存在，约 4.26 GB，625 个 tensor；前缀只有 model.language_model 和 model.visual，没有独立 lm_head.weight，因此必须正确绑定词嵌入权重。完整权重 checksum、可用 GPU 和 kernel smoke test 留到环境验收。

本轮 nvidia-smi 无法连接驱动；未执行模型推理或 GPU 测试。不能据此断言用户环境损坏，也不能声称显存或数值验证已通过。

本轮实际完成的只读验证：三份本地修改过的 Transformers 源文件与 HEAD 在移除 docstring 后 AST 一致；元数据确认上述依赖版本以及 Triton 3.5.1；离线读取 safetensors header 确认 625 个 tensor、文件大小 4,255,140,312 bytes，没有加载全模型。CPU AutoConfig 显示顶层 `dtype=None`、`text_config.dtype=torch.bfloat16`，因此 Runner 的 `set_default_dtype` 也必须取解析后的文字 dtype，不能只修 KV 分配。CPU Processor 两个合成 RGB 样例均通过 token/grid/模态 mask 检查：256×256 → grid `[1,16,16]`、pixels `[256,1536]`、64 个视觉 token；384×256 → `[1,16,24]`、`[384,1536]`、96 个视觉 token。未写入图片或修改实验脚本。当前 Processor 提示调用参数宜放在 `processor_kwargs` 中，正式参考工具需按固定版本使用该接口。

主要正确性来源是本地 `/home/thedai/transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py` 的 `Qwen3VLVisionModel`、`Qwen3VLTextRotaryEmbedding`、`Qwen3VLTextModel`、`Qwen3VLModel.get_rope_index`，以及同目录 `processing_qwen3_vl.py` 和 `src/transformers/vision_utils.py`。上游定位链接：[固定 Transformers 提交](https://github.com/huggingface/transformers/tree/2711ad22c82105e240a2877b92d082fe2857936f/src/transformers/models/qwen3_vl)、[固定模型配置](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/config.json)、[生成配置](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/generation_config.json)。本轮网页读取成功的是模型配置；模型计算细节依据本地源码，不以无法获取的 GitHub 网页作为已核验的证据。

## 模块边界和数据契约

拟新增 nanovllm/models/qwen3_vl.py：VL wrapper、文本 decoder、图文融合和 DeepStack；nanovllm/models/qwen3_vl_vision.py：视觉网络；nanovllm/layers/mrope.py：三轴位置和 interleaved rotary；nanovllm/utils/multimodal.py：Processor 输出归一化及输入验证。避免创建通用多模态插件框架。

Config 保留完整 hf_config，新增明确的文字配置访问路径；模型选择用 model_type 的小分支。KV 头数、层数、head_dim、dtype 从 text_config 取得。checkpoint 按命名空间映射；仅语言层使用 q/k/v 和 gate/up packed 规则，视觉 qkv 不应误处理。纯文本阶段允许显式跳过 model.visual 并报告；其余 missing/unexpected/shape mismatch 报错；完整阶段全部可训练参数必须加载或属于已验证的 tied alias。

建议保留 str/list[int] 的现有请求入口，新增 PreparedMultimodalPrompt 输入对象，由 prepare_multimodal(messages, processor) 产生。对象包含 token_ids [L]、pixel_values [P,1536]、image_grid_thw [1,3]、模态类型 [L]；不把 HF BatchFeature 原样传遍 scheduler。

请求状态拥有 prompt 三轴 positions [3,L]、rope_delta 标量、视觉 token mask [L]。视觉特征主分支 [V,2048] 与 3 组 DeepStack [V,2048] 仅在需要 Prefill 时传给模型，初期不保留跨请求视觉缓存。完整输入元数据保留到请求结束，供后续抢占重算策略使用；正常 Decode 不再执行 Vision Encoder。

生命周期约束：CPU 预处理数据归请求所有，只有已准入请求将 pixels 和特征搬到 GPU；完整 Prefill 成功后释放视觉 GPU 临时张量，保留 CPU 数据及 positions/delta，结束或异常时释放请求资源与已分配 KV。首版不支持抢占重算；保留 CPU 数据只是避免未来必须重新设计输入所有权。TP=1 不走 Sequence 的 pickle/共享内存分支，本轮不为多卡扩展序列化协议。

输入校验必须在 KV 分配前完成：图片数恰为 1、T=1、H/W 为正且可被 merge_size 整除、pixels 行数等于 T*H*W、image mask 与模态类型一致、图像占位符完整且数量等于 V。纯文本输入不得包含没有对应图片数据的 image/video 占位符。只验证输入契约，不接受缺失图片后退化为纯文本推理。

runner 使用无 padding 的展平 token 布局；语言位置为 [3,N]，与物理 KV slot、context_lens、cu_seqlens 分开。未来多请求在 token 维拼接，但本阶段拒绝 batch>1。模型输出仍为 hidden states [N,2048]，由现有 LM head 计算最后位置 logits；测试可逐位置单独做线性投影。

## 核心计算

文字配置：28 层，hidden=2048，intermediate=6144，16 Q heads、8 KV heads、head_dim=128，vocab=151936，rope_theta=5000000，mrope_section=[24,20,20]，共享 embedding/LM head。使用实际配置，不硬编码这些值进入运算。

纯文本将三轴位置设置成同一递增序列；Interleaved-MRoPE 此时退化为普通 RoPE。必须使用 VL checkpoint 和 VL text_config，不能把 Qwen3-0.6B 的通过当作 VL 文本通过。当前 Transformers 将旧 rope_scaling 字段归一为 rope_parameters；读取固定版本解析后的配置，显式适配该版本，不默默使用 Qwen3 默认 theta。

新 VL decoder 首版采用 HF 顺序的显式 residual add 和 norm，复用 linear、attention 等基础组件。现有 Qwen3 decoder 使用延迟 residual 合并，直接在其 hidden_states 上加 DeepStack 会改变 BF16 运算次序，因此不原样复用该层结构。保留原 Qwen3 路径。

视觉路径：HF Processor → patch embed Conv3d → 可学习二维绝对位置插值 → 二维视觉 RoPE → 24 个非因果 ViT blocks → 主 merger 与三组 DeepStack merger。视觉 attention 先用 PyTorch SDPA(is_causal=False)，不写语言 KV Cache。各图/时间组注意力边界必须遵循 HF，即使初期只有单图也在接口里保留 grid 信息。

P=T*H*W；patch_size=16、temporal_patch_size=2、channels=3，故 pixel_values 为 [P,1536]；patch embedding [P,1024]；空间合并 2x2 后 V=P/4，每路输出 [V,2048]。必须遵守 Processor 的块内 patch 顺序，不能再错误重排。绝对位置插值、视觉 RoPE 的坐标次序与 HF 逐项对齐。

视觉绝对位置表为 `[2304,1024]`，即 48×48；使用双线性插值、`align_corners=True`、border clamp。按 `(block_row, block_col, in_row, in_col)` 解出每个 patch 的 H/W 坐标，再计算四邻点索引和 FP32 权重；与 embedding 加权求和后转回 hidden dtype 再相加。首版直接复现该运算顺序，不仅凭数学等价就替换为不同 dtype 的 resize。二维视觉 RoPE 使用 theta=10000、head_dim=64，频率拼接为 `[H,W,H,W]`；它与语言三轴 Interleaved-MRoPE 是两套独立机制。视觉 block 使用 LayerNorm 和 `gelu_pytorch_tanh` MLP，merger 使用普通 GELU；不能直接复用文字 RMSNorm/SwiGLU。

主 merger 使用 pre-shuffle norm，DeepStack merger 使用 post-shuffle norm，不能只复用参数而忽略计算顺序。ViT 第 5/11/17 层（零起点）输出分别经 merger 后，在语言第 0/1/2 层完整 residual 输出之后，仅向图片位置添加。主视觉特征在 decoder 之前替换 image_pad 的词嵌入。占位符数量必须等于 V。

Interleaved-MRoPE：三轴 half-dim 频率先按 T 作底；H 取索引 1:60:3，W 取 2:60:3，其余保留 T，再复制组成 128 维 cos/sin。不能把 [24,20,20] 写成三个连续切片。用 t/h/w 不同的位置测试，纯文本无法暴露此错误。

文字 cos/sin 在 FP32 计算后按 HF 转为输入 dtype，再执行文字旋转；视觉旋转则按 HF 将 Q/K 和 cos/sin 转 FP32 运算后转回。现有 Qwen3 rotary 的 FP32 旋转路径不能未经对照直接当作 VL 文字旋转。为控制 BF16 误差，应同时核对 Q/K norm、旋转后的 Q/K、显式 residual 的 dtype 和舍入点。

位置生成以固定 HF get_rope_index 为 oracle：文本段三轴同步递增，图片段使用合并后网格坐标，后续文本从图片坐标最大值之后继续。delta=max(prompt_positions)+1-L。无 padding 单请求 Decode，第一个新输入 token 的物理序号为 L，三轴位置均为 L+delta；之后逐 token 加一。KV 长度仍计实际 token 数，不能加 delta。

## 调度与显存边界

初期 VL 禁用 prefix cache 的查找和发布，两处都要隔离；仅让 cache miss 不够。不同图片相同占位符 IDs 不得复用 KV。保留语言 paged KV 读写；不等于禁用 KV Cache。

单图 Prefill 初期必须完整调度，校验 L<=max_model_len 且 L<=max_num_batched_tokens，否则明确报错。先不实现跨图片片段的 chunking。batch=1 且按预算限制 prompt+max_new_tokens，并预先验证所需 KV 容量；容量不足报错，避免未经验证的多模态抢占重算。

这里的 batch=1 还要求初期 VL 引擎同一时刻仅一个已准入的活动请求：后续请求可以排队，但前一个完成、释放 KV 前，不调度后一个的 Prefill。仅设置 max_num_seqs=1 不足以保证这一点，当前 scheduler 会先处理等待队列 Prefill 并累积多个 running 请求。P2c 必须增加此准入规则，P3 测试队列中多个请求始终只有一个占用 KV。多请求并发准入和全体请求的容量预留属于 P4。新 VL 路径 TP>1 和 CUDA Graph 启用时明确拒绝，不能静默运行未经验证的分支。

该限制覆盖 qwen3_vl 引擎内的纯文本与图文请求。开始 Prefill 前保守检查 `ceil((L + max_new_tokens)/block_size)` 不超过可供该请求使用的块数，并限制 `L + max_new_tokens <= max_model_len`；单活动请求期间不再向别的请求分配 KV，因此可随 Decode 按需追加物理块。无法容纳的队首请求必须明确失败，不能永远等待或进入抢占循环。`max_new_tokens` 必须为正；长度终止保留当前“最多生成多少 token”的含义。

生成语义：VL 分支从固定 generation_config 解析 EOS 集合，`ignore_eos=True` 时忽略整个集合，长度限制始终有效；原 Qwen3 单 EOS 行为保留。greedy 必须真正 argmax，不能用极小 temperature 近似。HF 对照显式关闭 sampling；不在本期增加 top-p/top-k 等新采样功能，避免 checkpoint 默认 `do_sample=true` 造成参考不一致。

CUDA Graph 关闭；enforce_eager 只控制 CUDA Graph，现有层仍有 torch.compile，不能把它等同于全部 PyTorch eager。测试基线显式禁用 Dynamo 编译，后续可单独验证编译路径。FlashAttention 内核兼容性通过环境 smoke test 验证；若失败需另行选择可运行的正确性后端，而不是绕过缓存测试宣称已适配。

16GB 下 HF 与 nano-vLLM 分进程顺序运行，参考张量落 CPU，避免两个模型和 KV 池同时驻留。首批使用受控的短文本、低分辨率图片和 batch=1；以 Processor 实际 grid 为准，不用输入图像尺寸猜 token 数。扩大到非方图检验位置逻辑。

BF16 文本 KV 每 token 2*28*8*128*2=114688 bytes，即112 KiB，4096 token约448 MiB，不含分块取整。权重约3.96 GiB，剩余显存还需容纳视觉激活、logits、CUDA上下文和工作区，不能只按权重判断可行。采用带视觉输入的受控 warmup/profile 后分配 KV，并设置显式输入上限与余量；不是直接沿用文本 warmup 后占满90%显存。

建议初始验收上限为 `L+max_new_tokens<=4096`、单图 `V<=256`（P<=1024），以实际 Processor grid 校验，并对方图与允许的长宽比覆盖 profile。这是待批准的保守范围，不是已经实测的最大能力。Processor 像素上限与引擎 visual-token 上限一起设置；用户更大的输入可由显式的预处理策略缩放，Prepared 输入超限则报错。16GB 的可用余量与 profile 安全裕量在 P0/P2 GPU 验证后记录；不能把该初值当作免于显存测试的保证。

## 验证与阶段任务

每阶段都先写会失败的测试、确认失败原因、实现最小改动、运行对照和原 Qwen3 回归。下列是设计级任务拆解；详细逐步骤实施计划在设计批准后生成。

P0 基准：固定环境/模型 revision、修正参考采集方式、建立 tests/reference_qwen3_vl.py 与共享 fixtures。验证 GPU/FlashAttention，保存 input IDs、grid、positions、关键 hidden states、logits、KV；HF eval/inference_mode、关闭采样与 dropout。不会把模型 forward logits 当 generate token。

P1 纯文本：修改 config.py/model_runner.py/loader.py，新增 VL 文本模型、配置及 dtype 映射、完整加载报告与 tied head；sampling_params.py/sampler.py 增加真正的 argmax greedy，并接通 VL EOS 集合。先比 Prefill 最后 logits，再比 teacher-forced Decode 和固定短 prompt 的 greedy tokens，跨256-token分页边界；保留 Qwen3-0.6B 回归。可在本阶段用三轴相同的位置，但接口必须预留三轴。

P2a 位置与输入：新增 multimodal.py/mrope.py，扩展 Sequence/LLMEngine/runner。测试 Processor token/grid一致、三轴位置与delta精确一致、矩形网格、图片前后文字、占位符不匹配报错。跨请求状态不泄漏。

P2b 视觉：实现 qwen3_vl_vision.py 和相应权重加载；按 patch embed、绝对位置插值、视觉 RoPE、选定 ViT 层、主 merger、三组 DeepStack 输出逐级对照。使用合成小图与现有 candy.JPG、哈基米.jpg 的受控输入。此阶段可使用 HF 特征作为融合测试 fixture，但最终验收必须换成自实现视觉输出。

P2c 单图完整 Prefill：接通视觉替换、DeepStack、语言 attention、KV 写入，修改 scheduler/block_manager 的 VL 初期限制。对比融合 embedding、前三层注入后的完整 hidden states、最终 norm、最后 logits，并至少采样检查各层KV；测试两个相同token布局但不同像素请求不共享cache。阶段产物是正确的 Prefill 与首个 next-token prediction，不宣称完整多模态生成。

P3 单图 Decode：以请求级 delta 构造位置、复用 paged KV、视觉只算一次，测试 teacher-forced 多步 logits、固定短样例 greedy、256-token block 跨界、EOS/长度终止、连续请求状态隔离。需同时检查 HF cached Decode 与无 cache 全序列重算关系，防止 Prefill 数值误差掩盖位置/cache错误。

P4 后续功能按顺序逐项验收：多图及图文交错 → 多请求/不同长度混合batch → chunked Prefill/抢占重算 → 视频及时间戳 → 需要时再做包含视觉身份的prefix cache、TP>1与CUDA Graph。前三阶段不声称支持这些场景；输入功能与优化分别立项。

验收：IDs、grid、positions、mask、shape、权重映射要求精确；浮点误差按固定 dtype/backend记录 max_abs、mean_abs、相对L2和logit top-k。P0在相同权重的HF自身 eager/SDPA差异上校准并冻结分组件阈值，再做候选实现对照，不能失败后放宽阈值。接近零的元素不只看相对误差；不能仅凭文字看起来合理验收。短固定greedy样例应一致；有低margin分歧时报告logit差异，不隐藏分歧。绝不以BF16 bitwise相等作为跨kernel默认要求。

追加边界验收：prompt 长度 255/256/257、Decode 追加到新块、恰好达到长度/visual-token 上限、超限拒绝、两个 EOS 分别终止、ignore_eos、异常释放和后续请求恢复。低分辨率与非方图用于正确性定位，最大允许输入用于显存验收，二者不能互相替代。原实验脚本保留，新的参考采集工具独立建立；避免全仓自动收集 `test*.py` 意外触发下载、视频解码和 GPU 全模型加载。

## 主要风险与取舍

| 风险 | 本设计的控制方式 | 仍需验证 |
| --- | --- | --- |
| 三套位置逻辑混淆、图文 token 排序错位 | 绝对位置、视觉二维 RoPE、语言三轴 RoPE 分别与固定 HF 对照 | 非方图、图片前后文字、t/h/w 不同的频率用例 |
| DeepStack 加错位置或 BF16 顺序变化 | 新 VL decoder 显式 residual；只在前三层输出的视觉位置注入 | 注入前后 hidden states、最终 logits |
| 相同占位符导致跨图片 KV 污染 | VL prefix cache 查找与发布均禁用，保留请求内 paged KV | 相同 token IDs / 不同像素的连续请求 |
| chunking、抢占与视觉状态未协调 | 首期完整 Prefill、单活动请求、容量准入 | 排队、公平完成、容量不足报错与异常清理 |
| 16GB 显存或当前 GPU kernel 不可用 | 小输入起步、HF/nano 顺序运行、视觉 profile 后分配 KV | 实机 FlashAttention Prefill/Decode smoke 与峰值显存 |
| Transformers 开发版接口漂移 | 固定提交、模型 revision、依赖与参考 fixtures | 不将旧 requirements 快照当作当前可复现环境 |

保留现有 Qwen3 模型、Attention、线性层与 block 分配机制，新增的分支只处理 VL 语义差异。视觉模型自实现比直接调用 HF 增加逐层对齐工作，但符合既有路线且可清晰暴露 Scheduler/Runner/KV 的适配问题。首版牺牲吞吐与功能覆盖换取可定位的正确性证据；不能将 P3 完成表述为视频、并发或完整 Qwen3-VL 特性已支持。

## 审核点

沿用既有草案记录的路线 A；本轮用户已确认纯文本→单图完整Prefill→单图Decode的首轮范围，多图、视频和并发后续扩展。以下细节仍为方案建议，待整份设计批准：单活动请求准入、VL前缀缓存禁用、显式residual、HF版本固定到当前本地HEAD、视觉PyTorch SDPA、分进程顺序对照，以及4096总token/256视觉token的初始验收上限。设计批准后再写正式逐任务实施计划；本轮不安装依赖、不修改生产代码、不提交或推送Git。测试阈值、profile余量和辅助函数拆分留给实施者按规定实验确定，不能自行扩大输入范围或跳过验收。

此前草案记录的 builder check 已补充“单个活动请求”的准入约束。本轮更新后的独立 builder check 结果为 `nothing important left`：审阅者核对 Scheduler、BlockManager、Sequence 与 LLMEngine，确认单活动请求、完整 Prefill、全程 KV 容量准入、禁用缓存查找与发布、异常释放及阶段验证约束足以指导实施。该结论只表示设计可供规划；源码复核与CPU Processor检查不代替尚未完成的GPU数值验收。

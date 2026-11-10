# Qwen3-VL Native Task13a 实施结果

Task13a 已接通 Prepared CPU 输入经真实 `LLMEngine.add_request/step`、Scheduler 准入、Runner 自编码一次和完整图像 Prefill；实现及本地验证证据如下。独立 Review 已通过，规格符合、质量 Approved，Critical/Important/Minor均0；未自动暂存、Commit 或 Push。

## 实现范围

生产仅修改 `nanovllm/engine/llm_engine.py` 与 `nanovllm/engine/model_runner.py`。Engine 保留既有 Prepared 提前校验、请求私有 CPU 副本和 Scheduler 总预算校验；合法图像可以排队，排队时不搬 pixels、不分配 KV。Runner 始终构造 `text_only=False` 完整模型，通过公开严格 loader 实载 625 源 tensor、没有 visual skip。完整 Prefill 使用请求的 `[3,L]` MRoPE positions；物理 KV slots 仍由独立 block table 决定。只有已准入的完整请求才将 pixels 转 BF16/GPU，原生 encode 一次，再经 `run_model(..., visual_features=..., visual_mask=...)` 完成融合及最后 logits。pixels/features/mask 都是当前 run 局部变量，没有新增模型或 Runner 持久视觉状态。

不完整、cached 或多序列 Prefill 在搬 pixels/执行视觉网络前明确失败。图像 Decode 明确抛出 Task14 未实现异常；真实 public Engine 的第二步也验证该保护，不会悄悄使用纯文本位置。纯文本继续由完整 wrapper 运行，原 P1 Prefill/KV/Decode 与旧 Qwen3 回归保持严格验证。legacy TP 的 pickle 没有 multimodal 字段，新增分支仅在 multimodal Config 下读取该字段，不改 Sequence pickle 协议。

新增 `tests/test_qwen3_vl_prefill.py`。必要最小旧测试调整：`tests/test_qwen3_vl_input.py::test_engine_valid_prepared_explicitly_blocked_before_kv` 改为合法 Prepared 可排队且仍无 KV；`tests/test_qwen3_vl_runner.py::tiny_model_checkpoint` 保持原 text 配置、初始化顺序及 RNG 流，另用独立 RNG 生成真实 tiny HF vision state，替换过渡假视觉权重；对应 full-model 注册、完整实载和无 skip 断言更新。原实际 text tensor 和所有精确数值断言保留。原真实 checkpoint Runner 测试的 `(loaded, skipped)=(310,315)` 过渡审计必须更新为 `(625,0)`，不改变其语言/KV比较。

## 固定参考与小预算

Transformers 固定 `2711ad22c82105e240a2877b92d082fe2857936f`，checkpoint snapshot 固定 `89644892e4d85e24eaac8bacfd4f463576704203`；复用只读 Oracle 的输入，不修改正式 Oracle、fixtures 或 tolerances。参考在独立 HF 进程中全新执行完整图文 forward 并从 HF cache 采样，HF 退出后才启动 native Engine。HF BF16、视觉 SDPA、语言 FA2；native 相同算子后端，TP=1，eager/inference_mode，额外 `TORCHDYNAMO_DISABLE=1`，不使用未批准的宽 BF16 容差。所有浮点比较预先使用 `max_abs=mean_abs=rel_l2=0`，同时校验 shape、dtype 和 finite。

本轮明确配置 `max_model_len=512`、`max_num_batched_tokens=128`、`max_visual_tokens=96`，KV `[2,28,2,256,8,128]`。三图分别为 square：L83/V64/grid[1,16,16]；rect：L115/V96/grid[1,16,24]；same-layout-other-pixels：L83/V64/grid[1,16,16]。用真实 BlockManager 分配/释放一个先行请求旋转 free list，三次图像使用物理 block `[1]`、`[0]`、`[1]`，避免把逻辑位置等同物理 slot。

每图对照 positions、融合后完整 hidden、前三层完整 residual 注入后 hidden、最终 norm、最后 logits 及所有 28 层 K/V 的四个选定位置（首文本、首视觉、末视觉、末文本），共24对张量；严格三指标均0。没有把原生结果自产为 HF 参考。视觉 spy 在入队后为0，每次图像完整 Prefill 后恰为1。非分配 slots 的 KV sentinel 不变。非法 pixels/grid、超过视觉预算、L+max_tokens 和独立 batch token 预算、chunk 输入、未准入完整图像均提前拒绝；真实 GPU KV 全量前后逐元素相同、视觉调用0。

## TDD、故障灵敏度与验证

证据根目录 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13a/`。完整 CPU baseline313 passed10.31s；初次 collection8节点包含 PrefillRoutingTests，最终增补 legacy TP 回归后9节点。最初 CPU NotImplemented/guard失败只作过渡记录。先完成可运行完整 wrapper/Engine 接口桥接、暂不编码或融合后，真实 GPU 行为 RED 到达 `vision_forward_spy.call_count=0 !=1`（1 failed18.72s）。随后聚合记录同一缺失行为的实际数值 RED（1 failed14.65s）：square 最后 logits max_abs7.09375、mean_abs1.2972960607293857、rel_l2 0.43570292962265583，fused max_abs9.8216552734375，并记录 positions/28层KV失配。不是 setup/import 或单纯 NotImplemented。

最小实现后首次 GPU GREEN1 passed17.40s，CPU scoped8 passed，完整 CPU 最终321 passed14.01s；快速 CI31 passed，禁止模型栈导入的快速 CI31 passed。新发现 legacy TP 已反序列化 Sequence 无 multimodal 字段的兼容风险，先加真实 pickle CPU 行为 RED 重现 AttributeError，再将扫描条件限定于 VL，保持旧协议不变。

三项进程内受控数值错误均执行真实 CUDA forward 并 exit1，不改生产源码或正式参考：视觉区 height positions 错移使 logits max_abs0.4453125；融合 mask 错移使 logits max_abs2.83203125；把物理 slots 错归零起点使28层选定KV max_abs375.0。三项均保留输入、输出、逐项指标、原始命令/日志以及生产 SHA 前后一致证据。

最终有关 GPU 首次串行批次21 passed、1 failed，246.30s；唯一失败是原 full-checkpoint Runner 的 `(310,315)` 过渡加载审计，与完整模型实际 `(625,0)` 冲突。获准最小修正后，新图像节点及原失败P1/KV节点 host 覆盖重跑2 passed30.74s，Runner CPU26 passed2.21s。合并覆盖22个相关GPU节点，包含 Runner lifecycle/tiny fixture、旧Qwen3 smoke、Task12 fusion、text和vision；保留首次失败，不将首次批次写为22全部通过。最终图像目录 `routing-vbbu4pvh` 使用全新独立HF参考，24对三指标仍全0，最终test源码SHA `b24ee722561fbc0256e6ddfcb86f39396b293af4b043b0e8a2270602830b264e`。独立Review由另一代理完成，规格符合、质量Approved，Critical/Important/Minor均0。Reviewer独立CPU重算最终24对原始Tensor、核验identity/sourceSHA及JUnit；没有重跑套件/GPU，也未重复root的141项Oracle全量哈希。详见证据目录 `review.md` 与 `root-final-audit.json`。

## 资源与保留边界

正式新图像输出目录、最终来源 SHA、张量 SHA、原始命令、JUnit、故障报告与完整 diff 保存于证据目录；完整报告 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task-13a-report.md` 汇总最终结果。

最终覆盖图像 worker 实测 native peak allocated4,414,346,752B、reserved4,578,082,816B，GPU总量16,610,820,096B。峰值 reset 在模型构造/warmup/KV分配之后，含测试 hook、CPU拷贝前临时对象和额外 KV snapshot，不能当作生产联合 profile 或加载临时峰值。测试局部保留 model/spy 引用，因此不宣称 Engine.exit 后进程内所有 allocation 为0；独立进程退出完成其 CUDA context 回收。旧文本 lifecycle 回归单独验证原退出契约。

本轮只证明上述小预算图像完整 Prefill。Task13b 的视觉联合 profile、最大显存/4096与V256合法边界、Task13c及Task14图像Decode/生命周期均未实施或宣称通过。没有改 shared core、模型/vision/fusion、Scheduler、Sequence、loader、采样器或依赖；没有安装工具、改 Oracle 或扩大模型协议。远程 CI 未运行。

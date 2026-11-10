# Qwen3-VL Native Task12 实施结果

Task12 已实现主视觉替换与前三层 DeepStack 注入，最终本地 CPU/GPU 验收通过；独立 Review 已通过，规格与代码/测试质量均PASS，Critical/Important/Minor均0。本轮没有开始 Task13，也没有暂存、Commit 或 Push。

## 实现与边界

`Qwen3VLForConditionalGeneration(text_only=False)` 注册自实现 `model.visual`，`encode_images(pixel_values, grid_thw)` 返回其 `(main, list[3])`。默认 `text_only=True` 保留此前纯文本构造和运行。embedding 与 LM head 仍为同一 Parameter，完整模式使用公开严格 loader 实际流式加载 625 个源 tensor，没有视觉 skip。

wrapper 在任何 decoder/KV 操作前检查 features/mask 成对、bool 一维 mask、序列长度和设备、视觉数量、main 与全部三项 DeepStack 的二维 shape、隐藏维度、dtype 与设备。所有分支校验完成后，text model 才取 embedding：按 mask 顺序替换主特征，其余 token 不变；decoder0/1/2 各自完成 attention 与 MLP 两次 residual 后，仅向 mask 位置加对应 DeepStack；decoder3 及以后不注入，最后执行 norm。

生产改动仅 `nanovllm/models/qwen3_vl.py`。新增 `tests/test_qwen3_vl_fusion.py`。最小附加范围为 `tests/test_qwen3_vl_text.py` 原“视觉未接通”过渡测试：此前断言本次获批的接口必须抛 NotImplemented，现改为未配对 mask 提前 ValueError，并保留默认文本模式无 visual 注册检查；其他旧测试不变。未改 loader、vision 核心、共享 text 层、Runner、Sequence、Scheduler 或 KV 生命周期。

Task12 证明完整模型直接 forward 的融合行为；真实 image request 的 Engine/Runner 接线、准入、每层图像 KV 验收、Decode 与生命周期仍归 Task13/14。本轮虽在 native forward 使用真实 KV 缓冲，不据此声明 Runner 图像生命周期完成。

## 独立参考与数值协议

固定 Transformers `2711ad22c82105e240a2877b92d082fe2857936f`，checkpoint snapshot `89644892e4d85e24eaac8bacfd4f463576704203`。复用只读 Oracle 输入、positions/delta；不改写正式 Oracle、fixtures 或 tolerances。HF 和 native 在不同子进程串行运行，HF 进程退出后才构造 native，不共存两份完整模型。

HF `from_pretrained(dtype=bfloat16, attn_implementation={text_config: flash_attention_2, vision_config: sdpa})`，随后仅 `.cuda().eval()`，不做破坏 rotary buffer 的整体 `.to(bfloat16)`；实际断言视觉 inv_freq 为 FP32、两个子模型 backend 正确。native 语言 FA2、视觉非因果 SDPA，TP=1、Graph/compile 关闭、inference_mode，记录真实 GPU/版本及 BF16 reduction 设置。

使用预先声明的三指标严格零阈值 `max_abs=mean_abs=rel_l2=0`，通过既有 `assert_aligned` 同时检查 shape、dtype、finite；不采用未批准的宽 BF16 阈值。捕获 next-layer prehook，因此前三项 hidden 是真实注入后的完整 hidden，避开 HF decoder output hook 在注入前执行的陷阱。

两条保留的具名 GPU 测试：

- `FusionTests.test_replacement_and_deepstack` / `hf_fixed_features`：HF 固定 main/三项 DeepStack 输入隔离融合。六图共 60 项张量对照中，24 项为复制的视觉输入，36 项为 native 实际计算的 fused、前三层注入后完整 hidden、最终 norm、最后 logits。
- `FusionTests.test_native_encoded_visual_features` / `native_encoded_features`：实际调用 native `encode_images`，六图共 60 项均来自 native 计算，其中 24 项视觉输出、36 项融合/语言输出，与独立 HF 对照。

所有 case 还显式逐元素验证主特征替换以及非视觉 embedding 不变，positions/delta 精确一致，最后 logits 的 top5、top1 margin 和 argmax 一致性写入 JSON。

| case | L | V | 首次验收 HF top1 margin |
| --- | ---: | ---: | ---: |
| image_square | 83 | 64 | 0.75 |
| image_rect | 115 | 96 | 1.25 |
| image_limit | 275 | 256 | 2.25 |
| image_same_layout_other_pixels | 83 | 64 | 1.5 |
| image_candy | 83 | 64 | 2.0 |
| image_cat | 115 | 96 | 4.25 |

同布局不同像素用例独立运行；不开放产品多图、视频或混合 batch 范围。

## TDD 与初始证据

Task12 证据目录：`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task12/`。

- sandbox 完整 CPU baseline：282 passed、16 Gloo socket setup errors；真实限制 `Operation not permitted`，未按通过处理。宿主完整 baseline：298 passed、28 deselected，9.84s。sandbox CUDA probe=False，GPU gate 真实失败，未 mock/skip；正式 CPU/GPU 使用逐命令宿主权限。
- collection：17 nodes（15 CPU、2 GPU）正常收集，延迟导入重依赖，模块标记 hf_reference。
- CPU 初始 NotImplemented 失败为过渡证据，不冒充有效 RED。随后只打通可运行接口、暂忽略融合，CPU 数值/行为 RED：13 failed、2 passed，1.52s，包含字面算术顺序及 12 项提前拒绝。root 独立重复得到相同结果。
- GPU 有效 RED：两模式均真实构造并加载625源权重、完成 forward，实际 fused 与 main 数值断言失败，square 最大绝对差9.8125；2 failed，39.18s。不是 import、NotImplemented 或 setup 错误。两次 HF 完整六图参考和失败 native 输出、日志/XML均保留。
- 最小融合实现后首次 GREEN：CPU15 passed，1.48s；GPU2 passed，45.91s；六图两个模式所有正式比较三指标均0。
- root 完整 CPU：313 passed、30 deselected，9.88s。快速CI31 passed，1.67s；阻止 torch/transformers/triton/flash_attn 导入的 CI31 passed，1.77s。

CPU 字面测试使用可手算的完整 residual 边界来检查真实 wrapper 的替换/注入顺序，四层输出分别验证；真实注意力/MLP/residual 数值由全权重 GPU 测试验收。无效输入测试监视最早 decoder 入口，确保任何分支无效时没有 decoder/KV 副作用；不是用 fake 结果代替真实 GPU 证据。

## 复现命令

仓库根目录使用现有 `.venv`，不安装或更改依赖。完整命令和退出码保留于原始日志/XML；下列共同环境需用于所有模型测试：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_fusion.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_fusion.py -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_fusion.py -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_text.py tests/test_qwen3_vl_vision_positions.py tests/test_qwen3_vl_vision_block.py tests/test_qwen3_vl_vision_merger.py tests/test_qwen3_vl_vision.py -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python .superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11a/ci_quick_check.py
.venv/bin/python -m ruff check nanovllm/models/qwen3_vl.py tests/test_qwen3_vl_fusion.py tests/test_qwen3_vl_text.py
.venv/bin/python -m ruff format --check nanovllm/models/qwen3_vl.py tests/test_qwen3_vl_fusion.py tests/test_qwen3_vl_text.py
git diff --check
```

CPU/Gloo 与 GPU 命令在当前 sandbox 不可执行对应硬件能力，正式验证须在获准的宿主环境运行。GPU worker timeout360s，进程串行；缺权重/CUDA、NaN、shape差异和未满足门禁均失败，未使用 skip/xfail。worker 原始命令为 `python -m tests.test_qwen3_vl_fusion --worker {hf,hf_fixed_features,native_encoded_features} --output <directory>`。

## 最终验收与资源

最终 CPU313 passed、30 deselected，11.69s；最终 GPU2 passed、15 deselected，51.16s。既有 text 全4节点与视觉位置/block/merger/完整模型共8 GPU回归通过，141.88s。最终正式运行 FAILED/ERROR/SKIPPED 均0；初始真实失败保留原记录，不作删除或倒写。

最终目录 `hf_fixed_features-qq_v5fp0`、`native_encoded_features-odj73d4e` 各含独立 HF/native 六图 CPU tensor、identity、加载审计、逐case三指标/top5/margin及资源记录。120对tensor经额外CPU复核，shape/dtype/finite/exact完全通过；其中96对为native计算输出、24对为fixed模式复制视觉输入，不将复制项算作视觉计算验证。`final-tensor-audit.json` 保存复核与源绑定。

进程内受控错误使用 `--fault mask/order/placement`，不写生产源码：mask错位使fused max_abs6.28125；DeepStack逆序使第0层注入后max_abs19.5625；注入移至decoder0之前使该层输出max_abs12.8125；三者末位logits最大误差分别2.83203125、1.125、1.4375，均触发严格数值断言、exit1。`controlled-errors.json` 保存命令、metrics与生产SHA前后一致证明，三个fault目录保存实际tensor。诊断参考来自此前独立HF完整运行；最终正式验收另行新跑HF，无参考自产或阈值修改。

HF峰值allocated4,331,284,992B/reserved4,372,561,920B；native固定HF特征4,355,613,696B/5,043,650,560B；native自编码4,371,648,512B/5,045,747,712B，远低于本卡16GB。报告同时记录各case free/total。峰值统计在模型加载及native KV分配完成后reset，计入仍存活的权重/KV基线及随后各case forward，但不覆盖加载临时峰值。各worker删除模型、GC和empty_cache后，`weakref(next(model.parameters()))`这一代表性Parameter探针失效；未做逐个参数或全部GPU对象弱引用审计，不能据此声称所有GPU对象已释放。进程内尚有8,519,680B/9,568,256B allocated（native两模式），reserved分别622,854,144B/624,951,296B，不声称进程内allocation归零。HF先退出再启动native；测试不建立跨模型共存或Runner退出语义，完整worker进程退出完成其CUDA上下文回收。

56项基线保护SHA仅授权生产owner和过渡text测试变化；其余54项含vision/sharedcore/旧tests/Oracle顶层内容未变。Ruff check/format与diff检查通过；未安装工具或更改依赖。`source-sha256.json` 绑定最终5个交付文件，`tensor-sha256.json` 绑定原始张量；`implementation.diff` 包含相对BASE=f31bbe9301032449335150a11c7ef50a7c5c1d64的完整5文件diff，含两个未跟踪新文件，`summary.json` 汇总证据。

独立 Review `/root/task12_review` 已完成，规格和代码/测试质量均PASS，Critical0/Important0/Minor0。Reviewer只读核对完整五文件diff、固定HF源码及原始证据，独立复算120对Tensor、三项fault的实际偏差、68个Tensor文件SHA、56项保护记录和最终候选源码；另做9项内存中CPU输入校验探针，均在decoder之前ValueError。额外探针不是正式pytest节点，也不代替真实Gloo/CUDA门禁；Reviewer未重跑GPU或完整CPU suite。无修复请求或延期Minor，不需要二次Review。

执行裁定：沿用干净功能分支；额外修改唯一旧测试是替换与本次批准功能直接冲突的NotImplemented过渡断言，其他旧文件保持保护哈希。精确零门禁限固定BF16/视觉SDPA/语言FA2环境；模型forward不代表Runner接线、图像KV内容、Decode复用、跨请求/异常生命周期或联合预算已验收，这些由Task13/14负责。多图/视频/混合batch/TP/chunking/Graph/compile和跨backend性能/容差不在本轮范围。资源释放仅代表探针、峰值仅reset后窗口，不能外推全对象回收或加载临时峰值。代价是后续任务必须分别验证这些能力；远程CI未运行，不推断Commit/Push授权。Task12本地技术验收完成，未开始Task13。

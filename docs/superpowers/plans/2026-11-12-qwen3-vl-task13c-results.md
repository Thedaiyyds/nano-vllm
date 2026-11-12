# Qwen3-VL Native Task13c 实施结果

本轮实现真实请求的4096总预算/256视觉token边界验收，并对完整VL模型的最长纯文本Prefill作独立HF数值对照。只修改 `tests/test_qwen3_vl_prefill.py`、验证说明及必要的 `ModelRunner._load_model` 初始化；生产差异8行。Task13a/b既有功能复用，不重做实现；图像Decode guard保留，未开始Task14，未安装依赖或暂存/Commit/Push。首次修复后上限GPU验收1 passed33.18s，最终完整CPU324 passed10.34s、相关GPU18 passed149.23s，4项受控故障均命中预期断言；独立Review规格PASS、质量Approved，Critical0/Important0/Minor0。Task13c完成，父计划对应步骤已勾选，Task14仍未勾选。

## 请求、模型和数值协议

基线HEAD为 `162d67d104f338aff3a844335aa02f68f6773e95`、分支 `feat/multimodal`。固定Transformers `2711ad22c82105e240a2877b92d082fe2857936f`，Qwen3-VL-2B-Instruct snapshot `89644892e4d85e24eaac8bacfd4f463576704203`；`provenance()`检查模型全部文件指纹和HF可执行AST，允许既有注释。新HF参考只落本轮 `task13c` 证据目录，正式Oracle、fixture及阈值文件不变。

新增 `PrefillLimitTests.test_maximum_image_and_complete_model_text_prefill`：

- 图像输入复用只读Oracle `image_limit` 的原始275-token Processor结果、pixels/grid/types，再追加3820个确定的文本ID9707，得到L4095、P1024/V256、grid `[1,32,32]`；不是重新生成Oracle或改变图像数据。
- 纯文本输入为4095个9707。两例都经真实 `LLMEngine.add_request` 和一次 `step`，`max_tokens=1/temperature=0/ignore_eos=True`；完整625个源tensor实载、无skip，视觉模块存在、embedding与LM head绑权。图像encoder调用1次，纯文本0次，生成token与最后logits的argmax相等，结束后块全部归还。
- HF/native各自独立加载完整模型、串行运行，BF16、TP1、视觉SDPA/文字FA2、eager、`TORCHDYNAMO_DISABLE=1`；HF单独计算 `get_rope_index`，输入digest跨进程必须精确相同。两者仅保存最后位置logits `[1,151936]`，无 `[4095,151936]` 大投影长驻显存。
- 每例9对张量：FP32 `text_inv_freq[28,64]`（HF独立global频率扩展，native读取28层buffer）、positions、完整fused、前三层DeepStack注入后完整hidden、最终norm、最后logits、28层K/V选定物理槽。全部沿用Task13a `max_abs=mean_abs=rel_l2=0`，缺项/shape/dtype/NaN均不能通过。
- 图像逻辑位置 `[0,8,254,255,256,257,263,511,512,4093,4094]`，映射到真实物理位置 `[256,264,510,511,512,513,519,767,768,253,254]`；文字逻辑位置 `[0,254,255,256,257,511,512,4093,4094]`。物理表分别 `[1,2,...,15,0]` 与 `[0,15,14,...,1]`，不把逻辑位置当物理slot读回自比。池形状 `[2,28,16,256,8,128]`，每例4095个slot唯一，剩余slot的17 sentinel逐位保持。

V257使用完整一致的 `[1,2,514]`、P1028、257个连续image占位符输入，排除坏grid/行数误报。另检查图像L4096+1输出、文本L4096+1和文本L4095+2均拒绝；真实 `allocate` 与encoder spy均0次，queue/block pool不变，实际GPU KV始终为sentinel。这里“分配前”指请求准入前的KV块分配，Runner启动时的全局池已按13b正常创建。

## 真实RED、根因及最小修复

修改前host CPU基线324 passed、32 deselected、10.53s。collection10节点含 `PrefillLimitTests`，不是缺模块/接口错误。首次真实上限GPU运行1 failed31.70s，图像positions/fused相等，但首层输出max_abs0.03125、后两层0.09375/0.84375、norm4.25、logits0.3125、选定KV2.3125；同轮纯文本4095的原8对全部0。该数值断言失败是真实RED，不用setup错误或人为破坏生产代码冒充。

逐边界只读诊断确认视觉main及3个DeepStack、首层q_norm/k_norm/V均严格相等；旋转后Q有56个不同元素max0.0625、K有37个max0.03125。两进程实际HF调用dense FA2、native调用varlen FA2，但相同HF QKV分别回放两种kernel，结果三指标全部0，因此未更换参考后端或修改共享Attention。

根因为Runner在默认CUDA设备上创建文字RoPE常量，HF先在CPU创建再搬到CUDA。FP32 `inv_freq` 的索引19/33/44/56相差最多9.313225746154785e-10，长position放大相位差，跨越BF16 cos/sin舍入边界；纯文本也存在旋转后细微差异，只是该输入的后续结果恰好未放大。独立最小GPU诊断精确重现HF捕获Q/K；只把native频率改为CPU初始化值，Q/K立即严格相等，排除残差/视觉/物理KV映射为初始根因。

`ModelRunner._load_model` 现用既有 `Qwen3VLRotaryEmbedding(config.text_config)` 在显式CPU上下文产生一份微小FP32频率常量，复制到28层现有buffer；没有创建第二份完整CPU模型，不导入HF推理实现、不改模型公式、权重加载或后端。正式新增第9对独立HF频率检查，防止只看logits漏掉初始化漂移。修复后同一上限GPU1 passed33.18s，图像/文字两例共18对三指标全部0，4种非法请求均在分配/encoder之前拒绝。

## 最终资源和输出

RTX5060Ti，设备总量16,610,820,096B，实测驱动595.99.02（`host-environment.log`）；torch2.9.1+cu130、CUDA13.0、Transformers5.21.0.dev0、flash-attn2.8.3、Triton3.5.1、NumPy2.5.4。实际 `allow_bf16_reduced_precision_reduction=True`，不暗改数学设置。新worker使用完整默认联合profile后分配16块；profile预KV峰allocated4,542,145,024B/reserved4,747,952,128B，激活增量276,959,744B。

| 已分配KV的真实请求 | allocated peak(B) | reserved peak(B) | 保守free margin(B) | HF/native首token | top1 margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 图像L4095/V256，max_tokens1 | 5,011,924,480 | 5,200,936,960 | 10,256,820,736 | 9707 | 12.625 |
| 完整模型纯文本L4095，max_tokens1 | 5,007,656,448 | 5,200,936,960 | 10,261,088,768 | 358 | 0.625 |

每例前独立reset峰值，以上为最终18项GPU批次中的真实Prefill本例峰；保守margin为 `free_after−(peak_allocated−current_allocated)`，均大于配置1,073,741,824B余量。首次GREEN图像/文本margin分别10,469,157,376/10,474,080,768B；最终free变化后margin降低，allocated/reserved峰值及所有数值结果不变，不把两轮free混作同一观测。资源计数包含观测hook和选定KV读取，不能当无hook性能数据。报告顶层计数在最后一例reset后采样，包含后续负例sentinel检查，不能当作所有case的最大峰值；使用上表各case数据。没有测图像Decode；4096总预算是本轮承诺边界，不声称显卡最大能力。

证据 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13c/`：`initial.log/xml`、`limits-ocnx9apb/`、`initial-analysis.json`；`diagnostic-hf.log`、`diagnostic-native.log` 与 `diagnostic/comparison.json`；`rope-diagnostic.log`、`diagnostic/rope-initialization.json`；首次GREEN `green1.log/xml`、`limits-dpshd4r8/`。HF/native identity保存固定revision、模型全部文件SHA、软件/数学设置和production/test源码SHA；首次GREEN root重新核算18对tensor，见 `root-first-green-audit.json`。

最终GPU `gpu-final.log/xml`、上限参考 `limits-aluyb3xp/`；root再次独立重算最终18对tensor的shape/dtype/finite/精确数值及source绑定，全部通过。`root-final-audit.json`还确认Oracle141文件SHA不变、受保护的既有文件仅允许的Runner/Prefill test/验证说明3项改变、4类上限请求均拒绝、最小margin10,256,820,736B。`artifact-summary.json`、`frozen-source-sha256.json`、`source/`及 `production.diff/test.diff/implementation.diff` 提供审查入口、冻结副本和具体差异。

冻结源码SHA：Runner `4029d08b9c6c27d968a0ef445c89353d774592442ebe7ff8e348807dc412094e`；Prefill test `8d0cdd3e9b442e33b04b65a590b6e0479b2a3ec34d1844bef7cd52ec03d5e110`。整文件Ruff format/check与diff检查通过；冻结后不在最终测试期间编辑源码。

## 冻结后的回归与受控故障

完整host CPU324 passed、33 deselected、10.34s，exit0（`cpu-final.log/xml`）；相关GPU18 passed、47 deselected、149.23s，exit0（`gpu-final.log/xml`），无failure/error/skip/xfail。quick CPU31 passed、326 deselected、1.74s（`quick-final.log/xml`），禁止torch/Transformers模型栈导入CI31 passed、326 deselected、1.73s（`blocked-final.log`），均exit0。Prefill原8项CPU修复后8 passed、2 deselected、4.35s（`scoped-cpu-fix.xml`）。

4项故障在同一个orchestrator中逐个启动native子进程，前一个退出后才启动下一个；每个worker预期exit1，orchestrator验证全部实际断言后exit0。它只复用 `task13c/limits-dpshd4r8` 新参考的副本，不改正式Oracle。`faults/summary.json` 证明生产和test SHA前后完全相同；4种故障只存在于对应进程中，正常测试路径不启用。

| 进程内故障 | 实际断言证据 | worker退出码 | 耗时 |
| --- | --- | ---: | ---: |
| 图像H位置+1 | 最后logits max_abs0.53125、物理KV12.375超过0 | 1 | 21.33s |
| 融合mask滚动1位置 | fused max_abs13.84375、最后logits6.46875、物理KV154超过0 | 1 | 21.80s |
| 写入物理slot滚动256 | 最后logits仍严格相等，但选定物理KV max_abs168超过0 | 1 | 22.81s |
| 跳过Scheduler准入guard | 仅V257仍被输入层拒绝，3个总预算超限被错误接纳，`over_limit_rejected_before_allocation=False`命中`assertTrue` | 1 | 27.40s |

物理slot故障证明单看最后logits会漏掉KV写错；本轮确实用独立HF的28层物理内容检查抓到它。故障没有用setup/import错误替代行为失败，也没有嵌套启动pytest或出现anyio重写警告。对应完整日志为 `faults/{positions,fusion,kv_mapping,limits}/native.log`。

## 独立Review与交付

独立Reviewer完整读取候选diff、brief和父任务契约，定向检查未变更的MRoPE、比较器及DeepStack捕获helper，独立在CPU读取最终HF/native两例共18对张量并重算三个指标，全部0；同时核对身份/source SHA、输入digest、物理slot/sentinel、负例、最终JUnit及4个原始故障日志。规格PASS、质量Approved，Critical0/Important0/Minor0，无待修复或延期项。原始意见在 `task13c/review.md`；root已复核原始Review并完成计划/ledger收尾。

这是Task13c增量验收；多图、视频、混合batch、chunking、TP/Graph和prefix-cache等既有全局guard仍保留，本轮不重新证明整个分支的全部路径，也不宣称Task14图像Decode已完成。Review后仅更新报告/验证说明/父计划等元数据，源码及测试SHA保持最终GPU运行版本，无需重复无变化的套件。

正式交付5文件为 `nanovllm/engine/model_runner.py`、`tests/test_qwen3_vl_prefill.py`、`tests/qwen3_vl_validation.md`、本结果文档及父计划 `docs/superpowers/plans/2026-10-29-qwen3-vl-implementation.md`；内部报告和本轮HF参考/日志仍位于自忽略证据目录。`task13c/final-manifest.json`绑定5文件SHA、冻结源码、最终张量/身份、JUnit、Review和protected审计；`final-deliverables.diff`包含全部5文件的具体diff。未暂存、Commit或Push，远程CI未执行。

## 可复制的复现命令

在仓库根目录、有真实GPU/NCCL和Gloo本地socket权限的host环境运行。本轮由root串行运行所有GPU命令；不要求sudo或新依赖。缺GPU/权重/参考时测试失败或明确blocked，不以skip替代。

```bash
cd /home/thedai/nano-vllm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_prefill.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_prefill.py::PrefillLimitTests -m gpu --require-gpu -q --junitxml=/tmp/task13c-limits.xml
.venv/bin/python -m pytest tests -m 'not gpu' -q --junitxml=/tmp/task13c-cpu.xml
.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py tests/test_qwen3_vl_prefill.py tests/test_qwen3_vl_runner.py tests/test_qwen3_vl_text.py tests/test_original_baseline.py -m gpu --require-gpu -q --junitxml=/tmp/task13c-gpu.xml
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python .superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11a/ci_quick_check.py
.venv/bin/python -m ruff format --check nanovllm/engine/model_runner.py tests/test_qwen3_vl_prefill.py
.venv/bin/python -m ruff check nanovllm/engine/model_runner.py tests/test_qwen3_vl_prefill.py
git diff --check
```

最终相关GPU collection实际18项（65中47 deselected）：Memory1、Prefill2、Runner9、文本模型4、旧模型2。Runner既有真实32步teacher forcing/greedy文本Decode与分页边界、完整模型加载/失败清理、Task13b联合profile及旧Qwen3回归都在此集合中，未实现图像Decode。

本地受控故障脚本仅供诊断，先选择本轮新生成的HF证据路径；输出目录必须不存在，不覆盖正式Oracle或已有证据：

```bash
.venv/bin/python .superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13c/run_faults.py \
  .superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13c/limits-dpshd4r8 \
  /tmp/task13c-faults-new
```

它将位置、融合mask、物理slot和Scheduler准入四种进程内故障顺序执行，每个native worker须exit1且到达实际数值/状态断言；orchestrator在预期失败全部被证明且源码SHA前后相同时才exit0。源码无故障注入代码之外的变更，正常路径不启用fault选项。

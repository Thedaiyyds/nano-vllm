# Qwen3-VL Native Task13b 实施结果

本轮仅实现联合显存 profile 与 KV 净预算分配。生产文件仅 `nanovllm/engine/model_runner.py`，新增 `tests/test_qwen3_vl_memory.py`；没有修改旧测试、模型、视觉/融合、调度、输入契约、Oracle 或阈值。首次真实 GPU GREEN 与资源验证已通过；完整CPU首轮324 passed14.05s、fix1后324 passed13.69s，相关GPU13 passed110.60s，独立 Review 最终规格PASS、质量Approved，Critical0/Important0/Minor1（M1已记录deferred）；Task13b完成。未暂存、Commit 或 Push。

## 实现与预算

完整模型启动 warmup 调用 `profile_vl_memory()->dict[str,int]`，报告 allocated/reserved peak、清理后的 free、reserve、current、激活增量、净预算、设备总量和 image_case_count。默认分别执行纯文本 L4096，以及语言长度4096、P1024/V256 的 `[1,32,32]` 方图、`[1,2,512]` 与 `[1,512,2]` 两个极端方向，BF16、视觉 SDPA、语言 FA2、TP1、CUDA Graph off；验证额外设置 `TORCHDYNAMO_DISABLE=1`。

profile 使用内部合成 CPU Prepared 输入；依次经过公共 token/grid/pixels/mask 验证、CPU MRoPE、自编码及完整融合语言 forward。唯一有意区别是资源 forward 在 KV 分配前执行、无 Scheduler 准入的完整4096输入；图像不采样，文本沿用原 warmup 的采样调用并立即丢弃结果；它不接受任何用户未准入请求，也不修改公开准入保护。纯文本通过原 `run/prepare_prefill` 路由，仍按 max_model_len 保守测完整上下文，覆盖将来 Decode 容量；图像长度按 min(max_model_len,max_num_batched_tokens)，视觉预算按 min(max_visual_tokens,L−2)，小 Config 不强塞不可容纳的256视觉 token。保留 Task14 图像 Decode 拒绝。

Prepared 公共契约允许合并整除的宽高比256，因此覆盖2×512两个方向；天然图片 Processor 的 smart_resize 原图宽高比限制200独立存在，没有新增输入限制或静默缩小合法范围。tiny checkpoint 保留了超出 vocab64 的视觉保留 token ID，数学上没有合法图像 Prepared 输入，因此仅执行文本 profile、image_case_count=0；正式 checkpoint 必须三图，测试固定断言不允许跳过。

完整资源 forward 的局部栈退出后，才 synchronize、GC、empty_cache，并测 current/free。任何 logits/features/pixels 都不返回或挂到 Runner/模型；避免存活输出使 `peak-current` 抵消本应预留的激活。净预算是 `floor(total*utilization)−(total−free_after)−(peak_allocated−current_allocated)−reserve`；分配时重读 free/total 和当前 reserve，复用已测激活增量，避免外部显存或配置变化导致旧净预算失效。select_kv_block_count 消费带符号净预算，负值直接拒绝，容量不足直接报错；足够时完整分配 ceil(max_model_len/block_size)，默认16块、448MiB，余量只减一次。直接 allocator 调用若无 profile，会执行同一真实 profile，无需伪造旧测试状态。

## TDD 与证据

证据目录 `.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13b/`。初始 collection3节点包含 MemoryTests，补充 CPU 公式后为4节点。root host baseline 在新测试已写入后运行：321既有CPU通过、3新CPU失败、32 deselected，不能写为整轮全绿；其中负净预算 clamp 和错误重算净预算是有效行为 RED，缺 profile 方法是接口准备失败、不计行为 RED。

真实 GPU 行为 RED：完整 Runner 构造成功后，视觉调用0≠3，1 failed3.18s，不是 setup/import/NotImplemented。CPU 最终 scoped3 passed2.16s。首次 GPU GREEN1 passed26.70s。快速 CI31 passed1.78s，禁止模型栈导入快速 CI31 passed1.78s；scoped Ruff 与 git diff --check 通过。进程内受控故障、不改源码：激活抵消使公式断言失败；删除 reserve 与 clamp负预算均使 signed-net/精确净预算断言失败，原始日志和JUnit保留。真实 GPU live-output 故障将真实 forward 的 logits 留在进程列表，mem_get_info 前 weakref 断言失败，1 failed12.33s。四项故障均为进程内修改，源码不变；嵌套 pytest 产生的 anyio PytestAssertRewriteWarning 为故障诊断噪声，正式 GREEN 无此警告。完整 CPU 最终324 passed14.05s；最终13项相关GPU串行批次13 passed、46 deselected、110.60s，无失败/错误/跳过；覆盖新显存测试、13a图像Prefill、原Runner/lifecycle和旧Qwen3 smoke。

本轮只做资源容量和峰值验证，不替代 Task13c 的最大输入数值对齐，也未开始 Task14 图像 Decode 或生命周期扩展。没有租卡、安装依赖或运行远程 CI。

## 首次实机资源结果

RTX 5060 Ti，设备总量16,610,820,096B。固定 checkpoint/HF revision 及 model 文件 SHA、软件版本、完整实载625/no skip、模式和生产/测试 source SHA 在 `resources-first-green.json` 内；格式整理后 final source 已冻结，最终资源报告已重新绑定该 source，数值与首次GREEN一致。所有8次资源 forward 的最后 logits `[1,151936]` 均 finite，无 HF 数值对齐主张。

| 阶段/输入 | allocated peak(B) | reserved peak(B) | 实测保守 free margin(B) |
| --- | ---: | ---: | ---: |
| 分配前联合4case | 4,542,145,024 | 4,747,952,128 | free_after=11,650,531,328 |
| 已分配 KV，纯文本4096 | 5,003,466,240 | 5,196,742,656 | 10,574,608,896 |
| 已分配 KV，方图32×32 | 5,011,923,456 | 5,200,936,960 | 10,561,957,376 |
| 已分配 KV，宽图2×512 | 5,011,923,456 | 5,200,936,960 | 10,561,957,376 |
| 已分配 KV，高图512×2 | 5,011,923,456 | 5,200,936,960 | 10,561,957,376 |

预 KV current4,265,185,280B，激活增量276,959,744B，reserve1,073,741,824B，净预算8,638,747,750B；KV实际 `[2,28,16,256,8,128]`，469,762,048B。分配后各 case 独立 reset 峰值；margin=`free_after−(case_peak_allocated−case_current_allocated)`，最小10,561,957,376B显著超过1GiB。预 KV JSON 每case的峰值是累计峰值，不误当四次独立峰值。本次指定模型、单活动请求、完整 Prefill 的1GiB余量有充分实测空间，无需租卡或缩小承诺范围。

峰值测量在权重实载之后开始，不覆盖启动权重加载临时峰值；进程成功启动说明本机该加载可行，但不把联合 forward 峰值称为加载峰值。分配后测试不覆盖 Task14 图像 Decode，不能据此宣称 Decode 全链路已验收。测试 finite 检查增加少量临时 allocation，报告保留实际观测值。

root最终独立审计资源公式、full4096/P1024/V256/16block边界、冻结来源及Oracle141文件SHA全部通过。首轮13节点正式生产/test源码SHA分别为 `dfa3ffb94f9c52037a4b5a153b7d020cdb389a579763e51b1c5a89b96d3d1c28`、`aed64dadfca59445dd12379ef9c98aea4cf44912e3b51c4123738535b28802ec`。独立Review已完成，首轮I1关闭，M1 deferred。

## 独立 Review fix1

首轮Review：Critical0、Important1（I1）、Minor1（M1），规格不通过、质量Needs fixes。I1指出父任务要求启动失败给出预算，负净预算被拒绝但通用异常丢失实际负数，初始化清理后调用方无法读取profile。核实符合父任务要求后，先加强已有negative test为同时匹配 `nonnegative` 与 `-1`，真实CPU RED1 failed1.48s；只给现有异常增加 `got {available_bytes!r}`，不改公式、不clamp、不改GPU路径，完整新模块CPU GREEN3 passed1.56s，Ruff/diff检查通过。差异与原始命令/日志/JUnit保留 `review-fix1.diff`、`fix1-red.*`、`fix1-green.*`。M1只涉及故障harness的anyio警告，root记录deferred，不扩大本轮修复。

fix1后冻结生产/测试SHA：{"nanovllm/engine/model_runner.py": "161049c26f6f9520c7d54979647f385ecf1cf46e48a56f7dd1c1a58d46cc8485", "tests/test_qwen3_vl_memory.py": "03f5530609a20b5641f0d67b0eb2dc53076bcf8d004970cde7507ba5c9d6a279"}。先前完整CPU324/相关GPU13结果保留，fix1后完整CPU324 passed13.69s，真实资源GPU1 passed27.60s，原始日志/JUnit为 `cpu-fix1-final.*`、`gpu-fix1-final.*`；`resources.json`已重新绑定fix1冻结生产/测试SHA。原13项GPU计算回归保留，异常文字修复没有改变计算路径。定向复审规格PASS、质量Approved，Critical0/Important0/Minor1（M1 deferred），见 `review.md` 和 `review-fix1.md`。Reviewer只读审查及独立公式/SHA核验，没有重复GPU或套件。

## 复现命令

在仓库根目录执行。真实 GPU/NCCL 及完整 CPU 中的 Gloo 测试需要允许相应设备/本地 socket 的 host 环境；本轮由root获准后执行。GPU测试串行；显存报告输出在证据目录 `resources.json`。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py::MemoryTests -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_memory.py tests/test_qwen3_vl_prefill.py tests/test_qwen3_vl_runner.py tests/test_original_baseline.py -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python .superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11a/ci_quick_check.py
.venv/bin/python -m ruff format --check nanovllm/engine/model_runner.py tests/test_qwen3_vl_memory.py
.venv/bin/python -m ruff check nanovllm/engine/model_runner.py tests/test_qwen3_vl_memory.py
git diff --check
```

最终整文件格式检查发现仅新净预算表达式的换行需要整理；已格式化整个owner文件及新测试，`ruff format --check` 显示2 files already formatted，Ruff与diff检查通过。前后 AST 完全相同，证据 `pre-final-format-model_runner.py`、`final-format.diff`、`final-format-audit.json`；先前仅整理新profile范围，没有将其误记为整文件格式验收。格式整理不改变计算；root追加真实资源GPU1 passed25.86s（gpu-formatted-final.log/xml），resources.json绑定最终生产SHA22565e823e8838b2e5827a0e3f4342e9ddf33c6ad2c75e853e1aef800404f99f及测试SHA03f5530609a20b5641f0d67b0eb2dc53076bcf8d004970cde7507ba5c9d6a279，全部资源数值不变。root独立复核公式/来源，Reviewer格式补充审查Approved，最终Critical0/Important0/Minor1（M1 deferred）。不重复无变化的完整CPU/13项计算回归。

## 2026-11-12 Minor M1 专项关闭

M1 经局部 harness 修复关闭；独立专项 Review Approved，C0/I0/M0。根因是 pytest 初始化前模型依赖链提前导入 anyio，单次 pytest.main 即可复现；修正此前“嵌套 pytest”归因。故障注入移至 sessionstart，生产计算/正式测试/Oracle/阈值未改。新回归 RED3 failed→GREEN3 passed，严格警告回归3 passed；host完整CPU324 passed、正式资源GPU1 passed，真实GPU故障仍命中预期生命周期断言且无警告。详见[专项报告](2026-11-12-qwen3-vl-task13b-m1-results.md)及证据 `task13b/m1/review.md`。首轮与fix1的Minor1 deferred结论保留为历史，本次更新为closed。未Commit/Push。

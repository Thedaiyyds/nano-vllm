# Task 4：VL 权重映射与严格加载审计

2026-10-30，仅Native执行已批准Plan的Task4。分支`feat/multimodal`，HEAD仍为`f331e43406de157049028e716bca61926a38378e`。保留全部Task2/3未提交实现、报告和用户实验文件；未安装/升级依赖、未修改HF/Oracle或CUDA环境，未暂存、Commit或Push，未开始Task5。

## 实现说明

`nanovllm/utils/loader.py`新增`load_qwen3_vl_weights(model, path, *, text_only=False)`及`WeightLoadReport`。报告的`loaded_keys`是实际成功复制的checkpoint源key，`skipped_keys`是显式跳过的视觉源key，`tied_aliases`是已验证共享Parameter的模型别名（本checkpoint为`lm_head.weight`）。

加载分两阶段：

1. 只读safetensors header，根据模型参数建立期望源key及shape，逐项审计覆盖率。仅`model.language_model`下的`self_attn.qkv_proj`和`mlp.gate_up_proj`展开为Q/K/V及gate/up子shard；视觉qkv和其他视觉参数保持直接映射。缺少任何子shard、未知key、重复文件key、模型映射冲突及shape不符均在写入任何参数前拒绝，错误包含相应key及类别。
2. 审计通过后，逐文件、逐tensor读取并调用现有参数weight_loader；完成一次复制就释放源tensor引用，不缓存全量state_dict。public loader拒绝meta目标，防止PyTorch无数据复制仍被报告为加载成功。private header audit允许meta结构，且`loaded_keys`保持为空。

embedding/head必须为同一个`Parameter`，仅共享底层storage不算合法绑定；源embedding加载一次后，head共享相同参数。其他未知参数别名或重复源映射拒绝。`text_only=True`仅允许跳过`model.visual.`，语言漏项和其他命名空间错误照常失败。当前工具限制TP=1。

元数据错误在任何写入前失败；运行时copy/device异常可能留下部分加载状态，错误信息带源/目标名称，调用者必须丢弃该模型。这不是运行时失败回滚机制，生命周期处理归后续Runner任务。

旧`load_model()`及`default_weight_loader()`仅有格式换行，AST与HEAD完全一致。另用真实旧接口加载小型packed Q/K/V验证行为保持，没有为VL改动旧模型加载路径。

## TDD、测试与625-key证据

先写测试及可导入的空报告stub，正常收集10项；宿主CPU实跑为**10 failed**。核心RED为缺少K shard未抛ValueError、错误shape/未知key未拒绝、报告及625-key审计返回空；没有把ImportError、setup失败或NotImplementedError算作RED。实现后10项通过。

自查新增重复模型映射回归：模型同时保留`q_proj`与`qkv_proj`时，源字典会覆盖前一个目标；实跑**1 failed（未抛异常）**。将所有目标注册统一经过冲突检查后通过，Task模块达到11 passed。

独立Review指出public loader会把meta无数据复制报告成成功。新增回归先得到**1 failed（未抛异常）**，随后在public路径的payload读取/复制前拒绝meta目标；同一回归同时确认private audit仍接受meta。修复后Task模块12项及全套CPU重新通过。

| 验证 | 本次真实结果 |
| --- | --- |
| 开发前全部CPU基线 | 100 passed，8 deselected，8.49s |
| 最终Task4 CPU | **12 passed，1.65s** |
| 最终全部CPU | **112 passed，8 deselected，8.41s** |
| 最终快速CPU | **20 passed，100 deselected，1.66s** |
| Ruff check/format、git diff --check | 通过 |
| 测试模块阻断torch/transformers后的导入 | 通过，收集阶段不依赖模型栈 |
| 原loader/default callback AST比较 | 与HEAD一致 |
| GPU | NOT RUN；本任务为CPU加载和header审计，无GPU数值验收 |

微型safetensors测试实际使用原生`QKVParallelLinear`及`MergedColumnParallelLinear`，初始化真实单rank Gloo，不mock分布式rank或packed写入。测试检查所有加载结果、视觉qkv/bias、文本模式只跳过视觉、错误发生前参数保持哨兵值，以及源tensor的weakref释放。流式测试包装真实safetensors reader观察读取，不替代payload；每次读下一个tensor前，上一个源tensor必须已释放，最终10个源tensor均无存留引用。

Gloo初始化在默认沙箱的只读探测中报`Cannot resolve 127.0.0.1 to a (local) address`；在获得宿主本地socket权限后，真实rank0/world_size1初始化及所有CPU测试通过。测试限定`GLOO_SOCKET_IFNAME=lo`并在结束时恢复环境、销毁自己创建的进程组。不将该基础设施探测失败计作TDD RED或模型错误。快速CPU排除了hf_reference，因此不初始化Gloo。deselected项不是已运行通过项；没有以skip/xfail替代验收。远端GitHub CI本次未运行。

真实checkpoint固定目录为`Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`。测试从固定HF配置构建meta参数结构，再用原生packed线性层替换语言投影，作为独立名称/shape schema；只调用header audit，reader若尝试`get_tensor()`就使测试失败。

| header归属 | 实际数量 |
| --- | ---: |
| checkpoint源keys | 625 |
| 语言源keys | 310 |
| 视觉源keys | 315 |
| packed后的不同参数目标 | 541 |
| 已验证tied alias | lm_head.weight |
| header audit payload读取 | 0 |
| header audit loaded_keys | 空 |

每个packed子shard均单独核对，不能以625总数掩盖缺K。header SHA256为`a4c22c14e0d987dd572a283e92643374babeccc2f86cf2f8020e6144389c3f4d`。JUnit保存上述计数，`header-audit.json`关联源文件digest及header身份。该SHA仅覆盖header，本任务没有重复计算完整4GB权重指纹或声称真实完整权重实载成功；Task5/11继续承担实载验收。

## 独立Review与裁定

独立reviewer只读审查完整loader diff、新测试、Plan/Design和11-pass日志，并自行以现存小型checkpoint复现meta成功报告问题。初次结论为**With fixes**：无Critical，1项Important/P2，无Minor。按既有Native流程，主代理完成一次Review修复的RED→GREEN和最终全套回归；没有要求第二次Review，也不将初次结论改写为“初版无问题”。目前该问题已有真实通过的回归证据，无已知未修复的重要问题或延期Minor。

Reviewer未重跑需要Gloo socket的整套测试；其独立实测仅为meta问题的只读复现，其余依据源码、日志和JUnit。最终12项及全套112项由主代理实跑。

实施裁定：复用当前feature Native工作区，目标loader与既有Task2/3及用户实验不冲突；本轮只新增VL加载/审计工具及测试，不改Runner或模型。single-rank测试fixture先保留在本模块，避免提前扩展测试基础设施。

接受Reviewer明确留给后续的边界：完整625 tensor实载/GPU数值、TP>1、Task2/3既有改动不属于本轮；源dtype与目标dtype按PyTorch copy规则转换；固定本地checkpoint两遍audit/load之间被外部替换未在本轮加入锁定或文件替换防护。调用期间应保持checkpoint文件稳定；这些边界不等于后续GPU/模型通过。

持久HF Oracle未被改动或重采；本任务不使用`tolerances.json`，宽BF16包络仍未批准作正确性自动放行标准。

## 复现命令与交付文件

在仓库根目录、已有venv执行。Task4和完整CPU命令需要允许Gloo本地socket的宿主终端；不要求GPU，不安装或下载：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_VL_ARTIFACTS="$HOME/nano-vllm-artifacts/qwen3-vl"
export QWEN3_VL_ORACLE="$QWEN3_VL_ARTIFACTS/oracle-reviewed"
export QWEN3_VL_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"

.venv/bin/python -m pytest tests/test_qwen3_vl_loader.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_loader.py -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/utils/loader.py tests/test_qwen3_vl_loader.py
.venv/bin/python -m ruff format --check nanovllm/utils/loader.py tests/test_qwen3_vl_loader.py
git diff --check
```

本机持久化证据：`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task4/`，包括`baseline.log`、`collection.log`、`red.log/xml`、`green.log/xml`、`mapping-red.log`、`green-final.log/xml`、`review-red.log`、`task-final.log/xml`、`cpu-final.log/xml`、`quick-final.log/xml`、`header-audit.json`、`review.md`及含新增文件的`task4.diff`。该目录自忽略，不提交产物。

最终Python SHA256：

```text
8ff245b6a36ea4baa942a4ef414bc6e53b1265b0cd78f86e7ab220a940a179d1  nanovllm/utils/loader.py
f9db4ee82762388857761950090a5f10f1f36689bc535215337b02f360b25d4f  tests/test_qwen3_vl_loader.py
```

Task4交付清单：`nanovllm/utils/loader.py`、`tests/test_qwen3_vl_loader.py`、本报告、实施计划状态及Task4勾选更新。拟提交信息：`feat: audit and stream Qwen3-VL checkpoint weights`。技术验证及Review问题修复完成，工作区保留待审阅；未暂存、Commit或Push，本轮止于Task4。

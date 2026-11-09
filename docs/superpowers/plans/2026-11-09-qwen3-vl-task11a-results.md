# Task11a：非因果视觉Attention与单个ViT block

2026-11-09，从`feat/multimodal`、BASE=`ffeddc28889ef3fbbb76902a3fbf3dd28628821d`继续Native实施。Task1–10已完成，Task9+10已提交；本轮不重做旧任务。根目录、上级目录及目标源码/测试子树未找到AGENTS.md，已阅读批准的设计、实施计划、progress.md与Task10报告和独立Review。原个人未跟踪文件保持原状。

本轮只扩展`nanovllm/models/qwen3_vl_vision.py`，新增`tests/test_qwen3_vl_vision_block.py`，更新计划并交付本报告。范围止于11a；主/DeepStack merger、24层堆叠、视觉融合和Runner图像执行留11b以后。底层分段测试不表示开放产品多图/视频入口。不暂存、Commit或Push。

## 实现与接口

`Qwen3VLVisionAttention(vision_config).forward(hidden_states, cu_seqlens, position_embeddings)`消费`[P,1024]`及每图/时间组累计边界。biased QKV在token维投影，reshape为`[P,16,64]`，复用Task10的FP32 Q/K旋转并恢复BF16，然后逐段执行`torch.nn.functional.scaled_dot_product_attention`：`is_causal=False`、`dropout_p=0.0`、scale=`head_dim**-0.5`，拼接后输出投影。按HF相同转置、contiguous及dtype顺序执行，不使用语言Attention或KV cache。非法累计边界在运算前拒绝，必须从0到P且每段非空。

`Qwen3VLVisionBlock`沿用HF属性名，包含eps=`1e-6`的`norm1/norm2`、`attn.qkv/proj`、`mlp.linear_fc1/linear_fc2`，MLP为biased Linear→GELU-tanh→biased Linear。先`x+attn(norm1(x))`，再`x+mlp(norm2(x))`，保留BF16两个完整residual加法舍入点。不新增权重映射表，不改Loader或文本模块。

## TDD与数值证据

预先声明严格组件门禁：相同设备、BF16、固定HF SDPA参考，max_abs/mean_abs/relative L2全部0；此策略延续先前组件的更严格局部门禁，不读取、批准或放宽正式Oracle中尚未批准的宽BF16阈值，也不宣称跨内核bitwise策略。

先建立可导入、可严格加载真实权重的identity-forward脚手架，正常collection收集11个节点，含`VisionBlockTests`。初始CPU **10 FAILED，0.58s**，到达字面值非因果Attention、residual、独立标量GELU与非法边界断言。真实GPU **1 FAILED，26.89s**，严格加载真实block-0权重后到达`assert_aligned("vision_block_0", ...)`，**max_abs=18.625>0**。这是有效数值RED，不是setup/import/schema失败。

最小实现后CPU **10 PASSED，0.59s**，GPU **1 PASSED，41.27s**。GPU节点内含10例：方图、矩形、上限P=1024、相同布局不同像素、candy、cat、两张不同尺寸图及仅改变第二张图、两个时间组及仅改变第二组。两图/时间组是组件隔离测试，入口限制不变。

每例保存9项完整CPU张量：block-0输出、norm1、QKV、旋转Q、旋转K、attention输出、norm2、MLP输出、patch+absolute输入。每项核对shape、dtype、有限值和三项零误差。HF/native各自从同一只读Oracle像素构建输入；HF使用其独立patch/位置/旋转函数。两者分进程串行运行，HF进程退出后才启动native。按case保存输入digest、grid与cu_seqlens，HF与native真实权重均strict加载。

只读使用固定checkpoint snapshot `89644892e4d85e24eaac8bacfd4f463576704203`、HF提交`2711ad22c82105e240a2877b92d082fe2857936f`。现有provenance核验checkpoint指纹、HF源码revision及可执行AST，保存环境、源码SHA。实际消费block-0全部12个key和patch/absolute3个key，合计15项；不声称24层或全部视觉checkpoint keys已消费。

同形状两段计算中，修改第二段像素后第一段输出bitwise一致，第二段确实改变。另在独立诊断进程中施加仅测试用途的受控错误，不改生产源文件：

| 受控错误 | 有效断言失败 |
| --- | --- |
| 把两段合成全局attention | 手工均值期望不符，max_abs=4.0 |
| 改为causal=True | 第一token无法看到未来值，max_abs=1.0 |
| 跳过Q/K RoPE，真实GPU | block-0 max_abs=1.5703125>0 |

扰动后均恢复正常计算。独立Review后修复测试标记问题，再以最终源码指纹完成正式验收；初始RED与修复前GREEN证据保留，不覆盖历史记录。

## 验证记录

| 检查 | PASSED | FAILED/ERROR/SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | --- |
| 初始宿主完整CPU基线 | 280 | 0/0/0 | 25 | 9.78s |
| Task11a CPU GREEN | 10 | 0/0/0 | 1 | 0.59s |
| Task10+11a最终CPU | 21 | 0/0/0 | 2 | 1.87s |
| 最终完整CPU回归 | 290 | 0/0/0 | 26 | 9.84s |
| 最终快速CPU | 31 | 0/0/0 | 285 | 1.68s |
| 禁止模型栈导入的快速CPU | 31 | 0/0/0 | 285 | 1.66s |
| Task11a真实GPU首轮GREEN | 1 | 0/0/0 | 0 | 41.27s |
| Task11a真实GPU正式验收 | 1 | 0/0/0 | 0 | 42.34s |
| Task10真实GPU前置回归 | 1 | 0/0/0 | 0 | 23.21s |

GPU一项内执行10例，不能称为10个pytest节点。先前沙箱CPU为264通过、16个Gloo Socket setup error；CUDA False/0。宿主只读探测CUDA True/1、RTX5060Ti后，按命令授权重跑完整CPU与真实GPU；未mock/skip或为环境错误改断言。没有新依赖安装、驱动或全局权限修改。Ruff check/format两个文件与git diff --check通过。

首轮GPU证据`task11a/gpu-5_kv7k65/`：HF peak allocated **91,142,656 bytes**、reserved **113,246,208 bytes**；native peak allocated **84,850,176 bytes**、reserved **111,149,056 bytes**。对应最小采样空闲显存15,821,307,904 / 15,823,536,128 bytes。统计是从循环前重置后的诊断进程累计，包含patch、位置表、block与对照/采样开销，**不是完整视觉模型或端到端性能/峰值结论**。两进程参数弱引用均确认释放；进程退出释放上下文，不声称上下文零占用。

已有源码/测试/Oracle保护哈希审计中，仅目标vision模块按预期改变；Oracle未重采或写入，旧Qwen3基准及所有前置测试文件保持不变。最终正式GPU目录`task11a/gpu-jdievzxj/`再次完成十例、90项组件比较的三项零误差，并记录最终测试标记修正后的指纹；两段隔离和参数释放检查均通过。

## 独立Review、修复与裁定

Reviewer `/root/task11a_review`以fresh context只读审查完整四文件diff、批准设计/任务边界、固定HF源码及执行证据，结论**With fixes：Critical0、Important1、Minor0**。生产实现与计划相符；Reviewer独立解析10例×9组件零误差、输入摘要/分段/源码指纹一致、隔离和释放标志，以及JUnit。未独立重跑GPU，不将证据解析称为重新计算Tensor。

唯一Important为新CPU测试缺少重型测试标记：本地开发环境有torch，首轮快速CPU虽然41通过，轻量CI仅安装测试工具，实际上会运行这10项并缺torch失败。执行者自查与Reviewer均确认实际workflow；禁止模型栈导入的完整快速命令取得**10 FAILED、31 PASSED**，随后按既有vision_positions约定增加模块级`hf_reference`标记，修复后**31 PASSED、285 DESELECTED，1.66s**。这使重型CPU测试进入本地完整CPU验收，快速CI保持既有31项；没有安装torch或扩展CI依赖。最终完整CPU290项、所属CPU21项及正式GPU十例均覆盖该修正。按Native单次修复通道闭环，无二次Review，无遗留Critical/Important或延期Minor。

执行裁定：沿用当前干净功能分支，不搬移个人实验；同设备、同BF16/SDPA零误差仅为此组件验收，不替代跨内核校准。接受Reviewer未裁定范围：24层堆叠/选定后层、main/DeepStack merger及全部visual keys、产品多图/视频、融合/Runner/生成及端到端资源性能、TP/compile/Graph/其他dtype或checkpoint/跨kernel bitwise、宽BF16阈值批准，均保持后续或待用户批准。成本是本次单block正确性不能证明这些能力，后续任务必须各自实测。正式Task11a完成，下一未完成项为11b；本轮不启动。

## 复现与待提交范围

仓库根目录、现有venv、本地权重，无网络或新安装：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_block.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_block.py -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_block.py::VisionBlockTests -m gpu --require-gpu -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py::VisionPositionGPUTests -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/models/qwen3_vl_vision.py tests/test_qwen3_vl_vision_block.py
.venv/bin/python -m ruff format --check nanovllm/models/qwen3_vl_vision.py tests/test_qwen3_vl_vision_block.py
git diff --check
```

本轮正式四文件：vision模块、vision_block测试、实施计划、本报告。拟Commit说明`feat: add segmented Qwen3-VL vision attention and block`；未获人工批准不暂存/Commit/Push。持久化忽略证据目录`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11a/`保存原始RED/GREEN日志、JUnit、逐组件张量、provenance、显存、输入与源码摘要、扰动脚本与失败记录及保护审计，不加入Git。

# Task11b：主merger及DeepStack post-shuffle merger

2026-11-09，沿用`feat/multimodal`，BASE=`60ca881b95e7d6fff8dad627ac89f62624dfddde`。Task11a已完成、Commit并经用户确认Push，不重复实施；本轮开始时跟踪工作区和暂存区干净，原个人实验文件保留。AGENTS.md在仓库、上级及目标子树均未找到；执行依据为已批准设计、实施计划、progress.md与已验收Task11a及其独立Review。

本轮范围仅为Task11b：在`nanovllm/models/qwen3_vl_vision.py`末尾追加`Qwen3VLVisionPatchMerger`，新增`tests/test_qwen3_vl_vision_merger.py`，更新计划并交付本报告。原Task10/11a生产代码前缀逐字节保持不变，不改Loader、Runner、文本模型或旧测试。不实现24层视觉堆叠、图文融合、产品多图/视频或图像生成；下一项11c未启动。本轮不暂存/Commit/Push。

## 实现说明

`Qwen3VLVisionPatchMerger(vision_config, *, use_postshuffle_norm=False)`消费Processor块顺序下的固定`[P,1024]`中间hidden，返回`[P/4,2048]`。尺寸从config推导，实际`spatial_merge_size=2`，四个连续patch拼成4096维，不另做raster重排。

- 主merger：1024维LayerNorm（eps=1e-6）→view为`[V,4096]`→biased Linear 4096→4096→普通GELU（`approximate='none'`）→biased Linear 4096→2048。
- DeepStack merger：先view为`[V,4096]`→4096维LayerNorm（eps=1e-6）→同样的两层Linear与普通GELU。不能复用ViT MLP的tanh近似GELU。
- 属性名`norm/linear_fc1/act_fn/linear_fc2`沿用HF；同一个类通过flag表达两种norm顺序。三组DeepStack各自加载独立checkpoint权重，模型中的三组注册及层提取由11c负责。

## TDD与数值验收

测试首先声明与此前组件一致的严格局部门禁：同一GPU、BF16、固定HF基础算子，max_abs/mean_abs/relative L2全部0。不是启用或批准正式Oracle中待批准的宽BF16阈值。CPU解析期望由Python标量均值、方差和`math.erf`独立计算，FP64误差限预先设为rtol/atol=1e-13，不以HF同构随机模型互证。

Collection正常收集3个节点，含`VisionMergerTests`。可导入、可strict加载的脚手架只返回未归一化/未投影的hidden切片；两个CPU测试到达标量期望数值断言后**2 FAILED，0.60s**。首个真实GPU RED在主路输出失败（1 FAILED，13.11s）。实现前扩充测试收集各组数值失败，以免第一主路失败遮住DeepStack断言；再次真实GPU **1 FAILED，13.19s**，四组维度及真实权重schema均通过，方图输出的有效不匹配如下：

| merger | RED max_abs（门禁0） |
| --- | ---: |
| 主merger | 2720.1640625 |
| DeepStack 0 | 16.232421875 |
| DeepStack 1 | 326.1630859375 |
| DeepStack 2 | 359.701171875 |

最小实现后CPU **2 PASSED，0.62s**，真实GPU **1 PASSED，13.74s**。GPU一项内运行四组merger×六例，共24组对照；每组检查norm、fc1、普通GELU和最终输出，96项计算张量全部零误差，另保存并检查24项输入张量。已独立重新加载全部120对保存张量，复算shape/dtype/有限值及三项指标，均通过。

六例为方图、矩形、P=1024上限、相同布局不同像素、candy、cat。输入直接取持久化只读HF SDPA Oracle的`vision.layer.23/5/11/17`，分别供主路/三组DeepStack；本任务不重新实现或运行上游ViT堆叠。HF和native各自消费相同固定hidden，输入摘要一致。两者分进程串行，每个进程一次只将一组merger放到GPU。所有24个真实参数key均strict加载（每组norm、两层Linear的weight/bias共6个），无随机初始化参数残留，不声称已覆盖全部视觉keys。

固定模型snapshot `89644892e4d85e24eaac8bacfd4f463576704203`、HF提交`2711ad22c82105e240a2877b92d082fe2857936f`，现有provenance核验checkpoint指纹、HF可执行AST及环境。报告保存输入hidden摘要、grid、源key、源码SHA、dtype/backend、峰值及空闲显存。没有网络下载或新依赖。

CPU以相同hidden和相同identity Linear权重验证两条路径，独立期望明确不同。受控错误仅施加于测试诊断进程，不修改生产文件：

| 扰动 | 实际数值断言失败 |
| --- | --- |
| 主路错误地在合并后归一化 | max_abs=1.1908819798402281 |
| DeepStack错误地在合并前归一化 | max_abs=1.1908819798402277 |
| 普通GELU错误替换为tanh近似 | max_abs=0.00015275545538587232 |

错误轴诊断保持合法维度及affine布局，均到达数值断言，不将LayerNorm shape错误称为扰动成功。之后恢复正常代码并执行回归。

## 真实测试结果

| 验证 | PASSED | FAILED/ERROR/SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | --- |
| 既有CPU基线（不收集新增测试） | 290 | 0/0/0 | 26 | 10.22s |
| Task11b CPU GREEN | 2 | 0/0/0 | 1 | 0.62s |
| Task10+11a+11b CPU | 23 | 0/0/0 | 3 | 1.85s |
| 完整CPU回归 | 292 | 0/0/0 | 27 | 9.90s |
| 快速CPU | 31 | 0/0/0 | 288 | 1.69s |
| 禁止模型栈导入的轻量CI快速检查 | 31 | 0/0/0 | 288 | 1.65s |
| Task11b真实GPU GREEN | 1 | 0/0/0 | 0 | 13.74s |
| Task11b最终正式GPU验收 | 1 | 0/0/0 | 0 | 13.58s |
| Task10+11a真实GPU前置回归 | 2 | 0/0/0 | 0 | 64.85s |

所有GPU测试串行。轻量CI标记从测试创建时就存在，完整CPU仍覆盖新测试，没有通过扩大CI依赖规避上次的问题。Ruff按CI的tests范围加vision模块检查与格式检查共23文件通过，git diff --check通过。

初始沙箱完整CPU为274通过、16个Gloo Socket setup error，CUDA False/0；按命令获宿主权限后旧CPU基线290通过，真实GPU测试执行成功。没有skip/mock Gloo或GPU，也没有因环境错误调整断言。GPU为RTX5060Ti，环境沿用torch2.9.1+cu130；Dynamo/CUDA Graph关闭。

首轮GPU证据`task11b/gpu-mexk_7h5/`：HF/native各自累计peak allocated **68,710,400 bytes**、reserved **79,691,776 bytes**，最小采样空闲显存均为**15,857,680,384 bytes**。这些值包含串行各merger、当前输入/输出及诊断开销，不是完整视觉模型或引擎profile；不作为性能结论。每组merger参数弱引用确认释放，不声称所有GPU张量或上下文零占用。53项先前源码/测试/Oracle顶层保护哈希保持不变，目标vision旧代码前缀完全一致，正式Oracle未写入或重采。

最终正式验收目录`task11b/gpu-2l4voqjg/`对应当前生产/测试源码指纹，再次执行四组×六例全部对照。正式`task-done`命令1项通过13.58s，失败/错误/跳过均0；不将24组内部对照报告成24个pytest节点。

## 独立Review与范围裁定

Reviewer `/root/task11b_review`以fresh context只读审查完整四文件diff、批准设计/任务、固定HF源码与验证证据，结论规格符合性PASS、代码质量PASS，**Critical0 / Important0 / Minor0**。无修复请求、遗留问题或延期Minor，不进行二次Review。

Reviewer独立执行CPU **2 PASSED、1 DESELECTED，0.58s**；独立重载120对保存张量复算BF16/shape/有限值及零误差；核对全部24个参数key、层/merger对应关系、输入摘要、源码指纹、53项保护哈希与未改变的旧vision前缀，并检查原始RED/扰动/JUnit/资源证据。Reviewer未重新运行GPU，不能将其证据核对表述为独立GPU推理。之后执行者已完成上述正式GPU门禁。

执行裁定：沿用当前干净功能分支，无冲突则不搬移个人实验；精确零误差仅用于固定BF16/同后端组件，成本是不能外推跨内核或完整模型正确性。接受Reviewer未裁定范围：24层堆叠/三路注册/全部visual keys/上游native hidden提取和组装后的跨图隔离归11c；语言融合、DeepStack注入、Runner及Prefill/Decode归后续任务；产品多图/视频/混合batch、TP/Graph/compile、其他checkpoint/backend、任意非连续或非法内部hidden不在本次契约；整模型资源/性能及CUDA上下文全部释放未宣称；宽BF16正式阈值仍未批准。成本是本轮merger对齐不能证明这些能力，后续需分别验收。Commit/Push须另获用户批准，不因技术验收自动执行。

## 复现命令及交付

仓库根目录、现有venv及本地checkpoint；完整CPU需Gloo本地Socket权限，GPU命令需真实CUDA权限：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_merger.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_merger.py -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_merger.py::VisionMergerTests -m gpu --require-gpu -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py::VisionPositionGPUTests tests/test_qwen3_vl_vision_block.py::VisionBlockTests -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check tests nanovllm/__init__.py nanovllm/models/qwen3_vl_vision.py
.venv/bin/python -m ruff format --check tests nanovllm/__init__.py nanovllm/models/qwen3_vl_vision.py
git diff --check
```

正式交付四文件为vision模块、新merger测试、实施计划、本报告。拟Commit说明`feat: add Qwen3-VL main and DeepStack patch mergers`，未批准不暂存/Commit/Push。证据目录`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11b/`保存日志/JUnit、CPU及GPU原始RED、逐组件Tensor、identity/比较/显存、独立复算审计、受控扰动脚本和失败记录、保护哈希与Review；证据不加入Git。

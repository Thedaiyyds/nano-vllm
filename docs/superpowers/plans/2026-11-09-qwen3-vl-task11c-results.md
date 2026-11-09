# Task11c：完整视觉堆叠与三路DeepStack特征

2026-11-09，沿用`feat/multimodal`，BASE=`09aae57bebedf3b09df1f83cfb382f7a10a38c1b`。Task11b已Commit，用户确认已Push；本轮不重做Task10/11a/11b。开始时跟踪工作区和暂存区干净，个人实验文件保留。仓库、上级及目标子树未找到AGENTS.md，执行已批准设计/Plan与既有Native进度及Review流程。

只在`nanovllm/models/qwen3_vl_vision.py`末尾追加完整视觉模块，新增`tests/test_qwen3_vl_vision.py`，更新计划和本报告。原视觉组件代码前缀逐字节不变。范围止于Task11c和所属M2b门禁；没有语言融合、Runner图像入口、图像Decode、产品多图/视频、TP或编译优化，不推进Task12，不暂存/Commit/Push。

## 实现与接口

`Qwen3VLVisionModel(vision_config).forward(pixel_values, grid_thw)`返回`tuple(main, list[3])`，各路`[V,2048]`、`V=P/4`。构造器复用已有patch、二维RoPE、block和merger，按配置注册24层`blocks`、主`merger`及三组独立`deepstack_merger_list`，新增学习绝对位置表`pos_embed`；名称与HF/实际checkpoint一致。

forward先验证grid及patch总行数，错误在卷积前拒绝；随后按已验收顺序计算patch、绝对位置插值及BF16加法、二维RoPE。每图/时间组的空间长度构成int32累计边界，供所有24层分段非因果Attention。第5/11/17层（零起点）完整block输出立即经过对应post-shuffle merger，第23层输出经过主merger。特征列表是请求局部变量，不缓存跨调用hidden或视觉输出。

CPU以真实小模块和可手算的稀疏patch权重、常量位置表、block bias检查装配：三路merger必须接收到偏移1.5/3.5/6.5的完整block结果，主路为10.5；独立HF位置期望与字面边界`[0,16,40,64]`核对图/时间分段。无效grid四例必须在patch projection之前失败。CPU不以两个随机同构模型互证。

## 权重、参考与数值协议

固定checkpoint snapshot `89644892e4d85e24eaac8bacfd4f463576704203`、HF提交`2711ad22c82105e240a2877b92d082fe2857936f`，沿用原venv，无新依赖/下载。provenance检查checkpoint文件指纹、HF可执行AST、软件版本，保存源码SHA、GPU/CUDA、dtype/backend与输入摘要。

测试独立构造HF与native完整Vision模块，两个进程串行，不加载完整语言2B模型。参数在CPU BF16默认dtype上下文中创建并恢复上下文；HF RoPE显式FP32频率保持FP32，不通过整模块dtype转换将其量化。随后仅移动设备，保持原有HF初始化语义。

测试专用standalone视觉加载器先核对完整视觉key集合和全部shape，再逐tensor从safetensors读入CPU、复制到现有GPU Parameter并删除临时tensor；不保存整份state_dict。**315个视觉key全部实际加载**，另明确列出310个未加载的语言key，不允许其他未知namespace。该加载审计服务于独立Vision组件；公共VL Loader和完整模型注册不在本轮修改，后续Task12集成时仍须审核全模型。

浮点门禁在实现前声明：固定同设备/BF16/SDPA，max_abs、mean_abs、relative L2均为0，检查shape、dtype及有限值。延续此前更严格的同后端组件门禁，不读取或批准宽BF16正式Oracle阈值，不宣称跨内核bitwise通则。正式Oracle只读，没有重采、覆盖或放宽标准。

## TDD证据

先建立可导入、可真实加载且可观察block的装配脚手架：执行已有block但尚无正确位置、分段和中间层tap，三路临时使用末层状态。正常collection收集7项。CPU **6 FAILED、1 DESELECTED，1.82s**，到达位置/边界/选层或拒绝状态断言。真实GPU **1 FAILED，38.41s**，315个视觉key已严格加载后，到达计划指定`vision_layer_17`数值断言，**max_abs=331.75>0**；不是导入、setup或schema失败。

最小装配实现后CPU **6 PASSED，1.73s**，完整GPU **1 PASSED，58.32s**。每例检查patch、全部24个block和主/三路DeepStack，共29项；十二例合计**348项计算张量比较三指标全部0**。执行者另行重载全部保存Tensor独立复算并检查BF16/shape/有限值及指标，结果一致。

| 用例 | grid/内容 | patch数 | 单路输出行数 |
| --- | --- | ---: | ---: |
| image_square | [1,16,16] | 256 | 64 |
| image_rect | [1,16,24] | 384 | 96 |
| image_limit | [1,32,32] | 1024 | 256 |
| 同布局不同像素、candy | 各[1,16,16] | 各256 | 各64 |
| cat | [1,16,24] | 384 | 96 |
| 长宽交换 | [1,24,16] | 384 | 96 |
| 两图及仅改变图2 | [1,16,16]+[1,16,24] | 各640 | 各160 |
| 两图顺序交换 | [1,16,24]+[1,16,16] | 640 | 160 |
| 两时间组及仅改变组2 | [2,16,16] | 各512 | 各128 |

单图像素来自已验证的只读Oracle；长宽交换由固定HF Processor生成，多图/时间组为底层数学隔离测试，未开放对应产品入口。输出在原grid、Processor块顺序及HF对应输出顺序上逐元素对齐；矩形明确断言主路和三路均为`[96,2048]`、list长度3。

固定形状下改变第二图/时间组内容，第一段patch、全部24层和四路特征保持bitwise一致，第二段主特征确实改变。HF/native两边的保存结果均独立复算隔离成立。受控扰动仅发生在独立测试进程，不改生产文件：

| 错误候选 | 有效失败断言 |
| --- | --- |
| DeepStack索引整体后移一层 | CPU merger输入与手算期望不符，max_abs=2.0 |
| 颠倒三路返回顺序 | 真实GPU `deepstack_0` max_abs=19.578125>0 |
| 全部block错误使用跨图全局Attention | 真实GPU改变图2后，图1主特征130637/131072元素变化，max_abs=6.8125，隔离断言失败 |

## 测试和显存

| 验证 | PASSED | FAILED/ERROR/SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | --- |
| 旧CPU基线（排除新增模块） | 292 | 0/0/0 | 27 | 9.77s |
| Task11c CPU | 6 | 0/0/0 | 1 | 1.73s |
| 完整CPU | 298 | 0/0/0 | 28 | 10.00s |
| 所属视觉CPU | 29 | 0/0/0 | 4 | 1.93s |
| 快速CPU | 31 | 0/0/0 | 295 | 1.68s |
| 禁止模型栈导入的轻量CI检查 | 31 | 0/0/0 | 295 | 1.65s |
| Task11c GPU GREEN | 1 | 0/0/0 | 0 | 58.32s |
| Task11c最终正式GPU验收 | 1 | 0/0/0 | 0 | 57.12s |
| M2b GPU回归：Task10/11a/11b及旧Qwen3两项 | 5 | 0/0/0 | 0 | 87.23s |

一项Task11c GPU节点内部实际运行12例，不记作12个pytest节点。M2b回归包括旧Qwen3短生成及既有固定输入回归，不重建其基准或增加全模型数值矩阵。Ruff按CI tests范围加vision模块检查/格式检查共24文件通过。首次沙箱CPU基线为276通过、16个Gloo Socket setup error，已按命令获宿主权限重跑；没有mock/skip或因环境错误调整断言。GPU均在真实RTX5060Ti执行，所有GPU进程串行，Dynamo/CUDA Graph关闭。

最终正式GPU证据目录为`task11c/gpu-wsw5376c/`，源码指纹与独立Review候选一致。旧Qwen3两项worker原始记录另存`legacy-smoke.json`、`legacy-current.json`，已再次核对固定记录匹配未改变的原始reference，不依赖临时目录长期保留。

首轮完整GPU证据`task11c/gpu-g72vakcs/`：HF peak allocated **876,240,896 bytes**、reserved **905,969,664 bytes**，最低采样free **15,028,584,448 bytes**；native allocated **868,589,056 bytes**、reserved **903,872,512 bytes**，最低free **15,030,681,600 bytes**。从设备迁移前重置后累计，覆盖完整视觉权重加载、12例激活及诊断开销；仅为单个完整视觉模块，不是语言模型/KV/引擎联合profile，也不是性能Benchmark。参数弱引用确认释放，不声称所有输出Tensor或CUDA上下文零占用。

54项此前源码/测试/Oracle顶层保护哈希不变，原vision组件前缀完全不变。完整视觉测试采用局部worker timeout=240s，避免修改全局120s helper；超时仍失败，没有用timeout放行测试，初次RED/GREEN均在限内。

## 独立Review与范围裁定

Reviewer `/root/task11c_review`以fresh context只读审查完整四文件diff、整个已组装vision模块、新测试、批准设计/计划及原始证据，规格符合性PASS、代码/测试质量PASS，**Critical0 / Important0 / Minor0**。无修复请求、遗留问题或延期Minor，不进行二次Review。

Reviewer独立运行CPU **6 PASSED、1 DESELECTED，1.77s**，并执行Ruff及diff检查；独立复算348对Tensor三指标全0，以及HF/native两边全部29项图像/时间组隔离；复核候选源码SHA、54项保护哈希、旧vision前缀、315-key实际加载代码、RED/扰动/JUnit/显存/参数释放证据。Reviewer没有新增GPU推理，证据复算不等于独立GPU运行。之后执行者在未改变的候选代码上完成正式`task-done`GPU门禁：1项通过57.12s，12例/348项比较及315视觉参数审计再次通过。Task11c和M2b本地技术门禁完成，保留未提交状态。

裁定：沿用干净功能分支，不搬移个人实验；同设备BF16/SDPA零误差仅是当前视觉组件门禁。完整模型的语言融合/Runner注册/全模型加载/KV/图像Decode/生命周期、产品多图/视频、联合显存预算/性能、跨内核/TP/Graph/compile、其他checkpoint或非法DeepStack配置、宽BF16阈值批准均未由本轮证明。成本是后续任务须各自验证后才能声称支持。240s局部timeout只提供完整模块测试余量，真实挂起最多等待该时限才失败；不改变全局helper或把超时当通过。远程CI未执行，Commit/Push须另获用户授权，Task12未启动。

## 复现与交付

仓库根目录、现有venv、本地权重。完整CPU需Gloo本地Socket权限，GPU需真实CUDA权限：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_vision.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision.py -m 'not gpu' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision.py::VisionTests -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py::VisionPositionGPUTests tests/test_qwen3_vl_vision_block.py::VisionBlockTests tests/test_qwen3_vl_vision_merger.py::VisionMergerTests tests/test_original_baseline.py::test_qwen3_short_generation_smoke tests/test_original_baseline.py::test_qwen3_fixed_input_regression -m gpu --require-gpu -q
.venv/bin/python -m ruff check tests nanovllm/__init__.py nanovllm/models/qwen3_vl_vision.py
.venv/bin/python -m ruff format --check tests nanovllm/__init__.py nanovllm/models/qwen3_vl_vision.py
git diff --check
```

正式四文件：vision模块、新vision测试、实施计划、本报告。拟Commit说明`feat: assemble Qwen3-VL vision encoder and DeepStack outputs`；未获批准不暂存/Commit/Push。证据目录`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11c/`保留原始RED/GREEN日志/JUnit、全部视觉key审计、HF/native逐层CPU Tensor、输入及源码摘要、显存与释放记录、独立复算结果、三项扰动失败、Review和最终diff，不将证据或权重加入Git。

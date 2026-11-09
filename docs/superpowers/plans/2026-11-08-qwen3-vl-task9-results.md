# Task9：Processor输入契约与请求级CPU状态

2026-11-08，Task9完成。分支`feat/multimodal`、基线及当前HEAD为`4e7cb1700e42722541b1dd6e274908df2e21e64a`。用户确认Task8c已提交并推送；本轮只执行已批准Task9，未暂存/Commit/Push，未开始Task10。

## 实现与接口

新增`nanovllm/utils/multimodal.py`：

- `PreparedMultimodalPrompt`保存四项CPU输入：`token_ids: list[int]`、`pixel_values[P,1536]`、`image_grid_thw[I,3]`、`mm_token_type_ids[L]`。`copy()`复制列表、detach并clone各Tensor，使请求持有独立CPU存储。
- `prepare_multimodal(messages, processor, *, max_visual_tokens=256)`只处理一个会话、一张静态PIL图片。固定HF版本的选项全部放在`processor_kwargs`中，启用模态类型、禁用padding，并显式传入图像像素上下限。返回CPU Prepared，进一步校验真实输出grid/行数/占位符/预算，不以原始图片尺寸推断最终视觉token数。
- `validate_multimodal(prompt, hf_config, *, max_visual_tokens)`验证非空且合法的整数IDs、CPU设备/shape/dtype、浮点pixels有限值、静态单图grid、patch数量、视觉token上限、模态类型与image ID逐位置一致、连续且完整的vision_start/image_pad/vision_end语法。视频、多图、奇数或非正H/W、T!=1、像素行数不符等明确报错。空image字段可表示纯文本；此时不允许任何孤立image/video控制token。
- `validate_text_tokens`用于VL原始字符串编码后或list[int]入口，阻止孤立占位符绕过Prepared校验。

修改`Sequence`：增加keyword-only `multimodal=None`；Prepared与传入IDs必须一致，复制CPU数据后调用Task3位置函数，保存`prompt_positions[3,L]`、`rope_delta`和`visual_mask[L]`。纯文本旧调用保留默认None/0，无新增GPU张量。除构造器外原有方法AST不变，含六项TP序列化协议；旧方法的格式变化不改变行为。

修改`LLMEngine`：保留Config供输入边界使用；`add_request/generate`类型支持Prepared及原str/list[int]。先验证，再构造Sequence并交给既有Scheduler做长度/容量检查。**合法图片Prepared在入队/KV分配前抛出NotImplementedError，明确说明Task13尚未接通图像执行。** 不会剥掉图片后走纯文本。纯文本Prepared可入队，多个合法文本请求仍可排队；没有改写原generate批次原子性、调度准入或运行恢复语义。

## 范围裁定与使用限制

1. 沿用现有Native feature checkout和既有venv，保护个人未跟踪文件；没有新worktree或依赖安装。成本：在当前分支保留可审查的未提交diff。
2. Sequence固定接口没有config参数，按目标Qwen3-VL的merge2计算位置；validator明确拒绝其他merge布局，不默默产生错误位置。成本：不支持其他merge配置。
3. 预处理最大像素面积为`max_visual_tokens*(patch_size*merge_size)^2`，最小面积取固定Processor默认最小值与最大面积的较小者。helper允许HF按显式预算缩放，再检查实际grid；已构造Prepared自身从不自动缩放。成本：较大原图可能在显式预处理时缩小，不能将其等同于原尺寸输入。
4. **仅接受`{"type":"image", "image": PIL.Image.Image}`，且n_frames=1。** 路径、URL、base64/data URI以及`url/path/base64`别名全部在HF调用前明确拒绝。调用方先用PIL打开来源，并将尚保留帧信息的对象传入；helper负责检查后才委托HF做RGB转换和patch处理。成本：本阶段不提供来源下载/解码便利接口，不新增网络或文件I/O实现。
5. 图像模型执行、特征生命周期和真实图像KV留13/14；视觉编码/精度留10以后；已验收的Task7容量/缓存及Task8恢复不重复开发。TP>1、Graph、多图、视频执行和其他merge布局不扩展。成本：本次CPU输入/状态通过不代表单图生成可用。整批generate错误不承诺原子回滚，先前合法排队请求保留；错误请求自身不进入Scheduler/KV。

## TDD与独立Review

先建立可导入Prepared及显式NotImplemented边界，collection正常收集62项。初始RED **62 FAILED、0 ERROR，2.26s**：普通校验助手尚为NotImplemented过渡；Engine孤立IDs到达DID NOT RAISE；Sequence新keyword缺失属于接口未完成，不把TypeError冒称数值/所有权断言RED。最小实现后同一模块 **62 PASSED，1.99s**；补充CPU设备、预算及只读Oracle覆盖后 **70 PASSED，2.37s**。

为证明高风险断言有效，独立进程临时扰动本项目自己的方法，不修改生产文件：将Prepared.copy改为返回原对象，所有权测试 **1 FAILED，1.30s**，达到原列表被调用者改变的断言；将位置函数返回值位置/delta加1，真实HF位置对照 **1 FAILED，1.34s**。随后正常83项完整测试通过。没有Mock Gloo、模型、CUDA或HF Processor来取得通过。

独立Reviewer `/root/task9_review` 使用fresh-context gpt-6-astra high只读审查本任务完整diff及新文件，独立CPU **70 PASSED，2.36s**，检查hostCPU256通过和扰动失败日志，未独立执行完整CPU/GPU。原结论 **With fixes**，Critical0 / Important1 / Minor0。

唯一Important：动画的字符串或HF别名来源绕过原n_frames检查，真实固定HF Processor会把两帧APNG静默变成首帧。新增13项来源矩阵，原实现 **11 FAILED、2 PASSED、70 DESELECTED，2.08s**；然后按上述第4条最小限制来源形式，修复后Task9全 **83 PASSED，2.43s**。原canonical静态PIL及已有PIL动画拒绝两项控制用例保持通过。随后完整CPU和受影响文本GPU全部通过，关闭该问题。只进行一次Important修复验证，未派发二次Review；不冒称原Reviewer审过最终修复。

Reviewer未裁定的行为均列入上文范围裁定。无新增延期Minor；Task6原非阻塞测试建议保留原状态。

## 数值与状态证据

固定模型snapshot`89644892e4d85e24eaac8bacfd4f463576704203`，Transformers提交`2711ad22c82105e240a2877b92d082fe2857936f`，复用原本地editable环境。

| 输入图片 | 实际grid | pixels shape | image token数量 |
| --- | --- | --- | ---: |
| 256×256 | [1,16,16] | [256,1536] | 64 |
| 384×256 | [1,16,24] | [384,1536] | 96 |
| 512×512 | [1,32,32] | [1024,1536] | 256 |
| 1024×1024，显式max_visual_tokens=64 | [1,16,16] | [256,1536] | 64 |

前三种输入与直接固定HF Processor的IDs、pixels、grid、模态类型精确一致；还独立读取持久化只读Oracle的image_square/image_rect/image_limit输入并核对全部Tensor精确相等。不是仅比较shape或可读文本，不使用宽BF16阈值。对应Sequence三轴位置/delta与HF get_rope_index一致；独立小网格手工期望delta=-3。修改调用者列表、pixels、grid、模态类型后，请求自身数据、位置和mask不变，不同请求之间没有别名。

非法输入覆盖空输入、孤立image/video及vision_start/end、无效ID、缺/多grid、T/H/W、像素尺寸/NaN/Inf/类型、模态类型/长度/布局、预算、padding、批处理输出以及来源形式。真实Scheduler/BlockManager状态快照确认被拒绝请求没有入队或分配KV。非CPU输入用真实meta Tensor验证设备拒绝，不将其称为真实GPU输入验证。

## 最终测试结果

| 验证 | PASSED | FAILED | ERROR | SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Task9正式验收（task-done） | 83 | 0 | 0 | 0 | 0 | 2.92s |
| 完整CPU（含M1回归） | 269 | 0 | 0 | 0 | 24 | 9.56s |
| 快速CPU | 31 | 0 | 0 | 0 | 262 | 1.68s |
| 既有文本GPU组合 | 11 | 0 | 0 | 0 | 35 | 52.83s |

这些集合有重叠，不相加冒称独立测试数量。GPU11项沿用原8b/8c测试和旧Qwen3两项回归；真实目标checkpoint纯文本Prefill、32次Decode输入、greedy生成、分页KV及生命周期均通过。此类完整GPU worker按原测试将数据保存在task8b/task8c证据目录，本轮最终native Decode为`task8c/decode-b0xftip5/`，其数值/资源状态与本任务日志一并核对；没有为了Task9更改这些旧测试。

上述native worker退出前记录累计peak allocated=3,940,984,832 bytes、peak reserved=4,013,948,928 bytes、设备空闲11,723,407,360 bytes；退出后模型参数及KV弱引用全部失效、组已销毁，allocated=8,523,776 bytes、设备空闲15,701,704,704 bytes。峰值从既有warmup重置后累计，不是Task9 CPU helper的显存消耗或独立Decode性能基准；没有做性能比较。

沙箱完整CPU原为 **240 PASSED、16 ERROR、24 DESELECTED，10.00s**，16项均Gloo创建Socket的Operation not permitted；授权宿主后修复前256通过、最终269通过，原始失败及修复前日志均保留。没有skip、Mock Gloo或修改断言掩盖环境问题。CUDA独立探测：沙箱False/0，宿主True/1、RTX5060Ti、torch2.9.1+cu130/CUDA13.0；仅确认可见后运行原正式GPU测试，未用sudo、全局Full access或修改CUDA环境。

四个Python文件Ruff check/format以及git diff --check通过。其余nano/tests已跟踪源码、旧Qwen3基准及Oracle的tolerances.json、SHA256SUMS、cases/checkpoint保护哈希不变，未安装或替换依赖。正式Oracle持续位于`/home/thedai/nano-vllm-artifacts/qwen3-vl/oracle-reviewed/`，本轮只读。

## 复现命令与交付

仓库根目录，既有venv和本地checkpoint；完整CPU需本地Socket权限，GPU另需CUDA可见：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_input.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_input.py -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest tests/test_qwen3_vl_runner.py tests/test_original_baseline.py -m gpu --require-gpu -q
.venv/bin/python -m ruff check nanovllm/utils/multimodal.py nanovllm/engine/sequence.py nanovllm/engine/llm_engine.py tests/test_qwen3_vl_input.py
.venv/bin/python -m ruff format --check nanovllm/utils/multimodal.py nanovllm/engine/sequence.py nanovllm/engine/llm_engine.py tests/test_qwen3_vl_input.py
git diff --check
```

预处理用法（仅构造输入，此阶段传入Engine仍明确拒绝图像执行）：

```python
from PIL import Image
from transformers import AutoProcessor
from nanovllm.utils.multimodal import prepare_multimodal

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
with Image.open(image_path) as image:
    prepared = prepare_multimodal([{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image."},
    ]}], processor)
```

正式交付6文件：新增multimodal.py、test_qwen3_vl_input.py、本报告，修改sequence.py、llm_engine.py、实施计划。拟提交说明：`feat: validate Qwen3-VL processor inputs and own request state`；本轮只留可审查diff，不暂存/Commit/Push。

持久化忽略证据目录`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task9/`保存collection/red/green/input-expanded/input-final、review-red、cpu-sandbox/cpu-before-review/cpu-host、quick、gpu-final日志及JUnit，两项mutation日志与诊断脚本、CUDA探测、baseline/final source哈希、原源码、reviewed.diff、task9.diff、review.md和summary.json。正式Task9验收日志另在上级task-9-tests.log。不得把这些日志或大型Oracle数据作为正式交付文件加入Git。

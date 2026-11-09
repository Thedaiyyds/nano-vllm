# Task10：视觉patch embedding、绝对位置插值及二维RoPE

2026-11-09，Task10技术验收通过。沿用`feat/multimodal`当前工作区，HEAD为`4e7cb1700e42722541b1dd6e274908df2e21e64a`。Task9开发及再次Review已验收，其生产文件、测试和结果报告均按任务开始时SHA保护并保持不变。用户明确要求继续开发而非整理Git状态，因此保留现有未提交Task9文件，不做fetch/reset/暂存/Commit/Push。

本轮只新增`nanovllm/models/qwen3_vl_vision.py`和`tests/test_qwen3_vl_vision_positions.py`，更新实施计划并新增本报告。未开始Task11，未修改原Runner、Loader、Attention、模型或Task9输入边界。

## 实现说明

- `Qwen3VLVisionPatchEmbed(vision_config)`使用含bias的Conv3d，kernel/stride从配置推导，目标为`(2,16,16)`。输入Processor已展平的`[P,1536]`，恢复为`[P,3,2,16,16]`，按权重dtype卷积，返回`[P,1024]`。不重排patch、不自行复制时间帧；模块属性`proj.weight/bias`沿用HF命名。
- `vision_position_ids(grid_thw, merge_size)`返回int64`[P,2]`，顺序为`(block_row,block_col,in_row,in_col)`，匹配Processor的2×2块顺序；按T重复空间坐标，并按输入grid顺序拼接。
- `interpolate_vision_positions(weight, grid_thw, merge_size)`消费目标`[2304,1024]`绝对位置表。使用align_corners=True的`index*(side-1)/max(size-1,1)`坐标、边界clamp及四邻点FP32权重，按HF相同的四项乘法/求和顺序返回插值位置。BF16表与FP32权重相乘得到FP32位置；调用方先转回patch dtype，再相加。没有用不同精度或运算顺序的resize替代。
- `Qwen3VLVisionRotaryEmbedding(vision_config).forward(position_ids)`返回FP32 cos/sin`[P,64]`，目标theta=10000、head_dim=64，频率排列`[H,W,H,W]`，独立于语言三轴MRoPE。频率在CPU以FP32计算，保存为不可变数值，forward在目标设备构造小Tensor，避免父模块`.to(BF16)`量化频率。
- `apply_rotary_pos_emb_vision(q,k,cos,sin)`直接实现计划要求的视觉旋转：Q/K及cos/sin在FP32计算，结果分别恢复原Q/K dtype。该小函数不包含Attention、KV或ViT逻辑，供后续Task11消费。

这里的底层多grid/T数学接口不代表开放产品多图/视频入口；Task9的单图/静态图片校验及图像执行NotImplemented门槛保留。

## TDD证据与流程限制

先建立可导入接口，正常collection收集11项（后补一个时间对比用例，最终12项）。初始CPU **10 FAILED、1 DESELECTED，1.77s**，是NotImplemented过渡失败；宿主CUDA可见后，GPU **1 FAILED，5.75s**，到达真实checkpoint严格加载时发现尚无`proj.weight/bias`。**这两次初始RED尚未到达数值断言，不称为最初的数值/状态RED。**

最小实现后CPU **10 PASSED，1.99s**，真实GPU **1 PASSED，23.68s**。高风险测试进一步在独立进程中受控扰动本项目函数，不修改生产文件：

| 扰动候选 | 有效失败断言 | 结果 |
| --- | --- | --- |
| 把block-major坐标改成raster顺序 | 4×4手工坐标列表不一致 | 1 FAILED，1.20s |
| 插值输出整体加1 | 与HF位置输出不一致 | 1 FAILED，0.40s |
| 交换H/W轴生成频率 | 独立解析频率/cos断言不一致 | 1 FAILED，1.21s |

之后恢复正常函数并完成所属CPU及正式GPUGREEN。另补充不同时间槽的对比值与bias用例，确认模型保留输入两个时间槽的内容，而非复制第一个；补充多grid插值对照。

独立Reviewer指出严格时序限制：后续数值断言失败证明测试敏感性，但不能追认实现前已有数值RED。计划全局允许NotImplemented过渡和受控扰动；本轮功能与数值验收齐备，**初始数值RED的时间顺序证据不足仍如实保留**，不重写实现来制造历史证据。

## 真实权重与数值验证

固定HF提交`2711ad22c82105e240a2877b92d082fe2857936f`，checkpoint snapshot`89644892e4d85e24eaac8bacfd4f463576704203`。GPU为RTX5060Ti，torch2.9.1+cu130 / CUDA13.0，沿用既有环境；`TORCHDYNAMO_DISABLE=1`。只用`safetensors.safe_open`读取以下三项实际权重，patch两项对native及独立HF patch模块均strict加载：

```text
model.visual.patch_embed.proj.weight -> native_patch.proj.weight
model.visual.patch_embed.proj.bias   -> native_patch.proj.bias
model.visual.pos_embed.weight       -> interpolate_vision_positions(weight, ...)
```

不构造完整HF/nano模型，不加载ViT/merger权重；这不是全部视觉权重加载审计。HF参考使用固定源码的PatchEmbed、轴向RoPE、插值helper及旋转函数，生产实现不委托HF计算。两个小patch模块可同时放入单卡进行对应组件比较，不涉及两份完整2B模型共驻。

| 用例 | grid | patch数 | 浮点组件数 | max_abs / mean_abs / relative L2 |
| --- | --- | ---: | ---: | --- |
| image_square | [1,16,16] | 256 | 7 | 全部0 |
| image_rect | [1,16,24] | 384 | 7 | 全部0 |
| image_limit | [1,32,32] | 1024 | 7 | 全部0 |
| image_same_layout_other_pixels | [1,16,16] | 256 | 7 | 全部0 |
| image_candy | [1,16,16] | 256 | 7 | 全部0 |
| image_cat | [1,16,24] | 384 | 7 | 全部0 |
| image_rect_transposed | [1,24,16] | 384 | 7 | 全部0 |

七项为patch、FP32 absolute、BF16 patch+absolute、FP32 cos/sin、BF16 rotated Q/K，检查shape、dtype、有限值及三项零误差；positions另外精确比较。前六例输入取自持久化只读Oracle，长宽交换用例由固定HF Processor独立生成。原尺寸48×48、尺寸1轴、边界、FP32/BF16位置表、混合grid及T重复另由CPU与HF对照。4×4可辨认patch顺序使用真实Processor切块，并以手工稀疏卷积权重和字面值验证顺序，不以随机同构模型互证。

**未使用或批准宽BF16阈值**。正式Oracle及tolerances.json、SHA256SUMS、cases/checkpoint元数据哈希保持不变，新输出仅在本任务持久化忽略目录中。

## 最终测试结果

| 验证 | PASSED | FAILED | ERROR | SKIPPED | DESELECTED | 用时 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Task10 CPU模块 | 11 | 0 | 0 | 0 | 1 | 1.88s |
| 完整CPU（含此前回归） | 280 | 0 | 0 | 0 | 25 | 10.00s |
| 快速CPU | 31 | 0 | 0 | 0 | 274 | 1.67s |
| 最终正式Task10 GPU | 1 | 0 | 0 | 0 | 0 | 23.63s |

GPU一项内部实际执行上述七例，不能把它报告成七个pytest节点。快速CPU是完整CPU子集。此次没有修改任何既有生产模块，未重复运行Task9的旧文本GPU组合；此前已验收文件通过SHA保护，当前完整CPU再次通过。

Gloo的沙箱Socket限制已有原始证据，相关fixture及依赖哈希一致，本轮完整CPU直接采用针对命令授权的宿主执行；未skip/Mock Gloo、未改断言或为环境错误修改测试。CUDA单独探测：沙箱False/0、宿主True/1，确认可见后才运行GPU；没有sudo、全局Full access或CUDA环境修改。

最终GPU证据`task10/gpu-h4vtwnqv/`。诊断进程累计peak allocated **83,281,408 bytes**，peak reserved **98,566,144 bytes**，采样最小设备空闲 **15,836,053,504 bytes**。峰值包含native/HF两个小模块、保存对照张量和诊断开销，从循环前重置后累计，**不是单侧视觉模型峰值或性能基准**。退出检查确认两个patch模块的参数弱引用失效，不声称CUDA上下文零占用。

两个新Python文件Ruff check/format与git diff --check通过；所有已验收Task9及此前nano/tests源码、旧Qwen3基准、Oracle保护哈希不变，无依赖修改。

## 独立Review及裁定

Reviewer `/root/task10_review`使用fresh-context gpt-6-astra high只读审查，结论Ready Yes（限定Task10），Critical0 / Important0 / Minor0。独立CPU **11 PASSED、1 DESELECTED，1.85s**；检查执行者GPU七例零误差、完整CPU280通过和三项扰动断言日志，未独立重跑GPU或重新计算全部已保存Tensor。初始RED流程限制已在上文单列。没有Important代码修复或二次Review，无新增延期Minor。

执行裁定与成本：沿用当前工作区并以快照保护尚未体现在本地HEAD的Task9，不因Git状态重做已验收工作；测试只加载三项实际组件权重，不声称全部visual keys已消费；旋转小函数满足Task10显式FP32要求，ViT/Attention留Task11；immutable频率每次构造小Tensor，编译/性能优化后续再测。Reviewer未裁定的图像端到端生成、融合/DeepStack/KV、多图/视频产品入口、Graph、TP、其他设备后端和性能均保持后续范围，不因底层数学通过而开放。

## 复现命令与交付

仓库根目录、既有venv与本地checkpoint，无新增依赖。CPU/GPU需相应本地权限：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py -m 'not gpu' -q
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
.venv/bin/python -m pytest tests/test_qwen3_vl_vision_positions.py::VisionPositionGPUTests -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/models/qwen3_vl_vision.py tests/test_qwen3_vl_vision_positions.py
.venv/bin/python -m ruff format --check nanovllm/models/qwen3_vl_vision.py tests/test_qwen3_vl_vision_positions.py
git diff --check
```

本轮四项交付：新vision模块、新测试、本报告、实施计划中的Task10增量。拟提交说明`feat: add Qwen3-VL vision patch and spatial positions`；当前不暂存/Commit/Push，不开始Task11，Task9个人及已验收文件保持原样。

证据目录`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task10/`含collection、cpu-red/green/final/full、gpu-red/green/acceptance、quick日志/JUnit；三项mutation日志与仅测试诊断脚本；GPU各轮独立目录中保存真实HF/native小组件张量、identity、comparison、显存和源key记录；另有baseline-sha256、reviewed.diff、最终task10.diff、review.md、summary.json及最终SHA清单。正式task-done日志在上级task-10-tests.log，范围`4e7cb17..4e7cb17`不代表产生了新Commit。证据及权重不加入Git。

# Task 6：确定性 greedy 与 VL 双 EOS

2026-11-02，从前次中断继续完成Native Task6。基线及当前HEAD均为`89c1e40e9aa52867ad107954496b3d342f24170d`，分支`feat/multimodal`。Task2–5已提交，不重复实现；保留用户实验文件。未安装依赖、未改CUDA/HF/Oracle、未暂存/Commit/Push，未开始Task7。

## 实现说明

- `SamplingParams`允许`temperature=0`，拒绝负数和NaN/Inf温度；`max_tokens`必须为正Python整数（拒绝bool和非整数）。校验仅依赖标准库，保留轻量包导入。
- `Sampler.forward(logits, temperatures)`签名不变。零温度直接argmax，同分取首项；纯greedy不执行除法、不生成随机数。混合批次只对正温度行执行原softmax/指数随机数竞争采样，再按原行序写回结果。所有正温度直接走保留的原算法，数学表达式AST与基线一致。Sampler也拒绝非法温度Tensor。
- `Scheduler`在VL分支读取Config的完整`eos_token_ids`，旧Qwen3仍使用tokenizer标量EOS对应的单元素集合。两个VL EOS均可结束；`ignore_eos=True`忽略整个集合；长度上限始终有效，`max_tokens=1`在首个生成token后停止。终止后沿用原释放路径。
- `LLMEngine`只在旧文本模型分支将tokenizer EOS写入旧`config.eos`；VL保持GenerationConfig解析的集合，避免退回单EOS语义。

只改计划规定的四个生产文件及新增测试。修改文件按Ruff格式化，LLMEngine除EOS初始化外、Scheduler除初始化/停止判断外的函数AST与基线一致；未修改调度准入、prefix cache或请求生命周期。不新增top-k/top-p，不接完整VL Runner。

## TDD与测试边界

先正常收集33项，再执行CPU RED：**21 failed、9 passed、3 deselected**。实际失败包括合法0温度被原assert拒绝、greedy输出错误、非法参数被接受或异常类型不符、两种EOS未停止、VL的scalar EOS被覆盖。旧engine wiring测试缺新Scheduler集合属性的AttributeError属于新增接口过渡，不宣称所有失败均为数值断言；没有ImportError或基础设施setup失败冒充RED。9项第一次即通过的兼容行为不算新增实现先红后绿。

最小实现后CPU **30 passed、3 deselected**。中断后先核对日志和当前代码，确认这次运行已完成，再继续GPU、全套回归及Review，没有重做已完成Task。

主要验证内容：

| 检查 | 证据及边界 |
| --- | --- |
| 真greedy与tie | CPU/BF16 CUDA检查`[1,3,2]→1`、同分首项；真实151936词表中包含两个EOS及非零位置tie |
| 零温度不除法/不耗RNG | TorchDispatchMode仅观察真实aten.div_分母，仍执行原操作；纯greedy无除法，混合时实际分母均为正；检查CPU/CUDA RNG状态 |
| 正温度兼容 | 固定seed与独立保留的旧算法比较token IDs及RNG状态；全正温度均匀logits覆盖float32/BF16；混合行另含非均匀logits |
| 双EOS/ignore/length | 真实Scheduler、Sequence及BlockManager，分别结束151645/151643、连续忽略两者、max_tokens=1，并检查used blocks及ref_count释放 |
| engine EOS wiring | 使用真实Config、本地Tokenizer和Scheduler；仅替换GPU ModelRunner边界以隔离初始化，不是VL模型smoke、不mock Gloo |
| 旧模型兼容 | 既有Qwen3-0.6B真实GPU短生成及固定8-token回归；未重采或更新原期望 |

engine wiring单元测试不证明完整VL引擎已可初始化或生成；实际模型路由仍属于Task8。Task6释放检查只覆盖当前调度器的正常终止，不声称已完成Task7缓存隔离或Task14异常生命周期。

## 真实结果

| 命令范围 | PASSED | FAILED | ERROR | SKIPPED | DESELECTED |
| --- | ---: | ---: | ---: | ---: | ---: |
| 开发前完整CPU基线 | 113 | 0 | 0 | 0 | 12 |
| Task6 CPU | 30 | 0 | 0 | 0 | 3 |
| 最终完整CPU | 143 | 0 | 0 | 0 | 15 |
| 最终快速CPU | 31 | 0 | 0 | 0 | 127 |
| Task6 GPU+旧模型GPU组合 | 5 | 0 | 0 | 0 | 39 |

Task6 CPU 1.93s；完整CPU 9.25s；快速CPU 1.65s；GPU组合9.36s。GPU组合为Task6的3项采样测试加既有旧模型的2项兼容回归，不把后者算作新增VL模型验收。独立Reviewer另行只读复跑Task6 CPU：30 passed、3 deselected、1.91s。

GPU运行前单独探测：CUDA可见True、设备数1、RTX5060Ti、torch2.9.1+cu130、CUDA13.0。测试显式`--require-gpu`，没有skip/xfail代替运行。CPU全套使用已确认需要本地Socket权限的宿主Gloo环境；新Task6 CPU单元测试自身不要求Gloo。未把Gloo Socket限制推断为GPU权限问题，未使用sudo或全局Full access，未修改测试掩盖环境限制。

Ruff check及format检查五个改动Python文件通过，`git diff --check`通过。快速CPU新增11项仅用标准库的SamplingParams校验，其余模型栈相关测试均延迟导入并正确标记；不增加CI依赖。远端CI本轮未运行。

本任务检查的是离散token、EOS状态及RNG行为，不使用HF浮点宽阈值；持久Oracle及`tolerances.json`均未改动。原正温度算法保留，无新增模型数值对齐结论。

## 独立Review及延期Minor

独立Reviewer审查四个生产diff、新测试、Plan/Spec，读取GPU结果并独立复跑CPU。结论无Critical/Important，Task6可通过；没有要求第二次Review，也没有把读日志说成独立重跑GPU。

**延期Minor（非阻塞）：** 全正温度parity用例使用全零logits，能检验原随机算法/RNG消费，但温度缩放不改变均匀分布，不能单独发现将来移除或错误广播temperature的回归。建议保留该用例，并在后续增强中增加非均匀固定logits的全正温度对照。本轮实现经源码检查正确，已有混合非均匀行及旧模型固定生成回归；按Native流程记录非阻塞建议，不将其伪装成已修复。

Reviewer未判定Task7准入/容量/prefix cache隔离、完整VL生成、启用torch.compile后的行为和性能、用户实验脚本。本轮接受这些边界；验证使用`TORCHDYNAMO_DISABLE=1`，不声称已验证编译性能或M1整体完成。

实施裁定：沿用无目标冲突的Native feature工作区；旧正温度计算提取为私有helper以保留算术及随机过程，混合温度只采样正行；旧模型EOS分支保持单EOS。正整数预算和非法温度在入口明确拒绝，不引入通用采样配置框架。代价是Tensor检查及分支可能有同步/性能开销，性能与编译优化留后续独立验收。

## 复现命令与交付

仓库根目录、现有venv执行；GPU命令先在宿主确认CUDA，完整CPU/旧模型回归需要本地Socket权限。无需下载或安装：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
export QWEN3_VL_MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
export QWEN3_MODEL="$HOME/huggingface/Qwen3-0.6B"

.venv/bin/python -m pytest tests/test_qwen3_vl_sampling.py --collect-only -q
.venv/bin/python -m pytest tests/test_qwen3_vl_sampling.py -m 'not gpu' -q
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available()'
.venv/bin/python -m pytest tests/test_qwen3_vl_sampling.py::SamplingGPUTests -m gpu --require-gpu -q
# 本轮实际GPU组合命令，包含共享采样器影响的旧模型回归
.venv/bin/python -m pytest tests/test_qwen3_vl_sampling.py tests/test_original_baseline.py -m gpu --require-gpu -q
.venv/bin/python -m pytest tests -m 'not gpu' -q
.venv/bin/python -m pytest tests -m 'not gpu and not hf_reference and not checkpoint' -q
.venv/bin/python -m ruff check nanovllm/sampling_params.py nanovllm/layers/sampler.py nanovllm/engine/scheduler.py nanovllm/engine/llm_engine.py tests/test_qwen3_vl_sampling.py
.venv/bin/python -m ruff format --check nanovllm/sampling_params.py nanovllm/layers/sampler.py nanovllm/engine/scheduler.py nanovllm/engine/llm_engine.py tests/test_qwen3_vl_sampling.py
git diff --check
```

证据在持久化自忽略目录`.superpowers/sdd/2026-10-29-qwen3-vl-implementation/task6/`：`baseline.log/xml`、`collection.log`、`red.log/xml`、`green.log`、`cpu-unit.xml`、`cuda-probe.log`、`gpu.log/xml`、`cpu.log/xml`、`quick.log/xml`、`summary.json`、`review.md`、含新文件的`task6.diff`。失败RED日志保留，不覆盖为成功。

最终Python SHA256：

```text
2900d8f65d9c9d0da5ba3c0336a743f1b167a4fda5897de05e19f0fdee534c95  nanovllm/sampling_params.py
93423c6ac97f63c5d3b5bea488a2980c633e89c7c021ff1a36f1038264f34c3f  nanovllm/layers/sampler.py
53c31ecb9734c199b0a750428b9892439942664cc9242bc1906087e89967308c  nanovllm/engine/scheduler.py
b6e42407a2a87ea26a7d851eb9680934b67d3ad2198ffd1612d3597a2c2d5cdc  nanovllm/engine/llm_engine.py
f86f021979cc7c0051c6a02b52cdab1c925480b315a4a16db4151a5f3bd7aa0b  tests/test_qwen3_vl_sampling.py
```

正式交付为上述五个Python文件、本报告和实施计划Task6状态更新。拟提交信息`feat: add greedy sampling and Qwen3-VL EOS handling`，仅供审阅；本轮未暂存、Commit或Push，止于Task6。

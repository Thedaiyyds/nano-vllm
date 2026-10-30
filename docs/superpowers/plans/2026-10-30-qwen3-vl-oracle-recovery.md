# Qwen3-VL Task 1c 正式参考数据恢复记录

2026-10-30先只读检查原目录及用户目录、`/tmp`、`/var/tmp`、`/mnt`、`/media`内可能的数据和归档。原目录不存在，所查位置未找到正式张量或阈值副本。仓库中的生成器、输入清单、checkpoint指纹、旧摘要、历史日志和FP32诊断脚本仍在，因此按原Task1c流程重新采集，没有重新设计验证方法。

当前正式目录为`/home/thedai/nano-vllm-artifacts/qwen3-vl/oracle-reviewed/`，约1.5 GB（1.40 GiB；`du -sh`显示1.5G）。它位于用户磁盘分区；本机`/tmp`是tmpfs。目录封存为只读，并附完整`SHA256SUMS`；不纳入Git。全目录最终精确容量、文件数及封存核验结果见本次交付输出。

## 数据与结果

| 内容 | 本次真实结果 |
| --- | --- |
| HF正式参考 | 原10个用例 × SDPA/eager/SDPA重复，30个独立GPU进程全部成功 |
| 完整性审核 | 30组身份、文件SHA、必需组件、shape/dtype、finite、CPU输入指纹、teacher指纹及cache前缀检查通过 |
| KV核对 | 1680对首步KV逐位相同；后续步误差如实保留 |
| SDPA重复 | 所有张量逐位相同 |
| 历史摘要核对 | 所有10用例的已存数值、3164个浮点组件计数、140个零差异组件、23处argmax分歧记录完全复现 |
| 阈值核对 | 旧摘要保留的10组Decode logits三项阈值，共30个数值，完全相同 |
| 校准 | 使用未修改的`calibrate()`完成，仍为`2 × 最大HF实测差异 + one outward ULP`，无参数或规则调整 |
| CPU | Task1c：28 passed、4 deselected；完整CPU：54 passed、6 deselected |
| GPU Kernel | 4 passed、28 deselected；255/256/257三组指标与旧记录完全一致 |
| FP32诊断 | 原text_257/eager脚本重跑，各项数值完全复现，argmax 32/32一致 |

模型仍为snapshot `89644892e4d85e24eaac8bacfd4f463576704203`，Transformers仍为`2711ad22c82105e240a2877b92d082fe2857936f`；本地checkpoint全部固定文件通过原SHA256门禁，相关HF源码通过原AST门禁。生成器、checkpoint清单及FP32诊断源码的SHA与历史摘要相同。没有下载模型、安装/升级依赖、改CUDA环境或模型实现。

## 阈值证据的边界

原完整`tolerances.json`随临时目录丢失，仅其原SHA256和部分数值摘要仍在。因此不能证明全部3164组旧阈值与重采值逐项相等，也不能声称恢复了原30份张量/manifest的原始字节。已保留原摘要，未用新结果覆盖历史记录；本次校准提案独立保存，所有可核对的历史数值均精确一致，未放宽公式或已存阈值。

原阈值文件SHA256：`e3ca1659ec8924e8d480cf979fe5c238e961ac067bc15fa5362f8233519164f9`。

新阈值文件SHA256：`0a0029401ec73945930072300237f8dd45b7c14730c038619655a1de723600d9`。新文件引用本次manifest的哈希，manifest包含当次空闲显存等运行信息，不能与原文件作字节等同声明。

阈值状态继续为`PROPOSED_FROZEN_PENDING_USER_ACCEPTANCE`；重采不等于批准阈值，也不把宽BF16包络当作正确性的充分条件。后续候选实现失败时不得重采或调宽阈值来消除失败。

## 目录与后续使用

- `hf-sdpa/`、`hf-eager/`、`hf-repeat/`：每个用例各有`tensors.pt`和`manifest.json`。
- `tolerances.json`：本次冻结的校准提案及来源哈希。
- `oracle-source.py`、`cases.json`、`checkpoint.json`：不改动的来源快照。
- `kernel-evidence/`、`fp32-final/`：真实GPU Kernel JSON和原FP32脚本、张量及诊断结果。
- `historical-task1c-logs/`：原执行日志与FP32脚本。
- `recovery/`：本次30组采集日志、审核脚本/JSON、校准日志、测试日志、历史摘要、受控图片副本及环境前后核对。
- `SHA256SUMS`：封存时的全目录文件校验清单（不包含清单自身）。

后续Task使用`QWEN3_VL_ORACLE`指向这份只读参考；生成的候选结果、临时实验、Kernel复测及Benchmark应写入父目录下新建的独立目录。该变量只是路径约定，不是已有测试自动加载Oracle的开关；调用方明确传入路径或通过现有`validate_artifact()`加载对应case，比较时使用该case的`limits`及已批准的数值门槛。

```bash
cd /home/thedai/nano-vllm
export QWEN3_VL_ARTIFACTS="$HOME/nano-vllm-artifacts/qwen3-vl"
export QWEN3_VL_ORACLE="$QWEN3_VL_ARTIFACTS/oracle-reviewed"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1

# 只读验证封存的每个文件；全部成功时无输出、退出码0
(cd "$QWEN3_VL_ORACLE" && sha256sum --check --quiet SHA256SUMS)

# 只读回放既有完整性检查，并与保留的旧摘要比较；不重采、不覆盖审核JSON
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -B "$QWEN3_VL_ORACLE/recovery/audit.py" "$QWEN3_VL_ORACLE"
```

第二条命令预期逐个case输出PASS，最后报告30组通过、1680对首步KV相同、重复采集逐位相同、存留历史数值及30个Decode阈值一致。原生成器和校准CLI仍拒绝覆盖既有正式数据。当前工作仅恢复Task1c资料，不开始Task2，不暂存、Commit或Push。

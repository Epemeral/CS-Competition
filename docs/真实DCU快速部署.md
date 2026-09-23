# 真实 DCU 快速部署

真实比赛环境是单卡约 64 GiB 显存，Qwen2.5-7B FP16 权重约 14 GiB。因此主要目标是测出稳定的吞吐、延迟和显存余量，而不是只证明模型能加载。

## 1. 在登录节点准备代码

不要在登录节点运行 GPU 基准。把仓库和模型路径准备好，之后通过 Notebook 或 Slurm 计算节点执行：

```bash
cd ~/CS-Competition
git fetch origin feat/t4-triton-kernels
git checkout feat/t4-triton-kernels
git pull --ff-only origin feat/t4-triton-kernels
```

如果计算节点不能访问 GitHub，直接把本地仓库压缩包上传并解压；脚本不依赖在线拉取代码。

## 2. 先跑一条命令

组委会镜像已经提供 PyTorch/DTK 时，不要重新安装 torch、ROCm 或 Triton。直接运行：

```bash
cd ~/CS-Competition
chmod +x scripts/dcu_quickstart.sh
MODEL=/path/to/Qwen2.5-7B-Instruct \
MODEL_SOURCE=auto \
scripts/dcu_quickstart.sh
```

模型目录是本地目录时不会下载。模型目录不存在且镜像有 ModelScope 时，`MODEL_SOURCE=auto` 会尝试 ModelScope；也可以显式指定：

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct MODEL_SOURCE=modelscope scripts/dcu_quickstart.sh
```

脚本依次执行 GPU 环境检查、T4 算子正确性、T4 基准和 Qwen2.5 FP16 基线。结果分别保存到 `results/t4/` 和 `results/dcu/`。

## 3. 性能扫描建议

第一轮固定 `input_tokens=512`、`max_new_tokens=128`，扫描 batch：

```bash
MODEL=/path/to/Qwen2.5-7B-Instruct \
BATCH_SIZES=1,2,4,8,16 \
DTYPE=float16 ATTENTION=sdpa \
scripts/dcu_quickstart.sh
```

第二轮只运行基线，避免重复编译算子：

```bash
SKIP_KERNELS=1 MODEL=/path/to/Qwen2.5-7B-Instruct \
INPUT_TOKENS=2048 MAX_NEW_TOKENS=256 \
BATCH_SIZES=1,2,4,8 \
scripts/dcu_quickstart.sh
```

推荐先比较 `sdpa` 和 `eager`，再决定是否接入自定义 Triton/DTK 算子。每次只改变一个变量，并保留 JSON、设备型号、PyTorch/DTK 版本和 git commit。

## 4. 64 GiB 显存下的优化顺序

1. **确认基线**：FP16、`use_cache=True`、`model.eval()`、`torch.inference_mode()`，预热后同步计时。
2. **先调批量**：从 batch 1 逐步增加到 16，观察 tokens/s、requests/s 和峰值显存；OOM 前一个档位作为候选配置。
3. **按长度分桶**：真实评测请求按输入长度分桶，减少 padding。不要用统一的超长输入代替官方数据集结果。
4. **再测算子**：只有 `test_correctness.py` 通过后才比较 RMSNorm、Add+RMSNorm、SwiGLU 等 kernel；T4 的 v1/v2/v3 优势必须以 DCU 实测为准。
5. **最后做端到端**：把最快的 attention、batch 和 kernel 组合重新跑完整评测，避免只优化单算子却降低整体吞吐。

## 5. 结果判读

- `results/dcu/qwen25_*.json` 的 `median_output_tokens_per_second` 用于吞吐比较。
- `peak_allocated_bytes` 用于判断 batch 是否接近显存上限。
- `results/t4/bench_*.json` 用于单算子消融；不能直接和 NVIDIA A10/T4 的数字横向比较。
- 结果中的 `hip`、设备名和 `git_commit` 必须写入比赛报告，确保不同机器的数字可追溯。

## 常见问题

- **登录节点无 GPU**：正常，提交到 GPU 计算节点或 Notebook GPU 内核。
- **ModelScope 下载超时**：先在有网络的机器下载模型目录，再整体上传；不要依赖 Notebook 访问 GitHub。
- **`triton` 导入失败**：使用组委会提供的 DTK/Triton 环境；不要用 PyPI Triton 替换官方栈。
- **显存足够但吞吐低**：优先检查 batch、输入 padding、attention 实现和是否真的启用 KV cache，再看自定义 kernel。

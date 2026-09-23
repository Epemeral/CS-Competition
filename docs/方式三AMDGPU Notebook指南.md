# 方式三 AMDGPU Notebook 指南

当前方式三镜像为 `ubuntu22.04-rocm7.2.3-py312-torch2.12.0-1.40.1`，配置为 8 核、约 192 GiB 显存。这个配置足够加载 `Qwen2.5-7B-Instruct` 的 BF16 权重，并进行 batch、上下文长度和 attention 实现的消融。

## 先做什么

仓库中的 `notebooks/amdgpu_qwen25_7b.ipynb` 是入口。上传后按顺序执行：

1. 记录 Python、PyTorch、HIP、GPU、Transformers 和 ModelScope 版本。
2. 用 ModelScope 下载并缓存模型，检查 `config.json`。
3. 用 `scripts/qwen25_amdgpu_benchmark.py` 跑 eager 和 SDPA 两个基线。
4. 比较多个 batch 和输入长度，保存 JSON 原始结果。

脚本默认使用 BF16。若设备或算子不支持 BF16，再显式改成 `--dtype float16`，不要混用结果。

## 为什么先用 Transformers

Transformers 基线的目的，是固定 tokenizer、chat template、EOS 处理和吞吐计时口径。拿到组委会的官方镜像、评测集和 baseline 后，要把官方接口作为最终正确性标准。vLLM、连续批处理、PagedAttention 和 Triton/HIP kernel 应该在这个基线稳定后逐项接入。

## 推荐消融顺序

1. `eager` 与 `sdpa`：只改变 attention 实现。
2. 固定较快实现，扫描 batch size 和输入长度，记录显存峰值。
3. 接入 vLLM 或官方运行时，比较连续批处理/PagedAttention。
4. 运行 `kernels/tests/test_correctness.py`，再测 RMSNorm、SwiGLU、RoPE 和 QKV 融合。
5. 只有 profiling 证明算子是热点时，才把 kernel 接入端到端路径。

每一轮只改一个变量，并保留环境、git commit、参数、正确性结果和多次测量的 JSON。优化后的输出必须与参考实现逐 token 一致；只看单个 kernel 的加速比不能证明端到端变快。

## 直接命令

在 Notebook 的仓库目录中可以直接运行：

```bash
python scripts/qwen25_amdgpu_benchmark.py \
  --model /实际的模型目录 \
  --model-source transformers \
  --attention sdpa \
  --dtype bfloat16 \
  --batch-sizes 1,2,4,8 \
  --input-tokens 512 \
  --max-new-tokens 128 \
  --warmup 2 \
  --repeats 5 \
  --output results/amdgpu_sdpa.json
```

脚本会记录 `torch.version.hip`，因此可以确认 Notebook 实际使用的是 ROCm，而不是把 ROCm 误当成 CUDA 环境。

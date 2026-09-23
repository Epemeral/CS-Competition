#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 AMDGPU Notebook，避免手工复制长命令。"""

import json
from pathlib import Path


cells = []


def markdown(source: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    })


def code(source: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    })


markdown("""
# Qwen2.5-7B-Instruct · AMDGPU/ROCm 实验 Notebook

本 Notebook 面向 `ubuntu22.04-rocm7.2.3-py312-torch2.12.0-1.40.1` 镜像。
顺序固定为：环境记录 → 拉取代码 → ModelScope 模型缓存 → 正确性冒烟 →
Transformers 基线 → attention 消融 → 端到端优化。

Notebook 的结果只用于团队实验记录。拿到组委会官方 baseline、评测集和容器后，
必须用官方口径重新测量。
""")

markdown("""
## 0. 环境约束

- 设备通过 PyTorch 的 `torch.cuda` 接口访问，ROCm 版本从 `torch.version.hip` 读取。
- `Qwen2.5-7B-Instruct` 使用 BF16；显存不足时改成 FP16，并记录原因。
- 先跑正确性，再增加 batch、上下文长度或替换 attention。
- 不在同一个 Python 进程里同时常驻多个模型；每个方案结束后重启 kernel 或释放模型。
""")

code("""
import os, sys, platform, subprocess
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA alias available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("没有检测到 AMDGPU，请确认 Notebook 运行在方式三环境")
print("Device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory / 1024**3:.1f} GiB")
try:
    import transformers
    print("Transformers:", transformers.__version__)
except ImportError:
    print("Transformers 未安装，下一格安装")
try:
    import modelscope
    print("ModelScope:", modelscope.__version__)
except ImportError:
    print("ModelScope 未安装，下一格安装")
""")

code("""
# 镜像已经预装 ModelScope；只有缺包时才安装，避免覆盖 ROCm/torch。
%pip install -q "transformers>=4.51,<5" modelscope
""")

code("""
# 拉取当前实验分支。重跑时保留已有目录，避免重复下载。
import os
REPO = "https://github.com/Epemeral/CS-Competition.git"
BRANCH = "feat/t4-triton-kernels"
WORKDIR = "/mnt/data/CS-Competition"
if not os.path.exists(os.path.join(WORKDIR, ".git")):
    !git clone --depth 1 --branch {BRANCH} {REPO} {WORKDIR}
%cd /mnt/data/CS-Competition
!git log -1 --oneline
""")

markdown("""
## 1. 下载并检查模型

模型默认缓存到 `/mnt/data/model_cache`。这个目录可以按平台的持久化目录修改。
下载完成后先检查 `config.json`，确认是 `qwen2`、`Qwen2ForCausalLM` 和 7B 配置。
""")

code("""
from modelscope import snapshot_download
from pathlib import Path
import json, os

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CACHE = "/mnt/data/model_cache"
MODEL_DIR = snapshot_download(MODEL_ID, revision="master", cache_dir=MODEL_CACHE)
print("MODEL_DIR =", MODEL_DIR)
config = json.loads(Path(MODEL_DIR, "config.json").read_text(encoding="utf-8"))
for key in ("model_type", "architectures", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads"):
    print(f"{key}: {config.get(key)}")
""")

markdown("""
## 2. 先跑正确性和短基线

下面的命令只测 Transformers。`eager` 和 `sdpa` 生成相同 token 后，才能比较吞吐；
如果两者输出不一致，先停在这里检查版本、dtype 和 attention 实现。
""")

code("""
import subprocess, sys
base = [sys.executable, "scripts/qwen25_amdgpu_benchmark.py",
        "--model", MODEL_DIR, "--model-source", "transformers",
        "--batch-sizes", "1,2,4,8,16,32", "--input-tokens", "512",
        "--max-new-tokens", "128", "--warmup", "2", "--repeats", "5"]
subprocess.run(base + ["--attention", "eager", "--output", "results/amdgpu_eager.json"], check=True)
subprocess.run(base + ["--attention", "sdpa", "--output", "results/amdgpu_sdpa.json"], check=True)
""")

markdown("""
## 3. 读取对比结果

优先选择吞吐高且显存峰值稳定的方案。不要只看 batch=1；比赛的并发吞吐需要观察
多个 batch 和不同输入长度。每次只改变一个变量并保留 JSON。
""")

code("""
import json
from pathlib import Path

def show(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    print("\\n", path)
    print("attention:", data["settings"]["attention"], "dtype:", data["settings"]["dtype"])
    for row in data["measurements"]:
        peak = max((x["peak_allocated_bytes"] or 0) for x in row["samples"]) / 1024**3
        print(f"batch={row['batch_size']:<3} input={row['input_width']:<4} "
              f"tokens/s={row['median_output_tokens_per_second']:.2f} "
              f"peak={peak:.2f} GiB")
show("results/amdgpu_eager.json")
show("results/amdgpu_sdpa.json")
""")

markdown("""
## 4. 下一轮实验顺序

1. 固定最优 attention，扫描 `batch_size` 和 `input_tokens`，找到显存不溢出的吞吐峰值。
2. 有 vLLM/官方运行时后，在同一请求集上测试连续批处理和 PagedAttention。
3. 用 profiling 确认 RMSNorm、SwiGLU、RoPE 是否是热点，再启用 `kernels/` 中的融合实现。
4. 每个优化都做 token 级正确性、单项消融和端到端吞吐；只保留确实提升的方案。

不要把这个 Notebook 的 Transformers 数字直接填成比赛成绩。
""")

out = Path(__file__).resolve().parent.parent / "notebooks" / "amdgpu_qwen25_7b.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(out)

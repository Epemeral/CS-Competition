# -*- coding: utf-8 -*-
"""
生成可直接上传到 Google Colab 的 notebook。

用法：python build_colab_nb.py
产物：kernels_colab.ipynb
"""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ============================================================
md('''
# Triton Kernel 工程 · Colab 实测

在 Google Colab 上跑 `CS-Competition/kernels/` 的**正确性验证**和**性能基准**。

这个 notebook 会自动从 GitHub 拉代码，你不需要手动上传任何文件。

---

## 使用步骤

**① 开 GPU**（必须！）

菜单栏 → **代码执行程序** → **更改运行时类型** → 硬件加速器选 **T4 GPU** → 保存

> ⚠️ 不开 GPU 的话 Triton 跑不起来，会报 `no CUDA device`。

**② 全部运行**

按 **`Ctrl + F9`**（Mac 是 `Cmd + F9`），或者逐格按 `Shift + Enter`。

**③ 看结果**

最后一格会展示消融数据表格。

---

## 这个 notebook 会做什么

| 步骤 | 内容 |
|---|---|
| 1 | 检查环境（GPU / PyTorch / Triton 版本） |
| 2 | 从 GitHub 拉取代码 |
| 3 | **正确性验证**：Triton kernel 是否与 PyTorch 参考实现数值一致 |
| 4 | **性能基准**：测加速比和带宽利用率 |
| 5 | 展示消融数据 |

> 💡 第一次运行 `!` 开头的单元格时，Colab 会弹窗问「是否允许访问文件」，
> 点 **允许 / Allow**。
''')

# ============================================================
md('''
---

# ① 环境检查

先确认 Colab 给了什么环境。**每次开新会话都要先跑这一步**。
''')

code('''
import sys
import torch

print("=" * 62)
print("环境检查")
print("=" * 62)
print(f"Python   : {sys.version.split()[0]}")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA 可用 : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"显存      : {props.total_memory / 1024**3:.1f} GiB")
    print(f"计算能力  : {props.major}.{props.minor}")
else:
    print()
    print("❌ 没有检测到 GPU！")
    print("   请到「代码执行程序 → 更改运行时类型」里选 T4 GPU，然后重新运行。")

try:
    import triton
    print(f"Triton   : {triton.__version__}")
except ImportError:
    print("Triton   : 未安装（下一格会自动装）")

print("=" * 62)
''')

# ============================================================
code('''
# Colab 通常自带 triton，万一没有就装一个
try:
    import triton
    print("Triton 已就绪，跳过安装")
except ImportError:
    !pip install -q triton
    print("Triton 安装完成")
''')

# ============================================================
md('''
---

# ② 拉取代码

从 GitHub 拉 `feat/t4-triton-kernels` 分支。

> 仓库是**公开**的，所以不需要任何登录凭证。
''')

code('''
import os

REPO = "https://github.com/Epemeral/CS-Competition.git"
BRANCH = "feat/t4-triton-kernels"
WORKDIR = "/content/cs-comp"

# 清理旧目录（重跑时避免冲突）
if os.path.exists(WORKDIR):
    !rm -rf {WORKDIR}

# --depth 1 只拉最新一次提交，快很多
!git clone -b {BRANCH} --depth 1 {REPO} {WORKDIR}

print()
!ls -la {WORKDIR}/kernels
''')

# ============================================================
code('''
# 切到 kernels 目录（%cd 是持久生效的，后续单元格都在这里跑）
%cd /content/cs-comp/kernels
!find . -type f -name "*.py" | sort
''')

# ============================================================
md('''
---

# ③ 正确性验证

**这一步是入场券。** 比赛规则：正确性不通过 → 直接取消性能项资格。

这个脚本会检查：

- **形状覆盖**：含非 2 的幂 `(8,100)` `(16,1000)`，以及 Qwen2.5-7B 的真实尺寸 `(8,3584)` `(4,18944)`
- **dtype 覆盖**：fp32 / fp16 / bf16
- **极端情况**：全零输入、极大值 1e4、非连续输入、SwiGLU 尾部不对齐

**全绿才算通过。**
''')

code('''
!python tests/test_correctness.py
''')

# ============================================================
md('''
---

# ④ 性能基准

测三个 kernel 的 **PyTorch 原生 vs Triton** 加速比，以及**带宽利用率**。

## 为什么要看带宽利用率

RMSNorm / SwiGLU 都是 **memory-bound**（访存受限），对它们来说：

| 带宽利用率 | 含义 |
|---|---|
| 20% | 还有 5 倍空间，通常是访存不连续 |
| 80%+ | 接近硬件极限，再优化只能改算法 |

**这个数字比加速比更能说明问题**，答辩时非常有用。

## 关于计时方法

- **先预热 25 次**（排除编译、缓存未命中的影响）
- **正式跑 100 次取中位数**（单次测量不可信）

> 跑完需要一两分钟，耐心等。
''')

code('''
!python benchmarks/run_bench.py
''')

# ============================================================
md('''
---

# ⑤ 查看消融数据

基准跑完会自动落盘 JSON。这一格把结果整理成表格。
''')

code('''
import glob
import json
import os

files = sorted(glob.glob("results/bench_*.json"))

if not files:
    print("没找到结果文件。请先跑上一格（性能基准）。")
else:
    latest = files[-1]
    print(f"数据文件: {latest}")
    print()

    data = json.load(open(latest, encoding="utf-8"))

    print("=" * 74)
    print("运行环境")
    print("=" * 74)
    for k, v in data.get("_meta", {}).items():
        print(f"  {k:12s}: {v}")

    print()
    print("=" * 74)
    print("加速比汇总")
    print("=" * 74)
    print(f"  {'kernel':<22}{'平均加速比':>12}{'最佳加速比':>12}{'最佳带宽利用率':>18}")
    print("-" * 74)

    for name, rows in data.items():
        if name.startswith("_") or not rows:
            continue
        speedups = [r["speedup"] for r in rows]
        best = max(rows, key=lambda r: r["speedup"])
        util = best.get("bandwidth_util_pct")
        util_s = f"{util:.1f}%" if util else "—"
        print(f"  {name:<22}{sum(speedups)/len(speedups):>11.2f}x"
              f"{best['speedup']:>11.2f}x{util_s:>18}")

    print()
    print("=" * 74)
    print("💡 这些数据就是比赛要的『消融数据』，可以直接写进报告。")
    print("=" * 74)
''')

# ============================================================
md('''
---

# ⑥ 动手改一改（可选）

跑通之后，做这两个实验，理解会深一层：

## 实验 1：看 mask 式的边界处理有多重要

打开 `src/triton_kernels.py`，把 RMSNorm 的均方计算从：

```python
mean_square = tl.sum(x * x, axis=0) / N     # ✅ 用真实的 N
```

改成：

```python
mean_square = tl.sum(x * x, axis=0) / BLOCK_N   # ❌ 用编译期的 BLOCK_N
```

然后重跑正确性测试。**形状不是 2 的幂的那些用例会挂**——这就是 Triton 里最常见的坑。

## 实验 2：看看融合到底省了多少

在性能基准的输出里，找到 `fused_add_rmsnorm` 那一组。

对比「分开做（add + rmsnorm 两个 kernel）」和「融合成一个 kernel」的耗时。

**融合版的加速比通常明显高于单纯的 rmsnorm**——因为省掉了中间张量 `h` 的两次 HBM 往返。

## 实验 3：分块归约解决了什么问题

`src/triton_kernels.py` 里有两个 RMSNorm 实现：

| 版本 | 做法 | 适用场景 |
|---|---|---|
| `rmsnorm` (v1) | `BLOCK_N = next_power_of_2(N)`，整行一次处理 | N 较小 |
| `rmsnorm_v2` | `BLOCK_N` 固定上限（4096）+ 循环分块 | **N 很大**（如 18944） |

**为什么需要 v2**：N=18944 时 `next_power_of_2(18944) = 32768`，
一个 program 要同时持有 **32768 个 fp32** → 寄存器爆掉、spill 到 local memory
（走 HBM）→ **反而比 PyTorch 还慢**（实测 0.54x）。

v2 用「读两次 x」换「寄存器压力恒定」——第二次读大概率命中 L2，比 spill 便宜得多。

**看基准输出里 `(512, 18944)` 那一行**，对比 v1 和 v2 的耗时差距。

同一个思路也用在了 **`fused_add_rmsnorm_v2`** 上，所以「融合收益」那张表里
现在有三列：**分开做 / 融合 v1 / 融合 v2**。

这是 GPU kernel 的经典权衡，答辩时可以展开讲。

---

## 怎么在 Colab 里改代码

左边栏有个**文件夹图标**，点开能看到 `/content/cs-comp/kernels/`。
双击文件可以直接编辑，保存后重跑对应单元格即可。

> ⚠️ Colab 的修改**不会保存回 GitHub**，也不持久（会话断了就没了）。
> 要长期保留，需要下载下来再提交。
''')

# ============================================================
md('''
---

# 常见问题

## 报 `no CUDA device` / `Triton 无法运行`

没开 GPU。**代码执行程序 → 更改运行时类型 → T4 GPU**，然后**重新运行第 ① 格**。

## 报 `ModuleNotFoundError: No module named 'triton'`

跑一下第 ① 格后面的安装单元格（`!pip install -q triton`）。

## 报 `fatal: could not read Username for 'https://github.com'`

说明仓库变成了私有，或者分支名写错了。检查 `BRANCH` 变量。

## 跑到一半会话断了

Colab 免费版闲置约 90 分钟会断。重新跑第 ①②③ 格即可，代码不会丢（在 GitHub 上）。

## 想看显存占用

```python
!nvidia-smi
```

## 想装别的东西

```python
!pip install -q 包名
```
> ⚠️ 装的包**新会话就没了**，每次都要重装。

---

## 和 DCU 集群的关系

| | Colab | DCU 集群 |
|---|---|---|
| 硬件 | NVIDIA T4 | 海光 DCU（gfx936） |
| 用途 | **写代码、验证正确性、快速试错** | **跑真实性能、提交比赛结果** |
| 能跑 HIP 吗 | ❌ | ✅ |

**正确用法**：Colab 上把代码写对、逻辑跑通 → 拿到 DCU 上跑真实性能。

代码和方法 **100% 可迁移**（CUDA 和 HIP 语法 90% 相同），但**性能数字不能混用**。
''')

# ============================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("kernels_colab.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已生成: {out}")
print(f"共 {len(cells)} 个单元格")

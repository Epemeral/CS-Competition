# -*- coding: utf-8 -*-
"""Generate the portable T4/DCU optimization workbench notebook."""
import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": source.strip("\n").splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


md("""
# T4 / DCU Triton 优化工作台

这本 Notebook 用同一套入口完成环境检查、参考实现正确性、Qwen2.5-7B GQA 形状验证、单算子基准和结果导出。
它既可以在 NVIDIA T4 上验证 Triton 代码，也可以在 AMDGPU/DCU 镜像中运行真实测量。两类设备的性能数字必须分开记录。

## 运行顺序

1. 运行环境检查；没有 GPU 时仍可运行 CPU 参考检查。
2. 运行正确性脚本；通过后再测性能。
3. 运行单算子基准，并保存仓库根目录 `results/t4/` 下的 JSON。
4. 需要接入 vLLM 时，先在独立环境执行 integration patch，再做端到端 token 对齐。
""")

code("""
import json, os, platform, sys
import torch

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Memory GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
try:
    import triton
    print('Triton:', triton.__version__)
except Exception as exc:
    print('Triton unavailable:', repr(exc))
print('Platform:', platform.platform())
""")

code("""
# 设备指纹：ROCm / CUDA 两套栈要分清楚，否则报告里的设备描述会写错。
# 注意 torch.cuda.get_device_capability() 在 ROCm 环境下可能返回无意义的值，
# 判断设备真身要以 device_summary 打印的型号 + nvidia-smi / rocm-smi 为准。
print('设备名 :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
print('设备数 :', torch.cuda.device_count())
print('HIP    :', getattr(torch.version, 'hip', None))
print('CUDA   :', torch.version.cuda)
if torch.cuda.is_available():
    print('显存   :', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GiB')

import triton
print('Triton :', triton.__version__)
""")

code("""
from pathlib import Path
import importlib
import shutil
import subprocess
import sys

ROOT = Path.cwd()
REPO = 'https://github.com/Epemeral/CS-Competition.git'
BRANCH = 'feat/t4-triton-kernels'
CLONE_DIR = Path('/content/cs-comp')

# 判据不只是「文件在不在」，还要「版本够不够新」。
# 旧版 reference.py 的 qkv_split_rope 没有 num_key_value_heads 参数，
# 会导致后面整个 GQA 流程跑不通 —— 这是个典型的「文件在、但内容旧」陷阱。
GQA_MARKER = 'num_key_value_heads: int | None = None'


def is_kernel_tree(path):
    return (path / 'src' / 'reference.py').is_file() and \\
           (path / 'tests' / 'test_correctness.py').is_file()


def supports_gqa(path):
    ref = path / 'src' / 'reference.py'
    if not ref.is_file():
        return False
    return GQA_MARKER in ref.read_text(encoding='utf-8')


def sync_from_github(target):
    \"\"\"强制从 GitHub 拉最新代码，覆盖 target。失败则抛异常。\"\"\"
    if target.exists():
        shutil.rmtree(target)
    print(f'从 GitHub 拉取最新 {BRANCH} ...')
    subprocess.run(['git', 'clone', '-b', BRANCH, '--depth', '1',
                    REPO, str(target)], check=True)


# ---- 1. 先找本地已有的 kernels ----
candidates = [
    ROOT / 'kernels',
    ROOT,
    CLONE_DIR / 'kernels',
    Path('/content/CS-Competition/kernels'),
]
KERNELS = next((p for p in candidates if is_kernel_tree(p)), None)

# ---- 2. 没找到，或找到了但版本太旧 → 都去 GitHub 拉 ----
if KERNELS is None:
    print('未发现 kernels 源码，尝试从 GitHub 下载 ...')
    try:
        sync_from_github(CLONE_DIR)
        KERNELS = CLONE_DIR / 'kernels'
    except Exception as exc:
        raise RuntimeError(
            '无法自动获取仓库。请将仓库中的 kernels 文件夹上传到 '
            f'{CLONE_DIR}/kernels 后重新运行本单元。原始错误: ' + repr(exc)
        ) from exc
elif not supports_gqa(KERNELS):
    print(f'⚠️  本地 kernels 版本过旧（不支持 GQA）: {KERNELS}')
    print('   尝试从 GitHub 拉取最新版本 ...')
    try:
        sync_from_github(CLONE_DIR)
        KERNELS = CLONE_DIR / 'kernels'
    except Exception as exc:
        print('   拉取失败，继续用本地旧版 —— 后面会报错，请手动更新。')
        print('   原因:', repr(exc))
else:
    print(f'✅ 本地 kernels 已是最新（支持 GQA）: {KERNELS}')

# ---- 3. 最终校验 ----
if not is_kernel_tree(KERNELS):
    raise FileNotFoundError(
        f'找到的路径不完整: {KERNELS}。需要 src/reference.py 和 tests/test_correctness.py。'
    )
if not supports_gqa(KERNELS):
    raise RuntimeError(
        f'当前 kernels 仍不支持 GQA（缺少 num_key_value_heads 参数）: {KERNELS}\\n'
        '请强制更新：\\n'
        f'  !rm -rf {CLONE_DIR} && git clone -b {BRANCH} --depth 1 {REPO} {CLONE_DIR}\\n'
        '然后 Runtime -> Restart session，从头重跑。'
    )

# ---- 4. 清掉 Python 缓存的旧模块（换了文件但内存里还是旧对象）----
for name in ('reference', 'triton_kernels', 'bench_utils'):
    sys.modules.pop(name, None)
importlib.invalidate_caches()

SRC = KERNELS / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
print()
print('KERNELS =', KERNELS)
print('reference.py =', SRC / 'reference.py')
print('已清除 reference / triton_kernels / bench_utils 的模块缓存')
""")

md("""
## 1. CPU 参考实现与 Qwen2.5 配置

Qwen2.5-7B 使用 GQA：`hidden_size=3584`、`num_attention_heads=28`、`num_key_value_heads=4`、`head_dim=128`。因此 QKV 投影输出是 `(28+4+4)*128=4608`，不能按 `3*28*128` 切分。
""")

code("""
import importlib, inspect, sys
import torch

# Notebook 内核可能残留旧的 reference 模块；始终从本次定位的源码重新加载。
sys.modules.pop('reference', None)
importlib.invalidate_caches()
import reference as ref
print('reference loaded from:', ref.__file__)
print('qkv_split_rope signature:', inspect.signature(ref.qkv_split_rope))
if 'num_key_value_heads' not in inspect.signature(ref.qkv_split_rope).parameters:
    raise RuntimeError('当前 reference.py 不支持 GQA，请上传最新 kernels/src/reference.py')

qwen = dict(hidden_size=3584, num_attention_heads=28,
            num_key_value_heads=4, head_dim=128,
            intermediate_size=18944)
B, T = 1, 17
qkv = torch.randn(B, T, (qwen['num_attention_heads'] + 2*qwen['num_key_value_heads']) * qwen['head_dim'])
cos, sin = ref.build_rope_cache(T, qwen['head_dim'], device='cpu', dtype=qkv.dtype)
q, k, v = ref.qkv_split_rope(qkv, cos, sin, qwen['num_attention_heads'],
                              qwen['head_dim'], qwen['num_key_value_heads'])
print('Q/K/V:', tuple(q.shape), tuple(k.shape), tuple(v.shape))
assert q.shape[-1] == 28*128 and k.shape[-1] == v.shape[-1] == 4*128
print('GQA reference check: PASS')
""")

md("""
## 2. GPU 正确性

正确性是硬门槛。脚本覆盖非 2 的幂、FP16/BF16、极端值、非连续输入、分块阈值和真实 Qwen2.5 GQA。没有 GPU 时跳过 Triton，而不是把跳过误报成通过。
""")

code("""
import subprocess, sys
if not torch.cuda.is_available():
    print('No GPU: Triton correctness is deferred to T4 or DCU runtime.')
else:
    result = subprocess.run([sys.executable, str(KERNELS/'tests'/'test_correctness.py')], text=True)
    print('exit code:', result.returncode)
    if result.returncode:
        raise RuntimeError('correctness checks failed')
""")

md("""
## 3. 单算子 benchmark 与 autotune

RMSNorm 和 SwiGLU 是访存受限算子。SwiGLU 已按 block/warp autotune；RMSNorm 的 autotune 搜索 warp 配置，`rmsnorm_v2` 只在维度超过 block cap 时分块，避免 Qwen2.5 的 3584 维度被双遍读取。首次调用包含编译，不要把首次调用当作 steady-state。

### 融合 Add+RMSNorm 的三个版本

`fused_add_rmsnorm` 走了三代，每一代解决上一代的具体瓶颈：

| 版本 | 策略 | 适用场景 | 已知问题 |
|---|---|---|---|
| v1（整行处理） | 一个 program 处理整行 | 小 N | **N 大时寄存器溢出**：`next_power_of_2(18944)=32768`，一个 program 要持有 32768 个 fp32 |
| v2（分块归约） | BLOCK_N 固定上限，循环分块 | 中等 N | 第二遍**重复读取 X/R** 并重算 h，大 N 时带宽压力大 |
| **v3（两阶段）** | 阶段1 把 h 写回显存，阶段2 只读 h 做归一化 | **大 N** | 多一次 h 的写+读 |

v3 的取舍值得记一笔：它用「多一次 h 的 HBM 往返」换「不再重复读 X/R」。当 N 很大时，X 和 R 是两份数据、h 只有一份，所以 v3 的读取量更小。

```text
v2 的访存：读 X + 读 R  →  算 h（第一遍）
           读 X + 读 R  →  再算 h 用于归一化（第二遍）  ← 重复！
           ↓
v3 的访存：读 X + 读 R → 写 h → 存 partial（阶段1）
           读 h + 读 W → 写 y                    （阶段2）  ← 只读一份
```

三个版本都保持同一个 `(y, h)` 契约，所以可以直接互换、逐项消融。
`v3` 内部有自适应退化：`N <= block_cap` 时自动走 v1，避免小尺寸下被双阶段启动开销拖累。
""")

code("""
import subprocess, sys
if not torch.cuda.is_available():
    print('No GPU: benchmark is deferred to T4 or DCU runtime.')
else:
    result = subprocess.run([sys.executable, str(KERNELS/'benchmarks'/'run_bench.py')], text=True)
    print('exit code:', result.returncode)
    if result.returncode:
        raise RuntimeError('benchmark failed')
""")

code("""
from pathlib import Path
import json
result_dir = KERNELS.parent / 'results' / 't4'
result_files = sorted(result_dir.glob('bench_*.json'))
if result_files:
    latest = result_files[-1]
    data = json.loads(latest.read_text(encoding='utf-8'))
    print('Latest:', latest)
    for name, rows in data.items():
        if name.startswith('_') or not rows:
            continue
        best = max(rows, key=lambda row: row.get('speedup', 0))
        print(f'{name}: best speedup={best.get("speedup", 0):.2f}x, shape={best.get("shape")}')
else:
    print('No benchmark JSON yet.')
""")

md("""
## 4. 三个版本的横向对比

上面那格只打了「最佳加速比」。这一格把 `fused_add_rmsnorm` 的 v1 / v2 / v3 拉出来逐行对比 ——
这是写报告时最有说服力的一张表，因为它同时展示了**问题**（v1 在大 N 崩掉）和**解法**（v2/v3 修回来）。

bench 里额外加了一组 `N` 扫描（`1024 / 3584 / 4096 / 8192 / 12288 / 18944`），
专门跨过 `block_cap=4096` 这个分界点，这样才看得出**交叉发生在哪里**。
""")

code("""
from pathlib import Path
import json

result_dir = KERNELS.parent / 'results' / 't4'
files = sorted(result_dir.glob('bench_*.json'))
if not files:
    print('还没有 bench JSON，先跑上一格。')
else:
    rows = json.loads(files[-1].read_text(encoding='utf-8')).get('fused_add_rmsnorm', [])

    print('=' * 96)
    print('融合 Add+RMSNorm：v1（整行） / v2（分块归约） / v3（两阶段）')
    print('=' * 96)
    print(f"  {'形状':<16}{'dtype':<10}{'分开':>9}{'v1':>9}{'v2':>9}{'v3':>9}"
          f"{'v1加速':>8}{'v2加速':>8}{'v3加速':>8}  {'最优':<5}")
    print('-' * 96)

    for r in rows:
        s1 = r.get('speedup_v1') or 0
        s2 = r.get('speedup_v2') or 0
        s3 = r.get('speedup_v3') or 0
        best = max([('v1', s1), ('v2', s2), ('v3', s3)], key=lambda t: t[1])[0]

        print(f"  {str(r.get('shape')):<16}{str(r.get('dtype')):<10}"
              f"{r.get('unfused_ms', 0):>9.3f}"
              f"{r.get('v1_ms', 0):>9.3f}"
              f"{r.get('v2_ms', 0):>9.3f}"
              f"{r.get('v3_ms', 0):>9.3f}"
              f"{s1:>7.2f}x{s2:>7.2f}x{s3:>7.2f}x   {best}")

    # ---- 按 N 分组，找交叉点 ----
    print()
    print('按 N 看趋势（取该 N 下三种 dtype 的平均）：')
    print(f"  {'N':>8}{'形状':>14}   {'v1':>8}{'v2':>8}{'v3':>8}   最优")
    print('-' * 60)

    by_n = {}
    for r in rows:
        n = r['shape'][-1]
        by_n.setdefault(n, []).append(r)

    for n in sorted(by_n):
        group = by_n[n]
        avg = lambda key: sum(x.get(key) or 0 for x in group) / len(group)
        a1, a2, a3 = avg('speedup_v1'), avg('speedup_v2'), avg('speedup_v3')
        best = max([('v1', a1), ('v2', a2), ('v3', a3)], key=lambda t: t[1])[0]
        flag = '  ← 分界点' if n == 4096 else ''
        print(f"  {n:>8}{str(group[0]['shape']):>14}   "
              f"{a1:>7.2f}x{a2:>7.2f}x{a3:>7.2f}x   {best}{flag}")
""")

md("""
## 5. 结果解读与后续路线

- 单算子加速不等于端到端加速；必须测 vLLM/eager 的 TTFT、TPOT、吞吐、P50/P95 和峰值显存。
- 如果 RMSNorm/SwiGLU 占比低，继续堆小 kernel 的收益有限，应转向 attention、KV cache 或调度 profiling。
- **v1 → v2 → v3 这条链本身就是一份完整的优化叙事**：发现问题（寄存器 spill）→ 第一版解法（分块归约）→ 发现新问题（重复读输入）→ 第二版解法（两阶段）。报告里按这个顺序讲，比单纯堆加速比更有说服力。
- 小尺寸下（如 `(1, 3584)`）注意启动开销占比；SwiGLU 在极小尺寸可能低于 1.0x，属于正常现象，应记录而不是隐藏。
- FlashAttention、PagedAttention 和量化属于后续高风险方向，先复用参考公式和测试方法，再在 DCU 上重新验证，不能直接搬运 CUDA 数字。
- 保存设备名、PyTorch/Triton/DTK 版本、commit、warmup、重复次数和原始 JSON，报告才能复现。
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                    "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).with_name('t4_optimization_workbench.ipynb')
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('generated', out)

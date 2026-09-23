# -*- coding: utf-8 -*-
"""
性能基准 —— 产出比赛要的「消融数据」。

【这个脚本产出什么】
    1. 控制台表格：每个 kernel 的 PyTorch vs Triton 耗时、加速比、带宽利用率
    2. results/*.json：机器可读的原始数据（写报告、画图都用它）
    3. 融合收益对比：证明「融合」到底省了多少

【怎么用这些数据】
    · 加速比 → 性能分
    · before/after 对比 → 创新分（6~12 分的材料）
    · 带宽利用率 → 说明「还有多少优化空间」，答辩时非常有用

【一条重要提醒】
    RMSNorm / SwiGLU 都是 memory-bound（访存受限）。
    对它们来说带宽利用率比算力利用率重要得多：
        20%  → 还有 5 倍空间，通常是访存不连续
        80%+ → 接近硬件极限，再优化只能改算法
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import bench_utils as bu        # noqa: E402
import reference as ref         # noqa: E402
import triton_kernels as tk     # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ============================================================
# 基准配置
# ============================================================
BENCH_SHAPES = [
    (1, 3584),        # 单条推理（batch=1）
    (8, 3584),
    (32, 3584),
    (128, 3584),      # 高并发
    (128, 18944),     # SwiGLU 的中间维度
]

BENCH_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

WARMUP = 25
REP = 100


# ============================================================
# 一、RMSNorm
# ============================================================
def bench_rmsnorm(device, peak_bw, results):
    print()
    print("=" * 78)
    print("RMSNorm：PyTorch 原生 vs Triton")
    print("=" * 78)
    bu.header(f"  {'形状':<16}{'dtype':<10}{'PyTorch':>10}{'Triton':>10}"
              f"{'加速比':>9}{'带宽':>11}{'利用率':>9}")

    for shape in BENCH_SHAPES:
        for dtype in BENCH_DTYPES:
            x = torch.randn(*shape, device=device, dtype=dtype)
            w = torch.randn(shape[-1], device=device, dtype=dtype)

            t_ref = bu.bench(lambda: ref.rmsnorm(x, w, eps=1e-6),
                             warmup=WARMUP, rep=REP)
            t_tri = bu.bench(lambda: tk.rmsnorm(x, w, eps=1e-6),
                             warmup=WARMUP, rep=REP)

            nbytes = bu.bytes_for_rmsnorm(shape, x.element_size())
            bw = bu.bandwidth_gbps(nbytes, t_tri["median_ms"])
            util = bw / peak_bw * 100 if peak_bw else None

            line = (f"  {str(shape):<16}{str(dtype).split('.')[-1]:<10}"
                    f"{t_ref['median_ms']:>10.3f}{t_tri['median_ms']:>10.3f}"
                    f"{t_ref['median_ms'] / t_tri['median_ms']:>8.2f}x"
                    f"{bw:>10.1f}")
            line += f"{util:>8.1f}%" if util else f"{'—':>9}"
            if t_tri["spread"] > 0.2:
                line += "  ⚠️波动大"
            print(line)

            results["rmsnorm"].append({
                "shape": list(shape),
                "dtype": str(dtype).split(".")[-1],
                "pytorch_ms": t_ref["median_ms"],
                "triton_ms": t_tri["median_ms"],
                "speedup": t_ref["median_ms"] / t_tri["median_ms"],
                "bandwidth_gbps": bw,
                "bandwidth_util_pct": util,
                "spread": t_tri["spread"],
            })


# ============================================================
# 二、融合收益：add + rmsnorm
# ============================================================
def bench_fusion_gain(device, peak_bw, results):
    """对比「不融合」和「融合」—— 这是最有说服力的一组数据。"""
    print()
    print("=" * 78)
    print("融合收益：add + rmsnorm 分开做 vs 融合成一个 kernel")
    print("=" * 78)
    bu.header(f"  {'形状':<16}{'dtype':<10}{'分开':>10}{'融合':>10}"
              f"{'加速比':>9}{'带宽':>11}{'利用率':>9}")

    for shape in BENCH_SHAPES:
        for dtype in BENCH_DTYPES:
            x = torch.randn(*shape, device=device, dtype=dtype)
            r = torch.randn(*shape, device=device, dtype=dtype)
            w = torch.randn(shape[-1], device=device, dtype=dtype)

            def unfused():
                h = x + r                      # kernel 1：写 h 到 HBM
                return ref.rmsnorm(h, w, eps=1e-6)   # kernel 2：再读 h

            def fused():
                return tk.fused_add_rmsnorm(x, r, w, eps=1e-6)

            t_un = bu.bench(unfused, warmup=WARMUP, rep=REP)
            t_fu = bu.bench(fused, warmup=WARMUP, rep=REP)

            # 融合版搬运量：读 x + 读 r + 读 w + 写 y + 写 h
            m = 1
            for s in shape[:-1]:
                m *= s
            n = shape[-1]
            nb = m * n * x.element_size() * 4 + n * x.element_size()
            bw = bu.bandwidth_gbps(nb, t_fu["median_ms"])
            util = bw / peak_bw * 100 if peak_bw else None

            line = (f"  {str(shape):<16}{str(dtype).split('.')[-1]:<10}"
                    f"{t_un['median_ms']:>10.3f}{t_fu['median_ms']:>10.3f}"
                    f"{t_un['median_ms'] / t_fu['median_ms']:>8.2f}x"
                    f"{bw:>10.1f}")
            line += f"{util:>8.1f}%" if util else f"{'—':>9}"
            print(line)

            results["fused_add_rmsnorm"].append({
                "shape": list(shape),
                "dtype": str(dtype).split(".")[-1],
                "unfused_ms": t_un["median_ms"],
                "fused_ms": t_fu["median_ms"],
                "speedup": t_un["median_ms"] / t_fu["median_ms"],
                "bandwidth_gbps": bw,
                "bandwidth_util_pct": util,
            })


# ============================================================
# 三、SwiGLU
# ============================================================
def bench_swiglu(device, peak_bw, results):
    print()
    print("=" * 78)
    print("SwiGLU：PyTorch 原生 vs Triton")
    print("=" * 78)
    bu.header(f"  {'形状':<16}{'dtype':<10}{'PyTorch':>10}{'Triton':>10}"
              f"{'加速比':>9}{'带宽':>11}{'利用率':>9}")

    for shape in BENCH_SHAPES:
        for dtype in BENCH_DTYPES:
            g = torch.randn(*shape, device=device, dtype=dtype)
            u = torch.randn(*shape, device=device, dtype=dtype)

            t_ref = bu.bench(lambda: ref.swiglu(g, u), warmup=WARMUP, rep=REP)
            t_tri = bu.bench(lambda: tk.swiglu(g, u), warmup=WARMUP, rep=REP)

            nbytes = bu.bytes_for_swiglu(shape, g.element_size())
            bw = bu.bandwidth_gbps(nbytes, t_tri["median_ms"])
            util = bw / peak_bw * 100 if peak_bw else None

            line = (f"  {str(shape):<16}{str(dtype).split('.')[-1]:<10}"
                    f"{t_ref['median_ms']:>10.3f}{t_tri['median_ms']:>10.3f}"
                    f"{t_ref['median_ms'] / t_tri['median_ms']:>8.2f}x"
                    f"{bw:>10.1f}")
            line += f"{util:>8.1f}%" if util else f"{'—':>9}"
            print(line)

            results["swiglu"].append({
                "shape": list(shape),
                "dtype": str(dtype).split(".")[-1],
                "pytorch_ms": t_ref["median_ms"],
                "triton_ms": t_tri["median_ms"],
                "speedup": t_ref["median_ms"] / t_tri["median_ms"],
                "bandwidth_gbps": bw,
                "bandwidth_util_pct": util,
            })


# ============================================================
# 四、摘要
# ============================================================
def print_summary(results):
    print()
    print("=" * 78)
    print("消融数据摘要（可直接贴进报告）")
    print("=" * 78)

    for name, rows in results.items():
        if not rows or name.startswith("_"):
            continue
        speedups = [r["speedup"] for r in rows]
        best = max(rows, key=lambda r: r["speedup"])
        print(f"\n  {name}")
        print(f"    平均加速比 : {sum(speedups) / len(speedups):.2f}x")
        print(f"    最佳加速比 : {best['speedup']:.2f}x"
              f"  （{best['shape']} {best['dtype']}）")
        if best.get("bandwidth_util_pct"):
            print(f"    最佳带宽利用率 : {best['bandwidth_util_pct']:.1f}%")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 78)
    print("Triton Kernel 性能基准")
    print("=" * 78)

    if not torch.cuda.is_available():
        print()
        print("⚠️  当前环境没有 GPU，无法测性能。")
        print("请在 Colab（T4 GPU）或 DCU 集群上运行。")
        return 1

    device = "cuda"
    print()
    info = bu.device_summary()
    peak_bw = info["peak_bw_gbps"]

    results = {
        "_meta": {
            "device": info["device"],
            "torch": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "warmup": WARMUP,
            "rep": REP,
        },
        "rmsnorm": [],
        "fused_add_rmsnorm": [],
        "swiglu": [],
    }

    try:
        import triton
        results["_meta"]["triton"] = triton.__version__
    except ImportError:
        print("Triton 未安装 ❌")
        return 1

    bench_rmsnorm(device, peak_bw, results)
    bench_fusion_gain(device, peak_bw, results)
    bench_swiglu(device, peak_bw, results)

    print_summary(results)

    # ---- 落盘 ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print()
    print("=" * 78)
    print(f"原始数据已保存: {out}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())

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
    (32, 3584),       # 小批量
    (128, 3584),      # 中批量
    (512, 3584),      # 大批量 ← 新增：小尺寸喂不饱 GPU，必须加大才测得出真实带宽
    (2048, 3584),     # 超大 ← 新增
    (512, 18944),     # 大 N ← 新增：专门暴露 v1 的 BLOCK_N 爆炸问题
]

BENCH_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

WARMUP = 25
REP = 100


# ============================================================
# 一、RMSNorm
# ============================================================
def bench_rmsnorm(device, peak_bw, results):
    print()
    print("=" * 96)
    print("RMSNorm：PyTorch 原生 vs v1（整行处理） vs v2（分块归约）")
    print("=" * 96)
    bu.header(f"  {'形状':<15}{'dtype':<9}{'PyTorch':>9}{'v1':>9}{'v2':>9}"
              f"{'v1加速':>8}{'v2加速':>8}{'v2带宽':>10}{'利用率':>8}")

    for shape in BENCH_SHAPES:
        for dtype in BENCH_DTYPES:
            x = torch.randn(*shape, device=device, dtype=dtype)
            w = torch.randn(shape[-1], device=device, dtype=dtype)

            t_ref = bu.bench(lambda: ref.rmsnorm(x, w, eps=1e-6),
                             warmup=WARMUP, rep=REP)
            t_v1 = bu.bench(lambda: tk.rmsnorm(x, w, eps=1e-6),
                            warmup=WARMUP, rep=REP)
            t_v2 = bu.bench(lambda: tk.rmsnorm_v2(x, w, eps=1e-6),
                            warmup=WARMUP, rep=REP)

            nbytes = bu.bytes_for_rmsnorm(shape, x.element_size())
            bw = bu.bandwidth_gbps(nbytes, t_v2["median_ms"])
            util = bw / peak_bw * 100 if peak_bw else None

            line = (f"  {str(shape):<15}{str(dtype).split('.')[-1]:<9}"
                    f"{t_ref['median_ms']:>9.3f}"
                    f"{t_v1['median_ms']:>9.3f}"
                    f"{t_v2['median_ms']:>9.3f}"
                    f"{t_ref['median_ms'] / t_v1['median_ms']:>7.2f}x"
                    f"{t_ref['median_ms'] / t_v2['median_ms']:>7.2f}x"
                    f"{bw:>10.1f}")
            line += f"{util:>7.1f}%" if util else f"{'—':>8}"
            print(line)

            results["rmsnorm"].append({
                "shape": list(shape),
                "dtype": str(dtype).split(".")[-1],
                "pytorch_ms": t_ref["median_ms"],
                "v1_ms": t_v1["median_ms"],
                "v2_ms": t_v2["median_ms"],
                # 兼容旧字段名
                "triton_ms": t_v2["median_ms"],
                "speedup": t_ref["median_ms"] / t_v2["median_ms"],
                "speedup_v1": t_ref["median_ms"] / t_v1["median_ms"],
                "speedup_v2": t_ref["median_ms"] / t_v2["median_ms"],
                "bandwidth_gbps": bw,
                "bandwidth_util_pct": util,
            })


# ============================================================
# 二、融合收益：add + rmsnorm
# ============================================================
def bench_fusion_gain(device, peak_bw, results):
    """对比「不融合」和「融合」—— 这是最有说服力的一组数据。"""
    print()
    print("=" * 96)
    print("融合收益：分开做 vs 融合 v1（整行） vs 融合 v2（分块归约）")
    print("=" * 96)
    bu.header(f"  {'形状':<15}{'dtype':<9}{'分开':>9}{'融合v1':>9}{'融合v2':>9}"
              f"{'v1加速':>8}{'v2加速':>8}{'v2带宽':>10}{'利用率':>8}")

    for shape in BENCH_SHAPES:
        for dtype in BENCH_DTYPES:
            x = torch.randn(*shape, device=device, dtype=dtype)
            r = torch.randn(*shape, device=device, dtype=dtype)
            w = torch.randn(shape[-1], device=device, dtype=dtype)

            def unfused():
                h = x + r                             # kernel 1：写 h 到 HBM
                return ref.rmsnorm(h, w, eps=1e-6)    # kernel 2：再读 h

            t_un = bu.bench(unfused, warmup=WARMUP, rep=REP)
            t_v1 = bu.bench(lambda: tk.fused_add_rmsnorm(x, r, w, eps=1e-6),
                            warmup=WARMUP, rep=REP)
            t_v2 = bu.bench(lambda: tk.fused_add_rmsnorm_v2(x, r, w, eps=1e-6),
                            warmup=WARMUP, rep=REP)

            # 融合版搬运量：读 x + 读 r + 读 w + 写 y + 写 h
            m = 1
            for s in shape[:-1]:
                m *= s
            n = shape[-1]
            nb = m * n * x.element_size() * 4 + n * x.element_size()
            bw = bu.bandwidth_gbps(nb, t_v2["median_ms"])
            util = bw / peak_bw * 100 if peak_bw else None

            line = (f"  {str(shape):<15}{str(dtype).split('.')[-1]:<9}"
                    f"{t_un['median_ms']:>9.3f}"
                    f"{t_v1['median_ms']:>9.3f}"
                    f"{t_v2['median_ms']:>9.3f}"
                    f"{t_un['median_ms'] / t_v1['median_ms']:>7.2f}x"
                    f"{t_un['median_ms'] / t_v2['median_ms']:>7.2f}x"
                    f"{bw:>10.1f}")
            line += f"{util:>7.1f}%" if util else f"{'—':>8}"
            print(line)

            results["fused_add_rmsnorm"].append({
                "shape": list(shape),
                "dtype": str(dtype).split(".")[-1],
                "unfused_ms": t_un["median_ms"],
                "v1_ms": t_v1["median_ms"],
                "v2_ms": t_v2["median_ms"],
                # 兼容旧字段名（v2 为推荐版本）
                "fused_ms": t_v2["median_ms"],
                "speedup": t_un["median_ms"] / t_v2["median_ms"],
                "speedup_v1": t_un["median_ms"] / t_v1["median_ms"],
                "speedup_v2": t_un["median_ms"] / t_v2["median_ms"],
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
# 四、RoPE
# ============================================================
def bench_rope(device, peak_bw, results):
    print()
    print("=" * 96)
    print("RoPE：PyTorch 原生 vs Triton")
    print("=" * 96)
    bu.header(f"  {'(B,H,T,D)':<24}{'dtype':<9}{'PyTorch':>10}{'Triton':>10}"
              f"{'加速比':>9}{'带宽':>10}{'利用率':>8}")

    # Qwen2.5-7B: 28 个注意力头，head_dim=128
    cases = [
        (1, 28, 512, 128),
        (4, 28, 512, 128),
        (8, 28, 1024, 128),
        (8, 4, 1024, 128),      # GQA 场景下的 KV 头（4 个）
    ]

    for dtype in BENCH_DTYPES:
        for (B, H, T, D) in cases:
            q = torch.randn(B, H, T, D, device=device, dtype=dtype)
            cos, sin = ref.build_rope_cache(T, D, device=device, dtype=dtype)

            t_ref = bu.bench(lambda: ref.rope(q, cos, sin),
                             warmup=WARMUP, rep=REP)
            t_tri = bu.bench(lambda: tk.rope(q, cos, sin),
                             warmup=WARMUP, rep=REP)

            nbytes = bu.bytes_for_rope(B * H, T, D, q.element_size())
            bw = bu.bandwidth_gbps(nbytes, t_tri["median_ms"])
            util = bw / peak_bw * 100 if peak_bw else None

            line = (f"  {str((B, H, T, D)):<24}{str(dtype).split('.')[-1]:<9}"
                    f"{t_ref['median_ms']:>10.3f}{t_tri['median_ms']:>10.3f}"
                    f"{t_ref['median_ms'] / t_tri['median_ms']:>8.2f}x"
                    f"{bw:>10.1f}")
            line += f"{util:>7.1f}%" if util else f"{'—':>8}"
            print(line)

            results["rope"].append({
                "shape": [B, H, T, D],
                "dtype": str(dtype).split(".")[-1],
                "pytorch_ms": t_ref["median_ms"],
                "triton_ms": t_tri["median_ms"],
                "speedup": t_ref["median_ms"] / t_tri["median_ms"],
                "bandwidth_gbps": bw,
                "bandwidth_util_pct": util,
            })


# ============================================================
# 五、QKV 切分 + RoPE 融合
# ============================================================
def bench_qkv_rope(device, peak_bw, results):
    print()
    print("=" * 96)
    print("QKV 切分 + RoPE：PyTorch 原生（3 次 kernel） vs Triton 融合（1 次）")
    print("=" * 96)
    bu.header(f"  {'(B,T,H,D)':<24}{'dtype':<9}{'PyTorch':>10}{'Triton':>10}"
              f"{'加速比':>9}{'带宽':>10}{'利用率':>8}")

    cases = [
        (1, 512, 28, 128),
        (4, 512, 28, 128),
        (8, 1024, 28, 128),
        (8, 1024, 4, 128),
    ]

    for dtype in BENCH_DTYPES:
        for (B, T, H, D) in cases:
            HD = H * D
            qkv = torch.randn(B, T, 3 * HD, device=device, dtype=dtype)
            cos, sin = ref.build_rope_cache(T, D, device=device, dtype=dtype)

            t_ref = bu.bench(lambda: ref.qkv_split_rope(qkv, cos, sin, H, D),
                             warmup=WARMUP, rep=REP)
            t_tri = bu.bench(lambda: tk.qkv_split_rope(qkv, cos, sin, H, D),
                             warmup=WARMUP, rep=REP)

            nbytes = bu.bytes_for_qkv_rope(B, T, H, D, qkv.element_size())
            bw = bu.bandwidth_gbps(nbytes, t_tri["median_ms"])
            util = bw / peak_bw * 100 if peak_bw else None

            line = (f"  {str((B, T, H, D)):<24}{str(dtype).split('.')[-1]:<9}"
                    f"{t_ref['median_ms']:>10.3f}{t_tri['median_ms']:>10.3f}"
                    f"{t_ref['median_ms'] / t_tri['median_ms']:>8.2f}x"
                    f"{bw:>10.1f}")
            line += f"{util:>7.1f}%" if util else f"{'—':>8}"
            print(line)

            results["qkv_split_rope"].append({
                "shape": [B, T, H, D],
                "dtype": str(dtype).split(".")[-1],
                "pytorch_ms": t_ref["median_ms"],
                "triton_ms": t_tri["median_ms"],
                "speedup": t_ref["median_ms"] / t_tri["median_ms"],
                "bandwidth_gbps": bw,
                "bandwidth_util_pct": util,
            })


# ============================================================
# 六、摘要
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
        best_sp = max(rows, key=lambda r: r["speedup"])

        print(f"\n  {name}")
        print(f"    平均加速比 : {sum(speedups) / len(speedups):.2f}x")
        print(f"    最佳加速比 : {best_sp['speedup']:.2f}x"
              f"  （{best_sp['shape']} {best_sp['dtype']}）")

        # ⚠️ 带宽利用率要单独取最大值 —— 它和加速比的最优行往往不是同一个
        #    （小尺寸数据量小、加速比看着高，但利用率极低，会误导判断）
        bw_rows = [r for r in rows if r.get("bandwidth_util_pct")]
        if bw_rows:
            best_bw = max(bw_rows, key=lambda r: r["bandwidth_util_pct"])
            print(f"    最佳带宽利用率 : {best_bw['bandwidth_util_pct']:.1f}%"
                  f"  （{best_bw['shape']} {best_bw['dtype']}）")

        # v1 / v2 对比
        if rows[0].get("speedup_v1") is not None:
            s1 = [r["speedup_v1"] for r in rows]
            s2 = [r["speedup_v2"] for r in rows]
            print(f"    ├─ v1（整行处理）  平均 : {sum(s1) / len(s1):.2f}x")
            print(f"    └─ v2（分块归约）  平均 : {sum(s2) / len(s2):.2f}x")
            gains = [(r, r["speedup_v2"] / max(r["speedup_v1"], 1e-9)) for r in rows]
            g_best = max(gains, key=lambda t: t[1])
            print(f"       v2 相对 v1 最大提升 : {g_best[1]:.2f}x"
                  f"  （{g_best[0]['shape']} {g_best[0]['dtype']}）")


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
        "rope": [],
        "qkv_split_rope": [],
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
    bench_rope(device, peak_bw, results)
    bench_qkv_rope(device, peak_bw, results)

    print_summary(results)

    print()
    print("=" * 78)
    print("SwiGLU autotune 选出的最优配置")
    print("=" * 78)
    try:
        print(f"  {tk._swiglu_fwd_kernel.best_config}")
    except Exception:
        print("  （未获取到 autotune 配置）")

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

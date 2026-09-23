# -*- coding: utf-8 -*-
"""
基准测试工具 —— 把「快了多少」变成可复现的数字。

【三条纪律】
    1. 先预热：第一次运行包含编译、缓存未命中、显存分配等开销，不能算
    2. 多次取中位数：单次测量不可信（我们实测过同一段代码两次差 7 倍）
    3. 记录波动：中位数和最小值的差距大，说明环境不稳定，数据不可信

【为什么还要算带宽利用率】
    RMSNorm / SwiGLU 这类逐元素或归约操作都是 **memory-bound**（访存受限）。
    对它们来说，「算力用了多少」没意义，「带宽用了多少」才是关键。

    带宽利用率 = 实际搬运字节数 / 耗时 / 理论峰值带宽

    如果只有 20%，说明还有 5 倍空间（通常是访存模式不连续）；
    如果到了 80%+，说明已经接近硬件极限，再优化只能改算法。
"""

import statistics
import time

import torch


# ============================================================
# 一、计时核心
# ============================================================
def bench(fn, warmup: int = 25, rep: int = 100, device: str = "cuda") -> dict:
    """跑 fn 很多次，返回耗时统计（毫秒）。

    参数：
        fn      : 无参函数，执行一次要测的操作
        warmup  : 预热次数
        rep     : 正式测量次数
        device  : "cuda" 或 "cpu"

    返回：
        {"median_ms", "min_ms", "max_ms", "mean_ms", "spread"}
    """
    # ---- 预热 ----
    for _ in range(warmup):
        fn()

    if device == "cuda":
        torch.cuda.synchronize()
        times = []
        for _ in range(rep):
            # 用 CUDA Event 计时，比 time.perf_counter 准
            # （CPU 计时会把「提交任务」的时间算进去，而 GPU 是异步的）
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)

    med = statistics.median(times)
    mn = min(times)
    mx = max(times)
    return {
        "median_ms": med,
        "min_ms": mn,
        "max_ms": mx,
        "mean_ms": statistics.fmean(times),
        # 波动率：越小越可信。> 0.2 说明环境不稳，数据要打问号
        "spread": (mx - mn) / med if med > 0 else float("inf"),
    }


# ============================================================
# 二、带宽相关
# ============================================================
def bytes_for_rmsnorm(shape, dtype_bytes: int = 4) -> int:
    """RMSNorm 搬运的字节数：读 x + 读 weight + 写 y

    注意 weight 只读一次（形状 (N,)），相比 x 和 y 可以忽略，
    但严谨起见还是算进去。
    """
    m = 1
    for s in shape[:-1]:
        m *= s
    n = shape[-1]
    return m * n * dtype_bytes * 2 + n * dtype_bytes


def bytes_for_swiglu(shape, dtype_bytes: int = 4) -> int:
    """SwiGLU 搬运的字节数：读 gate + 读 up + 写 out"""
    numel = 1
    for s in shape:
        numel *= s
    return numel * dtype_bytes * 3


def bandwidth_gbps(nbytes: int, ms: float) -> float:
    """带宽 = 字节数 / 时间，单位 GB/s"""
    if ms <= 0:
        return 0.0
    return nbytes / (ms / 1000.0) / 1e9


# ============================================================
# 三、报告输出
# ============================================================
def print_result(name: str, stats: dict, ref_ms: float = None,
                 nbytes: int = None, peak_bw_gbps: float = None):
    """打印一行基准结果，并可选地算加速比和带宽利用率。"""
    med = stats["median_ms"]
    line = f"  {name:<28} {med:8.3f} ms"

    if ref_ms is not None and med > 0:
        speedup = ref_ms / med
        line += f"   {speedup:5.2f}x"

    if nbytes is not None:
        bw = bandwidth_gbps(nbytes, med)
        line += f"   {bw:7.1f} GB/s"
        if peak_bw_gbps:
            line += f"  ({bw / peak_bw_gbps * 100:4.1f}% 峰值)"

    if stats["spread"] > 0.2:
        line += "   ⚠️ 波动大"

    print(line)


def header(columns: str):
    print()
    print("-" * 78)
    print(columns)
    print("-" * 78)


# ============================================================
# 四、设备信息（用于判断带宽上限）
# ============================================================
def device_summary() -> dict:
    """打印设备信息，返回理论峰值带宽（GB/s，估算值）。"""
    info = {"device": "CPU", "peak_bw_gbps": None}

    if not torch.cuda.is_available():
        print("  设备: CPU（无 GPU，Triton 跑不了）")
        return info

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_gib = props.total_memory / 1024 ** 3

    print(f"  设备        : {name}")
    print(f"  显存        : {total_gib:.1f} GiB")
    print(f"  CUDA 能力   : {props.major}.{props.minor}")

    info["device"] = name

    # 粗略估算峰值带宽（用于判断「离极限还有多远」）
    # 这些是常见卡的标称值，仅作参考
    known = {
        "T4": 320, "A100": 1555, "V100": 900, "L4": 300,
        "A10": 600, "H100": 3350, "4090": 1008, "3090": 936,
    }
    for key, bw in known.items():
        if key.lower() in name.lower():
            info["peak_bw_gbps"] = bw
            print(f"  估算峰值带宽 : {bw} GB/s（{key} 标称值，仅参考）")
            break
    else:
        print("  估算峰值带宽 : 未知型号，带宽利用率将无法计算")
        print("                （海光 DCU 请查官方规格后填入 PEAK_BW）")

    return info


# ============================================================
# 五、自测（CPU 上也能跑）
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("bench_utils 自测")
    print("=" * 60)

    device_summary()

    # 用一个简单的 CPU 操作验证计时逻辑
    import numpy as np

    def work():
        a = np.random.randn(200, 200)
        return a @ a

    stats = bench(work, warmup=3, rep=20, device="cpu")
    print()
    print("  CPU 计时自测（矩阵乘 200x200）:")
    print(f"    中位数 {stats['median_ms']:.3f} ms")
    print(f"    最小值 {stats['min_ms']:.3f} ms")
    print(f"    波动率 {stats['spread']:.3f}")

    # 验证带宽计算
    nb = bytes_for_swiglu((1024, 4096), dtype_bytes=2)
    print()
    print("  SwiGLU 数据量 (1024x4096, fp16):")
    print(f"    {nb / 1e6:.1f} MB（读 gate + 读 up + 写 out）")

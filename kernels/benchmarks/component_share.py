# -*- coding: utf-8 -*-
"""
测算「RMSNorm + SwiGLU 在模型总时间里占多少」—— 决定端到端收益的上限。

【为什么必须先做这个测算】

    如果这两个算子只占模型总时间的 5%，
    那即使把它们优化到无限快，端到端也只能提升 5%。

    先量化，再优化 —— 这是性能优化最基本的一条纪律。

【怎么模拟】

    用真实的 Qwen2.5-7B 配置搭一个前向：
        hidden = 3584, intermediate = 18944, num_layers = 28

    每层做：
        h = rmsnorm(h)                    ← 我们优化了
        gate = h @ W_gate.T               ← 矩阵乘，没动
        up   = h @ W_up.T                 ← 矩阵乘，没动
        h    = swiglu(gate, up)           ← 我们优化了
        h    = h @ W_down.T               ← 矩阵乘，没动

    然后对比「全 PyTorch」和「替换两个算子后」的总耗时。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import bench_utils as bu        # noqa: E402
import triton_kernels as tk     # noqa: E402


# Qwen2.5-7B 的真实配置
CONFIG = {
    "num_layers": 28,
    "hidden": 3584,
    "intermediate": 18944,
}


def build_tensors(num_tokens, hidden, intermediate, dtype, device):
    """分配权重和张量（只分配一次，避免干扰计时）。"""
    return {
        "x": torch.randn(num_tokens, hidden, device=device, dtype=dtype),
        "w_norm": torch.randn(hidden, device=device, dtype=dtype),
        "w_gate": torch.randn(intermediate, hidden, device=device, dtype=dtype),
        "w_up": torch.randn(intermediate, hidden, device=device, dtype=dtype),
        "w_down": torch.randn(hidden, intermediate, device=device, dtype=dtype),
    }


def forward_pytorch(t, num_layers, eps=1e-6):
    """全 PyTorch 实现（baseline）。"""
    h = t["x"]
    for _ in range(num_layers):
        # RMSNorm（手写，和我们的参考实现一致）
        mean_sq = h.float().pow(2).mean(dim=-1, keepdim=True)
        h = (h.float() / torch.sqrt(mean_sq + eps) * t["w_norm"].float()).to(h.dtype)

        gate = h @ t["w_gate"].t()
        up = h @ t["w_up"].t()
        h = F.silu(gate) * up
        h = h @ t["w_down"].t()
    return h


def forward_triton(t, num_layers, eps=1e-6):
    """把 RMSNorm 和 SwiGLU 换成 Triton 实现。"""
    h = t["x"]
    for _ in range(num_layers):
        h = tk.rmsnorm_v2(h, t["w_norm"], eps=eps)

        gate = h @ t["w_gate"].t()
        up = h @ t["w_up"].t()
        h = tk.swiglu(gate, up)
        h = h @ t["w_down"].t()
    return h


def forward_norm_only(t, num_layers, eps=1e-6):
    """只替换 RMSNorm（用来拆解各部分贡献）。"""
    h = t["x"]
    for _ in range(num_layers):
        h = tk.rmsnorm_v2(h, t["w_norm"], eps=eps)

        gate = h @ t["w_gate"].t()
        up = h @ t["w_up"].t()
        h = F.silu(gate) * up
        h = h @ t["w_down"].t()
    return h


def main():
    print("=" * 78)
    print("组件占比测算：RMSNorm + SwiGLU 在模型前向里占多少时间")
    print("=" * 78)

    if not torch.cuda.is_available():
        print("\n需要 GPU 环境（Colab / DCU 集群）")
        return 1

    print()
    print(f"  模拟配置: Qwen2.5-7B")
    print(f"    层数        : {CONFIG['num_layers']}")
    print(f"    hidden      : {CONFIG['hidden']}")
    print(f"    intermediate: {CONFIG['intermediate']}")

    results = {}

    for dtype in (torch.bfloat16, torch.float16):
        for num_tokens in (128, 512, 2048):
            t = build_tensors(num_tokens, CONFIG["hidden"], CONFIG["intermediate"],
                              dtype, "cuda")
            n_layers = CONFIG["num_layers"]

            t_base = bu.bench(lambda: forward_pytorch(t, n_layers),
                              warmup=3, rep=10)
            t_norm = bu.bench(lambda: forward_norm_only(t, n_layers),
                              warmup=3, rep=10)
            t_full = bu.bench(lambda: forward_triton(t, n_layers),
                              warmup=3, rep=10)

            base_ms = t_base["median_ms"]
            full_ms = t_full["median_ms"]
            norm_ms = t_norm["median_ms"]

            speedup = base_ms / full_ms
            # 只换 RMSNorm 带来的时间减少，就是 RMSNorm 的「占比」
            norm_share = (base_ms - norm_ms) / base_ms
            # 全部替换带来的时间减少
            total_share = (base_ms - full_ms) / base_ms

            key = f"{str(dtype).split('.')[-1]}_{num_tokens}"
            results[key] = {
                "dtype": str(dtype).split(".")[-1],
                "num_tokens": num_tokens,
                "baseline_ms": base_ms,
                "triton_ms": full_ms,
                "speedup": speedup,
                "norm_share_pct": norm_share * 100,
                "total_share_pct": total_share * 100,
            }

            print()
            print(f"  [{str(dtype).split('.')[-1]}, {num_tokens} tokens]")
            print(f"    全 PyTorch        : {base_ms:8.2f} ms")
            print(f"    只换 RMSNorm      : {norm_ms:8.2f} ms"
                  f"   （省了 {norm_share*100:5.2f}%）")
            print(f"    全换（+SwiGLU）   : {full_ms:8.2f} ms"
                  f"   （省了 {total_share*100:5.2f}%，加速 {speedup:.3f}x）")

    # ---- 结论 ----
    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)

    shares = [r["total_share_pct"] for r in results.values()]
    speedups = [r["speedup"] for r in results.values()]
    print(f"  两个算子的时间占比 : {min(shares):.1f}% ~ {max(shares):.1f}%")
    print(f"  端到端加速比范围   : {min(speedups):.3f}x ~ {max(speedups):.3f}x")
    print()
    print("  💡 这个占比就是端到端收益的**上限**。")
    print("     如果只有个位数百分比，说明矩阵乘（我们没动的那部分）才是大头。")
    print("     想继续提升，要么优化矩阵乘，要么去动 attention（KV cache / 调度）。")

    # ---- 落盘 ----
    import json
    import time
    out_dir = Path(__file__).resolve().parents[2] / "results" / "t4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"component_share_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "_meta": {"config": CONFIG, "device": torch.cuda.get_device_name(0)},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  数据已保存: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
PyTorch 参考实现 —— 正确性验证的「标准答案」。

设计原则：
    1. 只求正确、清晰，不求速度（慢没关系，Triton 版要比它快）
    2. 用最直白的写法，让公式一眼能对上
    3. 每个函数都要能单独跑，不依赖其他模块

所有 Triton kernel 的输出，都必须和这里的实现对齐。
"""

import torch
import torch.nn.functional as F


# ============================================================
# 一、RMSNorm
# ============================================================
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm:  y = x / sqrt(mean(x^2) + eps) * weight

    维度契约：
        x      : (..., N)      N = 特征维度（最后一维）
        weight : (N,)
        输出    : (..., N)     形状与 x 相同

    与 LayerNorm 的区别：
        LayerNorm: (x - mean) / sqrt(var + eps)      ← 减均值、除标准差
        RMSNorm:    x         / sqrt(mean(x^2) + eps) ← 不减均值、除均方根

    RMSNorm 省掉了减均值，计算更少、效果接近，Qwen / LLaMA 都用它。

    注意 mean 是沿「最后一维」算的，keepdim=True 保证能正确广播回去。
    """
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)   # (..., 1)
    rrms = torch.rsqrt(mean_square + eps)               # 1 / sqrt(...)
    return x * rrms * weight


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    """融合「残差相加 + RMSNorm」：  h = x + residual,  y = rmsnorm(h) * weight

    为什么要把这两个操作融合？
        单独做的话，h 这个中间张量要写回 HBM、再从 HBM 读出来 —— 两次往返。
        融合后 h 只存在寄存器 / 共享内存里，省掉这两次往返。
        这就是「算子融合」省性能的本质。

    但注意：Transformer 里 h 后面还要用（残差连接），所以 h 也得输出。

    返回: (y, h)
        y : (..., N)  归一化后的结果
        h : (..., N)  相加后的结果（后面层还要用）
    """
    h = x + residual
    y = rmsnorm(h, weight, eps)
    return y, h


# ============================================================
# 二、SwiGLU
# ============================================================
def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU 激活:  silu(gate) * up

    维度契约：
        gate : (..., N)   门控分支
        up   : (..., N)   上投影分支
        输出  : (..., N)

    silu(x) = x * sigmoid(x)，也叫 swish。
    所以 SwiGLU = (gate * sigmoid(gate)) * up

    这是一个**逐元素**操作 —— 没有矩阵乘，纯粹是元素级计算。
    正因为简单，才特别适合融合：把 sigmoid、乘法合并到一个 kernel 里，
    中间结果完全不用落 HBM。
    """
    return F.silu(gate) * up


def swiglu_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """完整的 SwiGLU MLP 前向（Transformer 里的 FFN 层）

    流程：
        gate = x @ w_gate.T          (..., N) -> (..., H)
        up   = x @ w_up.T            (..., N) -> (..., H)
        h    = silu(gate) * up       (..., H)
        out  = h @ w_down.T          (..., H) -> (..., N)

    其中 H 通常是 N 的 2~3 倍（中间维度扩张）。
    这里给的是**未融合**的标准写法，用来当基准。
    """
    gate = F.linear(x, w_gate)
    up = F.linear(x, w_up)
    return F.linear(swiglu(gate, up), w_down)


# ============================================================
# 三、自测（不依赖 triton，纯 torch 就能跑）
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("参考实现自测")
    print("=" * 60)

    # ---- RMSNorm ----
    x = torch.randn(4, 8)
    w = torch.ones(8)
    y = rmsnorm(x, w, eps=0.0)
    # 验证：归一化后每行的均方应该等于 1
    row_ms = y.pow(2).mean(dim=-1)
    print(f"\n[RMSNorm] 输出每行均方（应该都是 1.0）：")
    print(f"  {row_ms.tolist()}")
    print(f"  全部接近 1？{torch.allclose(row_ms, torch.ones_like(row_ms), atol=1e-5)}")

    # 手算验证（用 Q2 里那个例子）
    x1 = torch.tensor([[3.0, 4.0]])
    y1 = rmsnorm(x1, torch.ones(2), eps=0.0)
    print(f"\n  手算验证 x=[3,4]: 期望 [0.84852814, 1.13137085]")
    print(f"  实际得到        : {[round(v, 8) for v in y1[0].tolist()]}")

    # ---- 融合 Add + RMSNorm ----
    x2 = torch.randn(4, 8)
    r2 = torch.randn(4, 8)
    y2, h2 = fused_add_rmsnorm(x2, r2, torch.ones(8), eps=1e-6)
    print(f"\n[融合 Add+RMSNorm]")
    print(f"  h = x + r 正确？{torch.allclose(h2, x2 + r2)}")
    print(f"  y = rmsnorm(h) 正确？{torch.allclose(y2, rmsnorm(x2 + r2, torch.ones(8), 1e-6))}")

    # ---- SwiGLU ----
    gate = torch.randn(4, 8)
    up = torch.randn(4, 8)
    out = swiglu(gate, up)
    print(f"\n[SwiGLU]")
    print(f"  输出形状 {tuple(out.shape)}（应和输入相同）")
    # silu(x) = x * sigmoid(x)
    manual = gate * torch.sigmoid(gate) * up
    print(f"  与手写 silu 一致？{torch.allclose(out, manual)}")

    # ---- SwiGLU MLP ----
    x3 = torch.randn(2, 16)
    wg = torch.randn(32, 16)
    wu = torch.randn(32, 16)
    wd = torch.randn(16, 32)
    o = swiglu_mlp(x3, wg, wu, wd)
    print(f"\n[SwiGLU MLP]")
    print(f"  输入 {tuple(x3.shape)} → 输出 {tuple(o.shape)}（应回到输入形状）")

    print("\n" + "=" * 60)
    print("参考实现自测完成")
    print("=" * 60)

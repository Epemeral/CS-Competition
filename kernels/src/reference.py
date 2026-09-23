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


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """vLLM 的 SiluAndMul 风格：输入是「gate 和 up 拼在一起」的单张量。

    维度契约：
        x    : (..., 2*H)     前半是 gate，后半是 up
        输出  : (..., H)

    为什么需要这个接口：
        vLLM 的 MLP 先用一次矩阵乘算出 (..., 2H)，再调 SiluAndMul 切开做激活。
        我们的 swiglu() 是「两个张量」接口，接不进去 —— 所以要写一个 vLLM 风格的适配版。

    ⚠️ 这是「适配层」，不是「性能优化」：真正的收益来自把它换成 Triton kernel
       （见 triton_kernels.silu_and_mul_triton）。
    """
    H = x.shape[-1] // 2
    return swiglu(x[..., :H], x[..., H:])


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
# 三、RoPE（旋转位置编码）
# ============================================================
def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device="cpu", dtype=torch.float32):
    """构造 RoPE 的 cos / sin 表。

    数学：
        inv_freq[i] = 1 / base^(2i/d)        i = 0 .. d/2-1
        angle(t, i) = t * inv_freq[i]
        cos[t, i]   = cos(angle(t, i))
        sin[t, i]   = sin(angle(t, i))

    返回形状 (T, D) 的表，**前后半相同** —— 这是为了配合 rotate_half 的写法
    （HuggingFace 也是这么存的）。

    base 越大，不同位置的频率差异越小（能区分的距离越远）。
    默认 10000 是绝大多数模型用的值。
    """
    half = head_dim // 2
    i = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (2.0 * i / head_dim))       # (half,)

    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = t[:, None] * inv_freq[None, :]                # (T, half)

    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    # 前后半拼成 (T, D)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos.to(dtype), sin.to(dtype)


def rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE：把位置信息「旋转」进 q / k 向量。

    维度契约：
        q   : (B, H, T, D)      D = head_dim
        cos : (T, D)
        sin : (T, D)
        输出 : (B, H, T, D)

    公式（HuggingFace 的 rotate_half 形式）：
        out = q * cos + rotate_half(q) * sin

        rotate_half(x) = cat(-x[D/2:], x[:D/2])

    展开看就是「对每一对维度做旋转」：
        out[:D/2] = q[:D/2] * cos - q[D/2:] * sin
        out[D/2:] = q[D/2:] * cos + q[:D/2] * sin

    这就是二维旋转矩阵 [cos -sin; sin cos] 作用在 (x1, x2) 上。
    它的妙处：**两个位置的相对关系只取决于它们的距离差**，与绝对位置无关。
    """
    def rotate_half(x):
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    # cos/sin 从 (T, D) 广播到 (1, 1, T, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin


def qkv_split_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
):
    """把一次投影出来的 QKV 切开，并对 Q / K 应用 RoPE。

    这是推理里的真实流程：
        qkv = x @ W_qkv^T          (B, T, 3*H*D)
        q, k, v = split(qkv)       ← 切分
        q, k = rope(q), rope(k)    ← 只有 q/k 需要位置信息，v 不需要

    为什么要融合成一个算子：
        不融合的话，「切分」+「rope(q)」+「rope(k)」是 3 次 kernel launch，
        中间张量还要落 HBM 再读出来。融合后只读写一次。

    维度契约：
        qkv : (B, T, 3*H*D)
        cos/sin : (T, D)
        输出 : (q, k, v)，各 (B, T, H*D)
    """
    B, T, three_hd = qkv.shape
    HD = num_heads * head_dim
    assert three_hd == 3 * HD, f"qkv 最后一维应为 3*H*D={3*HD}，实际 {three_hd}"

    q, k, v = qkv.split([HD, HD, HD], dim=-1)

    def rot(x):
        # (B, T, H*D) -> (B, H, T, D) -> rope -> 回到 (B, T, H*D)
        xr = x.view(B, T, num_heads, head_dim).transpose(1, 2)
        out = rope(xr, cos, sin)
        return out.transpose(1, 2).reshape(B, T, HD)

    return rot(q), rot(k), v


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

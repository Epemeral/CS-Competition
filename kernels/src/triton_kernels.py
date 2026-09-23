# -*- coding: utf-8 -*-
"""
Triton kernel 实现 —— 我们要优化的对象。

每个 kernel 都必须：
    1. 与 src/reference.py 的输出数值对齐（tests/ 验证）
    2. 相对 PyTorch 原生有加速（benchmarks/ 测量）

【为什么用 Triton 而不是直接写 HIP / CUDA】
    · 开发速度快 5~10 倍（Python DSL，不用管线程索引）
    · 性能通常能到手写的 80~95%
    · 先把算法逻辑跑通、把收益测出来，再决定要不要下沉到 HIP
    · 对比赛而言：「Triton 内核 + 完整消融数据」比「半成品 HIP 内核」得分高

【Triton 编程模型速览】
    · 你写的是「一个 program 怎么处理一块数据」，不是「一个线程做什么」
    · tl.program_id(0)  → 当前是第几个 program（相当于 blockIdx）
    · tl.arange(0, BLOCK) → 这一块里的偏移量（相当于一个向量）
    · 所有操作都是「张量级」的，不用写循环
    · BLOCK 必须是编译期常量（tl.constexpr）
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 一、RMSNorm
# ============================================================
@triton.jit
def _rmsnorm_fwd_kernel(
    X, W, Y,
    stride_x_row, stride_y_row,
    N,                                  # 特征维度（真实值）
    eps,
    BLOCK_N: tl.constexpr,              # 编译期常量，>= N 的 2 的幂
):
    """每个 program 处理一行。

    数据流：
        加载 x 和 w  →  算均方  →  算 rrms  →  乘回去  →  存储

    融合点：整个 RMSNorm 只读一次 x、写一次 y，中间结果全在寄存器里。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N                     # 边界：N 可能不是 2 的幂

    # 加载（mask 之外的位置填 0，不影响后面的求和）
    x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # 均方 —— ⚠️ 分母用真实的 N，不是 BLOCK_N！
    # 如果写成 / BLOCK_N，当 N 不是 2 的幂时结果就错了。
    # 这是 Triton 里最常见的坑之一。
    mean_square = tl.sum(x * x, axis=0) / N

    # rrms = 1 / sqrt(mean_square + eps)
    rrms = 1.0 / tl.sqrt(mean_square + eps)

    y = x * rrms * w

    # 转回原 dtype 再存（输入可能是 fp16/bf16，内部用 fp32 计算）
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton 版 RMSNorm 的 Python 封装。

    负责：整理形状、分配输出、计算 grid、启动 kernel。
    """
    assert x.is_cuda, "Triton kernel 需要 CUDA/ROCm 设备"
    x = x.contiguous()
    N = x.shape[-1]

    x2d = x.view(-1, N)                 # (M, N)，M = 行数
    M = x2d.shape[0]
    y = torch.empty_like(x2d)

    BLOCK_N = triton.next_power_of_2(N)  # 取 >= N 的最小 2 的幂
    grid = (M,)                          # 一行一个 program

    _rmsnorm_fwd_kernel[grid](
        x2d, weight, y,
        x2d.stride(0), y.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return y.view_as(x)


# ============================================================
# 二、融合 Add + RMSNorm
# ============================================================
@triton.jit
def _add_rmsnorm_fwd_kernel(
    X, R, W, Y, H,
    stride_x_row, stride_y_row, stride_h_row,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    """融合「残差相加 + RMSNorm」。

    公式：h = x + r;  y = h / sqrt(mean(h^2) + eps) * w

    为什么值得融合：
        不融合的话，h 要「写回 HBM → 再读出来」两次往返。
        融合后 h 只在寄存器里走一圈，**省掉 2 次 HBM 读写**。
        对于 memory-bound 的操作，这直接换成速度。

    为什么要输出 H：
        Transformer 里残差连接后面还要用 h，所以不能只输出 y。
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    h = x + r                                   # 残差相加
    mean_square = tl.sum(h * h, axis=0) / N
    rrms = 1.0 / tl.sqrt(mean_square + eps)
    y = h * rrms * w

    tl.store(H + row * stride_h_row + cols, h.to(H.dtype.element_ty), mask=mask)
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    """返回 (y, h)"""
    assert x.is_cuda
    x = x.contiguous()
    residual = residual.contiguous()
    N = x.shape[-1]

    x2d = x.view(-1, N)
    r2d = residual.view(-1, N)
    M = x2d.shape[0]

    y = torch.empty_like(x2d)
    h = torch.empty_like(x2d)

    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)

    _add_rmsnorm_fwd_kernel[grid](
        x2d, r2d, weight, y, h,
        x2d.stride(0), y.stride(0), h.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return y.view_as(x), h.view_as(x)


# ---- v2：分块归约版（和 RMSNorm v2 同样的思路）----
# 实测数据说明问题：
#   (512, 18944) fp32  分开做 1.645ms  融合 v1 5.282ms  → 0.31x（比不融合还慢！）
# 原因和 RMSNorm v1 一样：BLOCK_N = next_power_of_2(18944) = 32768，寄存器爆掉。
#
# v2 用分块归约：BLOCK_N 固定上限，循环两遍。
# 代价是 h = x + r 要算两次（很便宜，一次加法），换来寄存器压力恒定。

@triton.jit
def _add_rmsnorm_loop_kernel(
    X, R, W, Y, H,
    stride_x_row, stride_y_row, stride_h_row,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    # ---- 第一遍：算 h 并累加 sum(h²) ----
    sum_sq = 0.0
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        h = x + r
        sum_sq += tl.sum(h * h, axis=0)

    rrms = 1.0 / tl.sqrt(sum_sq / N + eps)

    # ---- 第二遍：重算 h，写出 h 和 y ----
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(R + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

        h = x + r
        y = h * rrms * w

        tl.store(H + row * stride_h_row + cols, h.to(H.dtype.element_ty), mask=mask)
        tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)


def fused_add_rmsnorm_v2(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    block_cap: int = 4096,
):
    """融合 Add + RMSNorm 的分块归约版。返回 (y, h)"""
    assert x.is_cuda
    x = x.contiguous()
    residual = residual.contiguous()
    N = x.shape[-1]

    x2d = x.view(-1, N)
    r2d = residual.view(-1, N)
    M = x2d.shape[0]

    y = torch.empty_like(x2d)
    h = torch.empty_like(x2d)

    BLOCK_N = min(triton.next_power_of_2(N), block_cap)
    grid = (M,)

    _add_rmsnorm_loop_kernel[grid](
        x2d, r2d, weight, y, h,
        x2d.stride(0), y.stride(0), h.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return y.view_as(x), h.view_as(x)


# ============================================================
# 三、SwiGLU
# ============================================================
@triton.jit
def _swiglu_fwd_kernel(
    GATE, UP, OUT,
    n_elements,
    BLOCK: tl.constexpr,
):
    """SwiGLU: silu(gate) * up

    这是纯逐元素操作 —— 最适合入门的融合 kernel。

    融合点：
        silu(g) = g * sigmoid(g)  需要算 sigmoid 再乘
        然后还要乘 up
        不融合的话 PyTorch 要启动 3~4 个 kernel，每个都读写一遍 HBM。
        融合后只读 2 次、写 1 次。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements            # 处理尾部不足一个 BLOCK 的情况

    g = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)

    # silu(g) = g * sigmoid(g)
    silu_g = g * tl.sigmoid(g)
    out = silu_g * u

    tl.store(OUT + offs, out.to(OUT.dtype.element_ty), mask=mask)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    assert gate.is_cuda
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)

    n = gate.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)     # cdiv = 向上取整除法

    _swiglu_fwd_kernel[grid](gate, up, out, n, BLOCK=BLOCK)
    return out


# ============================================================
# 四、RMSNorm v2：分块归约版（解决大 N 的寄存器爆炸）
# ============================================================
# 【为什么需要 v2】
#   v1 用 BLOCK_N = next_power_of_2(N) 一次性处理整行。
#   当 N = 18944 时，BLOCK_N = 32768 —— 一个 program 要同时持有 32768 个 fp32，
#   寄存器严重不足，编译器只能 spill 到 local memory（走 HBM），反而更慢。
#   实测：(128, 18944) fp32 下 v1 只有 0.54x，比 PyTorch 还慢。
#
# 【v2 怎么做】
#   把 BLOCK_N 固定成一个上限（如 4096），用循环分块处理：
#     第一遍：循环累加 sum(x²)
#     第二遍：循环归一化并写出
#
#   好处：寄存器压力恒定，不随 N 增长。
#   代价：x 要读两次 —— 但第二次大概率命中 L2 缓存，比寄存器 spill 便宜得多。
#
#   这是 GPU kernel 的经典权衡：**用一点额外的访存，换掉寄存器压力**。

@triton.jit
def _rmsnorm_loop_kernel(
    X, W, Y,
    stride_x_row, stride_y_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,          # 固定上限，不随 N 增长
):
    row = tl.program_id(0)

    # ---- 第一遍：循环累加 sum(x²) ----
    sum_sq = 0.0
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    # 注意分母用真实的 N
    rrms = 1.0 / tl.sqrt(sum_sq / N + eps)

    # ---- 第二遍：归一化并写出 ----
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm_v2(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
               block_cap: int = 4096) -> torch.Tensor:
    """分块归约版 RMSNorm。

    参数：
        block_cap : BLOCK_N 的上限。N 小于它时行为和 v1 一样；
                    N 大于它时循环分块，避免寄存器爆炸。
    """
    assert x.is_cuda, "Triton kernel 需要 CUDA/ROCm 设备"
    x = x.contiguous()
    N = x.shape[-1]

    x2d = x.view(-1, N)
    M = x2d.shape[0]
    y = torch.empty_like(x2d)

    # 关键：BLOCK_N 取 min(N 的 2 次幂, block_cap)，而不是直接用 next_power_of_2(N)
    BLOCK_N = min(triton.next_power_of_2(N), block_cap)
    grid = (M,)

    _rmsnorm_loop_kernel[grid](
        x2d, weight, y,
        x2d.stride(0), y.stride(0),
        N, eps,
        BLOCK_N=BLOCK_N,
    )
    return y.view_as(x)


# ============================================================
# 五、自测（需要 GPU）
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reference as ref

    if not torch.cuda.is_available():
        print("当前环境没有 GPU，Triton kernel 无法运行。")
        print("请在 Colab（开 T4 GPU）或 DCU 集群上运行。")
        sys.exit(0)

    torch.manual_seed(0)
    device = "cuda"

    print("=" * 60)
    print("Triton kernel 自测")
    print("=" * 60)
    print(f"设备: {torch.cuda.get_device_name(0)}")

    # ---- RMSNorm（v1 和 v2 都要和参考实现对齐）----
    for shape in [(4, 8), (32, 128), (8, 100), (1, 4096), (4, 18944)]:
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        w = torch.randn(shape[-1], device=device, dtype=torch.float32)
        y_ref = ref.rmsnorm(x, w, eps=1e-6)
        y_v1 = rmsnorm(x, w, eps=1e-6)
        y_v2 = rmsnorm_v2(x, w, eps=1e-6)
        ok1 = torch.allclose(y_ref, y_v1, atol=1e-4, rtol=1e-4)
        ok2 = torch.allclose(y_ref, y_v2, atol=1e-4, rtol=1e-4)
        print(f"  rmsnorm {str(shape):>12}   v1 {'✅' if ok1 else '❌'}"
              f"   v2 {'✅' if ok2 else '❌'}")

    # ---- 融合 Add + RMSNorm ----
    for shape in [(4, 8), (32, 128), (8, 100)]:
        x = torch.randn(*shape, device=device)
        r = torch.randn(*shape, device=device)
        w = torch.randn(shape[-1], device=device)
        y_ref, h_ref = ref.fused_add_rmsnorm(x, r, w, eps=1e-6)
        y_tri, h_tri = fused_add_rmsnorm(x, r, w, eps=1e-6)
        ok = (torch.allclose(y_ref, y_tri, atol=1e-4, rtol=1e-4)
              and torch.allclose(h_ref, h_tri, atol=1e-4, rtol=1e-4))
        print(f"  add_rmsnorm {str(shape):>12}  →  {'✅' if ok else '❌ 不一致'}")

    # ---- SwiGLU ----
    for shape in [(4, 8), (32, 128), (8, 1000), (1, 4097)]:
        g = torch.randn(*shape, device=device)
        u = torch.randn(*shape, device=device)
        out_ref = ref.swiglu(g, u)
        out_tri = swiglu(g, u)
        ok = torch.allclose(out_ref, out_tri, atol=1e-4, rtol=1e-4)
        print(f"  swiglu {str(shape):>12}  →  {'✅' if ok else '❌ 不一致'}")

    print("\n自测完成。更严格的验证请跑 tests/test_correctness.py")

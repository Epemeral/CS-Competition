# -*- coding: utf-8 -*-
"""
把 kernels 的 Triton 实现接进 vLLM —— 端到端集成。

【为什么这一步最重要】

    前面所有 kernel 都是「孤立测」的：给一个张量、测它多快。
    但比赛的评分口径是**端到端吞吐（tokens/s）**。

    一个 kernel 单独快 2 倍，不代表整体快 ——
    它可能只占总时间的 5%，那整体提升就微乎其微。

    只有接进真实推理流程，才能回答那个关键问题：
        **「我的优化到底让模型快了多少？」**

【怎么接：monkey-patch】

    最轻量的方式 —— 不改 vLLM 源码，运行时替换掉它的算子实现。

    好处：可逆、易对比、不需要重新编译 vLLM。

    ⚠️ 注意：vLLM 默认会用 torch.compile + CUDA Graph。
    我们的 Triton kernel 在里面可能编译失败或被绕过。
    所以端到端测试要先开 `enforce_eager=True` 验证收益，
    确认有效后再考虑编译兼容性。

【patch 哪两处】

    1. Qwen2RMSNorm.forward  —— 每层 2 次，28 层 = 56 次调用
    2. SiluAndMul.forward    —— 每层 1 次，28 层 = 28 次调用

    这两个都是 memory-bound 的小算子，正是融合收益最大的地方。
    （RoPE / QKV 融合的接入更复杂，等这两个验证有效再说。）
"""

import sys
from pathlib import Path

import torch

# 让脚本能直接 import src/ 下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import triton_kernels as tk  # noqa: E402


# 保存原始实现，用于恢复
_ORIG = {
    "qwen2_rmsnorm": None,
    "silu_and_mul": None,
}


# ============================================================
# 接入
# ============================================================
def patch_qwen2(verbose: bool = True):
    """把 Triton kernel 接进 vLLM 的 Qwen2 实现。

    返回 dict，记录 patch 了哪些地方（便于写进报告）。
    """
    import vllm.model_executor.models.qwen2 as qwen2_mod
    from vllm.model_executor.layers.activation import SiluAndMul

    patched = {}

    # ---------- 1. RMSNorm ----------
    if _ORIG["qwen2_rmsnorm"] is None:
        _ORIG["qwen2_rmsnorm"] = qwen2_mod.Qwen2RMSNorm.forward

    def triton_rmsnorm_forward(self, x, residual=None):
        """替换 vLLM 的 RMSNorm。

        vLLM 有两种调用形态：
            forward(x)               → 只做归一化
            forward(x, residual)     → 先做残差相加，再归一化（融合）
        两种我们都有对应实现。
        """
        if residual is None:
            return tk.rmsnorm_v2(x, self.weight, eps=self.variance_epsilon)
        # 融合路径：正好对应我们的 fused_add_rmsnorm_v2，返回 (y, h)
        return tk.fused_add_rmsnorm_v2(
            x, residual, self.weight, eps=self.variance_epsilon
        )

    qwen2_mod.Qwen2RMSNorm.forward = triton_rmsnorm_forward
    patched["Qwen2RMSNorm.forward"] = "triton_kernels.rmsnorm_v2 / fused_add_rmsnorm_v2"

    # ---------- 2. SiLU + Mul ----------
    if _ORIG["silu_and_mul"] is None:
        _ORIG["silu_and_mul"] = SiluAndMul.forward

    def triton_silu_and_mul_forward(self, x):
        """替换 vLLM 的 SiluAndMul。输入 (..., 2H) → 输出 (..., H)"""
        return tk.silu_and_mul_triton(x)

    SiluAndMul.forward = triton_silu_and_mul_forward
    patched["SiluAndMul.forward"] = "triton_kernels.silu_and_mul_triton"

    if verbose:
        print("=" * 66)
        print("已接入 vLLM（monkey-patch）")
        print("=" * 66)
        for k, v in patched.items():
            print(f"  {k:<26} -> {v}")
        print()

    return patched


# ============================================================
# 恢复
# ============================================================
def unpatch_qwen2(verbose: bool = True):
    """恢复 vLLM 的原始实现。"""
    import vllm.model_executor.models.qwen2 as qwen2_mod
    from vllm.model_executor.layers.activation import SiluAndMul

    if _ORIG["qwen2_rmsnorm"] is not None:
        qwen2_mod.Qwen2RMSNorm.forward = _ORIG["qwen2_rmsnorm"]
        _ORIG["qwen2_rmsnorm"] = None

    if _ORIG["silu_and_mul"] is not None:
        SiluAndMul.forward = _ORIG["silu_and_mul"]
        _ORIG["silu_and_mul"] = None

    if verbose:
        print("已恢复 vLLM 原始实现")


def is_patched() -> bool:
    return any(v is not None for v in _ORIG.values())


# ============================================================
# 正确性自检：patch 前后输出必须一致
# ============================================================
def verify_patch_equivalence(device="cuda", verbose: bool = True):
    """在接进 vLLM 之前，先验证「替换后的算子和原版输出一致」。

    这一步很重要 —— 如果算子本身就算错了，
    端到端跑出来的吞吐数字毫无意义（而且比赛会直接取消资格）。
    """
    if verbose:
        print("=" * 66)
        print("算子等价性自检（Triton 版 vs PyTorch 版）")
        print("=" * 66)

    ok_all = True

    # ---- RMSNorm ----
    x = torch.randn(128, 3584, device=device, dtype=torch.bfloat16)
    w = torch.randn(3584, device=device, dtype=torch.bfloat16)
    y_tri = tk.rmsnorm_v2(x, w, eps=1e-6)
    mean_sq = x.float().pow(2).mean(dim=-1, keepdim=True)
    y_ref = (x.float() / torch.sqrt(mean_sq + 1e-6) * w.float()).to(x.dtype)
    ok = torch.allclose(y_tri, y_ref, atol=1e-1, rtol=1e-1)
    ok_all &= ok
    if verbose:
        print(f"  rmsnorm_v2        最大误差 {(y_tri.float()-y_ref.float()).abs().max().item():.3e}  "
              f"{'OK' if ok else 'FAIL'}")

    # ---- 融合 Add + RMSNorm ----
    r = torch.randn(128, 3584, device=device, dtype=torch.bfloat16)
    y_tri, h_tri = tk.fused_add_rmsnorm_v2(x, r, w, eps=1e-6)
    h_ref = (x.float() + r.float())
    mean_sq = h_ref.pow(2).mean(dim=-1, keepdim=True)
    y_ref = (h_ref / torch.sqrt(mean_sq + 1e-6) * w.float()).to(x.dtype)
    ok1 = torch.allclose(y_tri, y_ref, atol=1e-1, rtol=1e-1)
    ok2 = torch.allclose(h_tri.float(), h_ref, atol=1e-1, rtol=1e-1)
    ok_all &= (ok1 and ok2)
    if verbose:
        print(f"  fused_add_rmsnorm y 误差 {(y_tri.float()-y_ref.float()).abs().max().item():.3e}  "
              f"{'OK' if ok1 else 'FAIL'}")
        print(f"  fused_add_rmsnorm h 误差 {(h_tri.float()-h_ref).abs().max().item():.3e}  "
              f"{'OK' if ok2 else 'FAIL'}")

    # ---- SiluAndMul ----
    # vLLM 的 intermediate_size，用 18944 的两倍作为输入宽度
    xx = torch.randn(128, 2 * 18944, device=device, dtype=torch.bfloat16)
    out_tri = tk.silu_and_mul_triton(xx)
    gate, up = xx[..., :18944], xx[..., 18944:]
    out_ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(xx.dtype)
    ok = torch.allclose(out_tri, out_ref, atol=1e-1, rtol=1e-1)
    ok_all &= ok
    if verbose:
        print(f"  silu_and_mul      最大误差 {(out_tri.float()-out_ref.float()).abs().max().item():.3e}  "
              f"{'OK' if ok else 'FAIL'}")

    if verbose:
        print()
        print(f"结论：{'全部一致，可以接进 vLLM' if ok_all else '有算子不一致，先修再接入！'}")
        print()

    return ok_all


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 GPU 环境（Colab / DCU 集群）")
        sys.exit(1)

    ok = verify_patch_equivalence()
    sys.exit(0 if ok else 1)

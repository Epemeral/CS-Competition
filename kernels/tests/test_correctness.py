# -*- coding: utf-8 -*-
"""
正确性验证 —— 每个 kernel 的必过关卡。

【为什么这一关最重要】
    比赛红线：**正确性不通过 → 直接取消性能项资格**。
    一个算错的 kernel 跑得再快也毫无价值。

【三条验证纪律】
    1. 必须和参考实现数值对齐（torch.allclose）
    2. 必须测「非 2 的幂」的形状 —— Triton 的 BLOCK 边界最容易出错
    3. 必须测极端情况：N=1、M=1、超大 N、不同 dtype

【容差怎么定】
    不同 dtype 精度不同，容差要相应放宽：
        fp32 : atol=1e-4
        fp16 : atol=1e-2
        bf16 : atol=1e-1   （bf16 只有 8 位尾数，精度本来就低）
"""

import sys
from pathlib import Path

import torch

# 让脚本能直接跑（不用装包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import reference as ref          # noqa: E402
import triton_kernels as tk      # noqa: E402


# ============================================================
# 测试配置
# ============================================================

# 形状列表：覆盖 最小 / 常规 / 非 2 的幂 / 真实模型尺寸
SHAPES = [
    (1, 1),          # 极端最小
    (4, 8),          # 规整的小尺寸
    (32, 128),       # 规整的中尺寸
    (8, 100),        # 非 2 的幂 ← Triton 边界
    (16, 1000),      # 非 2 的幂
    (2, 4096),       # 大 N（2 的幂）
    (8, 3584),       # ⭐ Qwen2.5-7B 的真实 hidden_size
    (4, 18944),      # ⭐ Qwen2.5-7B 的真实 intermediate_size
]

# dtype 与对应容差
DTYPES = [
    (torch.float32, 1e-4, 1e-4),
    (torch.float16, 1e-2, 1e-2),
    (torch.bfloat16, 1e-1, 1e-1),
]


# ============================================================
# 测试框架
# ============================================================
class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def check(self, label: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"  ✅ {label}")
        else:
            self.failed += 1
            self.failures.append(f"{label}  {detail}")
            print(f"  ❌ {label}   {detail}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        print()
        print("=" * 70)
        if self.failed == 0:
            print(f"全部通过：{self.passed}/{total} ✅")
        else:
            print(f"通过 {self.passed}/{total}，失败 {self.failed} ❌")
            print()
            print("失败列表：")
            for f in self.failures:
                print(f"  · {f}")
        print("=" * 70)
        return self.failed == 0


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


# ============================================================
# 一、RMSNorm
# ============================================================
def test_rmsnorm(rep: Report, device: str):
    print("\n【RMSNorm】v1=整行处理 / v2=分块归约")
    for shape in SHAPES:
        for dtype, atol, rtol in DTYPES:
            x = torch.randn(*shape, device=device, dtype=dtype)
            w = torch.randn(shape[-1], device=device, dtype=dtype)

            y_ref = ref.rmsnorm(x, w, eps=1e-6)
            y_v1 = tk.rmsnorm(x, w, eps=1e-6)
            y_v2 = tk.rmsnorm_v2(x, w, eps=1e-6)

            ok1 = torch.allclose(y_ref, y_v1, atol=atol, rtol=rtol)
            ok2 = torch.allclose(y_ref, y_v2, atol=atol, rtol=rtol)

            label = f"{str(shape):>14}  {str(dtype).split('.')[-1]:<16}"
            detail = ""
            if not ok1:
                detail += f"v1 误差 {max_abs_diff(y_ref, y_v1):.2e} "
            if not ok2:
                detail += f"v2 误差 {max_abs_diff(y_ref, y_v2):.2e}"
            rep.check(label, ok1 and ok2, detail)


# ============================================================
# 二、融合 Add + RMSNorm
# ============================================================
def test_fused_add_rmsnorm(rep: Report, device: str):
    print("\n【融合 Add + RMSNorm】v1=整行处理 / v2=分块归约")
    for shape in SHAPES:
        for dtype, atol, rtol in DTYPES:
            x = torch.randn(*shape, device=device, dtype=dtype)
            r = torch.randn(*shape, device=device, dtype=dtype)
            w = torch.randn(shape[-1], device=device, dtype=dtype)

            y_ref, h_ref = ref.fused_add_rmsnorm(x, r, w, eps=1e-6)
            y_v1, h_v1 = tk.fused_add_rmsnorm(x, r, w, eps=1e-6)
            y_v2, h_v2 = tk.fused_add_rmsnorm_v2(x, r, w, eps=1e-6)

            ok1 = (torch.allclose(y_ref, y_v1, atol=atol, rtol=rtol)
                   and torch.allclose(h_ref, h_v1, atol=atol, rtol=rtol))
            ok2 = (torch.allclose(y_ref, y_v2, atol=atol, rtol=rtol)
                   and torch.allclose(h_ref, h_v2, atol=atol, rtol=rtol))

            label = f"{str(shape):>14}  {str(dtype).split('.')[-1]:<16}"
            detail = ""
            if not ok1:
                detail += (f"v1 y误差 {max_abs_diff(y_ref, y_v1):.2e} "
                           f"h误差 {max_abs_diff(h_ref, h_v1):.2e} ")
            if not ok2:
                detail += (f"v2 y误差 {max_abs_diff(y_ref, y_v2):.2e} "
                           f"h误差 {max_abs_diff(h_ref, h_v2):.2e}")
            rep.check(label, ok1 and ok2, detail)


# ============================================================
# 三、SwiGLU
# ============================================================
def test_swiglu(rep: Report, device: str):
    print("\n【SwiGLU】")
    for shape in SHAPES:
        for dtype, atol, rtol in DTYPES:
            g = torch.randn(*shape, device=device, dtype=dtype)
            u = torch.randn(*shape, device=device, dtype=dtype)

            out_ref = ref.swiglu(g, u)
            out_tri = tk.swiglu(g, u)

            ok = torch.allclose(out_ref, out_tri, atol=atol, rtol=rtol)
            diff = max_abs_diff(out_ref, out_tri)
            label = f"{str(shape):>14}  {str(dtype).split('.')[-1]:<16}"
            rep.check(label, ok, f"最大误差 {diff:.2e}" if not ok else "")


# ============================================================
# 四、极端情况（最容易暴露 bug）
# ============================================================
def test_edge_cases(rep: Report, device: str):
    print("\n【极端情况】")

    # 1. 全零输入：mean_square = 0，加 eps 后不能变成 NaN
    x = torch.zeros(4, 128, device=device)
    w = torch.ones(128, device=device)
    y = tk.rmsnorm(x, w, eps=1e-6)
    rep.check("全零输入不产生 NaN/Inf",
              not (torch.isnan(y).any() or torch.isinf(y).any()),
              f"出现 {torch.isnan(y).sum().item()} 个 NaN")

    # 2. 极大值：验证数值稳定性
    x = torch.full((2, 128), 1e4, device=device)
    w = torch.ones(128, device=device)
    y = tk.rmsnorm(x, w, eps=1e-6)
    y_ref = ref.rmsnorm(x, w, eps=1e-6)
    rep.check("极大值输入数值稳定",
              torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3),
              f"最大误差 {max_abs_diff(y, y_ref):.2e}")

    # 3. 单行单列
    x = torch.randn(1, 1, device=device)
    w = torch.randn(1, device=device)
    y = tk.rmsnorm(x, w, eps=1e-6)
    y_ref = ref.rmsnorm(x, w, eps=1e-6)
    rep.check("N=1, M=1 的极端形状",
              torch.allclose(y, y_ref, atol=1e-4),
              f"最大误差 {max_abs_diff(y, y_ref):.2e}")

    # 4. 非连续输入（view 之前必须 contiguous）
    x = torch.randn(128, 64, device=device).t()      # 转置后不连续
    w = torch.ones(128, device=device)
    y = tk.rmsnorm(x, w, eps=1e-6)
    y_ref = ref.rmsnorm(x, w, eps=1e-6)
    rep.check("非连续输入（转置后）",
              torch.allclose(y, y_ref, atol=1e-4),
              f"最大误差 {max_abs_diff(y, y_ref):.2e}")

    # 5. SwiGLU 尾部对齐：元素数不是 BLOCK 的整数倍
    n = 1024 * 3 + 7
    g = torch.randn(n, device=device)
    u = torch.randn(n, device=device)
    out = tk.swiglu(g, u)
    out_ref = ref.swiglu(g, u)
    rep.check(f"SwiGLU 尾部不对齐（n={n}）",
              torch.allclose(out, out_ref, atol=1e-4),
              f"最大误差 {max_abs_diff(out, out_ref):.2e}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("Triton Kernel 正确性验证")
    print("=" * 70)

    if not torch.cuda.is_available():
        print()
        print("⚠️  当前环境没有 GPU，Triton kernel 无法运行。")
        print()
        print("请在以下环境之一运行：")
        print("  · Google Colab（代码执行程序 → 更改运行时类型 → T4 GPU）")
        print("  · DCU 集群（需要 DAS 官方 triton）")
        print()
        print("提示：纯 torch 的参考实现可以先在 CPU 上验证数学是否正确：")
        print("  python src/reference.py")
        return 1

    device = "cuda"
    print(f"\n设备: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton
        print(f"Triton : {triton.__version__}")
    except ImportError:
        print("Triton : 未安装 ❌")
        return 1

    rep = Report()
    test_rmsnorm(rep, device)
    test_fused_add_rmsnorm(rep, device)
    test_swiglu(rep, device)
    test_edge_cases(rep, device)

    ok = rep.summary()

    if ok:
        print()
        print("下一步：跑 benchmarks/run_bench.py 测性能。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

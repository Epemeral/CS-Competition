# Triton Kernel 工程 · T4 阶段工作台

> 用途：写、验证、测量 Triton 融合算子，产出比赛要的**消融数据**
> 对应教程：`1D/19 算子融合导论` + `3.1/02 SwiGLU` + `3.1/03 RMSNorm` + `3.1/05 Autotune`

---

## 这个工程解决什么问题

比赛里 T4 的验收标准不是「你写了 kernel」，而是：

> **「你让系统变快了多少，并且能说清楚为什么。」**

所以每个 kernel 都必须走完这条链：

```
写 kernel  →  数值对齐参考实现  →  测加速比  →  记录消融数据
   ↓              ↓                    ↓            ↓
kernels.py    tests/              benchmarks/   results/
```

**只写 kernel 不验证 = 白写；验证了不测性能 = 拿不到分。**

---

## 目录结构

```
triton_kernels/
├── README.md                  ← 你在这
├── requirements.txt
│
├── src/
│   ├── reference.py           PyTorch 参考实现（正确性的「标准答案」）
│   ├── triton_kernels.py      Triton kernel 实现（要优化的对象）
│   └── bench_utils.py         计时工具（预热 + 多次取中位数）
│
├── tests/
│   └── test_correctness.py    正确性验证（数值必须对齐）
│
├── benchmarks/
│   └── run_bench.py           性能基准（产出消融数据）
│
├── results/                   消融数据落盘（JSON）
│
└── notebooks/
    └── （Colab 版本，按需生成）
```

---

## 快速开始

### 方式 A：Google Colab（推荐，零配置）

```python
# 1. 上传整个 triton_kernels 文件夹，或者把代码粘贴进单元格
# 2. 装依赖（Colab 已自带 torch 和 triton）
!pip install -q triton

# 3. 跑验证
!python tests/test_correctness.py

# 4. 跑基准
!python benchmarks/run_bench.py
```

### 方式 B：本地 / DCU 集群

```bash
pip install -r requirements.txt

python tests/test_correctness.py     # 先过正确性
python benchmarks/run_bench.py       # 再测性能
```

> ⚠️ **DCU 集群上**：需要 DAS 官方 triton
> `triton-3.5.1+das.opt1.dtk2604.torch290-cp310`
> （不能用 PyPI 的通用版，见《环境重建清单》）

---

## 工作流：加一个新 kernel 的四步

以 RMSNorm 为例：

### 第 1 步：先在 `src/reference.py` 写参考实现

用最直白的 PyTorch 写，**只求正确，不求快**：

```python
def rmsnorm(x, weight, eps=1e-6):
    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(mean_square + eps) * weight
```

### 第 2 步：在 `src/triton_kernels.py` 写 kernel

```python
@triton.jit
def _rmsnorm_fwd_kernel(...):
    ...
```

### 第 3 步：在 `tests/test_correctness.py` 加测试

```python
check("rmsnorm", ref.rmsnorm, tk.rmsnorm, ...)
```

**必须测非 2 的幂的形状**——Triton 的 BLOCK 边界最容易出错。

### 第 4 步：在 `benchmarks/run_bench.py` 加基准

跑一次，数据自动落 `results/*.json`。

---

## 验收标准（对照比赛）

| 项目 | 标准 | 对应分数 |
|---|---|---|
| 数值正确性 | `allclose(atol=1e-4)` 通过所有形状 | **入场券**（不过直接取消资格） |
| 加速比 | 相对 PyTorch 原生有提升 | 性能分 |
| 消融数据 | 有 before/after + 多形状对比 | 创新分 6~12 |
| 可解释性 | 能说清「为什么快」（省了多少 HBM 往返） | 答辩 |

---

## 当前进度

| Kernel | 参考实现 | Triton 实现 | 正确性 | 性能基准 | 状态 |
|---|---|---|---|---|---|
| RMSNorm v1（整行） | ✅ | ✅ | ✅ | ✅ | 🟢 完成 |
| **RMSNorm v2（分块归约）** | ✅ | ✅ | ✅ | ✅ **2.22x** | 🟢 完成 |
| 融合 Add+RMSNorm v1 | ✅ | ✅ | ✅ | ✅ | 🟢 完成 |
| **融合 Add+RMSNorm v2** | ✅ | ✅ | 待跑 | 待跑 | 🟡 待验证 |
| SwiGLU | ✅ | ✅ | ✅ | ✅ **1.60x** | 🟢 完成 |
| RoPE | ✅ | ✅ | 待跑 | 待跑 | 🟡 待验证 |
| QKV 切分 + RoPE 融合 | ✅ | ✅ | 待跑 | 待跑 | 🟡 待验证 |

---

## 已验证的优化成果（Colab T4 实测）

### RMSNorm：分块归约解决寄存器爆炸

`(512, 18944)` 三方对比（PyTorch / v1 / v2）：

| dtype | PyTorch | v1 | v2 | v1 加速 | **v2 加速** |
|---|---|---|---|---|---|
| float32 | 1.162 ms | 2.418 ms | **0.547 ms** | 0.48x | **2.13x** |
| float16 | 0.645 ms | 0.608 ms | **0.291 ms** | 1.06x | **2.22x** |
| bfloat16 | 0.653 ms | 0.580 ms | **0.294 ms** | 1.12x | **2.22x** |

**v2 相对 v1 最大提升 4.42 倍。**

**根因**：`next_power_of_2(18944) = 32768`，v1 一个 program 要持有 32768 个 fp32，
寄存器爆掉 spill 到 local memory。
**方案**：v2 把 `BLOCK_N` 固定上限 4096，循环分块 —— 用「多读一次 x」换「寄存器压力恒定」。

### 融合的收益

`(2048, 3584)` 上，融合 Add+RMSNorm 相对「分开做」提升 **2.18x**，
带宽利用率 **63.6%**（不融合时中间张量 h 要两次 HBM 往返）。

### 一个方法论教训

**测试尺寸太小会得出错误结论。**
最初只测到 128 行（1.8 MB），带宽利用率只有 18.7% 且波动大；
加大到 2048 行后升到 47.6%~69.5%，加速比也从 1.37x 升到 1.97x。

---

## 三条纪律

**1. 先跑正确性，再测性能。**
一个错的 kernel 跑得再快也没意义——比赛里正确性不过直接取消资格。

**2. 每次改动都要留数据。**
改了什么、快了还是慢了、为什么。这些就是创新分的证据。

**3. 测性能必须多次取中位数。**
单次测量不可信（我们实测过同一段代码两次差 7 倍）。

---

## 和教程的对应

| 教程章节 | 对应本工程的 |
|---|---|
| 1D/19 算子融合导论 | 理解「为什么融合能省 HBM 往返」 |
| 3.1/02 Triton 融合 SwiGLU | `triton_kernels.swiglu` |
| 3.1/03 Triton 融合 RMSNorm | `triton_kernels.rmsnorm` |
| 3.1/05 Autotune 与 Profiling | `bench_utils.py` + `benchmarks/` |

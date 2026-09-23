# -*- coding: utf-8 -*-
"""Generate the standalone T2 attention/KV-cache workbench notebook."""
import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": source.strip("\n").splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": source.strip("\n").splitlines(keepends=True)})


md("""
# T2 / Attention 与 KV-cache 优化工作台

这是独立的 T2 实验入口，覆盖 Qwen2.5-7B 的 GQA 形状、causal attention 的 eager 与 SDPA 对比、KV-cache 两种布局消融，以及可选的 T4 融合算子检查。Notebook 只负责可复现实验和导出数据，真实比赛成绩必须在指定 DCU 镜像上重新测量。

运行顺序：环境检查 -> 形状/数值正确性 -> GPU 预热与中位数计时 -> JSON 导出。没有 GPU 时仍可完成 CPU 正确性检查，但不会伪造性能结果。
""")

code("""
import json, platform, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Memory GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print('Platform:', platform.platform())
""")

code("""
from pathlib import Path
import json

ROOT = Path.cwd()
if (ROOT / 'kernels' / 'src' / 'reference.py').is_file():
    KERNELS = ROOT / 'kernels'
elif Path('/content/cs-comp/kernels/src/reference.py').is_file():
    KERNELS = Path('/content/cs-comp/kernels')
else:
    KERNELS = None
print('KERNELS =', KERNELS)
""")

md("""
## 1. Qwen2.5-7B GQA 形状

Qwen2.5-7B 的 `28` 个 query heads 共享 `4` 个 key/value heads，`head_dim=128`。注意力输入采用 `(B, H, T, D)`；KV-cache 还会比较 `(B, H, T, D)` 与 `(B, T, H, D)` 两种布局。
""")

code("""
qwen = dict(hidden_size=3584, num_attention_heads=28,
            num_key_value_heads=4, head_dim=128)
B, T, S = 1, 17, 23
H, HKV, D = qwen['num_attention_heads'], qwen['num_key_value_heads'], qwen['head_dim']
assert qwen['hidden_size'] == H * D
q = torch.randn(B, H, T, D)
k = torch.randn(B, HKV, S, D)
v = torch.randn(B, HKV, S, D)
print('Q/K/V:', tuple(q.shape), tuple(k.shape), tuple(v.shape))
assert H % HKV == 0
print('GQA group size:', H // HKV)
""")

md("""
## 2. Eager 与 SDPA 正确性

`repeat_interleave` 展开 KV heads 后，eager 路径和 PyTorch `scaled_dot_product_attention` 使用相同的 causal mask。SDPA 是否走到 fused backend 由当前 PyTorch/ROCm/驱动决定，不能假设 CUDA 专属实现一定存在。
""")

code("""
def expand_kv(x, num_heads):
    groups = num_heads // x.shape[1]
    return x.repeat_interleave(groups, dim=1)

def eager_attention(q, k, v, causal=False):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        tq, tk = q.shape[-2], k.shape[-2]
        mask = torch.ones(tq, tk, device=q.device, dtype=torch.bool).tril(tk - tq)
        scores = scores.masked_fill(~mask, float('-inf'))
    return torch.softmax(scores, dim=-1).matmul(v)

def sdpa_attention(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

q0 = q.float()
k0, v0 = expand_kv(k.float(), H), expand_kv(v.float(), H)
y_eager = eager_attention(q0, k0, v0, causal=False)
y_sdpa = sdpa_attention(q0, k0, v0, causal=False)
max_err = (y_eager - y_sdpa).abs().max().item()
print('attention max abs error:', max_err)
assert torch.allclose(y_eager, y_sdpa, atol=2e-5, rtol=2e-5)
print('Attention correctness: PASS')
""")

md("""
## 3. KV-cache 布局消融

这里测的是同一批 token 的读取成本，不把布局转换时间隐藏掉。实际 decode 还应在 DCU 上结合 batch、历史长度和 paged cache 再测；这个单元先提供稳定的局部证据。
""")

code("""
def cache_read_bhtd(cache):
    return cache[:, :, -1, :].contiguous()

def cache_read_bthd(cache):
    return cache[:, -1, :, :].contiguous()

def median_ms(fn, warmup=10, repeats=30):
    for _ in range(warmup):
        fn()
    if not torch.cuda.is_available():
        start = time.perf_counter()
        for _ in range(repeats): fn()
        return (time.perf_counter() - start) * 1000 / repeats
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record(); fn(); end.record(); end.synchronize()
        samples.append(begin.elapsed_time(end))
    return float(torch.tensor(samples).median().item())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
cache_bhtd = torch.randn(B, HKV, S, D, device=device, dtype=dtype)
cache_bthd = cache_bhtd.transpose(1, 2).contiguous()
layout_results = {
    'BHTD': median_ms(lambda: cache_read_bhtd(cache_bhtd)),
    'BTHD': median_ms(lambda: cache_read_bthd(cache_bthd)),
}
print('KV-cache read median ms:', layout_results)
""")

md("""
## 4. Attention 基准与结果导出

只在 GPU 上记录 attention 性能；CPU 结果仅用于开发调试。结果中包含设备、版本、形状、误差和计时参数，便于与 T1/T3/T4 分开汇总。
""")

code("""
bench = {'eager_ms': None, 'sdpa_ms': None}
if torch.cuda.is_available():
    qd, kd, vd = q0.to(device=device, dtype=dtype), k0.to(device=device, dtype=dtype), v0.to(device=device, dtype=dtype)
    bench['eager_ms'] = median_ms(lambda: eager_attention(qd, kd, vd), warmup=20, repeats=50)
    bench['sdpa_ms'] = median_ms(lambda: sdpa_attention(qd, kd, vd), warmup=20, repeats=50)
    print('attention median ms:', bench)
else:
    print('No GPU: skip performance numbers.')

out = ROOT / 'results' / 't2'
out.mkdir(parents=True, exist_ok=True)
payload = {
    'task': 'T2',
    'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'torch': torch.__version__,
    'dtype': str(dtype),
    'qwen': qwen,
    'shape': {'B': B, 'T': T, 'S': S},
    'correctness_max_abs_error': max_err,
    'attention_ms': bench,
    'kv_cache_read_ms': layout_results,
    'warmup': 20 if torch.cuda.is_available() else 10,
    'repeats': 50 if torch.cuda.is_available() else 30,
}
(out / 't2_results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
print('saved:', out / 't2_results.json')
""")

md("""
## 5. 后续接入路线

先在固定形状上确认数值，再把 `T/S` 扩展到 prefill 与 decode 矩阵，记录 TTFT、TPOT、吞吐、P50/P95 和峰值显存。随后再评估 FlashAttention/PagedAttention 或 vLLM 接入。任何 fused kernel 都必须与本 Notebook 的 eager 参考结果做 token-level 对齐，不能只比较平均耗时。
""")

nb = {
    'cells': cells,
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
                 'language_info': {'name': 'python'}, 'colab': {'provenance': []}},
    'nbformat': 4,
    'nbformat_minor': 5,
}
out = Path(__file__).with_name('t2_optimization_workbench.ipynb')
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('generated', out)

# -*- coding: utf-8 -*-
"""
端到端 benchmark：vLLM 原版 vs 接入 Triton kernel 后。

【这是最重要的一张表】

    前面的 kernel 基准测的是「单个算子多快」，
    这里测的是「整个模型快了多少」—— **后者才是比赛评分口径**。

    官方 FAQ 原话：性能分核心是**并发吞吐（tokens/s）**。

【怎么跑】

    只加载一次模型，在两次测量之间切换 patch：
        1. 加载模型（enforce_eager=True）
        2. 跑 baseline（原版 vLLM）
        3. 应用 monkey-patch
        4. 再跑一次
        5. 对比吞吐

【⚠️ 为什么必须开 enforce_eager=True】

    vLLM 默认用 torch.compile + CUDA Graph。
    我们的 Triton kernel 在里面可能：
        · 编译失败（vLLM 的编译流程不认识它）
        · 被 CUDA Graph 绕过（捕获的是旧 kernel）
    所以先用 eager 模式验证「确实有收益」，再考虑编译兼容性。
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "integration"))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# T4（14.6 GiB）能跑 1.5B；7B 需要 DCU 的 64 GiB
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# 模拟真实请求：长短混合
PROMPTS = [
    "用一句话解释什么是 PagedAttention。",
    "海光 DCU 和 NVIDIA GPU 的主要区别是什么？",
    "为什么大模型推理的 decode 阶段是 memory-bound？",
    "解释一下 KV Cache 的作用。",
    "Triton 和 CUDA 有什么区别？",
    "什么是连续批处理（continuous batching）？",
    "为什么量化能提升推理吞吐？",
    "解释一下 RoPE 旋转位置编码的原理。",
] * 4   # 32 条请求，制造并发压力

MAX_TOKENS = 128
WARMUP_ROUNDS = 1


# ============================================================
# 测量
# ============================================================
def measure(llm, prompts, sampling_params, label=""):
    """跑一轮推理，返回吞吐指标。"""
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    result = {
        "label": label,
        "num_prompts": len(prompts),
        "total_output_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput_tok_per_s": total_tokens / elapsed,
        "latency_per_request_ms": elapsed / len(prompts) * 1000,
    }
    return result


def print_result(r):
    print(f"  {r['label']:<22}"
          f"{r['throughput_tok_per_s']:>10.1f} tokens/s"
          f"{r['elapsed_s']:>10.2f} s"
          f"{r['total_output_tokens']:>10d} tok"
          f"{r['latency_per_request_ms']:>12.1f} ms")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 78)
    print("端到端 benchmark：vLLM 原版 vs 接入 Triton kernel")
    print("=" * 78)

    if not torch.cuda.is_available():
        print("\n需要 GPU 环境（Colab / DCU 集群）")
        return 1

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("\n未安装 vLLM。请先：")
        print("    !pip install -q vllm")
        return 1

    from vllm_patch import patch_qwen2, unpatch_qwen2, verify_patch_equivalence

    # ---- 第 0 步：算子等价性自检（最重要的一步）----
    if not verify_patch_equivalence():
        print("算子不等价，先修正确性再测端到端！")
        return 1

    # ---- 第 1 步：加载模型（只加载一次）----
    print("=" * 78)
    print(f"加载模型：{DEFAULT_MODEL}")
    print("=" * 78)
    llm = LLM(
        model=DEFAULT_MODEL,
        enforce_eager=True,             # 关键：先不开编译
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        disable_log_stats=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # ---- 预热 ----
    print("\n预热中...")
    for _ in range(WARMUP_ROUNDS):
        llm.generate(PROMPTS[:4], sp)

    # ---- 第 2 步：baseline ----
    print()
    print("=" * 78)
    print(f"配置：{len(PROMPTS)} 条请求，max_tokens={MAX_TOKENS}，eager 模式")
    print("=" * 78)
    print(f"  {'配置':<22}{'吞吐':>18}{'耗时':>12}{'输出 token':>14}{'平均延迟':>16}")
    print("-" * 78)

    unpatch_qwen2(verbose=False)        # 确保干净
    base = measure(llm, PROMPTS, sp, label="原版 vLLM")
    print_result(base)

    # ---- 第 3 步：接入 Triton ----
    print()
    patch_qwen2(verbose=True)
    patched = measure(llm, PROMPTS, sp, label="接入 Triton kernel")
    print_result(patched)

    # ---- 第 4 步：对比 ----
    speedup = patched["throughput_tok_per_s"] / base["throughput_tok_per_s"]
    print()
    print("=" * 78)
    print("结果")
    print("=" * 78)
    print(f"  原版吞吐      : {base['throughput_tok_per_s']:.1f} tokens/s")
    print(f"  接入后吞吐    : {patched['throughput_tok_per_s']:.1f} tokens/s")
    print(f"  端到端提升    : {speedup:.4f}x  ({(speedup - 1) * 100:+.2f}%)")

    if speedup < 1.0:
        print()
        print("  ⚠️ 端到端变慢了。可能原因：")
        print("     · kernel 本身更快，但在整体中占比太小（被其它算子淹没）")
        print("     · 额外的 contiguous / 类型转换开销抵消了收益")
        print("     · eager 模式下 vLLM 自己的算子已经很快")

    # ---- 落盘 ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"e2e_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "_meta": {
            "model": DEFAULT_MODEL,
            "device": torch.cuda.get_device_name(0),
            "num_prompts": len(PROMPTS),
            "max_tokens": MAX_TOKENS,
            "enforce_eager": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baseline": base,
        "patched": patched,
        "speedup": speedup,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"原始数据已保存: {out}")

    # 收尾：恢复原状
    unpatch_qwen2(verbose=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

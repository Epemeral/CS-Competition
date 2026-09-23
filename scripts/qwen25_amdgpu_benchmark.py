#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可复现的 AMDGPU Qwen2.5-7B-Instruct 推理基准。

这个脚本面向 ROCm 环境。ROCm 下 PyTorch 仍通过 ``torch.cuda`` 暴露设备，
因此这里统一使用 ``cuda`` 设备名，但结果会额外记录 HIP 版本。

默认只测 Transformers 单进程基线；先用它确认模型、正确性和测量口径，
再把同一组请求交给 vLLM/自研 kernel 做对比。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


PROMPTS = [
    "用一句话解释什么是 KV Cache。",
    "解释连续批处理为什么能提高大模型推理吞吐。",
    "海光 DCU 上运行 Qwen2.5 时，应该先检查哪些软件版本？",
    "简要说明 RMSNorm 和 LayerNorm 的区别。",
    "为什么 decode 阶段通常受显存带宽限制？",
    "解释 RoPE 位置编码的基本原理。",
    "给出一个检查模型目录是否完整的命令。",
    "什么情况下应该使用 SDPA 而不是 eager attention？",
]


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_int_list(value: str) -> list[int]:
    values = [positive(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def resolve_model(args: argparse.Namespace) -> tuple[str, dict]:
    """解析本地目录，必要时通过 ModelScope 下载模型。"""
    requested = Path(args.model)
    if requested.exists():
        return str(requested.resolve()), {"source": "local", "requested": args.model}

    if args.model_source in {"auto", "modelscope"}:
        try:
            from modelscope import snapshot_download

            model_dir = snapshot_download(
                args.model,
                revision=args.revision,
                cache_dir=args.cache_dir,
            )
            return str(Path(model_dir).resolve()), {
                "source": "modelscope",
                "requested": args.model,
                "revision": args.revision,
            }
        except Exception as exc:
            if args.model_source == "modelscope":
                raise RuntimeError(f"ModelScope 下载失败: {exc}") from exc
            print(f"ModelScope 不可用，回退 Transformers/Hugging Face: {exc}")

    return args.model, {"source": "transformers", "requested": args.model}


def environment(torch, transformers, model_dir: str) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "model_dir": model_dir,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "device": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "total_memory_bytes": props.total_memory,
                "total_memory_gib": round(props.total_memory / 1024**3, 2),
            }
        )
    return info


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def chat_ids(tokenizer, prompt: str, max_input_tokens: int) -> list[int]:
    # 短问题无法代表长上下文；重复同一问题的语义内容只是为了构造稳定的
    # 压测长度，最终仍经过 chat template 和 tokenizer。真实评测必须换成官方输入。
    content = prompt
    ids = []
    while len(ids) < max_input_tokens:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if len(ids) >= max_input_tokens:
            break
        content = content + "\n" + prompt
    # 保留尾部可以保证 generation prompt 不被截掉；官方数据集有独立长度约束时，
    # 应以官方 tokenizer/evaluate 脚本为准，不要用这个截断结果替代比赛输入。
    return list(ids[-max_input_tokens:])


def make_batch(tokenizer, batch_size: int, max_input_tokens: int, pad_id: int):
    rows = [chat_ids(tokenizer, PROMPTS[i % len(PROMPTS)], max_input_tokens)
            for i in range(batch_size)]
    width = max(map(len, rows))
    input_ids = [[pad_id] * (width - len(row)) + row for row in rows]
    attention_mask = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
    return input_ids, attention_mask, width


def count_generated(row, eos_ids: set[int], pad_id: int) -> int:
    count = 0
    for token in row:
        token = int(token)
        if token in eos_ids or token == pad_id:
            break
        count += 1
    return count


def run_batch(torch, model, batch, device: str, max_new_tokens: int, eos_ids: set[int],
              pad_id: int) -> tuple[float, int]:
    input_ids, attention_mask, width = batch
    ids = torch.tensor(input_ids, device=device)
    mask = torch.tensor(attention_mask, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids,
            attention_mask=mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=list(eos_ids),
            pad_token_id=pad_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = sum(count_generated(row, eos_ids, pad_id)
                    for row in output[:, width:].detach().cpu().tolist())
    return elapsed, generated


def benchmark(torch, model, tokenizer, args, device: str) -> list[dict]:
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = next(iter(eos_ids))
    rows = []
    for batch_size in args.batch_sizes:
        batch = make_batch(tokenizer, batch_size, args.input_tokens, pad_id)
        for _ in range(args.warmup):
            run_batch(torch, model, batch, device, args.max_new_tokens, eos_ids, pad_id)
        samples = []
        for _ in range(args.repeats):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            elapsed, output_tokens = run_batch(
                torch, model, batch, device, args.max_new_tokens, eos_ids, pad_id
            )
            samples.append(
                {
                    "seconds": elapsed,
                    "output_tokens": output_tokens,
                    "output_tokens_per_second": output_tokens / elapsed,
                    "requests_per_second": batch_size / elapsed,
                    "peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated() if device == "cuda" else None
                    ),
                }
            )
        rows.append(
            {
                "batch_size": batch_size,
                "input_width": batch[2],
                "max_new_tokens": args.max_new_tokens,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "median_output_tokens_per_second": statistics.median(
                    item["output_tokens_per_second"] for item in samples
                ),
                "median_requests_per_second": statistics.median(
                    item["requests_per_second"] for item in samples
                ),
                "samples": samples,
            }
        )
        best = rows[-1]
        print(
            f"batch={batch_size:<3} input={batch[2]:<4} "
            f"tokens/s={best['median_output_tokens_per_second']:.2f} "
            f"req/s={best['median_requests_per_second']:.3f}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-source", choices=["auto", "modelscope", "transformers"], default="auto")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--cache-dir", default="./model_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--input-tokens", type=positive, default=512)
    parser.add_argument("--max-new-tokens", type=positive, default=128)
    parser.add_argument("--warmup", type=positive, default=2)
    parser.add_argument("--repeats", type=positive, default=5)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("没有检测到 AMDGPU/ROCm 设备；请在 AMDGPU Notebook 中运行。")

    device = "cuda"
    model_dir, source = resolve_model(args)
    dtype = getattr(torch, args.dtype)
    print(f"模型: {model_dir}\n来源: {source}\nattention: {args.attention}\ndtype: {args.dtype}")
    print(json.dumps(environment(torch, transformers, model_dir), ensure_ascii=False, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        attn_implementation=args.attention,
        trust_remote_code=True,
    ).to(device)
    if model.config.model_type != "qwen2":
        raise RuntimeError(f"模型类型不是 qwen2: {model.config.model_type}")
    model.eval()
    model.config.use_cache = True

    result = {
        "schema_version": 1,
        "purpose": "amdgpu_transformers_benchmark",
        "official_score": False,
        "model": source,
        "settings": vars(args),
        "environment": environment(torch, transformers, model_dir),
        "git": git_info(),
        "measurements": benchmark(torch, model, tokenizer, args, device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

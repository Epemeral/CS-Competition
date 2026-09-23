"""Preparation baseline, not the organizer's official baseline."""
import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from t1.core import digest, make_batch_indices, padding_stats, read_cases, trim_generated


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true", help="Offline random Qwen2, pipeline testing only")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--data", default="data/smoke.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--batch-size", type=positive, default=1)
    parser.add_argument("--max-new-tokens", type=positive, default=16)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--warmup", type=positive, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--length-bucket", action="store_true",
                        help="Sort cases by token length inside batches to reduce padding")
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        parser.error("Output exists; choose a new path to preserve experiments")
    cases = read_cases(args.data)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA/HIP device unavailable")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("Use float32 for the CPU smoke test")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, args.dtype)
    tokenizer = None
    if args.tiny:
        config = Qwen2Config(vocab_size=259, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=4096, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = args.attention
        model = Qwen2ForCausalLM(config).to(dtype=dtype, device=args.device)
        inputs = [[1] + [byte + 3 for byte in case["prompt"].encode("utf-8")] for case in cases]
        model_identity = {"name": "random-tiny-qwen2", "seed": args.seed, "config": config.to_dict()}
        # Attention implementation is experimental, not part of model identity.
        model_identity["config"].pop("_attn_implementation_autoset", None)
        pad, eos = 0, [2]
    else:
        options = {"revision": args.revision, "local_files_only": args.local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                    attn_implementation=args.attention, **options).to(args.device)
        if model.config.model_type != "qwen2":
            parser.error("Expected the Qwen2 architecture used by Qwen2.5")
        inputs = [tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                  tokenize=True, add_generation_prompt=True) for case in cases]
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        if not eos or any(token is None for token in eos):
            parser.error("Model has no EOS token")
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0]
        model_identity = {"name": args.model, "revision": args.revision,
                          "resolved_commit": getattr(model.config, "_commit_hash", None)}
    model.eval()
    if max(map(len, inputs)) + args.max_new_tokens > model.config.max_position_embeddings:
        parser.error("Input plus output exceeds model context; no silent truncation")
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                                  eos_token_id=eos, pad_token_id=pad, use_cache=not args.no_cache)

    lengths = list(map(len, inputs))
    batch_plan = make_batch_indices(lengths, args.batch_size, args.length_bucket)
    batches = []
    for indices in batch_plan:
        rows = [inputs[index] for index in indices]
        width = max(map(len, rows))
        # Decoder-only generation needs left padding and an explicit attention mask.
        ids = torch.tensor([[pad] * (width - len(row)) + row for row in rows], device=args.device)
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows], device=args.device)
        batches.append((ids, mask, indices))
    padding = padding_stats(lengths, batch_plan)

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def run():
        generated = [None] * len(inputs)
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for ids, mask, indices in batches:
                output = model.generate(input_ids=ids, attention_mask=mask, generation_config=generation)
                rows = output[:, ids.shape[1]:].cpu().tolist()
                for index, row in zip(indices, rows):
                    generated[index] = trim_generated(row, eos)
        synchronize()
        elapsed = time.perf_counter() - started
        return elapsed, generated

    for _ in range(args.warmup):
        run()
    measurements, first = [], None
    for _ in range(args.repeats):
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elapsed, generated = run()
        if first is not None and generated != first:
            raise RuntimeError("Generated token ids changed between repetitions")
        first = generated
        count = sum(map(len, generated))
        measurements.append({"seconds": elapsed, "output_tokens": count,
                             "output_tokens_per_second": count / elapsed,
                             "requests_per_second": len(cases) / elapsed,
                             "input_tokens": padding["input_tokens"],
                             "padded_input_tokens": padding["padded_input_tokens"],
                             "padding_waste_ratio": padding["padding_waste_ratio"],
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    outputs = [{"id": case["id"], "input_ids": inp, "token_ids": out,
                "text": tokenizer.decode(out, skip_special_tokens=True) if tokenizer else None}
               for case, inp, out in zip(cases, inputs, first)]
    result = {
        "schema_version": 1, "purpose": "development_smoke" if args.tiny else "preliminary_baseline",
        "official_score": False,
        "contract": {"model": model_identity, "dataset_sha256": digest(cases),
                     "max_new_tokens": args.max_new_tokens, "decoding": "greedy", "eos": eos, "pad": pad},
        "settings": vars(args),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "hip": torch.version.hip, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
                        "git_commit": commit, "git_dirty": dirty},
        "measurement_scope": "Pre-tokenized resident inputs; synchronized generate over static batches; excludes load/tokenize/decode; includes generated EOS",
        "batching": {"strategy": "length_sorted" if args.length_bucket else "input_order",
                     "batch_size": args.batch_size, "batch_count": len(batch_plan), **padding},
        "input_tokens": padding["input_tokens"], "runs": measurements,
        "median_output_tokens_per_second": statistics.median(row["output_tokens_per_second"] for row in measurements),
        "outputs": outputs,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"result": str(target), "purpose": result["purpose"],
                      "median_output_tokens_per_second": result["median_output_tokens_per_second"]}))


if __name__ == "__main__":
    main()

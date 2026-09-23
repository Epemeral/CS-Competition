#!/usr/bin/env python3
"""Convert the supplied benchmark extracts to one small, explicit JSONL contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: every line must be an object")
                yield value


def normalized(case_id: str, dataset: str, prompt: str, reference: Any, task_type: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {"id": case_id, "dataset": dataset, "prompt": prompt, "reference": reference, "task_type": task_type, "metadata": metadata}


def convert_longbench(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    for number, row in enumerate(jsonl(path)):
        if limit is not None and number >= limit:
            break
        yield normalized(row.get("_id", str(number)), "longbench", f"{row['context']}\n\nQuestion: {row['input']}", row.get("answers"), "generation", {"dataset": row.get("dataset"), "language": row.get("language"), "length": row.get("length")})


def convert_hellobench(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    for number, row in enumerate(jsonl(path)):
        if limit is not None and number >= limit:
            break
        yield normalized(str(row.get("id", number)), "hellobench", row["instruction"], None, "generation", {"category": row.get("category"), "subcategory": row.get("subcategory"), "checklists": row.get("formatted_checklists"), "raw_text": row.get("raw_text")})


def convert_ax(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    for number, row in enumerate(jsonl(path)):
        if limit is not None and number >= limit:
            break
        prompt = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nDoes the premise entail the hypothesis? Answer entailment or not_entailment."
        yield normalized(str(row.get("idx", number)), "superglue_ax", prompt, row.get("label"), "classification", {"pair_id": row.get("pair_id"), "logic": row.get("logic")})


def convert_tau2(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for number, row in enumerate(rows):
        if limit is not None and number >= limit:
            break
        instructions = row.get("user_scenario", {}).get("instructions", {})
        prompt = instructions.get("reason_for_call", "")
        yield normalized(str(row.get("id", number)), "tau2_telecom", prompt, row.get("evaluation_criteria"), "agent", {"description": row.get("description"), "ticket": row.get("ticket"), "initial_state": row.get("initial_state"), "user_instructions": instructions})


def convert_mmlu(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MMLU-Pro parquet conversion requires pyarrow") from exc
    rows = pq.read_table(path).to_pylist()
    for number, row in enumerate(rows):
        if limit is not None and number >= limit:
            break
        options = row.get("options", [])
        prompt = f"{row.get('question', '')}\n" + "\n".join(f"{chr(65+i)}. {option}" for i, option in enumerate(options))
        reference = row.get("answer", row.get("answer_index"))
        yield normalized(str(number), "mmlu_pro", prompt, reference, "multiple_choice", {"category": row.get("category"), "source_fields": sorted(row)})


def find(root: Path, parts: tuple[str, ...]) -> Path:
    matches = [p for p in root.rglob(parts[-1]) if all(part in p.parts for part in parts[:-1])]
    if not matches:
        raise FileNotFoundError(f"could not find {'/'.join(parts)} below {root}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing extracted datasets")
    parser.add_argument("dataset", choices=["longbench", "hellobench", "ax", "mmlu_pro", "tau2_telecom"])
    parser.add_argument("--input", type=Path, help="explicit source file; otherwise discover it below root")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    sources = {"longbench": ("data", "multifieldqa_en.jsonl"), "hellobench": ("HelloBench", "test.jsonl"), "ax": ("SuperGLUE-AX", "AX-g.jsonl"), "mmlu_pro": ("mmlu_pro", "test-00000-of-00001.parquet"), "tau2_telecom": ("tau2-telecom", "tasks_small.json")}
    source = args.input or find(args.root, sources[args.dataset])
    converters = {"longbench": convert_longbench, "hellobench": convert_hellobench, "ax": convert_ax, "mmlu_pro": convert_mmlu, "tau2_telecom": convert_tau2}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in converters[args.dataset](source, args.limit):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

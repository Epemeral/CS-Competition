#!/usr/bin/env python3
"""Inspect staged benchmark files without loading the whole corpus into memory."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _text_lengths(row: dict[str, Any]) -> list[int]:
    return [len(value) for value in row.values() if isinstance(value, str)]


def inspect_jsonl(path: Path, sample_size: int) -> dict[str, Any]:
    count = 0
    keys: set[str] = set()
    lengths: list[int] = []
    samples: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        count += 1
        keys.update(row)
        lengths.extend(_text_lengths(row))
        if len(samples) < sample_size:
            samples.append(row)
    return {
        "format": "jsonl",
        "records": count,
        "keys": sorted(keys),
        "string_length": _summary(lengths),
        "samples": samples,
    }


def inspect_json(path: Path, sample_size: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        sample = value[:sample_size]
        keys = sorted({key for row in value if isinstance(row, dict) for key in row})
        return {"format": "json", "records": len(value), "keys": keys, "samples": sample}
    return {"format": "json", "records": 1, "keys": sorted(value) if isinstance(value, dict) else [], "samples": [value]}


def inspect_parquet(path: Path, sample_size: int) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"format": "parquet", "error": "pyarrow is not installed; metadata unavailable"}
    table = pq.ParquetFile(path)
    sample = table.read_row_groups([0], use_threads=False).slice(0, sample_size).to_pylist()
    return {
        "format": "parquet",
        "records": table.metadata.num_rows,
        "columns": table.schema_arrow.names,
        "row_groups": table.metadata.num_row_groups,
        "samples": sample,
    }


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def inspect_path(path: Path, sample_size: int) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return inspect_jsonl(path, sample_size)
    if path.suffix == ".json":
        return inspect_json(path, sample_size)
    if path.suffix == ".parquet":
        return inspect_parquet(path, sample_size)
    return {"format": path.suffix.lstrip(".") or "unknown", "skipped": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing extracted datasets")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    parser.add_argument("--sample-size", type=int, default=2)
    args = parser.parse_args()
    # PowerShell on Windows often exposes a GBK stdout; reports contain data
    # from multilingual corpora, so always emit UTF-8 when the stream allows it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")
    files = [p for p in args.root.rglob("*") if p.is_file() and p.suffix in {".jsonl", ".json", ".parquet"}]
    report = {"root": str(args.root.resolve()), "files": {str(p.relative_to(args.root)): inspect_path(p, args.sample_size) for p in sorted(files)}}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

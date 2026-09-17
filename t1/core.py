import hashlib
import json
from pathlib import Path


def read_cases(path):
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError("Dataset must not be empty")
    ids = set()
    for case in cases:
        if not isinstance(case.get("id"), str) or not isinstance(case.get("prompt"), str):
            raise ValueError("Each case needs string id and prompt")
        if case["id"] in ids:
            raise ValueError("Duplicate case id: " + case["id"])
        ids.add(case["id"])
    return cases


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def trim_generated(tokens, eos_ids):
    # Include the first generated EOS; discard only padding after it.
    for index, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[:index + 1]
    return tokens


def compare(reference, candidate):
    if reference["contract"] != candidate["contract"]:
        raise ValueError("Incompatible model/input/generation contract")
    left, right = reference["outputs"], candidate["outputs"]
    for rows in (left, right):
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Empty outputs or duplicate ids")
    if {row["id"] for row in left} != {row["id"] for row in right}:
        raise ValueError("Output case ids differ")
    lookup = {row["id"]: row for row in right}
    errors = []
    for row in left:
        other = lookup[row["id"]]
        if row["input_ids"] != other["input_ids"]:
            raise ValueError("Input token ids differ for " + row["id"])
        a, b = row["token_ids"], other["token_ids"]
        if a != b:
            position = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
            errors.append({"id": row["id"], "first_difference": position,
                           "reference": a[position:position + 5], "candidate": b[position:position + 5]})
    return errors

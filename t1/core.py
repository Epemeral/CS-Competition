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


def make_batch_indices(lengths, batch_size, sort_by_length=False):
    """Return batch index groups without changing the caller's case order."""
    if not lengths:
        raise ValueError("At least one input length is required")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if any(length < 1 for length in lengths):
        raise ValueError("Input lengths must be positive")
    order = list(range(len(lengths)))
    if sort_by_length:
        # Stable sorting keeps equal-length cases deterministic.
        order.sort(key=lambda index: (lengths[index], index))
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


def padding_stats(lengths, batches):
    """Summarize useful and padded input tokens for a static batch plan."""
    actual = sum(lengths)
    padded = sum(max(lengths[index] for index in batch) * len(batch) for batch in batches)
    padding = padded - actual
    return {
        "input_tokens": actual,
        "padded_input_tokens": padded,
        "padding_tokens": padding,
        "padding_waste_ratio": padding / padded if padded else 0.0,
    }


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

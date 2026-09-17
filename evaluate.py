"""Token equality checker for preparation results, not official scoring."""
import argparse
import json
from pathlib import Path

from t1.core import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    try:
        reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.reference, args.candidate)]
        errors = compare(*reports)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        return 2
    print(json.dumps({"passed": not errors, "cases": len(reports[0]["outputs"]), "mismatches": errors}, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_datasets import convert_ax, convert_hellobench, convert_longbench, convert_tau2
from scripts.inspect_datasets import inspect_jsonl


class DatasetToolTests(unittest.TestCase):
    def test_longbench_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text(json.dumps({"_id": "x", "input": "q", "context": "c", "answers": ["a"], "dataset": "demo"}) + "\n", encoding="utf-8")
            row = next(convert_longbench(path, None))
            self.assertEqual(row["task_type"], "generation")
            self.assertEqual(row["reference"], ["a"])

    def test_classification_and_agent_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ax = root / "ax.jsonl"
            ax.write_text(json.dumps({"idx": "1", "premise": "p", "hypothesis": "h", "label": "entailment"}) + "\n", encoding="utf-8")
            self.assertEqual(next(convert_ax(ax, None))["task_type"], "classification")
            hello = root / "hello.jsonl"
            hello.write_text(json.dumps({"id": "h", "instruction": "answer", "category": "qa"}) + "\n", encoding="utf-8")
            self.assertEqual(next(convert_hellobench(hello, None))["task_type"], "generation")
            tau = root / "tau.json"
            tau.write_text(json.dumps([{"id": "t", "user_scenario": {"instructions": {"reason_for_call": "call"}}}]), encoding="utf-8")
            self.assertEqual(next(convert_tau2(tau, None))["task_type"], "agent")

    def test_inspection_counts_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            path.write_text('{"x":"a"}\n{"x":"bb"}\n', encoding="utf-8")
            report = inspect_jsonl(path, 1)
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["keys"], ["x"])
            self.assertEqual(len(report["samples"]), 1)

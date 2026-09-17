import copy
import tempfile
import unittest
from pathlib import Path

from t1.core import compare, read_cases, trim_generated


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"contract": {"model": "tiny"}, "outputs": [
            {"id": "a", "input_ids": [1], "token_ids": [4, 2]},
            {"id": "b", "input_ids": [1, 3], "token_ids": [5, 6, 2]}]}

    def test_eos_inclusive_padding_exclusive(self):
        self.assertEqual(trim_generated([4, 2, 0, 0], [2]), [4, 2])
        self.assertEqual(trim_generated([4, 0, 5], [2]), [4, 0, 5])
        self.assertEqual(trim_generated([7, 3, 0], [2, 3]), [7, 3])

    def test_reordered_cases(self):
        candidate = copy.deepcopy(self.reference)
        candidate["outputs"].reverse()
        self.assertEqual(compare(self.reference, candidate), [])

    def test_first_difference_and_shorter_output(self):
        candidate = copy.deepcopy(self.reference)
        candidate["outputs"][0]["token_ids"] = [4]
        self.assertEqual(compare(self.reference, candidate)[0]["first_difference"], 1)
        candidate["outputs"][0]["token_ids"] = [8, 2]
        self.assertEqual(compare(self.reference, candidate)[0]["first_difference"], 0)

    def test_invalid_comparisons_rejected(self):
        for mutation in (lambda c: c.update(contract={}),
                         lambda c: c.update(outputs=[]),
                         lambda c: c["outputs"].pop(),
                         lambda c: c["outputs"].append(c["outputs"][0]),
                         lambda c: c["outputs"][0].update(input_ids=[8])):
            candidate = copy.deepcopy(self.reference)
            mutation(candidate)
            with self.assertRaises(ValueError):
                compare(self.reference, candidate)

    def test_duplicate_dataset_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text('{"id":"a","prompt":"x"}\n' * 2, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_cases(path)


if __name__ == "__main__":
    unittest.main()

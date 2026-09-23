import copy
import tempfile
import unittest
from pathlib import Path

from t1.core import (compare, make_batch_indices, padding_stats, read_cases,
                     trim_generated)


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

    def test_length_sorted_batches_reduce_padding_without_reordering_cases(self):
        lengths = [8, 2, 7, 3]
        batches = make_batch_indices(lengths, 2, sort_by_length=True)
        input_order_stats = padding_stats(lengths, make_batch_indices(lengths, 2))
        self.assertEqual(batches, [[1, 3], [2, 0]])
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(4)))
        sorted_stats = padding_stats(lengths, batches)
        self.assertLess(sorted_stats["padded_input_tokens"], input_order_stats["padded_input_tokens"])
        self.assertEqual(sorted_stats, {
            "input_tokens": 20,
            "padded_input_tokens": 22,
            "padding_tokens": 2,
            "padding_waste_ratio": 2 / 22,
        })

    def test_padding_stats_for_input_order_batches(self):
        lengths = [8, 2, 7, 3]
        batches = make_batch_indices(lengths, 2)
        stats = padding_stats(lengths, batches)
        self.assertEqual(stats["padded_input_tokens"], 30)
        self.assertEqual(stats["padding_tokens"], 10)
        self.assertAlmostEqual(stats["padding_waste_ratio"], 1 / 3)


if __name__ == "__main__":
    unittest.main()

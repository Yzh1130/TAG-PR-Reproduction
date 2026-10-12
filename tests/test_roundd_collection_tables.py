"""Handwritten collector fixtures; no run scan, training, or real data writes."""
from __future__ import annotations

import unittest

from scripts.revision._revision_table_format import TableFormatter
from scripts.revision._revision_tables_collection import render_pairs, render_runs


def mock_pairs_documents():
    def row(seed, a, b, delta):
        return dict(pair="base|variant", metric="last", seed=str(seed), status="paired",
                    reason="", A_R1=str(a), B_R1=str(b), delta_R1=str(delta))
    first = row(0, 10, 9, -1)
    second = row(1, 12, 13, 1)
    absent = dict(pair="base|variant", metric="last", seed="2", status="unpaired",
                  reason="B directory absent", A_R1="14", B_R1="", delta_R1="")
    return [("first.csv", [first, absent]), ("second.csv", [first.copy(), second])]


def mock_runs_document():
    present = {
        "directory": "/mock/20261012_e2_nrw_seed0", "directory_name": "20261012_e2_nrw_seed0",
        "exp_name": "e2_nrw_seed0", "seed": 0, "training_git_commit": "b" * 40,
        "batch_size": 320, "itc_block_size": 80, "completed_epochs": 2,
        "epochs": [{"epoch": 1, "actual_iterations": 124}, {"epoch": 2, "actual_iterations": None}],
        "best": {"epoch": 1, "metrics": {"R1": 12.345}},
        "last": {"epoch": 2, "metrics": {"R1": 10.25}},
        "checkpoints": [{"name": "checkpoint_last.pth", "sha256_16": "a" * 16, "bytes": 42}],
        "missing_fields": ["epochs.2.actual_iterations"],
    }
    missing = {"directory_name": "old_run", "directory": "/mock/old_run", "missing_fields": ["seed"]}
    return {"runs": [present, missing]}


class CollectionTableTests(unittest.TestCase):
    def test_pairs_merge_manual_statistics_missing_and_conflict(self):
        documents = mock_pairs_documents()
        first = documents[0][1][0]
        # A/B means are 11; Δ [-1,+1] has mean=0, ddof=0 std=1,
        # range=2, negative/zero/positive=1/0/1. Duplicate seed 0 is deduped.
        fmt = TableFormatter(2)
        output = render_pairs(documents, fmt)
        self.assertIn("| 0 | 10.00 | 9.00 | −1.00 | paired |", output)
        self.assertIn("| 2 | n/a | n/a | n/a | unpaired | B directory absent |", output)
        self.assertIn("| mean | 11.00 | 11.00 | +0.00 |", output)
        self.assertIn("| population std (ddof=0) | 1.00 | 2.00 | 1.00 |", output)
        self.assertIn("| range [min, max] | 2.00 [10.00, 12.00] | 4.00 [9.00, 13.00] | 2.00 [−1.00, +1.00] |", output)
        self.assertIn("−/0/+: 1/0/1", output)
        self.assertTrue(any("B directory absent" in value for value in fmt.missing))
        conflicting = dict(first, delta_R1="-1.00001")
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            render_pairs([("first.csv", [first]), ("conflict.csv", [conflicting])], TableFormatter())
        missing_delta = dict(first, delta_R1="")
        missing_output = render_pairs([("missing.csv", [missing_delta])], TableFormatter())
        self.assertIn("| 0 | 10.00 | 9.00 | n/a |", missing_output)
        self.assertIn("| mean | 10.00 | 9.00 | n/a |", missing_output)
        separate = render_pairs([("last.csv", [first]), ("best.csv", [dict(first, metric="best")])], TableFormatter())
        self.assertIn("(last)", separate)
        self.assertIn("(best)", separate)

    def test_runs_recorded_fields_prefix_filter_and_na(self):
        data = mock_runs_document()
        fmt = TableFormatter(2)
        output = render_runs(data, fmt)
        self.assertIn("| e2_nrw_seed0 | 0 | " + "b" * 40 + " | 320 | 80 | 2 | epoch 1: 124; epoch 2: n/a | 1 | 12.35 | 10.25 |", output)
        self.assertIn("checkpoint_last.pth / aaaaaaaaaaaaaaaa / 42 bytes", output)
        self.assertIn("| /mock/old_run | n/a | n/a |", output)
        self.assertTrue(any("actual_iterations" in value for value in fmt.missing))
        filtered = render_runs(data, TableFormatter(), include=["e2_nrw"])
        self.assertIn("e2_nrw_seed0", filtered)
        self.assertNotIn("old_run", filtered)
        with self.assertRaisesRegex(ValueError, "containing a runs list"):
            render_runs([], TableFormatter())


if __name__ == "__main__":
    unittest.main()

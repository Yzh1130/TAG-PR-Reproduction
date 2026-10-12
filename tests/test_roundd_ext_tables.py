"""Hand-written formatting checks for recorded external/S13 table renderers."""

import copy
import unittest

from scripts.revision._revision_table_format import TableFormatter
from scripts.revision._revision_tables_ext import render_s13, render_table11, render_table12


class ExternalRevisionTablesTests(unittest.TestCase):
    def test_table11_recorded_values_population_summary_and_missing(self):
        entry = {
            "per_seed": [{"seed": 7, "baseline": {"r1": 12.345},
                          "transductive": {"delta": {"r1": 0.16}, "identity_ci95": [-0.42, 0.55]},
                          "inductive": {"delta": {"r1": -0.42}, "identity_ci95": None}}],
            "summary": {"baseline": {"r1": {"mean": 12.345, "std": 0.25, "min": 12, "max": 13}},
                        "transductive": {"delta": {"r1": {"mean": 0.16, "std": 0.10, "min": -0.1, "max": 0.5}},
                                         "sign_counts": {"positive": 2, "zero": 1, "negative": 0, "total": 3}},
                        "inductive": {"delta": {"r1": {"mean": -0.42, "std": 0.2, "min": -0.8, "max": -0.1}},
                                      "sign_counts": {"positive": 0, "zero": 0, "negative": 3, "total": 3}}}}
        fmt = TableFormatter()
        result = render_table11({"datasets": {"cuhkpedes": entry}}, fmt)
        self.assertIn("| CUHK-PEDES | 7 | 12.35 | +0.16 [−0.42, +0.55] | −0.42 [n/a] |", result)
        self.assertIn("| CUHK-PEDES | Population std (ddof=0) | 0.25 | 0.10 | 0.20 |", result)
        self.assertIn("| CUHK-PEDES | Range [min, max] | [12.00, 13.00] | [−0.10, +0.50] | [−0.80, −0.10] |", result)
        self.assertIn("2 / 1 / 0; 3", result)
        self.assertTrue(any("inductive.CI" in reason for reason in fmt.missing))
        self.assertTrue(any("aeri.per_seed" in reason for reason in fmt.missing))
        # Duplicate recorded training seeds must not become additional runs.
        duplicate = copy.deepcopy(entry)
        duplicate["per_seed"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate recorded training seed"):
            render_table11({"datasets": {"cuhkpedes": duplicate}}, TableFormatter())

    def test_table12_actual_training_seed_and_distinct_correlation_fields(self):
        doc = {"aeri": {"fits": {
            "trans_cohortQ_delta": {"slope_vs_log10D": -0.42, "slope_ci95": [-0.5, 0.16],
                                    "spearman_vs_D": -0.7, "spearman_vs_excess_skew": 0.8}},
                        "spearman_excess_skew_vs_D": 0.3},
               "_meta": {"provenance": {"seeds": {"gallery": [2026], "run_seeds": {"aeri": 2}}}}}
        fmt = TableFormatter()
        result = render_table12([("name_seed99.json", doc)], fmt)
        self.assertIn("| AERI-PEDES | 2 | trans_cohortQ_delta | −0.42 [−0.50, +0.16] | −0.70 | 0.30 | 0.80 |", result)
        self.assertIn("| AERI-PEDES | 2 | ind_delta | n/a [n/a] | n/a | 0.30 | n/a |", result)
        self.assertNotIn("| 99 |", result)
        self.assertTrue(any("fits.ind_delta" in reason for reason in fmt.missing))
        with self.assertRaisesRegex(ValueError, "duplicate recorded aeri training seed"):
            render_table12([("first", doc), ("second", doc)], TableFormatter())

    def test_s13_summary_rows_parallel_models_missing_and_known_protocol(self):
        metric = {"R1": 12.3, "R5": 23.4, "R10": 34.5, "mAP": 9.8}
        doc = {"rows": [
            {"setting": "uncorrected", "bank": None, "alpha": None, "k": None,
             "metrics": metric, "delta_R1_pp": 0.0, "identity_bootstrap": None},
            {"setting": "inductive_default", "bank": "train_full", "alpha": 0.5, "k": 10,
             "metrics": metric, "delta_R1_pp": 0.16,
             "identity_bootstrap": {"ci95_percentile_pp": [-0.42, 0.55]}}]}
        fmt = TableFormatter()
        result = render_s13([("L/14", doc), ("B/16", {"rows": []})], fmt)
        self.assertIn("L/14 R@1 (%)", result)
        self.assertIn("B/16 R@1 (%)", result)
        self.assertIn("| train_full (α=0.5, k=10) | 12.30 | 23.40 | 34.50 | 9.80 | +0.16 [−0.42, +0.55] | n/a |", result)
        self.assertIn("| Uncorrected | 12.30 | 23.40 | 34.50 | 9.80 | +0.00 [n/a] |", result)
        self.assertTrue(any("summary row is missing" in reason for reason in fmt.missing))
        broken = copy.deepcopy(doc)
        broken["rows"][1]["alpha"] = 0.25
        with self.assertRaisesRegex(ValueError, "disagrees"):
            render_s13([("wrong", broken)], TableFormatter())


if __name__ == "__main__":
    unittest.main()

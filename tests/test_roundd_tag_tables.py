"""Handwritten producer-schema fixtures for the three TAG table renderers."""

import copy
import unittest

from scripts.revision._revision_table_format import TableFormatter
from scripts.revision._revision_tables_tag import render_r3, render_table3, render_table5


def bootstrap(point=3.0):
    return {"observed_delta_pp": point, "ci95_percentile_pp": [point - 1, point + 1]}


def table5_fixture():
    original = {
        "same": {"metrics": {"R1": 10.0, "mAP": 15.0, "gallery_size_min": 9, "gallery_size_max": 9},
                 "zero_positive_queries": 0, "zero_positive_identities": 0},
        "cross": {"metrics": {"R1": 20.0, "mAP": 25.0, "gallery_size_min": 10, "gallery_size_max": 10},
                  "zero_positive_queries": 0, "zero_positive_identities": 0},
        "identity_bootstrap": bootstrap(10.0),
    }
    trials = [
        {"seed": 2026 + index,
         "same": {"metrics": {"R1": 1.0 if index % 2 == 0 else 3.0, "mAP": 4.0},
                  "zero_positive_queries": 0, "zero_positive_identities": 0},
         "cross": {"metrics": {"R1": 5.0, "mAP": 8.0},
                   "zero_positive_queries": 0, "zero_positive_identities": 0}}
        for index in range(10)
    ]
    rule = {"status": "complete", "mean_of_per_draw_paired_deltas_pp": 3.0,
            "std_of_per_draw_paired_deltas_pp": 1.0,
            "identity_bootstrap_conditional_on_fixed_draws": bootstrap(),
            "two_stage_identity_and_gallery_bootstrap": bootstrap(),
            "per_query_average_policy": "fixed common queries; lost positives count as misses"}
    group = {"n_queries": 20, "n_identities": 5, "effective_controlled_gallery_size": 8,
             "original": original, "authoritative_trials": trials,
             "table5_reference": {"original_cross_minus_same_pp": 10.0, "controlled_cross_minus_same_pp": 3.0},
             "zero_positive_policies": {
                 "a_fixed_misses": rule, "b_common_valid_queries": copy.deepcopy(rule),
                 "c_at_least_two_positives": {"status": "infeasible", "reason": "not enough original images"}}}
    return {"groups": {"A": group, "G": copy.deepcopy(group)},
            "unique_identity_counts": {"n_A": 5, "n_G": 5, "union": 8}}


def estimate(value, lo=None, hi=None):
    return {"estimate": value, "ci95_percentile": None if lo is None else [lo, hi]}


def table3_fixture():
    keys = ("same_source_same_view", "same_source_cross_view", "cross_source_same_view", "cross_source_cross_view")
    cells = {}
    for key, share, prior, ratio in zip(keys, [.5, .25, .25, 0], [.4, .3, .2, .1], [1.25, .8333, 1.25, None]):
        cells[key] = {"observed_share": estimate(share, 0, 1), "expected_share": estimate(prior),
                      "absolute_excess": estimate(share - prior),
                      "enrichment_ratio": estimate(ratio, .5, 2) if ratio is not None else
                      {"estimate": None, "ci95_percentile": None, "reason": "ratio not recorded"}}
    conditional = {"observed": estimate(.75, .5, 1), "expected": estimate(.5),
                   "enrichment_ratio": estimate(1.5, 1, 2), "absolute_excess": estimate(.25)}
    group = {"cells": cells, "same_view_given_same_source": conditional,
             "cross_view_given_same_source": {"observed": estimate(.25, 0, .5), "expected": estimate(.5),
                                              "enrichment_ratio": estimate(.5, 0, 1), "absolute_excess": estimate(-.25)}}
    pure = {"inapplicable_cells": {"same_source_cross_view": "pure-view source; no such negatives"}}
    return {"counts": dict(zip(keys, [2, 1, 1, 0])),
            "strata": {"all": group, "Market": pure, "PRAI-1581": pure, "uavhuman": pure},
            "mixed_source_conditionals": {
                name: {"same_view": conditional, "cross_view": group["cross_view_given_same_source"]}
                for name in ("AG-ReID.v2", "G2APS")},
            "definitions": {key: "recorded mock definition" for key in
                            ("negative_set", "observed", "expected", "conditional_expected", "bootstrap", "small_prior")}}


def cluster(mean, ci, count=4):
    return {"mean": mean, "ci95": ci, "n_queries": count, "n_identities": 2 if count else 0}


def r3_fixture():
    empty = {"quartile": 1, "n_source_images": 0, "n_queries": 0, "metrics": {"R1": None},
             "R1_identity_bootstrap_proportion": cluster(None, [None, None], 0),
             "same_source_same_view_error_share_among_errors": cluster(None, [None, None], 0)}
    nonempty = {"quartile": 4, "n_source_images": 2, "n_queries": 4, "metrics": {"R1": 25.0},
                "R1_identity_bootstrap_proportion": cluster(.25, [.1, .4]),
                "same_source_same_view_error_share_among_errors": cluster(.5, [.25, .75])}
    return {"image_size": {"quartile_basis": "equal-area ties grouped together",
                            "source_quartiles": {"Market": {"groups": [empty, nonempty]}}},
            "attribute_overlap": {
                "error_top1": cluster(.5, [.4, .6]), "error_top1_paired_subset": cluster(.5, [.4, .6]),
                "random_same_source_negative": cluster(.2, [.1, .3]),
                "paired_error_minus_random": cluster(.3, [.1, .5]),
                "queries_with_no_same_source_negative": 0},
            "similarity_margin": {"definition": "error top1 minus best positive",
                                  "distribution": {"count": 3, "mean": .2, "std": .1, "min": 0, "max": .4,
                                                   "quantiles": {"0.5": .2, "0": 0, "1": .4}}},
            "truth_rank": {"distribution": {"count": 3, "mean": 4, "std": 2, "min": 2, "max": 6,
                                            "quantiles": {"0.5": 4, "0": 2, "1": 6}}}}


class TagTableRenderTests(unittest.TestCase):
    def test_table5_six_protocol_rows_recorded_trials_and_infeasible(self):
        fmt = TableFormatter()
        markdown = render_table5(table5_fixture(), fmt, "mock B/16")
        self.assertIn("| A→A | 20 | 9–9 | 8 | 10.00 | 15.00 | 2.00 ± 1.00 | 4.00 ± 0.00 |", markdown)
        self.assertIn("| A→G | 20 | 10–10 | 8 | 20.00 | 25.00 | 5.00 ± 0.00 | 8.00 ± 0.00 |", markdown)
        self.assertIn("+10.00 [+9.00, +11.00]", markdown)
        self.assertIn("+3.00 [+2.00, +4.00]", markdown)
        for name in ("A→A", "A→G", "A paired cross−same", "G→G", "G→A", "G paired cross−same"):
            self.assertEqual(markdown.count("| " + name + " |"), 1)
        self.assertIn("infeasible | n/a | n/a | n/a | not enough original images", markdown)
        self.assertTrue(any("not enough original images" in item for item in fmt.missing))

    def test_table5_does_not_fill_missing_trial(self):
        data = table5_fixture()
        data["groups"]["A"]["authoritative_trials"][0]["same"]["metrics"]["R1"] = None
        fmt = TableFormatter()
        markdown = render_table5(data, fmt)
        self.assertIn("| A→A | 20 | 9–9 | 8 | 10.00 | 15.00 | n/a | 4.00 ± 0.00 |", markdown)
        self.assertTrue(any("ten" in item or "10 recorded" in item for item in fmt.missing))

    def test_table3_manual_shares_ratios_and_inapplicable(self):
        fmt = TableFormatter()
        markdown = render_table3(table3_fixture(), fmt)
        self.assertIn("| Same source / same view | 2 | 50.00 [0.00, 100.00] | 40.00 | 1.25 [0.50, 2.00] | +10.00 |", markdown)
        self.assertIn("| All: P(same view \\| same source) | 75.00 [50.00, 100.00] | 50.00 | 1.50 [1.00, 2.00] | +25.00 |", markdown)
        self.assertIn("−25.00", markdown)
        self.assertIn("| Market | inapplicable | pure-view source; no such negatives |", markdown)
        self.assertIn("n/a [n/a]", markdown)
        self.assertTrue(any("ratio not recorded" in item for item in fmt.missing))

    def test_r3_percent_units_empty_market_group_and_signed_pair(self):
        fmt = TableFormatter()
        markdown = render_r3(r3_fixture(), fmt)
        self.assertIn("| Market | 1 | 0 | 0 | n/a [n/a] | n/a [n/a] |", markdown)
        self.assertIn("| Market | 4 | 2 | 4 | 25.00 [10.00, 40.00] | 50.00 [25.00, 75.00] |", markdown)
        self.assertIn("| Paired error minus random | 4 | 2 | +0.30 [+0.10, +0.50] |", markdown)
        self.assertIn("| Best-positive rank among full gallery | 0.5 | 4.00 |", markdown)
        self.assertTrue(any("null endpoint" in item for item in fmt.missing))

    def test_nonfinite_trial_metric_is_not_formatted_as_missing(self):
        data = table5_fixture()
        data["groups"]["A"]["authoritative_trials"][0]["same"]["metrics"]["R1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            render_table5(data, TableFormatter())


if __name__ == "__main__":
    unittest.main()

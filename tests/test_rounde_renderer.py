"""Handwritten mock evidence for the four new Round-E rendering modes."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.revision import render_revision_tables as RENDER
from scripts.revision._revision_table_format import TableFormatter
from scripts.revision._revision_tables_rounde import (
    collect_e6_inputs, render_e5, render_e6, render_s7d, render_s13_compare,
)


def e5_fixture():
    return {"mock": True, "n_images": 100, "n_gumbel": 3,
            "main_report_block": "visual.block6", "block_order": ["visual.block6"],
            "cam_source_table": {"Market": {"0": 0, "1": 30}, "PRAI-1581": {"0": 30, "1": 0},
                                 "uavhuman": {"0": 0, "1": 0}, "AG-ReID.v2": {"0": 10, "1": 10}, "G2APS": {"0": 10, "1": 10}},
            "blocks": {"visual.block6": {
                "deterministic_argmax": {"accuracy": .75, "class_recall": [.5, 1], "balanced_accuracy": .75},
                "by_source": {"Market": {"accuracy": 1}, "PRAI-1581": {"accuracy": .5}},
                "logit_margin_quantiles": {"0.005": .01, "0.01": .02, "0.05": .3, "0.5": 1.2},
                "gumbel": {"accuracy_mean": .7, "accuracy_std": .1,
                           "mean_per_image_flip_rate": .2, "fraction_images_ever_flipped": .3}}}}


def e6_fixture():
    return {"mock": True, "n_repetitions": 2, "maximum_scope": "n=2 observed mock repetitions only",
            "argmax": {"eval_router_mode": "argmax", "eval_seed": 0,
                       "metrics": {"r1": 20, "r5": 30, "r10": 40, "mAP": 10}},
            "gumbel": [{"eval_router_mode": "gumbel", "eval_seed": 7,
                        "metrics": {"r1": 21, "r5": 30, "r10": 42, "mAP": 9},
                        "minus_argmax_pp": {"r1": 1, "r5": 0, "r10": 2, "mAP": -1}},
                       {"eval_router_mode": "gumbel", "eval_seed": 8,
                        "metrics": {"r1": 19, "r5": 32, "r10": 40, "mAP": None},
                        "minus_argmax_pp": {"r1": -1, "r5": 2, "r10": 0, "mAP": None}}],
            "observed_maximum_pairwise_difference_pp": {"r1": 2, "r5": 2, "r10": 2, "mAP": None}}


def s7d_fixture(seed=7):
    def protocol(r1=20, trans=1, ind=-1):
        return {"ng": 100, "baseline": {"r1": r1, "mAP": r1 / 2}, "excess_skew_N10": r1 / 10,
                "trans_delta_r1": trans, "trans_ci": [-.42, 1.55],
                "ind_delta_r1": ind, "ind_ci": [-1.5, .2]}
    controlled = [protocol(20 + 2 * index, index, index - 2) for index in range(5)]
    pairs = {"ground_minus_official_aerial": {
        "delta_r1_pp": -.42, "identity_ci95": [-1, .16], "nq": 200, "n_identity_clusters": 10,
        "correction_gain_difference": {"transductive": {"delta_gain_r1_pp": .16, "identity_ci95": [-.42, .55]},
                                       "inductive": {"delta_gain_r1_pp": -.2, "identity_ci95": [-.6, .4]}}},
        "size_controlled_aerial_minus_official_aerial": {
            "mean_delta_r1_pp": .25, "identity_ci95_conditional_on_five_draws": [-.3, .9],
            "nq": 200, "n_identity_clusters": 10,
            "correction_gain_difference": {"transductive": {"mean_delta_gain_r1_pp": .1, "identity_ci95_conditional_on_five_draws": [-.2, .3]},
                                           "inductive": {"mean_delta_gain_r1_pp": -.1, "identity_ci95_conditional_on_five_draws": [-.3, .2]}}}}
    return {"mock": True, "_meta": {"provenance": {"seeds": {"run_seed": seed}}},
            "protocols": {"T2A_official": protocol(), "T2G_ground_only": protocol(18), "T2M_mixed": {},
                          "T2A_size_controlled_4220": {"seeds": [2026, 2027, 2028, 2029, 2030], "runs": controlled,
                                                       "mean_r1": 24, "sd_r1": 2.828427124746,
                                                       "mean_trans_delta": 2, "mean_ind_delta": 0}},
            "paired_gallery_differences": pairs}


def s13_fixture():
    return {"mock": True, "rows": [
        {"setting": "uncorrected", "bank": None, "alpha": None, "k": None,
         "metrics": {"R1": 20, "R5": 30, "R10": 40, "mAP": 10}, "delta_R1_pp": 0, "identity_bootstrap": None},
        {"setting": "inductive_default", "bank": "train_full", "alpha": .5, "k": 10,
         "metrics": {"R1": 20.16, "R5": 30, "R10": 40, "mAP": 10}, "delta_R1_pp": .16,
         "identity_bootstrap": {"ci95_percentile_pp": [-.42, .55]}},
        {"setting": "transductive_k10", "bank": "test_full", "alpha": .5, "k": 10,
         "metrics": {"R1": 19.58, "R5": 30, "R10": 40, "mAP": 10}, "delta_R1_pp": -.42,
         "identity_bootstrap": None}]}


class RoundERendererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.out = self.root / "new_tables"

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, data):
        path = self.inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def run_table(self, command, flags):
        before = {path: path.read_bytes() for path in self.inputs.rglob("*.json")}
        with contextlib.redirect_stdout(io.StringIO()):
            result = RENDER.main([command, *flags, "--out", str(self.out)])
        text = Path(result["table"]).read_text(encoding="utf-8")
        self.assertIn("Provenance: inputs:", text)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        for source in result["provenance"]["inputs"]:
            self.assertEqual(source["sha256_16"], hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()[:16])
            self.assertIn(source["sha256_16"], text)
        self.assertIn(text, Path(result["all_tables"]).read_text(encoding="utf-8"))
        return text, result

    def test_table_e5_cli_handwritten_values_and_missing_sources(self):
        path = self.write_json("mock_router.json", e5_fixture())
        text, result = self.run_table("table-e5", ["--json", f"Mock B/16={path}"])
        self.assertIn("| Mock B/16 | visual.block6 (main report) | 100 | 75.00 | 50.00 | 100.00 | 75.00 | 3 | 70.00 ± 10.00 | 20.00 | 30.00 | 0.01 | 0.02 | 0.30 | 1.20 |", text)
        self.assertIn("| Mock B/16 | Market | 0 | 30 | PASS |", text)
        self.assertIn("| Mock B/16 | uavhuman | 0 | 0 | not present / not asserted |", text)
        self.assertIn("| Mock B/16 | visual.block6 (main report) | AG-ReID.v2 | n/a |", text)
        self.assertTrue(any("by_source.AG-ReID.v2" in reason for reason in result["provenance"]["missing_fields"]))
        broken = e5_fixture()
        broken["cam_source_table"]["Market"]["0"] = 1
        self.assertIn("| Mock | Market | 1 | 30 | FAIL |", render_e5([("Mock", broken)], TableFormatter()))

    def test_table_e6_cli_exact_summary_filename_seed_and_null(self):
        path = self.write_json("e6/mock_seed99/router_randomness.json", e6_fixture())
        text, result = self.run_table("table-e6", ["--root", f"Mock L/14={path.parent}"])
        self.assertIn("| Mock L/14 | gumbel | 7 | 21.00 | 30.00 | 42.00 | 9.00 | +1.00 | +0.00 | +2.00 | −1.00 |", text)
        self.assertIn("| Mock L/14 | gumbel | 8 | 19.00 | 32.00 | 40.00 | n/a | −1.00 | +2.00 | +0.00 | n/a |", text)
        self.assertIn("| Mock L/14 | 2 | 2.00 | 2.00 | 2.00 | n/a | n=2 observed mock repetitions only |", text)
        self.assertIn("这 N 次中观测到的最大差异", text)
        self.assertNotIn("| 99 |", text)
        self.assertEqual(collect_e6_inputs(path.parent), path)
        with self.assertRaisesRegex(ValueError, "exact E6 RUN_ROOT"):
            collect_e6_inputs(path.parent.parent)
        # Do not reconstruct a missing saved delta from complete metrics.
        data = e6_fixture()
        del data["gumbel"][0]["minus_argmax_pp"]["r1"]
        self.assertIn("| 9.00 | n/a | +0.00", render_e6([("Mock", data)], TableFormatter()))
        data["n_repetitions"] = 3
        with self.assertRaisesRegex(ValueError, "n_repetitions"):
            render_e6([("Mock", data)], TableFormatter())

    def test_table_s7d_cli_five_draws_and_paired_saved_cis(self):
        path = self.write_json("mock_s7d_seed99.json", s7d_fixture())
        text, result = self.run_table("table-s7d", ["--s7d-json", f"Mock={path}"])
        self.assertIn("| Mock | 7 | Official aerial | 100 | 20.00 | 10.00 | 2.00 | +1.00 [−0.42, +1.55] | −1.00 [−1.50, +0.20] |", text)
        self.assertIn("| Mock | 7 | Controlled aerial (5 gallery draws) | 100 | 24.00 ± 2.83 | 12.00 ± 1.41 | 2.40 ± 0.28 | +2.00 ± 1.41 [n/a] | +0.00 ± 1.41 [n/a] |", text)
        self.assertIn("| Mock | 7 | Ground − official aerial | −0.42 [−1.00, +0.16] | +0.16 [−0.42, +0.55] | −0.20 [−0.60, +0.40] | 200 | 10 | identity resampling |", text)
        self.assertIn("conditional on five gallery draws", text)
        self.assertEqual(sum(text.count(f"| Mock | 7 | {seed} |") for seed in (2026, 2027, 2028, 2029, 2030)), 5)
        self.assertIn("| Mock | 7 | Mixed | n/a | n/a | n/a | n/a | n/a [n/a] | n/a [n/a] |", text)
        self.assertTrue(any("CIs are not averaged" in value for value in result["provenance"]["missing_fields"]))
        second = self.write_json("mock_s7d_second.json", s7d_fixture(8))
        combined, _ = self.run_table("table-s7d", ["--s7d-json", f"Mock={path}", "--s7d-json", f"Mock={second}"])
        self.assertIn("| Mock | 8 | Official aerial |", combined)
        with self.assertRaisesRegex(ValueError, "duplicate model/training seed"):
            render_s7d([("Mock", s7d_fixture()), ("Mock", s7d_fixture())], TableFormatter())
        missing = s7d_fixture()
        missing["protocols"]["T2A_size_controlled_4220"]["runs"][0]["baseline"]["mAP"] = None
        self.assertIn("24.00 ± 2.83 | n/a ± n/a |", render_s7d([("Mock", missing)], TableFormatter()))

    def test_s13_compare_cli_models_are_rows_and_default_is_unchanged(self):
        path = self.write_json("mock_s13.json", s13_fixture())
        empty = self.write_json("mock_s13_empty.json", {"mock": True, "rows": []})
        text, result = self.run_table("s13", ["--summary", f"Mock L/14={path}", "--summary", f"Mock B/16={empty}", "--compare"])
        self.assertIn("| Mock L/14 | 20.00 | +0.00 [n/a] | 20.16 | +0.16 [−0.42, +0.55] | n/a | n/a [n/a] | 19.58 | −0.42 [n/a] |", text)
        self.assertIn("| Mock B/16 | n/a | n/a [n/a] |", text)
        self.assertNotIn("R@5 (%)", text)
        legacy, _ = self.run_table("s13", ["--summary", f"Mock L/14={path}"])
        self.assertIn("Mock L/14 R@5 (%)", legacy)
        self.assertIn("| Uncorrected |", legacy)
        bad = copy.deepcopy(s13_fixture())
        bad["rows"][1]["alpha"] = .25
        with self.assertRaisesRegex(ValueError, "setting differs"):
            render_s13_compare([("Mock", bad)], TableFormatter())

    def test_missing_documents_produce_na_not_inferred_values(self):
        for renderer in (render_e5, render_e6, render_s7d, render_s13_compare):
            fmt = TableFormatter()
            text = renderer([("Mock", {})], fmt)
            self.assertIn("n/a", text)
            self.assertTrue(fmt.missing)
        for renderer in (render_e5, render_e6, render_s13_compare):
            with self.assertRaisesRegex(ValueError, "labels must be unique"):
                renderer([("Mock", {}), ("Mock", {})], TableFormatter())


if __name__ == "__main__":
    unittest.main()

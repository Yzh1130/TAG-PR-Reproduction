"""200x100 synthetic regression for S14; no real retrieval numbers invented."""

import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import S14_identity_stats as s14
from . import S11_offline_stats as s11
from ._offline_common import compact_metrics, evaluate, json_ready, topk_mean
from ._s11_groups import balanced_group_halves, _crossfit_fold
from ._s11_descriptive import error_decomposition
from .test_S11_S12 import mock_data, reference_metrics
from tools import S9_error_stats as s9


class MemoryAudit:
    def __init__(self):
        self.report = {"checks": []}
        self.files = {}

    def check(self, name, actual, expected):
        row = dict(name=name, actual=json_ready(actual), expected=json_ready(expected), passed=actual == expected)
        self.report["checks"].append(row)
        if not row["passed"]:
            raise RuntimeError(f"exact check failed {name}: {actual!r} != {expected!r}")

    def write_json(self, name, payload):
        self.files[name] = json_ready(payload)
        return Path("_e1_s14") / name


def authority_fixture():
    records, ann, sim, _ = mock_data()
    raw, penalty = evaluate(sim, ann), topk_mean(sim, 10)
    halves = balanced_group_halves(ann.query_ids, seed=2026)
    with contextlib.redirect_stdout(io.StringIO()):
        folds = [_crossfit_fold(sim, ann, halves[f], halves[1-f], penalty, 10, "identity_group", f)[0]
                 for f in (0, 1)]
        controlled = s11.run_controlled(sim, ann, "common-min", bootstrap=2)
    with mock.patch.object(s9, "BOOTSTRAP_REPEATS", 2):
        prior = s9.analyze_errors(raw["details"]["top1"], ann)
    original = dict(raw=compact_metrics(raw), A=controlled,
                    B_error_decomposition=dict(raw=error_decomposition(ann, raw)),
                    F=dict(k=10, split_seed=2026, schemes=dict(identity_group=dict(folds=folds))),
                    seeds=dict(gallery=list(range(2026, 2036)), scans=[2026, 2027, 2028]))
    return records, ann, sim, raw, json_ready(original), json_ready(prior)


class S14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.ann, cls.sim, cls.raw, cls.s11, cls.s9 = authority_fixture()

    def test_replay_and_new_eval_gallery_against_independent_reference(self):
        audit = MemoryAudit()
        with contextlib.redirect_stdout(io.StringIO()):
            halves, penalties, trace = s14.restore_identity_folds(self.sim, self.ann, self.s11, audit)
            report = s14.identity_gallery_crossfit(self.sim, self.ann, self.s11, halves, penalties, audit,
                                                   scan_sizes=(10, 50, 500))
        self.assertIn("identity_folds_seed2026.json", audit.files)
        self.assertEqual(len(trace["folds"]), 2)
        for f, row in enumerate(report["folds"]):
            cal, queries = halves[f], halves[1-f]
            gallery = np.asarray(row["gallery_indices"])
            self.assertEqual(np.intersect1d(self.ann.query_source_indices[cal], gallery).size, 0)
            self.assertEqual(np.intersect1d(self.ann.query_ids[cal], self.ann.gallery_ids[gallery]).size, 0)
            choices = [("uncorrected", None), ("crossfit_csls", penalties[f]),
                       ("evaluation_query_transductive_csls", topk_mean(self.sim, 10, rows=queries))]
            for name, penalty in choices:
                hits, aps, _, _ = reference_metrics(self.sim, self.ann, queries, gallery, penalty=penalty, alpha=0.5)
                reference = float(np.float32(hits[:, 0].sum()) * np.float32(100) / np.float32(len(hits)))
                self.assertEqual(row["metrics"][name]["R1"], reference)
                self.assertAlmostEqual(row["metrics"][name]["mAP"], aps.mean() * 100, places=13)
            self.assertEqual(row["calibration_scan"][2]["status"], "unavailable")
            # Description-level rather than whole-identity sampling: exactly
            # the requested number is used (no group budget overshoot).
            self.assertEqual(row["calibration_scan"][0]["repetitions"][0]["calibration_texts"], 10)
            for trial in row["calibration_scan"][-1]["repetitions"]:
                self.assertEqual(trial["metrics"]["R1"], row["metrics"]["crossfit_csls"]["R1"])
        print("S14 MOCK A: shape=200x100; two disjoint identity folds; raw/crossfit/transductive independent-reference PASS")

    def test_each_fold_gate_is_exact_and_stops_before_trace(self):
        cases = [(0, "calibration", "images"), (1, "evaluation", "identities"),
                 (0, "metrics", "uncorrected"), (1, "metrics", "crossfit_csls")]
        for fold, section, key in cases:
            bad = copy.deepcopy(self.s11)
            target = bad["F"]["schemes"]["identity_group"]["folds"][fold]
            if section == "metrics":
                target[section][key]["R1"] = np.nextafter(target[section][key]["R1"], np.inf)
            else:
                target[section][key] += 1
            audit = MemoryAudit()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                s14.restore_identity_folds(self.sim, self.ann, bad, audit)
            self.assertEqual(audit.files, {})

    def test_B_exact_point_prior_and_identity_query_bootstrap(self):
        audit = MemoryAudit()
        with contextlib.redirect_stdout(io.StringIO()):
            result = s14.enrichment_statistics(self.ann, self.raw, self.s9, audit, repeats=10000)
        for name, row in result["groups"].items():
            self.assertEqual(row["identity"]["n_units"], len(np.unique(self.ann.query_ids)))
            for key, value in row["point"].items():
                self.assertEqual(value, self.s9["groups"][name][key])
            # Brute force prior using exclusion masks over all true-ID images.
            selected = [x for x in self.s9["error_query_audit"] if x["view"] == (0 if name == "aerial" else 1)]
            for item in selected:
                valid = self.ann.gallery_ids != item["identity"]
                expected = np.mean(self.ann.gallery_domains[valid] == item["source_domain"])
                self.assertEqual(item["conditional_prior"], expected)
            with mock.patch.object(s9, "BOOTSTRAP_REPEATS", 10000):
                baseline_interval, invalid = s9.paired_bootstrap_enrichment(
                    [x["same_source"] for x in selected], [x["conditional_prior"] for x in selected])
            self.assertEqual(invalid, 0)
            self.assertEqual(row["query"]["results"]["enrichment_ratio"]["ci95_percentile"], baseline_interval)
        print("S14 MOCK B: S9 point estimates EXACT; 10000 query/identity repeats; prior excludes all true-ID images")

    def test_B_tampered_definition_or_point_stops(self):
        for field in ("definition", "point"):
            bad = copy.deepcopy(self.s9)
            if field == "definition":
                bad["enrichment_definition"] = "mean over all queries"
            else:
                bad["groups"]["aerial"]["mean_conditional_prior"] = np.nextafter(
                    bad["groups"]["aerial"]["mean_conditional_prior"], np.inf)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                s14.enrichment_statistics(self.ann, self.raw, bad, MemoryAudit(), repeats=2)

    def test_C_four_counts_and_conditionals(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = s14.category_statistics(self.ann, self.raw, self.s11, MemoryAudit(), repeats=10000)
        expected = self.s11["B_error_decomposition"]["raw"]["strata"]["all"]
        self.assertEqual(sum(result["counts"].values()), result["n_errors"])
        for key, value in result["identity"]["results"].items():
            original = expected["cells" if key in s14.CELLS else "conditional_probabilities"][key]
            self.assertEqual(value["observed"], original["proportion"])
            self.assertIsNotNone(value["ci95_percentile"])
        bad = copy.deepcopy(self.s11)
        bad["B_error_decomposition"]["raw"]["strata"]["all"]["cells"][s14.CELLS[0]]["count"] += 1
        with self.assertRaises(RuntimeError):
            s14.category_statistics(self.ann, self.raw, bad, MemoryAudit(), repeats=2)
        print("S14 MOCK C: four-cell count conservation PASS; six identity-bootstrap intervals finite")

    def test_D_saved_indices_and_seed_replay_equal_without_mcnemar(self):
        with contextlib.redirect_stdout(io.StringIO()), mock.patch("scripts._s11_descriptive.mcnemar_exact",
                                                                  side_effect=AssertionError("forbidden")):
            saved = s14.controlled_identity_statistics(self.sim, self.ann, self.s11, MemoryAudit(), repeats=10000)
            no_indices = copy.deepcopy(self.s11)
            for control in no_indices["A"]["gallery_control"].values():
                for trial in control["trials"]:
                    trial.pop("same_gallery_indices")
                    trial.pop("cross_gallery_indices")
            replayed = s14.controlled_identity_statistics(self.sim, self.ann, no_indices, MemoryAudit(), repeats=10000)
        self.assertFalse(saved["mcnemar_computed"])
        for name in ("A", "G"):
            left, right = saved["groups"][name], replayed["groups"][name]
            self.assertEqual(left["original"], right["original"])
            for a, b in zip(left["controlled"]["trials"], right["controlled"]["trials"]):
                for key in ("same_gallery_indices", "cross_gallery_indices", "identity_ci"):
                    self.assertEqual(a[key], b[key])
                self.assertNotIn("mcnemar", a["identity_ci"])
                self.assertEqual(a["identity_ci"]["unit"], "cluster")
        print("S14 MOCK D: 20 saved-index draws match authorized seed replay EXACTLY; identity paired CI; no McNemar")

    def test_D_tampered_gallery_or_score_stops_before_any_bootstrap(self):
        for kind in ("indices", "metric", "common"):
            bad = copy.deepcopy(self.s11)
            if kind == "indices":
                trial = bad["A"]["gallery_control"]["A"]["trials"][0]
                trial["same_gallery_indices"] = trial["same_gallery_indices"][::-1]
            elif kind == "metric":
                value = bad["A"]["gallery_control"]["G"]["trials"][-1]["cross"]
                value["R1"] = np.nextafter(value["R1"], np.inf)
            else:
                bad["A"]["common_query_sets"]["A"]["query_indices"].reverse()
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(s14, "paired_test",
                                                                              side_effect=AssertionError("must not reach CI")):
                with self.assertRaises(RuntimeError):
                    s14.controlled_identity_statistics(self.sim, self.ann, bad, MemoryAudit(), repeats=2)

    def test_cluster_bootstrap_includes_zero_error_identities_and_undefined_draws(self):
        num = np.array([[1.0], [0.0], [1.0]])
        den = np.ones_like(num)
        groups, universe = np.array([10, 10, 20]), np.array([10, 20, 30])
        result = s14.ratio_bootstrap(num, den, groups, universe, repeats=100, seed=2026)
        rng = np.random.default_rng(2026)
        draw = rng.integers(0, 3, size=(100, 3), dtype=np.int64)
        ns = np.array([1, 1, 0])[draw].sum(1)
        ds = np.array([2, 1, 0])[draw].sum(1)
        self.assertEqual(result["n_units"], 3)
        self.assertEqual(result["results"][0]["undefined_replicates"], int((ds == 0).sum()))
        if (ds == 0).any():
            self.assertIsNone(result["results"][0]["ci95_percentile"])
        else:
            np.testing.assert_array_equal(result["results"][0]["ci95_percentile"], np.percentile(ns / ds, [2.5, 97.5]))
        for bad_num, bad_den in ((np.ones(2), np.ones(2)), (np.array([[np.nan]]), np.ones((1, 1))),
                                 (np.ones((1, 1)), -np.ones((1, 1)))):
            with self.assertRaises(ValueError):
                s14.ratio_bootstrap(bad_num, bad_den)

    def test_real_cli_all_sections_and_stop_report_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="s14_200x100_mock_") as directory:
            root = Path(directory)
            data = root / "inputs"
            data.mkdir()
            paths = [data / "sim.npy", data / "ann.json", data / "S11.json", data / "S9.json"]
            np.save(paths[0], self.sim)
            for path, value in zip(paths[1:], (self.records, self.s11, self.s9)):
                path.write_text(json.dumps(value), encoding="utf-8")
            before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            out = root / "_e1_s14"
            command = ["--sim", str(paths[0]), "--anno", str(paths[1]), "--s11-json", str(paths[2]),
                       "--s9-json", str(paths[3]), "--out", str(out), "--bootstrap", "10000"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(s14.main(command), 0)
            initial = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
            self.assertIn("identity_folds_seed2026.json", initial)
            reports = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*.json") if p.name.startswith("S14")]
            self.assertEqual(len(reports), 1)
            report = reports[0]
            self.assertEqual(report["status"], "complete")
            self.assertEqual(len(report["inputs"]), 4)
            self.assertEqual(len(report["git_HEAD"]), 40)
            self.assertIn("argv", report)
            for key in ("A_identity_gallery_crossfit", "B_source_enrichment", "C_error_categories", "D_controlled_paired"):
                self.assertIn(key, report)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(s14.main(command), 0)
            for name, content in initial.items():
                self.assertEqual((out / name).read_bytes(), content)
            self.assertEqual(len(list(out.glob("identity_folds_seed2026*.json"))), 2)
            self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            bad = copy.deepcopy(self.s11)
            bad["raw"]["R1"] = np.nextafter(bad["raw"]["R1"], np.inf)
            bad_path = data / "bad_s11.json"
            bad_path.write_text(json.dumps(bad), encoding="utf-8")
            bad_cmd = command.copy()
            bad_cmd[bad_cmd.index("--s11-json") + 1] = str(bad_path)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(s14.main(bad_cmd), 2)
            stopped = [json.loads(p.read_text(encoding="utf-8")) for p in out.glob("S14*.json")]
            self.assertEqual(sum(x["status"] == "stopped" for x in stopped), 1)
            print("S14 MOCK CLI: all A/B/C/D complete; 4 SHA256 audits; repeat creates timestamped files; tampered raw -> STOP")


if __name__ == "__main__":
    unittest.main()

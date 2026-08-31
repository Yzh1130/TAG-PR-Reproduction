"""S13 CPU mock regression tests: python -m unittest scripts.test_S13_bootstrap -v."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import S13_inductive_bootstrap as s13
from ._offline_common import annotation_from_records, evaluate


def fixture(n_gallery=100):
    records = []
    for image in range(n_gallery):
        records.append({"file_path": f"G2APS/mock_{image}.jpg", "id": image // 5,
                        "cam_id": image % 2, "captions": [f"person {image} a", f"person {image} b"]})
    ann = annotation_from_records(records)
    rng = np.random.default_rng(2026)
    sim = rng.normal(0, 0.16, (2 * n_gallery, n_gallery)).astype(np.float32)
    sim += (ann.query_ids[:, None] == ann.gallery_ids[None, :]) * np.float32(0.18)
    sim[:, 0] += np.float32(0.38)
    bank = rng.normal(0, 0.16, (3 * n_gallery, n_gallery)).astype(np.float32)
    bank[:, 0] += np.float32(0.38)
    return records, ann, sim, bank


def manual_bootstrap(first, second, groups, repeats=10000, seed=2026):
    difference = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    if groups is None:
        unit_difference = difference
        counts = np.ones(len(difference), dtype=np.int64)
    else:
        _, inverse = np.unique(groups, return_inverse=True)
        unit_difference = np.bincount(inverse, weights=difference)
        counts = np.bincount(inverse)
    draw = np.random.default_rng(seed).integers(0, len(counts), (repeats, len(counts)), dtype=np.int64)
    return 100.0 * unit_difference[draw].sum(axis=1) / counts[draw].sum(axis=1)


class S13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.ann, cls.sim, cls.bank = fixture()
        cls.report = s13.run_analysis(cls.sim, cls.bank, cls.ann, progress=None)

    def test_penalties_estimated_once_and_exact_s12_formula(self):
        with mock.patch.object(s13, "topk_values", wraps=s13.topk_values) as topk:
            report = s13.run_analysis(self.sim, self.bank, self.ann, bootstrap=50, progress=None)
        self.assertEqual(topk.call_count, 2)
        self.assertIs(topk.call_args_list[0].args[0], self.bank)
        self.assertEqual(topk.call_args_list[0].args[1], 30)
        self.assertIs(topk.call_args_list[1].args[0], self.sim)
        self.assertEqual(topk.call_args_list[1].args[1], 10)
        for label, bank_name, alpha, k in s13.CONFIGURATIONS:
            values = self.sim if bank_name == "test_full" else self.bank
            penalty = np.sort(values, axis=0)[-k:][::-1].mean(axis=0, dtype=np.float32)
            expected = evaluate(self.sim - np.float32(alpha) * penalty[None, :], self.ann)
            got = report["configurations"][label]
            for key in ("R1", "R5", "R10", "mAP"):
                self.assertEqual(got["metrics"][key], expected[key])
            self.assertEqual(got["penalty"]["sha256_native_array_bytes"], hashlib.sha256(penalty.tobytes()).hexdigest())
            self.assertTrue(got["penalty"]["fixed_during_bootstrap"])

    def test_bootstrap_matches_independent_shared_draws_for_all_units(self):
        raw = evaluate(self.sim, self.ann)["details"]["correct"]
        outcomes = {}
        for label, bank_name, alpha, k in s13.CONFIGURATIONS:
            values = self.sim if bank_name == "test_full" else self.bank
            penalty = np.sort(values, axis=0)[-k:][::-1].mean(axis=0, dtype=np.float32)
            outcomes[label] = evaluate(self.sim, self.ann, penalty=penalty, alpha=alpha)["details"]["correct"]
        for unit, groups in (("query", None), ("image", self.ann.query_source_indices),
                             ("identity", self.ann.query_ids)):
            draws = {}
            for label in ("a", "b", "c"):
                expected = manual_bootstrap(raw, outcomes[label], groups)
                draws[label] = expected
                got = self.report["configurations"][label]["bootstrap"][unit]
                self.assertAlmostEqual(got["bootstrap_mean_delta_pp"], expected.mean(), places=13)
                np.testing.assert_allclose(got["ci95_percentile_pp"], np.percentile(expected, [2.5, 97.5]), atol=1e-13, rtol=0)
                p = min(1.0, 2.0 * min((np.count_nonzero(expected <= 0) + 1) / 10001,
                                      (np.count_nonzero(expected >= 0) + 1) / 10001))
                self.assertEqual(got["p_two_sided"], p)
            direct = manual_bootstrap(outcomes["c"], outcomes["a"], groups)
            np.testing.assert_allclose(direct, draws["a"] - draws["c"], atol=1e-13, rtol=0)
            got = self.report["gain_difference_a_minus_c"]["bootstrap"][unit]
            self.assertAlmostEqual(got["bootstrap_mean_delta_pp"], direct.mean(), places=13)
            np.testing.assert_allclose(got["ci95_percentile_pp"], np.percentile(direct, [2.5, 97.5]), atol=1e-13, rtol=0)
        print("S13 paired bootstrap: all 12 estimates match independent draws; paired a-c draw error <= 1e-13 pp")

    def test_20x10_reference_metrics_and_source_exclusion(self):
        _, ann, sim, _ = fixture(10)
        for exclude in (False, True):
            result = evaluate(sim, ann, exclude_source=exclude)
            hits, aps, top1 = [], [], []
            for q in range(20):
                pool = np.arange(10)
                if exclude:
                    pool = pool[pool != ann.query_source_indices[q]]
                order = np.argsort(-sim[q, pool], kind="stable")
                matches = ann.gallery_ids[pool[order]] == ann.query_ids[q]
                hits.append([matches[:k].any() for k in (1, 5, 10)])
                aps.append(np.sum(matches.cumsum() / np.arange(1, len(pool) + 1) * matches) / matches.sum())
                top1.append(pool[order[0]])
            np.testing.assert_array_equal(result["details"]["top1"], top1)
            for index, key in enumerate(("R1", "R5", "R10")):
                self.assertEqual(result[key], float(np.float32(np.asarray(hits)[:, index].sum()) * np.float32(100) / np.float32(20)))
            self.assertAlmostEqual(result["mAP"], np.mean(aps) * 100, places=13)
            if exclude:
                self.assertTrue(np.all(result["details"]["top1"] != ann.query_source_indices))
        print("S13 20x10 metrics: stable-sort reference exact; source exclusion mask exact")

    def test_finite_outside_cosine_range_is_not_clipped(self):
        bank = self.bank * np.float32(4)
        before = bank.copy()
        report = s13.run_analysis(self.sim, bank, self.ann, bootstrap=20, progress=None)
        np.testing.assert_array_equal(bank, before)
        self.assertGreater(report["training_bank_gallery"]["max"], 1)
        self.assertFalse(report["training_bank_gallery"]["clipped"])
        self.assertTrue(any("without clipping" in item for item in report["warnings"]))
        invalid = bank.copy()
        invalid[13, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/infinity"):
            s13.run_analysis(self.sim, invalid, self.ann, bootstrap=20, progress=None)
        with self.assertRaisesRegex(ValueError, "gallery columns"):
            s13.run_analysis(self.sim, bank[:, :-1], self.ann, bootstrap=20, progress=None)

    def test_invalid_units_and_configuration_sizes(self):
        for options in ({"bootstrap": 0}, {"block_size": 0}, {"seed": -1}):
            with self.assertRaises(ValueError):
                s13.run_analysis(self.sim, self.bank, self.ann, progress=None, **options)
        with self.assertRaisesRegex(ValueError, ">=30"):
            s13.run_analysis(self.sim, self.bank[:29], self.ann, progress=None)
        parsed = s13.build_parser().parse_args(["--sim-qg", "s.npy", "--sim-bank-g", "b.npy", "--ann", "a.json", "--out", "out"])
        self.assertEqual(parsed.bootstrap, 10000)
        self.assertEqual(parsed.seed, 2026)
        self.assertEqual(parsed.sim, "s.npy")

    def test_real_cli_help_report_and_immutable_inputs(self):
        command = [sys.executable, str(Path(s13.__file__))]
        help_result = subprocess.run(command + ["--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for flag in ("--sim-qg", "--sim-bank-g", "--ann", "--bootstrap", "--seed"):
            self.assertIn(flag, help_result.stdout)
        with tempfile.TemporaryDirectory(prefix="tagpr_s13_") as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            sim_path, bank_path, ann_path = inputs / "sim.npy", inputs / "bank.npy", inputs / "ann.json"
            np.save(sim_path, self.sim)
            np.save(bank_path, self.bank)
            ann_path.write_text(json.dumps(self.records), encoding="utf-8")
            paths = [sim_path, bank_path, ann_path]
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            args = ["--sim-qg", str(sim_path), "--sim-bank-g", str(bank_path), "--ann", str(ann_path),
                    "--out", str(root / "out"), "--tag", "mock"]
            completed = subprocess.run(command + args, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            reports = list((root / "out").glob("S13_mock_*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete_with_warnings")
            self.assertFalse(report["pure_query_independent_inductive_pipeline"])
            self.assertEqual(report["bootstrap_protocol"]["repeats"], 10000)
            self.assertEqual(report["configurations"], self.report["configurations"])
            self.assertEqual(report["gain_difference_a_minus_c"], self.report["gain_difference_a_minus_c"])
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
            print("S13 CLI MOCK 200x100 qg, 300x100 bank; 10000 repeats; inputs unchanged; --help PASS")
            print(completed.stdout)
            forbidden = subprocess.run(command + args + ["--out", str(inputs / "out")], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(forbidden.returncode, 0)
            self.assertIn("read-only input archive", forbidden.stderr)
            self.assertFalse((inputs / "out").exists())
            wrong = subprocess.run(command + args + ["--bank", str(inputs / "missing.npy")], capture_output=True, text=True, timeout=60)
            self.assertEqual(wrong.returncode, 1)
            self.assertIn("cannot read NumPy matrix", wrong.stdout)
            self.assertNotIn("Traceback", wrong.stdout + wrong.stderr)


if __name__ == "__main__":
    unittest.main()

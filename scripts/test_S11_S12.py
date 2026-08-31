"""New-only regression tests: python -m unittest scripts.test_S11_S12 -v."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from . import S11_offline_stats as s11
from . import S12_inductive_bank as s12
from ._offline_common import (annotation_from_records, evaluate, json_ready, load_similarity,
                              safe_output_dir, topk_mean, write_report)
from ._s11_descriptive import mcnemar_exact, paired_test, reciprocal_statistics
from ._s11_groups import balanced_group_halves, random_identity_halves, run_grouped, sample_identity_groups


def mock_data():
    records = []
    domains = ("AG-ReID.v2", "G2APS", "Market", "PRAI-1581", "uavhuman")
    for pid in range(20):
        domain = domains[pid // 4]
        for j in range(5):
            view = int(j >= 3) if pid < 8 else (1 if domain == "Market" else 0)
            records.append(dict(file_path=f"{domain}/mock_{pid}_{j}.jpg", id=pid, cam_id=view,
                                captions=[f"person {pid} caption {j}a", f"person {pid} caption {j}b"], split="test"))
    ann = annotation_from_records(records)
    rng = np.random.default_rng(2026)
    sim = rng.normal(0, 0.16, (200, 100)).astype(np.float32)
    sim += (ann.query_ids[:, None] == ann.gallery_ids[None, :]) * np.float32(0.18)
    sim[:, 0] += np.float32(0.38)
    sim = np.clip(sim, -1, 1)
    features = rng.normal(size=(100, 17)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    gg = features @ features.T
    return records, ann, sim, gg


def reference_metrics(sim, ann, queries=None, gallery=None, exclude=False, penalty=None, alpha=1):
    q = np.arange(len(ann.query_ids)) if queries is None else np.asarray(queries)
    g = np.arange(len(ann.gallery_ids)) if gallery is None else np.sort(np.asarray(gallery))
    hits, aps, predictions, kept = [], [], [], []
    for i in q:
        pool = g[g != ann.query_source_indices[i]] if exclude else g
        positives = ann.gallery_ids[pool] == ann.query_ids[i]
        if not positives.any():
            continue
        scores = sim[i, pool] if penalty is None else sim[i, pool] - alpha * penalty[pool]
        order = np.argsort(-scores, kind="stable")
        matches = positives[order]
        hits.append([matches[:k].any() for k in (1, 5, 10)])
        aps.append(np.sum(matches.cumsum() / np.arange(1, len(pool) + 1) * matches) / matches.sum())
        predictions.append(pool[order[0]])
        kept.append(i)
    return np.asarray(hits), np.asarray(aps), np.asarray(predictions), np.asarray(kept)


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.ann, cls.sim, cls.gg = mock_data()

    def test_metrics_reference_ties_exclusion_and_float64(self):
        penalty = np.random.default_rng(2027).normal(0, 0.2, 100)
        for sim in (self.sim, np.zeros_like(self.sim)):
            for exclude in (False, True):
                for p in (None, penalty):
                    got = evaluate(sim, self.ann, exclude_source=exclude, penalty=p, alpha=0.5)
                    hits, ap, predictions, kept = reference_metrics(sim, self.ann, exclude=exclude, penalty=p, alpha=0.5)
                    np.testing.assert_array_equal(got["details"]["top1"], predictions)
                    np.testing.assert_array_equal(got["details"]["query_indices"], kept)
                    np.testing.assert_allclose(got["details"]["ap"], ap, atol=2e-16, rtol=0)
                    for j, key in enumerate(("R1", "R5", "R10")):
                        expected = float(np.float32(hits[:, j].sum()) * np.float32(100) / np.float32(len(hits)))
                        self.assertEqual(got[key], expected)
        # >16 positives/query exercises the full-sort fallback too.
        from dataclasses import replace
        all_positive = replace(self.ann, gallery_ids=np.zeros(100, np.int64), query_ids=np.zeros(200, np.int64))
        all_result = evaluate(self.sim, all_positive, exclude_source=True)
        self.assertEqual(all_result["R1"], 100)
        self.assertEqual(all_result["mAP"], 100)

    def test_common_queries_and_size_matched_galleries(self):
        sets, _, counts = s11.common_query_sets(self.ann)
        self.assertEqual(len(sets["A"]), 48)
        self.assertEqual(len(sets["G"]), 32)
        with contextlib.redirect_stdout(io.StringIO()):
            result = s11.run_controlled(self.sim, self.ann, mode="common-min", bootstrap=100)
        for group, c in result["gallery_control"].items():
            self.assertEqual(c["status"], "complete")
            self.assertEqual(len(c["trials"]), 10)
            for trial in c["trials"]:
                for key in ("same", "cross"):
                    r = trial[key]
                    self.assertEqual(r["skipped_no_positive"], 0)
                    self.assertEqual(r["n_query"], len(sets[group]))
                    self.assertEqual(r["gallery_size_min"], c["effective_gallery_size"])
                    self.assertEqual(r["gallery_size_max"], c["effective_gallery_size"])
        with contextlib.redirect_stdout(io.StringIO()):
            literal = s11.run_controlled(self.sim, self.ann, mode="literal", bootstrap=100)
        self.assertEqual(literal["gallery_control"]["A"]["status"], "infeasible_requested_downsampling")
        self.assertEqual(literal["gallery_control"]["G"]["status"], "complete")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "count check failed"):
            s11.run_controlled(self.sim, self.ann, expected_common=(1, 2), bootstrap=100)

    def test_streaming_topk_and_reciprocal_reference(self):
        for k in (1, 5, 10, 20, 50):
            expected = np.sort(self.sim, axis=0)[::-1][:k].mean(0, dtype=np.float32)
            np.testing.assert_array_equal(topk_mean(self.sim, k), expected)
        result = reciprocal_statistics(self.sim)
        row = np.argsort(-self.sim, axis=1, kind="stable")
        col = np.argsort(-self.sim, axis=0, kind="stable")
        for k in (5, 10, 20, 50):
            expected = sum(int(q in col[:k, g]) for q in range(200) for g in row[q, :k])
            self.assertEqual(result["results"][str(k)]["reciprocal_pairs"], expected)
        ties = reciprocal_statistics(np.zeros_like(self.sim))
        for k in (5, 10, 20, 50):
            self.assertEqual(ties["results"][str(k)]["reciprocal_pairs"], k * k)

    def test_paired_cluster_bootstrap_and_mcnemar(self):
        first = np.array([1] + [0] * 9)
        second = 1 - first
        self.assertAlmostEqual(mcnemar_exact(first, second)["p_two_sided"], 0.021484375)
        unchanged = paired_test(first, first, repeats=10000)
        self.assertEqual(unchanged["ci95_percentile_pp"], [0, 0])
        groups = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
        actual = paired_test(first, second, groups=groups, repeats=10000, seed=2026)
        rng = np.random.default_rng(2026)
        diff, counts = np.bincount(groups, weights=second - first), np.bincount(groups)
        draw = rng.integers(0, 3, (10000, 3), dtype=np.int64)
        expected = diff[draw].sum(1) / counts[draw].sum(1) * 100
        self.assertAlmostEqual(actual["bootstrap_mean_delta_pp"], expected.mean())
        np.testing.assert_allclose(actual["ci95_percentile_pp"], np.percentile(expected, [2.5, 97.5]))

    def test_group_no_leak_and_whole_identity_scans(self):
        first, second = balanced_group_halves(self.ann.query_ids, 2026)
        self.assertEqual(len(np.intersect1d(self.ann.query_ids[first], self.ann.query_ids[second])), 0)
        for budget in (1, 15, 70, len(first)):
            selected = sample_identity_groups(self.ann.query_ids, first, budget, 2026)
            # The helper returns metadata along with selected rows.
            rows = selected[0] if isinstance(selected, tuple) else selected
            self.assertGreaterEqual(len(rows), budget)
            for pid in np.unique(self.ann.query_ids[rows]):
                self.assertEqual(np.sum(self.ann.query_ids[rows] == pid), np.sum(self.ann.query_ids == pid))

    def test_S11_all_A_to_G_real_cli_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory(prefix="tagpr_s11_mock_") as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            ann_path, sim_path, gg_path = inputs / "ann.json", inputs / "sim.npy", inputs / "sim_gg.npy"
            ann_path.write_text(json.dumps(self.records), encoding="utf-8")
            np.save(sim_path, self.sim)
            np.save(gg_path, self.gg)
            paths = (ann_path, sim_path, gg_path)
            before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            command = [sys.executable, str(Path(s11.__file__)), "--sim", str(sim_path), "--anno", str(ann_path),
                       "--sim-gg", str(gg_path), "--out", str(root / "out"), "--tag", "mock.S11",
                       "--gallery-control", "common-min", "--expected-common", "48", "32"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            reports = list((root / "out").glob("*.json"))
            self.assertEqual(len(reports), 1)
            result = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            for key in ("A", "B_error_decomposition", "C_reciprocal", "D_hubness", "E_bootstrap", "F", "G"):
                self.assertIn(key, result)
            self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
            bootstrap = result["E_bootstrap"]["results"]
            print("S11 MOCK " + json.dumps(dict(shape=list(self.sim.shape), raw_R1=result["raw"]["R1"],
                  csls_R1=result["csls_k10"]["R1"], common_A=48, common_G=32,
                  reciprocal_pairs={k: r["reciprocal_pairs"] for k, r in result["C_reciprocal"]["results"].items()},
                  bootstrap={k: {"delta_mean_pp": v["bootstrap_mean_delta_pp"], "CI95": v["ci95_percentile_pp"]}
                             for k, v in bootstrap.items()},
                  inputs_unchanged=True, tasks="A/B/C/D/E/F/G complete")))

    def test_F_small_budgets_and_G_locked_k_independent_reference(self):
        full = topk_mean(self.sim, 10)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_grouped(self.sim, self.ann, full, scan_sizes=(10, 30, 50, 80))
        scan = result["F"]["calibration_scan"]
        self.assertEqual(len(scan), 5)
        for row in scan:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(len(row["repetitions"]), 3)
        self.assertTrue(all(r["matches_crossfit_exactly"] for r in scan[-1]["repetitions"]))
        for row in result["G"]["repetitions"]:
            first, second = random_identity_halves(self.ann.query_ids, row["seed"])
            selection, heldout = (first, second) if row["direction"] == "H1_to_H2" else (second, first)
            k = row["selected_k"]
            chosen = min(row["candidates"], key=lambda c: (-c["selection_R1"], c["k"]))["k"]
            self.assertEqual(k, chosen)
            for key, penalty in (("locked_held_out", topk_mean(self.sim, k, heldout)),
                                 ("fixed_held_out_bank_k10", topk_mean(self.sim, 10, heldout)),
                                 ("fixed_full_query_k10", full)):
                reference = evaluate(self.sim, self.ann, queries=heldout, penalty=penalty, alpha=0.5)
                self.assertEqual(row[key]["R1"], reference["R1"])
            frozen = evaluate(self.sim, self.ann, queries=heldout, penalty=topk_mean(self.sim, k, selection), alpha=0.5)
            self.assertEqual(row["supplementary_frozen_selection_bank"]["metrics"]["R1"], frozen["R1"])

    def test_S12_all_banks_and_methods(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = s12.self_test()
        print("S12 MOCK " + json.dumps(json_ready(result)))

    def test_S12_formulas_independent_full_matrix_reference(self):
        bank = np.random.default_rng(2028).uniform(-0.7, 0.7, (150, 100)).astype(np.float32)
        result = s12.run_corrections(self.sim, bank, self.gg, self.ann, subset_sizes=(40,),
                                    seeds=(2026,), ks=(5,), alphas=(0.5,), betas=(10,), progress=None)
        bank_info = {b["bank"]: b for b in result["banks"]}
        pg = np.log(np.exp(10 * self.gg.astype(np.float64)).sum(0)) / 10
        for row in result["rows"]:
            info = bank_info[row["bank"]]
            b = self.sim if row["bank"] == "test_full" else bank
            if not isinstance(info["row_indices"], str):
                b = b[info["row_indices"]]
            pq = np.log(np.exp(10 * b.astype(np.float64)).sum(0)) / 10
            if row["method"] == "CSLS/NNN":
                p = np.sort(b, axis=0)[::-1][:5].mean(0, dtype=np.float32)
                adjusted = self.sim - np.float32(0.5) * p
            elif row["method"] == "DualIS":
                adjusted = 2 * self.sim.astype(np.float64) - pq - pg
            else:
                adjusted = self.sim.astype(np.float64) - pq
                if row["method"] == "DIS":
                    hubs = np.unique(b.argmax(1))
                    enabled = np.isin(self.sim.argmax(1), hubs)
                    adjusted[~enabled] = self.sim[~enabled]
            hits, aps, _, _ = reference_metrics(adjusted, self.ann)
            for j, key in enumerate(("R1", "R5", "R10")):
                expected = float(np.float32(hits[:, j].sum()) * np.float32(100) / np.float32(len(hits)))
                self.assertEqual(row[key], expected, row)
            self.assertAlmostEqual(row["mAP"], float(aps.mean() * 100), places=12)

    def test_S12_text_features_and_parity_guard(self):
        from ._s12_features import feature_self_test
        with contextlib.redirect_stdout(io.StringIO()):
            result = feature_self_test()
        print("S12 FEATURES " + json.dumps(json_ready(result)))

    def test_invalid_inputs_archive_protection_and_help(self):
        with tempfile.TemporaryDirectory(prefix="tagpr_guards_") as directory:
            root = Path(directory)
            archive = root / "_archive_evidence"
            archive.mkdir()
            marker = archive / "sentinel.txt"
            marker.write_text("read-only evidence", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "read-only"):
                safe_output_dir(archive / "new")
            self.assertEqual(marker.read_text(), "read-only evidence")
            np.save(root / "bad.npy", np.zeros((2, 3), np.float32))
            with self.assertRaisesRegex(ValueError, "shape"):
                load_similarity(root / "bad.npy", self.ann)
            with self.assertRaisesRegex(ValueError, "k must"):
                topk_mean(self.sim, 0)
            with self.assertRaisesRegex(ValueError, "k must"):
                topk_mean(self.sim, 201)
            filenames = write_report(root / "new", "model.v1", {"ok": True}, "ok\n")
            self.assertTrue(all("model.v1_" in p.name for p in filenames))
        for script in (s11, s12):
            completed = subprocess.run([sys.executable, script.__file__, "--help"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--out", completed.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

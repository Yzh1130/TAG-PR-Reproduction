"""200x100 CPU mocks; every numerical expectation is computed from mock inputs."""

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

from scripts import R1_joint_enrichment as joint
from scripts._offline_common import annotation_from_records, evaluate
from scripts._s11_descriptive import error_decomposition


def fixture():
    records = []
    sources = ("Market", "PRAI-1581", "uavhuman", "AG-ReID.v2", "G2APS")
    for i in range(100):
        source = sources[i // 20]
        cam = 1 if source == "Market" else 0 if source in ("PRAI-1581", "uavhuman") else i % 2
        records.append(dict(file_path=f"{source}/mock_{i}.jpg", id=i // 4, cam_id=cam,
                            captions=[f"blue shirt black pants {i}", f"white coat person {i}"]))
    ann = annotation_from_records(records)
    sim = np.random.default_rng(2026).normal(size=(200, 100)).astype(np.float32)
    sim += np.float32(0.25) * (ann.query_ids[:, None] == ann.gallery_ids)
    raw = evaluate(sim, ann)
    s11 = dict(B_error_decomposition=dict(raw=error_decomposition(ann, raw)))
    return records, ann, sim, s11


class JointTests(unittest.TestCase):
    def test_joint_priors_equal_explicit_negative_gallery_and_conserve(self):
        _, ann, sim, s11 = fixture()
        raw = evaluate(sim, ann)
        errors, observed, priors = joint.joint_error_inputs(ann, raw)
        expected = np.zeros_like(priors)
        for q in np.flatnonzero(errors):
            negative = ann.gallery_ids != ann.query_ids[q]
            ss = ann.gallery_domains == ann.query_domains[q]
            sv = ann.gallery_views == ann.query_views[q]
            for c, mask in enumerate((ss & sv, ss & ~sv, ~ss & sv, ~ss & ~sv)):
                expected[q, c] = np.count_nonzero(negative & mask) / negative.sum()
        np.testing.assert_array_equal(priors, expected)
        report = joint.run_analysis(sim, ann, s11, bootstrap=200, seed=2026)
        self.assertEqual(sum(report["counts"].values()), int(errors.sum()))
        self.assertEqual(report["s11_four_cell_gate"], "PASS")
        self.assertEqual(set(report["mixed_source_conditionals"]), set(joint.MIXED))
        for pure in joint.PURE:
            self.assertNotIn("same_source_cross_view", report["strata"][pure]["cells"])
        print("R1 joint MOCK 200x100: explicit-negative prior max error=0; four-cell conservation/gate PASS")

    def test_cluster_draws_reform_both_numerator_and_denominator(self):
        _, ann, sim, _ = fixture()
        errors, observed, priors = joint.joint_error_inputs(ann, evaluate(sim, ann))
        repeats = 200
        selected = ann.query_views == 0
        result = joint.enrichment_stratum(ann, selected, errors, observed, priors, repeats, 2026)
        ids = np.unique(ann.query_ids)
        draws = np.random.default_rng(2026).integers(0, len(ids), (repeats, len(ids)))
        expected = []
        for draw in draws:
            multiplicity = np.bincount(draw, minlength=len(ids))
            weights = multiplicity[np.searchsorted(ids, ann.query_ids)] * selected * errors
            denominator = weights.sum()
            p = np.dot(weights, observed[:, 0]) / denominator
            prior = np.dot(weights, priors[:, 0]) / denominator
            expected.append((p, prior, p - prior, p / prior))
        item = result["cells"][joint.CELLS[0]]
        for i, key in enumerate(("observed_share", "expected_share", "absolute_excess", "enrichment_ratio")):
            np.testing.assert_allclose(item[key]["ci95_percentile"], np.percentile(np.asarray(expected)[:, i], [2.5, 97.5]), atol=1e-14, rtol=0)
        print("R1 joint MOCK: whole-ID shared-draw independent CI reference PASS")

    def test_reference_failure_zero_errors_and_ratio_suppression(self):
        _, ann, sim, s11 = fixture()
        changed = json.loads(json.dumps(s11))
        changed["B_error_decomposition"]["raw"]["strata"]["all"]["cells"][joint.CELLS[0]]["count"] += 1
        with self.assertRaisesRegex(RuntimeError, "exact gate FAILED"):
            joint.run_analysis(sim, ann, changed, bootstrap=10)
        perfect = (ann.query_ids[:, None] == ann.gallery_ids).astype(np.float32)
        reference = dict(B_error_decomposition=dict(raw=error_decomposition(ann, evaluate(perfect, ann))))
        report = joint.run_analysis(perfect, ann, reference, bootstrap=10)
        self.assertEqual(report["strata"]["all"]["n_errors"], 0)
        self.assertIsNone(report["strata"]["all"]["cells"][joint.CELLS[0]]["observed_share"]["estimate"])
        self.assertIsNone(report["strata"]["all"]["cells"][joint.CELLS[0]]["enrichment_ratio"]["estimate"])

    def test_cli_hashes_exclusive_collision_and_help(self):
        records, _, sim, s11 = fixture()
        command = [sys.executable, str(Path(joint.__file__))]
        self.assertEqual(subprocess.run(command + ["--help"], capture_output=True).returncode, 0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "inputs"
            source.mkdir()
            np.save(source / "sim.npy", sim)
            (source / "ann.json").write_text(json.dumps(records), encoding="utf-8")
            (source / "s11.json").write_text(json.dumps(s11), encoding="utf-8")
            paths = [source / n for n in ("sim.npy", "ann.json", "s11.json")]
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            args = ["--sim", str(paths[0]), "--anno", str(paths[1]), "--s11-json", str(paths[2]),
                    "--out", str(root / "new" / "joint.json"), "--bootstrap", "30"]
            for _ in range(2):
                run = subprocess.run(command + args, capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            outputs = list((root / "new").glob("*.json"))
            self.assertEqual(len(outputs), 2)
            report = json.loads(outputs[0].read_text(encoding="utf-8"))
            self.assertEqual(report["provenance"]["seeds"], {"bootstrap": 2026})
            for info in report["provenance"]["inputs"].values():
                self.assertEqual(info["sha256_16"], hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest()[:16])
                self.assertEqual(info["bytes"], Path(info["path"]).stat().st_size)
            self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            print("R1 joint MOCK CLI: --help, immutable inputs, hash16/bytes/argv/HEAD, collision-safe JSON/log PASS")


if __name__ == "__main__":
    unittest.main()

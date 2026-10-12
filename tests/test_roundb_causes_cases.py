"""200x100 CPU mock tests; values printed here are synthetic, never paper results."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts import R2_hard_case_pool as pool
from scripts import R2_make_figure as figure
from scripts import R3_finegrained_causes as causes
from scripts._offline_common import annotation_from_records, evaluate, topk_mean


def fixture(root):
    domains = ("AG-ReID.v2", "G2APS", "Market", "PRAI-1581", "uavhuman")
    records = []
    for pid in range(20):
        domain = domains[pid // 4]
        color = ("blue", "red", "green", "black")[pid % 4]
        clothing = ("jacket", "coat", "shirt", "hoodie")[pid % 4]
        for j in range(5):
            view = int(j >= 3) if pid < 8 else int(domain == "Market")
            relative = f"{domain}/mock_{pid}_{j}.png"
            records.append({"file_path": relative, "id": pid, "cam_id": view,
                            "captions": [f"person {pid} in {color} {clothing} and black pants behind umbrella",
                                         f"person {pid} wears white shirt and brown shoes"]})
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12 + (len(records) % 20) * 2, 24 + len(records) // 20 * 8),
                      (pid * 10, j * 30, 100)).save(path)
    ann = annotation_from_records(records)
    rng = np.random.default_rng(2026)
    sim = rng.normal(0, .16, (200, 100)).astype(np.float32)
    sim += (ann.query_ids[:, None] == ann.gallery_ids[None]) * np.float32(.18)
    sim[:, 0] += np.float32(.38)
    bank = rng.normal(0, .16, (300, 100)).astype(np.float32)
    bank[:, 0] += np.float32(.38)
    raw = evaluate(sim, ann)
    corrected = evaluate(sim, ann, penalty=topk_mean(sim, 10), alpha=.5)
    s13 = {"raw": {"R1": raw["R1"]}, "configurations": {
        "c": {"bank": "test_full", "alpha": .5, "k": 10, "metrics": {"R1": corrected["R1"]}}}}
    return records, ann, sim, bank, s13


class CausesCasesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="roundb_causes_mock_")
        self.root = Path(self.temp.name)
        self.images = self.root / "inputs" / "images"
        self.records, self.ann, self.sim, self.bank, self.s13 = fixture(self.images)
        self.sizes, self.areas, self.manifest = causes.read_image_sizes(self.ann, self.images)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_source_quartiles_margin_rank_and_attribute_paths(self):
        report = causes.analyze(self.sim, self.ann, self.records, self.areas, self.sizes, bootstrap=40)
        self.assertEqual(set(report["image_size"]["source_quartiles"]), set(self.ann.gallery_domains))
        for domain, entry in report["image_size"]["source_quartiles"].items():
            self.assertEqual(sum(group["n_queries"] for group in entry["groups"]), int((self.ann.query_domains == domain).sum()))
            self.assertEqual(len(entry["groups"]), 4)
        raw = evaluate(self.sim, self.ann)
        for entry in report["per_error_margin_rank"]:
            q = entry["query_index"]
            order = np.argsort(-self.sim[q], kind="stable")
            expected_rank = np.flatnonzero(self.ann.gallery_ids[order] == self.ann.query_ids[q])[0] + 1
            self.assertEqual(entry["best_correct_rank"], expected_rank)
            self.assertEqual(entry["margin"], float(self.sim[q, order[0]] - self.sim[q, entry["best_correct_gallery_index"]]))
        for entry in report["attribute_overlap"]["per_error"]:
            q, g = entry["query_index"], entry["random_gallery_index"]
            self.assertEqual(self.ann.gallery_domains[g], self.ann.query_domains[q])
            self.assertNotEqual(self.ann.gallery_ids[g], self.ann.query_ids[q])
            expanded_texts = [caption for row in self.records for caption in row["captions"]]
            query_set = causes.attribute_tokens(expanded_texts[q])
            expected_max = max(causes.jaccard(query_set, causes.attribute_tokens(caption))
                               for caption in self.records[entry["error_gallery_index"]]["captions"])
            self.assertEqual(entry["error_jaccard"], expected_max)
        self.assertEqual(report["raw"]["R1"], raw["R1"])
        self.assertEqual(len(self.manifest), 100)
        print("R3 MOCK", json.dumps({"shape": list(self.sim.shape), "source_quartiles": 20,
              "raw_R1": report["raw"]["R1"], "attribute_paired_difference": report["attribute_overlap"]["paired_error_minus_random"],
              "margin": report["similarity_margin"]["distribution"], "rank": report["truth_rank"]["distribution"]}))

    def test_cluster_bootstrap_matches_independent_reference_and_empty(self):
        values, ids = np.array([0., 1., 2., 4., 8.]), np.array([0, 0, 1, 2, 2])
        result = causes.cluster_mean(values, ids, 100, 2026)
        rng = np.random.default_rng(2026)
        draw = rng.integers(0, 3, (100, 3))
        sums, counts = np.bincount(ids, weights=values), np.bincount(ids)
        expected = sums[draw].sum(1) / counts[draw].sum(1)
        np.testing.assert_array_equal(result["ci95"], np.percentile(expected, [2.5, 97.5]))
        self.assertEqual(causes.cluster_mean([], [], 10, 2026)["mean"], None)
        with self.assertRaisesRegex(ValueError, "aligned"):
            causes.cluster_mean([np.nan], [0], 10, 2026)

    def test_jaccard_max_over_captions_and_missing_same_source_negative(self):
        self.assertEqual(causes.jaccard(set(), set()), 0.)
        self.assertEqual(causes.attribute_tokens("BLUE t-shirt!"), {"blue", "tshirt"})
        self.assertEqual(causes.jaccard({"blue", "shirt"}, {"blue"}), .5)
        from dataclasses import replace
        ann = replace(self.ann, gallery_domains=np.array([f"single_{i}" for i in range(100)]),
                      query_domains=np.repeat([f"single_{i}" for i in range(100)], 2))
        raw = evaluate(self.sim, ann)
        result = causes.attribute_comparison(self.records, ann, raw["details"]["top1"], raw["details"]["correct"], 2026)
        self.assertTrue(np.all(result["random_gallery_indices"] == -1))

    def test_no_errors_tied_areas_and_invalid_dimensions(self):
        sim = np.zeros_like(self.sim)
        sim[np.arange(len(sim)), self.ann.query_source_indices] = .5
        report = causes.analyze(sim, self.ann, self.records, np.ones(100, dtype=np.int64),
                                self.sizes, bootstrap=10)
        self.assertEqual(report["similarity_margin"]["distribution"]["count"], 0)
        self.assertEqual(report["truth_rank"]["distribution"]["count"], 0)
        self.assertEqual(report["attribute_overlap"]["error_top1"]["mean"], None)
        for entry in report["image_size"]["source_quartiles"].values():
            self.assertEqual(sum(group["n_queries"] == 0 for group in entry["groups"]), 3)
        with self.assertRaisesRegex(ValueError, "align"):
            causes.analyze(sim, self.ann, self.records, np.ones(99), self.sizes, bootstrap=10)

    def test_all_seven_pools_seeded_sampling_parity_and_no_mutation(self):
        before = hashlib.sha256(self.sim.tobytes()).hexdigest()
        report = pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas, self.images, self.s13)
        repeat = pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas, self.images, self.s13)
        self.assertEqual(set(report["pools"]), set(pool.POOL_NAMES))
        for name, entry in report["pools"].items():
            self.assertEqual(entry["selected_query_indices"], repeat["pools"][name]["selected_query_indices"])
            self.assertLessEqual(entry["selected_count"], 30)
            self.assertEqual(len(set(entry["selected_query_indices"])), entry["selected_count"])
            for case in entry["cases"]:
                self.assertEqual(len(case["before_top3"]), 3)
                self.assertEqual(len(case["after_top3"]), 3)
                if "success" in name:
                    self.assertFalse(case["before_correct"])
                    self.assertTrue(case["after_correct"])
                if "failure" in name:
                    self.assertTrue(case["before_correct"])
                    self.assertFalse(case["after_correct"])
        self.assertEqual(before, hashlib.sha256(self.sim.tobytes()).hexdigest())
        bad = json.loads(json.dumps(self.s13))
        bad["configurations"]["c"]["metrics"]["R1"] = np.nextafter(bad["configurations"]["c"]["metrics"]["R1"], np.inf)
        with self.assertRaisesRegex(AssertionError, "exact check"):
            pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas, self.images, bad)
        print("R2 POOL MOCK", json.dumps({name: {"candidates": item["candidate_count"], "selected": item["selected_count"]}
              for name, item in report["pools"].items()}))

    def test_contact_sheet_text_free_fallback_and_manual_selection(self):
        report = pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas, self.images, self.s13, cases_per_pool=2)
        cases = report["pools"]["similar_clothing"]["cases"]
        self.assertTrue(cases)
        cases[0]["query_text"] = "中文人工说明"
        with patch.object(pool, "choose_font", return_value=(None, None)), redirect_stdout(io.StringIO()):
            rendered = pool.render_cases(cases, self.root / "fallback.png", "人工选择")
        self.assertFalse(rendered["text_enabled"])
        layout = json.loads(Path(rendered["layout_json"]).read_text(encoding="utf-8"))
        self.assertEqual(layout["rows"][0]["query_text"], "中文人工说明")
        self.assertEqual(len(layout["rows"][0]["slots"]), 8)
        with self.assertRaises(FileExistsError):
            with patch.object(pool, "choose_font", return_value=(None, None)), redirect_stdout(io.StringIO()):
                pool.render_cases(cases, self.root / "fallback.png")
        pool_json = self.root / "pool.json"
        pool_json.write_text(json.dumps(report), encoding="utf-8")
        item = {"pool": "similar_clothing", "query_index": cases[0]["query_index"]}
        select = self.root / "select.json"
        select.write_text(json.dumps({"pool_json": "pool.json", "items": [item]}), encoding="utf-8")
        _, _, selected, _ = figure.load_selection(select)
        self.assertEqual(selected[0]["query_index"], cases[0]["query_index"])
        select.write_text(json.dumps({"pool_json": "pool.json", "items": [item, item]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            figure.load_selection(select)
        with self.assertRaisesRegex(ValueError, "font file"):
            pool.choose_font(["text"], str(self.root / "no-font.ttf"))
        with self.assertRaisesRegex(ValueError, "outside"):
            causes.image_path(self.images, "../../absent.png")

    def test_real_cli_help_outputs_provenance_immutable_inputs_and_manual_figure(self):
        inputs = self.root / "inputs"
        anno, sim, bank, s13 = (inputs / name for name in ("ann.json", "sim.npy", "bank.npy", "s13.json"))
        anno.write_text(json.dumps(self.records), encoding="utf-8")
        s13.write_text(json.dumps(self.s13), encoding="utf-8")
        np.save(sim, self.sim)
        np.save(bank, self.bank)
        sources = [anno, sim, bank, s13]
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
        for module in (causes, pool, figure):
            completed = subprocess.run([sys.executable, module.__file__, "--help"], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        out3, out2 = self.root / "r3", self.root / "r2"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(causes.main(["--sim", str(sim), "--anno", str(anno), "--image-root", str(self.images),
                                         "--out", str(out3), "--bootstrap", "40"]), 0)
            self.assertEqual(pool.main(["--sim", str(sim), "--bank", str(bank), "--anno", str(anno), "--image-root", str(self.images),
                                       "--s13-json", str(s13), "--out", str(out2), "--cases-per-pool", "2"]), 0)
            self.assertEqual(pool.main(["--sim", str(sim), "--bank", str(bank), "--anno", str(anno), "--image-root", str(self.images),
                                       "--s13-json", str(s13), "--out", str(out2), "--cases-per-pool", "2"]), 0)
        self.assertEqual(len(list(out2.glob("R2_hard_case_pool*.json"))), 2)
        report_file = next(out2.glob("R2_hard_case_pool*.json"))
        report = json.loads(report_file.read_text(encoding="utf-8"))
        for manifest in report["provenance"]["inputs"].values():
            self.assertEqual(len(manifest["sha256_16"]), 16)
            self.assertGreater(manifest["bytes"], 0)
        self.assertEqual(len(report["image_input_manifest"]), 100)
        item = report["pools"]["similar_clothing"]["cases"][0]
        select = inputs / "select.json"
        select.write_text(json.dumps({"pool_json": str(report_file), "items": [{"pool": "similar_clothing", "query_index": item["query_index"]}]}), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(figure.main(["--select", str(select), "--out", str(self.root / "figure")]), 0)
        figure_report = json.loads(next((self.root / "figure").glob("R2_make_figure*.json")).read_text(encoding="utf-8"))
        self.assertEqual(figure_report["rendered"]["n_rows"], 1)
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources])
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            pool.main(["--sim", str(sim), "--bank", str(bank), "--anno", str(anno), "--image-root", str(self.images),
                       "--s13-json", str(s13), "--out", str(inputs / "badout")])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Round C R2 tests use a 200x100 synthetic matrix, never paper measurements."""

from contextlib import redirect_stderr, redirect_stdout
import copy
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
from scripts._offline_common import compact_metrics, evaluate, topk_mean
from tests.test_roundb_causes_cases import fixture


class RoundCR2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="roundc_r2_mock_")
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.images = self.inputs / "images"
        self.records, self.ann, self.sim, self.bank, self.s13 = fixture(self.images)
        _, self.areas, _ = causes.read_image_sizes(self.ann, self.images)
        self.report = pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas,
                                   self.images, self.s13, cases_per_pool=3)
        self.sample_pool = next(name for name in pool.POOL_NAMES if self.report["pools"][name]["cases"])
        self.case = self.report["pools"][self.sample_pool]["cases"][0]

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, document):
        path = self.inputs / name
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return path

    def single_pool(self, condition=None):
        case = copy.deepcopy(self.case)
        if condition:
            case["correction_condition"] = condition
        return {"pools": {self.sample_pool: {"cases": [case], "seed": 2026,
                    "selected_query_indices": [case["query_index"]],
                    "candidate_query_indices": [case["query_index"], 999999],
                    "correction_condition": case["correction_condition"]}}}

    def test_default_and_custom_inductive_math_and_parity(self):
        expected = compact_metrics(evaluate(self.sim, self.ann, penalty=topk_mean(self.bank, 10), alpha=.5))
        self.assertEqual({key: self.report["inductive"][key] for key in expected}, expected)
        self.s13["training_bank_provenance_note"] = "original note must stay unchanged"
        default = pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas,
                               self.images, self.s13, cases_per_pool=2)
        self.assertEqual(default["training_bank_provenance_note"], self.s13["training_bank_provenance_note"])
        custom = pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas, self.images,
                              self.s13, cases_per_pool=30, inductive_alpha=.75, inductive_k=30,
                              bank_label="E3 fresh mock")
        expected = compact_metrics(evaluate(self.sim, self.ann, penalty=topk_mean(self.bank, 30), alpha=.75))
        self.assertEqual({key: custom["inductive"][key] for key in expected}, expected)
        self.assertEqual(custom["inductive"]["bank_label"], "E3 fresh mock")
        self.assertIn("does not certify", custom["training_bank_provenance_note"])
        for name in ("inductive_success", "inductive_failure"):
            for case in custom["pools"][name]["cases"]:
                q = case["query_index"]
                scores = self.sim[q] - np.float32(.75) * topk_mean(self.bank, 30)
                expected_indices = np.argsort(-scores, kind="stable")[:3].tolist()
                self.assertEqual([item["gallery_index"] for item in case["after_top3"]], expected_indices)
        bad = copy.deepcopy(self.s13)
        bad["configurations"]["c"]["metrics"]["R1"] = np.nextafter(bad["configurations"]["c"]["metrics"]["R1"], np.inf)
        with self.assertRaisesRegex(AssertionError, "exact check"):
            pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas, self.images, bad,
                         inductive_alpha=.75, inductive_k=30)
        with self.assertRaisesRegex(ValueError, "positive"):
            pool.analyze(self.sim, self.bank, self.ann, self.records, self.areas, self.images,
                         self.s13, inductive_k=0)

    def test_cli_bank_metadata_and_provenance(self):
        anno = self.write_json("ann.json", self.records)
        s13 = self.write_json("s13.json", self.s13)
        sim, bank = self.inputs / "sim.npy", self.inputs / "bank.npy"
        np.save(sim, self.sim)
        np.save(bank, self.bank)
        before = hashlib.sha256(bank.read_bytes()).hexdigest()
        out = self.root / "pool_out"
        argv = ["--sim", str(sim), "--bank", str(bank), "--anno", str(anno),
                "--image-root", str(self.images), "--s13-json", str(s13), "--out", str(out),
                "--cases-per-pool", "1", "--inductive-alpha", ".75", "--inductive-k", "30",
                "--bank-label", "E3_fresh_forward e3_c0_b0_seed0"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(pool.main(argv), 0)
        saved = json.loads(next(out.glob("R2_hard_case_pool*.json")).read_text(encoding="utf-8"))
        self.assertEqual(saved["inductive"]["alpha"], .75)
        self.assertEqual(saved["inductive"]["k"], 30)
        self.assertEqual(saved["inductive"]["bank"]["sha256_16"], before[:16])
        self.assertEqual(saved["inductive"]["bank"]["bytes"], bank.stat().st_size)
        self.assertEqual(saved["inductive"]["bank"]["path"], str(bank.resolve()))
        self.assertEqual(saved["provenance"]["arguments"]["bank_label"], argv[-1])
        self.assertEqual(saved["provenance"]["arguments"]["inductive_alpha"], .75)
        self.assertEqual(saved["provenance"]["arguments"]["inductive_k"], 30)
        self.assertEqual(saved["provenance"]["inputs"]["bank"]["sha256_16"], before[:16])
        self.assertEqual(hashlib.sha256(bank.read_bytes()).hexdigest(), before)
        print("R2 ROUND C MOCK bank provenance PASS; shape=200x100; alpha=.75 k=30")

    def test_panels_missing_candidates_ambiguity_and_explicit_conditions(self):
        q = self.case["query_index"]
        trans = self.write_json("trans.json", self.single_pool())
        ind = self.write_json("ind.json", self.single_pool("training-text bank CSLS alpha=.75 k=30"))
        specs = [f"trans={trans}", f"ind={ind}"]
        with self.assertRaisesRegex(ValueError, "different correction"):
            figure.load_panels([f"a:{q}"], specs)
        sources, cases, selected = figure.load_panels([f"a:{q}@trans", f"e:{q}@ind"], specs)
        self.assertEqual(len(cases), 2)
        self.assertEqual([row["panel"] for row in selected], ["a", "e"])
        self.assertEqual(selected[0]["pool_memberships"][0]["pool_sampling_seed"], 2026)
        self.assertTrue(selected[0]["pool_memberships"][0]["in_seeded_30_sample"])
        with self.assertRaisesRegex(ValueError, "candidate only"):
            figure.load_panels(["a:999999@trans"], specs)
        with self.assertRaisesRegex(ValueError, "missing"):
            figure.load_panels(["a:999998@trans"], specs)
        both = self.single_pool()
        both["pools"]["inductive_success"] = copy.deepcopy(both["pools"][self.sample_pool])
        both["pools"]["inductive_success"]["cases"][0]["correction_condition"] = "training-text bank CSLS alpha=.75 k=30"
        both_path = self.write_json("both.json", both)
        with self.assertRaisesRegex(ValueError, "different correction"):
            figure.load_panels([f"a:{q}@both"], [f"both={both_path}"])
        _, cases, selected = figure.load_panels([f"a:{q}@both/inductive_success"], [f"both={both_path}"])
        self.assertEqual(selected[0]["pool_memberships"][0]["pool"], "inductive_success")

    def test_same_condition_memberships_and_unknown_legacy_seed(self):
        q = self.case["query_index"]
        first = self.single_pool()
        first["pools"][self.sample_pool].pop("seed")
        first["pools"][self.sample_pool].pop("selected_query_indices")
        a = self.write_json("a.json", first)
        b = self.write_json("b.json", self.single_pool())
        _, cases, selected = figure.load_panels([f"b:{q}"], [f"old={a}", f"other={b}"])
        self.assertEqual(len(cases), 1)
        self.assertEqual(len(selected[0]["pool_memberships"]), 2)
        self.assertIsNone(selected[0]["pool_memberships"][0]["pool_sampling_seed"])
        self.assertIsNone(selected[0]["pool_memberships"][0]["in_seeded_30_sample"])

    def test_panel_cli_exports_english_only_png_pdf_selected_json_and_errors(self):
        document = self.single_pool()
        document["pools"][self.sample_pool]["cases"][0]["query_text"] = "中文描述 not drawn"
        path = self.write_json("panels.json", document)
        q = self.case["query_index"]
        argv = ["--pool-json", f"trans={path}", "--panel", f"a:{q}@trans", "--out", str(self.root / "fig"),
                "--tag", "mock"]
        with patch.object(figure, "choose_font", return_value=(None, None)), redirect_stdout(io.StringIO()):
            self.assertEqual(figure.main(argv), 0)
        report = json.loads(next((self.root / "fig").glob("R2_make_figure*.json")).read_text(encoding="utf-8"))
        selected_path = Path(report["selected_cases_json"])
        self.assertEqual(selected_path.name, "selected_cases_mock.json")
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        self.assertEqual(selected["cases"][0]["query_text"], "中文描述 not drawn")
        self.assertIn("argv", selected["provenance"])
        self.assertEqual(selected["manual_selection"][0]["pool_memberships"][0]["source_json_sha256_16"],
                         hashlib.sha256(path.read_bytes()).hexdigest()[:16])
        rendered = report["rendered"]
        self.assertTrue(Path(rendered["pdf"]).read_bytes().startswith(b"%PDF"))
        with Image.open(rendered["png"]) as image:
            self.assertAlmostEqual(image.info["dpi"][0], 300, delta=.01)
        layout = json.loads(Path(rendered["layout_json"]).read_text(encoding="utf-8"))
        self.assertFalse(layout["query_text_drawn"])
        self.assertEqual(layout["language"], "English")
        self.assertEqual(len(layout["rows"][0]["slots"]), 8)
        self.assertEqual(layout["rows"][0]["slots"][0]["frame_color"], "green")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            figure.main(["--pool-json", f"trans={path}", "--panel", "b:999999@trans", "--out", str(self.root / "missing")])
        self.assertEqual(error.exception.code, 2)
        print("R2 ROUND C MOCK explicit selection PNG300dpi/PDF/layout/selected JSON PASS; missing row rejected")

    def test_help_and_legacy_selection_retained(self):
        for module in (pool, figure):
            result = subprocess.run([sys.executable, module.__file__, "--help"], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        path = self.write_json("old_pool.json", self.single_pool())
        select = self.write_json("old_select.json", {"pool_json": str(path), "items": [
            {"pool": self.sample_pool, "query_index": self.case["query_index"]}]})
        _, _, cases, selected = figure.load_selection(select)
        self.assertEqual(cases[0], self.case)
        self.assertEqual(selected[0]["pool"], self.sample_pool)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(figure.main(["--select", str(select), "--out", str(self.root / "legacy")]), 0)
        report = json.loads(next((self.root / "legacy").glob("R2_make_figure*.json")).read_text(encoding="utf-8"))
        self.assertNotIn("pdf", report["rendered"])
        self.assertEqual(Path(report["rendered"]["png"]).name, "selected_panel.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)

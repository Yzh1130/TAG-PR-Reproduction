"""S13 filename-convention regressions on a 200x100 MOCK (no archive writes)."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from . import S13_inductive_bootstrap as s13
from .test_S13_bootstrap import fixture


class S13FreshBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.ann, cls.sim, cls.bank = fixture(100)

    def test_fresh_only_changes_labels_not_computation_including_extra_d(self):
        options = dict(bootstrap=100, extra_configs=["train_full:0.5:10"])
        old_messages, fresh_messages = [], []
        original = s13.run_analysis(self.sim, self.bank, self.ann,
                                    progress=old_messages.append, **options)
        fresh = s13.run_analysis(self.sim, self.bank, self.ann, fresh_bank=True,
                                 progress=fresh_messages.append, **options)
        self.assertEqual(original["raw"], fresh["raw"])
        self.assertEqual(original["gain_difference_a_minus_c"], fresh["gain_difference_a_minus_c"])
        for label in "abcd":
            old = {key: value for key, value in original["configurations"][label].items()
                   if key != "regime"}
            new = {key: value for key, value in fresh["configurations"][label].items()
                   if key != "regime"}
            self.assertEqual(old, new)
        self.assertEqual(original["warnings"], [s13.REGIME_WARNING])
        self.assertEqual(fresh["warnings"], [])
        self.assertTrue(any(s13.REGIME_WARNING in value for value in old_messages))
        self.assertFalse(any(s13.REGIME_WARNING in value for value in fresh_messages))
        self.assertTrue(any("filename classification" in value for value in fresh_messages))
        self.assertFalse(original["pure_query_independent_inductive_pipeline"])
        self.assertTrue(fresh["pure_query_independent_inductive_pipeline"])
        self.assertFalse(fresh["training_bank_provenance_classification"]["verified"])
        for label in "abd":
            self.assertIn("filename convention", fresh["configurations"][label]["regime"])
            self.assertNotIn("test-calibrated W", fresh["configurations"][label]["regime"])
        self.assertTrue(fresh["configurations"]["c"]["regime"].startswith("transductive:"))
        print("S13 FRESH MOCK 200x100: a/b/c/extra-d metrics and bootstrap identical; max abs error=0")

    def test_real_range_warning_is_preserved_for_fresh_without_mutating_scores(self):
        bank = self.bank * np.float32(4)
        before = bank.copy()
        messages = []
        result = s13.run_analysis(self.sim, bank, self.ann, bootstrap=10,
                                 fresh_bank=True, progress=messages.append)
        np.testing.assert_array_equal(bank, before)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("Fresh bank scores extend beyond [-1,1]", result["warnings"][0])
        self.assertIn("without clipping or renormalization", result["warnings"][0])
        self.assertTrue(any(value.startswith("S13 WARN: Fresh bank") for value in messages))
        self.assertFalse(any(s13.REGIME_WARNING in value for value in messages))

    def test_legacy_cli_does_not_read_new_hashes_or_emit_optional_provenance(self):
        with tempfile.TemporaryDirectory(prefix="tagpr_s13_legacy_mock_") as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            sim_path, bank_path, ann_path = inputs / "sim.npy", inputs / "bank.npy", inputs / "ann.json"
            np.save(sim_path, self.sim)
            np.save(bank_path, self.bank)
            ann_path.write_text(json.dumps(self.records), encoding="utf-8")
            arguments = ["--sim", str(sim_path), "--bank", str(bank_path),
                         "--anno", str(ann_path), "--out", str(root / "out"),
                         "--bootstrap", "10", "--tag", "mock_legacy"]
            import contextlib
            import io
            with mock.patch.object(s13, "_extra_config_provenance", side_effect=AssertionError("legacy audit called")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(s13.main(arguments), 0)
            report_path, = (root / "out").glob("S13_mock_legacy_*.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn("extra_config_provenance", report)

    def test_cli_uses_bank_filename_not_parent_and_never_changes_input_files(self):
        with tempfile.TemporaryDirectory(prefix="tagpr_s13_fresh_mock_") as directory:
            root = Path(directory)
            inputs = root / "inputs_fresh"
            inputs.mkdir()
            sim_path, ann_path = inputs / "sim.npy", inputs / "ann.json"
            np.save(sim_path, self.sim)
            ann_path.write_text(json.dumps(self.records), encoding="utf-8")
            reports = []
            for filename, is_fresh in (("S12_sim_bank_g.npy", False),
                                        ("E3_sim_bank_g_FRESH.npy", True)):
                bank_path = inputs / filename
                np.save(bank_path, self.bank)
                paths = [sim_path, bank_path, ann_path]
                hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
                command = [sys.executable, str(Path(s13.__file__)),
                           "--sim", str(sim_path), "--bank", str(bank_path),
                           "--anno", str(ann_path), "--out", str(root / filename.replace(".npy", "")),
                           "--bootstrap", "100", "--tag", "mock_fresh",
                           "--extra-config", "train_full:0.5:10"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                out = root / filename.replace(".npy", "")
                report_path, = out.glob("S13_mock_fresh_*.json")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                reports.append(report)
                log_path, = out.glob("S13_mock_fresh_*.log")
                self.assertEqual(s13.REGIME_WARNING in result.stdout, not is_fresh)
                self.assertEqual(s13.REGIME_WARNING in log_path.read_text(encoding="utf-8"), not is_fresh)
                self.assertEqual(report["status"], "complete" if is_fresh else "complete_with_warnings")
                self.assertEqual(report["pure_query_independent_inductive_pipeline"], is_fresh)
                self.assertEqual(report["arguments"]["extra_config"], ["train_full:0.5:10"])
                extra_provenance = report["extra_config_provenance"]
                self.assertEqual(extra_provenance["argv"], command[1:])
                self.assertEqual(extra_provenance["seeds"], {"bootstrap": 2026})
                expected_head = subprocess.run(["git", "rev-parse", "HEAD"],
                                               cwd=Path(s13.__file__).resolve().parents[1],
                                               capture_output=True, text=True, check=True).stdout.strip()
                self.assertEqual(extra_provenance["git_HEAD"], expected_head)
                for name, path, digest in zip(("sim", "bank", "anno"), paths, hashes):
                    self.assertEqual(extra_provenance["inputs"][name],
                                     {"path": str(path.resolve()), "sha256_16": digest[:16]})
                self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
            self.assertEqual(reports[0]["raw"], reports[1]["raw"])
            self.assertEqual(reports[0]["gain_difference_a_minus_c"], reports[1]["gain_difference_a_minus_c"])
            print("S13 CLI FRESH MOCK 200x100: filename-only old/fresh branches PASS; JSON/log PASS; inputs unchanged")


if __name__ == "__main__":
    unittest.main()

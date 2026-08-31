"""S13 optional-config regressions on a 200-query x 100-gallery MOCK fixture.

Run: python -m unittest scripts.test_S13_extra_config -v
No archived experiment files are read or written by these tests.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from . import S13_inductive_bootstrap as s13
from ._offline_common import compact_metrics, evaluate
from ._s11_descriptive import paired_test
from .test_S13_bootstrap import fixture


def _without_elapsed(report):
    return {key: value for key, value in report.items() if key != "penalty_elapsed_sec"}


class S13ExtraConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.ann, cls.sim, cls.bank = fixture(100)
        cls.default = s13.run_analysis(cls.sim, cls.bank, cls.ann, progress=None)
        cls.extra = s13.run_analysis(
            cls.sim, cls.bank, cls.ann, progress=None,
            extra_configs=["train_full:0.5:10", "test_full:0.25:20",
                           "train_full:0.75:35", "train_full:0.75:35"])

    def test_default_report_matches_head_bitwise(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "show", "HEAD:scripts/S13_inductive_bootstrap.py"], cwd=root,
            check=True, capture_output=True, text=True, encoding="utf-8")
        baseline = types.ModuleType("scripts._s13_head_reference")
        baseline.__package__ = "scripts"
        exec(compile(result.stdout, "<HEAD:S13>", "exec"), baseline.__dict__)
        before = baseline.run_analysis(self.sim, self.bank, self.ann, progress=None)
        self.assertEqual(_without_elapsed(self.default), _without_elapsed(before))
        arguments = ["--sim", "s.npy", "--bank", "b.npy", "--anno", "a.json", "--out", "out"]
        self.assertEqual(vars(s13.build_parser().parse_args(arguments)),
                         vars(baseline.build_parser().parse_args(arguments)))
        self.assertEqual(list(self.default["configurations"]), ["a", "b", "c"])
        print("S13 EXTRA MOCK 200x100: default full report vs HEAD exact (timing excluded); metric/bootstrap max abs error=0")

    def test_extras_append_without_changing_existing_comparisons(self):
        self.assertEqual(list(self.extra["configurations"]), list("abcdefg"))
        for label in ("a", "b", "c"):
            self.assertEqual(self.extra["configurations"][label], self.default["configurations"][label])
        self.assertEqual(self.extra["gain_difference_a_minus_c"],
                         self.default["gain_difference_a_minus_c"])
        self.assertEqual(self.extra["configurations"]["f"], self.extra["configurations"]["g"])

    def test_extra_penalties_and_all_three_units_match_independent_reference(self):
        raw = evaluate(self.sim, self.ann)
        for label in "defg":
            result = self.extra["configurations"][label]
            values = self.bank if result["bank"] == "train_full" else self.sim
            # Full-sort reference, independent of the streaming top-k implementation.
            penalty = np.sort(values, axis=0)[-result["k"]:][::-1].mean(axis=0, dtype=np.float32)
            corrected = evaluate(self.sim, self.ann, penalty=penalty, alpha=result["alpha"])
            self.assertEqual(result["metrics"], compact_metrics(corrected))
            self.assertEqual(result["penalty"]["sha256_native_array_bytes"],
                             hashlib.sha256(penalty.tobytes()).hexdigest())
            self.assertEqual(result["penalty"]["n_bank"], len(values))
            for unit, groups in (("query", None), ("image", self.ann.query_source_indices),
                                 ("identity", self.ann.query_ids)):
                expected = paired_test(raw["details"]["correct"], corrected["details"]["correct"],
                                       groups=groups, repeats=10000, seed=2026)
                expected["resampling_unit"] = unit
                self.assertEqual(result["bootstrap"][unit], expected)
        self.assertTrue(self.extra["configurations"]["e"]["regime"].startswith("transductive:"))
        item = self.extra["configurations"]["d"]["bootstrap"]["identity"]
        print("S13 EXTRA MOCK d=train_full:0.5:10; 10000 identity resamples: "
              f"observed_delta_pp={item['observed_delta_pp']}, "
              f"mean_delta_pp={item['bootstrap_mean_delta_pp']}, "
              f"CI95={item['ci95_percentile_pp']}, p={item['p_two_sided']}; "
              "all d/e/f/g reference comparisons exact")

    def test_new_bank_topk_is_computed_once_per_key_and_default_calls_unchanged(self):
        with mock.patch.object(s13, "topk_values", wraps=s13.topk_values) as topk:
            s13.run_analysis(self.sim, self.bank, self.ann, bootstrap=10, progress=None,
                             extra_configs=["train_full:0.5:10", "test_full:0.25:20",
                                            "train_full:0.5:35", "train_full:0.75:35"])
        self.assertEqual([call.args[1] for call in topk.call_args_list], [30, 10, 20, 35])
        self.assertIs(topk.call_args_list[0].args[0], self.bank)
        self.assertIs(topk.call_args_list[1].args[0], self.sim)
        self.assertIs(topk.call_args_list[2].args[0], self.sim)
        self.assertIs(topk.call_args_list[3].args[0], self.bank)

    def test_argument_defaults_validation_and_repetition(self):
        base = ["--sim", "s.npy", "--bank", "b.npy", "--anno", "a.json", "--out", "out"]
        self.assertNotIn("extra_config", vars(s13.build_parser().parse_args(base)))
        self.assertEqual(s13.build_parser().parse_args(base).bootstrap, 10000)
        parsed = s13.build_parser().parse_args(
            base + ["--extra-config", "train_full:0.5:10", "--extra-config", "test_full:0:1"])
        self.assertEqual(parsed.extra_config, ["train_full:0.5:10", "test_full:0:1"])
        invalid = ["", "train_full:0.5", "train_full:0.5:10:extra", "unknown:0.5:10",
                   "train_full:nan:10", "train_full:inf:10", "train_full:-0.5:10",
                   "train_full:0.5:0", "train_full:0.5:-1", "train_full:0.5:1.5"]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    s13._parse_extra_config(value)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    s13.build_parser().parse_args(base + ["--extra-config", value])
        self.assertEqual(s13._parse_extra_config("train_full:0:1"), ("train_full", 0.0, 1))
        self.assertEqual(s13._parse_extra_config("train_full:1:300"), ("train_full", 1.0, 300))
        for value in ("train_full:0.5:301", "test_full:0.5:201"):
            with self.assertRaisesRegex(ValueError, "exceeds .* rows="):
                s13.run_analysis(self.sim, self.bank, self.ann, progress=None, extra_configs=[value])
        self.assertEqual([s13._config_label(i) for i in (3, 4, 25, 26, 27, 51, 52)],
                         ["d", "e", "z", "aa", "ab", "az", "ba"])

    def test_cli_repeated_option_report_and_inputs_immutable(self):
        with tempfile.TemporaryDirectory(prefix="tagpr_s13_extra_mock_") as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            sim_path, bank_path, ann_path = inputs / "sim.npy", inputs / "bank.npy", inputs / "ann.json"
            np.save(sim_path, self.sim)
            np.save(bank_path, self.bank)
            ann_path.write_text(json.dumps(self.records), encoding="utf-8")
            paths = [sim_path, bank_path, ann_path]
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            command = [sys.executable, str(Path(s13.__file__)), "--sim", str(sim_path),
                       "--bank", str(bank_path), "--anno", str(ann_path), "--out", str(root / "out"),
                       "--tag", "mock_extra", "--extra-config", "train_full:0.5:10",
                       "--extra-config", "test_full:0.25:20"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            reports = list((root / "out").glob("S13_mock_extra_*.json"))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(list(report["configurations"]), list("abcde"))
            self.assertEqual(report["arguments"]["extra_config"],
                             ["train_full:0.5:10", "test_full:0.25:20"])
            for label in "abcde":
                self.assertEqual(report["configurations"][label], self.extra["configurations"][label])
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
            self.assertEqual(report["bootstrap_protocol"]["repeats"], 10000)
            self.assertTrue(list((root / "out").glob("S13_mock_extra_*.log")))
            print("S13 EXTRA CLI MOCK 200x100: repeated --extra-config JSON/log PASS; 10000 repeats; input hashes unchanged")


if __name__ == "__main__":
    unittest.main()

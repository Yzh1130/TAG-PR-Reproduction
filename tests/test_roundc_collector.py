"""CPU-only fixture trees; no training and no real experiment mutation."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from scripts.revision import collect_revision_runs as collector


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "experiments"
        self.root.mkdir()

    def run_directory(self, name, seed=0, rows=None):
        path = self.root / name
        path.mkdir()
        config = {"misc": {"seed": seed}, "data": {"batch_size": 320},
                  "schedule": {"epoch": 99, "niter_per_ep": 777},
                  "experiment": {"itc_block_size": 0, "grad_checkpoint": True,
                                 "eval_router_mode": "gumbel", "eval_seed": None}}
        (path / "config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        argv = ["main.py", "--config", "/original/cfg_b16.yaml", "--exp_name", name,
                "--seed", str(seed), "--batch_size", "320", "--grad_checkpoint"]
        (path / "run_command.txt").write_text(
            "command: python " + " ".join(argv) + "\ncwd: /repo\n"
            "config_source: /original/cfg_b16.yaml\nsys.argv: " + json.dumps(argv) + "\n",
            encoding="utf-8")
        (path / "git_commit.txt").write_text(
            "git rev-parse HEAD:\n" + "a" * 40 + "\n\nWORKTREE: CLEAN\n", encoding="utf-8")
        if rows is not None:
            lines = ["epoch,R1,R5,R10,mAP,train_loss,epoch_time_sec"]
            for epoch, r1 in enumerate(rows, 1):
                lines.append(f"{epoch},{r1},{r1 + 10},{r1 + 20},{r1 - 10},1,2")
            (path / "metrics_by_epoch.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_01_new_run_exact_fields_and_real_iterations(self):
        path = self.run_directory("e3_c0_b0_seed0", rows=[51.123456789, 53.0])
        (path / "train.log").write_text(
            "2026 TAG INFO: Epoch[1] Iteration[50/124], loss: 1.0\n"
            "2026 TAG INFO: Epoch[1] Iteration[100/124], loss: 1.0\n"
            "2026 TAG INFO: Epoch 1 done. Time per batch: 1[s]\n"
            "2026 TAG INFO: Acc@1 51.12346 Acc@5 61.12346 Acc@10 71.12346 mAP 41.12346\n"
            "2026 TAG INFO: CSLS Acc@1 99.00000 Acc@5 99.00000 Acc@10 99.00000 mAP 99.00000\n"
            "2026 TAG INFO: Epoch[2] Iteration[50/124], loss: 1.0\n"
            "2026 TAG INFO: Epoch 2 done. Time per batch: 1[s]\n"
            "2026 TAG INFO: Acc@1 53.00000 Acc@5 63.00000 Acc@10 73.00000 mAP 43.00000\n",
            encoding="utf-8")
        (path / "final_results.txt").write_text(
            "best_epoch: 2\nR@1: 53.0\nR@5: 63.0\nR@10: 73.0\nmAP: 43.0\n"
            "completed_epochs: 2\nrun_status: completed\n", encoding="utf-8")
        checkpoint = path / "checkpoint_last.pth"
        checkpoint.write_bytes(b"fixture checkpoint without pickle")
        run = collector.parse_run(path)
        for key, value in {"exp_name": path.name, "seed": 0, "batch_size": 320,
                           "itc_block_size": 0, "grad_checkpoint": True,
                           "eval_router_mode": "gumbel", "eval_seed": None,
                           "training_git_commit": "a" * 40, "completed_epochs": 2}.items():
            self.assertEqual(run[key], value)
        self.assertNotIn("eval_seed", run["missing_fields"])
        self.assertIsNone(run["config_source_sha256_16"])
        self.assertEqual(run["config_used_sha256_16"], hashlib.sha256(
            (path / "config_used.yaml").read_bytes()).hexdigest()[:16])
        self.assertEqual(run["epochs"][0]["metrics"]["R1"], 51.123456789)
        self.assertEqual([entry["actual_iterations"] for entry in run["epochs"]], [124, 124])
        self.assertEqual(run["epochs"][0]["observed_iteration_max"], 100)
        self.assertEqual(run["best"]["epoch"], 2)
        self.assertEqual(run["last"]["metrics"], {"R1": 53., "R5": 63., "R10": 73., "mAP": 43.})
        self.assertEqual(run["checkpoints"][0]["bytes"], checkpoint.stat().st_size)
        self.assertEqual(run["checkpoints"][0]["sha256_16"], hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16])

    def test_02_historical_stdout_and_namespace_no_archive_defaults(self):
        path = self.root / "20260821_bs320_ep10_seed99"
        path.mkdir()
        (path / "stdout.log").write_text(
            "Namespace(config='/history/cfg.yaml', exp_name='bs320_ep10', seed=7, batch_size=320, grad_checkpoint=False)\n"
            "{'misc': {'seed': 7}, 'data': {'batch_size': 320}, 'experiment': {'grad_checkpoint': False}}\n"
            "Epoch[1] Iteration[50/124], loss: 1.0\nEpoch 1 done.\n"
            "Acc@1 47.12500 Acc@5 57.12500 Acc@10 67.12500 mAP 37.12500\n"
            "Epoch[2] Iteration[50/124], loss: 1.0\nEpoch 2 done.\n"
            "Acc@1 47.12500 Acc@5 58.00000 Acc@10 68.00000 mAP 38.00000\n"
            "RERANK Acc@1 90.00000 Acc@5 90.00000 Acc@10 90.00000 mAP 90.00000\n",
            encoding="utf-8")
        run = collector.discover_runs([self.root], ["20260821_bs320"])[0]
        self.assertEqual(run["seed"], 7)  # Not the filename's seed99.
        self.assertEqual(run["exp_name"], "bs320_ep10")
        self.assertEqual(run["completed_epochs"], 2)
        self.assertEqual(run["best"]["epoch"], 1)  # Strict R1 improvement, earliest tie.
        self.assertEqual(run["last"]["metrics"]["R5"], 58.)
        self.assertIsNone(run["itc_block_size"])
        self.assertIsNone(run["eval_router_mode"])
        self.assertIsNone(run["training_git_commit"])
        self.assertEqual(run["effective_configuration_source"], "training_log_literal_configuration")
        self.assertIn("config_used_sha256_16", run["missing_fields"])

    def test_03_partial_epoch_does_not_infer_completion_or_total(self):
        path = self.run_directory("e2_partial_seed0")
        (path / "train.log").write_text("Epoch[1] Iteration[50/124], loss: 1.0\n", encoding="utf-8")
        run = collector.parse_run(path)
        self.assertIsNone(run["completed_epochs"])
        self.assertIsNone(run["last"])
        self.assertIsNone(run["epochs"][0]["actual_iterations"])
        self.assertEqual(run["epochs"][0]["measured_loader_iterations"], 124)
        self.assertIn("epochs.1.actual_iterations", run["missing_fields"])
        self.assertNotEqual(run["completed_epochs"], 99)

    def test_04_conflicting_rows_and_missing_last_metrics_are_not_guessed(self):
        path = self.run_directory("e2_partial_csv", rows=[10])
        with (path / "metrics_by_epoch.csv").open("a", encoding="utf-8") as handle:
            handle.write("1,11,21,31,1,1,2\n")
        (path / "final_results.txt").write_text("completed_epochs: 2\n", encoding="utf-8")
        run = collector.parse_run(path)
        self.assertIsNone(run["epochs"][0]["metrics"]["R1"])
        self.assertEqual(run["last"]["epoch"], 2)
        self.assertIsNone(run["last"]["metrics"]["R1"])
        self.assertTrue(run["warnings"])

    def test_05_pair_hand_calculated_summary_and_last_best_are_separate(self):
        runs = []
        for seed, delta in enumerate((-1., 0., 2.)):
            a = collector.parse_run(self.run_directory(f"e3_c0_b0_seed{seed}", seed, [50., 40.]))
            b = collector.parse_run(self.run_directory(f"e3_c1_b80_seed{seed}", seed, [50., 40. + delta]))
            runs += [a, b]
        pair = collector.pair_runs(runs, "e3_c0_b0|e3_c1_b80", "last")
        summary = pair["summary"]["R1"]
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean"], 1 / 3)
        self.assertAlmostEqual(summary["population_std"], math.sqrt(14) / 3)
        self.assertEqual(summary["range"], 3.)
        self.assertEqual(summary["sign_counts"], {"negative": 1, "zero": 1, "positive": 1})
        self.assertEqual(pair["summary_A"]["R1"]["mean"], 40.)
        self.assertAlmostEqual(pair["summary_B"]["R1"]["mean"], 40. + 1 / 3)
        best_pair = collector.pair_runs(runs, "e3_c0_b0|e3_c1_b80", "best")
        self.assertEqual(best_pair["summary"]["R1"]["mean"], 0.)
        self.assertEqual(pair["metric"], "last")
        self.assertEqual(best_pair["metric"], "best")

    def test_06_pair_missing_duplicate_seed_and_no_name_inference(self):
        runs = [collector.parse_run(self.run_directory("A_seed0", 0, [50])),
                collector.parse_run(self.run_directory("B_seed0", 0, [52])),
                collector.parse_run(self.run_directory("B_seed0_duplicate", 0, [52])),
                collector.parse_run(self.run_directory("A_seed1", 1, [50]))]
        unknown = self.root / "A_seed999"
        unknown.mkdir()
        (unknown / "checkpoint_best.pth").write_bytes(b"legacy")
        runs.append(collector.parse_run(unknown))
        pair = collector.pair_runs(runs, "A_|B_", "last")
        self.assertEqual([row["status"] for row in pair["rows"]], ["unpaired", "unpaired"])
        self.assertEqual(pair["summary"]["R1"]["n"], 0)
        self.assertEqual(len(pair["unknown_seed_unpaired"]), 1)
        self.assertIsNone(runs[-1]["seed"])

    def test_07_cli_exclusive_collision_outputs_and_input_unchanged(self):
        path = self.run_directory("e3_c0_b0_seed0", rows=[50.])
        other_root = self.base / "second_root"
        other_root.mkdir()
        before = {item: item.read_bytes() for item in path.iterdir() if item.is_file()}
        output = self.base / "analysis"
        argv = ["--root", str(self.root), "--root", str(other_root), "--out", str(output),
                "--tag", "fixture", "--pair", "e3_c0_b0|e3_c1_b80", "--by", "seed", "--metric", "last"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(collector.main(argv), 0)
            self.assertEqual(collector.main(argv), 0)
        self.assertEqual(len(list(output.glob("runs_*.json"))), 2)
        report = json.loads((output / "runs_fixture.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["provenance"]["roots"]), 2)
        self.assertTrue(report["provenance"]["read_only_scan"])
        self.assertEqual(report["metric_scope"], "last")
        self.assertTrue((output / "pairs_last_fixture.csv").is_file())
        self.assertEqual({item: item.read_bytes() for item in before}, before)

    def test_08_errors_do_not_write_into_excluded_experiments(self):
        path = self.run_directory("not_included", rows=[50])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                collector.main(["--root", str(self.root), "--include", "e3_",
                                "--out", str(path / "reports"), "--tag", "bad"])
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse((path / "reports").exists())
        with self.assertRaises(ValueError):
            collector.discover_runs([self.base / "missing"], [])
        with self.assertRaises(ValueError):
            collector.pair_runs([], "malformed", "last")

    def test_09_help_and_config_source_hash_only_if_archived(self):
        help_text = collector.build_parser().format_help()
        for option in ("--root", "--include", "--pair", "--by", "--metric", "--out", "--tag"):
            self.assertIn(option, help_text)
        path = self.run_directory("known_hash", rows=[50])
        with (path / "run_command.txt").open("a", encoding="utf-8") as handle:
            handle.write("config_source_sha256: " + "b" * 64 + "\n")
        run = collector.parse_run(path)
        self.assertEqual(run["config_source_sha256_16"], "b" * 16)
        self.assertNotIn("config_source_sha256_16", run["missing_fields"])

    def test_10_logged_denominator_conflict_and_final_iteration_evidence(self):
        path = self.run_directory("mixed_progress")
        (path / "train.log").write_text("Epoch[1] Iteration[50/100]\nEpoch[1] Iteration[60/124]\nEpoch 1 done.\n"
                                       "Epoch[2] Iteration[124/124]\n", encoding="utf-8")
        run = collector.parse_run(path)
        self.assertIsNone(run["epochs"][0]["actual_iterations"])
        self.assertEqual(run["epochs"][1]["actual_iterations"], 124)
        self.assertEqual(run["completed_epoch_numbers_observed"], [1])
        self.assertTrue(run["warnings"])

    def test_11_include_filters_before_checkpoint_hash_and_uses_archived_exp_name(self):
        excluded = self.run_directory("large_unwanted", rows=[50])
        (excluded / "checkpoint_best.pth").write_bytes(b"must not be hashed")
        included = self.run_directory("dated_name_not_matching", rows=[50])
        command = included / "run_command.txt"
        command.write_text(command.read_text(encoding="utf-8").replace(
            "dated_name_not_matching", "e3_explicit_name"), encoding="utf-8")
        original = collector.file_evidence
        def guarded(path):
            if path.parent == excluded:
                raise AssertionError("excluded run must not be hashed")
            return original(path)
        with mock.patch.object(collector, "file_evidence", side_effect=guarded):
            runs = collector.discover_runs([self.root], ["e3_"])
        self.assertEqual([run["exp_name"] for run in runs], ["e3_explicit_name"])

    def test_12_no_grad_precedence_and_eval_only_no_training_completion(self):
        path = self.root / "eval_only"
        path.mkdir()
        (path / "run_command.txt").write_text("sys.argv: " + json.dumps([
            "main.py", "--grad_checkpoint", "--no_grad_checkpoint", "--eval_only"]) + "\n", encoding="utf-8")
        (path / "metrics_by_epoch.csv").write_text(
            "epoch,R1,R5,R10,mAP\n0,50,60,70,40\n", encoding="utf-8")
        run = collector.parse_run(path)
        self.assertEqual(run["grad_checkpoint"], False)
        self.assertTrue(run["evaluation_only"])
        self.assertIsNone(run["completed_epochs"])
        self.assertIsNone(run["best"])
        self.assertIsNone(run["last"])
        self.assertTrue(any("epoch=0" in warning for warning in run["warnings"]))

    def test_13_excluded_stdout_only_history_is_also_read_only(self):
        path = self.root / "historical_not_included"
        path.mkdir()
        (path / "stdout.log").write_text("Epoch[1] Iteration[50/100]\nEpoch 1 done.\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                collector.main(["--root", str(self.root), "--include", "e3_", "--out",
                                str(path / "analysis"), "--tag", "forbidden"])
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse((path / "analysis").exists())

    def test_14_changing_input_is_not_assigned_a_snapshot_hash(self):
        path = self.base / "checkpoint.pth"
        path.write_bytes(b"fixture")
        metadata = path.stat()
        before = SimpleNamespace(st_size=metadata.st_size, st_mtime_ns=metadata.st_mtime_ns)
        after = SimpleNamespace(st_size=metadata.st_size + 1, st_mtime_ns=metadata.st_mtime_ns)
        with mock.patch.object(Path, "stat", side_effect=[before, after]):
            with self.assertRaisesRegex(ValueError, "input changed while hashing"):
                collector.file_evidence(path)


if __name__ == "__main__":
    unittest.main()

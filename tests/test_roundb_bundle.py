"""Round-B D4/D7 CPU mock integration; no training or GPU eval is run.

All retrieval values come from actual 200x100 mock matrices. Model loading and
feature extraction alone are mocked; score writing, metric computation, scans,
bootstrap, audit hashing and the shell's Python aggregator run for real.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

import numpy as np

from scripts import E3_fresh_forward as e3
from scripts import S12_inductive_bank as s12
from scripts import S13_inductive_bootstrap as s13
from scripts._offline_common import evaluate, topk_mean
from scripts._s11_descriptive import paired_test
from scripts.test_E3_fresh_forward import (_close_memmaps, manual_scores,
                                          synthetic_fixture)
from scripts.test_S13_bootstrap import fixture


REPO = Path(__file__).resolve().parents[1]
BASELINE = "8622dc2540fb28e20c4c344204ef75603c8f3b3b"
FIELDS = ("R1", "R5", "R10", "mAP")


def git_source(relative):
    return subprocess.check_output(["git", "show", f"{BASELINE}:{relative}"],
                                   cwd=REPO, text=True, encoding="utf-8")


def reference_module(relative, name):
    module = ModuleType(name)
    module.__package__ = "scripts"
    module.__file__ = str(REPO / relative)
    exec(compile(git_source(relative), module.__file__, "exec"), module.__dict__)
    return module


def without_runtime(value):
    """Exclude only measured wall-clock fields, never scores or scan rows."""
    if isinstance(value, dict):
        return {key: without_runtime(item) for key, item in value.items()
                if key not in ("elapsed_sec", "penalty_elapsed_sec")}
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


@unittest.skipIf(torch is None, "PyTorch required for real CPU score writing")
class E3BundleTests(unittest.TestCase):
    def pipeline(self, archive_present):
        data = synthetic_fixture(queries=200, galleries=100, training=300)
        with tempfile.TemporaryDirectory(prefix="roundb_e3_200x100_mock_") as directory:
            root = Path(directory)
            inputs, archive_dir, out = root / "inputs", root / "archive", root / "out"
            inputs.mkdir()
            archive_dir.mkdir()
            out.mkdir()
            anno, train = inputs / "test_reid.json", inputs / "train_reid.json"
            anno.write_text(json.dumps(data.records), encoding="utf-8")
            train.write_text(json.dumps(data.train_records), encoding="utf-8")
            checkpoint, config_path = inputs / "checkpoint.pth", inputs / "config.yaml"
            checkpoint.write_bytes(b"MOCK checkpoint bytes; no actual model load")
            config_path.write_text("mock: true\n", encoding="utf-8")
            archive = archive_dir / "sim_qg.npy"
            np.save(archive, manual_scores(data.test, data.gallery, data.ann.gallery_views))
            hash_before = hashlib.sha256(archive.read_bytes()).hexdigest()
            config = SimpleNamespace(experiment=SimpleNamespace(max_words=30, text_length=77,
                view_prompt=False, eval_router_mode="argmax", eval_seed=17), image_dir="mock")
            events, captured = [], {}

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.parameter = torch.nn.Parameter(torch.ones(2))
                    self.register_buffer("ema", torch.zeros(2))

            def load(*args):
                seed = args[5]
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                events.append("load")
                return Model().eval(), config, {"mock": True, "seed": seed}

            def images(*args, **kwargs):
                events.append("gallery_images")
                torch.rand(3)
                return data.gallery

            def texts(model, captions, **kwargs):
                branch = "test" if captions[0].startswith("test") else "train"
                events.append(branch + "_text")
                return getattr(data, branch)

            def scan(sim, bank, gg, ann):
                captured.update(sim=np.asarray(sim).copy(), bank=np.asarray(bank).copy(), gg=np.asarray(gg).copy())
                # Only the calibration budgets adapt to the MOCK bank size;
                # all 352 original method/bank/hyperparameter computations run.
                return s12.run_corrections(sim, bank, gg, ann, subset_sizes=(100, 200, 250), progress=None)

            real_archive_loader = e3.load_similarity

            def read_archive(*args, **kwargs):
                self.assertTrue(archive_present, "omitted --sim must never read any archive")
                self.assertEqual(events, ["load", "gallery_images", "test_text", "train_text"])
                self.assertEqual(len(list(out.glob("*.npy"))), 3)
                return real_archive_loader(*args, **kwargs)

            command = ["--fresh-forward", "--anno", str(anno), "--train-anno", str(train),
                       "--resume", str(checkpoint), "--config", str(config_path), "--out", str(out),
                       "--device", "cpu", "--seed", "0", "--score-block-size", "19", "--tag", "mock bundle"]
            if archive_present:
                command += ["--sim", str(archive)]
            stdout = io.StringIO()
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(stdout))
                stack.enter_context(redirect_stderr(stdout))
                stack.enter_context(patch.object(e3, "load_model", side_effect=load))
                image_mock = stack.enter_context(patch.object(e3, "extract_gallery_features", side_effect=images))
                text_mock = stack.enter_context(patch.object(e3, "extract_text_features", side_effect=texts))
                stack.enter_context(patch.object(e3, "preprocess_texts", side_effect=lambda rows, max_words:
                    [caption for row in rows for caption in row["captions"]]))
                archive_mock = stack.enter_context(patch.object(e3, "load_similarity", side_effect=read_archive))
                scan_mock = stack.enter_context(patch.object(e3, "run_corrections", side_effect=scan))
                code = e3.main(command)
                self.assertEqual(code, 0, stdout.getvalue())
                reports = list(out.glob("E3_mock_bundle_*.json"))
                self.assertEqual(len(reports), 1, stdout.getvalue())
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                self.assertEqual(image_mock.call_count, 1)
                self.assertEqual(text_mock.call_count, 2)
                self.assertEqual(archive_mock.call_count, int(archive_present))
                self.assertEqual(scan_mock.call_count, 1)
                _close_memmaps(scan_mock.call_args.args)
                scan_mock.reset_mock()
                self.assertEqual(hash_before, hashlib.sha256(archive.read_bytes()).hexdigest())
                self.assertEqual(report["status"], "complete")
                self.assertEqual(len(report["corrections"]["rows"]), 352)
                self.assertEqual(report["fresh_shapes"]["sim_qg"], [200, 100])
                self.assertEqual(report["fresh_shapes"]["sim_bank_g"], [300, 100])
                provenance = report["provenance"]
                self.assertEqual(provenance["checkpoint_sha256_16"], hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16])
                self.assertEqual(provenance["inputs"]["checkpoint"]["bytes"], checkpoint.stat().st_size)
                self.assertEqual(provenance["inputs"]["checkpoint"]["sha256_16"], provenance["checkpoint_sha256_16"])
                self.assertEqual(provenance["eval_router_mode"], "argmax")
                self.assertEqual(provenance["initial_extraction_seed"], 0)
                self.assertEqual(provenance["configured_eval_seed"], 17)
                self.assertIsNone(provenance["eval_seed"])
                self.assertFalse(provenance["eval_seed_reset_applied"])
                self.assertEqual(len(provenance["git_HEAD"]), 40)
                self.assertEqual(len(provenance["gallery_image_inputs"]), 100)
                self.assertTrue(all(not item["available"] for item in provenance["gallery_image_inputs"]))
                self.assertIn("--tag", provenance["argv"])
                self.assertTrue(report["pure_query_independent_inductive_pipeline"])
                self.assertTrue(all(g["rng_unchanged"] for g in report["independence_guards"]))
                if not archive_present:
                    self.assertNotIn("archived_similarity", provenance["inputs"])
                    self.assertTrue(report["archive_comparison_skipped"])
                    self.assertIsNone(report["archived_metrics"])
                    self.assertIsNone(report["fresh_minus_archived_metrics_pp"])
                    self.assertIn("COMPARISON SKIPPED", stdout.getvalue())
                else:
                    archive_result = evaluate(np.load(archive), data.ann)
                    for field in FIELDS:
                        self.assertEqual(report["archived_metrics"][field], archive_result[field])
                result = report, captured, events
            gc.collect()
            return result

    def test_no_archive_access_352_rows_and_full_provenance(self):
        report, captured, events = self.pipeline(False)
        self.assertEqual(events, ["load", "gallery_images", "test_text", "train_text"])
        self.assertEqual(captured["sim"].shape, (200, 100))
        self.assertEqual(captured["bank"].shape, (300, 100))
        print("ROUND B E3 MOCK omitted --sim: 200x100 qg, 300x100 bank, 352 real scan rows; archive reads=0; checkpoint/hash/router/seed/git provenance PASS")

    def test_explicit_archive_changes_only_reference_reporting(self):
        omitted, left, left_events = self.pipeline(False)
        explicit, right, right_events = self.pipeline(True)
        self.assertEqual(left_events, right_events)
        for matrix in ("sim", "bank", "gg"):
            np.testing.assert_array_equal(left[matrix], right[matrix])
        for field in ("fresh_metrics", "corrections", "feature_shapes", "independence_guards"):
            self.assertEqual(without_runtime(omitted[field]), without_runtime(explicit[field]))
        self.assertFalse(explicit["archive_comparison_skipped"])
        print("ROUND B E3 MOCK explicit vs omitted --sim: all fresh matrix bytes, 352 correction rows, extraction order EXACT; only archive-reference report differs")


class S13BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.ann, cls.sim, cls.bank = fixture()

    def test_legacy_default_run_analysis_exact_against_baseline_commit(self):
        baseline = reference_module("scripts/S13_inductive_bootstrap.py", "scripts._roundb_s13_reference")
        kwargs = dict(bootstrap=10000, seed=2026, block_size=19, progress=None)
        before = baseline.run_analysis(self.sim, self.bank, self.ann, **kwargs)
        after = s13.run_analysis(self.sim, self.bank, self.ann, **kwargs)
        before.pop("penalty_elapsed_sec")
        after.pop("penalty_elapsed_sec")
        self.assertEqual(before, after)
        self.assertEqual(list(after["configurations"]), ["a", "b", "c"])
        parser = s13.build_parser().parse_args(["--sim", "s", "--bank", "b", "--anno", "a", "--out", "o"])
        self.assertFalse(hasattr(parser, "summary_table"))
        print("ROUND B S13 MOCK no flag vs git 8622dc2 run_analysis: complete a/b/c report EXACT (timing excluded), bootstrap=10000")

    def test_summary_four_metrics_and_paired_identity_intervals_exact(self):
        summary = s13.build_summary_table(self.sim, self.bank, self.ann, bootstrap=10000, seed=2026, block_size=19)
        self.assertEqual(len(summary["rows"]), 4)
        raw = evaluate(self.sim, self.ann)
        self.assertEqual(summary["rows"][0]["metrics"], {k: raw[k] for k in FIELDS})
        for row in summary["rows"][1:]:
            matrix = self.sim if row["bank"] == "test_full" else self.bank
            penalty = np.sort(matrix, axis=0)[-row["k"]:][::-1].mean(0, dtype=np.float32)
            expected = evaluate(self.sim, self.ann, penalty=penalty, alpha=row["alpha"])
            self.assertEqual(row["metrics"], {k: expected[k] for k in FIELDS})
            ci = paired_test(raw["details"]["correct"], expected["details"]["correct"],
                             groups=self.ann.query_ids, repeats=10000, seed=2026)
            self.assertEqual(row["identity_bootstrap"], ci)
            self.assertEqual(row["delta_R1_pp"], ci["observed_delta_pp"])
        self.assertFalse(summary["selection_from_test_metrics"])
        print("ROUND B S13 MOCK summary: four rows, all four metrics EXACT; identity paired CI EXACT x10000; no test-based selection")

    def test_cli_optional_summary_new_file_and_non_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="roundb_s13_200x100_mock_") as directory:
            root = Path(directory)
            data = root / "inputs"
            data.mkdir()
            sim, bank, anno = data / "sim.npy", data / "bank_fresh.npy", data / "ann.json"
            np.save(sim, self.sim)
            np.save(bank, self.bank)
            anno.write_text(json.dumps(self.records), encoding="utf-8")
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sim, bank, anno)}
            output = root / "out"
            command = ["--sim", str(sim), "--bank", str(bank), "--anno", str(anno), "--out", str(output),
                       "--tag", "mock", "--bootstrap", "100", "--summary-table"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(s13.main(command), 0)
            summary_file = output / "S13_mock_summary_table.json"
            initial = summary_file.read_bytes()
            report = json.loads(initial)
            self.assertEqual(len(report["rows"]), 4)
            self.assertEqual(len(report["provenance"]["inputs"]), 3)
            self.assertEqual(report["provenance"]["seeds"], {"bootstrap": 2026})
            for role in report["provenance"]["inputs"].values():
                self.assertEqual(len(role["sha256_16"]), 16)
                self.assertGreater(role["bytes"], 0)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(s13.main(command), 0)
            self.assertEqual(summary_file.read_bytes(), initial)
            self.assertEqual(len(list(output.glob("S13_mock_summary_table*.json"))), 2)
            self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (sim, bank, anno)})
            legacy_reports = [json.loads(p.read_text()) for p in output.glob("S13_mock_*.json") if "summary_table" not in p.name]
            self.assertEqual(len(legacy_reports), 2)
            self.assertTrue(all(list(r["configurations"]) == ["a", "b", "c"] for r in legacy_reports))
            print("ROUND B S13 MOCK CLI --summary-table: audited four-row file, legacy a/b/c preserved; repeated name timestamped; all input hashes unchanged")


class RouterShellTests(unittest.TestCase):
    shell_path = REPO / "scripts/revision/e6_router_randomness.sh"

    def test_bash_help_and_syntax_no_gpu_execution(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            candidate = Path("C:/Program Files/Git/bin/bash.exe")
            bash = str(candidate) if candidate.is_file() else bash
        if bash is None:
            self.skipTest("Bash unavailable; no GPU command is executed")
        for options in (("-n", str(self.shell_path)), (str(self.shell_path), "--help")):
            completed = subprocess.run([bash, *options], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            if "--help" in options:
                self.assertIn("N_GUMBEL_SEEDS", completed.stdout)
        print("ROUND B E6 MOCK bash -n / --help PASS; no shell eval/training command executed")

    def aggregate(self, bad=None):
        _, ann, sim, bank = fixture()
        code = self.shell_path.read_text(encoding="utf-8").split("<<'PY_SUMMARY'\n", 1)[1].split("\nPY_SUMMARY", 1)[0]
        raw = evaluate(sim, ann)
        candidates = [evaluate(sim, ann, penalty=topk_mean(bank, k), alpha=alpha)
                      for k, alpha in ((10, .5), (30, .75))]
        with tempfile.TemporaryDirectory(prefix="roundb_e6_200x100_mock_") as directory:
            root = Path(directory)
            config, checkpoint = root / "config.yaml", root / "checkpoint.pth"
            config.write_text("seed: 0\n", encoding="utf-8")
            checkpoint.write_bytes(b"mock-checkpoint-no-model-load")
            for label, mode, seed, metrics in (("argmax", "argmax", 0, raw),
                ("gumbel_seed0", "gumbel", 0, candidates[0]),
                ("gumbel_seed1", "gumbel", 1, candidates[1])):
                run = root / label / "mock_run"
                run.mkdir(parents=True)
                payload = dict(eval_router_mode=mode, eval_seed=seed, csls=None, reranked=None,
                               original={old: metrics[new] for old, new in zip(("r1", "r5", "r10", "mAP"), FIELDS)})
                if bad == "router" and label == "gumbel_seed0":
                    payload["eval_router_mode"] = "argmax"
                if bad == "nonfinite" and label == "gumbel_seed0":
                    payload["original"]["r1"] = float("nan")
                (run / "evaluation_revision.json").write_text(json.dumps(payload), encoding="utf-8")
                (run / "config_used.yaml").write_text("misc:\n  seed: 7\n", encoding="utf-8")
            completed = subprocess.run([sys.executable, "-c", code, str(root), "2", str(config), str(checkpoint),
                                        "ViT-B/16", str(REPO), str(root)],
                                       capture_output=True, text=True, timeout=30)
            if bad:
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse((root / "router_randomness.json").exists())
                return completed.stderr
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((root / "router_randomness.json").read_text(encoding="utf-8"))
            fields = ("r1", "r5", "r10", "mAP")
            for row, expected in zip(report["gumbel"], candidates):
                for old, new in zip(fields, FIELDS):
                    self.assertEqual(row["metrics"][old], expected[new])
                    self.assertEqual(row["minus_argmax_pp"][old], expected[new]-raw[new])
            for old, new in zip(fields, FIELDS):
                self.assertEqual(report["observed_maximum_pairwise_difference_pp"][old],
                                 abs(candidates[0][new]-candidates[1][new]))
            self.assertFalse(report["training_executed"])
            self.assertFalse(report["postprocessing_enabled"])
            self.assertIn("n=2", report["maximum_scope"])
            self.assertEqual(report["provenance"]["seeds"]["gumbel_eval"], [0, 1])
            self.assertEqual(report["provenance"]["seeds"]["startup_misc_seed_by_run"],
                             {"argmax": 7, "gumbel_seed0": 7, "gumbel_seed1": 7})
            self.assertEqual(len(report["provenance"]["git_HEAD"]), 40)
            self.assertEqual(report["provenance"]["argv"][0], "bash")
            for path, record in report["provenance"]["inputs"].items():
                source = Path(path)
                self.assertEqual(record["bytes"], source.stat().st_size)
                self.assertEqual(record["sha256_16"], hashlib.sha256(source.read_bytes()).hexdigest()[:16])
            # Exact same code uses exclusive output creation, even in a mock.
            repeated = subprocess.run([sys.executable, "-c", code, str(root), "2", str(config), str(checkpoint),
                                       "ViT-B/16", str(REPO), str(root)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("FileExistsError", repeated.stderr)
            return report

    def test_heredoc_aggregator_on_derived_mock_metrics(self):
        report = self.aggregate()
        print("ROUND B E6 MOCK heredoc: 3 derived 200x100 evaluations; relative/max-difference arithmetic EXACT; input SHA16/bytes + seed/argv/git audited; exclusive output")
        self.assertEqual(report["n_repetitions"], 2)

    def test_bad_router_or_nonfinite_stops_clearly(self):
        self.assertIn("router/seed provenance mismatch", self.aggregate("router"))
        self.assertIn("nonfinite metric", self.aggregate("nonfinite"))
        print("ROUND B E6 MOCK invalid router/nonfinite metrics STOP before any completed summary")


if __name__ == "__main__":
    unittest.main()

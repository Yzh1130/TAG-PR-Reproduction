"""Synthetic CPU tests for E3; no archive, checkpoint or GPU is required.

Run ``python -m unittest scripts.test_E3_fresh_forward -v``.  On Windows conda,
use ``MKL_THREADING_LAYER=SEQUENTIAL`` in the test process if NumPy MKL and
Torch have conflicting OpenMP libraries (never KMP_DUPLICATE_LIB_OK).
Only expensive
model construction and extraction are mocked in integration tests; score
construction, float32 files, retrieval metrics and the correction scan are real.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import gc
import hashlib
import io
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    # Load Torch before the CPU fixture first calls NumPy BLAS. This does not
    # modify either library or suppress duplicate-runtime safety checks.
    import torch
except ImportError:
    torch = None

import numpy as np

from . import E3_fresh_forward as e3
from . import S12_inductive_bank as s12
from ._offline_common import annotation_from_records, evaluate


def synthetic_fixture(seed=2026, queries=200, galleries=100, training=300, prompted=False):
    rng = np.random.default_rng(seed)

    def branches(count, prefix="text"):
        result = {}
        for suffix, width in (("feat", 4), ("tse_feat", 6)):
            values = rng.normal(size=(count, width))
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            result[f"{prefix}_{suffix}"] = values.astype(np.float32)
        return result

    test, train = branches(queries), branches(training)
    if prompted:
        test = {key + "_view0": value for key, value in test.items()} | {
            key + "_view1": value for key, value in branches(queries).items()}
        train = {key + "_view0": value for key, value in train.items()} | {
            key + "_view1": value for key, value in branches(training).items()}
    gallery = branches(galleries, "image")
    if queries != galleries * 2 or training % 2:
        raise ValueError("test fixture uses two captions per image")
    records = [{"file_path": f"G2APS/{'D' if index % 2 == 0 else 'S'}1_{index:04d}.jpg",
                "id": index // 2, "cam_id": index % 2,
                "captions": [f"test caption {index} a", f"test caption {index} b"], "split": "test"}
               for index in range(galleries)]
    train_records = [{"file_path": f"G2APS/D2_{index:04d}.jpg", "id": 1000 + index // 2,
                      "captions": [f"train caption {index} a", f"train caption {index} b"], "split": "train"}
                     for index in range(training // 2)]
    ann = annotation_from_records(records)
    return SimpleNamespace(test=test, train=train, gallery=gallery, records=records,
                           train_records=train_records, ann=ann, prompted=prompted)


def manual_scores(text, gallery, views):
    count = len(next(iter(text.values())))
    output = np.empty((count, len(views)), np.float32)
    prompted = "text_feat_view0" in text
    for view in ((0, 1) if prompted else (None,)):
        suffix = f"_view{view}" if prompted else ""
        selected = np.asarray(views) == view if prompted else np.ones(len(views), dtype=bool)
        output[:, selected] = (
            text[f"text_feat{suffix}"] @ gallery["image_feat"][selected].T
            + text[f"text_tse_feat{suffix}"] @ gallery["image_tse_feat"][selected].T
        ) / 2
    return output


def _close_memmaps(value):
    if isinstance(value, np.memmap):
        value._mmap.close()
    elif isinstance(value, dict):
        for item in value.values():
            _close_memmaps(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _close_memmaps(item)


@unittest.skipIf(torch is None, "PyTorch is needed for actual CPU score construction")
class FreshScoreTests(unittest.TestCase):
    def _check_matrices(self, prompted):
        fixture = synthetic_fixture(queries=20, galleries=10, training=30, prompted=prompted)
        with tempfile.TemporaryDirectory(prefix="tagpr-e3-scores-") as directory:
            outputs = e3.write_fresh_matrices(Path(directory), "small", fixture.test, fixture.train,
                                              fixture.gallery, fixture.ann.gallery_views, "cpu", block_size=7)
            try:
                self.assertTrue({"sim_qg", "sim_bank_g", "sim_gg", "paths"} <= outputs.keys())
                expected = {
                    "sim_qg": manual_scores(fixture.test, fixture.gallery, fixture.ann.gallery_views),
                    "sim_bank_g": manual_scores(fixture.train, fixture.gallery, fixture.ann.gallery_views),
                    "sim_gg": (fixture.gallery["image_feat"] @ fixture.gallery["image_feat"].T
                               + fixture.gallery["image_tse_feat"] @ fixture.gallery["image_tse_feat"].T) / 2,
                }
                errors = {}
                for name, reference in expected.items():
                    self.assertIsInstance(outputs[name], np.memmap)
                    self.assertEqual(outputs[name].dtype, np.float32)
                    self.assertFalse(outputs[name].flags.writeable)
                    self.assertEqual(outputs[name].shape, reference.shape)
                    error = float(np.max(np.abs(outputs[name] - reference)))
                    errors[name] = error
                    self.assertLess(error, 3e-7)
                    self.assertTrue(Path(outputs["paths"][name]).is_file())
                np.testing.assert_allclose(np.diag(outputs["sim_gg"]), 1, atol=3e-7, rtol=0)
                print("E3 SCORE MOCK " + json.dumps({"prompted": prompted, "global_width": 4,
                    "tse_width": 6, "qg_shape": list(outputs["sim_qg"].shape),
                    "bank_shape": list(outputs["sim_bank_g"].shape), "max_abs_errors": errors}, sort_keys=True))
            finally:
                _close_memmaps(outputs)

    def test_unprompted_unequal_widths_and_partial_final_block(self):
        self._check_matrices(False)

    def test_view_prompt_selects_gallery_view_columns(self):
        self._check_matrices(True)

    def test_matrix_output_never_overwrites_existing_files(self):
        fixture = synthetic_fixture(queries=20, galleries=10, training=30)
        with tempfile.TemporaryDirectory(prefix="tagpr-e3-no-overwrite-") as directory:
            out = Path(directory)
            first = e3.write_fresh_matrices(out, "same", fixture.test, fixture.train, fixture.gallery,
                                            fixture.ann.gallery_views, "cpu", block_size=7)
            hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for name, path in first["paths"].items()}
            _close_memmaps(first)
            with self.assertRaises(FileExistsError):
                e3.write_fresh_matrices(out, "same", fixture.test, fixture.train, fixture.gallery,
                                         fixture.ann.gallery_views, "cpu", block_size=7)
            self.assertEqual(hashes, {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                     for name, path in first["paths"].items()})

    def test_invalid_feature_width_normalization_and_prompt_keys(self):
        fixture = synthetic_fixture(queries=20, galleries=10, training=30)
        wrong_width = np.ones((20, 3), np.float32) / np.sqrt(np.float32(3))
        bad_inputs = [
            {**fixture.test, "text_feat": wrong_width},
            {**fixture.test, "text_feat": fixture.test["text_feat"] * .5},
            {"text_feat_view0": fixture.test["text_feat"],
             "text_tse_feat_view0": fixture.test["text_tse_feat"]},
            {**fixture.test, "text_feat": np.full((20, 4), np.nan, np.float32)},
        ]
        for index, bad in enumerate(bad_inputs):
            with self.subTest(case=index), tempfile.TemporaryDirectory(prefix="tagpr-e3-invalid-") as directory:
                with self.assertRaises(ValueError):
                    e3.write_fresh_matrices(Path(directory), "bad", bad, fixture.train,
                                             fixture.gallery, fixture.ann.gallery_views, "cpu", block_size=7)
                self.assertFalse(list(Path(directory).glob("*.npy")))


@unittest.skipIf(torch is None, "PyTorch is needed for extraction-state and integration guards")
class FreshPipelineTests(unittest.TestCase):
    def pipeline(self, seed=0, full_scan=False, change_test=False, change_archive=False,
                 corrupt_text_rng=False, corrupt_text_buffer=False, prompted=False):
        fixture = synthetic_fixture(prompted=prompted)
        if change_test:
            fixture.test = {key: -value for key, value in fixture.test.items()}
        with tempfile.TemporaryDirectory(prefix="tagpr-e3-pipeline-") as directory:
            root = Path(directory)
            archive_dir, out = root / "archive", root / "out"
            archive_dir.mkdir()
            out.mkdir()
            archive_scores = manual_scores(fixture.test, fixture.gallery, fixture.ann.gallery_views)
            if change_archive:
                archive_scores.fill(-.9)
                archive_scores[np.arange(len(archive_scores)), fixture.ann.query_source_indices] = .9
            archive = archive_dir / "sim_qg.npy"
            np.save(archive, archive_scores)
            before_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            ann_path, train_path = root / "test_reid.json", root / "train_reid.json"
            ann_path.write_text(json.dumps(fixture.records), encoding="utf-8")
            train_path.write_text(json.dumps(fixture.train_records), encoding="utf-8")
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"mock trusted checkpoint")
            config_path = root / "config.yaml"
            config_path.write_text("{}", encoding="utf-8")
            command = ["--fresh-forward", "--sim", str(archive), "--anno", str(ann_path),
                       "--resume", str(checkpoint), "--config", str(config_path), "--out", str(out),
                       "--device", "cpu", "--seed", str(seed), "--score-block-size", "19", "--tag", "mock"]
            config = SimpleNamespace(experiment=SimpleNamespace(max_words=30, text_length=77,
                                                                 view_prompt=prompted), image_dir="unused")
            events, captured, output = [], {}, io.StringIO()

            class MockModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.ones(2))
                    self.register_buffer("statistics", torch.zeros(2))

            def load(*positional, **keywords):
                requested_seed = positional[5] if len(positional) > 5 else keywords["seed"]
                self.assertEqual(requested_seed, seed)
                random.seed(requested_seed)
                np.random.seed(requested_seed)
                torch.manual_seed(requested_seed)
                events.append("load")
                return MockModel().eval(), config, {"seed": requested_seed, "checkpoint": str(checkpoint)}

            def images(model, *unused, **kwargs):
                events.append("image")
                # Image-side Gumbel routing is allowed to advance RNG exactly
                # once. The text-state guard starts AFTER this extraction.
                torch.rand(3)
                return fixture.gallery

            def texts(model, strings, **kwargs):
                branch = "test" if strings[0].startswith("test") else "train"
                events.append(branch)
                if corrupt_text_rng:
                    torch.rand(1)
                if corrupt_text_buffer:
                    model.statistics.add_(1)
                return getattr(fixture, branch)

            def corrections(sim, bank, gg, ann):
                captured["sim_qg"] = np.asarray(sim).copy()
                captured["sim_bank_g"] = np.asarray(bank).copy()
                captured["sim_gg"] = np.asarray(gg).copy()
                if full_scan:
                    return s12.run_corrections(sim, bank, gg, ann,
                                               subset_sizes=(100, 200, 250), progress=None)
                return {"baseline": {key: value for key, value in evaluate(sim, ann).items()
                                      if key in ("R1", "R5", "R10", "mAP", "n_query")},
                        "rows": [], "banks": []}

            real_load_similarity = e3.load_similarity

            def read_archive(*values, **kwargs):
                self.assertEqual(events, ["load", "image", "test", "train"])
                self.assertEqual(len(list(out.glob("*.npy"))), 3,
                                 "Archive must not be read until all three fresh matrices exist")
                captured["archive_loaded_after_all_fresh_matrices"] = True
                return real_load_similarity(*values, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(output))
                stack.enter_context(redirect_stderr(output))
                load_mock = stack.enter_context(patch.object(e3, "load_model", side_effect=load))
                image_mock = stack.enter_context(patch.object(e3, "extract_gallery_features", side_effect=images))
                text_mock = stack.enter_context(patch.object(e3, "extract_text_features", side_effect=texts))
                stack.enter_context(patch.object(e3, "preprocess_texts", side_effect=lambda records, max_words:
                                                 [text for row in records for text in row["captions"]]))
                archive_mock = stack.enter_context(patch.object(e3, "load_similarity", side_effect=read_archive))
                correction_mock = stack.enter_context(patch.object(e3, "run_corrections", side_effect=corrections))
                code = e3.main(command)
                reports = list(out.glob("E3_*.json"))
                self.assertEqual(len(reports), 1, output.getvalue())
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                self.assertTrue(list(out.glob("E3_*.log")))
                self.assertEqual(before_hash, hashlib.sha256(archive.read_bytes()).hexdigest())
                self.assertEqual(load_mock.call_count, 1)
                self.assertEqual(image_mock.call_count, 1)
                if corrupt_text_rng or corrupt_text_buffer:
                    self.assertEqual(code, 1, output.getvalue())
                    correction_mock.assert_not_called()
                    archive_mock.assert_not_called()
                    self.assertFalse(list(out.glob("*sim_bank_g*.npy")))
                else:
                    self.assertEqual(code, 0, output.getvalue())
                    self.assertEqual(text_mock.call_count, 2)
                    correction_mock.assert_called_once()
                    archive_mock.assert_called_once()
                    self.assertEqual(len(correction_mock.call_args.args), 4)
                    self.assertFalse(correction_mock.call_args.kwargs)
                    _close_memmaps(correction_mock.call_args.args)
                # Release mock-held arrays/tracebacks before Windows temp-file cleanup.
                correction_mock.reset_mock()
                text_mock.reset_mock()
                image_mock.reset_mock()
                load_mock.reset_mock()
            gc.collect()
            return report, captured, events, output.getvalue()

    def test_order_seed_no_parity_gate_and_full_352_scan(self):
        report, matrices, events, output = self.pipeline(full_scan=True, change_archive=True)
        self.assertEqual(events, ["load", "image", "test", "train"])
        self.assertTrue(report["pure_query_independent_inductive_pipeline"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["corrections"]["rows"]), 352)
        self.assertEqual(len(report["corrections"]["banks"]), 11)
        self.assertNotIn("gallery_reconstruction", report)
        self.assertNotIn("qg_parity", report)
        self.assertEqual(matrices["sim_qg"].shape, (200, 100))
        self.assertEqual(matrices["sim_bank_g"].shape, (300, 100))
        self.assertTrue(matrices["archive_loaded_after_all_fresh_matrices"])
        self.assertIn("extraction_elapsed_sec", report)
        for key in ("gallery_images", "test_text", "train_text"):
            self.assertGreaterEqual(report["extraction_elapsed_sec"][key], 0)
        train_banks = [bank for bank in report["corrections"]["banks"] if bank["bank"].startswith("train")]
        self.assertTrue(all("inductive" in bank["regime"] for bank in train_banks))
        self.assertIn("transductive", report["corrections"]["banks"][0]["regime"])
        print("E3 PIPELINE MOCK " + json.dumps({"shape": list(matrices["sim_qg"].shape),
              "bank_shape": list(matrices["sim_bank_g"].shape), "events": events,
              "rows": len(report["corrections"]["rows"]), "banks": 11,
              "baseline": report["corrections"]["baseline"], "archive_R1": 100,
              "archive_is_not_a_parity_gate": True,
              "pure_query_independent_inductive_pipeline": True}, sort_keys=True))

    def test_training_bank_is_independent_of_test_text_and_archive(self):
        _, reference, _, _ = self.pipeline()
        _, changed, _, _ = self.pipeline(change_test=True, change_archive=True)
        np.testing.assert_array_equal(reference["sim_bank_g"], changed["sim_bank_g"])
        np.testing.assert_array_equal(reference["sim_gg"], changed["sim_gg"])
        self.assertFalse(np.array_equal(reference["sim_qg"], changed["sim_qg"]))
        print("E3 INDEPENDENCE MOCK train_bank_max_abs_error=0 gallery_gram_max_abs_error=0 test_qg_changed=true")

    def test_view_prompt_pipeline_and_explicit_seed(self):
        report, matrices, events, _ = self.pipeline(seed=17, prompted=True)
        self.assertEqual(events, ["load", "image", "test", "train"])
        fixture = synthetic_fixture(prompted=True)
        np.testing.assert_allclose(matrices["sim_bank_g"],
                                   manual_scores(fixture.train, fixture.gallery, fixture.ann.gallery_views),
                                   rtol=0, atol=3e-7)
        self.assertTrue(report["pure_query_independent_inductive_pipeline"])

    def test_text_rng_consumption_stops_before_bank_or_scan(self):
        report, _, events, output = self.pipeline(corrupt_text_rng=True)
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["pure_query_independent_inductive_pipeline"])
        self.assertEqual(report["independence_status"], "not_verified")
        self.assertEqual(events, ["load", "image", "test"])
        self.assertRegex(report["error"].lower(), "rng|random")
        self.assertIn("STOP", output.upper())
        print("E3 TEXT RNG MOCK status=STOP bank_created=false scan_called=false")

    def test_text_model_buffer_mutation_stops(self):
        report, _, events, _ = self.pipeline(corrupt_text_buffer=True)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(events, ["load", "image", "test"])
        self.assertRegex(report["error"].lower(), "buffer|state|model")


class FreshCommandLineTests(unittest.TestCase):
    def test_help_defaults_and_explicit_seed(self):
        parser = e3.build_parser()
        required = ["--anno", "test_reid.json", "--resume", "checkpoint.pth", "--config", "config.yaml"]
        defaults = parser.parse_args(required)
        self.assertFalse(defaults.fresh_forward)
        self.assertEqual(defaults.seed, 0)
        self.assertEqual(defaults.num_workers, 2)
        self.assertEqual(defaults.score_block_size, 256)
        explicit = parser.parse_args(required + ["--fresh-forward", "--seed", "73"])
        self.assertTrue(explicit.fresh_forward)
        self.assertEqual(explicit.seed, 73)
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            e3.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        for flag in ("--fresh-forward", "--seed", "--image-batch-size", "--score-block-size"):
            self.assertIn(flag, output.getvalue())

    def test_explicit_mode_and_invalid_sizes_error_before_loading(self):
        required = ["--anno", "test_reid.json", "--resume", "checkpoint.pth", "--config", "config.yaml"]
        for arguments in ([], ["--fresh-forward", "--score-block-size", "0"],
                          ["--fresh-forward", "--image-batch-size", "-1"],
                          ["--fresh-forward", "--text-batch-size", "0"],
                          ["--fresh-forward", "--num-workers", "-1"]):
            with self.subTest(arguments=arguments), patch.object(e3, "load_model") as loader:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    e3.main(required + arguments)
                self.assertEqual(caught.exception.code, 2)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

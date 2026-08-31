"""CPU regression tests for the explicit, text-only S12 Gram recovery path.

Run with ``python -m unittest scripts.test_S12_reconstruction -v``.  All fixtures
are synthetic (200 queries x 100 gallery images), not archived TAG-PEDES data.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import gc
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import S12_inductive_bank as s12
from . import _s12_reconstruct as reconstruction
from ._offline_common import annotation_from_records, evaluate, json_ready


def synthetic_features(seed=2026, antithetic=False):
    """Unequal branch widths exercise the real global/TSE shape contract."""
    rng = np.random.default_rng(seed)

    def branches(count):
        parts = []
        for width in (4, 6):
            values = rng.normal(size=(count // 2 if antithetic else count, width))
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            values = values.astype(np.float32)
            if antithetic:
                values = np.concatenate((values, -values))
            parts.append(values)
        return {"text_feat": parts[0], "text_tse_feat": parts[1]}

    test, train, gallery = branches(200), branches(300), branches(100)
    raw_test = reconstruction.concatenate_text_features(test)
    raw_train = reconstruction.concatenate_text_features(train)
    raw_gallery = reconstruction.concatenate_text_features(gallery)
    qg = (raw_test @ raw_gallery.T).astype(np.float32)
    gg = (raw_gallery @ raw_gallery.T).astype(np.float32)
    bank = (raw_train @ raw_gallery.T).astype(np.float32)
    records = [{"file_path": f"G2APS/{'D' if index % 2 == 0 else 'S'}1_{index:04d}.jpg",
                "id": index // 2, "cam_id": index % 2,
                "captions": [f"caption {index} a", f"caption {index} b"], "split": "test"}
               for index in range(100)]
    ann = annotation_from_records(records)
    return SimpleNamespace(test=test, train=train, gallery=gallery, raw_test=raw_test,
                           raw_train=raw_train, raw_gallery=raw_gallery, qg=qg, gg=gg,
                           bank=bank, records=records, ann=ann)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReconstructionNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic_features()

    def test_float64_least_squares_and_bank_against_literal_formulas(self):
        f = self.fixture
        g, g_report = reconstruction.reconstruct_gallery(f.gg, rank=10, block_size=17)
        t, t_report = reconstruction.coordinate_texts(f.qg, g, block_size=23, rcond=1e-10)
        w, w_report = reconstruction.fit_text_mapping(f.raw_test, t, block_size=31, rcond=1e-10)
        self.assertEqual(g.shape, (100, 10))
        self.assertEqual(t.shape, (200, 10))
        self.assertEqual(w.shape, (10, 10))
        self.assertTrue(all(value.dtype == np.float64 for value in (g, t, w)))
        reference_t = f.qg.astype(np.float64) @ g @ np.linalg.pinv(g.T @ g, rcond=1e-10)
        reference_w = np.linalg.pinv(f.raw_test, rcond=1e-10) @ reference_t
        np.testing.assert_allclose(t, reference_t, atol=1e-14, rtol=0)
        np.testing.assert_allclose(w, reference_w, atol=1e-14, rtol=0)
        blocks = list(reconstruction.iter_mapped_scores(f.raw_train, w, g, block_size=19))
        self.assertEqual([(a, b) for a, b, _ in blocks],
                         [(a, min(a + 19, 300)) for a in range(0, 300, 19)])
        self.assertTrue(all(block.dtype == np.float64 for _, _, block in blocks))
        bank = np.concatenate([block for _, _, block in blocks])
        bank_error = float(np.abs(bank - f.bank).max())
        self.assertLess(bank_error, 1e-6)
        self.assertLess(g_report["residual"]["max_abs_error"], 1e-6)
        self.assertLess(t_report["residual"]["max_abs_error"], 1e-6)
        self.assertLess(w_report["residual"]["max_abs_error"], 1e-6)
        self.assertEqual(t_report["rcond"], 1e-10)
        self.assertEqual(w_report["rcond"], 1e-10)
        self.assertEqual(g_report["spectrum"]["rank_selected"], 10)
        print("S12 RECON NUMERICS " + json.dumps({
            "qg_shape": list(f.qg.shape), "global_width": 4, "tse_width": 6,
            "computation_dtype": str(g.dtype),
            "gram_max_abs_error": g_report["residual"]["max_abs_error"],
            "coordinate_max_abs_error": t_report["residual"]["max_abs_error"],
            "mapping_max_abs_error": w_report["residual"]["max_abs_error"],
            "train_bank_max_abs_error": bank_error,
            "numpy_pinv_text_max_abs_error": float(np.abs(t - reference_t).max()),
            "numpy_pinv_mapping_max_abs_error": float(np.abs(w - reference_w).max()),
        }, sort_keys=True))

    def test_auto_rank_negative_clipping_symmetrization_and_energy(self):
        spectrum = np.array([5.0, 1e-5, 6e-8, 4e-8, 0.0, -2e-8], dtype=np.float64)
        gram = np.diag(spectrum)
        gram[0, 1] = 2e-9
        gram[1, 0] = -2e-9
        g, report = reconstruction.reconstruct_gallery(gram, rank="auto", block_size=2)
        diagnostics = report["spectrum"]
        self.assertEqual(g.shape, (6, 3))
        self.assertEqual(diagnostics["negative_eigenvalues_clipped"], 1)
        self.assertEqual(diagnostics["negative_eigenvalue_max_abs"], 2e-8)
        self.assertEqual(diagnostics["symmetry_max_abs_error"], 4e-9)
        self.assertEqual(diagnostics["auto_eigenvalue_threshold"], 5e-8)
        self.assertAlmostEqual(diagnostics["retained_energy_ratio"],
                               spectrum[:3].sum() / np.maximum(spectrum, 0).sum(), places=15)
        expected = np.diag([5.0, 1e-5, 6e-8, 0, 0, 0])
        np.testing.assert_allclose(g @ g.T, expected, atol=1e-15, rtol=0)
        self.assertEqual(report["residual"]["max_abs_error"], 4e-8)
        full, explicit_report = reconstruction.reconstruct_gallery(gram, rank=6)
        self.assertEqual(full.shape, (6, 6))
        self.assertEqual(explicit_report["spectrum"]["rank_selected"], 6)
        self.assertEqual(explicit_report["spectrum"]["rank_positive_selected"], 4)
        self.assertTrue(explicit_report["spectrum"]["warnings"])
        print("S12 AUTO RANK " + json.dumps(diagnostics, sort_keys=True))

    def test_rank_deficient_mapping_uses_explicit_pinv_without_ridge(self):
        rng = np.random.default_rng(2027)
        raw = rng.normal(size=(200, 4))
        raw = np.concatenate((raw, raw[:, :2]), axis=1)
        target = rng.normal(size=(200, 3))
        w, report = reconstruction.fit_text_mapping(raw, target, block_size=9, rcond=1e-10)
        expected = np.linalg.pinv(raw, rcond=1e-10) @ target
        np.testing.assert_allclose(w, expected, atol=1e-14, rtol=0)
        self.assertEqual(report["raw_feature_rank"], 4)
        self.assertTrue(report["warnings"])

    def test_nonfinite_shapes_rank_and_block_errors_are_explicit(self):
        for matrix, rank in ((np.ones((4, 3)), 2), (np.zeros((3, 3)), "auto"),
                             (np.diag([1.0, np.nan]), 1), (np.eye(3), 0),
                             (np.eye(3), 4), (np.eye(3), -1), (np.eye(3), "bad"),
                             (np.eye(3), True)):
            with self.subTest(shape=matrix.shape, rank=rank), self.assertRaises(ValueError):
                reconstruction.reconstruct_gallery(matrix, rank=rank)
        for block in (0, -1, True, 1.5):
            with self.subTest(block=block), self.assertRaises(ValueError):
                reconstruction.reconstruct_gallery(np.eye(3), rank=2, block_size=block)
        with self.assertRaisesRegex(ValueError, "columns"):
            reconstruction.coordinate_texts(np.ones((5, 4)), np.eye(3))
        with self.assertRaisesRegex(ValueError, "row"):
            reconstruction.fit_text_mapping(np.ones((4, 3)), np.ones((5, 3)))
        for rcond in (0, -1, np.nan, 1):
            with self.subTest(rcond=rcond), self.assertRaises(ValueError):
                reconstruction.coordinate_texts(np.eye(3), np.eye(3), rcond=rcond)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            list(reconstruction.iter_mapped_scores(np.ones((5, 3)), np.eye(4), np.eye(4)))

    def test_concatenation_does_not_normalize_or_force_equal_branch_widths(self):
        f = self.fixture
        raw = reconstruction.concatenate_text_features(f.test)
        expected = np.concatenate((f.test["text_feat"].astype(np.float64),
                                   f.test["text_tse_feat"].astype(np.float64)), axis=1) / np.sqrt(2)
        np.testing.assert_array_equal(raw, expected)
        self.assertEqual(raw.shape, (200, 10))
        with self.assertRaisesRegex(ValueError, "view-prompt"):
            reconstruction.concatenate_text_features({"text_feat_view0": f.test["text_feat"]})
        with self.assertRaisesRegex(ValueError, "not L2"):
            reconstruction.concatenate_text_features({**f.test, "text_feat": 0.5 * f.test["text_feat"]})
        with self.assertRaisesRegex(ValueError, "row counts"):
            reconstruction.concatenate_text_features({**f.test, "text_tse_feat": f.test["text_tse_feat"][:-1]})


class ReconstructionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic_features()

    def test_acceptance_uses_exact_count_delta_and_reports_warn_and_stop(self):
        n = 16504

        def result(count):
            correct = np.arange(n) < count
            return {"n_query": n, "R1": float(np.float32(100 * count / n)),
                    "details": {"correct": correct, "top1": np.zeros(n, dtype=np.int64),
                                "query_indices": np.arange(n)}}

        base = result(10000)
        small = {name: {"max_abs_error": 1e-7} for name in ("b", "c", "e")}
        passing = s12._reconstruction_acceptance(small, base, result(10003), tolerance=1e-5)
        self.assertEqual(passing["status"], "PASS")
        self.assertAlmostEqual(passing["r1_delta_pp"], 3 * 100 / n, places=14)
        warning = s12._reconstruction_acceptance({**small, "e": {"max_abs_error": .001}},
                                                base, result(10003), tolerance=1e-5)
        self.assertEqual(warning["status"], "WARN")
        stopped = s12._reconstruction_acceptance(small, base, result(10004), tolerance=1e-5)
        self.assertEqual(stopped["status"], "STOP")
        print("S12 R1 GATE " + json.dumps({"three_query_delta_pp": passing["r1_delta_pp"],
                                           "four_query_delta_pp": stopped["r1_delta_pp"],
                                           "states": [passing["status"], warning["status"], stopped["status"]]}))

    def _run_bank(self, fixture, sim=None):
        temporary = tempfile.TemporaryDirectory(prefix="s12_gram_test_")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        archive, out = root / "_archive_evidence", root / "new_output"
        archive.mkdir()
        out.mkdir()
        sim = fixture.qg if sim is None else sim
        qg_path, gg_path = archive / "sim_qg.npy", archive / "sim_gg.npy"
        np.save(qg_path, sim)
        np.save(gg_path, fixture.gg)
        before = {path.name: file_hash(path) for path in archive.iterdir()}
        qg = np.load(qg_path, mmap_mode="r", allow_pickle=False)
        gg = np.load(gg_path, mmap_mode="r", allow_pickle=False)
        args = s12.build_parser().parse_args(["--reconstruct-gallery-from-gg", "--gg-rank", "10",
                                             "--score-block-size", "17", "--out", str(out)])
        report = {}
        output = io.StringIO()
        with redirect_stdout(output):
            bank = s12._reconstructed_bank(args, out, "mock", report, qg, gg, fixture.ann,
                                           fixture.train, fixture.test)
        self.assertEqual(before, {path.name: file_hash(path) for path in archive.iterdir()})
        self.assertFalse(qg.flags.writeable)
        self.assertFalse(gg.flags.writeable)
        self.assertIsInstance(bank, np.memmap)
        self.assertEqual(bank.dtype, np.float32)
        self.assertEqual(bank.shape, (300, 100))
        json.dumps(json_ready(report), allow_nan=False)
        return bank, report, output.getvalue(), out

    def test_integration_float32_outputs_reports_and_readonly_archives(self):
        bank, report, output, out = self._run_bank(self.fixture)
        np.testing.assert_allclose(bank, self.fixture.bank, atol=1e-6, rtol=0)
        recon = report["gallery_reconstruction"]
        self.assertEqual(recon["gate"]["status"], "PASS")
        for key in ("a_gram_spectrum", "b_gram_residual", "c_coordinate_residual",
                    "d_procrustes_residual", "e_end_to_end", "gate"):
            self.assertIn(key, recon)
        for path in out.glob("*.npy"):
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            self.assertEqual(values.dtype, np.float32, str(path))
        self.assertEqual(recon["e_end_to_end"]["metric_deltas_pp"]["R1"], 0)
        self.assertIn("PASS", output)
        print("S12 END-TO-END " + json.dumps(recon["e_end_to_end"], sort_keys=True))

    def test_constant_offset_is_warn_not_silent_or_stop(self):
        f = synthetic_features(antithetic=True)
        shifted = (f.qg + np.float32(.001)).astype(np.float32)
        bank, report, output, _ = self._run_bank(f, sim=shifted)
        recon = report["gallery_reconstruction"]
        self.assertEqual(recon["gate"]["status"], "WARN")
        self.assertEqual(recon["gate"]["r1_delta_pp"], 0)
        self.assertGreater(recon["e_end_to_end"]["max_abs_error"], .0009)
        self.assertIn("WARN", output)
        np.testing.assert_allclose(bank, f.bank, atol=1e-6, rtol=0)
        print("S12 WARN FIXTURE " + json.dumps(recon["gate"], sort_keys=True))

    def _pipeline(self, reconstruct=False, metadata=True, corrupt=False, through_main=False):
        """Real orchestration, with only expensive repository extraction mocked."""
        f = self.fixture
        temporary = tempfile.TemporaryDirectory(prefix="s12_path_test_")
        self.addCleanup(temporary.cleanup)
        # Mock call histories/tracebacks may retain memmaps in cycles. Release
        # them before TemporaryDirectory deletion, especially on Windows.
        self.addCleanup(gc.collect)
        root = Path(temporary.name)
        archive, out = root / "dump", root / "out"
        archive.mkdir()
        out.mkdir()
        scores = f.qg.copy()
        if corrupt:
            scores[np.arange(200), f.ann.query_source_indices] = 1.0
        np.save(archive / "sim_qg.npy", scores)
        np.save(archive / "sim_gg.npy", f.gg)
        # A genuine retrieval snapshot is deliberately not a feature archive.
        (archive / "output.json").write_text(json.dumps([{"query": i} for i in range(200)]), encoding="utf-8")
        if metadata:
            (archive / "sim_meta.json").write_text("{}", encoding="utf-8")
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"mock checkpoint: load_model is mocked")
        ann_path = root / "test_reid.json"
        ann_path.write_text(json.dumps(f.records), encoding="utf-8")
        command = ["--sim", str(archive / "sim_qg.npy"), "--anno", str(ann_path),
                   "--out", str(out), "--resume", str(checkpoint), "--device", "cpu",
                   "--gg-rank", "10", "--score-block-size", "19"]
        command += ["--reconstruct-gallery-from-gg"] if reconstruct else []
        args = s12.build_parser().parse_args(command)
        config = SimpleNamespace(experiment=SimpleNamespace(max_words=30, text_length=77, view_prompt=False),
                                 image_dir="unused")
        provenance = {"seed": 2026}
        gallery = {"image_feat": f.gallery["text_feat"], "image_tse_feat": f.gallery["text_tse_feat"]}
        train_records = f.records + f.records[:50]
        info = {"identities": 50, "captions": 300}
        fake_torch = SimpleNamespace(device=lambda value: SimpleNamespace(type="cpu"))
        before = {path.name: file_hash(path) for path in archive.iterdir()}
        report, output = {}, io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch.dict("sys.modules", {"torch": fake_torch}))
            stack.enter_context(patch.object(s12, "load_annotations", return_value=f.ann))
            stack.enter_context(patch.object(s12, "discover_train_annotations", return_value=(ann_path, train_records, info)))
            stack.enter_context(patch.object(s12, "load_model", return_value=(object(), config, provenance)))
            stack.enter_context(patch.object(s12, "preprocess_texts", side_effect=lambda records, words: [text for row in records for text in row["captions"]]))
            extraction = stack.enter_context(patch.object(s12, "extract_text_features", side_effect=
                [f.train, {key: value.copy() for key, value in f.train.items()}, f.test,
                 {key: value.copy() for key, value in f.test.items()}]))
            image_extraction = stack.enter_context(patch.object(s12, "extract_gallery_features", return_value=gallery))
            strict_guard = stack.enter_context(patch.object(s12, "verify_query_gallery", return_value={"max_abs_error": 0}))
            writer = stack.enter_context(patch.object(s12, "write_bank_similarity", side_effect=lambda path, *unused:
                self._save_bank(path, f.bank)))
            corrections = stack.enter_context(patch.object(s12, "run_corrections", return_value={"mock": True}))
            if through_main:
                exit_code = s12.main(command)
                files = list(out.glob("S12_*.json"))
                self.assertEqual(len(files), 1)
                report = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(exit_code, 1 if corrupt else 0)
                self.assertEqual(len(list(out.glob("S12_*.log"))), 1)
            else:
                s12._run_real(args, out, "mock", report)
            if corrupt:
                corrections.assert_not_called()
                self.assertFalse(list(out.glob("*_sim_bank_g.npy")))
            else:
                corrections.assert_called_once()
                # Do not leave fixture files held open by Mock.call_args.
                for value in corrections.call_args.args:
                    if isinstance(value, np.memmap):
                        value._mmap.close()
        self.assertEqual(before, {path.name: file_hash(path) for path in archive.iterdir()})
        return report, output.getvalue(), extraction, image_extraction, strict_guard, writer

    @staticmethod
    def _save_bank(path, values):
        np.save(path, values)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def test_default_retains_direct_image_extraction_and_strict_parity(self):
        report, _, extraction, images, guard, writer = self._pipeline()
        self.assertEqual(extraction.call_count, 3)
        images.assert_called_once()
        guard.assert_called_once()
        self.assertEqual(guard.call_args.kwargs, {"tolerance": 1e-5})
        writer.assert_called_once()
        self.assertEqual(len(guard.call_args.args), 6)
        self.assertNotIn("gallery_reconstruction", report)

    def test_explicit_reconstruction_never_calls_image_encoder_and_checks_both_text_sets(self):
        report, output, extraction, images, guard, writer = self._pipeline(reconstruct=True, metadata=False)
        self.assertEqual(extraction.call_count, 4)
        images.assert_not_called()
        guard.assert_not_called()
        writer.assert_not_called()
        self.assertEqual(report["gallery_reconstruction"]["gate"]["status"], "PASS")
        self.assertIn("WARN", output)
        self.assertIn("sim_meta.json", output)

    def test_direct_mode_still_requires_archive_metadata(self):
        with self.assertRaisesRegex(FileNotFoundError, "sim_meta.json"):
            self._pipeline(reconstruct=False, metadata=False)

    def test_stop_writes_complete_json_diagnostics_without_bank_or_corrections(self):
        report, output, extraction, images, guard, writer = self._pipeline(
            reconstruct=True, metadata=False, corrupt=True, through_main=True)
        self.assertEqual(report["status"], "stopped")
        recon = report["gallery_reconstruction"]
        self.assertEqual(recon["gate"]["status"], "STOP")
        self.assertGreater(abs(recon["gate"]["r1_delta_pp"]), .02)
        for key in ("a_gram_spectrum", "b_gram_residual", "c_coordinate_residual",
                    "d_procrustes_residual", "e_end_to_end"):
            self.assertIn(key, recon)
        self.assertIn("STOP", output)
        self.assertEqual(extraction.call_count, 4)
        images.assert_not_called()
        guard.assert_not_called()
        writer.assert_not_called()
        print("S12 STOP PIPELINE " + json.dumps({"status": report["status"], "gate": recon["gate"],
                                               "all_a_to_e_saved": True, "bank_not_created": True}))

    def test_success_main_reports_provenance_warning_in_json(self):
        report, output, _, images, _, _ = self._pipeline(reconstruct=True, metadata=False, through_main=True)
        self.assertEqual(report["status"], "complete_with_warnings")
        self.assertFalse(report["archive_metadata"]["metadata_available"])
        self.assertEqual(report["gallery_reconstruction"]["gate"]["status"], "PASS")
        self.assertIn("output.json is retrieval results, NOT a feature cache", output)
        images.assert_not_called()

    def test_parser_defaults_and_rank_validation(self):
        parser = s12.build_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.reconstruct_gallery_from_gg)
        self.assertEqual(defaults.gg_rank, 1536)
        self.assertEqual(defaults.parity_tol, 1e-5)
        self.assertEqual(parser.parse_args(["--gg-rank", "auto"]).gg_rank, "auto")
        self.assertEqual(parser.parse_args(["--gg-rank", "10"]).gg_rank, 10)
        for value in ("0", "-1", "garbage", "1.5"):
            with self.subTest(value=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--gg-rank", value])
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(tolerance=value), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                s12.main(["--parity-tol", value])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            s12.main(["--reconstruct-gallery-from-gg", "--gallery-features", "output.json"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Regression tests: python -m unittest tools.test_analysis_tools -v."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import _analysis_common as common
from tools import S8_controlled_eval as s8
from tools import S9_error_stats as s9
from tools import S10_csls_crossfit as s10


ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.ann = common.make_mock_annotation()
        self.scores = common.make_mock_similarity(self.ann)

    def test_s8_independent_mask_metric_and_csls_oracles(self):
        with contextlib.redirect_stdout(io.StringIO()):
            s8.self_test()

    def test_s9_conditional_prior_bootstrap_and_prediction_validation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            s9.self_test()

    def test_s10_fixed_evaluation_no_leakage_and_endpoint(self):
        with contextlib.redirect_stdout(io.StringIO()):
            s10.self_test()

    def test_real_annotation_explicit_and_path_only_anchors(self):
        path = ROOT / "datasets" / "TAG-PEDES" / "test_reid.json"
        if not path.exists():
            self.skipTest("local TAG-PEDES annotations unavailable")
        records = json.loads(path.read_text(encoding="utf-8"))
        explicit = common.annotation_from_records(records)
        stripped = [{key: value for key, value in record.items() if key != "cam_id"}
                    for record in records]
        inferred = common.annotation_from_records(stripped)
        np.testing.assert_array_equal(explicit.gallery_views, inferred.gallery_views)
        np.testing.assert_array_equal(explicit.query_source_indices, inferred.query_source_indices)
        for key, count in common.ANCHORS.items():
            self.assertEqual(explicit.summary[key], count)
        expected = {(0, 0): (4472, 70), (0, 1): (4542, 0),
                    (1, 0): (4194, 0), (1, 1): (3938, 256)}
        for (qview, gview), (count, skipped) in expected.items():
            q = np.flatnonzero(np.isin(explicit.query_ids, explicit.dual_ids)
                               & (explicit.query_views == qview))
            g = np.flatnonzero(explicit.gallery_views == gview)
            selected, dropped = common.eligible_queries(explicit, q, g, True)
            self.assertEqual((len(selected), dropped), (count, skipped))

    def test_bad_annotations_fail_closed(self):
        records = common.make_mock_records()
        with self.assertRaisesRegex(ValueError, "anchor assertion failed"):
            common.annotation_from_records(records)
        records[0]["cam_id"] = 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            common.annotation_from_records(records, False)
        records = common.make_mock_records()
        records[0]["file_path"] = "AG-ReID.v2/unknown.jpg"
        with self.assertRaisesRegex(ValueError, "cannot infer"):
            common.annotation_from_records(records, False)
        records = common.make_mock_records()
        records[0]["file_path"] = "mystery/x.jpg"
        with self.assertRaisesRegex(ValueError, "unknown source"):
            common.annotation_from_records(records, False)

    def test_matrix_shape_dtype_and_nan_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.npy"
            np.save(path, self.scores[:, :-1])
            with self.assertRaisesRegex(ValueError, "shape"):
                common.load_similarity(path, self.ann)
            np.save(path, self.scores.astype(np.int32))
            with self.assertRaisesRegex(ValueError, "dtype"):
                common.load_similarity(path, self.ann)
            values = self.scores.copy()
            values[0, 1] = np.nan
            np.save(path, values)
            with self.assertRaisesRegex(ValueError, "NaN"):
                common.load_similarity(path, self.ann)

    def test_selfcheck_fp32_gate_is_not_silently_disabled(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "fp32 self-check failed"):
                s8.self_check_gate(self.scores, self.ann)
            with patch.object(s8, "EXPECTED_R1", 100.0):
                gate = s8.self_check_gate(self.scores, self.ann)
        self.assertTrue(gate["aligned_to_reference"])
        self.assertEqual(float(np.float32(63.63306)),
                         common._online_percent(10502, 16504))

    def test_streaming_topk_is_chunk_and_query_order_invariant(self):
        rng = np.random.default_rng(99)
        values = rng.normal(size=(20, 10)).astype(np.float32)
        expected = np.sort(values, axis=0)[::-1][:6].mean(axis=0, dtype=np.float32)
        for block_size in (1, 3, 7, 40):
            actual = common.column_topk_mean(values, 6, rng.permutation(20),
                                             block_size=block_size)
            np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            common.column_topk_mean(values, 21)

    def test_ties_exclusion_and_empty_queries(self):
        values = np.zeros((20, 10), dtype=np.float32)
        result = common.evaluate(values, self.ann)
        self.assertEqual(result["top1_tied_queries"], 20)
        self.assertEqual(result["R1"], 30.0)
        empty = common.evaluate(values, self.ann, query_indices=np.array([], dtype=np.int64))
        self.assertIsNone(empty["mAP"])
        self.assertEqual(empty["n_query"], 0)
        subset = np.array([0])
        result = common.evaluate(values, self.ann, query_indices=np.array([0, 1]),
                                 gallery_indices=subset, exclude_source=True)
        self.assertEqual(result["n_query_excluded_no_positive"], 2)

    def test_json_outputs_roundtrip_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "result.json"
            result = common.evaluate(self.scores, self.ann)
            common.write_json(path, result)
            self.assertEqual(json.loads(path.read_text()), result)
            with self.assertRaises(ValueError):
                common.write_json(path, {"bad": float("nan")})

    def test_all_entrypoints_help_and_missing_inputs(self):
        for script in ("S8_controlled_eval.py", "S9_error_stats.py", "S10_csls_crossfit.py",
                       "probe_ortho.py"):
            with self.subTest(script=script):
                command = [sys.executable, str(ROOT / "tools" / script)]
                help_result = subprocess.run(command + ["--help"], capture_output=True,
                                             text=True, timeout=60)
                self.assertEqual(help_result.returncode, 0, help_result.stderr)
                self.assertIn("--self_test", help_result.stdout)
                missing = subprocess.run(command, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(missing.returncode, 0)
                self.assertIn("error:", missing.stderr)
                self.assertNotIn("Traceback", missing.stderr)

    def test_probe_two_backwards_and_checkpoint_hooks(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools" / "probe_ortho.py"),
                                 "--self_test"], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("independent unscaled backward errors=0", result.stdout)
        self.assertIn("loss_scale 1/65536", result.stdout)
        self.assertIn("parameters unchanged", result.stdout)
        self.assertIn("hooks count=4", result.stdout)


if __name__ == "__main__":
    unittest.main()

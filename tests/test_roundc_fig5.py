"""Figure-5 checks on three 200-query x 100-gallery synthetic seed runs.

Metrics are calculated from mock scores, never hard-coded acceptance targets.
"""

import contextlib
import csv
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


FIG = importlib.import_module("scripts.revision.plot_fig5_aeri")
S7D = importlib.import_module("scripts.ext.S7d_aeri")
REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(directory):
    """Use S7d's actual recorded schema and its existing metric computations."""
    paths, aggregate_rows, expected = [], [], {}
    qpid = np.tile(np.arange(20), 10)
    gpid = np.repeat(np.arange(20), 5)
    for seed in range(3):
        rng = np.random.default_rng(2026 + seed)
        scores = rng.normal(size=(200, 100)).astype(np.float32)
        scores += (qpid[:, None] == gpid[None, :]).astype(np.float32) * .8
        bank = rng.normal(size=(250, 100)).astype(np.float32)

        def protocol(indices):
            matrix = scores[:, indices]
            pids = gpid[indices]
            penalty_t = np.sort(np.partition(matrix, matrix.shape[0] - 10, axis=0)[-10:], axis=0).astype(np.float64).mean(0).astype(np.float32)
            penalty_i = np.sort(np.partition(bank[:, indices], bank.shape[0] - 10, axis=0)[-10:], axis=0).astype(np.float64).mean(0).astype(np.float32)
            f0, ap0, t0 = S7D.evaluate(matrix, pids)
            ft, apt, _ = S7D.evaluate(matrix, pids, penalty_t)
            fi, api, _ = S7D.evaluate(matrix, pids, penalty_i)
            base, trans, ind = [S7D.metrics(f, ap) for f, ap in ((f0, ap0), (ft, apt), (fi, api))]
            excess, distinct, mx = S7D.ex_skew(matrix, t0)
            return {"ng": len(indices), "baseline": base, "transductive": trans, "inductive": ind,
                    "trans_delta_r1": round(trans["r1"] - base["r1"], 3),
                    "ind_delta_r1": round(ind["r1"] - base["r1"], 3),
                    "trans_ci": S7D.boot(f0 < 1, ft < 1), "ind_ci": S7D.boot(f0 < 1, fi < 1),
                    "excess_skew_N10": excess, "distinct_top1": distinct, "max_top1": mx}, f0 < 1, ft < 1, fi < 1

        with mock.patch.multiple(S7D, qpid=qpid, CHUNK=30, K=10, NBOOT=24, NULL=3, create=True):
            official, b0, t0, i0 = protocol(np.arange(100))
            ground, bg, tg, ig = protocol(np.arange(100)[::-1])
            mixed, bm, tm, im = protocol(np.roll(np.arange(100), 8))
            draws, corrected = [], []
            for draw_seed in range(2026, 2031):
                indices = np.random.default_rng(draw_seed).choice(100, 80, replace=False)
                row, bb, tt, ii = protocol(indices)
                draws.append(row)
                corrected.append((bb, tt, ii))
            controlled = {"seeds": list(range(2026, 2031)), "runs": draws,
                          "mean_r1": round(float(np.mean([r["baseline"]["r1"] for r in draws])), 3),
                          "mean_trans_delta": round(float(np.mean([r["trans_delta_r1"] for r in draws])), 3),
                          "mean_ind_delta": round(float(np.mean([r["ind_delta_r1"] for r in draws])), 3)}

            def paired_row(bb, tt, ii, controlled=False):
                ci_key = "identity_ci95_conditional_on_five_draws" if controlled else "identity_ci95"
                value_key = "mean_delta_r1_pp" if controlled else "delta_r1_pp"
                gain_key = "mean_delta_gain_r1_pp" if controlled else "delta_gain_r1_pp"
                result = {value_key: float(100 * (bb.mean() - b0.mean())), ci_key: S7D.boot(b0, bb), "correction_gain_difference": {}}
                for mode, current, reference in (("transductive", tt, t0), ("inductive", ii, i0)):
                    gain = current.astype(float) - bb.astype(float)
                    ref_gain = reference.astype(float) - b0.astype(float)
                    result["correction_gain_difference"][mode] = {gain_key: float(100 * (gain.mean() - ref_gain.mean())), ci_key: S7D.boot(ref_gain, gain)}
                return result

            mean_draw = np.mean(np.asarray(corrected, dtype=float), axis=0)
            paired = {"ground_minus_official_aerial": paired_row(bg, tg, ig),
                      "mixed_minus_official_aerial": paired_row(bm, tm, im),
                      "size_controlled_aerial_minus_official_aerial": paired_row(*mean_draw, controlled=True)}
        result = {"protocols": {"T2A_official": official, "T2G_ground_only": ground, "T2M_mixed": mixed,
                                "T2A_size_controlled_4220": controlled},
                  "paired_gallery_differences": paired,
                  "_meta": {"provenance": {"seeds": {"run_seed": seed, "bootstrap": 2026, "gallery": list(range(2026, 2031))}}}}
        path = directory / f"mock_S7d_seed{seed}.json"
        write_json(path, result)
        paths.append(str(path))
        expected[seed] = result
        aggregate_rows.append({"seed": seed, "baseline": official["baseline"]})
    aggregate = directory / "mock_aggregate.json"
    write_json(aggregate, {"datasets": {"aeri": {"per_seed": aggregate_rows}}})
    return paths, str(aggregate), expected


class RoundCFig5Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.inputs = self.directory / "inputs"
        self.inputs.mkdir()
        self.paths, self.aggregate, self.expected = fixture(self.inputs)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_schema_and_every_point_is_from_recorded_fields(self):
        data = FIG.prepare_data(self.paths, self.aggregate)
        self.assertEqual(len(data["points"]), 36)
        self.assertEqual(len(data["paired_points"]), 27)
        self.assertEqual(data["training_seeds"], [0, 1, 2])
        for row in data["points"]:
            source = self.expected[row["seed"]]
            cursor = source
            for field in row["source_field"].split("."):
                cursor = cursor[field]
            self.assertEqual(row["value"], cursor)
            if row["composition"] == "controlled_aerial":
                draws = source["protocols"]["T2A_size_controlled_4220"]["runs"]
                self.assertEqual(row["hub_skew"], np.mean([r["excess_skew_N10"] for r in draws]))
                self.assertIsNone(row["ci_low"])
                self.assertEqual(row["gallery_draws"], len(draws))
        for row in data["paired_points"]:
            cursor = self.expected[row["seed"]]
            for field in row["source_field"].split("."):
                cursor = cursor[field]
            self.assertEqual(row["value"], cursor)

    def test_missing_font_falls_back_and_artifacts_are_audited_and_exclusive(self):
        argv = ["--aggregate-json", self.aggregate, "--out", str(self.directory / "new_outputs"),
                "--tag", "mock", "--font", str(self.directory / "not_a_font.ttf")]
        for path in self.paths:
            argv.extend(["--s7d-json", path])
        captured = io.StringIO()
        before = {path: Path(path).read_bytes() for path in self.paths + [self.aggregate]}
        with contextlib.redirect_stdout(captured):
            result = FIG.main(argv)
        self.assertFalse(result["chinese_font_used"])
        self.assertIn("using English figure labels", captured.getvalue())
        self.assertEqual(before, {path: Path(path).read_bytes() for path in before})
        for figure in ("main", "paired"):
            png, pdf = [Path(result["artifacts"][figure][suffix]) for suffix in ("png", "pdf")]
            self.assertTrue(png.stat().st_size > 0)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            with Image.open(png) as im:
                self.assertAlmostEqual(im.info["dpi"][0], 300, places=1)
        with Path(result["artifacts"]["data_csv"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 87)
        self.assertEqual(len(result["provenance"]["inputs"]), 4)
        for evidence in result["provenance"]["inputs"]:
            self.assertEqual(evidence, FIG._evidence(evidence["path"]))
        directory = Path(result["artifacts"]["data_csv"]).parent
        self.assertTrue((directory / "fig5_aeri_plot_mock.json").exists())
        self.assertTrue((directory / "fig5_aeri_mock.log").exists())
        other = FIG._directory(self.directory / "new_outputs", "mock", [])
        self.assertNotEqual(directory, other)

    def test_missing_ci_is_not_inferred_and_aggregate_rounding_only_warns(self):
        result = self.expected[0]
        result["protocols"]["T2A_official"].pop("trans_ci")
        result["paired_gallery_differences"]["ground_minus_official_aerial"].pop("identity_ci95")
        write_json(Path(self.paths[0]), result)
        aggregate = json.loads(Path(self.aggregate).read_text(encoding="utf-8"))
        aggregate["datasets"]["aeri"]["per_seed"][0]["baseline"]["r1"] += 1e-7
        write_json(Path(self.aggregate), aggregate)
        data = FIG.prepare_data(self.paths, self.aggregate)
        row = next(row for row in data["points"] if row["seed"] == 0 and row["composition"] == "official_aerial" and row["metric"] == "trans_delta_r1")
        self.assertIsNone(row["ci_low"])
        row = next(row for row in data["paired_points"] if row["seed"] == 0 and row["composition"] == "ground_only" and row["metric"] == "baseline_difference_vs_official")
        self.assertIsNone(row["ci_low"])
        self.assertTrue(any("aggregate/S7d" in item for item in data["warnings"]))

    def test_bad_inputs_do_not_guess_or_fill_missing_metrics(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            FIG.prepare_data(self.paths[:2], self.aggregate)
        with self.assertRaisesRegex(ValueError, "distinct recorded"):
            FIG.prepare_data([self.paths[0], self.paths[0], self.paths[2]], self.aggregate)
        result = self.expected[0]
        result["_meta"]["provenance"]["seeds"].pop("run_seed")
        write_json(Path(self.paths[0]), result)
        with self.assertRaisesRegex(ValueError, "filename is not a seed source"):
            FIG.prepare_data(self.paths, self.aggregate)
        with self.assertRaisesRegex(ValueError, "finite recorded number"):
            FIG._number(float("nan"), "mock")

    def test_help_and_invalid_cli_are_clear(self):
        process = subprocess.run([sys.executable, str(REPO / "scripts/revision/plot_fig5_aeri.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        for parameter in ("--s7d-json", "--aggregate-json", "--font", "--out", "--tag"):
            self.assertIn(parameter, process.stdout)
        with contextlib.redirect_stderr(io.StringIO()) as captured, self.assertRaises(SystemExit):
            FIG.main(["--s7d-json", self.paths[0], "--aggregate-json", self.aggregate,
                      "--out", str(self.directory / "invalid"), "--tag", "mock"])
        self.assertIn("exactly three", captured.getvalue())
        self.assertFalse((self.directory / "invalid").exists())


if __name__ == "__main__":
    unittest.main()

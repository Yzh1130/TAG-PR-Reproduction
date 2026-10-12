"""Exercise every offline table CLI on explicit, handwritten mock archives."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.revision import render_revision_tables as RENDER
from scripts.revision._revision_table_format import TableFormatter
from tests.test_roundd_collection_tables import mock_pairs_documents, mock_runs_document
from tests.test_roundd_tag_tables import r3_fixture, table3_fixture, table5_fixture


class RoundDRendererCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.out = self.root / "new_tables"

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, data):
        path = self.inputs / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def run_table(self, command, flags, expected):
        before = {path: path.read_bytes() for path in self.inputs.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            result = RENDER.main([command, *flags, "--out", str(self.out)])
        content = Path(result["table"]).read_text(encoding="utf-8")
        self.assertIn(expected, content)
        self.assertIn("Provenance: inputs:", content)
        self.assertIn("script git HEAD=", content)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        for record in result["provenance"]["inputs"]:
            digest = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()[:16]
            self.assertEqual(record["sha256_16"], digest)
            self.assertIn(digest, content)
        self.assertIn(content, Path(result["all_tables"]).read_text(encoding="utf-8"))
        return result

    def test_table5_cli(self):
        path = self.write_json("mock_table5.json", table5_fixture())
        self.run_table("table5", ["--json", path, "--label", "Mock B/16"], "2.00 ± 1.00")

    def test_table3_cli(self):
        path = self.write_json("mock_table3.json", table3_fixture())
        self.run_table("table3", ["--json", path], "50.00 [0.00, 100.00]")

    def test_table11_cli_missing_stays_na(self):
        path = self.write_json("mock_aggregate.json", {"datasets": {"aeri": {
            "per_seed": [{"seed": 0, "baseline": {"r1": 12.345},
                          "transductive": {"delta": {"r1": .16}, "identity_ci95": [-.42, .55]}}]}}})
        self.run_table("table11", ["--aggregate", path], "+0.16 [−0.42, +0.55]")

    def test_table12_cli_actual_seed(self):
        path = self.write_json("mock_distractor_seed99.json", {"aeri": {
            "fits": {"trans_cohortQ_delta": {"slope_vs_log10D": -.42, "slope_ci95": [-.5, .16]}},
            "spearman_excess_skew_vs_D": .3},
            "_meta": {"provenance": {"seeds": {"run_seeds": {"aeri": 2}}}}})
        self.run_table("table12", ["--s7b", path], "| AERI-PEDES | 2 | trans_cohortQ_delta | −0.42 [−0.50, +0.16]")

    def test_pairs_cli_merges_two_csvs(self):
        flags = []
        for name, rows in mock_pairs_documents():
            path = self.inputs / name
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            flags += ["--csv", str(path)]
        self.run_table("pairs", flags, "| population std (ddof=0) | 1.00 | 2.00 | 1.00 |")

    def test_r3_cli_market_null(self):
        path = self.write_json("mock_r3.json", r3_fixture())
        self.run_table("table-r3", ["--json", path], "| Market | 1 | 0 | 0 | n/a [n/a] | n/a [n/a]")

    def test_runs_cli_and_filter(self):
        path = self.write_json("mock_runs.json", mock_runs_document())
        self.run_table("runs", ["--json", path, "--include", "e2_nrw"], "checkpoint_last.pth / aaaaaaaaaaaaaaaa / 42 bytes")

    def test_s13_cli_four_complete_rows_and_baseline_na(self):
        settings = [("uncorrected", None, None, None), ("inductive_default", "train_full", .5, 10),
                    ("inductive_075_30", "train_full", .75, 30), ("transductive_k10", "test_full", .5, 10)]
        rows = [{"setting": setting, "bank": bank, "alpha": alpha, "k": k,
                 "metrics": {"R1": 12.3, "R5": 23.4, "R10": 34.5, "mAP": 9.8}, "delta_R1_pp": .16,
                 "identity_bootstrap": {"ci95_percentile_pp": [-.42, .55]} if bank else None}
                for setting, bank, alpha, k in settings]
        path = self.write_json("mock_s13.json", {"rows": rows})
        self.run_table("s13", ["--summary", "Mock L/14=" + path], "| train_full (α=0.75, k=30) | 12.30 | 23.40 | 34.50 | 9.80 | +0.16 [−0.42, +0.55]")

    def test_accumulation_collision_and_modified_output_guard(self):
        a = self.write_json("mock_table3.json", table3_fixture())
        b = self.write_json("mock_r3.json", r3_fixture())
        first = self.run_table("table3", ["--json", a], "Table 3")
        second = self.run_table("table-r3", ["--json", b], "R3 —")
        combined = Path(second["all_tables"]).read_text(encoding="utf-8")
        self.assertIn("Table 3", combined)
        self.assertIn("R3 —", combined)
        again = self.run_table("table3", ["--json", a], "Table 3")
        self.assertNotEqual(first["table"], again["table"])
        table = Path(first["table"])
        table.write_text(table.read_text(encoding="utf-8") + "manual change", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            RENDER.main(["table3", "--json", a, "--out", str(self.out)])
        self.assertIn("existing generated table changed", error.getvalue())

    def test_clear_input_errors_before_outputs_and_help(self):
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            RENDER.main(["table3", "--json", str(self.inputs / "absent.json"), "--out", str(self.out)])
        self.assertIn("input file does not exist", error.getvalue())
        self.assertFalse(self.out.exists())
        path = self.write_json("mock_table3.json", table3_fixture())
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            RENDER.main(["table3", "--json", path, "--out", str(self.inputs / "forbidden")])
        self.assertIn("read-only input", error.getvalue())
        self.assertFalse((self.inputs / "forbidden").exists())
        for command in ("table5", "table3", "table11", "table12", "pairs", "table-r3", "runs", "s13"):
            process = subprocess.run([sys.executable, str(Path(RENDER.__file__)), command, "--help"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("--decimals", process.stdout)

    def test_ambiguous_csv_corrupt_manifest_and_format_overflow_stop(self):
        path = self.inputs / "bad.csv"
        path.write_text("pair,metric,seed,status,A_R1,A_R1\na|b,last,0,paired,1,2\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            RENDER.main(["pairs", "--csv", str(path), "--out", str(self.out)])
        self.assertIn("duplicate CSV headers", error.getvalue())
        self.assertFalse(self.out.exists())
        json_path = self.write_json("mock_table3.json", table3_fixture())
        result = self.run_table("table3", ["--json", json_path], "Table 3")
        manifest_path = Path(result["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["entries"] = [None]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            RENDER.main(["table3", "--json", json_path, "--out", str(self.out)])
        self.assertIn("entries must be objects", error.getvalue())
        with self.assertRaisesRegex(ValueError, "unit conversion overflowed"):
            TableFormatter().number(1e308, "mock", percent=True)


if __name__ == "__main__":
    unittest.main()

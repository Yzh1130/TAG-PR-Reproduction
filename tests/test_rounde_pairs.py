"""Handwritten alias fixtures: no archive scan, model evaluation, or metrics."""
from __future__ import annotations

import json
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from scripts.revision._revision_table_format import TableFormatter
from scripts.revision._revision_tables_collection import render_pairs
from scripts.revision import render_revision_tables as RENDER


HISTORICAL = "bs320_ep10|nrw_view_norm_ep10"
TARGET = "bs320_ep10|e2_nrw_view_norm"


def row(pair, seed, a, b, delta, status="paired", scope="last", reason=""):
    return {
        "pair": pair, "metric": scope, "seed": str(seed), "status": status,
        "reason": reason, "A_R1": str(a) if a is not None else "",
        "B_R1": str(b) if b is not None else "",
        "delta_R1": str(delta) if delta is not None else "",
        "A_directories": json.dumps([f"/mock/baseline_seed{seed}"]),
        "B_directories": json.dumps([f"C:\\mock\\variant_seed{seed}"]) if b is not None else "[]",
    }


class PairsAliasTests(unittest.TestCase):
    def _cli_inputs(self, root, collision=False):
        inputs = root / "inputs"
        inputs.mkdir()
        paths = []
        for name, data in (("historical.csv", row(HISTORICAL, 0, 10, 9, -1)),
                           ("current.csv", row(TARGET, 0 if collision else 1, 12, 13, 1))):
            path = inputs / name
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data))
                writer.writeheader()
                writer.writerow(data)
            paths.append(path)
        return paths

    def test_cli_alias_is_forwarded_and_recorded_in_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._cli_inputs(root)
            before = [path.read_bytes() for path in paths]
            alias = HISTORICAL + "=" + TARGET
            argv = ["pairs", "--csv", str(paths[0]), "--csv", str(paths[1]),
                    "--alias", alias, "--out", str(root / "output")]
            with contextlib.redirect_stdout(io.StringIO()):
                result = RENDER.main(argv)
            text = Path(result["table"]).read_text(encoding="utf-8")
            self.assertEqual(text.count("### "), 1)
            self.assertIn("| mean | 11.00 | 11.00 | +0.00 |", text)
            # Markdown cells escape Windows backslashes; the path is unchanged.
            self.assertIn(str(paths[0]).replace("\\", "\\\\") + ":2", text)
            self.assertIn("name=variant_seed1", text)
            self.assertEqual(result["provenance"]["argv"][1:], argv)
            self.assertEqual(before, [path.read_bytes() for path in paths])
            self.assertEqual(RENDER.build_parser().parse_args(
                ["pairs", "--csv", str(paths[0]), "--out", str(root / "unused")]).alias, [])

    def test_cli_alias_conflict_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._cli_inputs(root, collision=True)
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
                RENDER.main(["pairs", "--csv", str(paths[0]), "--csv", str(paths[1]),
                             "--alias", HISTORICAL + "=" + TARGET, "--out", str(root / "output")])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("alias collision", errors.getvalue())
            self.assertFalse((root / "output").exists())

    def test_alias_combines_disjoint_seeds_manual_statistics_and_sources(self):
        documents = [
            ("historical.csv", [row(HISTORICAL, 0, 10, 9, -1)]),
            ("current.csv", [row(TARGET, 1, 12, 13, 1)]),
        ]
        text = render_pairs(documents, TableFormatter(), aliases=[HISTORICAL + "=" + TARGET])
        self.assertEqual(text.count("### "), 1)
        self.assertIn("### bs320_ep10|e2_nrw_view_norm (last)", text)
        self.assertIn("| 0 | 10.00 | 9.00 | −1.00 | paired |", text)
        self.assertIn("| 1 | 12.00 | 13.00 | +1.00 | paired |", text)
        self.assertIn("| mean | 11.00 | 11.00 | +0.00 |", text)
        self.assertIn("| population std (ddof=0) | 1.00 | 2.00 | 1.00 |", text)
        self.assertIn("| range [min, max] | 2.00 [10.00, 12.00] | 4.00 [9.00, 13.00] | 2.00 [−1.00, +1.00] |", text)
        self.assertIn("historical.csv:2", text)
        self.assertIn("current.csv:2", text)
        self.assertIn("name=baseline_seed0", text)
        self.assertIn("name=variant_seed1", text)

    def test_alias_collision_fails_even_when_values_equal(self):
        same = row(HISTORICAL, 0, 10, 11, 1)
        other = dict(same, pair=TARGET)
        with self.assertRaisesRegex(ValueError, "alias collision.*not overwriting even equal values"):
            render_pairs([("a.csv", [same]), ("b.csv", [other])], TableFormatter(),
                         aliases=[HISTORICAL + "=" + TARGET])

    def test_no_alias_preserves_exact_duplicate_and_conflict_rules(self):
        existing = row(HISTORICAL, 0, 10, 11, 1)
        documents = [("a.csv", [existing]), ("b.csv", [existing.copy()])]
        text = render_pairs(documents, TableFormatter())
        self.assertEqual(text, render_pairs(documents, TableFormatter(), aliases=[]))
        self.assertEqual(text.count("| 0 | 10.00 | 11.00 | +1.00 |"), 1)
        self.assertIn("a.csv:2, b.csv:2", text)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            render_pairs([("a.csv", [existing]), ("b.csv", [dict(existing, B_R1="12")])], TableFormatter())

    def test_unpaired_placeholder_supplies_no_cells_and_keeps_provenance(self):
        missing = row(HISTORICAL, 1, 999, None, None, "unpaired", reason="missing B")
        active = row(TARGET, 1, 12, 13, 1)
        text = render_pairs([("hist.csv", [missing]), ("new.csv", [active])], TableFormatter(),
                            aliases=[HISTORICAL + "=" + TARGET])
        self.assertIn("| 1 | 12.00 | 13.00 | +1.00 | paired |", text)
        self.assertIn("hist.csv:2: missing B", text)
        self.assertIn("(recorded empty list)", text)
        self.assertNotIn("999.00", text)
        self.assertIn("| mean | 12.00 | 13.00 | +1.00 |", text)

    def test_multiple_unpaired_rows_remain_na_without_partial_value_filling(self):
        a = row(HISTORICAL, 0, 10, None, None, "unpaired", reason="missing B")
        b = row(TARGET, 0, None, 12, None, "unpaired", reason="missing A")
        fmt = TableFormatter()
        text = render_pairs([("a.csv", [a]), ("b.csv", [b])], fmt,
                            aliases=[HISTORICAL + "=" + TARGET])
        self.assertIn("| 0 | n/a | n/a | n/a | unpaired | missing B; missing A |", text)
        self.assertIn("| mean | n/a | n/a | n/a |", text)
        self.assertNotIn("10.00", text)
        self.assertNotIn("12.00", text)

    def test_alias_chain_is_explicit_and_checkpoint_scopes_stay_separate(self):
        middle = "base|intermediate"
        documents = [("last.csv", [row(HISTORICAL, 0, 10, 11, 1)]),
                     ("best.csv", [row(TARGET, 0, 12, 13, 1, scope="best")])]
        text = render_pairs(documents, TableFormatter(),
                            aliases=[HISTORICAL + "=" + middle, middle + "=" + TARGET])
        self.assertIn("### bs320_ep10|e2_nrw_view_norm (last)", text)
        self.assertIn("### bs320_ep10|e2_nrw_view_norm (best)", text)

    def test_alias_input_rejects_cycles_duplicates_and_ambiguity(self):
        documents = [("a.csv", [row(HISTORICAL, 0, 10, 11, 1)])]
        invalid = [
            ([HISTORICAL + "=" + HISTORICAL], "cycle"),
            ([HISTORICAL + "=" + TARGET, TARGET + "=" + HISTORICAL], "cycle"),
            ([HISTORICAL + "=" + TARGET, HISTORICAL + "=" + TARGET], "duplicate or ambiguous"),
            ([HISTORICAL + "=" + TARGET, HISTORICAL + "=a|b"], "duplicate or ambiguous"),
            (["bad"], "OLD_PAIR=TARGET_PAIR"),
            (["bad=target"], "two nonempty prefixes"),
            (["=" + TARGET], "nonempty"),
            ([HISTORICAL + "=" + TARGET + " "], "whitespace"),
        ]
        for aliases, message in invalid:
            with self.subTest(aliases=aliases), self.assertRaisesRegex(ValueError, message):
                render_pairs(documents, TableFormatter(), aliases=aliases)

    def test_missing_directory_provenance_is_na_not_derived(self):
        data = row(HISTORICAL, 0, 10, 11, 1)
        del data["A_directories"]
        fmt = TableFormatter()
        text = render_pairs([("a.csv", [data])], fmt)
        self.assertIn("A directories", text)
        self.assertTrue(any("A_directories" in value for value in fmt.missing))
        with self.assertRaisesRegex(ValueError, "collector JSON directory array"):
            render_pairs([("a.csv", [dict(data, B_directories="['not JSON']")])], TableFormatter())


if __name__ == "__main__":
    unittest.main()

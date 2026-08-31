"""Filesystem-only audit tests; all artifacts live in a disposable temp tree."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ._analysis_audit import Audit, read_json


class AuditTests(unittest.TestCase):
    def test_exclusive_outputs_hash_exact_gate_and_failure_report(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / "input.json"
            source.write_text('{"synthetic_fixture": true}', encoding="utf-8")
            before = source.read_bytes()
            audit = Audit("S14", argparse.Namespace(out=root / "_e1_s14"),
                          argv=["MOCK"], seeds={"mock": 2026})
            with audit.capture():
                print("MOCK AUDIT LOG")
                self.assertEqual(audit.read_json(source, "fixture"), {"synthetic_fixture": True})
                self.assertEqual(audit.report["inputs"]["fixture"]["sha256_16"],
                                 hashlib.sha256(before).hexdigest()[:16])
                first = audit.write_json("identity_folds_seed2026.json", {"mock": 1})
                second = audit.write_json("identity_folds_seed2026.json", {"mock": 2})
                self.assertNotEqual(first, second)
                self.assertEqual(read_json(first), {"mock": 1})
                self.assertEqual(read_json(second), {"mock": 2})
                self.assertNotEqual(audit.new_run_dir(), audit.new_run_dir())
                audit.check("exact", {"R1": 1.25}, {"R1": 1.25})
                with self.assertRaisesRegex(RuntimeError, "exact check FAILED"):
                    audit.check("must fail", {"R1": 1.25}, {"R1": 1.250000000000001})
                report, log = audit.finish("failed", "intentional MOCK failure")
            payload = read_json(report)
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["argv"], ["MOCK"])
            self.assertEqual(len(payload["git_HEAD"]), 40)
            self.assertIn("MOCK AUDIT LOG", log.read_text(encoding="utf-8"))
            self.assertEqual(source.read_bytes(), before)
            next_audit = Audit("S14", argparse.Namespace(out=root / "_e1_s14"))
            next_report, next_log = next_audit.finish()
            self.assertNotEqual(next_report, report)
            self.assertNotEqual(next_log, log)

    def test_constructor_never_opens_e3_input(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            args = argparse.Namespace(out=root / "_e1_s15", e3_json=root / "DO_NOT_READ.json")
            audit = Audit("S15", args)
            self.assertEqual(audit.report["inputs"], {})
            audit.finish()

    def test_protected_roots_and_nested_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for destination in (root / "_dump" / "_e1_s14", root / "_archive_evidence" / "_e1_s14",
                                root / "_e1" / "_e1_s14", root / "arbitrary"):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    Audit("S14", argparse.Namespace(out=destination))


if __name__ == "__main__":
    unittest.main()

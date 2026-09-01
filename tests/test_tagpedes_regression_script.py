"""Static safety contract for the opt-in TAG-PEDES regression runner."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "regression_tagpedes.sh"


class TagPedesRegressionScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_script_is_strict_and_checkpoint_is_mandatory(self):
        self.assertRegex(self.source, r"(?m)^set\s+-euo\s+pipefail\s*$")
        self.assertRegex(
            self.source,
            r"\$\{TAGPEDES_CHECKPOINT:\?[^}]+\}",
            "TAGPEDES_CHECKPOINT must fail before running when unset",
        )

    def test_environment_defaults_are_explicit(self):
        self.assertRegex(
            self.source,
            r"\$\{TAGPEDES_CONFIG:-config/config\.yaml\}",
        )
        for variable in (
            "TAGPEDES_OUTPUT_ROOT",
            "TAGPEDES_NPROC",
            "TAGPEDES_PYTHON",
        ):
            with self.subTest(variable=variable):
                self.assertRegex(
                    self.source,
                    rf"\$\{{{variable}:-[^}}]+\}}",
                    f"{variable} must remain an optional override with a default",
                )

    def test_invocation_is_eval_only_and_dumps_float32_similarity(self):
        required_arguments = (
            "--config",
            "--eval_only",
            "--resume",
            "--dump_sim",
            "--dump_sim_dtype",
            "float32",
        )
        for argument in required_arguments:
            with self.subTest(argument=argument):
                self.assertIn(argument, self.source)
        self.assertNotIn("_archive_evidence", self.source)
        self.assertNotIn("_dump/", self.source)
        self.assertNotRegex(self.source, r"(?:^|/)\_e1[^A-Za-z0-9_]",)

    def test_prints_the_first_16_hex_digits_of_sim_qg_sha256(self):
        self.assertIn("sim_qg.npy", self.source)
        self.assertRegex(self.source.lower(), r"sha-?256|hashlib\.sha256")
        # Accept either `cut -c1-16`, shell substring syntax, or a Python
        # `hexdigest()[:16]` helper, but never an undocumented full digest only.
        prefix_forms = (
            r"cut\s+-c\s*1-16",
            r":0:16",
            r"hexdigest\(\)\s*\[:16\]",
        )
        self.assertTrue(
            any(re.search(pattern, self.source) for pattern in prefix_forms),
            "script must print exactly the first 16 SHA-256 hex characters",
        )

    def test_shell_syntax_when_bash_is_available(self):
        # Windows' legacy WSL launcher corrupts stdin/stderr encoding when it
        # is driven through subprocess pipes.  The same script is checked by
        # real Bash on the Linux/AutoDL target where it is intended to run.
        if os.name == "nt":
            self.skipTest("Bash stdin syntax check is Linux-only")
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable for the optional syntax check")
        result = subprocess.run(
            [bash, "-n"],
            # Supplying bytes preserves LF line endings on Windows.  Text-mode
            # stdin rewrites them to CRLF, which breaks the embedded heredoc
            # even though the checked-in shell script is valid on Linux.
            input=self.source.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", errors="backslashreplace"),
        )


if __name__ == "__main__":
    unittest.main()

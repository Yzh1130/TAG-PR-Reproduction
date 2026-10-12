"""Round-D font regressions; numeric fixture uses the existing S7d schema."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from scripts.revision import plot_fig5_aeri as FIG
from tests.test_roundc_fig5 import fixture


class RoundDFig5Tests(unittest.TestCase):
    def test_font_precedes_layout_and_preserves_style_without_global_change(self):
        fig, axis = plt.subplots()
        title = axis.set_title("Mock title", fontsize=19, fontweight="bold")
        label = axis.set_xlabel("Mock label", fontsize=13)
        original = matplotlib.rcParams["axes.unicode_minus"]
        layout = fig.tight_layout
        font = FontProperties(family="DejaVu Serif")

        def check_layout():
            self.assertEqual(title.get_fontfamily(), ["DejaVu Serif"])
            self.assertEqual(label.get_fontfamily(), ["DejaVu Serif"])
            self.assertEqual(title.get_fontsize(), 19)
            self.assertEqual(title.get_fontweight(), "bold")
            self.assertEqual(label.get_fontsize(), 13)
            self.assertFalse(matplotlib.rcParams["axes.unicode_minus"])
            layout()

        with mock.patch.object(fig, "tight_layout", side_effect=check_layout) as called:
            FIG._layout_figure(fig, font)
        called.assert_called_once()
        self.assertEqual(matplotlib.rcParams["axes.unicode_minus"], original)
        plt.close(fig)

    def test_mock_figures_have_no_missing_glyph_warning_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / "inputs"
            inputs.mkdir()
            paths, aggregate, _ = fixture(inputs)
            evidence = {path: Path(path).read_bytes() for path in paths + [aggregate]}
            expected = FIG.prepare_data(paths, aggregate)
            font = Path("C:/Windows/Fonts/simhei.ttf")
            argv = ["--aggregate-json", aggregate, "--out", str(root / "outputs"), "--tag", "mock"]
            if font.is_file():
                argv += ["--font", str(font)]
            for path in paths:
                argv += ["--s7d-json", path]
            with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                result = FIG.main(argv)
            self.assertFalse([str(item.message) for item in caught if "Glyph" in str(item.message)])
            self.assertEqual(result["points"], expected["points"])
            self.assertEqual(result["paired_points"], expected["paired_points"])
            self.assertEqual(evidence, {path: Path(path).read_bytes() for path in evidence})
            for kind in ("main", "paired"):
                for extension in ("png", "pdf"):
                    self.assertGreater(Path(result["artifacts"][kind][extension]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

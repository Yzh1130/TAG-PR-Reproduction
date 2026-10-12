"""Paper-layout checks using actual S7d schema and 200 x 100 mock scores.

All plotted values are compared with the saved mock input fields, not with
hard-coded retrieval scores. Tests never modify any input JSON.
"""

import contextlib
import copy
import csv
import importlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from tests.test_roundc_fig5 import fixture


FIG = importlib.import_module("scripts.revision.plot_fig5_aeri")


class Fig5PaperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        inputs = self.directory / "inputs"
        inputs.mkdir()
        self.paths, self.aggregate, _ = fixture(inputs)
        self.before = {path: Path(path).read_bytes()
                       for path in self.paths + [self.aggregate]}

    def tearDown(self):
        self.temp.cleanup()

    def argv(self, layout=None):
        args = ["--aggregate-json", self.aggregate,
                "--out", str(self.directory / "outputs"), "--tag", "mock"]
        for path in self.paths:
            args.extend(["--s7d-json", path])
        if layout is not None:
            args.extend(["--layout", layout])
        return args

    def test_paper_outputs_three_panels_fixed_canvas_and_recorded_values(self):
        from matplotlib.collections import PathCollection, LineCollection
        from matplotlib.text import Text

        prepared = FIG.prepare_data(self.paths, self.aggregate)
        captured_figures = []
        original_save = FIG._save_paper_figure

        def save(figure, *args):
            result = original_save(figure, *args)
            captured_figures.append(figure)
            return result

        stdout = io.StringIO()
        with mock.patch.object(FIG, "_save_paper_figure", side_effect=save), \
                contextlib.redirect_stdout(stdout):
            result = FIG.main(self.argv("paper") + ["--font", str(self.directory / "missing.ttf")])
        self.assertEqual(result["points"], prepared["points"])
        self.assertEqual(result["paired_points"], prepared["paired_points"])
        self.assertEqual(self.before, {path: Path(path).read_bytes() for path in self.before})
        self.assertIn("using English figure labels", stdout.getvalue())
        self.assertIn("Suggested caption:", stdout.getvalue())
        self.assertFalse(result["chinese_font_used"])

        self.assertEqual(len(captured_figures), 1)
        figure = captured_figures[0]
        self.assertEqual(len(figure.axes), 3)
        self.assertAlmostEqual(figure.get_figwidth() * 2.54, 17)
        self.assertTrue(all(text.get_fontsize() >= 7 for text in figure.findobj(Text)))
        self.assertIsNone(figure._suptitle)
        for axis, letter in zip(figure.axes, "abc"):
            self.assertEqual(axis.get_title(), "")
            self.assertEqual(axis.get_title(loc="left"), "")
            self.assertEqual(axis.get_title(loc="right"), "")
            self.assertTrue(any(text.get_text() == f"({letter})" for text in axis.texts))
            self.assertIsNone(axis.get_legend())
        self.assertEqual(len(figure.legends), 1)
        legend = figure.legends[0]
        self.assertEqual(legend._ncols, len(legend.get_texts()))
        self.assertIn("Controlled (5-draw mean)", [text.get_text() for text in legend.get_texts()])
        self.assertEqual(result["paper_layout"]["panel_count"], len(figure.axes))

        gain_axis, paired_axis, hub_axis = figure.axes
        gain_rows = [row for mode in ("trans_delta_r1", "ind_delta_r1")
                     for row in prepared["points"] if row["metric"] == mode]
        gain_points = [item for item in gain_axis.collections if isinstance(item, PathCollection)]
        self.assertEqual(len(gain_points), len(gain_rows))
        for point, row in zip(gain_points, gain_rows):
            self.assertEqual(float(point.get_offsets()[0, 1]), row["value"])
        # No original R@1 is plotted: only the two delta modes contribute points.
        self.assertEqual(len(gain_points), sum(row["metric"] != "baseline_r1"
                                              for row in prepared["points"]))
        self.assertEqual(len([item for item in gain_axis.collections if isinstance(item, LineCollection)]),
                         2 * len(FIG.COMPOSITIONS))
        self.assertTrue(any(np.array_equal(line.get_ydata(), [0, 0]) for line in gain_axis.lines))

        paired_rows = [row for row in prepared["paired_points"]
                       if row["metric"] == "transductive_gain_difference_vs_official"]
        paired_points = [item for item in paired_axis.collections if isinstance(item, PathCollection)]
        paired_intervals = [item for item in paired_axis.collections if isinstance(item, LineCollection)]
        self.assertEqual(len(paired_points), len(paired_rows))
        self.assertEqual(len(paired_intervals), 2 * len(paired_rows))
        for index, (point, row) in enumerate(zip(paired_points, paired_rows)):
            self.assertEqual(float(point.get_offsets()[0, 1]), row["value"])
            segment = paired_intervals[index * 2].get_segments()[0]
            np.testing.assert_array_equal(segment[:, 1], [row["ci_low"], row["ci_high"]])
        self.assertTrue(any(np.array_equal(line.get_ydata(), [0, 0]) for line in paired_axis.lines))

        hub_points = [item for item in hub_axis.collections if isinstance(item, PathCollection)]
        self.assertEqual(len(hub_points), len(gain_rows))
        hollow_count = 0
        for point, row in zip(hub_points, gain_rows):
            np.testing.assert_array_equal(point.get_offsets()[0], [row["hub_skew"], row["value"]])
            hollow = point.get_facecolors().size == 0
            self.assertEqual(hollow, row["composition"] == "controlled_aerial")
            hollow_count += hollow
        self.assertEqual(hollow_count, 2 * len(prepared["training_seeds"]))
        self.assertFalse(hub_axis.lines)  # No fitted line.

        for suffix in ("png", "pdf"):
            path = Path(result["artifacts"]["paper"][suffix])
            self.assertGreater(path.stat().st_size, 0)
        with Image.open(result["artifacts"]["paper"]["png"]) as image:
            self.assertLessEqual(abs(image.size[0] - 17 / 2.54 * 300), 1)
            self.assertAlmostEqual(image.info["dpi"][0], 300, places=1)
        self.assertTrue(Path(result["artifacts"]["paper"]["pdf"]).read_bytes().startswith(b"%PDF"))
        with Path(result["artifacts"]["data_csv"]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(prepared["points"]) + len(gain_rows) + len(prepared["paired_points"]))
        csv_gain = [row for row in rows if row["plot"] == "gallery_composition"]
        self.assertEqual([float(row["value"]) for row in csv_gain],
                         [row["value"] for row in prepared["points"]])

    def test_omitted_and_explicit_legacy_route_to_original_plotter(self):
        self.assertEqual(FIG.build_parser().parse_args(self.argv()).layout, "legacy")
        for layout in (None, "legacy"):
            with self.subTest(layout=layout), \
                    mock.patch.object(FIG, "plot_data", return_value=({}, False)) as legacy, \
                    mock.patch.object(FIG, "plot_paper_data") as paper, \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                result = FIG.main(self.argv(layout))
            legacy.assert_called_once()
            paper.assert_not_called()
            self.assertNotIn("paper_layout", result)
            self.assertNotIn("Suggested caption:", stdout.getvalue())
            self.assertEqual(self.before, {path: Path(path).read_bytes() for path in self.before})

    def test_paper_does_not_infer_missing_pairs_or_five_draw_evidence(self):
        data = FIG.prepare_data(self.paths, self.aggregate)
        bad = copy.deepcopy(data)
        bad["paired_points"] = [row for row in bad["paired_points"]
                                if not (row["seed"] == data["training_seeds"][0]
                                        and row["metric"] == "transductive_gain_difference_vs_official")]
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "paired differences"):
            FIG.plot_paper_data(bad, self.directory, "mock")
        bad = copy.deepcopy(data)
        next(row for row in bad["points"] if row["composition"] == "controlled_aerial")["gallery_draws"] = 4
        with contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(ValueError, "five controlled-gallery draws"):
            FIG.plot_paper_data(bad, self.directory, "mock")
        self.assertEqual(self.before, {path: Path(path).read_bytes() for path in self.before})


if __name__ == "__main__":
    unittest.main()

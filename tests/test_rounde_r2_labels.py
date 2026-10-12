"""Round E panel-label tests use mock images, never formal retrieval results."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from scripts import R2_make_figure as figure


class RoundER2LabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rounde_r2_labels_mock_")
        self.root = Path(self.temp.name)
        inputs = self.root / "inputs"
        inputs.mkdir()
        self.images = []
        for index in range(8):
            path = inputs / f"gallery_{index}.png"
            Image.new("RGB", (20, 40), (index * 20, 70, 120)).save(path)
            self.images.append({"path": str(path), "gallery_index": index,
                                "pid": index % 2, "hit": index % 2 == 0})
        self.case = {"query_index": 17, "query_text": "中文描述 must stay outside the image",
                     "source_image": self.images[0], "before_top3": self.images[1:4],
                     "after_top3": self.images[4:7], "truth_image": self.images[7],
                     "correction_condition": "mock saved correction alpha=.5 k=10"}
        self.selected = {"panel": "a", "query_index": 17}
        self.pool = inputs / "pool.json"
        self.pool.write_text(json.dumps({"pools": {"transductive_success": {
            "cases": [self.case], "seed": 2026, "selected_query_indices": [17],
            "correction_condition": self.case["correction_condition"]}}}, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def render(self, name, font=None, labels="en"):
        return figure.render_panels([self.case], [self.selected], self.root / f"{name}.png", font, labels)

    @staticmethod
    def layout(result):
        return json.loads(Path(result["layout_json"]).read_text(encoding="utf-8"))

    def test_default_english_is_the_original_rendering_path(self):
        default = figure.render_panels([self.case], [self.selected], self.root / "default.png")
        explicit = self.render("english", labels="en")
        self.assertEqual(Path(default["png"]).read_bytes(), Path(explicit["png"]).read_bytes())
        self.assertEqual(self.layout(default), self.layout(explicit))
        self.assertEqual(self.layout(default)["language"], "English")
        self.assertNotIn("column_headers", self.layout(default))
        self.assertNotIn("rendered_labels", default)
        self.assertFalse(default["query_text_drawn"])
        self.assertEqual(figure.build_parser().parse_args(["--out", "mock"]).labels, "en")

    def test_chinese_headers_are_drawn_from_the_explicit_font(self):
        candidates = (Path("C:/Windows/Fonts/simhei.ttf"), Path("C:/Windows/Fonts/msyh.ttc"),
                      Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
        font_path = next((path for path in candidates if path.is_file()), None)
        if font_path is None:
            self.skipTest("no explicit CJK font available for the Chinese pixel rendering test")
        drawn = []
        original = ImageDraw.ImageDraw.text

        def capture(draw, xy, text, *args, **kwargs):
            drawn.append(text)
            return original(draw, xy, text, *args, **kwargs)

        stdout = io.StringIO()
        with patch.object(ImageDraw.ImageDraw, "text", capture), redirect_stdout(stdout):
            result = self.render("chinese", str(font_path), "zh")
        layout = self.layout(result)
        self.assertEqual(drawn[:8], ["查询源图", "校正前#1", "校正前#2", "校正前#3",
                                     "校正后#1", "校正后#2", "校正后#3", "最优正样本"])
        self.assertEqual(drawn[8], "(a)\nq=17")
        self.assertNotIn(self.case["query_text"], drawn)
        self.assertEqual(layout["requested_labels"], "zh")
        self.assertEqual(layout["rendered_labels"], "zh")
        self.assertEqual(layout["language"], "Chinese")
        self.assertEqual(layout["font"], str(font_path.resolve()))
        self.assertIsNone(layout["label_font_fallback"])
        self.assertNotIn("WARNING", stdout.getvalue())
        self.assertTrue(Path(result["pdf"]).read_bytes().startswith(b"%PDF"))
        self.assertEqual(len(layout["rows"][0]["slots"]), 8)

    def test_missing_glyphs_fall_back_to_english_without_another_cjk_font(self):
        candidates = (Path("C:/Windows/Fonts/arial.ttf"),
                      Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        font_path = next((path for path in candidates if path.is_file()), None)
        calls = []
        original = figure.choose_font

        def capture(texts, font_path=None, size=15):
            calls.append((list(texts), font_path))
            return original(texts, font_path, size)

        stdout = io.StringIO()
        if font_path is None:
            with patch.object(figure, "choose_font", return_value=(None, None)), redirect_stdout(stdout):
                result = self.render("missing_glyphs", "mock_non_cjk_font.ttf", "zh")
        else:
            with patch.object(figure, "choose_font", capture), redirect_stdout(stdout):
                result = self.render("missing_glyphs", str(font_path), "zh")
            self.assertEqual(calls[0], (figure.CHINESE_PANEL_HEADERS, str(font_path)))
            self.assertFalse(any(any("查询" in text for text in texts) and path is None
                                 for texts, path in calls))
        layout = self.layout(result)
        self.assertEqual(layout["language"], "English")
        self.assertEqual(layout["column_headers"], figure.ENGLISH_PANEL_HEADERS)
        self.assertEqual(result["requested_labels"], "zh")
        self.assertEqual(result["rendered_labels"], "en")
        self.assertIn("WARNING:", stdout.getvalue())
        self.assertIn("falling back to English panel labels", stdout.getvalue())
        self.assertTrue(layout["label_font_fallback"])

    def test_cli_missing_font_fallback_records_language_and_requested_path(self):
        missing_font = self.root / "absent_font.ttf"
        out = self.root / "output"
        argv = ["--pool-json", f"mock={self.pool}", "--panel", "a:17@mock", "--out", str(out),
                "--labels", "zh", "--font", str(missing_font)]
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(figure.main(argv), 0)
        report = json.loads(next(out.glob("R2_make_figure*.json")).read_text(encoding="utf-8"))
        self.assertEqual(report["provenance"]["arguments"]["labels"], "zh")
        self.assertEqual(report["provenance"]["arguments"]["font"], str(missing_font))
        self.assertNotIn("font", report["provenance"]["inputs"])
        self.assertEqual(report["rendered"]["rendered_labels"], "en")
        self.assertIn("font file does not exist:", stdout.getvalue())
        self.assertEqual(self.layout(report["rendered"])["column_headers"], figure.ENGLISH_PANEL_HEADERS)
        selected = json.loads(Path(report["selected_cases_json"]).read_text(encoding="utf-8"))
        self.assertEqual(selected["cases"][0]["query_text"], self.case["query_text"])
        self.assertEqual(selected["rendered"]["requested_labels"], "zh")

    def test_help_and_legacy_selection_stays_readable(self):
        result = subprocess.run([sys.executable, figure.__file__, "--help"], capture_output=True,
                                text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--labels {en,zh}", result.stdout)
        selection = self.pool.parent / "selection.json"
        selection.write_text(json.dumps({"pool_json": str(self.pool), "items": [
            {"pool": "transductive_success", "query_index": 17}]}), encoding="utf-8")
        _, _, cases, selected = figure.load_selection(selection)
        self.assertEqual(cases[0], self.case)
        self.assertEqual(selected[0]["query_index"], 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)

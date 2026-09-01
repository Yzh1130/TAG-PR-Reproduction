import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from datasets.external.common import path_resolution
from datasets.external.icfgpedes_ext import DATASET_NAME, load_adapter


KNOWN_DUPLICATE = "test/2868/2868_005_05_0114afternoon_0833_1_ex.jpg"


def _external_data_root(test_case):
    candidate = Path(os.environ.get("TAGPR_EXTERNAL_DATA_ROOT", "/root/autodl-tmp/datasets"))
    annotation = candidate / "ICFG-PEDES" / "ICFG-PEDES.json"
    if not annotation.is_file():
        test_case.skipTest(
            "real ICFG-PEDES annotation is unavailable; set TAGPR_EXTERNAL_DATA_ROOT "
            "or run on /root/autodl-tmp/datasets"
        )
    return candidate


class ICFGPedesAdapterMockTests(unittest.TestCase):
    def _fixture(self, root):
        records = [
            {"split": "train", "captions": ["duplicate id one"], "file_path": KNOWN_DUPLICATE, "id": 1},
            {"split": "train", "captions": ["duplicate id five"], "file_path": KNOWN_DUPLICATE, "id": 5},
            # This path is under test/, but the JSON split makes it a train sample.
            {"split": "train", "captions": ["json train"], "file_path": "test/7/train.jpg", "id": 7},
            # Conversely, this path is under train/, but remains a test sample.
            {"split": "test", "captions": ["json test zero"], "file_path": "train/20/test.jpg", "id": 20},
            {"split": "test", "captions": ["json test one"], "file_path": "test/10/test.jpg", "id": 10},
        ]
        annotation = root / "ICFG-PEDES" / "ICFG-PEDES.json"
        annotation.parent.mkdir(parents=True)
        annotation.write_text(json.dumps(records), encoding="utf-8")
        for record in records:
            image = root / "ICFG-PEDES" / "imgs" / record["file_path"]
            image.parent.mkdir(parents=True, exist_ok=True)
            image.touch(exist_ok=True)
        return records

    def test_mock_preserves_duplicate_and_uses_json_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                bundle = load_adapter(root, strict=False)

            self.assertEqual(bundle.dataset, DATASET_NAME)
            self.assertEqual(len(bundle.train_samples), 3)
            self.assertEqual(bundle.train_pid_remap, {1: 0, 5: 1, 7: 2})
            duplicate = [
                sample for sample in bundle.train_samples
                if sample.image_path.replace("\\", "/").endswith(KNOWN_DUPLICATE)
            ]
            self.assertEqual([sample.original_pid for sample in duplicate], [1, 5])
            self.assertIn("preserved_as_independent_training_samples: true", stream.getvalue())
            test = bundle.evaluations["test"]
            self.assertEqual(test.pid_remap, {10: 0, 20: 1})
            self.assertEqual(test.query_pid, [1, 0])
            self.assertEqual(test.gallery_pid, [1, 0])
            self.assertEqual(test.query_source_gallery_idx, [0, 1])
            self.assertTrue(test.gallery_paths[0].replace("\\", "/").endswith("ICFG-PEDES/imgs/train/20/test.jpg"))
            self.assertEqual(path_resolution(bundle.all_image_paths())["ratio"], 1.0)

    def test_mock_rejects_any_additional_duplicate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._fixture(root)
            records.append(
                {"split": "test", "captions": ["extra duplicate"], "file_path": "train/20/test.jpg", "id": 20}
            )
            annotation = root / "ICFG-PEDES" / "ICFG-PEDES.json"
            annotation.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected exactly"):
                load_adapter(root, strict=False)


class ICFGPedesAdapterRealAnnotationTests(unittest.TestCase):
    def test_real_counts_duplicate_and_all_paths(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            bundle = load_adapter(_external_data_root(self))
        test = bundle.evaluations["test"]

        self.assertEqual(len(bundle.train_samples), 34674)
        self.assertEqual(bundle.num_classes, 3102)
        self.assertEqual(set(bundle.train_pid_remap.values()), set(range(3102)))
        self.assertEqual((len(test.query_texts), len(test.gallery_paths)), (19848, 19848))
        duplicate = [
            sample for sample in bundle.train_samples
            if sample.image_path.replace("\\", "/").endswith(KNOWN_DUPLICATE)
        ]
        self.assertEqual(len(duplicate), 2)
        self.assertEqual(sorted(sample.original_pid for sample in duplicate), [1, 5])
        self.assertIn("preserved_as_independent_training_samples: true", stream.getvalue())
        resolution = path_resolution(bundle.all_image_paths())
        self.assertEqual(resolution["resolved"], resolution["checked"])
        self.assertEqual(resolution["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()

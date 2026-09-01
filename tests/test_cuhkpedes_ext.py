import json
import os
import tempfile
import unittest
from pathlib import Path

from datasets.external.common import path_resolution
from datasets.external.cuhkpedes_ext import DATASET_NAME, load_adapter


def _external_data_root(test_case):
    candidate = Path(os.environ.get("TAGPR_EXTERNAL_DATA_ROOT", "/root/autodl-tmp/datasets"))
    annotation = candidate / "CUHK-PEDES" / "reid_raw.json"
    if not annotation.is_file():
        test_case.skipTest(
            "real CUHK-PEDES annotation is unavailable; set TAGPR_EXTERNAL_DATA_ROOT "
            "or run on /root/autodl-tmp/datasets"
        )
    return candidate


class CUHKPedesAdapterMockTests(unittest.TestCase):
    def test_mock_expansion_protocol_mapping_and_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {"split": "train", "captions": ["t0", "t1"], "file_path": "train/a.jpg", "id": 42},
                {"split": "train", "captions": ["t2"], "file_path": "train/b.jpg", "id": 7},
                {"split": "val", "captions": ["v0", "v1"], "file_path": "val/v.jpg", "id": 100},
                {"split": "test", "captions": ["q0"], "file_path": "test/b.jpg", "id": 90},
                {"split": "test", "captions": ["q1", "q2"], "file_path": "test/a.jpg", "id": 80},
            ]
            annotation = root / "CUHK-PEDES" / "reid_raw.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text(json.dumps(records), encoding="utf-8")
            for record in records:
                image = root / "CUHK-PEDES" / "imgs" / record["file_path"]
                image.parent.mkdir(parents=True, exist_ok=True)
                image.touch()

            bundle = load_adapter(root, strict=False)

            self.assertEqual(bundle.dataset, DATASET_NAME)
            self.assertEqual(len(bundle.train_samples), 3)
            self.assertEqual(bundle.train_pid_remap, {7: 0, 42: 1})
            self.assertEqual([sample.caption for sample in bundle.train_samples], ["t0", "t1", "t2"])
            self.assertTrue(all("CUHK-PEDES" in sample.image_path for sample in bundle.train_samples))
            test = bundle.evaluations["test"]
            self.assertEqual(test.pid_remap, {80: 0, 90: 1})
            self.assertEqual(test.query_texts, ["q0", "q1", "q2"])
            self.assertEqual(test.gallery_pid, [1, 0])
            self.assertEqual(test.query_pid, [1, 0, 0])
            self.assertEqual(test.query_source_gallery_idx, [0, 1, 1])
            self.assertEqual(len(bundle.evaluations["val"].gallery_paths), 1)
            self.assertEqual(path_resolution(bundle.all_image_paths())["ratio"], 1.0)


class CUHKPedesAdapterRealAnnotationTests(unittest.TestCase):
    def test_real_counts_identities_and_all_paths(self):
        bundle = load_adapter(_external_data_root(self))
        test = bundle.evaluations["test"]
        val = bundle.evaluations["val"]

        self.assertEqual(len(bundle.train_samples), 68126)
        self.assertEqual(bundle.num_classes, 11003)
        self.assertEqual(set(bundle.train_pid_remap.values()), set(range(11003)))
        self.assertEqual((len(test.query_texts), len(test.gallery_paths)), (6156, 3074))
        self.assertEqual((len(val.query_texts), len(val.gallery_paths)), (6158, 3078))
        self.assertEqual(len(set(val.gallery_pid)), 1000)
        resolution = path_resolution(bundle.all_image_paths())
        self.assertEqual(resolution["resolved"], resolution["checked"])
        self.assertEqual(resolution["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()

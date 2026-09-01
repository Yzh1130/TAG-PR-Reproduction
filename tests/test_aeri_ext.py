import json
import os
import tempfile
import unittest
from pathlib import Path

from datasets.external.aeri_ext import DATASET_NAME, load_adapter
from datasets.external.common import path_resolution, resolve_path


def _external_data_root(test_case):
    candidate = Path(os.environ.get("TAGPR_EXTERNAL_DATA_ROOT", "/root/autodl-tmp/datasets"))
    annotation = candidate / "AERI-PEDES" / "train_caption.json"
    if not annotation.is_file():
        test_case.skipTest(
            "real AERI-PEDES annotation is unavailable; set TAGPR_EXTERNAL_DATA_ROOT "
            "or run on /root/autodl-tmp/datasets"
        )
    return candidate


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


class AERIAdapterMockTests(unittest.TestCase):
    def test_mock_w_o_ground_prefixes_unique_galleries_and_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "AERI-PEDES"
            base.mkdir(parents=True)
            train = [
                {
                    "pid": 20,
                    "split": "train",
                    "image_id": 0,
                    "aerial_img": "AERI-PEDES/train/a0.jpg",
                    "ground_img": "AERI-PEDES/train/GROUND_MUST_NOT_BE_USED.jpg",
                    "caption": "aerial person zero",
                },
                {
                    "pid": 10,
                    "split": "train",
                    "image_id": 1,
                    "aerial_img": "AERI-PEDES/train/a1.jpg",
                    "ground_img": "AERI-PEDES/train/GROUND_MUST_NOT_BE_USED_2.jpg",
                    "caption": "aerial person one",
                },
            ]
            test = [
                {
                    "pid": 90,
                    "split": "test",
                    "captions": ["query 90 a", "query 90 b"],
                    "file_paths": ["test/a90.jpg", "test/a90.jpg"],
                    "ground_file_paths": ["test/g90.jpg", "test/g90.jpg"],
                },
                {
                    "pid": 80,
                    "split": "test",
                    "captions": ["query 80"],
                    "file_paths": ["test/a80.jpg"],
                    "ground_file_paths": ["test/g80.jpg"],
                },
            ]
            (base / "train_caption.json").write_text(json.dumps(train), encoding="utf-8")
            (base / "test_caption.json").write_text(json.dumps(test), encoding="utf-8")
            for row in train:
                _touch(root / row["aerial_img"])
            for row in test:
                for relative in row["file_paths"] + row["ground_file_paths"]:
                    _touch(base / relative)

            bundle = load_adapter(root, strict=False)
            protocol = bundle.evaluations["test"]
            extra = bundle.extras["ground"]

            self.assertEqual(bundle.dataset, DATASET_NAME)
            self.assertEqual(bundle.train_pid_remap, {10: 0, 20: 1})
            self.assertEqual([sample.pid for sample in bundle.train_samples], [1, 0])
            self.assertTrue(all(sample.cam_id == 0 for sample in bundle.train_samples))
            self.assertFalse(any("GROUND_MUST_NOT_BE_USED" in path for path in bundle.all_image_paths()))
            self.assertTrue(bundle.train_samples[0].image_path.replace("\\", "/").endswith(
                "AERI-PEDES/train/a0.jpg"
            ))
            self.assertEqual(protocol.pid_remap, {80: 0, 90: 1})
            self.assertEqual(protocol.query_pid, [1, 1, 0])
            self.assertEqual(protocol.gallery_pid, [1, 0])
            self.assertEqual(protocol.query_source_gallery_idx, [-1, -1, -1])
            self.assertEqual(protocol.query_view, ["aerial"] * 3)
            self.assertEqual(protocol.gallery_view, ["aerial"] * 2)
            self.assertEqual(len(extra.paths), 2)
            self.assertEqual(
                [Path(path).name for path in extra.paths],
                ["g90.jpg", "g80.jpg"],
            )
            self.assertEqual(extra.pid, [1, 0])
            self.assertEqual(extra.view, ["ground", "ground"])
            self.assertEqual(extra.pid_remap, protocol.pid_remap)
            self.assertEqual(extra.split, "test")
            self.assertFalse(set(protocol.gallery_paths) & set(extra.paths))
            self.assertEqual(path_resolution(bundle.all_image_paths())["ratio"], 1.0)

    def test_mock_rejects_aerial_ground_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "AERI-PEDES"
            base.mkdir(parents=True)
            train = [{
                "pid": 0, "split": "train", "image_id": 0,
                "aerial_img": "AERI-PEDES/train/a.jpg",
                "ground_img": "AERI-PEDES/train/g.jpg", "caption": "caption",
            }]
            test = [{
                "pid": 1, "split": "test", "captions": ["query"],
                "file_paths": ["same.jpg"], "ground_file_paths": ["same.jpg"],
            }]
            (base / "train_caption.json").write_text(json.dumps(train), encoding="utf-8")
            (base / "test_caption.json").write_text(json.dumps(test), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sets overlap"):
                load_adapter(root, strict=False)


class AERIAdapterRealAnnotationTests(unittest.TestCase):
    def test_real_counts_identities_protocol_and_all_paths(self):
        root = _external_data_root(self)
        bundle = load_adapter(root)
        protocol = bundle.evaluations["test"]
        extra = bundle.extras["ground"]

        self.assertEqual(len(bundle.train_samples), 76007)
        self.assertEqual(bundle.num_classes, 3522)
        self.assertEqual(set(bundle.train_pid_remap.values()), set(range(3522)))
        self.assertEqual((len(protocol.query_texts), len(protocol.gallery_paths)), (6141, 5525))
        self.assertEqual(len(protocol.pid_remap), 1000)
        self.assertEqual(len(extra.paths), 4220)
        self.assertEqual(len(extra.pid), len(extra.paths))
        self.assertEqual(extra.view, ["ground"] * len(extra.paths))
        self.assertEqual(extra.pid_remap, protocol.pid_remap)
        self.assertEqual(extra.split, "test")
        self.assertFalse(set(protocol.gallery_paths) & set(extra.paths))
        test_rows = json.loads((root / "AERI-PEDES" / "test_caption.json").read_text(
            encoding="utf-8-sig"
        ))
        expected_paths = []
        expected_pid = []
        seen = set()
        for row in test_rows:
            mapped_pid = protocol.pid_remap[int(row["pid"])]
            for relative in row["ground_file_paths"]:
                path = resolve_path(root, "AERI-PEDES", relative)
                if path not in seen:
                    seen.add(path)
                    expected_paths.append(path)
                    expected_pid.append(mapped_pid)
        self.assertEqual(extra.paths, expected_paths)
        self.assertEqual(extra.pid, expected_pid)
        train_ground = json.loads((root / "AERI-PEDES" / "train_caption.json").read_text(
            encoding="utf-8-sig"
        ))
        forbidden = {resolve_path(root, "", row["ground_img"]) for row in train_ground}
        self.assertTrue(forbidden.isdisjoint(sample.image_path for sample in bundle.train_samples))
        resolution = path_resolution(bundle.all_image_paths())
        self.assertEqual(resolution["resolved"], resolution["checked"])
        self.assertEqual(resolution["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest
from pathlib import Path

from datasets.external.common import path_resolution
from datasets.external.tbapr_ext import DATASET_NAME, load_adapter


def _external_data_root(test_case):
    candidate = Path(os.environ.get("TAGPR_EXTERNAL_DATA_ROOT", "/root/autodl-tmp/datasets"))
    annotation = candidate / "TBAGPR" / "AG-ReID" / "agtext_train.json"
    if not annotation.is_file():
        test_case.skipTest(
            "real TBAPR annotations are unavailable; set TAGPR_EXTERNAL_DATA_ROOT "
            "or run on /root/autodl-tmp/datasets"
        )
    return candidate


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


class TBAPRAdapterMockTests(unittest.TestCase):
    def _fixture(self, root):
        ag = root / "TBAGPR" / "AG-ReID"
        g2 = root / "TBAGPR" / "G2APS-AG"
        ag.mkdir(parents=True)
        g2.mkdir(parents=True)
        ag_train = [
            [5, 0, ["train", "a0.jpg"], ["pair_train", "unused0.jpg"], "ag caption 0"],
            [9, 1, ["train", "a1.jpg"], ["pair_train", "unused1.jpg"], "ag caption 1"],
        ]
        g2_train = [
            [100, 0, ["train", "g0.jpg"], ["pair_train", "unused2.jpg"], "g caption 0"],
            [101, 1, ["train", "g1.jpg"], ["pair_train", "unused3.jpg"], "g caption 1"],
        ]
        ag_test = {
            "image_pids": [20, 21],
            "img_paths": [["test", "a20.jpg"], ["test", "a21.jpg"]],
            "pair_img_paths": [["pair", "pa20.jpg"], ["pair", "pa21.jpg"]],
            "caption_pids": [20, 21],
            "captions": ["query a20", "query a21"],
        }
        # Duplicate official gallery image rows must remain two rows.
        g2_test = {
            "image_pids": [30, 30],
            "img_paths": [["test", "duplicate.jpg"], ["test", "duplicate.jpg"]],
            "pair_img_paths": [["pair", "pg30a.jpg"], ["pair", "pg30b.jpg"]],
            "caption_pids": [30, 30],
            "captions": ["query g30 a", "query g30 b"],
        }
        for path, value in (
            (ag / "agtext_train.json", ag_train),
            (g2 / "g2aps_train.json", g2_train),
            (ag / "agtext_test.json", ag_test),
            (g2 / "g2aps_test.json", g2_test),
        ):
            path.write_text(json.dumps(value), encoding="utf-8")
        for base, rows in ((ag, ag_train), (g2, g2_train)):
            for _pid, _iid, image, _pair, _caption in rows:
                _touch(base.joinpath(*image))
        for base, value in ((ag, ag_test), (g2, g2_test)):
            for components in value["img_paths"] + value["pair_img_paths"]:
                _touch(base.joinpath(*components))

    def test_mock_offsets_preserve_rows_and_ignore_training_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            bundle = load_adapter(root, strict=False)
            protocol = bundle.evaluations["test"]
            extra = bundle.extras["pair"]

            self.assertEqual(bundle.dataset, DATASET_NAME)
            self.assertEqual(bundle.train_pid_remap, {5: 0, 9: 1, 102: 2, 103: 3})
            self.assertEqual([sample.original_pid for sample in bundle.train_samples], [5, 9, 102, 103])
            self.assertEqual([sample.source_subset for sample in bundle.train_samples], [
                "AG-ReID", "AG-ReID", "G2APS-AG", "G2APS-AG"
            ])
            self.assertFalse(any("pair_train" in path for path in bundle.all_image_paths()))
            self.assertIn("image_id +2", bundle.notes[0])
            self.assertEqual(len(protocol.gallery_paths), 4)
            self.assertEqual(len(set(protocol.gallery_paths)), 3)
            self.assertEqual(protocol.gallery_paths[2], protocol.gallery_paths[3])
            self.assertEqual(protocol.query_source_gallery_idx, [0, 1, 2, 3])
            self.assertEqual(protocol.pid_remap, {20: 0, 21: 1, 32: 2})
            self.assertEqual(protocol.gallery_pid, [0, 1, 2, 2])
            self.assertEqual(protocol.query_pid, [0, 1, 2, 2])
            self.assertTrue(all(
                protocol.query_pid[query_index]
                == protocol.gallery_pid[source_index]
                for query_index, source_index in enumerate(
                    protocol.query_source_gallery_idx
                )
            ))
            self.assertEqual(protocol.query_view, ["img"] * 4)
            self.assertEqual(protocol.gallery_view, ["img"] * 4)
            self.assertEqual(protocol.query_source_subset, ["AG-ReID"] * 2 + ["G2APS-AG"] * 2)
            self.assertEqual(extra.view, ["pair"] * 4)
            self.assertEqual(extra.source_subset, protocol.gallery_source_subset)
            self.assertEqual(extra.pid, protocol.gallery_pid)
            self.assertEqual(
                [Path(path).name for path in extra.paths],
                ["pa20.jpg", "pa21.jpg", "pg30a.jpg", "pg30b.jpg"],
            )
            self.assertEqual(path_resolution(bundle.all_image_paths())["ratio"], 1.0)

    def test_mock_rejects_unequal_official_test_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            path = root / "TBAGPR" / "AG-ReID" / "agtext_test.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["captions"].pop()
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unequal lengths"):
                load_adapter(root, strict=False)

    def test_mock_rejects_source_row_with_a_different_caption_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._fixture(root)
            path = root / "TBAGPR" / "G2APS-AG" / "g2aps_test.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["caption_pids"][0] = 999
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "official row alignment"):
                load_adapter(root, strict=False)


class TBAPRAdapterRealAnnotationTests(unittest.TestCase):
    def test_real_counts_offsets_rows_and_all_paths(self):
        bundle = load_adapter(_external_data_root(self))
        protocol = bundle.evaluations["test"]
        extra = bundle.extras["pair"]

        self.assertEqual(len(bundle.train_samples), 31340)
        self.assertEqual(bundle.num_classes, 1180)
        self.assertEqual(set(bundle.train_pid_remap.values()), set(range(1180)))
        self.assertEqual((len(protocol.query_texts), len(protocol.gallery_paths)), (3200, 3200))
        self.assertEqual(len(protocol.pid_remap), 529)
        self.assertEqual(protocol.query_source_gallery_idx, list(range(3200)))
        self.assertTrue(all(
            protocol.query_pid[query_index]
            == protocol.gallery_pid[source_index]
            for query_index, source_index in enumerate(
                protocol.query_source_gallery_idx
            )
        ))
        self.assertEqual(len(extra.paths), 3200)
        self.assertEqual(extra.pid, protocol.gallery_pid)
        self.assertEqual(extra.view, ["pair"] * 3200)
        self.assertEqual(extra.source_subset, protocol.gallery_source_subset)
        self.assertEqual(len(set(protocol.gallery_paths)), 3083)  # 1149 AG + 1934 G2APS unique images.
        resolution = path_resolution(bundle.all_image_paths())
        self.assertEqual(resolution["resolved"], resolution["checked"])
        self.assertEqual(resolution["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()

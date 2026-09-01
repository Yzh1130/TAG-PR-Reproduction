import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.external.dump_contract import (
    create_output_directory,
    evaluate_scores_and_write_output,
    gallery_identity_image_index,
    text_sha256_16,
    write_and_verify_similarity,
    write_core_features,
    write_meta,
    write_optional_features,
    write_pid_remap,
)
from misc.eval import compose_global_local_similarity


def _normalized(rng, rows, width):
    values = rng.normal(size=(rows, width)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class ExternalDumpContractTest(unittest.TestCase):
    def test_gallery_identity_index_uses_global_rows_and_survives_json(self):
        gallery_pid = [1, 0, 1, 2]
        expected = {1: [0, 2], 0: [1], 2: [3]}
        actual = gallery_identity_image_index(gallery_pid)
        self.assertEqual(actual, expected)
        self.assertEqual(
            sorted(index for indices in actual.values() for index in indices),
            list(range(len(gallery_pid))),
        )
        self.assertTrue(all(
            gallery_pid[index] == pid
            for pid, indices in actual.items()
            for index in indices
        ))
        with tempfile.TemporaryDirectory() as temporary:
            path = write_meta(
                temporary,
                {"gallery_identity_image_index": actual},
            )
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            persisted["gallery_identity_image_index"],
            {"1": [0, 2], "0": [1], "2": [3]},
        )

    def test_complete_200_by_100_mock_dump(self):
        rng = np.random.default_rng(2026)
        nq, ng, width = 200, 100, 16
        text_global = _normalized(rng, nq, width)
        text_local = _normalized(rng, nq, width)
        image_global = _normalized(rng, ng, width)
        image_local = _normalized(rng, ng, width)
        gallery_pid = np.arange(ng) % 20
        query_pid = np.arange(nq) % 20
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            directory.mkdir()
            write_core_features(
                directory,
                text_global,
                text_local,
                image_global,
                image_local,
            )
            sim_path, parity = write_and_verify_similarity(directory, block_size=31)
            self.assertTrue(parity["array_equal"])
            self.assertEqual(parity["compared_elements"], nq * ng)
            saved = np.load(sim_path, allow_pickle=False)
            expected = compose_global_local_similarity(
                torch.from_numpy(text_global),
                torch.from_numpy(text_local),
                torch.from_numpy(image_global),
                torch.from_numpy(image_local),
            ).numpy()
            # A single large BLAS call may use a different accumulation tiling
            # than the bounded-memory writer.  The strict bitwise check is the
            # writer's second identical blockwise pass recorded above; this
            # independent full call verifies the score formula numerically.
            self.assertTrue(np.allclose(saved, expected, rtol=0, atol=2e-7))
            metrics = evaluate_scores_and_write_output(
                sim_path,
                query_pid,
                gallery_pid,
                directory / "output.json",
                block_size=23,
            )
            self.assertEqual(set(metrics), {"r1", "r5", "r10", "mAP"})
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            output = json.loads((directory / "output.json").read_text())
            self.assertEqual(len(output), nq)
            first = output[0]["0"]
            self.assertEqual(len(first["pred"]), 50)
            self.assertEqual(first["rank"], list(range(1, 51)))

            bank = _normalized(rng, 37, width)
            extra = _normalized(rng, 11, width)
            written = write_optional_features(
                directory,
                bank_global=bank,
                bank_local=bank.copy(),
                extra_global=extra,
                extra_local=extra.copy(),
                extra_meta={
                    "n": 11,
                    "paths": [f"ground/{index}.jpg" for index in range(11)],
                    "pid": [index // 2 for index in range(11)],
                    "view": ["ground"] * 11,
                },
            )
            self.assertIn("feats_bank_text_global.npy", written)
            self.assertIn("extra_meta.json", written)
            persisted_extra = json.loads(
                (directory / "extra_meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                persisted_extra["paths"],
                [f"ground/{index}.jpg" for index in range(11)],
            )
            self.assertEqual(persisted_extra["pid"], [index // 2 for index in range(11)])
            self.assertEqual(persisted_extra["view"], ["ground"] * 11)
            pid_path = write_pid_remap(directory, {10: 0}, {20: 0})
            self.assertTrue(pid_path.is_file())
            meta_path = write_meta(directory, {"nq": nq, "ng": ng})
            self.assertEqual(json.loads(meta_path.read_text())["nq"], nq)

    def test_output_directory_is_unique_and_protected_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiments_ext"
            first = create_output_directory(
                root, "mock", "ViT-B/16", "test", 0,
                timestamp="20260901T000000000000Z",
            )
            self.assertTrue(first.is_dir())
            with self.assertRaises(FileExistsError):
                create_output_directory(
                    root, "mock", "ViT-B/16", "test", 0,
                    timestamp="20260901T000000000000Z",
                )
            with self.assertRaises(ValueError):
                create_output_directory(
                    Path(temporary) / "_dump" / "new",
                    "mock", "b16", "test", 0,
                )
            with self.assertRaises(ValueError):
                create_output_directory(
                    Path(temporary) / "_e1_something" / "new",
                    "mock", "b16", "test", 0,
                )

    def test_validation_and_no_overwrite_fail_clearly(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            feature = _normalized(rng, 4, 3)
            write_core_features(directory, feature, feature, feature, feature)
            with self.assertRaises(FileExistsError):
                write_core_features(directory, feature, feature, feature, feature)
            with self.assertRaises(ValueError):
                write_optional_features(directory, bank_global=feature)
        self.assertEqual(text_sha256_16(["a", "bc"]), text_sha256_16(["a", "bc"]))
        self.assertNotEqual(text_sha256_16(["a", "bc"]), text_sha256_16(["ab", "c"]))


if __name__ == "__main__":
    unittest.main()

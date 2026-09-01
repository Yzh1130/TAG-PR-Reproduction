"""Configuration and dispatch invariants for the external-dataset pipeline.

These tests intentionally stop before loading annotations, images, or model
weights.  Adapter cardinalities and path resolution are covered by the four
adapter-specific test modules.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

import main
import datasets
from misc.build import build_optimizer
from misc.utils import parse_config
from datasets.external.runtime import assert_external_hyperparameters
from datasets.external.runtime import _local_batch_size
from experiment import ExperimentArchive


ROOT = Path(__file__).resolve().parents[1]
CONFIG_EXPECTATIONS = {
    "cfg_cuhk.yaml": {
        "dataset_name": "cuhkpedes",
        "annotation_paths": ["CUHK-PEDES/reid_raw.json"],
        "train_samples": 68126,
        "num_classes": 11003,
    },
    "cfg_icfg.yaml": {
        "dataset_name": "icfgpedes",
        "annotation_paths": ["ICFG-PEDES/ICFG-PEDES.json"],
        "train_samples": 34674,
        "num_classes": 3102,
    },
    "cfg_aeri.yaml": {
        "dataset_name": "aeri",
        "annotation_paths": [
            "AERI-PEDES/train_caption.json",
            "AERI-PEDES/test_caption.json",
        ],
        "train_samples": 76007,
        "num_classes": 3522,
    },
    "cfg_tbapr.yaml": {
        "dataset_name": "tbapr",
        "annotation_paths": [
            "TBAGPR/AG-ReID/agtext_train.json",
            "TBAGPR/G2APS-AG/g2aps_train.json",
            "TBAGPR/AG-ReID/agtext_test.json",
            "TBAGPR/G2APS-AG/g2aps_test.json",
        ],
        "train_samples": 31340,
        "num_classes": 1180,
    },
}
DATASET_SPECIFIC_EXTERNAL_KEYS = {
    "annotation_paths",
    "train_samples",
    "num_classes",
}


def _load_yaml(name):
    with (ROOT / name).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _without_dataset_specific_fields(config):
    value = copy.deepcopy(config)
    value.pop("dataset_name")
    for key in DATASET_SPECIFIC_EXTERNAL_KEYS:
        value["external"].pop(key)
    return value


class ExternalConfigTests(unittest.TestCase):
    def test_repository_local_datasets_package_wins_name_resolution(self):
        package_path = Path(datasets.__file__).resolve()
        self.assertEqual(package_path, ROOT / "datasets" / "__init__.py")

    def test_dataset_specific_fields_match_the_contract(self):
        for filename, expected in CONFIG_EXPECTATIONS.items():
            with self.subTest(config=filename):
                config = _load_yaml(filename)
                self.assertEqual(config["dataset_name"], expected["dataset_name"])
                for key in DATASET_SPECIFIC_EXTERNAL_KEYS:
                    self.assertEqual(config["external"][key], expected[key])

    def test_all_non_dataset_hyperparameters_are_identical(self):
        normalized = {
            filename: _without_dataset_specific_fields(_load_yaml(filename))
            for filename in CONFIG_EXPECTATIONS
        }
        reference_name, reference = next(iter(normalized.items()))
        for filename, candidate in normalized.items():
            with self.subTest(reference=reference_name, config=filename):
                self.assertEqual(candidate, reference)

    def test_shared_protocol_hyperparameters(self):
        for filename in CONFIG_EXPECTATIONS:
            with self.subTest(config=filename):
                config = _load_yaml(filename)
                self.assertEqual(config["model"]["backbone"], "ViT-B/16")
                self.assertEqual(config["model"]["checkpoint"], "${CLIP_B16_CHECKPOINT}")
                self.assertEqual(config["model"]["saved_path"], "experiments_ext")
                self.assertTrue(config["experiment"]["disable_moe"])
                self.assertEqual(config["experiment"]["view_ratio"], 1.0)
                self.assertEqual(config["experiment"]["ortho_ratio"], 1.0)
                self.assertTrue(config["experiment"]["grad_checkpoint"])
                self.assertEqual(config["experiment"]["nitc_ratio"], 1.0)
                self.assertEqual(config["experiment"]["ritc_ratio"], 1.0)
                self.assertEqual(config["experiment"]["id_ratio"], 0.5)
                self.assertFalse(config["experiment"]["rerank"])
                self.assertFalse(config["experiment"]["csls"])
                self.assertEqual(config["experiment"]["max_words"], 77)
                self.assertEqual(config["experiment"]["text_length"], 77)
                self.assertEqual(config["experiment"]["input_resolution"], [224, 224])
                self.assertEqual(config["data"]["global_batch_size"], 320)
                self.assertEqual(config["data"]["batch_size"], 320)
                self.assertEqual(config["data"]["num_workers"], 2)
                self.assertEqual(config["schedule"]["epoch"], 10)
                self.assertEqual(config["schedule"]["lr"], 1.0e-4)
                self.assertEqual(config["schedule"]["weight_decay"], 0.02)
                self.assertEqual(config["schedule"]["betas"], [0.9, 0.98])
                self.assertEqual(config["external"]["data_root"], "/root/autodl-tmp/datasets")
                self.assertEqual(config["external"]["output_root"], "experiments_ext")
                self.assertEqual(config["external"]["eval_split"], "test")
                self.assertTrue(config["external"]["extract_bank_text"])
                self.assertTrue(config["external"]["extract_extra_images"])

    def test_external_runtime_rejects_locked_protocol_overrides(self):
        config = parse_config(str(ROOT / "cfg_cuhk.yaml"))
        self.assertTrue(assert_external_hyperparameters(config))
        incompatible_values = {
            "rerank": True,
            "csls": True,
            "grad_checkpoint": False,
            "view_ratio": 0.0,
            "ortho_ratio": 0.0,
            "batch_size": 256,
        }
        for field, incompatible in incompatible_values.items():
            with self.subTest(field=field):
                candidate = copy.deepcopy(config)
                section = candidate.data if field == "batch_size" else candidate.experiment
                section[field] = incompatible
                with self.assertRaisesRegex(ValueError, field):
                    assert_external_hyperparameters(candidate)

    def test_config_used_records_locked_global_and_effective_local_batch(self):
        config = parse_config(str(ROOT / "cfg_cuhk.yaml"))
        self.assertTrue(assert_external_hyperparameters(config))
        with mock.patch(
            "datasets.external.runtime._world_size",
            return_value=2,
        ):
            self.assertEqual(_local_batch_size(config), 160)
        with tempfile.TemporaryDirectory() as temporary:
            ExperimentArchive(temporary, enabled=True).write_pre_run(
                config,
                str(ROOT / "cfg_cuhk.yaml"),
            )
            archived = yaml.safe_load(
                (Path(temporary) / "config_used.yaml").read_text(encoding="utf-8")
            )
        self.assertEqual(archived["data"]["global_batch_size"], 320)
        self.assertEqual(archived["data"]["batch_size"], 160)
        self.assertTrue(archived["experiment"]["grad_checkpoint"])
        self.assertEqual(archived["experiment"]["view_ratio"], 1.0)
        self.assertEqual(archived["experiment"]["ortho_ratio"], 1.0)

    def test_existing_optimizer_builder_is_adamw(self):
        config = parse_config(str(ROOT / "cfg_cuhk.yaml"))
        optimizer = build_optimizer(config, torch.nn.Linear(2, 2))
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 1.0e-4)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.02)
        self.assertEqual(tuple(optimizer.defaults["betas"]), (0.9, 0.98))

    def test_smoke_parser_default_and_explicit_step_count(self):
        parser = main.build_parser()
        self.assertIsNone(parser.parse_args([]).smoke)
        self.assertEqual(parser.parse_args(["--smoke"]).smoke, 200)
        self.assertEqual(parser.parse_args(["--smoke", "17"]).smoke, 17)

    def test_default_tag_config_does_not_dispatch_external_runtime(self):
        class ReachedTagPath(Exception):
            pass

        fake_runtime = types.ModuleType("datasets.external.runtime")
        fake_runtime.run_external = mock.Mock(
            side_effect=AssertionError("default TAG-PEDES run entered external runtime")
        )
        with mock.patch.dict(
            sys.modules,
            {"datasets.external.runtime": fake_runtime},
        ), mock.patch.object(
            main,
            "configure_experiment",
            side_effect=ReachedTagPath,
        ):
            with self.assertRaises(ReachedTagPath):
                main.main([])
        fake_runtime.run_external.assert_not_called()

    def test_external_dataset_name_is_the_only_dispatch_trigger(self):
        marker = object()
        fake_runtime = types.ModuleType("datasets.external.runtime")
        fake_runtime.run_external = mock.Mock(return_value=marker)
        with mock.patch.dict(
            sys.modules,
            {"datasets.external.runtime": fake_runtime},
        ):
            result = main.main(
                [
                    "--config",
                    str(ROOT / "cfg_cuhk.yaml"),
                    "--smoke",
                    "1",
                ]
            )
        self.assertIs(result, marker)
        fake_runtime.run_external.assert_called_once()
        call = fake_runtime.run_external.call_args
        self.assertEqual(call.args[0].dataset_name, "cuhkpedes")
        self.assertEqual(call.kwargs["smoke"], 1)


if __name__ == "__main__":
    unittest.main()

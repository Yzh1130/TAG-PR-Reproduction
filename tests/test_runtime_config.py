import io
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import main
from experiment import ExperimentArchive, METRICS_HEADER
from misc.utils import parse_config


class RuntimeConfigTests(unittest.TestCase):
    def test_overrides_mutate_the_single_config_object(self):
        args = main.build_parser().parse_args(
            [
                "--batch_size",
                "16",
                "--test_batch_size",
                "24",
                "--epoch",
                "2",
            ]
        )
        config = parse_config(args.config)
        original_id = id(config)

        overrides = main.apply_cli_overrides(config, args)

        self.assertEqual(id(config), original_id)
        self.assertEqual(config.data.batch_size, 16)
        self.assertEqual(config.data.test_batch_size, 24)
        self.assertEqual(config.schedule.epoch, 2)
        self.assertEqual(
            [item[0] for item in overrides],
            ["data.batch_size", "data.test_batch_size", "schedule.epoch"],
        )

    def test_training_loader_mismatch_is_fatal(self):
        config = parse_config("config/config.yaml")
        config.data.batch_size = 16
        loader = SimpleNamespace(batch_size=80)

        with self.assertRaisesRegex(
            RuntimeError,
            r"config\.data\.batch_size=16, train_loader\.batch_size=80",
        ):
            main.assert_training_batch_size(config, loader)

    def test_override_summary_has_explicit_delimiters(self):
        stream = io.StringIO()
        logger = logging.getLogger("override-summary-test")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))

        main.log_cli_overrides(logger, [("data.batch_size", 80, 16)])

        self.assertEqual(
            stream.getvalue().splitlines(),
            [
                "OVERRIDE SUMMARY",
                "data.batch_size: 80 -> 16",
                "END OVERRIDE SUMMARY",
            ],
        )

    def test_test_batch_override_rebuilds_only_when_explicit(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(9))
        original = torch.utils.data.DataLoader(dataset, batch_size=3)
        dataloader = {"test_loader": original}
        config = parse_config("config/config.yaml")
        config.data.test_batch_size = 2

        main.apply_test_loader_override(config, dataloader, ())
        self.assertIs(dataloader["test_loader"], original)

        overrides = (("data.test_batch_size", 256, 2),)
        main.apply_test_loader_override(config, dataloader, overrides)
        self.assertEqual(dataloader["test_loader"].batch_size, 2)

    def test_metrics_header_contains_peak_memory_columns(self):
        self.assertEqual(
            METRICS_HEADER[-2:],
            ["peak_mem_alloc_gb", "peak_mem_reserved_gb"],
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = ExperimentArchive(Path(directory), enabled=True)
            archive.initialize_metrics()
            header = (Path(directory) / "metrics_by_epoch.csv").read_text(
                encoding="utf-8"
            )
        self.assertIn("peak_mem_alloc_gb,peak_mem_reserved_gb", header)


if __name__ == "__main__":
    unittest.main()

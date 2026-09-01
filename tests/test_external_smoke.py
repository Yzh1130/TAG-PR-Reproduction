import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from datasets.external.runtime import (
    ExternalTrainDataset,
    assert_smoke_payload_blinded,
    count_token_overflow,
    print_external_evaluation,
    run_smoke_protocol,
    save_external_checkpoint,
)
from datasets.external.common import TrainSample


class _TrainDataset(Dataset):
    def __init__(self):
        self.images = torch.arange(20 * 6, dtype=torch.float32).reshape(20, 6) / 100

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "caption": f"caption {index}",
            "cam_id": 1,
            "id": index % 5,
        }


class _VisualLayer(nn.Module):
    def forward(self, attention, dense, training=False):
        self.last_training = training
        return dense.mean(dim=1)


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(6, 4)
        self.visual_emb_layer = _VisualLayer()
        self.config = SimpleNamespace(
            experiment=SimpleNamespace(nrw_mode="off")
        )

    def forward(self, batch, alpha, training=False, epoch=0):
        del alpha, epoch
        values = self.projection(batch["image"])
        return {"mock_loss": values.square().mean()}

    def encode_image(self, images, return_dense=False):
        self.assert_return_dense = return_dense
        feature = self.projection(images)
        dense = feature.unsqueeze(1).repeat(1, 2, 1)
        attention = torch.zeros(images.shape[0], 2, 2)
        return feature, dense, attention


class _Tokenizer:
    def encode(self, text):
        return list(range(int(text)))


class _WordTokenizer:
    def encode(self, text):
        return text.split()


class ExternalSmokeTest(unittest.TestCase):
    def test_engineering_smoke_is_result_blinded(self):
        train_loader = DataLoader(_TrainDataset(), batch_size=4, shuffle=False)
        images = torch.randn(16, 6)
        eval_loader = DataLoader(
            TensorDataset(images, torch.arange(16) % 5, torch.arange(16)),
            batch_size=2,
            shuffle=False,
        )
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        report = {
            "checks": {
                "train_samples": True,
                "num_classes": True,
                "split_available": True,
                "annotation_paths": True,
            },
            "train_samples": 20,
            "num_classes": 5,
            "path_resolution": {"checked": 20, "resolved": 20, "ratio": 1.0},
        }
        with tempfile.TemporaryDirectory() as temporary:
            payload = run_smoke_protocol(
                model,
                train_loader,
                eval_loader,
                optimizer,
                [0.01] * 3,
                torch.device("cpu"),
                steps=3,
                adapter_report=report,
                overflow_count=2,
                text_count=20,
                output_directory=temporary,
            )
            self.assertTrue(assert_smoke_payload_blinded(payload))
            saved = json.loads((Path(temporary) / "smoke_engineering.json").read_text())
            self.assertEqual(saved, payload)
            self.assertEqual(saved["counts"]["completed_steps"], 3)
            self.assertGreaterEqual(saved["counts"]["evaluation_batches_checked"], 1)

    def test_smoke_checkpoint_matches_last_checkpoint_schema_and_loads(self):
        model = _MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        config = SimpleNamespace(name="mock")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint_smoke.pth"
            save_external_checkpoint(path, model, optimizer, config, epoch=3)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                set(checkpoint),
                {"model", "optimizer", "config", "epoch"},
            )
            self.assertEqual(checkpoint["epoch"], 3)
            restored = _MockModel()
            restored.load_state_dict(checkpoint["model"], strict=True)
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
            for expected, actual in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(expected, actual))
            with self.assertRaises(FileExistsError):
                save_external_checkpoint(path, model, optimizer, config, epoch=3)

    def test_quiet_metrics_prints_no_metric_values(self):
        metrics = {"r1": 11.25, "r5": 22.5, "r10": 33.75, "mAP": 44.0}
        output = io.StringIO()
        with redirect_stdout(output):
            print_external_evaluation(
                metrics,
                {"array_equal": True},
                quiet_metrics=True,
            )
        self.assertEqual(
            output.getvalue().splitlines(),
            ["EVALUATION COMPLETE", "array_equal=True"],
        )
        self.assertFalse(any(str(value) in output.getvalue() for value in metrics.values()))

        default_output = io.StringIO()
        with redirect_stdout(default_output):
            print_external_evaluation(
                metrics,
                {"array_equal": True},
                quiet_metrics=False,
            )
        self.assertIn("R@1=11.25000", default_output.getvalue())
        self.assertIn("mAP=44.00000", default_output.getvalue())

    def test_recursive_blinding_contract_rejects_metrics_and_nonintegers(self):
        with self.assertRaises(ValueError):
            assert_smoke_payload_blinded({"metrics": {"anything": 1}})
        with self.assertRaises(ValueError):
            assert_smoke_payload_blinded({"top_1": 1})
        with self.assertRaises(TypeError):
            assert_smoke_payload_blinded({"counts": {"elapsed": 0.1}})
        self.assertTrue(assert_smoke_payload_blinded({"ok": True, "count": 1}))

    def test_context_overflow_count(self):
        overflow, total = count_token_overflow(
            ["2", "5", "10"],
            context_length=7,
            tokenizer=_Tokenizer(),
        )
        self.assertEqual(total, 3)
        self.assertEqual(overflow, 1)

    def test_context_count_uses_caption_dataset_word_truncation_first(self):
        sample = TrainSample(
            image_path="unused.jpg",
            caption="one two three four five six seven eight nine ten",
            pid=0,
            original_pid=0,
        )
        dataset = ExternalTrainDataset([sample], transform=None, max_words=3)
        self.assertEqual(dataset.captions, ["one two three"])
        overflow, total = count_token_overflow(
            dataset.captions,
            context_length=7,
            tokenizer=_WordTokenizer(),
        )
        self.assertEqual((overflow, total), (0, 1))


if __name__ == "__main__":
    unittest.main()

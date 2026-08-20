"""Exercise main.py's non-dry-run training loop with lightweight test doubles.

This script is intentionally launched through torchrun.  It preserves the real
CLI/config/archive/DataLoader/assertion/optimizer/backward/logging path while
replacing the unavailable CUHK-PEDES files and large TAG-PR model.
"""

import argparse
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import main as training_main


DATASET_SIZE = 19954


class SyntheticTrainingDataset(torch.utils.data.Dataset):
    person2text = {0: ["synthetic caption"]}

    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, index):
        return {
            "image": torch.tensor([float(index % 7)]),
            "caption": "synthetic caption",
            "cam_id": torch.tensor(1),
            "id": torch.tensor(0),
        }


class InterruptingModel(torch.nn.Module):
    def __init__(self, stop_after, simulate_oom):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.stop_after = stop_after
        self.simulate_oom = simulate_oom
        self.iterations = 0

    def to(self, *args, **kwargs):
        return self

    def forward(self, batch, alpha, training=False):
        self.iterations += 1
        if self.simulate_oom:
            raise torch.cuda.OutOfMemoryError("synthetic OOM for exit-path test")
        if self.iterations > self.stop_after:
            raise KeyboardInterrupt
        target = batch["image"].float().mean()
        return {"loss": (self.weight - target).square()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--stop_after", type=int, default=50)
    parser.add_argument("--force_mismatch", action="store_true")
    parser.add_argument("--simulate_oom", action="store_true")
    parser.add_argument("--scenario", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    work_root = Path(tempfile.gettempdir()) / "tagpr-real-training-smoke"
    workdir = work_root / args.scenario
    if workdir.exists():
        shutil.rmtree(workdir)
    (workdir / "config").mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "config" / "config.yaml", workdir / "config" / "config.yaml")
    os.chdir(workdir)

    def build_data(config):
        batch_size = config.data.batch_size + (1 if args.force_mismatch else 0)
        train_loader = torch.utils.data.DataLoader(
            SyntheticTrainingDataset(),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.arange(8)),
            batch_size=config.data.test_batch_size,
        )
        return {
            "train_loader": train_loader,
            "train_sampler": None,
            "test_loader": test_loader,
        }

    fake_model_module = types.ModuleType("model.tbps_model_MoE")
    fake_model_module.clip_vitb = lambda config, num_classes: InterruptingModel(
        args.stop_after,
        args.simulate_oom,
    )
    sys.modules["model.tbps_model_MoE"] = fake_model_module

    training_main.build_pedes_data = build_data
    training_main.init_distributed_mode = lambda config: None
    training_main.set_seed = lambda config: None
    training_main.is_using_distributed = lambda: False
    training_main.is_master = lambda: True
    training_main.get_rank = lambda: 0
    training_main.load_checkpoint = lambda model, config: (model, None)
    training_main.test_tse = lambda model, loader, text_length, device: {
        "r1": 0.0,
        "r5": 0.0,
        "r10": 0.0,
        "mAP": 0.0,
    }
    training_main.cosine_scheduler = lambda config: [
        config.schedule.lr
    ] * (config.schedule.epoch * config.schedule.niter_per_ep)

    def build_optimizer(config, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=config.schedule.lr)
        for group in optimizer.param_groups:
            group["ratio"] = 1.0
        return optimizer

    training_main.build_optimizer = build_optimizer

    main_argv = []
    if args.batch_size is not None:
        main_argv.extend(["--batch_size", str(args.batch_size)])
    if args.test_batch_size is not None:
        main_argv.extend(["--test_batch_size", str(args.test_batch_size)])
    if args.epoch is not None:
        main_argv.extend(["--epoch", str(args.epoch)])

    try:
        training_main.main(main_argv)
    except KeyboardInterrupt:
        print(f"SMOKE INTERRUPTED after {args.stop_after} completed iterations")
        print(f"SMOKE ARTIFACT DIR: {workdir / 'ckpts'}")
        return 0
    except RuntimeError as exc:
        if args.force_mismatch and "DataLoader batch_size mismatch" in str(exc):
            print(f"EXPECTED ASSERTION: {exc}")
            return 0
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

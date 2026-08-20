import csv
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml


METRICS_HEADER = [
    "epoch",
    "R1",
    "R5",
    "R10",
    "mAP",
    "train_loss",
    "epoch_time_sec",
    "peak_mem_alloc_gb",
    "peak_mem_reserved_gb",
]

RECORDED_ENVIRONMENT_VARIABLES = [
    "CUDA_VISIBLE_DEVICES",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
]


class ExperimentDirectoryConflict(RuntimeError):
    """Raised when a managed experiment would overwrite its best checkpoint."""


def resolve_experiment_directory(saved_path, output_dir=None, exp_name=None):
    root = Path(output_dir) if output_dir is not None else Path(saved_path)
    if exp_name is None:
        return root
    return root / f"{datetime.now():%Y%m%d}_{exp_name}"


def prepare_experiment_directory(directory, protect_checkpoint=False):
    directory = Path(directory)
    checkpoint = directory / "checkpoint_best.pth"
    if protect_checkpoint and checkpoint.exists():
        raise ExperimentDirectoryConflict(
            f"Experiment directory '{directory}' already contains "
            "checkpoint_best.pth; choose a different --exp_name or --output_dir."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def add_file_handler(logger, directory):
    """Add the experiment train.log handler without changing existing handlers."""
    log_path = (Path(directory) / "train.log").resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_path:
                return

    formatter = next(
        (handler.formatter for handler in logger.handlers if handler.formatter),
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"),
    )
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def _to_plain_data(value):
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


def _run_git_command(arguments):
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}"

    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        return f"unavailable: {detail}"
    return result.stdout.rstrip()


def _git_summary():
    commit = _run_git_command(["rev-parse", "HEAD"])
    status = _run_git_command(["status", "--porcelain"])
    status_unavailable = status.startswith("unavailable:")
    state = "UNKNOWN" if status_unavailable else ("DIRTY" if status else "CLEAN")
    status_text = status if status else "(clean)"
    return (
        f"git rev-parse HEAD:\n{commit}\n\n"
        f"WORKTREE: {state}\n"
        f"git status --porcelain:\n{status_text}\n"
    )


def _run_command_summary(config_path):
    full_argv = [sys.executable, *sys.argv]
    if os.name == "nt":
        command = subprocess.list2cmdline(full_argv)
    else:
        command = shlex.join(full_argv)

    lines = [
        f"command: {command}",
        f"cwd: {os.getcwd()}",
        f"config_source: {config_path}",
        f"sys.executable: {sys.executable}",
        f"sys.argv: {json.dumps(sys.argv, ensure_ascii=False)}",
        "environment:",
    ]
    lines.extend(
        f"{name}={os.environ.get(name, '<unset>')}"
        for name in RECORDED_ENVIRONMENT_VARIABLES
    )
    return "\n".join(lines) + "\n"


def _torchvision_version():
    try:
        import torchvision

        return torchvision.__version__
    except Exception as exc:  # torchvision can fail while loading compiled extensions
        return f"unavailable ({type(exc).__name__}: {exc})"


def _world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return os.environ.get("WORLD_SIZE", "unknown")


def _environment_summary():
    cuda_available = torch.cuda.is_available()
    try:
        gpu_count = torch.cuda.device_count() if cuda_available else 0
    except Exception:
        gpu_count = "unavailable"

    lines = [
        f"python: {platform.python_version()}",
        f"python_full: {sys.version.replace(os.linesep, ' ')}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"torchvision: {_torchvision_version()}",
        f"cuda_runtime: {torch.version.cuda or 'not available'}",
        f"cuda_available: {cuda_available}",
        f"gpu_count: {gpu_count}",
        f"world_size: {_world_size()}",
    ]

    if isinstance(gpu_count, int):
        for index in range(gpu_count):
            try:
                name = torch.cuda.get_device_name(index)
            except Exception as exc:
                name = f"unavailable ({type(exc).__name__}: {exc})"
            lines.append(f"gpu_{index}: {name}")
    return "\n".join(lines) + "\n"


def _write_text(path, content):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


class ExperimentArchive:
    def __init__(self, directory, enabled):
        self.directory = Path(directory)
        self.enabled = enabled

    def write_pre_run(self, config, config_path):
        if not self.enabled:
            return

        # Capture Git state before creating untracked archive files.
        git_summary = _git_summary()
        command_summary = _run_command_summary(config_path)
        environment_summary = _environment_summary()
        config_yaml = yaml.safe_dump(
            _to_plain_data(config),
            sort_keys=False,
            allow_unicode=True,
        )

        _write_text(self.directory / "config_used.yaml", config_yaml)
        _write_text(self.directory / "git_commit.txt", git_summary)
        _write_text(self.directory / "run_command.txt", command_summary)
        _write_text(
            self.directory / "environment_summary.txt", environment_summary
        )

    def initialize_metrics(self):
        if not self.enabled:
            return
        metrics_path = self.directory / "metrics_by_epoch.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(METRICS_HEADER)
            handle.flush()
            os.fsync(handle.fileno())

    def append_metrics(
        self,
        epoch,
        rank_1,
        rank_5,
        rank_10,
        mean_average_precision,
        train_loss,
        epoch_time_sec,
        peak_mem_alloc_gb,
        peak_mem_reserved_gb,
    ):
        if not self.enabled:
            return
        metrics_path = self.directory / "metrics_by_epoch.csv"
        with metrics_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    rank_1,
                    rank_5,
                    rank_10,
                    mean_average_precision,
                    train_loss,
                    epoch_time_sec,
                    peak_mem_alloc_gb,
                    peak_mem_reserved_gb,
                ]
            )
            handle.flush()
            os.fsync(handle.fileno())

    def write_final_results(
        self,
        best_epoch,
        best_metrics,
        total_time_sec,
        peak_mem_alloc_gb=0.0,
        peak_mem_reserved_gb=0.0,
        completed_epochs=0,
        run_status="completed",
    ):
        if not self.enabled:
            return

        if best_metrics is None:
            result_lines = [
                "best_epoch: not available",
                "R@1: not available",
                "R@5: not available",
                "R@10: not available",
                "mAP: not available",
            ]
        else:
            result_lines = [
                f"best_epoch: {best_epoch}",
                f"R@1: {best_metrics['r1']}",
                f"R@5: {best_metrics['r5']}",
                f"R@10: {best_metrics['r10']}",
                f"mAP: {best_metrics['mAP']}",
            ]
        result_lines.append(f"total_training_time_sec: {total_time_sec}")
        result_lines.extend(
            [
                f"completed_epochs: {completed_epochs}",
                f"peak_mem_alloc_gb: {peak_mem_alloc_gb}",
                f"peak_mem_reserved_gb: {peak_mem_reserved_gb}",
                f"run_status: {run_status}",
            ]
        )
        _write_text(
            self.directory / "final_results.txt",
            "\n".join(result_lines) + "\n",
        )

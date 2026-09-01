"""Independent training/evaluation runtime for external retrieval datasets.

Nothing in this module is imported by the TAG-PEDES data path.  ``main.py``
dispatches here only when an explicit external ``dataset_name`` is configured.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datasets.external.common import AdapterBundle, path_resolution
from datasets.external.dump_contract import (
    annotation_provenance,
    create_output_directory,
    evaluate_scores_and_write_output,
    gallery_identity_image_index,
    git_head,
    text_sha256_16,
    write_and_verify_similarity,
    write_core_features,
    write_meta,
    write_optional_features,
    write_pid_remap,
)


_ADAPTER_MODULES = {
    "cuhkpedes": "datasets.external.cuhkpedes_ext",
    "cuhk-pedes": "datasets.external.cuhkpedes_ext",
    "icfgpedes": "datasets.external.icfgpedes_ext",
    "icfg-pedes": "datasets.external.icfgpedes_ext",
    "aeri": "datasets.external.aeri_ext",
    "aeri-pedes": "datasets.external.aeri_ext",
    "tbapr": "datasets.external.tbapr_ext",
    "tbagpr": "datasets.external.tbapr_ext",
}


def _canonical_name(value):
    return str(value).strip().lower().replace("_", "-")


def is_external_dataset(config):
    return _canonical_name(getattr(config, "dataset_name", "")) in _ADAPTER_MODULES


def load_external_adapter(dataset_name, data_root):
    name = _canonical_name(dataset_name)
    try:
        module_name = _ADAPTER_MODULES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported external dataset_name={dataset_name!r}; "
            f"choose one of {sorted(_ADAPTER_MODULES)}"
        ) from exc
    module = importlib.import_module(module_name)
    if not hasattr(module, "load_adapter"):
        raise RuntimeError(f"{module_name} does not export load_adapter(data_root)")
    bundle = module.load_adapter(data_root)
    if not isinstance(bundle, AdapterBundle):
        raise TypeError(f"{module_name}.load_adapter returned {type(bundle).__name__}")
    return bundle.validate()


def _dist_active():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank():
    return torch.distributed.get_rank() if _dist_active() else 0


def _world_size():
    return torch.distributed.get_world_size() if _dist_active() else 1


def _master():
    return _rank() == 0


def _initialize_runtime(config):
    names = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    supplied = [name in os.environ for name in names]
    if any(supplied) and not all(supplied):
        raise RuntimeError(
            "distributed external runs require RANK, WORLD_SIZE, and LOCAL_RANK together"
        )
    if all(supplied) and not _dist_active():
        torch.distributed.init_process_group(
            backend=config.distributed.backend,
            init_method=config.distributed.url,
        )
    if torch.cuda.is_available():
        index = int(os.environ["LOCAL_RANK"]) if all(supplied) else int(config.device)
        torch.cuda.set_device(index)
        config.device = f"cuda:{index}"
    else:
        config.device = "cpu"
    seed = int(config.misc.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # model.all_gather consults a historical constant which always reports
    # distributed mode.  Disable gathering for an ordinary one-process run.
    if not _dist_active():
        config.model.use_gather = False
    return torch.device(config.device)


def _broadcast_path(path):
    if not _dist_active():
        return Path(path)
    objects = [str(path) if _master() else None]
    torch.distributed.broadcast_object_list(objects, src=0)
    return Path(objects[0])


def _resolve_original_checkpoint(config):
    raw = str(config.model.checkpoint)
    expanded = os.path.expandvars(raw)
    if remainders := [token for token in ("$", "%") if token in expanded]:
        raise RuntimeError(
            f"unresolved environment variable in external model.checkpoint={raw!r}; "
            "set CLIP_B16_CHECKPOINT to the ViT-B/16 .pt file"
        )
    path = Path(expanded).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"external CLIP checkpoint does not exist: {path}")
    config.model.checkpoint = str(path)
    return path


def _make_transforms(config):
    # Import torchvision only for actual image pipelines, keeping annotation
    # inspection and CPU contract tests independent of compiled image ops.
    from torchvision import transforms
    from misc.data import Choose

    size = config.experiment.input_resolution
    if isinstance(size, int):
        size = (size, size)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = Choose([
        transforms.ColorJitter(.1, .1, .1, 0),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size, (0.9, 1.0), antialias=True),
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomErasing(scale=(0.10, 0.20)),
    ], size)
    eval_transform = transforms.Compose([
        transforms.Resize(
            size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


class ExternalTrainDataset(Dataset):
    def __init__(self, samples, transform, max_words):
        from misc.caption_dataset import pre_caption

        self.samples = list(samples)
        self.transform = transform
        self.captions = [
            pre_caption(sample.caption, max_words=max_words)
            for sample in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.image_path) as opened:
            image = self.transform(opened.convert("RGB"))
        return {
            "image": image,
            "caption": self.captions[index],
            "cam_id": int(sample.cam_id),
            "id": int(sample.pid),
        }


class ExternalImageDataset(Dataset):
    def __init__(self, paths, pids, transform):
        self.paths = list(paths)
        self.pids = [int(value) for value in pids]
        if len(self.paths) != len(self.pids):
            raise ValueError("external image path/pid row counts differ")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as opened:
            image = self.transform(opened.convert("RGB"))
        return image, self.pids[index], index


def _local_batch_size(config):
    global_batch = int(config.data.global_batch_size)
    world = _world_size()
    if global_batch < 1 or global_batch % world:
        raise ValueError(
            f"global_batch_size={global_batch} is not divisible by world_size={world}"
        )
    local = global_batch // world
    configured = int(config.data.batch_size)
    if configured not in (global_batch, local):
        raise ValueError(
            f"data.batch_size={configured} disagrees with global/local batch "
            f"{global_batch}/{local}"
        )
    config.data.batch_size = local
    return local


def assert_external_hyperparameters(config):
    resolution = list(config.experiment.input_resolution) if not isinstance(
        config.experiment.input_resolution, int
    ) else [config.experiment.input_resolution] * 2
    values = {
        "backbone": (str(config.model.backbone), "ViT-B/16"),
        "disable_moe": (bool(config.experiment.disable_moe), True),
        # Keep the paper's ViT-B/16-without-HR-MoE configuration verbatim.
        # Transformer_MoE removes the view token and constructs no MoE blocks,
        # so these retained ratios have no numerical or gradient effect.
        "view_ratio": (float(config.experiment.view_ratio), 1.0),
        "ortho_ratio": (float(config.experiment.ortho_ratio), 1.0),
        "grad_checkpoint": (bool(config.experiment.grad_checkpoint), True),
        "global_batch_size": (int(config.data.global_batch_size), 320),
        "batch_size": (int(config.data.batch_size), 320),
        "epochs": (int(config.schedule.epoch), 10),
        "lr": (float(config.schedule.lr), 1e-4),
        "weight_decay": (float(config.schedule.weight_decay), 0.02),
        "max_words": (int(config.experiment.max_words), 77),
        "input_resolution": (resolution, [224, 224]),
        "num_workers": (int(config.data.num_workers), 2),
        # External feature dumps are the unprocessed retrieval scores.  Reject
        # CLI/config attempts to enable either online post-processor here.
        "rerank": (bool(config.experiment.rerank), False),
        "csls": (bool(config.experiment.csls), False),
    }
    mismatches = {
        key: {"actual": actual, "required": required}
        for key, (actual, required) in values.items()
        if actual != required
    }
    if mismatches:
        raise ValueError(
            "external protocol hyperparameters differ from the locked common "
            f"configuration: {mismatches}"
        )
    return True


def build_external_loaders(bundle, protocol, config):
    train_transform, eval_transform = _make_transforms(config)
    train_dataset = ExternalTrainDataset(
        bundle.train_samples,
        train_transform,
        int(config.experiment.max_words),
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if _dist_active()
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=_local_batch_size(config),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(config.data.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    evaluation_dataset = ExternalImageDataset(
        protocol.gallery_paths,
        protocol.gallery_pid,
        eval_transform,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=int(config.data.test_batch_size),
        shuffle=False,
        num_workers=int(config.data.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return {
        "train_dataset": train_dataset,
        "train_loader": train_loader,
        "train_sampler": train_sampler,
        "eval_transform": eval_transform,
        "evaluation_loader": evaluation_loader,
    }


def assert_adapter_contract(bundle, config, split, *, path_sample=2000):
    expected_samples = int(config.external.train_samples)
    expected_classes = int(config.external.num_classes)
    checks = {
        "train_samples": len(bundle.train_samples) == expected_samples,
        "num_classes": bundle.num_classes == expected_classes,
        "split_available": split in bundle.evaluations,
    }
    configured = [
        str((Path(config.external.data_root) / relative).resolve())
        for relative in config.external.annotation_paths
    ]
    actual = [str(Path(path).resolve()) for path in bundle.annotation_paths]
    checks["annotation_paths"] = configured == actual
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"external adapter/config contract failed {failed}: "
            f"samples={len(bundle.train_samples)}/{expected_samples}, "
            f"classes={bundle.num_classes}/{expected_classes}, "
            f"annotations={actual}/{configured}, splits={sorted(bundle.evaluations)}"
        )
    resolution = path_resolution(bundle.all_image_paths(), limit=path_sample)
    if resolution["checked"] != min(path_sample, len(bundle.all_image_paths())):
        raise RuntimeError("stable external path sample checked an unexpected count")
    if resolution["ratio"] != 1.0:
        raise FileNotFoundError(
            f"external path sample resolution is {resolution['ratio']:.6f}: {resolution}"
        )
    return {
        "checks": checks,
        "train_samples": len(bundle.train_samples),
        "num_classes": bundle.num_classes,
        "path_resolution": resolution,
    }


def count_token_overflow(captions, context_length, tokenizer=None):
    if tokenizer is None:
        from text_utils.tokenizer import _tokenizer as tokenizer
    total = len(captions)
    overflow = sum(
        len(tokenizer.encode(caption)) + 2 > int(context_length)
        for caption in captions
    )
    return int(overflow), int(total)


@torch.no_grad()
def extract_text_features(model, texts, config, device, batch_size=256):
    from misc.caption_dataset import pre_caption
    from text_utils.tokenizer import tokenize

    if config.experiment.view_prompt:
        raise ValueError("external dump currently requires view_prompt=false")
    texts = [
        pre_caption(text, max_words=int(config.experiment.max_words))
        for text in texts
    ]
    global_parts, local_parts = [], []
    model.eval()
    for start in range(0, len(texts), batch_size):
        tokens = tokenize(
            texts[start:start + batch_size],
            context_length=int(config.experiment.text_length),
        ).to(device)
        global_feature, dense, attention = model.encode_text(
            tokens,
            return_dense=True,
        )
        local_feature = model.textual_emb_layer(
            attention,
            tokens,
            dense,
            training=False,
        )
        global_parts.append(F.normalize(global_feature, dim=-1).cpu().numpy())
        local_parts.append(F.normalize(local_feature, dim=-1).cpu().numpy())
    return (
        np.concatenate(global_parts).astype(np.float32, copy=False),
        np.concatenate(local_parts).astype(np.float32, copy=False),
    )


@torch.no_grad()
def extract_image_features(model, loader, device, *, expected_rows=None):
    global_parts, local_parts, pid_parts, index_parts = [], [], [], []
    model.eval()
    for images, pids, indices in loader:
        global_feature, dense, attention = model.encode_image(
            images.to(device, non_blocking=True),
            return_dense=True,
        )
        local_feature = model.visual_emb_layer(
            attention,
            dense,
            training=False,
        )
        global_parts.append(F.normalize(global_feature, dim=-1).cpu().numpy())
        local_parts.append(F.normalize(local_feature, dim=-1).cpu().numpy())
        pid_parts.append(torch.as_tensor(pids).cpu().numpy())
        index_parts.append(torch.as_tensor(indices).cpu().numpy())
    global_values = np.concatenate(global_parts).astype(np.float32, copy=False)
    local_values = np.concatenate(local_parts).astype(np.float32, copy=False)
    pids = np.concatenate(pid_parts).astype(np.int64, copy=False)
    indices = np.concatenate(index_parts).astype(np.int64, copy=False)
    if expected_rows is not None and len(global_values) != expected_rows:
        raise RuntimeError(
            f"extracted {len(global_values)} images, expected {expected_rows}"
        )
    if not np.array_equal(indices, np.arange(len(indices))):
        raise RuntimeError("external gallery feature row order changed")
    return global_values, local_values, pids


def _extra_loader(extra, transform, config):
    return DataLoader(
        ExternalImageDataset(extra.paths, extra.pid, transform),
        batch_size=int(config.data.test_batch_size),
        shuffle=False,
        num_workers=int(config.data.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _archive_pre_run(directory, config, config_path):
    from experiment import ExperimentArchive

    archive = ExperimentArchive(directory, enabled=True)
    archive.write_pre_run(config, config_path)
    # The baseline archive predates environment-backed checkpoint paths.  Add
    # this external-only variable without changing its TAG-PEDES allow-list.
    with (Path(directory) / "run_command.txt").open(
        "a", encoding="utf-8", newline=""
    ) as stream:
        stream.write(
            "external_environment:\n"
            f"CLIP_B16_CHECKPOINT={os.environ.get('CLIP_B16_CHECKPOINT', '<unset>')}\n"
        )
    return archive


def _load_model(config, bundle, device, resume_path=None):
    from model.tbps_model_MoE import (
        clip_vitb,
        configure_backbone_from_checkpoint,
    )

    if resume_path is None:
        _resolve_original_checkpoint(config)
        configure_backbone_from_checkpoint(config)
    else:
        resume_path = Path(resume_path).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        configure_backbone_from_checkpoint(config, checkpoint_path=str(resume_path))
    config.num_classes = bundle.num_classes
    model = clip_vitb(config, bundle.num_classes)
    if resume_path is None:
        from misc.build import load_checkpoint
        model, result = load_checkpoint(model, config)
        # Original CLIP intentionally lacks task heads. Unexpected source keys,
        # however, indicate a wrong backbone/checkpoint and are never ignored.
        if result.unexpected_keys:
            raise RuntimeError(
                f"unexpected original checkpoint keys: {result.unexpected_keys}"
            )
        allowed_visual_missing = (
            "visual.view_token",
            ".router.",
            ".img_router.",
            ".fea_router.",
        )
        missing_backbone = [
            key
            for key in result.missing_keys
            if (
                key.startswith("visual.") or key.startswith("encode_text.")
            ) and not any(marker in key for marker in allowed_visual_missing)
        ]
        if missing_backbone:
            raise RuntimeError(
                "original checkpoint did not initialize required backbone keys:\n  "
                + "\n  ".join(missing_backbone)
            )
    else:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError("external resume checkpoint must contain model weights")
        state_name = (
            "model_ema"
            if config.experiment.use_ema and "model_ema" in checkpoint
            else "model"
        )
        model.load_state_dict(checkpoint[state_name], strict=True)
    return model.to(device)


def _autocast(device):
    return (
        torch.autocast(device_type="cuda")
        if device.type == "cuda"
        else nullcontext()
    )


def _training_step(model, batch, alpha, device, epoch=0):
    model_config = model.module.config if hasattr(model, "module") else model.config
    with _autocast(device):
        if model_config.experiment.nrw_mode == "off":
            result = model(batch, alpha, training=True)
        else:
            result = model(batch, alpha, training=True, epoch=epoch)
        losses = [value for key, value in result.items() if "loss" in key]
        if not losses:
            raise RuntimeError("model forward returned no losses")
        loss = sum(losses)
    return result, loss


def assert_smoke_payload_blinded(payload):
    banned = ("metric", "r1", "r5", "r10", "map", "top1", "top-1", "accuracy")

    def visit(value, path="root"):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower().replace("_", "").replace("@", "")
                if any(token.replace("-", "") in normalized for token in banned):
                    raise ValueError(f"smoke payload contains blinded field {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif type(value) not in (bool, int):
            raise TypeError(
                f"smoke payload leaves must be bool/int, got {type(value).__name__} at {path}"
            )

    visit(payload)
    return True


def write_smoke_payload(directory, payload):
    assert_smoke_payload_blinded(payload)
    path = Path(directory) / "smoke_engineering.json"
    with path.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return path


def save_external_checkpoint(path, model, optimizer, config, epoch):
    """Persist the common external checkpoint schema without overwriting."""
    model_without_ddp = model.module if hasattr(model, "module") else model
    path = Path(path)
    with path.open("xb") as stream:
        torch.save({
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": int(epoch),
        }, stream)
    return path


def run_smoke_protocol(
    model,
    train_loader,
    evaluation_loader,
    optimizer,
    lr_schedule,
    device,
    *,
    steps,
    adapter_report,
    overflow_count,
    text_count,
    output_directory,
):
    """Engineering-only smoke run.  Retrieval metrics are forbidden here."""
    if steps < 1:
        raise ValueError("smoke steps must be positive")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(train_loader)
    completed = 0
    epoch = 0
    all_finite = True
    pid_in_range = True
    while completed < steps:
        load_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if getattr(train_loader, "sampler", None) is not None and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        load_time = time.perf_counter() - load_start
        pids = torch.as_tensor(batch["id"])
        pid_min, pid_max = int(pids.min()), int(pids.max())
        classes = int(adapter_report["num_classes"])
        pid_in_range &= 0 <= pid_min <= pid_max < classes
        for group in optimizer.param_groups:
            group["lr"] = float(lr_schedule[min(completed, len(lr_schedule) - 1)]) * group.get("ratio", 1.0)
        result, loss = _training_step(model, batch, 0.0, device, epoch=epoch)
        finite = bool(torch.isfinite(loss).item())
        all_finite &= finite
        if not finite:
            raise FloatingPointError(f"smoke batch {completed} loss is non-finite")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        peak = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )
        print(
            "SMOKE BATCH "
            f"step={completed + 1}/{steps} shape={tuple(batch['image'].shape)} "
            f"pid_range=[{pid_min},{pid_max}] loss_finite={finite} "
            f"gpu_peak_gib={peak:.4f} load_time_sec={load_time:.6f}"
        )
        completed += 1

    # Result-blinding invariant: only feature dimensions and row/label alignment
    # are checked.  No ranking, retrieval metric, or top-1 statistic is computed.
    eval_batches = 0
    eval_rows = 0
    eval_aligned = True
    model_without_ddp = model.module if hasattr(model, "module") else model
    model_without_ddp.eval()
    with torch.no_grad():
        for images, pids, indices in evaluation_loader:
            if eval_batches == 8:
                break
            global_feature, dense, attention = model_without_ddp.encode_image(
                images.to(device),
                return_dense=True,
            )
            local_feature = model_without_ddp.visual_emb_layer(
                attention,
                dense,
                training=False,
            )
            rows = int(images.shape[0])
            aligned = (
                global_feature.ndim == 2
                and local_feature.ndim == 2
                and global_feature.shape[0] == rows
                and local_feature.shape[0] == rows
                and len(pids) == rows
                and len(indices) == rows
            )
            eval_aligned &= bool(aligned)
            if not aligned:
                raise RuntimeError("smoke evaluation feature/PID rows are misaligned")
            print(
                "SMOKE EVAL BATCH "
                f"batch={eval_batches + 1} image_shape={tuple(images.shape)} "
                f"global_shape={tuple(global_feature.shape)} "
                f"local_shape={tuple(local_feature.shape)} pid_rows={len(pids)}"
            )
            eval_batches += 1
            eval_rows += rows
    print(
        "SMOKE TEXT TRUNCATION "
        f"overflow={overflow_count} total={text_count} "
        f"ratio={(overflow_count / text_count if text_count else 0.0):.6f}"
    )
    payload = {
        "checks": {
            "adapter_counts": bool(all(adapter_report["checks"].values())),
            "path_resolution_complete": adapter_report["path_resolution"]["resolved"] == adapter_report["path_resolution"]["checked"],
            "pid_in_range": bool(pid_in_range),
            "all_losses_finite": bool(all_finite),
            "evaluation_rows_aligned": bool(eval_aligned),
            "result_blinding_enforced": True,
        },
        "counts": {
            "requested_steps": int(steps),
            "completed_steps": int(completed),
            "train_samples": int(adapter_report["train_samples"]),
            "num_classes": int(adapter_report["num_classes"]),
            "paths_checked": int(adapter_report["path_resolution"]["checked"]),
            "paths_resolved": int(adapter_report["path_resolution"]["resolved"]),
            "captions_checked": int(text_count),
            "captions_truncated_by_context": int(overflow_count),
            "evaluation_batches_checked": int(eval_batches),
            "evaluation_rows_checked": int(eval_rows),
        },
    }
    if output_directory is not None:
        write_smoke_payload(output_directory, payload)
    return payload


def _train_full(model, loaders, config, device):
    from misc.build import build_optimizer, cosine_scheduler

    if int(config.schedule.epoch) != 10:
        raise ValueError("external protocol requires exactly 10 training epochs")
    train_loader = loaders["train_loader"]
    config.schedule.niter_per_ep = len(train_loader)
    optimizer = build_optimizer(config, model)
    schedule = cosine_scheduler(config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iteration = 0
    for epoch in range(10):
        if loaders["train_sampler"] is not None:
            loaders["train_sampler"].set_epoch(epoch)
        model.train()
        for index, batch in enumerate(train_loader):
            for group in optimizer.param_groups:
                group["lr"] = float(schedule[iteration]) * group.get("ratio", 1.0)
            alpha = (
                config.model.softlabel_ratio * min(1.0, index / len(train_loader))
                if epoch == 0
                else config.model.softlabel_ratio
            )
            _, loss = _training_step(model, batch, alpha, device, epoch=epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite external loss at epoch={epoch + 1}, batch={index + 1}"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iteration += 1
            if _master() and (index + 1) % int(config.log.print_period) == 0:
                print(
                    f"External epoch[{epoch + 1}/10] "
                    f"batch[{index + 1}/{len(train_loader)}] loss={loss.item():.6f}"
                )
        if _master():
            print(f"External epoch {epoch + 1}/10 complete; evaluation deferred to epoch 10")
    return optimizer


def print_external_evaluation(metrics, parity, quiet_metrics=False):
    """Print evaluation completion without changing persisted metric data."""
    if quiet_metrics:
        print("EVALUATION COMPLETE")
        print(f"array_equal={parity['array_equal']}")
        return
    print(
        "External evaluation "
        f"R@1={metrics['r1']:.5f} R@5={metrics['r5']:.5f} "
        f"R@10={metrics['r10']:.5f} mAP={metrics['mAP']:.5f}"
    )


def _dump_evaluation(
    model,
    bundle,
    protocol,
    loaders,
    config,
    device,
    directory,
    *,
    split,
    epoch,
    resume_path,
    argv,
    exp_name,
    quiet_metrics=False,
):
    text_global, text_local = extract_text_features(
        model,
        protocol.query_texts,
        config,
        device,
    )
    image_global, image_local, image_pid = extract_image_features(
        model,
        loaders["evaluation_loader"],
        device,
        expected_rows=len(protocol.gallery_paths),
    )
    if not np.array_equal(image_pid, np.asarray(protocol.gallery_pid)):
        raise RuntimeError("extracted gallery PID order does not match adapter protocol")
    write_core_features(
        directory,
        text_global,
        text_local,
        image_global,
        image_local,
    )
    sim_path, parity = write_and_verify_similarity(directory, device=device)
    metrics = evaluate_scores_and_write_output(
        sim_path,
        protocol.query_pid,
        protocol.gallery_pid,
        Path(directory) / "output.json",
    )

    optional = {}
    if bool(config.external.extract_bank_text):
        bank_global, bank_local = extract_text_features(
            model,
            [sample.caption for sample in bundle.train_samples],
            config,
            device,
        )
        optional.update(write_optional_features(
            directory,
            bank_global=bank_global,
            bank_local=bank_local,
        ))
    matching_extras = [
        (key, extra)
        for key, extra in bundle.extras.items()
        if extra.split == split
    ]
    if len(matching_extras) > 1:
        raise RuntimeError(
            f"multiple extra-image protocols match split={split}: "
            f"{[key for key, _ in matching_extras]}"
        )
    extra_key, extra = matching_extras[0] if matching_extras else (None, None)
    if bool(config.external.extract_extra_images) and extra is not None:
        extra_global, extra_local, extra_pid = extract_image_features(
            model,
            _extra_loader(extra, loaders["eval_transform"], config),
            device,
            expected_rows=len(extra.paths),
        )
        if not np.array_equal(extra_pid, np.asarray(extra.pid)):
            raise RuntimeError("extra image PID order does not match adapter protocol")
        optional.update(write_optional_features(
            directory,
            extra_global=extra_global,
            extra_local=extra_local,
            extra_meta={
                "dataset": bundle.dataset,
                "split": split,
                "type": extra_key,
                "n": len(extra.paths),
                "paths": extra.paths,
                "pid": extra.pid,
                "view": extra.view,
                "source_subset": extra.source_subset,
            },
        ))
    pid_path = write_pid_remap(
        directory,
        bundle.train_pid_remap,
        protocol.pid_remap,
    )
    meta = {
        "dataset": bundle.dataset,
        "split": split,
        "protocol": protocol.protocol,
        "backbone": config.model.backbone,
        "seed": int(config.misc.seed),
        "epoch": int(epoch),
        "nq": len(protocol.query_texts),
        "ng": len(protocol.gallery_paths),
        "D_global": int(text_global.shape[1]),
        "D_local": int(text_local.shape[1]),
        "query_pid": protocol.query_pid,
        "gallery_pid": protocol.gallery_pid,
        "gallery_identity_image_index": gallery_identity_image_index(
            protocol.gallery_pid
        ),
        "query_source_gallery_idx": protocol.query_source_gallery_idx,
        "query_view": protocol.query_view,
        "gallery_view": protocol.gallery_view,
        "query_source_subset": protocol.query_source_subset,
        "gallery_source_subset": protocol.gallery_source_subset,
        "gallery_paths": protocol.gallery_paths,
        "query_texts_sha256_16": text_sha256_16(protocol.query_texts),
        "pid_remap_table": str(pid_path.resolve()),
        "argv": list(sys.argv if argv is None else argv),
        "git_HEAD": git_head(),
        "input_annotations": annotation_provenance(bundle.annotation_paths),
        "checkpoint": str(Path(resume_path).resolve()) if resume_path else "checkpoint_last.pth",
        "exp_name": exp_name,
        "score_definition": "0.5 * cosine(text_global, image_global) + 0.5 * cosine(text_local_TSE, image_local_TSE); all rows L2-normalized",
        "strict_score_recomputation": parity,
        "metrics": metrics,
        "optional_files": optional,
    }
    write_meta(directory, meta)
    print_external_evaluation(metrics, parity, quiet_metrics=quiet_metrics)
    return metrics


def run_external(
    config,
    experiment=None,
    cli_overrides=(),
    config_path=None,
    eval_only=False,
    resume_path=None,
    exp_name=None,
    external_split=None,
    smoke=None,
    smoke_save_checkpoint=False,
    quiet_metrics=False,
    output_root=None,
    argv=None,
):
    del experiment  # external runs always use a fresh immutable output directory
    if eval_only and resume_path is None:
        raise ValueError("external --eval_only requires --resume PATH")
    if not eval_only and resume_path is not None:
        raise ValueError("external --resume is supported only with --eval_only")
    if smoke is not None and eval_only:
        raise ValueError("--smoke and --eval_only are mutually exclusive")
    if smoke_save_checkpoint and smoke is None:
        raise ValueError("--smoke_save_checkpoint requires --smoke [N]")
    device = _initialize_runtime(config)
    assert_external_hyperparameters(config)
    # Resolve the effective per-rank batch before archiving so config_used.yaml
    # records both the locked global batch and the actual local batch consumed
    # by each DataLoader. A second call during loader construction is
    # intentionally idempotent.
    _local_batch_size(config)
    # Environment expansion belongs exclusively to this explicit external
    # branch and is captured in config_used.yaml as the effective path.
    if not eval_only:
        _resolve_original_checkpoint(config)
    else:
        config.external.resume_checkpoint = str(
            Path(resume_path).expanduser().resolve()
        )
    split = str(external_split or config.external.eval_split)
    bundle = load_external_adapter(config.dataset_name, config.external.data_root)
    report = assert_adapter_contract(bundle, config, split)
    protocol = bundle.evaluations[split]
    if _master():
        directory = create_output_directory(
            output_root or config.external.output_root,
            bundle.dataset,
            config.model.backbone,
            split,
            config.misc.seed,
        )
    else:
        directory = Path(".")
    directory = _broadcast_path(directory)
    config.model.saved_path = str(directory)
    if _master():
        _archive_pre_run(directory, config, config_path)
        print(f"External output directory: {directory}")
        print(
            "EXTERNAL ADAPTER SUMMARY "
            f"train_samples={report['train_samples']} "
            f"num_classes={report['num_classes']} "
            f"paths_checked={report['path_resolution']['checked']} "
            f"paths_resolved={report['path_resolution']['resolved']} "
            f"path_ratio={report['path_resolution']['ratio']:.6f}"
        )
        for key, old, new in cli_overrides:
            print(f"EXTERNAL OVERRIDE {key}: {old!r} -> {new!r}")
    loaders = build_external_loaders(bundle, protocol, config)
    model = _load_model(config, bundle, device, resume_path if eval_only else None)

    if smoke is not None:
        from misc.build import build_optimizer, cosine_scheduler
        if _dist_active():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index],
                find_unused_parameters=True,
            )
        config.schedule.niter_per_ep = len(loaders["train_loader"])
        optimizer = build_optimizer(config, model)
        schedule = cosine_scheduler(config)
        # Count tokenizer truncation after the same pre_caption word truncation
        # already applied by ExternalTrainDataset.
        overflow, total = count_token_overflow(
            loaders["train_dataset"].captions,
            int(config.experiment.text_length),
        )
        payload = run_smoke_protocol(
            model,
            loaders["train_loader"],
            loaders["evaluation_loader"],
            optimizer,
            schedule,
            device,
            steps=int(smoke),
            adapter_report=report,
            overflow_count=overflow,
            text_count=total,
            output_directory=directory if _master() else None,
        )
        if _master() and smoke_save_checkpoint:
            save_external_checkpoint(
                directory / "checkpoint_smoke.pth",
                model,
                optimizer,
                config,
                # One-based epoch containing the final smoke optimizer step.
                (int(smoke) - 1) // len(loaders["train_loader"]) + 1,
            )
        if _dist_active():
            torch.distributed.barrier()
        return payload if _master() else None

    if eval_only:
        if not _master():
            if _dist_active():
                torch.distributed.barrier()
            return None
        checkpoint = torch.load(
            Path(resume_path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        epoch = int(checkpoint.get("epoch", -1))
        del checkpoint
        result = _dump_evaluation(
            model,
            bundle,
            protocol,
            loaders,
            config,
            device,
            directory,
            split=split,
            epoch=epoch,
            resume_path=resume_path,
            argv=argv,
            exp_name=exp_name,
            quiet_metrics=quiet_metrics,
        )
        if _dist_active():
            torch.distributed.barrier()
        return result

    if _dist_active():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
        )
    optimizer = _train_full(model, loaders, config, device)
    checkpoint_path = directory / "checkpoint_last.pth"
    if _master():
        save_external_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config,
            epoch=10,
        )
        model_without_ddp = model.module if hasattr(model, "module") else model
        result = _dump_evaluation(
            model_without_ddp,
            bundle,
            protocol,
            loaders,
            config,
            device,
            directory,
            split=split,
            epoch=10,
            # The dump records the exact final checkpoint that produced it.
            resume_path=checkpoint_path,
            argv=argv,
            exp_name=exp_name,
            quiet_metrics=quiet_metrics,
        )
    else:
        result = None
    if _dist_active():
        torch.distributed.barrier()
    return result


__all__ = [
    "ExternalImageDataset",
    "ExternalTrainDataset",
    "assert_adapter_contract",
    "assert_smoke_payload_blinded",
    "build_external_loaders",
    "count_token_overflow",
    "extract_image_features",
    "extract_text_features",
    "is_external_dataset",
    "load_external_adapter",
    "print_external_evaluation",
    "run_external",
    "run_smoke_protocol",
    "save_external_checkpoint",
    "write_smoke_payload",
]

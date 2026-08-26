import argparse
import csv
import os
import random
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from experiment import (
    ExperimentArchive,
    ExperimentDirectoryConflict,
    METRICS_HEADER,
    add_file_handler,
    prepare_experiment_directory,
    resolve_experiment_directory,
)
from misc.utils import get_rank
from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import metric_eval, test_tse, test
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
from options import get_args
# from eva_clip import create_model_and_transforms
from text_utils.logger import setup_logger


def build_parser():
    parser = argparse.ArgumentParser(description="TAG-PR training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="configuration file path (default: config/config.yaml)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="experiment name (default: None)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="experiment output root (default: config model.saved_path)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="override data.batch_size (default: config value)",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="override data.test_batch_size (default: config value)",
    )
    parser.add_argument(
        "--input_resolution",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="override experiment.input_resolution as height width (default: config value)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=("ViT-B/16", "ViT-L/14"),
        default=None,
        help="override model.backbone (default: config value)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="override schedule.epoch (default: config value)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override misc.seed (default: config value)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override schedule.lr (default: config value)",
    )
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="enable activation checkpointing (default: config value)",
    )
    parser.add_argument(
        "--no_grad_checkpoint",
        action="store_true",
        help="disable activation checkpointing (overrides config and --grad_checkpoint)",
    )
    parser.add_argument(
        "--use_bridge",
        action="store_true",
        help="enable cross-view bridge image loading (default: config value)",
    )
    parser.add_argument(
        "--no_use_bridge",
        action="store_true",
        help="disable cross-view bridge image loading (overrides config and --use_bridge)",
    )
    parser.add_argument(
        "--bridge_cv_ratio",
        type=float,
        default=None,
        help="override experiment.bridge_cv_ratio (default: config value)",
    )
    parser.add_argument(
        "--bridge_tg_ratio",
        type=float,
        default=None,
        help="override experiment.bridge_tg_ratio (default: config value)",
    )
    parser.add_argument(
        "--dhn_margin",
        type=float,
        default=None,
        help="override experiment.dhn_margin (default: config value)",
    )
    parser.add_argument(
        "--sdm_ratio",
        type=float,
        default=None,
        help="override experiment.sdm_ratio (default: config value)",
    )
    parser.add_argument(
        "--tal_ratio",
        type=float,
        default=None,
        help="override experiment.tal_ratio (default: config value)",
    )
    parser.add_argument(
        "--tal_margin",
        type=float,
        default=None,
        help="override experiment.tal_margin (default: config value)",
    )
    parser.add_argument(
        "--tal_tau",
        type=float,
        default=None,
        help="override experiment.tal_tau (default: config value)",
    )
    parser.add_argument(
        "--nrw_mode",
        choices=("off", "view_norm", "global_norm", "fixed"),
        default=None,
        help="override experiment.nrw_mode (default: config value)",
    )
    parser.add_argument(
        "--nrw_alpha",
        type=float,
        default=None,
        help="override experiment.nrw_alpha (default: config value)",
    )
    parser.add_argument(
        "--nrw_min_weight",
        type=float,
        default=None,
        help="override experiment.nrw_min_weight (default: config value)",
    )
    parser.add_argument(
        "--nrw_warmup_epochs",
        type=int,
        default=None,
        help="override experiment.nrw_warmup_epochs (default: config value)",
    )
    parser.add_argument(
        "--nrw_ema",
        type=float,
        default=None,
        help="override experiment.nrw_ema (default: config value)",
    )
    parser.add_argument(
        "--nrw_fixed_aerial",
        type=float,
        default=None,
        help="override experiment.nrw_fixed_aerial (default: config value)",
    )
    parser.add_argument(
        "--view_ratio",
        type=float,
        default=None,
        help="override experiment.view_ratio (default: config value)",
    )
    parser.add_argument(
        "--ortho_ratio",
        type=float,
        default=None,
        help="override experiment.ortho_ratio (default: config value)",
    )
    parser.add_argument(
        "--disable_moe",
        action="store_true",
        help="replace expert blocks with standard ViT blocks (default: config value)",
    )
    parser.add_argument(
        "--no_disable_moe",
        action="store_true",
        help="enable MoE blocks (overrides config and --disable_moe)",
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=None,
        help="override experiment.max_words (default: config value)",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="enable model exponential moving average (default: config value)",
    )
    parser.add_argument(
        "--no_use_ema",
        action="store_true",
        help="disable model EMA (overrides config and --use_ema)",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=None,
        help="override experiment.ema_decay (default: config value)",
    )
    parser.add_argument(
        "--view_prompt",
        action="store_true",
        help="enable view-conditioned text prompts (default: config value)",
    )
    parser.add_argument(
        "--no_view_prompt",
        action="store_true",
        help="disable view-conditioned prompts (overrides config and --view_prompt)",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="enable k-reciprocal evaluation re-ranking (default: config value)",
    )
    parser.add_argument(
        "--no_rerank",
        action="store_true",
        help="disable re-ranking (overrides config and --rerank)",
    )
    parser.add_argument(
        "--rerank_k1",
        type=int,
        default=None,
        help="override experiment.rerank_k1 (default: config value)",
    )
    parser.add_argument(
        "--rerank_k2",
        type=int,
        default=None,
        help="override experiment.rerank_k2 (default: config value)",
    )
    parser.add_argument(
        "--rerank_lambda",
        type=float,
        default=None,
        help="override experiment.rerank_lambda (default: config value)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="load --resume and run one evaluation without training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint path for --eval_only (default: None)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="resolve configuration and write pre-run archives without training",
    )
    # Retain the previously accepted option even though the baseline does not use it.
    parser.add_argument("--simplified", action="store_true", help=argparse.SUPPRESS)
    return parser


def apply_cli_overrides(config, args):
    overrides = []
    override_specs = [
        (args.backbone, config.model, "backbone", "model.backbone"),
        (args.batch_size, config.data, "batch_size", "data.batch_size"),
        (
            args.test_batch_size,
            config.data,
            "test_batch_size",
            "data.test_batch_size",
        ),
        (
            args.input_resolution,
            config.experiment,
            "input_resolution",
            "experiment.input_resolution",
        ),
        (args.epoch, config.schedule, "epoch", "schedule.epoch"),
        (args.seed, config.misc, "seed", "misc.seed"),
        (args.lr, config.schedule, "lr", "schedule.lr"),
        (
            args.bridge_cv_ratio,
            config.experiment,
            "bridge_cv_ratio",
            "experiment.bridge_cv_ratio",
        ),
        (
            args.bridge_tg_ratio,
            config.experiment,
            "bridge_tg_ratio",
            "experiment.bridge_tg_ratio",
        ),
        (
            args.dhn_margin,
            config.experiment,
            "dhn_margin",
            "experiment.dhn_margin",
        ),
        (
            args.sdm_ratio,
            config.experiment,
            "sdm_ratio",
            "experiment.sdm_ratio",
        ),
        (
            args.tal_ratio,
            config.experiment,
            "tal_ratio",
            "experiment.tal_ratio",
        ),
        (
            args.tal_margin,
            config.experiment,
            "tal_margin",
            "experiment.tal_margin",
        ),
        (
            args.tal_tau,
            config.experiment,
            "tal_tau",
            "experiment.tal_tau",
        ),
        (
            args.nrw_mode,
            config.experiment,
            "nrw_mode",
            "experiment.nrw_mode",
        ),
        (
            args.nrw_alpha,
            config.experiment,
            "nrw_alpha",
            "experiment.nrw_alpha",
        ),
        (
            args.nrw_min_weight,
            config.experiment,
            "nrw_min_weight",
            "experiment.nrw_min_weight",
        ),
        (
            args.nrw_warmup_epochs,
            config.experiment,
            "nrw_warmup_epochs",
            "experiment.nrw_warmup_epochs",
        ),
        (
            args.nrw_ema,
            config.experiment,
            "nrw_ema",
            "experiment.nrw_ema",
        ),
        (
            args.nrw_fixed_aerial,
            config.experiment,
            "nrw_fixed_aerial",
            "experiment.nrw_fixed_aerial",
        ),
        (
            args.view_ratio,
            config.experiment,
            "view_ratio",
            "experiment.view_ratio",
        ),
        (
            args.ortho_ratio,
            config.experiment,
            "ortho_ratio",
            "experiment.ortho_ratio",
        ),
        (
            args.max_words,
            config.experiment,
            "max_words",
            "experiment.max_words",
        ),
        (
            args.ema_decay,
            config.experiment,
            "ema_decay",
            "experiment.ema_decay",
        ),
        (
            args.rerank_k1,
            config.experiment,
            "rerank_k1",
            "experiment.rerank_k1",
        ),
        (
            args.rerank_k2,
            config.experiment,
            "rerank_k2",
            "experiment.rerank_k2",
        ),
        (
            args.rerank_lambda,
            config.experiment,
            "rerank_lambda",
            "experiment.rerank_lambda",
        ),
    ]
    for new_value, section, key, dotted_key in override_specs:
        if new_value is None:
            continue
        old_value = section[key]
        section[key] = new_value
        overrides.append((dotted_key, old_value, new_value))
    if args.no_grad_checkpoint or args.grad_checkpoint:
        old_value = config.experiment.grad_checkpoint
        new_value = not args.no_grad_checkpoint
        config.experiment.grad_checkpoint = new_value
        overrides.append(
            ("experiment.grad_checkpoint", old_value, new_value)
        )
    if args.no_use_bridge or args.use_bridge:
        old_value = config.experiment.use_bridge
        new_value = not args.no_use_bridge
        config.experiment.use_bridge = new_value
        overrides.append(("experiment.use_bridge", old_value, new_value))
    if args.no_disable_moe or args.disable_moe:
        old_value = config.experiment.disable_moe
        new_value = not args.no_disable_moe
        config.experiment.disable_moe = new_value
        overrides.append(("experiment.disable_moe", old_value, new_value))
    if args.no_use_ema or args.use_ema:
        old_value = config.experiment.use_ema
        new_value = not args.no_use_ema
        config.experiment.use_ema = new_value
        overrides.append(("experiment.use_ema", old_value, new_value))
    if args.no_view_prompt or args.view_prompt:
        old_value = config.experiment.view_prompt
        new_value = not args.no_view_prompt
        config.experiment.view_prompt = new_value
        overrides.append(("experiment.view_prompt", old_value, new_value))
    if args.no_rerank or args.rerank:
        old_value = config.experiment.rerank
        new_value = not args.no_rerank
        config.experiment.rerank = new_value
        overrides.append(("experiment.rerank", old_value, new_value))
    return overrides


def resolve_runtime_config(args):
    config = parse_config(args.config)
    overrides = apply_cli_overrides(config, args)
    return config, overrides


def configure_experiment(config, args):
    directory = resolve_experiment_directory(
        config.model.saved_path,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
    )

    if args.output_dir is not None or args.exp_name is not None:
        old_saved_path = config.model.saved_path
        config.model.saved_path = str(directory)
        path_override = ("model.saved_path", old_saved_path, str(directory))
    else:
        path_override = None

    # Preserve the pure no-argument baseline while protecting every explicitly
    # managed run from overwriting a completed experiment.
    protect_checkpoint = any(
        value is not None
        for value in (
            args.exp_name,
            args.output_dir,
            args.batch_size,
            args.test_batch_size,
            args.input_resolution,
            args.backbone,
            args.epoch,
            args.seed,
            args.lr,
            args.bridge_cv_ratio,
            args.bridge_tg_ratio,
            args.dhn_margin,
            args.sdm_ratio,
            args.tal_ratio,
            args.tal_margin,
            args.tal_tau,
            args.nrw_mode,
            args.nrw_alpha,
            args.nrw_min_weight,
            args.nrw_warmup_epochs,
            args.nrw_ema,
            args.nrw_fixed_aerial,
            args.view_ratio,
            args.ortho_ratio,
            args.max_words,
            args.ema_decay,
            args.rerank_k1,
            args.rerank_k2,
            args.rerank_lambda,
        )
    ) or any(
        (
            args.grad_checkpoint,
            args.no_grad_checkpoint,
            args.use_bridge,
            args.no_use_bridge,
            args.disable_moe,
            args.no_disable_moe,
            args.use_ema,
            args.no_use_ema,
            args.view_prompt,
            args.no_view_prompt,
            args.rerank,
            args.no_rerank,
            args.dry_run,
        )
    ) or args.config != "config/config.yaml"
    if (
        args.eval_only
        and args.output_dir is None
        and args.exp_name is None
    ):
        # Evaluation never writes a checkpoint, so it may safely read one from
        # an otherwise protected experiment directory.
        protect_checkpoint = False
    directory = prepare_experiment_directory(directory, protect_checkpoint)
    return directory, path_override


def log_cli_overrides(logger, overrides, config=None):
    logger.info("OVERRIDE SUMMARY")
    if not overrides:
        logger.info("no overrides")
    else:
        for key, old_value, new_value in overrides:
            logger.info("%s: %r -> %r", key, old_value, new_value)
    if config is None:
        logger.info("END OVERRIDE SUMMARY")
        return
    logger.info(
        "experiment.grad_checkpoint effective: %r",
        config.experiment.grad_checkpoint,
    )
    logger.info(
        "experiment.use_bridge effective: %r",
        config.experiment.use_bridge,
    )
    logger.info(
        "experiment.bridge_cv_ratio effective: %r",
        config.experiment.bridge_cv_ratio,
    )
    logger.info(
        "experiment.bridge_tg_ratio effective: %r",
        config.experiment.bridge_tg_ratio,
    )
    logger.info("experiment.dhn_margin effective: %r", config.experiment.dhn_margin)
    logger.info("experiment.sdm_ratio effective: %r", config.experiment.sdm_ratio)
    logger.info("experiment.tal_ratio effective: %r", config.experiment.tal_ratio)
    logger.info("experiment.tal_margin effective: %r", config.experiment.tal_margin)
    logger.info("experiment.tal_tau effective: %r", config.experiment.tal_tau)
    logger.info("experiment.nrw_mode effective: %r", config.experiment.nrw_mode)
    logger.info("experiment.nrw_alpha effective: %r", config.experiment.nrw_alpha)
    logger.info(
        "experiment.nrw_min_weight effective: %r",
        config.experiment.nrw_min_weight,
    )
    logger.info(
        "experiment.nrw_warmup_epochs effective: %r",
        config.experiment.nrw_warmup_epochs,
    )
    logger.info("experiment.nrw_ema effective: %r", config.experiment.nrw_ema)
    logger.info(
        "experiment.nrw_fixed_aerial effective: %r",
        config.experiment.nrw_fixed_aerial,
    )
    logger.info("experiment.view_ratio effective: %r", config.experiment.view_ratio)
    logger.info("experiment.ortho_ratio effective: %r", config.experiment.ortho_ratio)
    logger.info("experiment.disable_moe effective: %r", config.experiment.disable_moe)
    logger.info("experiment.max_words effective: %r", config.experiment.max_words)
    logger.info("experiment.use_ema effective: %r", config.experiment.use_ema)
    logger.info("experiment.ema_decay effective: %r", config.experiment.ema_decay)
    logger.info("experiment.view_prompt effective: %r", config.experiment.view_prompt)
    logger.info("experiment.rerank effective: %r", config.experiment.rerank)
    logger.info("experiment.rerank_k1 effective: %r", config.experiment.rerank_k1)
    logger.info("experiment.rerank_k2 effective: %r", config.experiment.rerank_k2)
    logger.info(
        "experiment.rerank_lambda effective: %r",
        config.experiment.rerank_lambda,
    )
    logger.info("model.backbone effective: %r", config.model.backbone)
    logger.info(
        "model.expert_blocks_auto effective: %r",
        config.model.expert_blocks_auto,
    )
    logger.info("END OVERRIDE SUMMARY")

    if 'vision_width' in config.model:
        logger.info("BACKBONE STRUCTURE SUMMARY")
        logger.info("  source: %s", config.model.architecture_source)
        logger.info("  patch_size: %s", config.model.patch_size)
        logger.info("  vision_width: %s", config.model.vision_width)
        logger.info("  vision_layers: %s", config.model.vision_layers)
        logger.info("  vision_heads: %s", config.model.vision_heads)
        logger.info("  embed_dim: %s", config.model.embed_dim)
        logger.info("  text_width: %s", config.model.text_width)
        logger.info("  text_layers: %s", config.model.text_layers)
        logger.info("  text_heads: %s", config.model.text_heads)
        logger.info(
            "  pretrained_grid_size: %s",
            list(config.model.pretrained_grid_size),
        )
        patch_rows = (
            config.experiment.input_resolution[0] // config.model.patch_size
        )
        patch_columns = (
            config.experiment.input_resolution[1] // config.model.patch_size
        )
        logger.info("  actual_patch_count: %s", patch_rows * patch_columns)
        logger.info(
            "  expert_blocks: %s -> %s (auto=%s)",
            list(config.model.expert_blocks_source),
            list(config.expert_blocks),
            config.model.expert_blocks_auto,
        )
        logger.info("END BACKBONE STRUCTURE SUMMARY")


def assert_training_batch_size(config, train_loader):
    expected = config.data.batch_size
    actual = train_loader.batch_size
    if actual != expected:
        raise RuntimeError(
            "DataLoader batch_size mismatch: "
            f"config.data.batch_size={expected}, "
            f"train_loader.batch_size={actual}"
        )


def _rebuild_loader_with_batch_size(loader, batch_size):
    kwargs = {
        "dataset": loader.dataset,
        "batch_size": batch_size,
        "sampler": loader.sampler,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader.generator,
    }
    if loader.num_workers > 0:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["persistent_workers"] = loader.persistent_workers
    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return torch.utils.data.DataLoader(**kwargs)


def apply_test_loader_override(config, dataloader, cli_overrides):
    if not any(key == "data.test_batch_size" for key, _, _ in cli_overrides):
        return
    expected = config.data.test_batch_size
    test_loader = dataloader["test_loader"]
    if test_loader.batch_size != expected:
        test_loader = _rebuild_loader_with_batch_size(test_loader, expected)
        dataloader["test_loader"] = test_loader
    if test_loader.batch_size != expected:
        raise RuntimeError(
            "DataLoader test batch_size mismatch: "
            f"config.data.test_batch_size={expected}, "
            f"test_loader.batch_size={test_loader.batch_size}"
        )


class ModelEMA:
    """Parameter EMA with direct-copy semantics for model buffers."""

    def __init__(self, model, decay):
        if not 0.0 <= decay < 1.0:
            raise ValueError("experiment.ema_decay must be in [0, 1)")
        self.decay = decay
        self.parameter_names = set(dict(model.named_parameters()))
        self.shadow = OrderedDict(
            (name, value.detach().clone())
            for name, value in model.state_dict().items()
        )

    @torch.no_grad()
    def update(self, model, update_parameters=True):
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError("model state changed after EMA initialization")
        for name, average in self.shadow.items():
            value = current[name].detach()
            if name in self.parameter_names:
                if update_parameters:
                    average.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                average.copy_(value)

    def state_dict(self):
        return OrderedDict(
            (name, value.detach().clone())
            for name, value in self.shadow.items()
        )

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if state_dict.keys() != self.shadow.keys():
            raise RuntimeError("checkpoint EMA keys do not match the model")
        for name, average in self.shadow.items():
            saved = state_dict[name]
            if saved.shape != average.shape or saved.dtype != average.dtype:
                raise RuntimeError(f"checkpoint EMA tensor mismatch for {name}")
        for name, average in self.shadow.items():
            average.copy_(state_dict[name].to(device=average.device))

    @torch.no_grad()
    def _swap(self, model):
        live_state = model.state_dict()
        if live_state.keys() != self.shadow.keys():
            raise RuntimeError("model state does not match the EMA state")
        for name, value in live_state.items():
            average = self.shadow[name]
            if (
                value.shape != average.shape
                or value.dtype != average.dtype
                or value.device != average.device
            ):
                raise RuntimeError(f"EMA tensor mismatch for {name}")
        for name, value in live_state.items():
            temporary = value.detach().clone()
            value.copy_(self.shadow[name])
            self.shadow[name].copy_(temporary)

    @contextmanager
    def average_parameters(self, model):
        """Swap EMA values in place and restore the exact live state on exit."""
        was_training = model.training
        self._swap(model)
        try:
            yield
        finally:
            self._swap(model)
            model.train(was_training)


@torch.no_grad()
def test_tse_view_prompt(model, data_loader, max_length, device):
    """Evaluate text twice and select scores using each gallery image's view."""
    from text_utils.tokenizer import tokenize

    model.eval()
    dataset = data_loader.dataset
    if not hasattr(dataset, "img_cam_ids"):
        raise RuntimeError("view-prompt evaluation requires gallery img_cam_ids")

    texts = dataset.text
    text_features_by_view = {}
    text_tse_features_by_view = {}
    text_batch_size = 256
    for view in (0, 1):
        text_features = []
        text_tse_features = []
        for start in range(0, len(texts), text_batch_size):
            batch_texts = texts[start: min(len(texts), start + text_batch_size)]
            tokens = tokenize(
                batch_texts,
                context_length=max_length - 1,
            ).to(device)
            cam_ids = torch.full(
                (tokens.shape[0],),
                view,
                dtype=torch.long,
                device=device,
            )
            text_feature, text_dense, text_attention = (
                model.encode_text_with_view_prompt(
                    tokens,
                    cam_ids,
                    return_dense=True,
                )
            )
            prompted_tokens = model.view_prompt_token_ids(tokens)
            text_features.append(F.normalize(text_feature, dim=-1))
            text_tse_features.append(
                F.normalize(
                    model.textual_emb_layer(
                        text_attention,
                        prompted_tokens,
                        text_dense,
                        training=False,
                    ),
                    dim=-1,
                )
            )
        text_features_by_view[view] = torch.cat(text_features, dim=0)
        text_tse_features_by_view[view] = torch.cat(text_tse_features, dim=0)

    image_features = []
    image_tse_features = []
    for images in data_loader:
        images = images.to(device)
        image_feature, image_dense, image_attention = model.encode_image(
            images,
            return_dense=True,
        )
        image_features.append(F.normalize(image_feature, dim=-1))
        image_tse_features.append(
            F.normalize(
                model.visual_emb_layer(
                    image_attention,
                    image_dense,
                    training=False,
                ),
                dim=-1,
            )
        )
    image_features = torch.cat(image_features, dim=0)
    image_tse_features = torch.cat(image_tse_features, dim=0)

    gallery_cam_ids = dataset.img_cam_ids.to(device)
    if gallery_cam_ids.shape[0] != image_features.shape[0]:
        raise RuntimeError("gallery cam_id count does not match gallery image count")
    if torch.any((gallery_cam_ids < 0) | (gallery_cam_ids > 1)):
        raise ValueError("view prompt requires gallery cam_id values in {0, 1}")

    similarities = image_features.new_empty(len(texts), image_features.shape[0])
    for view in (0, 1):
        image_mask = gallery_cam_ids == view
        if not image_mask.any():
            continue
        similarities[:, image_mask] = (
            text_features_by_view[view] @ image_features[image_mask].t()
            + text_tse_features_by_view[view] @ image_tse_features[image_mask].t()
        ) / 2

    return metric_eval(
        similarities,
        dataset.img2person,
        dataset.txt2person,
    )


def _fused_retrieval_features(*features):
    return torch.cat(features, dim=-1).float() / (len(features) ** 0.5)


@torch.no_grad()
def test_tse_with_rerank(model, data_loader, config, logger):
    """Evaluate raw scores and their label-free k-reciprocal re-ranking."""
    from text_utils.tokenizer import tokenize
    from misc.rerank import (
        ENCODING_BLOCK_SIZE,
        JACCARD_QUERY_BLOCK_SIZE,
        QUERY_EXPANSION_BLOCK_SIZE,
        RANK_BLOCK_SIZE,
        k_reciprocal_rerank,
        self_cosine_distance,
    )

    if not torch.cuda.is_available() or not str(config.device).startswith('cuda'):
        raise RuntimeError(
            "k-reciprocal re-ranking requires CUDA; CPU fallback is disabled"
        )
    model.eval()
    dataset = data_loader.dataset
    texts = dataset.text
    text_batch_size = 256
    max_length = config.experiment.text_length
    device = config.device

    if config.experiment.view_prompt:
        text_features_by_view = {}
        text_tse_features_by_view = {}
        for view in (0, 1):
            text_features = []
            text_tse_features = []
            for start in range(0, len(texts), text_batch_size):
                batch_texts = texts[start: min(len(texts), start + text_batch_size)]
                tokens = tokenize(
                    batch_texts,
                    context_length=max_length - 1,
                ).to(device)
                cam_ids = torch.full(
                    (tokens.shape[0],),
                    view,
                    dtype=torch.long,
                    device=device,
                )
                text_feature, text_dense, text_attention = (
                    model.encode_text_with_view_prompt(
                        tokens,
                        cam_ids,
                        return_dense=True,
                    )
                )
                prompted_tokens = model.view_prompt_token_ids(tokens)
                text_features.append(F.normalize(text_feature, dim=-1))
                text_tse_features.append(
                    F.normalize(
                        model.textual_emb_layer(
                            text_attention,
                            prompted_tokens,
                            text_dense,
                            training=False,
                        ),
                        dim=-1,
                    )
                )
            text_features_by_view[view] = torch.cat(text_features, dim=0)
            text_tse_features_by_view[view] = torch.cat(
                text_tse_features,
                dim=0,
            )
        # Average both prompt-conditioned text graphs into one QQ relation.
        query_relation_features = _fused_retrieval_features(
            text_features_by_view[0],
            text_tse_features_by_view[0],
            text_features_by_view[1],
            text_tse_features_by_view[1],
        )
    else:
        text_features = []
        text_tse_features = []
        for start in range(0, len(texts), text_batch_size):
            batch_texts = texts[start: min(len(texts), start + text_batch_size)]
            tokens = tokenize(batch_texts, context_length=max_length).to(device)
            text_feature, text_dense, text_attention = model.encode_text(
                tokens,
                return_dense=True,
            )
            text_features.append(F.normalize(text_feature, dim=-1))
            text_tse_features.append(
                F.normalize(
                    model.textual_emb_layer(
                        text_attention,
                        tokens,
                        text_dense,
                        training=False,
                    ),
                    dim=-1,
                )
            )
        text_features = torch.cat(text_features, dim=0)
        text_tse_features = torch.cat(text_tse_features, dim=0)
        query_relation_features = _fused_retrieval_features(
            text_features,
            text_tse_features,
        )

    image_features = []
    image_tse_features = []
    for images in data_loader:
        images = images.to(device)
        image_feature, image_dense, image_attention = model.encode_image(
            images,
            return_dense=True,
        )
        image_features.append(F.normalize(image_feature, dim=-1))
        image_tse_features.append(
            F.normalize(
                model.visual_emb_layer(
                    image_attention,
                    image_dense,
                    training=False,
                ),
                dim=-1,
            )
        )
    image_features = torch.cat(image_features, dim=0)
    image_tse_features = torch.cat(image_tse_features, dim=0)
    gallery_relation_features = _fused_retrieval_features(
        image_features,
        image_tse_features,
    )

    if config.experiment.view_prompt:
        if not hasattr(dataset, "img_cam_ids"):
            raise RuntimeError("view-prompt re-ranking requires gallery img_cam_ids")
        gallery_cam_ids = dataset.img_cam_ids.to(device)
        if gallery_cam_ids.shape[0] != image_features.shape[0]:
            raise RuntimeError("gallery cam_id count does not match gallery images")
        if torch.any((gallery_cam_ids < 0) | (gallery_cam_ids > 1)):
            raise ValueError("view prompt requires cam_id values in {0, 1}")
        similarities = image_features.new_empty(
            len(texts),
            image_features.shape[0],
        )
        for view in (0, 1):
            image_mask = gallery_cam_ids == view
            if not image_mask.any():
                continue
            similarities[:, image_mask] = (
                text_features_by_view[view] @ image_features[image_mask].t()
                + text_tse_features_by_view[view]
                @ image_tse_features[image_mask].t()
            ) / 2
    else:
        similarities = (
            text_features @ image_features.t()
            + text_tse_features @ image_tse_features.t()
        ) / 2

    raw_result = metric_eval(
        similarities,
        dataset.img2person,
        dataset.txt2person,
    )
    # The normalized cosine distance is symmetric, non-negative, and bounded.
    query_gallery_distance = (
        1.0 - similarities.float()
    ).mul_(0.5).clamp_(0.0, 1.0)
    del similarities
    if config.experiment.view_prompt:
        del text_features_by_view, text_tse_features_by_view
    else:
        del text_features, text_tse_features
    del image_features, image_tse_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    rerank_start = time.time()
    logger.info(
        "RERANK BLOCKS rank=%s encoding=%s query_expansion=%s jaccard_query=%s",
        RANK_BLOCK_SIZE,
        ENCODING_BLOCK_SIZE,
        QUERY_EXPANSION_BLOCK_SIZE,
        JACCARD_QUERY_BLOCK_SIZE,
    )
    try:
        query_query_distance = self_cosine_distance(query_relation_features)
        gallery_gallery_distance = self_cosine_distance(
            gallery_relation_features
        )
        del query_relation_features, gallery_relation_features
        reranked_distance = k_reciprocal_rerank(
            query_gallery_distance,
            query_query_distance,
            gallery_gallery_distance,
            k1=config.experiment.rerank_k1,
            k2=config.experiment.rerank_k2,
            lambda_value=config.experiment.rerank_lambda,
        )
        del (
            query_gallery_distance,
            query_query_distance,
            gallery_gallery_distance,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA out of memory during k-reciprocal re-ranking; lower "
            "--test_batch_size for feature extraction, free model/EMA memory, "
            "or reduce the internal re-ranking block sizes"
        ) from exc
    torch.cuda.synchronize()
    rerank_time_sec = time.time() - rerank_start
    logger.info("K-reciprocal re-ranking completed in %.3f seconds", rerank_time_sec)
    reranked_result = metric_eval(
        reranked_distance.neg_(),
        dataset.img2person,
        dataset.txt2person,
    )
    return raw_result, reranked_result


def evaluate_model(model, data_loader, config, logger):
    if config.experiment.rerank:
        return test_tse_with_rerank(model, data_loader, config, logger)
    if config.experiment.view_prompt:
        result = test_tse_view_prompt(
            model,
            data_loader,
            config.experiment.text_length,
            config.device,
        )
    else:
        result = test_tse(
            model,
            data_loader,
            config.experiment.text_length,
            config.device,
        )
    return result, None


def log_evaluation_results(logger, result, reranked_result=None):
    logger.info(
        'Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(
            top1=result['r1'],
            top5=result['r5'],
            top10=result['r10'],
            mAP=result['mAP'],
        )
    )
    if reranked_result is not None:
        logger.info(
            'RERANK Acc@1 {top1:.5f} Acc@5 {top5:.5f} '
            'Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(
                top1=reranked_result['r1'],
                top5=reranked_result['r5'],
                top10=reranked_result['r10'],
                mAP=reranked_result['mAP'],
            )
        )


RERANK_METRICS_HEADER = METRICS_HEADER + [
    "R1_rr",
    "R5_rr",
    "R10_rr",
    "mAP_rr",
]


def initialize_metrics(experiment, rerank):
    if not rerank:
        experiment.initialize_metrics()
        return
    if not experiment.enabled:
        return
    metrics_path = experiment.directory / "metrics_by_epoch.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(RERANK_METRICS_HEADER)
        handle.flush()
        os.fsync(handle.fileno())


def append_metrics(
    experiment,
    epoch,
    result,
    train_loss,
    epoch_time_sec,
    peak_mem_alloc_gb,
    peak_mem_reserved_gb,
    reranked_result=None,
):
    if reranked_result is None:
        experiment.append_metrics(
            epoch,
            result['r1'],
            result['r5'],
            result['r10'],
            result['mAP'],
            train_loss,
            epoch_time_sec,
            peak_mem_alloc_gb,
            peak_mem_reserved_gb,
        )
        return
    if not experiment.enabled:
        return
    metrics_path = experiment.directory / "metrics_by_epoch.csv"
    with metrics_path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            [
                epoch,
                result['r1'],
                result['r5'],
                result['r10'],
                result['mAP'],
                train_loss,
                epoch_time_sec,
                peak_mem_alloc_gb,
                peak_mem_reserved_gb,
                reranked_result['r1'],
                reranked_result['r5'],
                reranked_result['r10'],
                reranked_result['mAP'],
            ]
        )
        handle.flush()
        os.fsync(handle.fileno())


def load_resume_weights(model, resume_path, use_ema, logger):
    checkpoint_path = Path(resume_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise RuntimeError("resume checkpoint must contain a 'model' state dict")
    state_name = 'model'
    if use_ema:
        if 'model_ema' in checkpoint:
            state_name = 'model_ema'
        else:
            logger.warning(
                "Resume checkpoint has no model_ema; evaluating model weights"
            )
    load_result = model.load_state_dict(checkpoint[state_name], strict=True)
    logger.info(
        "Loaded %s weights from %s (missing=%s unexpected=%s)",
        state_name,
        checkpoint_path,
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )
    return load_result


def validate_checkpoint_mapping(load_result, config):
    if load_result is None:
        return
    unexpected = list(load_result.unexpected_keys)
    if unexpected:
        raise RuntimeError(
            "Checkpoint contains keys that cannot be mapped:\n  "
            + "\n  ".join(unexpected)
        )
    if config.model.ckpt_type != 'original_clip':
        if load_result.missing_keys:
            raise RuntimeError(
                "Saved checkpoint is missing model keys:\n  "
                + "\n  ".join(load_result.missing_keys)
            )
        return
    allowed_visual_missing = (
        'visual.view_token',
        '.router.',
        '.img_router.',
        '.fea_router.',
    )
    missing_backbone = [
        key
        for key in load_result.missing_keys
        if (
            key.startswith('visual.') or key.startswith('encode_text.')
        ) and not any(marker in key for marker in allowed_visual_missing)
    ]
    if missing_backbone:
        raise RuntimeError(
            "Checkpoint did not initialize required backbone keys:\n  "
            + "\n  ".join(missing_backbone)
        )


def _cuda_peak_memory_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    gib = 1024 ** 3
    return (
        torch.cuda.max_memory_allocated() / gib,
        torch.cuda.max_memory_reserved() / gib,
    )


def environment_rank():
    try:
        return int(os.environ.get("RANK", "0"))
    except ValueError:
        return 0


def run(
    config,
    experiment=None,
    cli_overrides=(),
    config_path=None,
    eval_only=False,
    resume_path=None,
):
    logger = setup_logger('TAG', distributed_rank=get_rank())
    logger.propagate = False
    if experiment is not None:
        experiment.write_pre_run(config, config_path)
    if experiment is not None and experiment.enabled:
        add_file_handler(logger, experiment.directory)
    log_cli_overrides(logger, cli_overrides, config)
    logger.info(f'\n{config}')
    if experiment is not None:
        initialize_metrics(experiment, config.experiment.rerank)

    training_start_time = time.time()
    train_loader = None
    completed_epochs = 0
    full_peak_alloc_gb = 0.0
    full_peak_reserved_gb = 0.0
    best_rank_1 = 0.0
    best_epoch = 0
    best_metrics = None
    best_metrics_epoch = None

    try:
        # Keep training-only dependencies out of import/help/dry-run code paths.
        from model.tbps_model_MoE import clip_vitb

        dataloader = build_pedes_data(config)
        apply_test_loader_override(config, dataloader, cli_overrides)
        train_loader = dataloader['train_loader']
        assert_training_batch_size(config, train_loader)
        logger.info(
            "DataLoader batch_size verified: expected=%s, actual=%s",
            config.data.batch_size,
            train_loader.batch_size,
        )
        logger.info(
            "Evaluation DataLoader batch_size: config=%s, actual=%s",
            config.data.test_batch_size,
            dataloader['test_loader'].batch_size,
        )
        num_classes = len(train_loader.dataset.person2text)
        config.num_classes = num_classes

        meters = {
            "loss": AverageMeter(),
            "itc_loss": AverageMeter(),
            "itc_tse_loss": AverageMeter(),
            "view_loss": AverageMeter(),
            "view_acc": AverageMeter(),
            "sdm_loss": AverageMeter(),
            "tal_loss": AverageMeter(),
            "ortho_loss": AverageMeter(),
            "ritc_loss": AverageMeter(),
            "ritc_tse_loss": AverageMeter(),
            "balance_loss": AverageMeter(),
            "router_z_loss": AverageMeter(),
            "id_loss": AverageMeter(),
            "bridge_cv_loss": AverageMeter(),
            "bridge_tg_loss": AverageMeter(),
        }
        if config.experiment.dhn_margin > 0:
            meters["dhn_hard_ratio"] = AverageMeter()
        if config.experiment.nrw_mode != 'off':
            meters.update({
                "nrw_w_mean_aerial": AverageMeter(),
                "nrw_w_mean_ground": AverageMeter(),
                "nrw_mu_aerial": AverageMeter(),
                "nrw_mu_ground": AverageMeter(),
            })

        # model
        # model_name = "EVA02-CLIP-B-16"
        # pretrained = "/data/zxy/UAV/checkpoints/EVA02_CLIP_B_psz16_s8B.pt"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
        # model, _, preprocess = create_model_and_transforms(config, model_name, pretrained, force_custom_clip=True)
        model = clip_vitb(config, num_classes)
        if eval_only:
            if not is_master():
                return None
            load_resume_weights(
                model,
                resume_path,
                config.experiment.use_ema,
                logger,
            )
            model.to(config.device)
            eval_result, reranked_result = evaluate_model(
                model,
                dataloader['test_loader'],
                config,
                logger,
            )
            log_evaluation_results(logger, eval_result, reranked_result)
            torch.cuda.empty_cache()
            return {
                'original': eval_result,
                'reranked': reranked_result,
            }

        model.to(config.device)
        model, load_result = load_checkpoint(model, config)
        validate_checkpoint_mapping(load_result, config)

        if is_using_distributed():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.device],
                find_unused_parameters=True,
            )

        model_without_ddp = model.module if hasattr(model, "module") else model
        model_ema = None
        if config.experiment.use_ema and is_master():
            model_ema = ModelEMA(
                model_without_ddp,
                config.experiment.ema_decay,
            )
            if config.model.ckpt_type == 'saved':
                saved_checkpoint = torch.load(
                    os.path.join(
                        config.model.saved_path,
                        'checkpoint_best.pth',
                    ),
                    map_location='cpu',
                    weights_only=False,
                )
                if 'model_ema' in saved_checkpoint:
                    model_ema.load_state_dict(saved_checkpoint['model_ema'])
                    logger.info("Restored model EMA from saved checkpoint")
                else:
                    logger.warning(
                        "Saved checkpoint has no model_ema; initialized EMA "
                        "from the restored model weights"
                    )
                del saved_checkpoint

        config.schedule.niter_per_ep = len(train_loader)
        lr_schedule = cosine_scheduler(config)
        optimizer = build_optimizer(config, model)

        it = 0
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(config.schedule.epoch):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if is_using_distributed():
                dataloader['train_sampler'].set_epoch(epoch)

            start_time = time.time()
            for meter in meters.values():
                meter.reset()
            model.train()

            for i, batch in enumerate(train_loader):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[it] * param_group['ratio']

                if epoch == 0:
                    alpha = config.model.softlabel_ratio * min(1.0, i / len(train_loader))
                else:
                    alpha = config.model.softlabel_ratio

                with torch.autocast(device_type='cuda'):
                    if config.experiment.nrw_mode == 'off':
                        ret = model(batch, alpha, training=True)
                    else:
                        ret = model(batch, alpha, training=True, epoch=epoch)
                    loss = sum([v for k, v in ret.items() if "loss" in k])

                batch_size = batch['image'].shape[0]
                meters['loss'].update(loss.item(), batch_size)
                meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
                meters['itc_tse_loss'].update(ret.get('itc_tse_loss', 0), batch_size)
                meters['sdm_loss'].update(
                    ret['sdm_loss'].item() if 'sdm_loss' in ret else 0,
                    batch_size,
                )
                meters['tal_loss'].update(
                    ret['tal_loss'].item() if 'tal_loss' in ret else 0,
                    batch_size,
                )
                meters['view_loss'].update(ret.get('view_loss', 0), batch_size)
                meters['ortho_loss'].update(ret.get('ortho_loss', 0), batch_size)
                meters['view_acc'].update(ret.get('view_acc', 0), batch_size)
                meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
                meters['ritc_tse_loss'].update(ret.get('ritc_tse_loss', 0), batch_size)
                meters['balance_loss'].update(ret.get('balance_loss', 0), batch_size)
                meters['router_z_loss'].update(ret.get('router_z_loss', 0), batch_size)
                meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
                meters['bridge_cv_loss'].update(ret.get('bridge_cv_loss', 0), batch_size)
                meters['bridge_tg_loss'].update(ret.get('bridge_tg_loss', 0), batch_size)
                if config.experiment.dhn_margin > 0:
                    meters['dhn_hard_ratio'].update(
                        ret['dhn_hard_ratio'].item(),
                        batch_size,
                    )
                if config.experiment.nrw_mode != 'off':
                    meters['nrw_w_mean_aerial'].update(ret['nrw_w_mean_aerial'], batch_size)
                    meters['nrw_w_mean_ground'].update(ret['nrw_w_mean_ground'], batch_size)
                    meters['nrw_mu_aerial'].update(ret['nrw_mu_aerial'], batch_size)
                    meters['nrw_mu_ground'].update(ret['nrw_mu_ground'], batch_size)

                scaler.scale(loss).backward()
                if model_ema is None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scale_before_step = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_succeeded = (
                        scaler.get_scale() >= scale_before_step
                    )
                    model_ema.update(
                        model_without_ddp,
                        update_parameters=optimizer_step_succeeded,
                    )
                model.zero_grad()
                optimizer.zero_grad()
                it += 1

                if (i + 1) % config.log.print_period == 0:
                    info_str = f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"
                    for key, meter in meters.items():
                        if meter.val != 0:
                            info_str += f", {key}: {meter.val:.4f}"
                    info_str += f", Base Lr: {param_group['lr']:.2e}"
                    logger.info(info_str)

            if is_master():
                end_time = time.time()
                time_per_batch = (end_time - start_time) / (i + 1)
                logger.info(
                    "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                        epoch + 1,
                        time_per_batch,
                        train_loader.batch_size / time_per_batch,
                    )
                )

                ema_context = (
                    model_ema.average_parameters(model_without_ddp)
                    if model_ema is not None
                    else nullcontext()
                )
                with ema_context:
                    eval_result, reranked_result = evaluate_model(
                        model_without_ddp,
                        dataloader['test_loader'],
                        config,
                        logger,
                    )
                rank_1, rank_5, rank_10, map = (
                    eval_result['r1'],
                    eval_result['r5'],
                    eval_result['r10'],
                    eval_result['mAP'],
                )
                log_evaluation_results(logger, eval_result, reranked_result)
                torch.cuda.empty_cache()

            epoch_peak_alloc_gb, epoch_peak_reserved_gb = _cuda_peak_memory_gb()
            full_peak_alloc_gb = max(full_peak_alloc_gb, epoch_peak_alloc_gb)
            full_peak_reserved_gb = max(
                full_peak_reserved_gb,
                epoch_peak_reserved_gb,
            )
            completed_epochs = epoch + 1
            logger.info(
                "Epoch %s peak CUDA memory: allocated=%.3f GiB, reserved=%.3f GiB",
                epoch + 1,
                epoch_peak_alloc_gb,
                epoch_peak_reserved_gb,
            )

            if is_master():
                epoch_time_sec = time.time() - start_time
                if experiment is not None:
                    append_metrics(
                        experiment,
                        epoch + 1,
                        eval_result,
                        meters['loss'].avg,
                        epoch_time_sec,
                        epoch_peak_alloc_gb,
                        epoch_peak_reserved_gb,
                        reranked_result,
                    )
                if best_metrics is None or best_metrics['r1'] < rank_1:
                    best_metrics = {
                        'r1': rank_1,
                        'r5': rank_5,
                        'r10': rank_10,
                        'mAP': map,
                    }
                    best_metrics_epoch = epoch
                if best_rank_1 < rank_1:
                    best_rank_1 = rank_1
                    best_epoch = epoch
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                    }
                    if model_ema is not None:
                        save_obj['model_ema'] = model_ema.state_dict()
                    torch.save(
                        save_obj,
                        os.path.join(config.model.saved_path, 'checkpoint_best.pth'),
                    )
                    del save_obj

    except torch.cuda.OutOfMemoryError:
        current_alloc_gb, current_reserved_gb = _cuda_peak_memory_gb()
        full_peak_alloc_gb = max(full_peak_alloc_gb, current_alloc_gb)
        full_peak_reserved_gb = max(full_peak_reserved_gb, current_reserved_gb)
        actual_batch_size = (
            train_loader.batch_size
            if train_loader is not None
            else config.data.batch_size
        )
        logger.exception(
            "CUDA OOM: batch_size=%s, completed_epochs=%s, "
            "peak_mem_alloc_gb=%.3f, peak_mem_reserved_gb=%.3f",
            actual_batch_size,
            completed_epochs,
            full_peak_alloc_gb,
            full_peak_reserved_gb,
        )
        if is_master() and experiment is not None:
            experiment.write_final_results(
                best_metrics_epoch + 1 if best_metrics_epoch is not None else None,
                best_metrics,
                time.time() - training_start_time,
                full_peak_alloc_gb,
                full_peak_reserved_gb,
                completed_epochs,
                run_status="cuda_oom",
            )
        raise

    total_training_time_sec = time.time() - training_start_time
    if is_master() and experiment is not None:
        experiment.write_final_results(
            best_metrics_epoch + 1 if best_metrics_epoch is not None else None,
            best_metrics,
            total_training_time_sec,
            full_peak_alloc_gb,
            full_peak_reserved_gb,
            completed_epochs,
        )
    print(f"best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval_only and args.resume is None:
        parser.error("--eval_only requires --resume PATH")
    if args.resume is not None and not args.eval_only:
        parser.error("--resume is currently supported only with --eval_only")
    config, overrides = resolve_runtime_config(args)
    if config.experiment.dhn_margin < 0:
        parser.error("dhn_margin must be non-negative")
    if config.experiment.rerank:
        if config.experiment.rerank_k1 < 1:
            parser.error("rerank_k1 must be positive")
        if not 1 <= config.experiment.rerank_k2 <= config.experiment.rerank_k1 + 1:
            parser.error("rerank_k2 must be in [1, rerank_k1 + 1]")
        if not 0.0 <= config.experiment.rerank_lambda <= 1.0:
            parser.error("rerank_lambda must be in [0, 1]")

    try:
        experiment_directory, path_override = configure_experiment(config, args)
    except ExperimentDirectoryConflict as exc:
        parser.error(str(exc))
    if path_override is not None:
        overrides.append(path_override)

    if args.dry_run:
        rank = environment_rank()
        logger = setup_logger('TAG', distributed_rank=rank)
        logger.propagate = False
        experiment = ExperimentArchive(
            experiment_directory,
            enabled=rank == 0,
        )
        experiment.write_pre_run(config, args.config)
        if experiment.enabled:
            add_file_handler(logger, experiment.directory)
        log_cli_overrides(logger, overrides, config)
        logger.info(f'\n{config}')
        logger.info("Dry run complete; training was not started.")
        return 0

    # Match the baseline import-before-seeding order for actual training runs.
    import model.tbps_model_MoE as tbps_model_module

    init_distributed_mode(config)
    configure_backbone = getattr(
        tbps_model_module,
        'configure_backbone_from_checkpoint',
        None,
    )
    if configure_backbone is not None:
        configure_backbone(
            config,
            checkpoint_path=args.resume if args.eval_only else None,
        )
    set_seed(config)
    archive_eval_only = (
        args.output_dir is not None or args.exp_name is not None
    )
    experiment = ExperimentArchive(
        experiment_directory,
        enabled=is_master() and (not args.eval_only or archive_eval_only),
    )
    run(
        config,
        experiment=experiment,
        cli_overrides=overrides,
        config_path=args.config,
        eval_only=args.eval_only,
        resume_path=args.resume,
    )
    return 0


if __name__ == '__main__':
    main()

import argparse
import os
import random
import time
from pathlib import Path

import torch
from experiment import (
    ExperimentArchive,
    ExperimentDirectoryConflict,
    add_file_handler,
    prepare_experiment_directory,
    resolve_experiment_directory,
)
from misc.utils import get_rank
from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import test_tse, test
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
        (args.batch_size, config.data, "batch_size", "data.batch_size"),
        (
            args.test_batch_size,
            config.data,
            "test_batch_size",
            "data.test_batch_size",
        ),
        (args.epoch, config.schedule, "epoch", "schedule.epoch"),
        (args.seed, config.misc, "seed", "misc.seed"),
        (args.lr, config.schedule, "lr", "schedule.lr"),
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
            args.epoch,
            args.seed,
            args.lr,
        )
    ) or any(
        (args.grad_checkpoint, args.no_grad_checkpoint, args.dry_run)
    ) or args.config != "config/config.yaml"
    directory = prepare_experiment_directory(directory, protect_checkpoint)
    return directory, path_override


def log_cli_overrides(logger, overrides, config):
    logger.info("OVERRIDE SUMMARY")
    if not overrides:
        logger.info("no overrides")
    else:
        for key, old_value, new_value in overrides:
            logger.info("%s: %r -> %r", key, old_value, new_value)
    logger.info(
        "experiment.grad_checkpoint effective: %r",
        config.experiment.grad_checkpoint,
    )
    logger.info("END OVERRIDE SUMMARY")


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


def run(config, experiment=None, cli_overrides=(), config_path=None):
    logger = setup_logger('TAG', distributed_rank=get_rank())
    logger.propagate = False
    if experiment is not None:
        experiment.write_pre_run(config, config_path)
    if experiment is not None and experiment.enabled:
        add_file_handler(logger, experiment.directory)
    log_cli_overrides(logger, cli_overrides, config)
    logger.info(f'\n{config}')
    if experiment is not None:
        experiment.initialize_metrics()

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
            "ortho_loss": AverageMeter(),
            "ritc_loss": AverageMeter(),
            "ritc_tse_loss": AverageMeter(),
            "balance_loss": AverageMeter(),
            "router_z_loss": AverageMeter(),
            "id_loss": AverageMeter(),
        }

        # model
        # model_name = "EVA02-CLIP-B-16"
        # pretrained = "/data/zxy/UAV/checkpoints/EVA02_CLIP_B_psz16_s8B.pt"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"
        # model, _, preprocess = create_model_and_transforms(config, model_name, pretrained, force_custom_clip=True)
        model = clip_vitb(config, num_classes)
        model.to(config.device)
        model, load_result = load_checkpoint(model, config)

        if is_using_distributed():
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.device],
                find_unused_parameters=True,
            )

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
                    ret = model(batch, alpha, training=True)
                    loss = sum([v for k, v in ret.items() if "loss" in k])

                batch_size = batch['image'].shape[0]
                meters['loss'].update(loss.item(), batch_size)
                meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
                meters['itc_tse_loss'].update(ret.get('itc_tse_loss', 0), batch_size)
                meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
                meters['view_loss'].update(ret.get('view_loss', 0), batch_size)
                meters['ortho_loss'].update(ret.get('ortho_loss', 0), batch_size)
                meters['view_acc'].update(ret.get('view_acc', 0), batch_size)
                meters['ritc_loss'].update(ret.get('ritc_loss', 0), batch_size)
                meters['ritc_tse_loss'].update(ret.get('ritc_tse_loss', 0), batch_size)
                meters['balance_loss'].update(ret.get('balance_loss', 0), batch_size)
                meters['router_z_loss'].update(ret.get('router_z_loss', 0), batch_size)
                meters['id_loss'].update(ret.get('id_loss', 0), batch_size)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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

                model_without_ddp = model.module if hasattr(model, "module") else model
                eval_result = test_tse(
                    model_without_ddp,
                    dataloader['test_loader'],
                    77,
                    config.device,
                )
                rank_1, rank_5, rank_10, map = (
                    eval_result['r1'],
                    eval_result['r5'],
                    eval_result['r10'],
                    eval_result['mAP'],
                )
                logger.info(
                    'Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'.format(
                        top1=rank_1,
                        top5=rank_5,
                        top10=rank_10,
                        mAP=map,
                    )
                )
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
                    experiment.append_metrics(
                        epoch + 1,
                        rank_1,
                        rank_5,
                        rank_10,
                        map,
                        meters['loss'].avg,
                        epoch_time_sec,
                        epoch_peak_alloc_gb,
                        epoch_peak_reserved_gb,
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
                    model_without_ddp = (
                        model.module if hasattr(model, "module") else model
                    )
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                    }
                    torch.save(
                        save_obj,
                        os.path.join(config.model.saved_path, 'checkpoint_best.pth'),
                    )

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
    config, overrides = resolve_runtime_config(args)

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
    import model.tbps_model_MoE

    init_distributed_mode(config)
    set_seed(config)
    experiment = ExperimentArchive(
        experiment_directory,
        enabled=is_master(),
    )
    run(
        config,
        experiment=experiment,
        cli_overrides=overrides,
        config_path=args.config,
    )
    return 0


if __name__ == '__main__':
    main()

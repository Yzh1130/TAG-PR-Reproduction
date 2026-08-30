"""Measure orthogonality-loss gradients without taking optimization steps.

Run from the repository root with the same config used for the checkpoint.
This standalone process reuses the training DataLoader (two workers), model
factory, checkpoint loader, loss sum, and CUDA autocast. It never calls main.run.
"""

import argparse
from contextlib import contextmanager, nullcontext
import json
import logging
import math
import os
import pickle
from pathlib import Path
import statistics
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
ORTHO_ALPHA = 0.001


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", help="trusted checkpoint containing model weights (not EMA)")
    parser.add_argument("--config", help="training config YAML, with local dataset paths")
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--device", default="cuda:0", help="single CUDA device (default: cuda:0)")
    parser.add_argument("--batch_size", type=int, default=None, help="optional probe-only batch size")
    parser.add_argument(
        "--loss_scale", type=float, default=65536.0,
        help="fixed scale for both AMP backwards; norms are unscaled (default: 65536, training GradScaler initial value)",
    )
    parser.add_argument(
        "--epoch", type=int, default=1,
        help="zero-based epoch for sampler/NRW/softlabel warmup; default 1 (checkpoint has no epoch)",
    )
    parser.add_argument("--out", help="optional JSON report path")
    parser.add_argument("--self_test", action="store_true", help="run CPU mock tests; no checkpoint required")
    return parser


class OrthoCapture:
    """Capture the exact CLS/view inputs used by each MoE orthogonal loss.

    The first ln_2 call per layer belongs to the main image branch. Later
    bridge-image forwards and backward checkpoint recomputations are ignored.
    """

    def __init__(self, model, moe_type=None):
        if moe_type is None:
            from model.base_transformer import ResidualAttentionBlockMoE
            moe_type = ResidualAttentionBlockMoE

        self.active = False
        self.values = {}
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, moe_type) and module.compute_ortho_loss:
                self.handles.append(module.ln_2.register_forward_hook(self._hook(name)))
        if not self.handles:
            raise ValueError("No enabled MoE orthogonal-loss blocks; check disable_moe/expert_blocks/ortho_ratio")

    def _hook(self, name):
        def capture(_module, _inputs, output):
            if not self.active or name in self.values:
                return
            import torch
            import torch.nn.functional as F

            if output.ndim != 3 or output.shape[0] < 2:
                raise RuntimeError(f"Unexpected ln_2 shape in {name}: {tuple(output.shape)}")
            # ln_2 emits [sequence,batch,width]; forward_experts permutes it.
            tokens = output.detach().permute(1, 0, 2)
            cos = torch.cosine_similarity(
                F.normalize(tokens[:, 0, :], dim=-1),
                F.normalize(tokens[:, -1, :]), dim=-1,
            ).abs()
            self.values[name] = cos.float().cpu()
        return capture

    def start(self):
        self.values.clear()
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.stop()
        for handle in self.handles:
            handle.remove()


def gradient_norm(model):
    """L2 norm over all parameter gradients; disconnected parameters add zero."""
    import torch

    squares = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            norm = parameter.grad.detach().float().norm()
            squares.append(norm.double().square())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def measure_batch(model, batch, capture, alpha, epoch, amp=True, nrw=False, loss_scale=65536.0):
    """Two independent gradient accumulations from one identical forward graph."""
    import torch

    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss_scale must be positive and finite")
    model.zero_grad(set_to_none=True)
    capture.start()
    try:
        with torch.autocast(device_type="cuda") if amp else nullcontext():
            if nrw:
                ret = model(batch, alpha, training=True, epoch=epoch)
            else:
                ret = model(batch, alpha, training=True)
            total = sum(value for key, value in ret.items() if "loss" in key)
            ortho = ret.get("ortho_loss")
    finally:
        capture.stop()
    if not torch.is_tensor(ortho) or not ortho.requires_grad:
        raise RuntimeError("ortho_loss is absent or not differentiable; enable ortho_ratio and MoE")
    if not torch.is_tensor(total) or not torch.isfinite(total).all() or not torch.isfinite(ortho).all():
        raise RuntimeError("Non-finite or missing total/orthogonal loss")

    # Match training's AMP loss scaling to avoid FP16 gradient underflow. A
    # fixed shared scale makes both measurements comparable without an optimizer.
    (ortho.float() * loss_scale).backward(retain_graph=True)
    g_ortho = gradient_norm(model) / loss_scale
    model.zero_grad(set_to_none=True)
    if not math.isfinite(g_ortho):
        raise RuntimeError("Non-finite orthogonal gradients; lower --loss_scale and rerun (no automatic fallback)")
    (total.float() * loss_scale).backward()
    g_total = gradient_norm(model) / loss_scale
    model.zero_grad(set_to_none=True)
    if not math.isfinite(g_total):
        raise RuntimeError("Non-finite total gradients; lower --loss_scale and rerun (no automatic fallback)")
    if g_total <= 0:
        raise RuntimeError(f"Invalid zero total gradient norm: ortho={g_ortho}, total={g_total}")
    if len(capture.values) != len(capture.handles):
        raise RuntimeError("Not all orthogonal-loss layers ran in the main forward")
    all_cos = torch.cat(list(capture.values.values()))
    return {
        "ortho_loss": float(ortho.detach().item()),
        "total_loss": float(total.detach().item()),
        "g_ortho": g_ortho,
        "g_total": g_total,
        "gradient_ratio": g_ortho / g_total,
        "loss_scale": loss_scale,
        "mean_abs_cos": float(all_cos.mean().item()),
        "fraction_abs_cos_ge_alpha": float((all_cos >= ORTHO_ALPHA).float().mean().item()),
        "fraction_zero_clamped": float((all_cos <= ORTHO_ALPHA).float().mean().item()),
        "layer_sample_count": int(all_cos.numel()),
        "layers": {
            name: {
                "mean_abs_cos": float(values.mean().item()),
                "fraction_abs_cos_ge_alpha": float((values >= ORTHO_ALPHA).float().mean().item()),
                "fraction_zero_clamped": float((values <= ORTHO_ALPHA).float().mean().item()),
            }
            for name, values in capture.values.items()
        },
    }


def summarize(records):
    ratios = [row["gradient_ratio"] for row in records]
    samples = sum(row["layer_sample_count"] for row in records)
    return {
        "num_batches": len(records),
        "gradient_ratio_mean": statistics.mean(ratios),
        "gradient_ratio_std": statistics.pstdev(ratios),
        "std_convention": "population (ddof=0), across batches",
        **{
            key: sum(row[key] * row["layer_sample_count"] for row in records) / samples
            for key in ("mean_abs_cos", "fraction_abs_cos_ge_alpha", "fraction_zero_clamped")
        },
    }


@contextmanager
def single_process_group(device):
    """Keep the repository's distributed sampler and all_gather unmodified."""
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This probe is single-GPU; launch python directly, not multi-rank torchrun")
    old_environment = {key: os.environ.get(key) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
    created = False
    try:
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK=str(device.index or 0))
        if dist.is_initialized():
            if dist.get_world_size() != 1:
                raise ValueError("Existing distributed process group must have world_size=1")
            yield
        else:
            backend = "nccl" if dist.is_nccl_available() else "gloo"
            with tempfile.TemporaryDirectory(prefix="tagpr-ortho-dist-") as directory:
                rendezvous = (Path(directory) / "rendezvous").as_uri()
                dist.init_process_group(backend=backend, init_method=rendezvous, rank=0, world_size=1)
                created = True
                try:
                    yield
                finally:
                    dist.destroy_process_group()
                    created = False
    finally:
        if created:
            dist.destroy_process_group()
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_probe(args):
    import torch

    resume = Path(args.resume).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    if not resume.is_file():
        raise ValueError(f"Checkpoint does not exist: {resume}")
    if not config_path.is_file():
        raise ValueError(f"Config does not exist: {config_path}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Real probe requires CUDA; use --self_test for a CPU mock")
    torch.cuda.set_device(device)
    import main as training_main
    from misc.utils import set_seed
    from model.tbps_model_MoE import clip_vitb, configure_backbone_from_checkpoint

    config_args = training_main.build_parser().parse_args(["--config", str(config_path)])
    config, _ = training_main.resolve_runtime_config(config_args)
    if config.experiment.disable_moe or config.experiment.ortho_ratio <= 0:
        raise ValueError("Probe requires disable_moe=false and ortho_ratio>0 in the supplied config")
    config.device = str(device)
    config.data.num_workers = 2
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    logger = logging.getLogger("probe_ortho")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    with single_process_group(device):
        configure_backbone_from_checkpoint(config, checkpoint_path=resume)
        set_seed(config)
        data = training_main.build_pedes_data(config)
        loader = data["train_loader"]
        if loader.num_workers != 2:
            raise RuntimeError(f"Expected num_workers=2, got {loader.num_workers}")
        if len(loader) < args.num_batches:
            raise ValueError(f"Only {len(loader)} complete batches are available; requested {args.num_batches}")
        if data["train_sampler"] is not None:
            data["train_sampler"].set_epoch(args.epoch)
        model = clip_vitb(config, len(loader.dataset.person2text))
        training_main.load_resume_weights(model, str(resume), False, logger)
        model.to(device)
        model.train()
        capture = OrthoCapture(model)
        records = []
        print("ORTHO PROBE: original model weights, no EMA, no optimizer, world_size=1, workers=2")
        print(f"epoch={args.epoch} (zero-based); ortho_ratio={config.experiment.ortho_ratio}; alpha={ORTHO_ALPHA}")
        print("Source loss: 100 * mean(clamp(abs(cos)-0.001, min=0)); >=alpha is the active side, NOT an upper saturation.")
        print("Reported g_ortho uses the weighted ret['ortho_loss']; g_total includes every returned '*loss*' term.")
        print(f"Both backwards use loss_scale={args.loss_scale:g}; reported gradient norms are divided by this scale.")
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for index, batch in enumerate(loader):
                # DDP normally performs this input transfer; no DDP reducer is
                # needed for two backwards in this single-process diagnostic.
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                softlabel_alpha = config.model.softlabel_ratio
                if args.epoch == 0:
                    softlabel_alpha *= min(1.0, index / len(loader))
                row = measure_batch(
                    model, batch, capture, softlabel_alpha, args.epoch,
                    nrw=config.experiment.nrw_mode != "off",
                    loss_scale=args.loss_scale,
                )
                row["batch"] = index + 1
                records.append(row)
                print(f"batch={index + 1:02d} g_ortho={row['g_ortho']:.8g} g_total={row['g_total']:.8g} ratio={row['gradient_ratio']:.8g} mean|cos|={row['mean_abs_cos']:.8g} >=alpha={row['fraction_abs_cos_ge_alpha']:.6f}")
                if len(records) == args.num_batches:
                    break
        finally:
            capture.close()
        torch.cuda.synchronize(device)
        return {
            "checkpoint": str(resume), "config": str(config_path),
            "device": str(device), "batch_size": loader.batch_size, "num_workers": loader.num_workers,
            "epoch_zero_based": args.epoch, "model_weights": "model (not model_ema)",
            "ortho_ratio": config.experiment.ortho_ratio, "alpha": ORTHO_ALPHA,
            "loss_scale": args.loss_scale,
            "loss_formula": "100 * mean(clamp(abs(cos)-alpha,min=0)); then ortho_ratio",
            "branch_note": ">=alpha is requested active-side occupancy; <=alpha is actual zero-clamped region",
            "gradient_note": "all model parameters, same fixed loss scale for both backwards under training CUDA autocast; reported norms divided by loss_scale; nonfinite gradients error without fallback; no optimizer step",
            "buffer_policy": "training-mode BN/NRW buffers evolve once per batch in this standalone process; no parameters or checkpoint files are updated",
            "elapsed_sec": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "summary": summarize(records), "batches": records,
        }


def self_test():
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as activation_checkpoint

    class MockBlock(nn.Module):
        """Same token layout/orthogonal formula, without full model dependencies."""

        def __init__(self):
            super().__init__()
            self.compute_ortho_loss = True
            self.projection = nn.Linear(8, 8)
            self.ln_2 = nn.LayerNorm(8)
            self.router = nn.Linear(8, 2)

        def forward(self, tokens, cam_id):
            output = self.ln_2(self.projection(tokens))
            x = output.permute(1, 0, 2)
            cos = torch.cosine_similarity(
                F.normalize(x[:, 0, :], dim=-1), F.normalize(x[:, -1, :]), dim=-1,
            ).abs()
            ortho = (cos - ORTHO_ALPHA).clamp(min=0).mean() * 100
            view = F.cross_entropy(self.router(x[:, -1, :]), cam_id)
            return output, view, ortho

    class MockModel(nn.Module):
        def __init__(self, checkpoint):
            super().__init__()
            self.block = MockBlock()
            self.checkpoint = checkpoint

        def forward(self, batch, alpha, training=False, epoch=0):
            if self.checkpoint:
                output, view, ortho = activation_checkpoint(
                    self.block, batch["tokens"], batch["cam_id"], use_reentrant=False,
                )
            else:
                output, view, ortho = self.block(batch["tokens"], batch["cam_id"])
            return {"ortho_loss": ortho, "view_loss": view, "mock_task_loss": output.square().mean()}

    torch.set_num_threads(1)
    results = []
    for checkpoint, loss_scale in ((enabled, scale) for enabled in (False, True) for scale in (1.0, 65536.0)):
        torch.manual_seed(2026)
        model = MockModel(checkpoint).train()
        batch = {"tokens": torch.randn(5, 4, 8), "cam_id": torch.tensor([0, 1, 0, 1])}
        saved = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        capture = OrthoCapture(model, moe_type=MockBlock)
        try:
            torch.manual_seed(17)
            row = measure_batch(model, batch, capture, 0.5, 1, amp=False, loss_scale=loss_scale)
            assert len(capture.values) == 1 and row["layer_sample_count"] == 4
            assert all(torch.equal(saved[name], parameter) for name, parameter in model.named_parameters())
            assert all(parameter.grad is None for parameter in model.parameters())
            # Independent reference backward verifies zero_grad separated the
            # total gradient from the first orthogonal gradient accumulation.
            torch.manual_seed(17)
            ret = model(batch, 0.5, training=True)
            sum(ret.values()).backward()
            error = abs(gradient_norm(model) - row["g_total"])
            assert error < 1e-7, error
            model.zero_grad(set_to_none=True)
            torch.manual_seed(17)
            model(batch, 0.5, training=True)["ortho_loss"].backward()
            ortho_error = abs(gradient_norm(model) - row["g_ortho"])
            assert ortho_error < 1e-7, ortho_error
            model.zero_grad(set_to_none=True)
            expected_loss = 100 * (capture.values["block"] - ORTHO_ALPHA).clamp(min=0).mean()
            assert abs(expected_loss.item() - row["ortho_loss"]) < 1e-5
            results.append({"checkpoint": checkpoint, "total_gradient_norm_error": error, "ortho_gradient_norm_error": ortho_error, **row})
            print(f"checkpoint={checkpoint} loss_scale={loss_scale:g} g_ortho={row['g_ortho']:.14g} g_total={row['g_total']:.14g} reference_max_error={max(error, ortho_error):.8g}")
        finally:
            capture.close()
    assert all(abs(row["gradient_ratio"] - results[0]["gradient_ratio"]) < 1e-7 for row in results)
    # Exercise the fail-fast policy rather than silently lowering the scale.
    overflow_hook = model.block.projection.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    capture = OrthoCapture(model, moe_type=MockBlock)
    try:
        try:
            measure_batch(model, batch, capture, 0.5, 1, amp=False, loss_scale=65536.0)
        except RuntimeError as exc:
            assert "lower --loss_scale" in str(exc), exc
            assert all(parameter.grad is None for parameter in model.parameters())
        else:
            raise AssertionError("Nonfinite gradients must error without fallback")
    finally:
        capture.close()
        overflow_hook.remove()
    print("SELF_TEST PASS: torch-only mock block with source ortho formula; checkpoint off/on x loss_scale 1/65536; independent unscaled backward errors=0; parameters unchanged; hooks count=4 (not recomputations); nonfinite gradients fail fast")
    return {"self_test": True, "nonfinite_gradient_rejected": True, "summary": summarize(results), "batches": results}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_batches < 1 or args.epoch < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("num_batches/batch_size must be positive and epoch must be nonnegative")
    if not math.isfinite(args.loss_scale) or args.loss_scale <= 0:
        parser.error("loss_scale must be positive and finite")
    if not args.self_test and (not args.resume or not args.config):
        parser.error("--resume and --config are required (unless --self_test)")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        report = self_test() if args.self_test else run_probe(args)
        summary = report["summary"]
        print(f"g_ortho/g_total: {summary['gradient_ratio_mean']:.10g} +/- {summary['gradient_ratio_std']:.10g}")
        print(f"mean|cos|: {summary['mean_abs_cos']:.10g}; |cos|>=alpha: {summary['fraction_abs_cos_ge_alpha']:.6%}; zero-clamped: {summary['fraction_zero_clamped']:.6%}")
        if args.out:
            output = Path(args.out).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Wrote {output}")
        return 0
    except ImportError as exc:
        parser.error(f"Missing runtime dependency: {exc}. Use the same Python environment as training.")
    except (ValueError, RuntimeError, OSError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

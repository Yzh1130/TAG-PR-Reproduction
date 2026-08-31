"""S12: query-bank CSLS/NNN, IS, DIS and DualIS on immutable archived scores.

Only the feature-extraction stage needs PyTorch/the repository GPU environment.
The correction scan is NumPy-only and never uses labels to estimate penalties.
Metrics use test identities solely after all calibration statistics are fixed.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

try:
    from ._offline_common import evaluate, load_annotations, load_similarity, safe_output_dir, json_ready
    from ._s12_features import (
        _read_records, assert_identical_features, discover_gallery_features, discover_train_annotations,
        extract_gallery_features, extract_text_features, load_model, preprocess_texts,
        resolve_checkpoint, save_training_features, text_fingerprint, verify_query_gallery,
        write_bank_similarity,
    )
except ImportError:
    from _offline_common import evaluate, load_annotations, load_similarity, safe_output_dir, json_ready
    from _s12_features import (
        _read_records, assert_identical_features, discover_gallery_features, discover_train_annotations,
        extract_gallery_features, extract_text_features, load_model, preprocess_texts,
        resolve_checkpoint, save_training_features, text_fingerprint, verify_query_gallery,
        write_bank_similarity,
    )


DEFAULT_ROOT = Path("/root/autodl-tmp")
DEFAULT_DUMP = DEFAULT_ROOT / "experiments/_dump/20260830_dump_fk32"
METHOD_SOURCES = {
    "DBNorm": "https://aclanthology.org/2023.emnlp-main.652/",
    "DIS": "https://github.com/ioanacroi/qb-norm/blob/master/dynamic_inverted_softmax.py",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", default=str(DEFAULT_DUMP / "sim_qg.npy"), help="read-only float32 query-gallery scores")
    parser.add_argument("--sim-gg", default=None, help="read-only gallery-gallery scores; default sibling sim_gg.npy")
    parser.add_argument("--anno", default=str(DEFAULT_ROOT / "datasets/TAG-PEDES/test_reid.json"))
    parser.add_argument("--train-anno", default=None, help="optional explicit annotation after autodiscovery ambiguity")
    parser.add_argument("--out", default=str(DEFAULT_ROOT / "experiments/_e1"), help="new output directory; never an archive")
    parser.add_argument("--tag", default="fk32", help="tag included in report/log/bank-matrix filenames")
    parser.add_argument("--resume", default=str(DEFAULT_ROOT / "experiments/20260827_l14_final_keep"), help="trusted checkpoint or directory containing checkpoint_best.pth")
    parser.add_argument("--config", default=str(DEFAULT_ROOT / "cfg_l14.yaml"))
    parser.add_argument("--backbone", default="ViT-L/14", choices=("ViT-B/16", "ViT-L/14"))
    parser.add_argument("--repo", default=None, help="repository root when standalone scripts are copied outside the checkout")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=None, type=int, help="single extraction seed; default config.misc.seed, explicitly recorded; never reset between text repetitions")
    parser.add_argument("--text-batch-size", default=256, type=int, help="online text evaluation batch size (256)")
    parser.add_argument("--image-batch-size", default=32, type=int, help="match archived evaluation's ACTUAL image loader batch size (baseline 32)")
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--image-root", default=None, help="default config.image_dir")
    parser.add_argument("--gallery-features", default=None, help="optional read-only npz with image_feat/image_tse_feat; always validated against archived qg")
    parser.add_argument("--score-block-size", default=256, type=int, help="FP32 text-gallery GEMM row block size")
    parser.add_argument("--self-test", action="store_true", help="run 200x100 NumPy mock scan without checkpoint/CUDA")
    return parser


def _validate_matrix(values, name, columns=None):
    if values.ndim != 2 or min(values.shape) < 1 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name}: expected a nonempty 2D floating matrix, got {values.shape}/{values.dtype}")
    if columns is not None and values.shape[1] != columns:
        raise ValueError(f"{name}: expected {columns} gallery columns, got {values.shape[1]}")
    for start in range(0, len(values), 256):
        block = values[start:start + 256]
        if not np.isfinite(block).all():
            raise ValueError(f"{name}: non-finite scores near row {start}")
        if block.min() < -1.0001 or block.max() > 1.0001:
            raise ValueError(f"{name}: scores outside cosine-mixture range [-1,1]")


def bank_statistics(bank, ks, betas, rows=None, block_size=256):
    """One bank pass; bounded top-k, stable float64 streaming log-sum-exp.

    A bank's self rows remain eligible in DualIS, matching the softmax-bank
    definition. Self exclusion is specific to gallery-CSLS, not DualIS.
    """
    if block_size < 1:
        raise ValueError("Bank block size must be positive")
    rows = np.arange(len(bank), dtype=np.int64) if rows is None else np.sort(np.asarray(rows, dtype=np.int64))
    if rows.ndim != 1 or len(rows) == 0 or len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= len(bank):
        raise ValueError("Bank row selection must contain unique valid indices")
    ks, betas = tuple(ks), tuple(betas)
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 1 or k > len(rows) for k in ks):
        raise ValueError(f"top-k values {ks} must be integers in [1,{len(rows)}]")
    if not betas or any(not np.isfinite(beta) or beta <= 0 for beta in betas):
        raise ValueError("IS beta values must be positive and finite")
    columns, maximum_k = bank.shape[1], max(ks)
    keep = np.full((maximum_k, columns), -np.inf, dtype=np.float32)
    lse = {beta: np.full(columns, -np.inf, dtype=np.float64) for beta in betas}
    hubs = np.zeros(columns, dtype=bool)
    for start in range(0, len(rows), block_size):
        block = np.asarray(bank[rows[start:start + block_size]], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Bank contains non-finite values")
        hubs[np.argmax(block, axis=1)] = True
        merged = np.concatenate((keep, block), axis=0)
        keep = np.partition(merged, len(merged) - maximum_k, axis=0)[-maximum_k:]
        for beta in betas:
            scaled = block.astype(np.float64) * beta
            maximum = scaled.max(axis=0)
            block_lse = maximum + np.log(np.exp(scaled - maximum).sum(axis=0))
            lse[beta] = np.logaddexp(lse[beta], block_lse)
    keep = np.sort(keep, axis=0)[::-1]
    return {
        "topk": {k: keep[:k].mean(axis=0, dtype=np.float32) for k in ks},
        "is": {beta: lse[beta] / beta for beta in betas},
        "hubs": hubs,
        "rows": rows,
    }


def _metric_summary(result):
    return {key: result[key] for key in ("R1", "R5", "R10", "mAP", "n_query") if key in result}


def run_corrections(sim, train_bank, sim_gg, ann, subset_sizes=(1000, 5000, 16504),
                    seeds=(2026, 2027, 2028), ks=(5, 10, 16, 20, 30),
                    alphas=(0.25, 0.5, 0.75, 1.0), betas=(5, 10, 20, 50), progress=print):
    """CPU correction engine, independent of model, training and GPU imports."""
    _validate_matrix(sim, "query-gallery")
    _validate_matrix(train_bank, "training bank-gallery", sim.shape[1])
    _validate_matrix(sim_gg, "gallery-gallery", sim.shape[1])
    if sim_gg.shape[0] != sim.shape[1]:
        raise ValueError("gallery-gallery matrix must be square with the same gallery order")
    if any(not np.isfinite(alpha) or alpha < 0 for alpha in alphas):
        raise ValueError("NNN alpha values must be finite and nonnegative")
    if any(size < max(ks) or size > len(train_bank) for size in subset_sizes):
        raise ValueError(f"Requested subset sizes must lie in [{max(ks)},{len(train_bank)}]; got {subset_sizes}")
    baseline = _metric_summary(evaluate(sim, ann))
    if progress:
        progress("RAW       no calibration bank                                     "
                 + " ".join(f"{baseline[key]:9.5f}" for key in ("R1", "R5", "R10", "mAP")))
        progress("BANK REGIMES: test_full = transductive (requires complete test queries); "
                 "train_* = inductive (offline training text only). All share the fixed test gallery.")
    original_top1 = np.argmax(sim, axis=1)
    # Gallery-bank normalizers are independent of the query bank; compute once.
    gallery_stats = bank_statistics(sim_gg, (min(1, len(sim_gg)),), betas)
    banks = [("test_full", "transductive: complete test query bank", sim, None, None),
             ("train_full", "inductive: offline training text bank", train_bank, None, None)]
    for size in subset_sizes:
        for seed in seeds:
            rows = np.random.default_rng(seed).choice(len(train_bank), size=size, replace=False)
            banks.append((f"train_{size}_seed{seed}", "inductive: offline training text subset", train_bank, rows, seed))
    results, summaries = [], []
    for name, regime, values, selected, seed in banks:
        started = time.perf_counter()
        statistics = bank_statistics(values, ks, betas, rows=selected)
        apply_dis = statistics["hubs"][original_top1]
        summaries.append({"bank": name, "regime": regime, "n_bank": len(statistics["rows"]), "seed": seed,
                          "hub_gallery_count": int(statistics["hubs"].sum()),
                          "dis_applied_query_count": int(apply_dis.sum()),
                          "row_indices": statistics["rows"] if selected is not None else "all in original order"})

        def record(method, parameters, penalty, alpha=1.0, mask=None):
            metrics = _metric_summary(evaluate(sim, ann, penalty=penalty, alpha=alpha, apply_mask=mask))
            row = {"method": method, "bank": name, "regime": regime, "n_bank": len(statistics["rows"]),
                   "seed": seed, "hyperparameters": parameters, **metrics}
            results.append(row)
            if progress:
                progress(f"{method:<9} {name:<23} {json.dumps(parameters, sort_keys=True):<29} "
                         + " ".join(f"{metrics[key]:9.5f}" for key in ("R1", "R5", "R10", "mAP")))

        for k in ks:
            for alpha in alphas:
                record("CSLS/NNN", {"alpha": alpha, "k": k}, statistics["topk"][k], alpha=alpha)
        for beta in betas:
            query_penalty = statistics["is"][beta]
            record("IS", {"beta": beta}, query_penalty)
            record("DIS", {"beta": beta}, query_penalty, mask=apply_dis)
            record("DualIS", {"beta": beta}, 0.5 * (query_penalty + gallery_stats["is"][beta]))
        summaries[-1]["elapsed_sec"] = time.perf_counter() - started
    return {"baseline": baseline, "rows": results, "banks": summaries,
            "seeds": list(seeds), "subset_sizes": list(subset_sizes),
            "grid": {"alpha": list(alphas), "k": list(ks), "beta": list(betas)},
            "formulas": {
                "CSLS/NNN": "s - alpha * mean(topk(bank[:,g])); alpha=.5 is ranking-equivalent to 2s-r",
                "IS": "s - logsumexp(beta*bank[:,g])/beta",
                "DIS": "IS if original query top1 is top1 of at least one bank query; otherwise unchanged s",
                "DualIS": "s - .5*(logsumexp(beta*query_bank[:,g])/beta + logsumexp(beta*gallery_bank[:,g])/beta)",
            },
            "dual_is_note": "Rank-equivalent log of the product of two inverted softmax factors (DBNorm Eq.1). "
                            "The requested fixed test gallery is used as gallery bank; the original paper uses train/validation "
                            "gallery-bank features. This is an explicitly labelled test-gallery-bank adaptation. Self terms "
                            "are retained. No test labels enter either bank penalty. The gallery can be fixed offline.",
            "sources": METHOD_SOURCES,
            "metric_ties": "deterministic ascending gallery index; identical convention across all compared methods"}


class _Tee:
    def __init__(self, terminal, buffer):
        self.terminal, self.buffer = terminal, buffer

    def write(self, value):
        self.terminal.write(value)
        self.buffer.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()


def _load_meta(sim_path):
    path = Path(sim_path).with_name("sim_meta.json")
    if not path.is_file():
        raise FileNotFoundError(f"Archived provenance metadata is required: {path}")
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("sim_gg_composition_matches_sim_qg") is False:
        raise ValueError("Archived sim_gg does not match qg global/TSE composition; DualIS comparison would be confounded")
    return metadata


def _run_real(args, out, run_name, report):
    sim_path = Path(args.sim).expanduser().resolve()
    ann = load_annotations(args.anno)
    sim = load_similarity(sim_path, ann)
    if sim.dtype != np.float32:
        raise ValueError("S12 requires the archived float32 qg matrix, not a float16 replacement")
    gg_path = Path(args.sim_gg) if args.sim_gg else sim_path.with_name("sim_gg.npy")
    sim_gg = np.load(gg_path, mmap_mode="r", allow_pickle=False)
    for directory in (sim_path.parent, gg_path.expanduser().resolve().parent):
        if out == directory or directory in out.parents:
            raise ValueError(f"Output must not be inside a read-only input archive directory: {directory}")
    _validate_matrix(sim_gg, "gallery-gallery", sim.shape[1])
    if sim_gg.shape != (sim.shape[1], sim.shape[1]) or sim_gg.dtype != np.float32:
        raise ValueError(f"Expected float32 square gallery matrix {(sim.shape[1], sim.shape[1])}; got {sim_gg.shape}/{sim_gg.dtype}")
    metadata = _load_meta(sim_path)
    checkpoint = resolve_checkpoint(args.resume)
    recorded_checkpoint = metadata.get("checkpoint_path")
    if recorded_checkpoint and Path(recorded_checkpoint).expanduser().resolve() != checkpoint:
        raise ValueError(f"Archive checkpoint provenance differs from --resume: {recorded_checkpoint} != {checkpoint}")
    train_path, train_records, train_info = discover_train_annotations(args.anno, args.train_anno)
    test_records, test_info = _read_records(args.anno, split="test")
    report.update({"archive_metadata": metadata, "annotation": ann.summary, "train_annotation": train_info,
                   "test_annotation": test_info, "input_files": {"sim_qg": str(sim_path), "sim_gg": str(gg_path.resolve())}})
    model, config, provenance = load_model(args.config, checkpoint, args.backbone, args.device,
                                          train_info["identities"], args.seed, args.repo)
    report["feature_provenance"] = provenance
    report["extraction_seed"] = provenance["seed"]
    print("FEATURE EXTRACTION SETTINGS " + json.dumps(provenance, sort_keys=True))
    import torch
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_texts = preprocess_texts(train_records, config.experiment.max_words)
    test_texts = preprocess_texts(test_records, config.experiment.max_words)
    kwargs = {"max_length": config.experiment.text_length, "device": args.device,
              "view_prompt": bool(config.experiment.view_prompt), "batch_size": args.text_batch_size}
    started = time.perf_counter()
    first = extract_text_features(model, training_texts, **kwargs)
    second = extract_text_features(model, training_texts, **kwargs)
    report["consecutive_text_extraction"] = assert_identical_features(first, second)
    report["text_extract_twice_elapsed_sec"] = time.perf_counter() - started
    del second
    feature_meta = {**provenance, "train_annotation": str(train_path.resolve()),
                    "preprocessed_text_sha256": text_fingerprint(training_texts),
                    "max_words": config.experiment.max_words, "text_length": config.experiment.text_length,
                    "view_prompt": bool(config.experiment.view_prompt), "text_batch_size": args.text_batch_size}
    report["train_feature_file"] = save_training_features(out / "train_text_feats.npz", first, feature_meta)
    # Test text has no image-side routing either. Fresh extraction supplies the
    # exact same global/TSE text construction for the mandatory archive guard.
    test_features = extract_text_features(model, test_texts, **kwargs)
    gallery, archive_feature_path = discover_gallery_features(sim_path.parent, args.gallery_features)
    if gallery is None:
        print("No archived image features: extracting gallery ONCE, then requiring qg max_abs_error < 1e-5.")
        gallery = extract_gallery_features(model, config, args.anno, args.image_root or config.image_dir,
                                           args.device, args.image_batch_size, args.num_workers)
        gallery_source = "one fresh image extraction; no retries or RNG search"
    else:
        gallery_source = archive_feature_path
    report["gallery_source"] = gallery_source
    report["qg_parity"] = verify_query_gallery(test_features, gallery, ann.gallery_views, sim,
                                               args.device, args.score_block_size)
    bank_path = out / f"{run_name}_sim_bank_g.npy"
    train_bank = write_bank_similarity(bank_path, first, gallery, ann.gallery_views, args.device, args.score_block_size)
    report["bank_matrix"] = {"path": str(bank_path), "shape": list(train_bank.shape), "dtype": str(train_bank.dtype),
                             "bytes": bank_path.stat().st_size}
    if torch.device(args.device).type == "cuda":
        report["feature_peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    del model, first, test_features, gallery
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    print("METHOD    BANK                    HYPERPARAMETERS                      R@1       R@5      R@10       mAP")
    started = time.perf_counter()
    report["corrections"] = run_corrections(sim, train_bank, sim_gg, ann)
    report["correction_scan_elapsed_sec"] = time.perf_counter() - started


def self_test():
    """200x100 mock matrix; every correction, bank type and subset seed executes."""
    try:
        from ._offline_common import annotation_from_records
    except ImportError:
        from _offline_common import annotation_from_records
    rng = np.random.default_rng(2026)
    records = [{"file_path": f"G2APS/{'D' if index % 2 == 0 else 'S'}1_{index:04d}.jpg",
                "id": index // 2, "cam_id": index % 2, "captions": [f"caption {index} a", f"caption {index} b"]}
               for index in range(100)]
    ann = annotation_from_records(records)
    sim = rng.uniform(-0.8, 0.8, (200, 100)).astype(np.float32)
    train_bank = rng.uniform(-0.8, 0.8, (300, 100)).astype(np.float32)
    gg_features = rng.normal(size=(100, 17))
    gg_features /= np.linalg.norm(gg_features, axis=1, keepdims=True)
    gg = (gg_features @ gg_features.T).astype(np.float32)
    statistics = bank_statistics(train_bank, (5, 10, 16, 20, 30), (5, 10, 20, 50), block_size=47)
    direct_topk = np.sort(train_bank, axis=0)[-10:][::-1].mean(axis=0, dtype=np.float32)
    topk_error = float(np.abs(statistics["topk"][10] - direct_topk).max())
    assert topk_error == 0
    is_errors = {}
    for beta in (5, 10, 20, 50):
        scaled = train_bank.astype(np.float64) * beta
        maxima = scaled.max(axis=0)
        direct = (maxima + np.log(np.exp(scaled - maxima).sum(axis=0))) / beta
        error = float(np.abs(statistics["is"][beta] - direct).max())
        assert error < 1e-12
        is_errors[beta] = error
    result = run_corrections(sim, train_bank, gg, ann, subset_sizes=(40, 100, 200), progress=None)
    assert len(result["banks"]) == 11
    assert len(result["rows"]) == 352
    assert {row["method"] for row in result["rows"]} == {"CSLS/NNN", "IS", "DIS", "DualIS"}
    assert all(np.isfinite(row[key]) for row in result["rows"] for key in ("R1", "R5", "R10", "mAP"))
    # A bank with a single hub exercises both DIS application and bypass.
    partial_bank = np.full((40, 100), -0.7, dtype=np.float32)
    partial_bank[:, 0] = 0.7
    partial_stats = bank_statistics(partial_bank, (5,), (10,))
    applied = partial_stats["hubs"][np.argmax(sim, axis=1)]
    assert applied.any() and (~applied).any()
    dis_result = evaluate(sim, ann, penalty=partial_stats["is"][10], alpha=1, apply_mask=applied)
    manual = sim.astype(np.float64).copy()
    manual[applied] -= partial_stats["is"][10]
    manual_result = evaluate(manual, ann)
    assert all(dis_result[key] == manual_result[key] for key in ("R1", "R5", "R10", "mAP"))
    summary = {"shape": list(sim.shape), "bank_shape": list(train_bank.shape), "seed": 2026,
               "topk_max_abs_error": topk_error, "is_max_abs_errors": is_errors,
               "banks": len(result["banks"]), "correction_rows": len(result["rows"]),
               "baseline": result["baseline"], "dis_applied": int(applied.sum()),
               "dis_bypassed": int((~applied).sum()), "status": "PASS"}
    print("S12 MOCK SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        try:
            self_test()
            return 0
        except (OSError, ValueError, RuntimeError, AssertionError) as exc:
            parser.exit(1, f"S12 mock error: {exc}\n")
    if min(args.text_batch_size, args.image_batch_size, args.score_block_size) < 1 or args.num_workers < 0:
        parser.error("batch/block sizes must be positive and num-workers nonnegative")
    proposed_out = Path(args.out).expanduser().resolve()
    input_directories = [Path(args.sim).expanduser().resolve().parent]
    if args.sim_gg:
        input_directories.append(Path(args.sim_gg).expanduser().resolve().parent)
    if any(proposed_out == directory or directory in proposed_out.parents for directory in input_directories):
        parser.error("--out must not be inside a read-only input archive directory")
    try:
        out = safe_output_dir(proposed_out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.tag).strip("._") or "run"
    run_name = f"S12_{tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    report = {"tool": "S12", "tag": args.tag, "arguments": vars(args), "extraction_seed": args.seed,
              "subset_seeds": [2026, 2027, 2028], "started_utc": datetime.now(timezone.utc).isoformat()}
    buffer, started, failure = io.StringIO(), time.perf_counter(), None
    with redirect_stdout(_Tee(sys.stdout, buffer)):
        try:
            _run_real(args, out, run_name, report)
            report["status"] = "complete"
        except Exception as exc:
            failure = exc
            report["status"], report["error"] = "stopped", f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "parity_report"):
                report["qg_parity"] = exc.parity_report
            print("S12 STOPPED: " + report["error"])
        report["elapsed_sec"] = time.perf_counter() - started
        print(f"S12 SUMMARY status={report['status']} elapsed_sec={report['elapsed_sec']:.3f}")
        print(f"JSON: {out / (run_name + '.json')}")
        print(f"LOG: {out / (run_name + '.log')}")
    with (out / f"{run_name}.json").open("x", encoding="utf-8") as stream:
        json.dump(json_ready(report), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with (out / f"{run_name}.log").open("x", encoding="utf-8") as stream:
        stream.write(buffer.getvalue())
    if failure:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

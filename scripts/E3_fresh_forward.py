"""E3: one fresh gallery forward and query-independent training-text calibration.

Opt in with --fresh-forward. Archived scores are read ONLY to report baseline
metric differences, never for alignment, reconstruction, seeding or acceptance.
The train-bank methods are inductive; S12's test_full rows remain transductive
controls. No existing training/evaluation entrypoint is modified.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

try:
    from ._analysis_audit import git_head, sha256_16
    from ._offline_common import evaluate, json_ready, load_annotations, load_similarity, safe_output_dir
    from ._s12_features import (
        _read_records, discover_train_annotations, extract_gallery_features,
        extract_text_features, load_model, preprocess_texts, resolve_checkpoint,
        text_fingerprint, validate_features, write_bank_similarity,
    )
    from .S12_inductive_bank import DEFAULT_DUMP, DEFAULT_ROOT, _metric_summary, _Tee, run_corrections
except ImportError:
    from _analysis_audit import git_head, sha256_16
    from _offline_common import evaluate, json_ready, load_annotations, load_similarity, safe_output_dir
    from _s12_features import (
        _read_records, discover_train_annotations, extract_gallery_features,
        extract_text_features, load_model, preprocess_texts, resolve_checkpoint,
        text_fingerprint, validate_features, write_bank_similarity,
    )
    from S12_inductive_bank import DEFAULT_DUMP, DEFAULT_ROOT, _metric_summary, _Tee, run_corrections


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-forward", action="store_true",
                        help="required opt-in: fresh scores, no archived-score parity gate")
    parser.add_argument("--sim", "--sim-qg", dest="sim", default=None,
                        help="optional read-only archived qg; omitted: explicitly skip archive comparison")
    parser.add_argument("--anno", "--ann", dest="anno", required=True, help="test_reid.json in gallery order")
    parser.add_argument("--train-anno", default=None, help="explicit training annotation if discovery is ambiguous")
    parser.add_argument("--resume", required=True, help="trusted checkpoint or directory with checkpoint_best.pth")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", choices=("ViT-B/16", "ViT-L/14"), default=None)
    parser.add_argument("--repo", default=None, help="repository root if scripts are copied outside the checkout")
    parser.add_argument("--out", default=str(DEFAULT_ROOT / "experiments/_e1"), help="new artifacts directory, never the archive")
    parser.add_argument("--tag", default="fk32_fresh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0, help="single initial seed, default 0; no routing retries or seed search")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--score-block-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-root", default=None, help="default config.image_dir")
    return parser


def _state_snapshot(model, device):
    """Detect query extraction affecting subsequent bank extraction; do not reset RNG."""
    import torch
    numpy_state = np.random.get_state()
    return {
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].copy(), numpy_state[2:]),
        "torch_rng": torch.get_rng_state().clone(),
        "cuda_rng": [value.clone() for value in torch.cuda.get_rng_state_all()]
                    if torch.device(device).type == "cuda" else [],
        "model_buffers": {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
    }


def _assert_state_unchanged(before, after, stage):
    """No test-text-dependent RNG/buffer changes may leak into the train bank."""
    import torch
    changed = []
    if before["python_rng"] != after["python_rng"]:
        changed.append("Python RNG")
    left, right = before["numpy_rng"], after["numpy_rng"]
    if left[0] != right[0] or not np.array_equal(left[1], right[1]) or left[2] != right[2]:
        changed.append("NumPy RNG")
    if not torch.equal(before["torch_rng"], after["torch_rng"]):
        changed.append("Torch CPU RNG")
    if len(before["cuda_rng"]) != len(after["cuda_rng"]) or any(
            not torch.equal(a, b) for a, b in zip(before["cuda_rng"], after["cuda_rng"])):
        changed.append("Torch CUDA RNG")
    left, right = before["model_buffers"], after["model_buffers"]
    if left.keys() != right.keys() or any(not torch.equal(left[key], right[key]) for key in left if key in right):
        changed.append("model buffers")
    if changed:
        raise RuntimeError(f"E3 independence guard: {stage} changed {', '.join(changed)}; "
                           "subsequent bank extraction cannot be certified query-independent. No RNG reset/retry performed.")
    return {"stage": stage, "rng_unchanged": True, "model_buffers_unchanged": True,
            "action": "checked only; no resetting/reseeding or state replacement"}


def _synchronize(device):
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _validate_text_gallery(text, gallery, name):
    validate_features(text)
    validate_features(gallery)
    if set(gallery) != {"image_feat", "image_tse_feat"}:
        raise ValueError("Gallery features need exactly image_feat and image_tse_feat")
    prompted = "text_feat_view0" in text
    suffixes = ("_view0", "_view1") if prompted else ("",)
    expected = {key + suffix for suffix in suffixes for key in ("text_feat", "text_tse_feat")}
    if set(text) != expected:
        raise ValueError(f"{name}: expected complete global/TSE feature keys {sorted(expected)}, got {sorted(text)}")
    for suffix in suffixes:
        for text_key, image_key in (("text_feat", "image_feat"), ("text_tse_feat", "image_tse_feat")):
            if text[text_key + suffix].shape[1] != gallery[image_key].shape[1]:
                raise ValueError(f"{name}: {text_key + suffix} width does not match {image_key}")


def write_fresh_matrices(out, run_name, test_features, train_features, gallery_features,
                         gallery_views, device, block_size=256):
    """Use the same frozen gallery for qg, bank-g and gg, with unchanged FP32 scoring."""
    if block_size < 1:
        raise ValueError("score-block-size must be positive")
    _validate_text_gallery(test_features, gallery_features, "test text")
    _validate_text_gallery(train_features, gallery_features, "train text")
    if set(test_features) != set(train_features):
        raise ValueError("Train and test text must use the same view-prompt configuration")
    out = safe_output_dir(out)
    paths = {"sim_qg": out / f"{run_name}_sim_qg_fresh.npy",
             "sim_bank_g": out / f"{run_name}_sim_bank_g_fresh.npy",
             "sim_gg": out / f"{run_name}_sim_gg_fresh.npy"}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite fresh matrix: {path}")
    # Re-label the already extracted gallery's two branches, without recomputing
    # or reshaping any features. This gives .5*(G_global G_global.T + G_TSE G_TSE.T).
    gallery_as_query = {"text_feat": gallery_features["image_feat"],
                        "text_tse_feat": gallery_features["image_tse_feat"]}
    sources = {"sim_qg": test_features, "sim_bank_g": train_features, "sim_gg": gallery_as_query}
    result = {"paths": {key: str(path) for key, path in paths.items()}}
    for key, features in sources.items():
        result[key] = write_bank_similarity(paths[key], features, gallery_features, gallery_views, device, block_size)
        matrix = result[key]
        expected = (len(next(iter(features.values()))), len(gallery_features["image_feat"]))
        if matrix.shape != expected or matrix.dtype != np.float32:
            raise ValueError(f"Fresh {key} shape/dtype mismatch: {matrix.shape}/{matrix.dtype}; expected {expected}/float32")
        for start in range(0, len(matrix), block_size):
            if not np.isfinite(matrix[start:start + block_size]).all():
                raise ValueError(f"Fresh {key} has non-finite scores near row {start}")
    return result


def _run_real(args, out, run_name, report):
    sim_path = Path(args.sim).expanduser().resolve() if args.sim else None
    out = Path(out).expanduser().resolve()
    if sim_path is not None and (out == sim_path.parent or sim_path.parent in out.parents):
        raise ValueError("--out must not be inside the read-only similarity archive")
    ann = load_annotations(args.anno)
    checkpoint = resolve_checkpoint(args.resume)
    train_path, train_records, train_info = discover_train_annotations(args.anno, args.train_anno)
    test_records, test_info = _read_records(args.anno, split="test")
    report.update({"annotation": ann.summary, "train_annotation": train_info, "test_annotation": test_info,
                   "fresh_forward": True, "seed": args.seed, "image_batch_size": args.image_batch_size,
                   "text_batch_size": args.text_batch_size, "score_block_size": args.score_block_size,
                   "num_workers": args.num_workers, "extraction_order": ["gallery_images", "test_text", "train_text"],
                   "pure_query_independent_inductive_pipeline": False, "independence_status": "pending",
                   "inductive_scope": "train_* bank construction/calibration only; test_full scan rows remain transductive controls",
                   "test_query_use_in_training_bank": False, "alignment_W_used": False,
                   "archive_parity_checked": False,
                   "archive_role": "metrics-only reference; no reconstruction, alignment, acceptance gate or routing retries",
                   "archived_similarity": str(sim_path) if sim_path else None,
                   "archive_comparison_skipped": sim_path is None, "warnings": []})
    model, config, provenance = load_model(args.config, checkpoint, args.backbone, args.device,
                                          train_info["identities"], args.seed, args.repo)
    report["feature_provenance"] = provenance
    paths = {"checkpoint": checkpoint, "config": Path(args.config),
             "test_annotation": Path(args.anno), "train_annotation": train_path}
    report["provenance"] = {
        "argv": report.get("argv", list(sys.argv)),
        "git_HEAD": git_head(args.repo), "seeds": {"initial_extraction": args.seed,
                                                    "correction_subsets": [2026, 2027, 2028]},
        "checkpoint_sha256_16": sha256_16(checkpoint),
        "eval_router_mode": getattr(config.experiment, "eval_router_mode", "gumbel"),
        # E3 uses its existing SINGLE initial seed before construction. It does
        # not call main's post-load eval_seed reset; distinguish configured from applied.
        "eval_seed": None, "initial_extraction_seed": args.seed,
        "configured_eval_seed": getattr(config.experiment, "eval_seed", None),
        "eval_seed_reset_applied": False,
        "inputs": {key: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
                         "sha256_16": sha256_16(path)} for key, path in paths.items()},
    }
    report["view_prompt"] = bool(config.experiment.view_prompt)
    print("E3 FRESH SETTINGS " + json.dumps({"seed": args.seed, "image_batch_size": args.image_batch_size,
                                            "text_batch_size": args.text_batch_size, "provenance": provenance}, sort_keys=True))
    import torch
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elapsed = report["extraction_elapsed_sec"] = {}

    # No feature cache, no parity-based rerun: this ONE image pass freezes the
    # stochastic routing realization used by every subsequent score matrix.
    _synchronize(args.device)
    started = time.perf_counter()
    gallery = extract_gallery_features(model, config, args.anno, args.image_root or config.image_dir,
                                       args.device, args.image_batch_size, args.num_workers)
    _synchronize(args.device)
    elapsed["gallery_images"] = time.perf_counter() - started
    if len(gallery["image_feat"]) != len(ann.gallery_ids):
        raise ValueError("Fresh gallery features do not cover all annotation images in order")
    report["gallery_extraction_passes"] = 1
    image_root = Path(args.image_root or config.image_dir).expanduser()
    report["provenance"]["gallery_image_inputs"] = []
    for relative in ann.paths:
        image_path = (image_root / relative).resolve()
        evidence = {"path": str(image_path), "available": image_path.is_file()}
        if image_path.is_file():
            evidence.update(bytes=image_path.stat().st_size, sha256_16=sha256_16(image_path))
        report["provenance"]["gallery_image_inputs"].append(evidence)
    print(f"E3 EXTRACT gallery_images rows={len(gallery['image_feat'])} elapsed_sec={elapsed['gallery_images']:.3f}")

    text_kwargs = dict(max_length=config.experiment.text_length, device=args.device,
                       view_prompt=bool(config.experiment.view_prompt), batch_size=args.text_batch_size)
    guards = report["independence_guards"] = []
    before = _state_snapshot(model, args.device)
    started = time.perf_counter()
    test_texts = preprocess_texts(test_records, config.experiment.max_words)
    test_features = extract_text_features(model, test_texts, **text_kwargs)
    _synchronize(args.device)
    elapsed["test_text"] = time.perf_counter() - started
    guards.append(_assert_state_unchanged(before, _state_snapshot(model, args.device), "test_text"))
    if len(next(iter(test_features.values()))) != len(ann.query_ids):
        raise ValueError("Fresh test features do not cover every annotation caption in order")
    print(f"E3 EXTRACT test_text rows={len(test_texts)} elapsed_sec={elapsed['test_text']:.3f}")

    before = _state_snapshot(model, args.device)
    started = time.perf_counter()
    training_texts = preprocess_texts(train_records, config.experiment.max_words)
    train_features = extract_text_features(model, training_texts, **text_kwargs)
    _synchronize(args.device)
    elapsed["train_text"] = time.perf_counter() - started
    guards.append(_assert_state_unchanged(before, _state_snapshot(model, args.device), "train_text"))
    if len(next(iter(train_features.values()))) != train_info["captions"]:
        raise ValueError("Fresh training features do not cover every training caption in order")
    elapsed["total"] = sum(elapsed.values())
    print(f"E3 EXTRACT train_text rows={len(training_texts)} elapsed_sec={elapsed['train_text']:.3f}")
    report["text_fingerprints"] = {"test": text_fingerprint(test_texts), "train": text_fingerprint(training_texts)}
    report["feature_shapes"] = {name: {key: list(value.shape) for key, value in features.items()}
                                for name, features in (("gallery", gallery), ("test", test_features), ("train", train_features))}
    del before, model
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()

    _synchronize(args.device)
    started = time.perf_counter()
    matrices = write_fresh_matrices(out, run_name, test_features, train_features, gallery,
                                    ann.gallery_views, args.device, args.score_block_size)
    _synchronize(args.device)
    report["score_computation_elapsed_sec"] = time.perf_counter() - started
    report["fresh_files"] = matrices["paths"]
    report["fresh_shapes"] = {key: list(matrices[key].shape) for key in ("sim_qg", "sim_bank_g", "sim_gg")}
    report["pure_query_independent_inductive_pipeline"] = True
    report["independence_status"] = "verified_for_train_bank_paths"
    report["scoring"] = {
        "qg_and_bank_g": "0.5*(normalized text_global @ image_global.T + normalized text_TSE @ image_TSE.T), FP32",
        "gg": "0.5*(image_global @ image_global.T + image_TSE @ image_TSE.T), SAME one-pass normalized gallery",
        "view_prompt": "if enabled, encode text separately for views 0/1 and select by gallery column view, including train texts",
        "gallery_self_terms": "retained for the unchanged S12 DualIS scan; no gallery-CSLS self-exclusion applied",
    }
    if torch.device(args.device).type == "cuda":
        report["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        report["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    else:
        report["cuda_peak_allocated_bytes"] = None
        report["cuda_peak_reserved_bytes"] = None
    del gallery, train_features, test_features

    # Archive access is intentionally after the fresh bank exists. It cannot
    # select a seed, change gallery routing, estimate W, or supply bank scores.
    raw = evaluate(matrices["sim_qg"], ann)
    report["fresh_metrics"] = _metric_summary(raw)
    if sim_path is None:
        report["archived_metrics"] = None
        report["fresh_minus_archived_metrics_pp"] = None
        report["fresh_minus_archived_correct_queries"] = None
        print("E3 ARCHIVE COMPARISON SKIPPED: --sim not supplied; fresh metrics only, no archive read.")
    else:
        archived = load_similarity(sim_path, ann)
        report["provenance"]["inputs"]["archived_similarity"] = {
            "path": str(sim_path), "bytes": sim_path.stat().st_size, "sha256_16": sha256_16(sim_path)}
        archived_eval = evaluate(archived, ann)
        report["archived_metrics"] = _metric_summary(archived_eval)
        delta = {key: float(raw[key] - archived_eval[key]) for key in ("R1", "R5", "R10", "mAP")}
        report["fresh_minus_archived_metrics_pp"] = delta
        report["fresh_minus_archived_correct_queries"] = int(np.asarray(raw["details"]["correct"]).sum()
                                                              - np.asarray(archived_eval["details"]["correct"]).sum())
        print("E3 ARCHIVE COMPARISON " + json.dumps({"fresh": report["fresh_metrics"], "archived": report["archived_metrics"],
                                                    "delta_pp": delta, "parity_gate": False}, sort_keys=True))
    print("E3 INDUCTIVE SCOPE: train_* is query-independent; test_full is a transductive control. "
          "No archive parity threshold, rerouting, W or test-query calibration of train banks.")
    print("METHOD    BANK                    HYPERPARAMETERS                      R@1       R@5      R@10       mAP")
    started = time.perf_counter()
    report["corrections"] = run_corrections(matrices["sim_qg"], matrices["sim_bank_g"], matrices["sim_gg"], ann)
    report["correction_scan_elapsed_sec"] = time.perf_counter() - started


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.fresh_forward:
        parser.error("E3 requires explicit --fresh-forward; use unchanged S12 for archived-matrix analysis")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be an integer in [0, 2**32)")
    if min(args.image_batch_size, args.text_batch_size, args.score_block_size) < 1 or args.num_workers < 0:
        parser.error("batch/block sizes must be positive and --num-workers must be nonnegative")
    proposed_out = Path(args.out).expanduser().resolve()
    if args.sim:
        archive_directory = Path(args.sim).expanduser().resolve().parent
        if proposed_out == archive_directory or archive_directory in proposed_out.parents:
            parser.error("--out must not be inside the read-only similarity archive")
    try:
        out = safe_output_dir(proposed_out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.tag).strip("._") or "run"
    run_name = f"E3_{tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    report = {"tool": "E3", "tag": args.tag, "arguments": vars(args),
              "argv": list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    buffer, started, failure = io.StringIO(), time.perf_counter(), None
    with redirect_stdout(_Tee(sys.stdout, buffer)):
        try:
            _run_real(args, out, run_name, report)
            report["status"] = "complete"
        except Exception as exc:
            failure = exc
            report.update(status="stopped", error=f"{type(exc).__name__}: {exc}")
            if not report.get("pure_query_independent_inductive_pipeline", False):
                report["independence_status"] = "not_verified"
            print("E3 STOPPED: " + report["error"])
        report["elapsed_sec"] = time.perf_counter() - started
        print(f"E3 SUMMARY status={report['status']} elapsed_sec={report['elapsed_sec']:.3f}")
        print(f"JSON: {out / (run_name + '.json')}")
        print(f"LOG: {out / (run_name + '.log')}")
    with (out / f"{run_name}.json").open("x", encoding="utf-8") as stream:
        json.dump(json_ready(report), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with (out / f"{run_name}.log").open("x", encoding="utf-8") as stream:
        stream.write(buffer.getvalue())
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

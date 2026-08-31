"""S15: lock inductive CSLS hyperparameters using only training identities.

The E3 JSON (and every test file named by it) is opened only AFTER the proxy
scan has written an immutable parameter lock. Test matrices are never rebuilt.
The final test phase has one block traversal and two already-frozen settings.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from . import E3_fresh_forward as e3
    from ._analysis_audit import Audit
    from ._offline_common import (
        annotation_from_records, compact_metrics, evaluate, load_annotations,
        load_matrix, print_table, topk_mean,
    )
except ImportError:
    import E3_fresh_forward as e3
    from _analysis_audit import Audit
    from _offline_common import (
        annotation_from_records, compact_metrics, evaluate, load_annotations,
        load_matrix, print_table, topk_mean,
    )


ALPHAS = (0.25, 0.5, 0.75, 1.0)
KS = (5, 10, 16, 20, 30)
METRICS = ("R1", "R5", "R10", "mAP")
LIMITATION = "代理验证图像为训练图像，模型在训练期已见"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-anno", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", choices=("ViT-B/16", "ViT-L/14"), default=None)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--out", required=True, help="new output root named _e1_s15")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-identities", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--e3-json", required=True,
                        help="authoritative E3 JSON; not opened until the parameters are locked")
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--score-block-size", type=int, default=256)
    parser.add_argument("--image-root", default=None, help="default config.image_dir")
    parser.add_argument("--repo", default=None, help="repository root when scripts were copied elsewhere")
    return parser


def split_training_records(records, val_identities=1000, split_seed=2026):
    """Use sorted unique IDs and exactly one permutation, preserving image order."""
    if any(str(row.get("split", "train")).lower() not in ("train", "training") for row in records):
        raise ValueError("--train-anno contains a non-training record; test/validation records are forbidden")
    ann = annotation_from_records(records)
    identities = np.unique(ann.gallery_ids)
    if not 1 <= val_identities < len(identities):
        raise ValueError(f"--val-identities must be in [1, {len(identities) - 1}], got {val_identities}")
    order = np.random.default_rng(split_seed).permutation(identities)
    validation, bank = order[:val_identities], order[val_identities:]
    vi = np.flatnonzero(np.isin(ann.gallery_ids, validation))
    bi = np.flatnonzero(np.isin(ann.gallery_ids, bank))
    vr, br = [records[int(i)] for i in vi], [records[int(i)] for i in bi]
    v_ann, b_ann = annotation_from_records(vr), annotation_from_records(br)
    checks = {
        "identity_intersection_empty": len(np.intersect1d(validation, bank)) == 0,
        "identity_union_equals_training": np.array_equal(np.sort(np.r_[validation, bank]), identities),
        "image_intersection_empty": len(np.intersect1d(vi, bi)) == 0,
        "image_partition_conserved": np.array_equal(np.sort(np.r_[vi, bi]), np.arange(len(records))),
        "gallery_identities_in_V": bool(np.isin(v_ann.gallery_ids, validation).all()),
        "bank_caption_identities_in_B": bool(np.isin(b_ann.query_ids, bank).all()),
        "query_sources_are_proxy_gallery": bool(np.array_equal(
            v_ann.query_ids, v_ann.gallery_ids[v_ann.query_source_indices])),
        "caption_partition_conserved": len(v_ann.query_ids) + len(b_ann.query_ids) == len(ann.query_ids),
        "gallery_and_bank_image_paths_disjoint": not bool(set(v_ann.paths) & set(b_ann.paths)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Training-only partition conservation failed: {checks}")
    info = {
        "split_seed": int(split_seed), "method": "np.unique(ids), default_rng(split_seed).permutation, first V then B",
        "validation_identities": validation.tolist(), "bank_identities": bank.tolist(),
        "validation_image_indices": vi.tolist(), "bank_image_indices": bi.tolist(), "checks": checks,
        "counts": {"all_identities": len(identities), "V": len(validation), "B": len(bank),
                   "G_V": len(vi), "Q_V": len(v_ann.query_ids), "T_B": len(b_ann.query_ids),
                   "all_images": len(records), "all_captions": len(ann.query_ids)},
    }
    return vr, br, v_ann, info


def scan_proxy(sim, bank, ann):
    """Only the proxy split is accepted here. Source images are excluded before ranking."""
    if bank.ndim != 2 or bank.shape[1] != sim.shape[1] or len(bank) < max(KS):
        raise ValueError(f"Proxy bank must cover the proxy gallery and contain at least {max(KS)} captions")
    raw = compact_metrics(evaluate(sim, ann, exclude_source=True))
    if not raw["n_query"]:
        raise ValueError("No proxy query has a positive image after excluding its source; use more images per identity")
    penalties = {k: topk_mean(bank, k) for k in KS}
    rows = []
    for alpha in ALPHAS:
        for k in KS:
            row = {"alpha": alpha, "k": k, **compact_metrics(evaluate(
                sim, ann, exclude_source=True, penalty=penalties[k], alpha=alpha))}
            if row["n_query"] != raw["n_query"]:
                raise RuntimeError("Proxy CSLS changed the eligible query set")
            rows.append(row)
    # k resolves the user-requested tie; alpha is a deterministic tertiary tie.
    ranked = sorted(rows, key=lambda row: (-row["R1"], row["k"], row["alpha"]))
    return {"uncorrected": raw, "rows": rows, "selected": dict(ranked[0]),
            "runner_up": dict(ranked[1]),
            "selection_rule": "largest proxy R1; ties: smaller k, then smaller alpha",
            "score": "float32 S - alpha * mean(top-k training-bank scores by gallery column)",
            "source_image_excluded": True,
            "n_query_skipped_no_positive": raw["skipped_no_positive"]}


def evaluate_frozen_methods(sim, bank, ann, selected, block_size=256):
    """One score-row traversal, no test search, all choices immutable on entry.

    We reuse the shared exact evaluator per block/method. Integer hit counts and
    AP vectors are reassembled in original query order, preserving its float32
    R@k arithmetic and float64 mAP bit for bit. The default is deduplicated.
    """
    selected = (float(selected["alpha"]), int(selected["k"]))
    if selected[0] not in ALPHAS or selected[1] not in KS:
        raise ValueError("Locked parameters do not belong to the predefined proxy grid")
    if block_size < 1 or sim.shape != (len(ann.query_ids), len(ann.gallery_ids)):
        raise ValueError("Invalid final test shape or score block size")
    default = (0.5, 10)
    params = list(dict.fromkeys((selected, default)))
    if bank.ndim != 2 or bank.shape[1] != sim.shape[1] or len(bank) < max(k for _, k in params):
        raise ValueError("E3 full training bank is incompatible with the test gallery or frozen k")
    penalties = {k: topk_mean(bank, k) for _, k in params}
    configs = {"uncorrected": None, **{f"frozen_{i}": param for i, param in enumerate(params)}}
    accum = {name: {"hits": np.zeros(3, dtype=np.int64), "ap": [], "n": 0,
                    "ties": 0, "correct_count": 0} for name in configs}
    rows_seen = 0
    for start in range(0, len(sim), block_size):
        stop = min(start + block_size, len(sim))
        scores = np.array(sim[start:stop], dtype=np.float32, copy=True)
        batch_ann = replace(ann, query_source_indices=ann.query_source_indices[start:stop],
                            query_ids=ann.query_ids[start:stop], query_views=ann.query_views[start:stop],
                            query_domains=ann.query_domains[start:stop])
        for name, param in configs.items():
            kwargs = {} if param is None else {"penalty": penalties[param[1]], "alpha": param[0]}
            result = evaluate(scores, batch_ann, **kwargs)
            n = result["n_query"]
            if n != stop - start or result["skipped_no_positive"]:
                raise RuntimeError("An E3 test query has no positive: annotation/matrix order is inconsistent")
            hit_counts = np.rint(np.array([result[key] for key in METRICS[:3]]) * n / 100).astype(np.int64)
            if not ((hit_counts >= 0) & (hit_counts <= n)).all():
                raise RuntimeError("Invalid metric aggregation")
            a = accum[name]
            a["hits"] += hit_counts
            a["ap"].append(result["details"]["ap"])
            a["n"] += n
            a["ties"] += result["top1_tied_queries"]
            a["correct_count"] += int(result["details"]["correct"].sum())
        rows_seen += stop - start
    outputs = {}
    for name, a in accum.items():
        if a["n"] != len(sim) or a["hits"][0] != a["correct_count"]:
            raise RuntimeError("Final test traversal conservation failed")
        outputs[name] = {key: float(np.float32(a["hits"][i]) * np.float32(100) / np.float32(a["n"]))
                         for i, key in enumerate(METRICS[:3])}
        outputs[name].update(mAP=float(np.concatenate(a["ap"]).mean() * 100), n_query=a["n"],
                             correct_count=a["correct_count"], top1_tied_queries=a["ties"])
    return {"uncorrected": outputs["uncorrected"], "selected": outputs["frozen_0"],
            "literature_default": outputs[f"frozen_{params.index(default)}"],
            "frozen_parameters": {"selected": {"alpha": selected[0], "k": selected[1]},
                                  "literature_default": {"alpha": default[0], "k": default[1]}},
            "test_evaluation_passes": 1, "rows_visited": rows_seen,
            "unique_corrected_settings": len(params), "selected_equals_default": selected == default,
            "source_image_excluded": False,
            "note": "one final test phase / one qg row traversal; raw is an integrity gate, not a parameter candidate"}


def _metrics_table(title, rows):
    print(title)
    print_table(["setting", "alpha", "k", "R@1", "R@5", "R@10", "mAP", "queries"], [
        [row.get("setting", "proxy"), row.get("alpha", "-"), row.get("k", "-"),
         *[f"{row[key]:.10f}" for key in METRICS], row["n_query"]] for row in rows])


def _close_matrices(*matrices):
    for matrix in matrices:
        if isinstance(matrix, np.memmap):
            matrix._mmap.close()


def _path_from_authority(value, e3_path):
    if not isinstance(value, str) or not value:
        raise ValueError("E3 authority has an empty/missing artifact path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(e3_path).resolve().parent / path
    return path.resolve()


def final_test_phase(args, audit, selected, train_info, train_text_fingerprint, provenance, lock_path):
    """The ONLY function allowed to open E3/test inputs. Caller must have locked."""
    if not Path(lock_path).is_file():
        raise RuntimeError("No durable parameter lock exists; refusing to open E3/test inputs")
    authority = audit.read_json(args.e3_json, "E3 authority (first read, after parameter lock)")
    report = audit.report
    report["test_phase_started_after_lock"] = str(lock_path)
    if authority.get("status") != "complete" or not authority.get("fresh_forward") or not authority.get(
            "pure_query_independent_inductive_pipeline") or authority.get("alignment_W_used", False):
        raise ValueError("--e3-json must certify a completed fresh, W-free inductive E3 run")
    prior = authority.get("feature_provenance", {})
    for key in ("backbone", "loaded_state", "seed"):
        audit.check(f"E3/proxy feature provenance {key}", provenance.get(key), prior.get(key))
    for key in ("checkpoint", "config"):
        audit.check(f"E3/proxy {key} path", str(Path(provenance[key]).resolve()),
                    str(_path_from_authority(prior.get(key), args.e3_json)))
    audit.check("E3/full training identities", train_info["identities"], authority["train_annotation"]["identities"])
    audit.check("E3/full training captions", train_info["captions"], authority["train_annotation"]["captions"])
    audit.check("E3/full training text fingerprint", train_text_fingerprint, authority["text_fingerprints"]["train"])
    report["E3_reuse_provenance"] = {
        "feature_provenance": prior, "fresh_files": authority.get("fresh_files"),
        "checkpoint_config_history_caveat": "E3 stores paths, not historical file hashes; current files are hashed, historical byte equality cannot be certified",
        "proxy_image_batch_size": args.image_batch_size, "E3_image_batch_size": authority.get("image_batch_size"),
        "test_matrices_recomputed": False,
    }
    files = authority.get("fresh_files", {})
    qg_path = _path_from_authority(files.get("sim_qg"), args.e3_json)
    bank_path = _path_from_authority(files.get("sim_bank_g"), args.e3_json)
    ann_path = _path_from_authority(authority.get("test_annotation", {}).get("path"), args.e3_json)
    for path, role in ((qg_path, "E3 saved test qg"), (bank_path, "E3 saved full training bank scores"),
                       (ann_path, "E3 test annotation")):
        audit.record_input(path, role)
    ann = load_annotations(ann_path)
    audit.check("E3/test annotation counts", ann.summary, authority["annotation"])
    shape = (len(ann.query_ids), len(ann.gallery_ids))
    audit.check("E3/test qg declared shape", list(shape), authority["fresh_shapes"]["sim_qg"])
    bank_shape = (train_info["captions"], len(ann.gallery_ids))
    audit.check("E3/full training bank declared shape", list(bank_shape), authority["fresh_shapes"]["sim_bank_g"])
    sim = bank = None
    try:
        sim, bank = load_matrix(qg_path, shape), load_matrix(bank_path, bank_shape)
        result = evaluate_frozen_methods(sim, bank, ann, selected, args.score_block_size)
        for key in METRICS:
            audit.check(f"E3 uncorrected exact {key}", result["uncorrected"][key], authority["fresh_metrics"][key])
        result["selected_minus_uncorrected_pp"] = {
            key: result["selected"][key] - result["uncorrected"][key] for key in METRICS}
        result["default_minus_uncorrected_pp"] = {
            key: result["literature_default"][key] - result["uncorrected"][key] for key in METRICS}
        result["training_bank_rows"] = len(bank)
        report["test"] = result
        _metrics_table("S15 FINAL TEST (frozen once; no parameter search)", [
            {"setting": name, **result[name], **result["frozen_parameters"].get(name, {})}
            for name in ("uncorrected", "selected", "literature_default")])
    finally:
        _close_matrices(sim, bank)


def run(args, audit):
    """Training-only until parameter lock; do not inspect args.e3_json here first."""
    report = audit.report
    report.update(limitation=LIMITATION, test_used_for_selection=False, test_matrices_recomputed=False,
                  pure_query_independent_inductive_pipeline=False,
                  extraction_order=["proxy_gallery_images_V", "proxy_text_V", "bank_text_B"],
                  formula="S - alpha * column_topk_mean(training_text_bank_scores)",
                  test_phase_started_after_lock=None)
    for path, role in ((args.train_anno, "training annotation"), (args.config, "proxy model config")):
        audit.record_input(path, role)
    checkpoint = e3.resolve_checkpoint(args.resume)
    audit.record_input(checkpoint, "trusted model checkpoint")
    records, train_info = e3._read_records(args.train_anno, split="train")
    vr, br, ann, split = split_training_records(records, args.val_identities, args.split_seed)
    report["split"] = split
    report["train_annotation"] = train_info
    report["split_file"] = str(audit.write_json(f"training_identity_split_seed{args.split_seed}.json", split))
    if split["counts"]["T_B"] < max(KS):
        raise ValueError(f"Training bank B has fewer than {max(KS)} captions, insufficient for the fixed k grid")
    print("S15 TRAINING PARTITION " + json.dumps(split["counts"], sort_keys=True))
    print("S15 LIMITATION: " + LIMITATION)
    directory = audit.new_run_dir()
    proxy_ann = directory / "proxy_validation_reid.json"
    with proxy_ann.open("x", encoding="utf-8") as stream:
        json.dump(vr, stream, ensure_ascii=False, indent=2)
    report["proxy_annotation"] = str(proxy_ann)
    # Crucially the classifier is constructed with ALL training identities,
    # not the V split cardinality; this preserves checkpoint shape compatibility.
    model, config, provenance = e3.load_model(args.config, checkpoint, args.backbone, args.device,
                                             train_info["identities"], args.seed, args.repo)
    report["feature_provenance"] = provenance
    report["model_num_classes"] = train_info["identities"]
    image_root = Path(args.image_root or config.image_dir).expanduser()
    for relative in ann.paths:
        audit.record_input(image_root / relative, f"proxy training image V:{relative}")
    elapsed, guards = {}, []
    e3._synchronize(args.device)
    started = time.perf_counter()
    gallery = e3.extract_gallery_features(model, config, proxy_ann, image_root,
                                          args.device, args.image_batch_size, args.num_workers)
    e3._synchronize(args.device)
    elapsed["gallery_images"] = time.perf_counter() - started
    audit.check("proxy gallery extracted count", len(gallery["image_feat"]), len(ann.gallery_ids))
    report["gallery_extraction_passes"] = 1
    text_kwargs = dict(max_length=config.experiment.text_length, device=args.device,
                       view_prompt=bool(config.experiment.view_prompt), batch_size=args.text_batch_size)
    extracted = []
    for name, subset, count in (("proxy_text_V", vr, split["counts"]["Q_V"]),
                                ("bank_text_B", br, split["counts"]["T_B"])):
        before = e3._state_snapshot(model, args.device)
        started = time.perf_counter()
        texts = e3.preprocess_texts(subset, config.experiment.max_words)
        features = e3.extract_text_features(model, texts, **text_kwargs)
        e3._synchronize(args.device)
        elapsed[name] = time.perf_counter() - started
        guards.append(e3._assert_state_unchanged(before, e3._state_snapshot(model, args.device), name))
        audit.check(f"{name} extracted count", len(next(iter(features.values()))), count)
        extracted.append(features)
    elapsed["total"] = sum(elapsed.values())
    report["extraction_elapsed_sec"], report["independence_guards"] = elapsed, guards
    report["pure_query_independent_inductive_pipeline"] = True
    report["feature_shapes"] = {name: {key: list(value.shape) for key, value in features.items()}
                                 for name, features in (("gallery_V", gallery), ("query_V", extracted[0]),
                                                        ("bank_B", extracted[1]))}
    # No extra extraction: hash processed full TRAIN captions for E3 row-order
    # compatibility. This stage touches no test annotation/text/score.
    train_fingerprint = e3.text_fingerprint(e3.preprocess_texts(records, config.experiment.max_words))
    report["full_train_text_fingerprint"] = train_fingerprint
    del model, before
    import torch
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()
    sim = bank = None
    try:
        qg_path, bank_path = directory / "proxy_sim_qg_fresh.npy", directory / "proxy_sim_bank_g_fresh.npy"
        e3._validate_text_gallery(extracted[0], gallery, "proxy text V")
        e3._validate_text_gallery(extracted[1], gallery, "bank text B")
        sim = e3.write_bank_similarity(qg_path, extracted[0], gallery, ann.gallery_views,
                                        args.device, args.score_block_size)
        bank = e3.write_bank_similarity(bank_path, extracted[1], gallery, ann.gallery_views,
                                         args.device, args.score_block_size)
        report["proxy_files"] = {"sim_qg": str(qg_path), "sim_bank_g": str(bank_path)}
        report["proxy_shapes"] = {"sim_qg": list(sim.shape), "sim_bank_g": list(bank.shape)}
        proxy = scan_proxy(sim, bank, ann)
    finally:
        _close_matrices(sim, bank)
    report["proxy"] = proxy
    _metrics_table("S15 PROXY VALIDATION (source image excluded)",
                   [{"setting": "uncorrected", **proxy["uncorrected"]}] + proxy["rows"])
    selected = {key: proxy["selected"][key] for key in ("alpha", "k")}
    lock = {"selected": selected, "runner_up": proxy["runner_up"], "selection_rule": proxy["selection_rule"],
            "selected_proxy_metrics": proxy["selected"], "test_inputs_read": False,
            "seed": args.seed, "split_seed": args.split_seed, "limitation": LIMITATION,
            "git_HEAD": report.get("git_HEAD"), "inputs_at_lock": dict(report.get("inputs", {}))}
    lock_path = audit.write_json("parameters_lock.json", lock)
    report["parameter_lock"] = str(lock_path)
    print(f"S15 PARAMETERS LOCKED alpha={selected['alpha']} k={selected['k']} lock={lock_path}")
    # This boundary is deliberate: no E3 JSON read, matrix discovery, or test
    # annotation inspection occurs before the immutable training-only lock.
    final_test_phase(args, audit, selected, train_info, train_fingerprint, provenance, lock_path)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if any(not 0 <= value < 2 ** 32 for value in (args.seed, args.split_seed)):
        parser.error("--seed and --split-seed must be integers in [0, 2**32)")
    if min(args.val_identities, args.image_batch_size, args.text_batch_size, args.score_block_size) < 1:
        parser.error("identity/batch/block sizes must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    try:
        audit = Audit("S15", args, argv=argv, seeds={"feature_seed": args.seed, "identity_split": args.split_seed},
                      repo=args.repo)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    failure = None
    with audit.capture():
        try:
            run(args, audit)
        except Exception as exc:
            failure = exc
            print(f"S15 STOPPED: {type(exc).__name__}: {exc}")
        audit.finish(status="stopped" if failure else "complete",
                     error=f"{type(failure).__name__}: {failure}" if failure else None)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())

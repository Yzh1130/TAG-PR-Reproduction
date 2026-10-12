"""Numerical A/B and short-training gates; provenance is not an acceptance target.

All comparisons are exact. Differences are reported without changing inputs,
rounding results, re-seeding, or switching the router. GPU runs are performed by
the shell entrypoints on the training machine; ``mock`` only tests this utility.
"""

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np


METRIC_KEYS = ("R@1", "R@5", "R@10", "mAP")
TRACE_KEYS = (
    "update", "it", "epoch", "lr", "alpha", "niter_per_ep",
    "itc_block_size", "loss_weights", "losses", "total_loss", "grad_norm", "finite",
)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_report(path, report):
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A failed/repeated gate must not overwrite earlier evidence.
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


def prediction_rows(path):
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected TAG-PEDES per-query result list")
    rows = {}
    for record in records:
        if not isinstance(record, dict) or len(record) != 1:
            raise ValueError(f"{path}: each record must map one query index to its result")
        index_text, payload = next(iter(record.items()))
        index = int(index_text)
        if index in rows or not isinstance(payload, dict) or "pred" not in payload:
            raise ValueError(f"{path}: duplicate query or missing 'pred': {index}")
        arrays = {}
        for key, value in payload.items():
            if isinstance(value, list):
                array = np.asarray(value)
                if array.dtype.kind not in "biuf":
                    raise ValueError(f"{path}: nonnumeric per-query array '{key}'")
                arrays[key] = array
        if "pred" not in arrays:
            raise ValueError(f"{path}: query {index} has no prediction array")
        rows[index] = arrays
    if sorted(rows) != list(range(len(rows))):
        raise ValueError(f"{path}: query indices must be contiguous from zero")
    return rows


def compare_predictions(old_path, new_path):
    before, after = prediction_rows(old_path), prediction_rows(new_path)
    mismatches = []
    if before.keys() != after.keys():
        return {"equal": False, "old_queries": len(before), "new_queries": len(after),
                "first_differences": [{"reason": "query indices/count differ"}]}
    for query in sorted(before):
        if before[query].keys() != after[query].keys():
            mismatches.append({"query": query, "reason": "numeric array keys differ"})
            continue
        for key in before[query]:
            left, right = before[query][key], after[query][key]
            if left.shape != right.shape:
                mismatches.append({"query": query, "array": key, "reason": "shape differs",
                                   "old_shape": list(left.shape), "new_shape": list(right.shape)})
            elif not np.array_equal(left, right):
                coordinates = np.argwhere(left != right)
                first = tuple(coordinates[0])
                mismatches.append({"query": query, "array": key,
                                   "position": list(first),
                                   "old": left[first].item(), "new": right[first].item()})
        if len(mismatches) >= 20:
            break
    return {"equal": not mismatches, "old_queries": len(before), "new_queries": len(after),
            "first_differences": mismatches[:20]}


def compare_matrices(old_path, new_path, block_size=256):
    before = np.load(old_path, mmap_mode="r", allow_pickle=False)
    after = np.load(new_path, mmap_mode="r", allow_pickle=False)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("similarity dumps must be 2D matrices")
    if before.dtype != np.float32 or after.dtype != np.float32:
        raise ValueError("A/B requires --dump_sim_dtype float32 on both runs")
    report = {"old_shape": list(before.shape), "new_shape": list(after.shape),
              "old_dtype": str(before.dtype), "new_dtype": str(after.dtype)}
    if before.shape != after.shape:
        return {**report, "equal": False, "max_abs_diff": None,
                "reason": "similarity shapes differ"}
    maximum, first, differing = 0.0, None, 0
    for start in range(0, len(before), block_size):
        left = np.asarray(before[start:start + block_size])
        right = np.asarray(after[start:start + block_size])
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("similarity dumps contain NaN/Inf")
        delta = left.astype(np.float64) - right.astype(np.float64)
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
        differs = left != right
        differing += int(np.count_nonzero(differs))
        if first is None and differs.any():
            row, col = np.argwhere(differs)[0]
            first = {"query": int(start + row), "gallery": int(col),
                     "old": float(left[row, col]), "new": float(right[row, col])}
    return {**report, "equal": differing == 0, "max_abs_diff": maximum,
            "differing_entries": differing, "first_difference": first}


def compare_evaluation(old_dir, new_dir, block_size=256):
    before, after = Path(old_dir), Path(new_dir)
    old_meta = read_json(before / "sim_meta.json")
    new_meta = read_json(after / "sim_meta.json")
    old_metrics, new_metrics = old_meta["raw_metrics"], new_meta["raw_metrics"]
    metrics = {}
    for key in METRIC_KEYS:
        left, right = float(old_metrics[key]), float(new_metrics[key])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"nonfinite metric {key}")
        metrics[key] = {"old": left, "new": right, "difference": right - left,
                        "equal": left == right}
    predictions = compare_predictions(before / "output.json", after / "output.json")
    matrix = compare_matrices(before / "sim_qg.npy", after / "sim_qg.npy", block_size)
    runtime = None
    runtime_files = [before / "ab_runtime_checks.json", after / "ab_runtime_checks.json"]
    if any(path.exists() for path in runtime_files):
        if not all(path.is_file() for path in runtime_files):
            raise ValueError("only one A/B run contains runtime checks")
        old_checks, new_checks = map(read_json, runtime_files)
        fields = ("misc_seed", "test_batch_size", "num_workers", "annotation_hashes",
                  "shuffle_false", "sequential_sampler", "router_mode", "eval_seed")
        runtime = {key: {"old": old_checks[key], "new": new_checks[key],
                         "equal": old_checks[key] == new_checks[key]} for key in fields}
        for checks in (old_checks, new_checks):
            if not checks["shuffle_false"] or not checks["sequential_sampler"]:
                raise ValueError("A/B runtime did not verify sequential evaluation order")
            if checks["image_eval_calls"] < 1:
                raise ValueError("A/B runtime did not verify model.eval during image encoding")
    passed = all(item["equal"] for item in metrics.values()) and predictions["equal"] and matrix["equal"]
    if runtime is not None:
        passed = passed and all(item["equal"] for item in runtime.values())
    return {"status": "PASS" if passed else "FAIL", "gate": "default_eval_numerical_parity",
            "metrics": metrics, "predictions": predictions, "sim_qg": matrix,
            "runtime_checks": runtime,
            "note": "No provenance-byte comparison, numerical tolerance, router change or seed retry."}


def read_trace(path):
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if "DEBUG SKIPPED" in line:
                raise ValueError(f"{path}: GradScaler skipped an overflow iteration; "
                                 "the no-NaN/Inf short-run gate cannot pass")
            marker = "DEBUG ITER "
            if marker not in line:
                continue
            text = line.split(marker, 1)[1].strip()
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: malformed DEBUG ITER JSON: {exc}") from exc
            missing = set(TRACE_KEYS) - record.keys()
            if missing:
                raise ValueError(f"{path}: missing trace fields {sorted(missing)}")
            result.append(record)
    if not result:
        raise ValueError(f"{path}: no DEBUG ITER trace; run the Round A debug path")
    return result


def validate_trace(records, expected_updates, expected_block):
    if len(records) != expected_updates:
        raise ValueError(f"expected exactly {expected_updates} updates, got {len(records)}")
    for number, record in enumerate(records, 1):
        if record["update"] != number:
            raise ValueError(f"nonsequential successful update at row {number}")
        if record["itc_block_size"] != expected_block:
            raise ValueError(f"row {number}: unexpected itc_block_size")
        if record["finite"] is not True:
            raise ValueError(f"row {number}: finite check failed")
        values = [record["total_loss"], record["grad_norm"], record["alpha"],
                  *record["lr"], *record["losses"].values(), *record["loss_weights"].values()]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"row {number}: NaN/Inf in trace")


def compare_training(old_log, new_log, expected_updates):
    baseline, blocked = read_trace(old_log), read_trace(new_log)
    validate_trace(baseline, expected_updates, 0)
    validate_trace(blocked, expected_updates, 80)
    # Different candidates deliberately change losses and gradient norms.
    # Only the unchanged schedule/protocol fields are exact parity targets.
    keys = ("it", "epoch", "lr", "alpha", "niter_per_ep", "loss_weights")
    differences = []
    for number, (left, right) in enumerate(zip(baseline, blocked), 1):
        for key in keys:
            if left[key] != right[key]:
                differences.append({"update": number, "field": key,
                                    "block0": left[key], "block80": right[key]})
    return {"status": "PASS" if not differences else "FAIL", "gate": "short_training_protocol",
            "updates_per_run": expected_updates, "block_sizes": [0, 80],
            "finite": True, "exactly_compared": list(keys),
            "differences": differences,
            "losses_or_gradients_required_equal": False}


def run_mock():
    rng = np.random.default_rng(2026)
    matrix = rng.normal(size=(200, 100)).astype(np.float32)
    all_ranks = np.argsort(-matrix, axis=1)
    ranks = all_ranks[:, :10]
    # Toy metrics are independently derived; no expected retrieval value is hardcoded.
    ids = np.repeat(np.arange(50), 2)
    query_ids = np.arange(200) % 50
    matches = ids[all_ranks] == query_ids[:, None]
    mock_metrics = {
        f"R@{k}": float(100 * np.mean(matches[:, :k].any(axis=1)))
        for k in (1, 5, 10)
    }
    precision = np.cumsum(matches, axis=1) / np.arange(1, matrix.shape[1] + 1)
    mock_metrics["mAP"] = float(100 * np.mean(
        (precision * matches).sum(axis=1) / matches.sum(axis=1)
    ))
    with tempfile.TemporaryDirectory(prefix="tagpr_ab_mock_") as temporary:
        folders = [Path(temporary) / name for name in ("old", "new")]
        for folder in folders:
            folder.mkdir()
            np.save(folder / "sim_qg.npy", matrix)
            records = [{str(i): {"pred": row.tolist(), "match":
                        (ids[row] == query_ids[i]).astype(int).tolist()}}
                       for i, row in enumerate(ranks)]
            (folder / "output.json").write_text(json.dumps(records), encoding="utf-8")
            (folder / "sim_meta.json").write_text(json.dumps({"raw_metrics": mock_metrics}), encoding="utf-8")
        identical = compare_evaluation(*folders)
        assert identical["status"] == "PASS"
        changed = matrix.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
        np.save(folders[1] / "sim_qg.npy", changed)
        divergent = compare_evaluation(*folders)
        assert divergent["status"] == "FAIL" and divergent["sim_qg"]["max_abs_diff"] > 0
        np.save(folders[1] / "sim_qg.npy", matrix)
        altered_predictions = read_json(folders[1] / "output.json")
        altered_predictions[0]["0"]["pred"][0] = (altered_predictions[0]["0"]["pred"][0] + 1) % matrix.shape[1]
        (folders[1] / "output.json").write_text(json.dumps(altered_predictions), encoding="utf-8")
        prediction_difference = compare_evaluation(*folders)
        assert prediction_difference["status"] == "FAIL" and not prediction_difference["predictions"]["equal"]
        (folders[1] / "output.json").write_text((folders[0] / "output.json").read_text(encoding="utf-8"), encoding="utf-8")
        altered_meta = read_json(folders[1] / "sim_meta.json")
        altered_meta["raw_metrics"]["R@1"] = float(np.nextafter(altered_meta["raw_metrics"]["R@1"], np.inf))
        (folders[1] / "sim_meta.json").write_text(json.dumps(altered_meta), encoding="utf-8")
        metric_difference = compare_evaluation(*folders)
        assert metric_difference["status"] == "FAIL" and not metric_difference["metrics"]["R@1"]["equal"]
        log_paths = [Path(temporary) / "block0.log", Path(temporary) / "block80.log"]
        for path, block in zip(log_paths, (0, 80)):
            records = [{"update": i, "it": i - 1, "epoch": 0, "lr": [1e-4 * i],
                        "alpha": i / 10, "niter_per_ep": 20, "itc_block_size": block,
                        "loss_weights": {"nitc_ratio": 1.0}, "losses": {"itc_loss": float(i)},
                        "total_loss": float(i), "grad_norm": 0.5 * i, "finite": True}
                       for i in range(1, 4)]
            path.write_text("\n".join("DEBUG ITER " + json.dumps(row) for row in records), encoding="utf-8")
        train = compare_training(*log_paths, 3)
        assert train["status"] == "PASS"
        records[1]["lr"][0] *= 2
        log_paths[1].write_text("\n".join("DEBUG ITER " + json.dumps(row) for row in records), encoding="utf-8")
        schedule_difference = compare_training(*log_paths, 3)
        assert schedule_difference["status"] == "FAIL"
        records[1]["finite"] = False
        log_paths[1].write_text("\n".join("DEBUG ITER " + json.dumps(row) for row in records), encoding="utf-8")
        try:
            compare_training(*log_paths, 3)
        except ValueError:
            nonfinite_failure = True
        else:
            raise AssertionError("nonfinite trace was not rejected")
        records[1]["finite"] = True
        log_paths[1].write_text("DEBUG SKIPPED overflow at iteration=0\n" +
            "\n".join("DEBUG ITER " + json.dumps(row) for row in records), encoding="utf-8")
        try:
            compare_training(*log_paths, 3)
        except ValueError:
            overflow_failure = True
        else:
            raise AssertionError("skipped overflow was not rejected")
    return {"status": "PASS", "scope": "CPU mock only", "seed": 2026,
            "similarity_shape": [200, 100], "identical_eval": identical["status"],
            "one_ulp_matrix_difference_detected": divergent["status"] == "FAIL",
            "prediction_difference_detected": prediction_difference["status"] == "FAIL",
            "full_precision_metric_difference_detected": metric_difference["status"] == "FAIL",
            "schedule_difference_detected": schedule_difference["status"] == "FAIL",
            "nonfinite_trace_rejected": nonfinite_failure,
            "skipped_overflow_rejected": overflow_failure,
            "mock_short_train_gate": train,
            "training_machine_AB_or_short_training_executed": False}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    evaluation = subparsers.add_parser("eval", help="compare two fp32 eval dumps exactly")
    evaluation.add_argument("--old-dir", required=True)
    evaluation.add_argument("--new-dir", required=True)
    evaluation.add_argument("--block-size", type=int, default=256)
    evaluation.add_argument("--out", default=None)
    training = subparsers.add_parser("train", help="compare short-run DEBUG ITER JSON traces")
    training.add_argument("--old-log", required=True)
    training.add_argument("--new-log", required=True)
    training.add_argument("--expected-updates", type=int, required=True)
    training.add_argument("--out", default=None)
    subparsers.add_parser("mock", help="CPU-only synthetic comparator self-check")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "block_size", 1) < 1 or getattr(args, "expected_updates", 1) < 1:
        parser.error("block size and expected updates must be positive")
    try:
        if args.mode == "eval":
            report = compare_evaluation(args.old_dir, args.new_dir, args.block_size)
        elif args.mode == "train":
            report = compare_training(args.old_log, args.new_log, args.expected_updates)
        else:
            report = run_mock()
        write_report(getattr(args, "out", None), report)
        return 0 if report["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        parser.exit(2, f"comparison error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

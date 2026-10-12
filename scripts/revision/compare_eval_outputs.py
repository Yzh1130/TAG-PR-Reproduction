"""Numerical A/B and short-training gates; provenance is not an acceptance target.

All comparisons are exact. Differences are reported without changing inputs,
rounding results, re-seeding, or switching the router. GPU runs are performed by
the shell entrypoints on the training machine; ``mock`` only tests this utility.
"""

import argparse
import json
import math
from pathlib import Path
import re
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


def read_trace(path, with_calibration=False):
    """Read the whole trace; with_calibration controls evidence return shape only."""
    result, calibration_skips = [], []
    next_it = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if "DEBUG SKIPPED" in line:
                if result:
                    raise ValueError(f"{path}:{line_number}: GradScaler skip after the "
                                     "first successful update is not allowed")
                match = re.search(r"DEBUG SKIPPED overflow at iteration=(\d+)(.*)$", line.strip())
                if match is None:
                    raise ValueError(f"{path}:{line_number}: malformed DEBUG SKIPPED event")
                logged_iteration = int(match.group(1))
                # SKIPPED is logged after it += 1; DEBUG ITER captures it before.
                iteration = logged_iteration - 1
                if iteration != next_it:
                    raise ValueError(f"{path}:{line_number}: calibration skips must be "
                                     "contiguous from logged iteration=1")
                scale_fields = {}
                # Current training logs have no scales. Record only explicit
                # fields if supplied; never infer defaults or presumed halving.
                for field in match.group(2).strip().split():
                    scale_match = re.fullmatch(r"(scale|scale_before|scale_after)=([^\s]+)", field)
                    if scale_match is None or scale_match.group(1) in scale_fields:
                        raise ValueError(f"{path}:{line_number}: unsupported/duplicate skip field {field!r}")
                    value = float(scale_match.group(2))
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"{path}:{line_number}: invalid logged scale")
                    scale_fields[scale_match.group(1)] = value
                calibration_skips.append({
                    "logged_iteration": logged_iteration, "it": iteration,
                    "scale": scale_fields if scale_fields else "scale_not_logged",
                })
                next_it += 1
                continue
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
            if type(record["it"]) is not int or record["it"] != next_it:
                raise ValueError(f"{path}:{line_number}: successful iteration must be "
                                 f"contiguous; expected it={next_it}, got {record['it']!r}")
            result.append(record)
            next_it += 1
    if not result:
        raise ValueError(f"{path}: no DEBUG ITER trace; run the Round A debug path")
    if with_calibration:
        return result, {
            "count": len(calibration_skips), "events": calibration_skips,
            "first_successful_update": {
                "update": result[0]["update"], "it": result[0]["it"],
                "iteration_1based": result[0]["it"] + 1,
            },
            "post_first_success_skip_count": 0,
        }
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
    baseline, calibration0 = read_trace(old_log, with_calibration=True)
    blocked, calibration80 = read_trace(new_log, with_calibration=True)
    validate_trace(baseline, expected_updates, 0)
    validate_trace(blocked, expected_updates, 80)
    # Different candidates deliberately change losses and gradient norms.
    # Only the unchanged schedule/protocol fields are exact parity targets.
    keys = ("it", "epoch", "lr", "alpha", "niter_per_ep", "loss_weights")
    differences = []
    calibration_equal = calibration0["events"] == calibration80["events"]
    if not calibration_equal:
        differences.append({"field": "calibration_skips", "block0": calibration0["events"],
                            "block80": calibration80["events"]})
    for number, (left, right) in enumerate(zip(baseline, blocked), 1):
        for key in keys:
            if left[key] != right[key]:
                differences.append({"update": number, "field": key,
                                    "block0": left[key], "block80": right[key]})
    return {"status": "PASS" if not differences else "FAIL", "gate": "short_training_protocol",
            "updates_per_run": expected_updates, "block_sizes": [0, 80],
            "finite": True, "finite_scope": "successful_optimizer_updates_only",
            "exactly_compared": list(keys),
            "differences": differences,
            "numbering_convention": {
                "canonical": "zero-based attempted iteration: it",
                "DEBUG_SKIPPED": "logged iteration is post-increment; it = logged_iteration - 1",
                "DEBUG_ITER": "it is captured pre-increment; already zero-based",
                "human_iteration": "it + 1",
            },
            "calibration_skips": {
                "rule": "only matching skips before the first successful update are allowed",
                "matched": calibration_equal,
                "block0": calibration0, "block80": calibration80,
                "scale_evidence": "scale_not_logged means unavailable; no numerical scales were inferred",
            },
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
        def write_calibration_trace(path, block, count=6, scale_fields=""):
            mock_records = [{
                "update": i, "it": count + i - 1, "epoch": 0, "lr": [1e-4 * i],
                "alpha": i / 10, "niter_per_ep": 20, "itc_block_size": block,
                "loss_weights": {"nitc_ratio": 1.0}, "losses": {"itc_loss": float(i)},
                "total_loss": float(i), "grad_norm": 0.5 * i, "finite": True,
            } for i in range(1, 4)]
            lines = [f"DEBUG SKIPPED overflow at iteration={i}{scale_fields}"
                     for i in range(1, count + 1)]
            lines.extend("DEBUG ITER " + json.dumps(row) for row in mock_records)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        def require_rejected(text):
            log_paths[1].write_text(text, encoding="utf-8")
            try:
                compare_training(*log_paths, 3)
            except ValueError:
                return True
            raise AssertionError("invalid training trace was not rejected")

        for path, block in zip(log_paths, (0, 80)):
            write_calibration_trace(path, block)
        calibration = compare_training(*log_paths, 3)
        assert calibration["status"] == "PASS" and calibration["calibration_skips"]["matched"]
        for name in ("block0", "block80"):
            evidence = calibration["calibration_skips"][name]
            assert evidence["count"] == 6
            assert [event["logged_iteration"] for event in evidence["events"]] == list(range(1, 7))
            assert [event["it"] for event in evidence["events"]] == list(range(6))
            assert all(event["scale"] == "scale_not_logged" for event in evidence["events"])
            assert evidence["first_successful_update"] == {"update": 1, "it": 6, "iteration_1based": 7}
        valid_text = log_paths[1].read_text(encoding="utf-8")
        write_calibration_trace(log_paths[1], 80, count=5)
        calibration_difference = compare_training(*log_paths, 3)
        assert calibration_difference["status"] == "FAIL"
        assert calibration_difference["differences"][0]["field"] == "calibration_skips"
        invalid_checks = {
            "post_success_skip_rejected": require_rejected(valid_text.replace(
                'DEBUG ITER {"update": 2', 'DEBUG SKIPPED overflow at iteration=8\nDEBUG ITER {"update": 2')),
            "skip_after_final_success_rejected": require_rejected(
                valid_text + "DEBUG SKIPPED overflow at iteration=10\n"),
            "invalid_skip_number_rejected": require_rejected(valid_text.replace("iteration=1\n", "iteration=0\n")),
            "duplicate_skip_rejected": require_rejected(valid_text.replace("iteration=2\n", "iteration=1\n")),
            "skip_gap_rejected": require_rejected(valid_text.replace("iteration=2\n", "iteration=3\n")),
            "first_success_number_mismatch_rejected": require_rejected(valid_text.replace('"it": 6', '"it": 5')),
            "success_gap_rejected": require_rejected(valid_text.replace('"it": 7', '"it": 8')),
            "malformed_skip_rejected": require_rejected(valid_text.replace("iteration=1\n", "iteration=unknown\n")),
        }
        # All unchanged schedule fields still require exact equality after calibration.
        schedule_fields = {}
        for field, replacement in (("epoch", 1), ("lr", [4e-4]), ("alpha", 0.75),
                                   ("niter_per_ep", 21), ("loss_weights", {"nitc_ratio": 0.5})):
            lines = valid_text.splitlines()
            record = json.loads(lines[7].split("DEBUG ITER ", 1)[1])
            record[field] = replacement
            lines[7] = "DEBUG ITER " + json.dumps(record)
            log_paths[1].write_text("\n".join(lines), encoding="utf-8")
            difference = compare_training(*log_paths, 3)
            assert difference["status"] == "FAIL"
            assert any(item["field"] == field for item in difference["differences"])
            schedule_fields[field] = True
        assert require_rejected(valid_text.replace('"finite": true', '"finite": false', 1))
        assert require_rejected(valid_text.replace('"grad_norm": 0.5', '"grad_norm": NaN', 1))
        assert require_rejected(valid_text.replace('"update": 2', '"update": 3', 1))
        try:
            log_paths[1].write_text(valid_text, encoding="utf-8")
            compare_training(*log_paths, 2)
        except ValueError:
            wrong_update_count_rejected = True
        else:
            raise AssertionError("wrong successful update count was not rejected")
        # These are explicitly logged mock scales, not guesses about the real logs.
        for path, block in zip(log_paths, (0, 80)):
            write_calibration_trace(path, block, scale_fields=" scale_before=64 scale_after=32")
        logged_scales = compare_training(*log_paths, 3)
        assert logged_scales["status"] == "PASS"
        assert logged_scales["calibration_skips"]["block0"]["events"][0]["scale"] == {
            "scale_before": 64.0, "scale_after": 32.0,
        }
        write_calibration_trace(log_paths[1], 80, scale_fields=" scale_before=64 scale_after=16")
        assert compare_training(*log_paths, 3)["status"] == "FAIL"
        write_calibration_trace(log_paths[1], 80)
        assert compare_training(*log_paths, 3)["status"] == "FAIL"
    return {"status": "PASS", "scope": "CPU mock only", "seed": 2026,
            "similarity_shape": [200, 100], "identical_eval": identical["status"],
            "one_ulp_matrix_difference_detected": divergent["status"] == "FAIL",
            "prediction_difference_detected": prediction_difference["status"] == "FAIL",
            "full_precision_metric_difference_detected": metric_difference["status"] == "FAIL",
            "schedule_difference_detected": schedule_difference["status"] == "FAIL",
            "nonfinite_trace_rejected": nonfinite_failure,
            "matched_initial_calibration_allowed": calibration["status"] == "PASS",
            "missing_scales_explicitly_recorded": True,
            "calibration_count_difference_detected": calibration_difference["status"] == "FAIL",
            **invalid_checks,
            "schedule_checks_after_calibration": schedule_fields,
            "nonfinite_after_calibration_rejected": True,
            "nonsequential_update_rejected": True,
            "wrong_update_count_rejected": wrong_update_count_rejected,
            "explicit_logged_scales_preserved_and_compared": True,
            "one_sided_scale_evidence_rejected": True,
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

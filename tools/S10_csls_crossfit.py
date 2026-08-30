"""Offline CSLS cross-fitting and calibration-size sensitivity analysis.

The evaluation half is fixed.  Every calibration sample, including each size
scan, comes exclusively from the other half; labels never enter CSLS.
"""

import argparse
from pathlib import Path
import sys

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools._analysis_common import (
    column_topk_mean,
    evaluate,
    load_annotations,
    load_similarity,
    make_mock_annotation,
    make_mock_similarity,
    print_table,
    write_json,
)


SPLIT_SEED = 2026
SCAN_SEEDS = (2026, 2027, 2028)
CALIBRATION_SIZES = (100, 500, 1000, 5000, 8252)


def split_queries(n_query):
    """Return a reproducible, disjoint, equal-size calibration/eval split."""
    if n_query < 2 or n_query % 2:
        raise ValueError("CSLS cross-fitting requires a positive even query count")
    order = np.random.default_rng(SPLIT_SEED).permutation(n_query)
    midpoint = n_query // 2
    return order[:midpoint], order[midpoint:]


def calibration_subset(calibration_indices, size, seed):
    if not 0 < size <= len(calibration_indices):
        raise ValueError(
            f"n_cal={size} must be within [1, {len(calibration_indices)}]"
        )
    # Keep the same order as crossfit, not a new permutation, at the endpoint.
    # This also preserves floating-point top-k reduction order exactly.
    if size == len(calibration_indices):
        return calibration_indices
    return np.random.default_rng(seed).choice(
        calibration_indices, size=size, replace=False
    )


def analyze(scores, annotation, k, calibration_sizes=CALIBRATION_SIZES):
    if k <= 0:
        raise ValueError("--k must be a positive integer")
    calibration, evaluation = split_queries(scores.shape[0])
    if k > len(calibration):
        raise ValueError(
            f"--k={k} exceeds the calibration half size {len(calibration)}"
        )
    for size in calibration_sizes:
        if size < k or size > len(calibration):
            raise ValueError(
                f"n_cal={size} must be at least k={k} and no larger than "
                f"the calibration pool ({len(calibration)})"
            )
    if np.intersect1d(calibration, evaluation).size:
        raise AssertionError("calibration and evaluation queries overlap")

    print(
        f"SPLIT seed={SPLIT_SEED} calibration={len(calibration)} "
        f"evaluation={len(evaluation)} overlap=0; fixed evaluation half"
    )
    raw = evaluate(scores, annotation, query_indices=evaluation)
    crossfit_penalties = column_topk_mean(scores, k, query_indices=calibration)
    crossfit = evaluate(
        scores, annotation, query_indices=evaluation, penalties=crossfit_penalties
    )
    transductive_penalties = column_topk_mean(scores, k)
    transductive = evaluate(
        scores,
        annotation,
        query_indices=evaluation,
        penalties=transductive_penalties,
    )
    settings = {
        "uncorrected": raw,
        "crossfit_csls": crossfit,
        "full_transductive_csls": transductive,
    }
    print_table(
        ["setting", "n_cal", "n_eval", "R@1", "R@5", "R@10", "mAP"],
        [
            [
                name,
                count,
                values["n_query"],
                *[f"{values[key]:.8f}" for key in ("R1", "R5", "R10", "mAP")],
            ]
            for name, count, values in (
                ("uncorrected", 0, raw),
                ("crossfit_csls", len(calibration), crossfit),
                ("full_transductive_csls", scores.shape[0], transductive),
            )
        ],
    )

    scan = []
    for size in calibration_sizes:
        repetitions = []
        for seed in SCAN_SEEDS:
            subset = calibration_subset(calibration, size, seed)
            if size == len(calibration):
                # The exact same set/order gives the exact same estimator.
                # Reuse it to avoid recomputing three identical reductions.
                if not np.array_equal(subset, calibration):
                    raise AssertionError("full calibration subset changed order")
                penalties = crossfit_penalties
            else:
                penalties = column_topk_mean(scores, k, query_indices=subset)
            metrics = evaluate(
                scores,
                annotation,
                query_indices=evaluation,
                penalties=penalties,
                r1_only=True,
            )
            r1 = float(metrics["R1"])
            if size == len(calibration) and r1 != crossfit["R1"]:
                raise AssertionError(
                    "full-calibration scan R@1 differs from crossfit: "
                    f"{r1!r} != {crossfit['R1']!r}"
                )
            repetitions.append({"seed": seed, "R1": r1})
        values = np.asarray([row["R1"] for row in repetitions], dtype=np.float64)
        scan.append(
            {
                "n_cal": int(size),
                "R1_mean": float(values.mean()),
                "R1_std": float(values.std(ddof=0)),
                "repetitions": repetitions,
                "matches_crossfit_exactly": (
                    bool(np.all(values == crossfit["R1"]))
                    if size == len(calibration)
                    else None
                ),
            }
        )
    print_table(
        ["n_cal", "R@1 mean", "R@1 std (ddof=0)", "seeds", "endpoint exact"],
        [
            [
                row["n_cal"],
                f"{row['R1_mean']:.8f}",
                f"{row['R1_std']:.8f}",
                "/".join(map(str, SCAN_SEEDS)),
                row["matches_crossfit_exactly"],
            ]
            for row in scan
        ],
    )
    return {
        "tool": "S10_csls_crossfit",
        "k": int(k),
        "split_seed": SPLIT_SEED,
        "scan_seeds": list(SCAN_SEEDS),
        "n_query": int(scores.shape[0]),
        "n_gallery": int(scores.shape[1]),
        "n_calibration_pool": int(len(calibration)),
        "n_evaluation": int(len(evaluation)),
        "calibration_query_indices": calibration.tolist(),
        "evaluation_query_indices": evaluation.tolist(),
        "calibration_evaluation_overlap": 0,
        "protocol": (
            "Query-level equal split using numpy.default_rng(2026).permutation. "
            "All scans sample without replacement exclusively from the initial "
            "calibration half; the evaluation half and full gallery stay fixed. "
            "No source image or same-identity image is excluded. Calibration "
            "uses no labels; CSLS = 2 * sim_qg - column_top_k_mean. "
            "All metrics are percentages; scan standard deviation uses ddof=0."
        ),
        "settings": settings,
        "calibration_scan": scan,
    }


def self_test():
    annotation = make_mock_annotation()
    scores = make_mock_similarity(annotation)
    # Add an artificial hub so the test exercises a nontrivial correction,
    # rather than merely confirming three already-perfect rankings.
    scores[:, 0] += np.float32(2.7)
    if scores.shape != (20, 10):
        raise AssertionError(f"expected mock shape (20, 10), got {scores.shape}")
    result = analyze(scores, annotation, k=2, calibration_sizes=(2, 5, 10))
    calibration = np.asarray(result["calibration_query_indices"])
    evaluation = np.asarray(result["evaluation_query_indices"])
    for size in (2, 5, 10):
        for seed in SCAN_SEEDS:
            subset = calibration_subset(calibration, size, seed)
            if len(np.unique(subset)) != size:
                raise AssertionError("calibration scan sampled with replacement")
            if not np.isin(subset, calibration).all():
                raise AssertionError("calibration scan leaked evaluation queries")
            if np.intersect1d(subset, evaluation).size:
                raise AssertionError("calibration/evaluation overlap")
    k = result["k"]
    manual_penalties = np.sort(scores[calibration], axis=0)[-k:].mean(axis=0)
    penalties = column_topk_mean(scores, k, query_indices=calibration)
    error = float(np.max(np.abs(manual_penalties - penalties)))
    if not np.allclose(manual_penalties, penalties, rtol=0, atol=1e-7):
        raise AssertionError(f"top-k penalty mismatch: {error}")
    for name, correction in (
        ("uncorrected", None),
        ("crossfit_csls", manual_penalties),
        ("full_transductive_csls", np.sort(scores, axis=0)[-k:].mean(axis=0)),
    ):
        subset_scores = scores[evaluation]
        if correction is not None:
            subset_scores = 2 * subset_scores - correction
        # Full argsort uses the same descending-score tie policy as the metric
        # helper; this independently computes the ID match on the first rank.
        order = np.argsort(-subset_scores, axis=1, kind="stable")
        correct = (
            annotation.gallery_ids[order[:, 0]] == annotation.query_ids[evaluation]
        )
        manual_r1 = float(correct.mean() * 100.0)
        actual_r1 = result["settings"][name]["R1"]
        if abs(manual_r1 - actual_r1) > 1e-12:
            raise AssertionError(f"{name}: manual R1={manual_r1} != {actual_r1}")
    print(f"SELF_TEST top_k_mean_max_abs_error={error:.17g}")
    print("SELF_TEST calibration/evaluation overlap=0; all scan samples in calibration pool")
    print("SELF_TEST n_cal=10 equals crossfit exactly for all three seeds")
    print("SELF_TEST PASS: S10 (20 queries x 10 gallery)")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim_qg", type=Path, help="raw exported sim_qg.npy")
    parser.add_argument("--ann", type=Path, help="TAG-PEDES test_reid.json")
    parser.add_argument("--k", type=int, default=10, help="CSLS neighbors (default: 10)")
    parser.add_argument("--out", type=Path, help="destination results JSON")
    parser.add_argument(
        "--self_test", action="store_true", help="run a deterministic 20 x 10 mock test"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            result = self_test()
        else:
            missing = [flag for flag in ("sim_qg", "ann", "out") if getattr(args, flag) is None]
            if missing:
                parser.error("required for analysis: " + ", ".join("--" + key for key in missing))
            annotation = load_annotations(args.ann, enforce_anchors=True)
            scores = load_similarity(args.sim_qg, annotation)
            result = analyze(scores, annotation, args.k)
            result["sim_qg_path"] = str(args.sim_qg.resolve())
            result["annotation_path"] = str(args.ann.resolve())
            result["annotation_summary"] = annotation.summary
        if args.out is not None:
            write_json(args.out, result)
            print(f"Wrote {args.out.resolve()}")
    except (OSError, ValueError, TypeError, KeyError, MemoryError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

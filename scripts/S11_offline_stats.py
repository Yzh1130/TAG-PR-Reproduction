"""S11: read-only CPU statistics for archived TAG-PEDES similarities."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from ._offline_common import (compact_metrics, eligible_queries, evaluate, json_ready,
                                  load_annotations, load_matrix, load_similarity, print_table,
                                  provenance, report_stem, tee_log, topk_mean)
    from ._s11_descriptive import paired_test, run_descriptive
    from ._s11_groups import run_grouped
except ImportError:
    from _offline_common import (compact_metrics, eligible_queries, evaluate, json_ready,
                                 load_annotations, load_matrix, load_similarity, print_table,
                                 provenance, report_stem, tee_log, topk_mean)
    from _s11_descriptive import paired_test, run_descriptive
    from _s11_groups import run_grouped


def common_query_sets(ann):
    """Intersection of legal queries, NOT an intersection selected by score."""
    galleries = [np.flatnonzero(ann.gallery_views == v) for v in (0, 1)]
    sets, report = {}, {}
    for view, name in ((0, "A"), (1, "G")):
        q = np.flatnonzero((ann.query_views == view) & np.isin(ann.query_ids, ann.dual_ids))
        legal = [eligible_queries(ann, q, g, exclude_source=True)[0] for g in galleries]
        common = np.intersect1d(*legal)
        sets[name] = common
        report[name] = dict(input_dual_view_queries=len(q), valid_to_A=len(legal[0]),
                            valid_to_G=len(legal[1]), common_queries=len(common),
                            excluded_from_common=len(q) - len(common), query_indices=common)
    return sets, galleries, report


def sample_gallery(ann, queries, candidates, size, rng, same_view):
    """Positive-preserving, label-stratified gallery sample, never score-based.

    Same-view samples retain every source image, then exclusion removes exactly
    one candidate per query. Thus matched protocols have equal EFFECTIVE size,
    not nominal sizes that differ by one after exclusion.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if not 1 <= size <= len(candidates):
        raise ValueError(f"cannot downsample {len(candidates)} gallery images to {size}")
    required = set(ann.query_source_indices[queries].tolist()) if same_view else set()
    for pid in np.unique(ann.query_ids[queries]):
        available = candidates[ann.gallery_ids[candidates] == pid]
        minimum = 2 if same_view else 1
        if len(available) < minimum:
            raise ValueError(f"identity {pid}: insufficient positives after source exclusion")
        have = sum(int(g) in required for g in available)
        if have < minimum:
            pool = np.asarray([g for g in available if int(g) not in required])
            required.update(rng.choice(pool, minimum - have, replace=False).tolist())
    if len(required) > size:
        raise ValueError(f"sample size {size} cannot retain {len(required)} required source/positive images")
    if not required.issubset(set(candidates.tolist())):
        raise ValueError("required source image is outside target view")
    pool = candidates[~np.isin(candidates, list(required))]
    fill = rng.choice(pool, size - len(required), replace=False)
    selected = np.sort(np.asarray(list(required) + fill.tolist(), dtype=np.int64))
    eligible, skipped = eligible_queries(ann, queries, selected, exclude_source=True)
    if skipped or not np.array_equal(eligible, queries):
        raise AssertionError("positive-preserving gallery sampling lost a legal query")
    return selected


def run_controlled(sim, ann, mode="literal", expected_common=None,
                   seeds=tuple(range(2026, 2036)), bootstrap=10000):
    if mode not in ("literal", "common-min"):
        raise ValueError("gallery control must be literal or common-min")
    sets, galleries, counts = common_query_sets(ann)
    print("COMMON QUERY SETS " + json.dumps({k: {n: v for n, v in r.items() if n != "query_indices"}
                                           for k, r in counts.items()}))
    if expected_common is not None:
        actual = [len(sets["A"]), len(sets["G"])]
        if actual != list(expected_common):
            raise ValueError(f"common-query count check failed: got {actual}, command-line expected {expected_common}")
    if not len(sets["A"]) or not len(sets["G"]):
        raise ValueError("controlled evaluation needs legal common queries in both source views")
    original, original_details = {}, {}
    for name, source in (("A", 0), ("G", 1)):
        for target, target_name in ((source, name), (1 - source, "G" if source == 0 else "A")):
            key = name + "->" + target_name
            result = evaluate(sim, ann, queries=sets[name], gallery=galleries[target], exclude_source=True)
            original[key], original_details[key] = compact_metrics(result), result
    comparisons, controlled, notices = {}, {}, []
    for name, source in (("A", 0), ("G", 1)):
        same_key, cross_key = name + "->" + name, name + "->" + ("G" if source == 0 else "A")
        comparisons[name] = paired_test(original_details[same_key]["details"]["correct"],
                                        original_details[cross_key]["details"]["correct"],
                                        repeats=bootstrap, seed=2026)
        same_available, cross_available = len(galleries[source]) - 1, len(galleries[1 - source])
        if mode == "literal" and cross_available < same_available:
            message = (f"{cross_key}: literal downsampling impossible: target has {cross_available} images, "
                       f"same-view protocol has {same_available} candidates AFTER source exclusion. "
                       "No upsampling/duplicate gallery invented. With approval, use --gallery-control common-min.")
            print("PROTOCOL WARNING: " + message)
            notices.append(message)
            controlled[name] = dict(status="infeasible_requested_downsampling", reason=message)
            continue
        target_size = (same_available if mode == "literal" else min(same_available, cross_available))
        trials = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            same_g = (galleries[source] if target_size == same_available else
                      sample_gallery(ann, sets[name], galleries[source], target_size + 1, rng, True))
            cross_g = sample_gallery(ann, sets[name], galleries[1 - source], target_size, rng, False)
            same = evaluate(sim, ann, queries=sets[name], gallery=same_g, exclude_source=True)
            cross = evaluate(sim, ann, queries=sets[name], gallery=cross_g, exclude_source=True)
            if not (same["gallery_size_min"] == same["gallery_size_max"] ==
                    cross["gallery_size_min"] == cross["gallery_size_max"] == target_size):
                raise AssertionError("effective gallery-size equality failed")
            trials.append(dict(seed=seed, same=compact_metrics(same), cross=compact_metrics(cross),
                               same_gallery_indices=same_g, cross_gallery_indices=cross_g,
                               paired=paired_test(same["details"]["correct"], cross["details"]["correct"],
                                                  repeats=bootstrap, seed=seed)))
        summary = {direction: {m: dict(mean=float(np.mean([t[direction][m] for t in trials])),
                                      std=float(np.std([t[direction][m] for t in trials], ddof=0)))
                               for m in ("R1", "R5", "R10", "mAP")} for direction in ("same", "cross")}
        controlled[name] = dict(status="complete", effective_gallery_size=target_size, trials=trials,
                                metrics_mean_std=summary,
                                inference="paired query bootstrap/McNemar separately per gallery draw; no pooled pseudo-replication")
    rows = [[key, "original", r["n_query"], f'{r["gallery_size_min"]}..{r["gallery_size_max"]}',
             *[f'{r[m]:.5f}' for m in ("R1", "R5", "R10", "mAP")]] for key, r in original.items()]
    for name, c in controlled.items():
        if c["status"] != "complete":
            continue
        for side, key in (("same", name + "->" + name), ("cross", name + "->" + ("G" if name == "A" else "A"))):
            rows.append([key, "size-controlled", len(sets[name]), c["effective_gallery_size"],
                         *[f'{c["metrics_mean_std"][side][m]["mean"]:.5f}+/-{c["metrics_mean_std"][side][m]["std"]:.5f}'
                           for m in ("R1", "R5", "R10", "mAP")]])
    print_table(["Protocol", "Gallery", "Queries", "Candidates/query", "R1", "R5", "R10", "mAP"], rows)
    paired_rows = []
    for name, paired in comparisons.items():
        paired_rows.append([name, "original", paired["observed_delta_pp"], paired["ci95_percentile_pp"],
                            paired["p_two_sided"], paired["mcnemar"]["p_two_sided"]])
    for name, control in controlled.items():
        for trial in control.get("trials", []):
            paired = trial["paired"]
            paired_rows.append([name, f'size-controlled seed={trial["seed"]}', paired["observed_delta_pp"],
                                paired["ci95_percentile_pp"], paired["p_two_sided"], paired["mcnemar"]["p_two_sided"]])
    print_table(["Source", "Setting", "Cross-minus-same(pp)", "Percentile95(pp)", "Bootstrap p", "McNemar p"], paired_rows)
    return dict(common_query_sets=counts, expected_common_from_cli=expected_common, original=original,
                original_paired_cross_minus_same=comparisons, gallery_control=controlled, mode=mode,
                notices=notices, seeds=list(seeds), std_ddof=0,
                sampling="preserve positives; same-view subsets also preserve every query source; no similarity-based sampling")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="read-only raw float32 sim_qg.npy")
    parser.add_argument("--anno", required=True, help="test image-record JSON in dump order")
    parser.add_argument("--out", required=True, help="new analysis directory (never _archive_evidence)")
    parser.add_argument("--sim-gg", help="optional read-only sim_gg.npy: validate and record provenance")
    parser.add_argument("--tag", default="S11")
    parser.add_argument("--gallery-control", choices=("literal", "common-min"), default="literal",
                        help="literal reports impossible upward downsampling; common-min is an explicit alternative protocol")
    parser.add_argument("--expected-common", nargs=2, type=int, metavar=("A_QUERIES", "G_QUERIES"),
                        help="optional supplied count check; no hardcoded score/count gate")
    return parser


def print_descriptive_tables(report):
    print("B: error categories and Wilson 95% intervals (fractions, not percentages)")
    rows = []
    for setting, result in report["B_error_decomposition"].items():
        for group, stratum in result["strata"].items():
            for cell, values in stratum["cells"].items():
                rows.append([setting, group, cell, values["count"], values["denominator"],
                             values["proportion"], values["ci95"]])
            for condition, values in stratum["conditional_probabilities"].items():
                rows.append([setting, group, condition, values["count"], values["denominator"],
                             values["proportion"], values["ci95"]])
    print_table(["Setting", "Stratum", "Event", "Count", "Denominator", "Fraction", "Wilson95"], rows)
    print("C: cross-modal reciprocal neighbors")
    print_table(["k", "Pairs", "Fraction(query top-k)", "Mean/query", "Histogram"],
                [[k, r["reciprocal_pairs"], r["fraction_of_query_topk_candidates"],
                  r["mean_reciprocal_per_query"], r["per_query_distribution"]]
                 for k, r in report["C_reciprocal"]["results"].items()])
    print("D: hubness N1, including zero-count gallery samples")
    hubness = report["D_hubness"]
    print_table(["Setting", "Queries", "skew(N1)", "Gini", "Head1% share", "Head5% share"],
                [[setting, subset, r["skewness_N1"], r["gini_N1"],
                  r["head_1pct"]["query_share"], r["head_5pct"]["query_share"]]
                 for setting in ("raw", "csls_k10") for subset in ("all_queries", "errors_only")
                 for r in [hubness[setting][subset]]])
    print(f'Delta skew={hubness["delta_skewness_all_queries"]}; Delta R1={hubness["delta_R1_pp"]} pp')
    print("E: fixed-penalty paired bootstrap (10,000 resamples)")
    print_table(["Unit", "Delta mean(pp)", "Percentile95(pp)", "Two-sided sign-tail p"],
                [[unit, r["bootstrap_mean_delta_pp"], r["ci95_percentile_pp"], r["p_two_sided"]]
                 for unit, r in report["E_bootstrap"]["results"].items()])


def run(args):
    ann, start = load_annotations(args.anno), time.perf_counter()
    sim = load_similarity(args.sim, ann)
    report = dict(task="S11", annotation=ann.summary, input=provenance(args.sim),
                  precision=str(sim.dtype), seeds=dict(bootstrap=2026, gallery=list(range(2026, 2036)),
                  scans=[2026, 2027, 2028], half_select=list(range(2026, 2031))))
    print("ANNOTATION " + json.dumps(ann.summary))
    if sim.dtype == np.float16:
        print("WARNING: float16 ties may differ from the original float32 ranking; no performance gate is imposed.")
    if args.sim_gg:
        gg = load_matrix(args.sim_gg, (sim.shape[1], sim.shape[1]))
        report["gallery_input"] = provenance(args.sim_gg)
        report["gallery_input"]["used_in_S11"] = False
        del gg
    print("A: controlled queries and galleries")
    report["A"] = run_controlled(sim, ann, args.gallery_control, args.expected_common)
    print("B-E: original and fixed-full-query CSLS(k=10) outcomes; bootstrap never refits r(g)")
    raw, penalties = evaluate(sim, ann), topk_mean(sim, 10)
    corrected = evaluate(sim, ann, penalty=penalties, alpha=0.5)
    report["raw"], report["csls_k10"] = compact_metrics(raw), compact_metrics(corrected)
    report.update(run_descriptive(sim, ann, raw, corrected, penalties))
    print_descriptive_tables(report)
    print("F-G: grouped crossfit, whole-identity calibration scan, held-out k selection")
    report.update(run_grouped(sim, ann, penalties))
    report["elapsed_sec"] = time.perf_counter() - start
    report["status"] = "partial_protocol_infeasible" if report["A"]["notices"] else "complete"
    print_table(["Setting", "R1", "R5", "R10", "mAP"],
                [[name, *[f'{r[m]:.6f}' for m in ("R1", "R5", "R10", "mAP")]]
                 for name, r in (("raw", raw), ("CSLS(k=10)", corrected))])
    print(f'S11 status={report["status"]}; elapsed_sec={report["elapsed_sec"]:.3f}')
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        proposed_out = Path(args.out).expanduser().resolve()
        for source in (args.sim, args.sim_gg):
            if source is not None:
                directory = Path(source).expanduser().resolve().parent
                if proposed_out == directory or directory in proposed_out.parents:
                    raise ValueError("--out must not be inside a read-only similarity archive directory")
        stem = report_stem(args.out, args.tag)
        json_path, log_path = Path(str(stem) + ".json"), Path(str(stem) + ".log")
        with tee_log(log_path):
            try:
                result = run(args)
            except (OSError, ValueError, AssertionError, RuntimeError) as exc:
                result = dict(status="failed", error=str(exc), arguments=vars(args))
                print("ERROR: " + str(exc))
                with json_path.open("x", encoding="utf-8") as handle:
                    json.dump(json_ready(result), handle, indent=2, ensure_ascii=False, allow_nan=False)
                raise
            with json_path.open("x", encoding="utf-8") as handle:
                json.dump(json_ready(result), handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            print(f"JSON: {json_path}\nLOG: {log_path}")
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError) as exc:
        print(f"S11 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

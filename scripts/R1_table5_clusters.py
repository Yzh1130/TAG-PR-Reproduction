"""Round B / R1-2: audited Table-5 paired identity-cluster intervals.

CPU-only, read-only inputs. The authoritative S11 common queries and gallery
draws are restored before any new intervals are computed. Zero-positive
policies are reported separately; this script does not select a main policy.
"""

import argparse
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from ._analysis_audit import _exact
    from ._offline_common import (compact_metrics, eligible_queries, evaluate,
                                  indices, load_annotations, load_similarity,
                                  print_table)
    from .S11_offline_stats import common_query_sets, sample_gallery
except ImportError:
    from _analysis_audit import _exact
    from _offline_common import (compact_metrics, eligible_queries, evaluate,
                                 indices, load_annotations, load_similarity,
                                 print_table)
    from S11_offline_stats import common_query_sets, sample_gallery


METRICS = ("R1", "R5", "R10", "mAP")


def exact_gate(checks, name, actual, expected):
    passed = _exact(actual, expected)
    checks.append(dict(name=name, passed=passed, actual=actual, expected=expected))
    if not passed:
        raise RuntimeError(f"exact S11 gate FAILED: {name}: {actual!r} != {expected!r}")


def query_distribution(ann, queries):
    labels, counts = np.unique(ann.query_ids[queries], return_counts=True)
    return dict(n_queries=len(queries), n_identities=len(labels),
                identity_ids=labels.tolist(),
                query_counts_by_identity={str(k): int(v) for k, v in zip(labels, counts)},
                query_count_distribution={str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())})


def pair_outcomes(sim, ann, query, same_gallery, cross_gallery):
    """Call the unchanged S11 evaluator; scatter eligible results back to Q.

    Scattering zeros is ONLY the explicit fixed-denominator policy. The valid
    masks preserve the evaluator's actual zero-positive exclusion semantics.
    """
    query = indices(query, len(ann.query_ids))
    result, vectors, valid = {}, {}, {}
    for name, gallery in (("same", same_gallery), ("cross", cross_gallery)):
        gallery = indices(gallery, len(ann.gallery_ids))
        evaluation = evaluate(sim, ann, queries=query, gallery=gallery, exclude_source=True)
        eligible = evaluation["details"]["query_indices"]
        lookup = {int(q): i for i, q in enumerate(query)}
        locations = np.asarray([lookup[int(q)] for q in eligible], np.int64)
        if not np.array_equal(query[locations], eligible):
            raise AssertionError("evaluation changed the requested query order")
        hit, mask = np.zeros(len(query), bool), np.zeros(len(query), bool)
        hit[locations] = evaluation["details"]["correct"]
        mask[locations] = True
        lost = query[~mask]
        result[name] = dict(metrics=compact_metrics(evaluation),
                            gallery_indices=gallery.tolist(),
                            zero_positive_queries=len(lost),
                            zero_positive_identities=int(np.unique(ann.query_ids[lost]).size),
                            zero_positive_query_indices=lost.tolist(),
                            zero_positive_identity_ids=np.unique(ann.query_ids[lost]).tolist(),
                            correct_by_requested_query=hit.tolist(),
                            has_legal_positive_by_requested_query=mask.tolist())
        vectors[name], valid[name] = hit, mask
    both = valid["same"] & valid["cross"]
    result["common_valid_queries"] = int(both.sum())
    result["excluded_by_either_protocol"] = int((~both).sum())
    return result, vectors["cross"].astype(np.float64) - vectors["same"], both


def restore_s11(sim, ann, s11, checks):
    """Saved indices take priority; seed replay is used only if absent."""
    saved = s11["A"]
    common, galleries, counts = common_query_sets(ann)
    seeds = s11["seeds"]["gallery"]
    if len(seeds) != 10 or len(set(seeds)) != len(seeds) or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("S11 seeds.gallery must contain ten distinct nonnegative integer seeds")
    if "seeds" in saved:
        exact_gate(checks, "S11.A.seeds", saved["seeds"], seeds)
    verified = {}
    for name, source in (("A", 0), ("G", 1)):
        expected_queries = saved["common_query_sets"][name]
        for key, value in counts[name].items():
            if key in expected_queries:
                actual = value.tolist() if isinstance(value, np.ndarray) else value
                exact_gate(checks, f"common.{name}.{key}", actual, expected_queries[key])
        query = common[name]
        if not len(query):
            raise ValueError(f"no common legal queries for {name}")
        original, difference, both = pair_outcomes(sim, ann, query, galleries[source], galleries[1-source])
        for direction, key in (("same", name + "->" + name),
                               ("cross", name + "->" + ("G" if name == "A" else "A"))):
            reference = saved["original"][key]
            if reference.get("exclude_source") is not True:
                raise ValueError(f"S11 {key} does not use the required source-image exclusion")
            for metric in METRICS:
                exact_gate(checks, f"original.{key}.{metric}", original[direction]["metrics"][metric], reference[metric])
        if not both.all():
            raise AssertionError("common-query restoration produced an invalid original query")
        control = saved["gallery_control"][name]
        if control["status"] != "complete":
            raise ValueError(f"S11 {name} gallery control is not complete: {control['status']}")
        target = int(control["effective_gallery_size"])
        trials = control["trials"]
        exact_gate(checks, f"{name}.trial_seeds", [trial["seed"] for trial in trials], seeds)
        draws = []
        for trial in trials:
            draw_seed = trial["seed"]
            keys = ("same_gallery_indices", "cross_gallery_indices")
            replay = None
            if any(key not in trial for key in keys):
                rng = np.random.default_rng(draw_seed)
                same = (galleries[source] if target == len(galleries[source]) - 1 else
                        sample_gallery(ann, query, galleries[source], target + 1, rng, True))
                cross = sample_gallery(ann, query, galleries[1-source], target, rng, False)
                replay = (same, cross)
            selected, origins = [], {}
            for j, direction in enumerate(("same", "cross")):
                key = keys[j]
                gallery = indices(trial[key], len(ann.gallery_ids)) if key in trial else replay[j]
                if not np.all(ann.gallery_views[gallery] == (source if j == 0 else 1-source)):
                    raise ValueError(f"S11 {name} seed={draw_seed} {direction} contains the wrong gallery view")
                selected.append(gallery)
                origins[direction] = "saved_indices" if key in trial else "same_seed_same_S11_code_replay"
            detail, delta, valid = pair_outcomes(sim, ann, query, *selected)
            for direction in ("same", "cross"):
                for metric in METRICS:
                    exact_gate(checks, f"{name}.seed{draw_seed}.{direction}.{metric}",
                               detail[direction]["metrics"][metric], trial[direction][metric])
                # Historical S11 preserves sources so both EFFECTIVE sizes
                # equal target. Verify the saved metric-size evidence exactly.
                for field in ("n_query", "skipped_no_positive", "gallery_size_min", "gallery_size_max"):
                    if field in trial[direction]:
                        exact_gate(checks, f"{name}.seed{draw_seed}.{direction}.{field}",
                                   detail[direction]["metrics"][field], trial[direction][field])
            draws.append(dict(seed=draw_seed, detail=detail, difference=delta,
                              valid=valid, index_source=origins))
        verified[name] = dict(query=query, source=source, original=original,
                              original_difference=difference, draws=draws, target=target)
    return verified, galleries, list(seeds)


def summarize_bootstrap(values, observed, seed, repeats, n_identities, n_queries, two_stage):
    invalid = int((~np.isfinite(values)).sum())
    return dict(observed_delta_pp=float(observed) if np.isfinite(observed) else None,
                bootstrap_mean_delta_pp=float(values.mean()) if not invalid else None,
                ci95_percentile_pp=np.percentile(values, [2.5, 97.5]).tolist() if not invalid else None,
                undefined_replicates=invalid, repeats=repeats, seed=seed,
                n_identities=n_identities, n_queries=n_queries,
                unit="identity_and_gallery_draw" if two_stage else "identity",
                estimator="mean per-query cross-minus-same over eligible fixed draws; query-weighted whole-identity bootstrap",
                empty_policy="no redraw or silent deletion; undefined replicate makes the interval null")


def identity_interval(difference, valid, labels, repeats=10000, seed=2026, two_stage=False):
    """Per-query average across fixed draws, then whole-identity bootstrap.

    For the common-valid policy, a query's average uses only draws where BOTH
    protocols have a positive; queries absent from all draws have zero weight.
    This is explicitly not the mean of per-draw ratios with varying denominators.
    The sensitivity resamples both draws and identities, recomputing those query
    averages. Empty denominators are recorded instead of silently redrawn.
    """
    difference = np.asarray(difference, np.float64)
    valid = np.asarray(valid, bool)
    labels = np.asarray(labels)
    if difference.ndim == 1:
        difference, valid = difference[None, :], valid[None, :]
    if difference.shape != valid.shape or labels.shape != (difference.shape[1],) or not len(labels):
        raise ValueError("need matching [draws, queries] values/masks and one identity per query")
    if not np.isfinite(difference).all() or repeats < 1 or seed < 0:
        raise ValueError("bootstrap inputs must be finite; repeats positive and seed nonnegative")
    unique, inverse = np.unique(labels, return_inverse=True)
    n_draws = len(difference)
    masked = difference * valid
    counts = valid.sum(0)
    per_query = np.divide(masked.sum(0), counts, out=np.zeros(len(labels)), where=counts > 0)
    active = counts > 0
    observed = per_query[active].mean() * 100 if active.any() else np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    # Optimize the ordinary fixed-draw identity bootstrap by reducing queries
    # to their identity sums/counts once, without changing the weighting.
    num = np.bincount(inverse, weights=per_query * active, minlength=len(unique))
    den = np.bincount(inverse, weights=active, minlength=len(unique))
    if not two_stage:
        for start in range(0, repeats, 64):
            stop = min(repeats, start + 64)
            chosen = rng.integers(0, len(unique), (stop-start, len(unique)))
            denominator = den[chosen].sum(1)
            numerator = num[chosen].sum(1) * 100
            estimates[start:stop] = np.divide(numerator, denominator,
                                             out=np.full(stop-start, np.nan), where=denominator > 0)
    else:
        for j in range(repeats):
            chosen_ids = rng.integers(0, len(unique), len(unique))
            multiplicity = np.bincount(chosen_ids, minlength=len(unique))[inverse]
            chosen_draws = rng.integers(0, n_draws, n_draws)
            draw_multiplicity = np.bincount(chosen_draws, minlength=n_draws)
            nvalid = draw_multiplicity @ valid
            totals = draw_multiplicity @ masked
            means = np.divide(totals, nvalid, out=np.zeros(len(labels)), where=nvalid > 0)
            weight = multiplicity * (nvalid > 0)
            estimates[j] = (np.sum(means * weight) / weight.sum() * 100) if weight.sum() else np.nan
    return summarize_bootstrap(estimates, observed, seed, repeats, len(unique), int(active.sum()), two_stage)


def at_least_two_gallery(ann, query, candidates, size, rng, retain_sources):
    """Source-preserving same-view and >=2 positives/identity in both views.

    Selection is label-stratified, NOT a uniform-gallery draw. It raises the
    positive-retention rate; insufficient originals/size is an error, never
    repaired by duplication or replacing the rule with >=1.
    """
    candidates = np.asarray(candidates, np.int64)
    if not 1 <= size <= len(candidates):
        raise ValueError(f"cannot sample {size} originals from {len(candidates)} candidates")
    required = set(ann.query_source_indices[query].tolist()) if retain_sources else set()
    for pid in np.unique(ann.query_ids[query]):
        available = candidates[ann.gallery_ids[candidates] == pid]
        if len(available) < 2:
            raise ValueError(f"identity {pid} has only {len(available)} target-view images; >=2 is infeasible")
        retained = sum(int(g) in required for g in available)
        if retained < 2:
            pool = available[~np.isin(available, list(required))]
            required.update(rng.choice(pool, 2-retained, replace=False).tolist())
    if not required.issubset(set(candidates.tolist())) or len(required) > size:
        raise ValueError(f"{size} slots cannot retain {len(required)} required source/positive images")
    pool = candidates[~np.isin(candidates, list(required))]
    fill = rng.choice(pool, size-len(required), replace=False)
    selected = np.sort(np.asarray([*required, *fill.tolist()], np.int64))
    legal, lost = eligible_queries(ann, query, selected, exclude_source=True)
    if lost or not np.array_equal(legal, query):
        raise AssertionError(">=2-positive sampler lost a common legal query")
    return selected


def policy_summary(draws, query_ids, policy, repeats, seed):
    differences = np.vstack([d["difference"] for d in draws])
    valid = np.vstack([d["valid"] for d in draws])
    mask = valid if policy == "b_common_valid_queries" else np.ones_like(valid)
    per_draw = []
    for draw, diff, use in zip(draws, differences, mask):
        same_correct = np.asarray(draw["detail"]["same"]["correct_by_requested_query"], bool)
        cross_correct = np.asarray(draw["detail"]["cross"]["correct_by_requested_query"], bool)
        per_draw.append(dict(seed=draw["seed"], denominator=int(use.sum()),
                             paired_delta_pp=float(diff[use].mean()*100) if use.any() else None,
                             same_R1_percent_float64=float(same_correct[use].mean()*100) if use.any() else None,
                             cross_R1_percent_float64=float(cross_correct[use].mean()*100) if use.any() else None,
                             excluded=int((~use).sum()),
                             zero_positive_queries_same=draw["detail"]["same"]["zero_positive_queries"],
                             zero_positive_queries_cross=draw["detail"]["cross"]["zero_positive_queries"],
                             zero_positive_identities_same=draw["detail"]["same"]["zero_positive_identities"],
                             zero_positive_identities_cross=draw["detail"]["cross"]["zero_positive_identities"]))
    values = np.asarray([d["paired_delta_pp"] if d["paired_delta_pp"] is not None else np.nan for d in per_draw])
    finite = np.isfinite(values).all()
    return dict(status="complete", per_draw=per_draw,
                per_query_average_policy=("average only jointly-valid draws, omit a query only if invalid in ALL draws"
                                          if policy == "b_common_valid_queries" else "fixed common queries; lost positives count as misses"),
                identity_bootstrap_conditional_on_fixed_draws=identity_interval(differences, mask, query_ids, repeats, seed),
                two_stage_identity_and_gallery_bootstrap=identity_interval(differences, mask, query_ids, repeats, seed, True),
                mean_of_per_draw_paired_deltas_pp=float(values.mean()) if finite else None,
                std_of_per_draw_paired_deltas_pp=float(values.std(ddof=0)) if finite else None,
                range_of_per_draw_paired_deltas_pp=[float(values.min()), float(values.max())] if finite else None,
                draw_variation_std_ddof=0,
                note="for varying valid denominators, mean of per-draw ratios can differ from mean per-query delta")


def analyze(sim, ann, s11, seed=2026, bootstrap=10000):
    start = time.perf_counter()
    checks = []
    verified, galleries, seeds = restore_s11(sim, ann, s11, checks)
    result = dict(task="R1_table5_clusters", annotation=ann.summary, checks=checks,
                  definition="paired cross-view R1 minus same-view R1, percentage points",
                  zero_positive_original_policy=dict(
                      evaluator="_offline_common.evaluate calls eligible_queries and skips no-positive queries",
                      source_exclusion="remove only the query source image before sorting",
                      common_queries="S11.common_query_sets intersects legal same-view and cross-view queries",
                      historical_sampling=s11["A"].get("sampling"),
                      historical_sampler="S11.sample_gallery retains all same-view sources, >=2 same-view and >=1 cross-view positives per identity",
                      conclusion="saved S11 draws, not newly invented uniform draws, define the Table-5 reproduction"),
                  policies=dict(a_fixed_misses="fixed common query set; no legal positive means a miss",
                                b_common_valid_queries="within each draw both directions use the same jointly-valid query set",
                                c_at_least_two_positives="retain >=2 original target-view images/identity in both directions; source-preserving same-view sampling"),
                  positive_retention_warning="c is label-stratified, raises positive retention, and is not uniform subsampling; impossible >=2 retention is reported, never relaxed",
                  groups={})
    query_id_sets = []
    for name, values in verified.items():
        query, source = values["query"], values["source"]
        query_id_sets.append(np.unique(ann.query_ids[query]))
        original_ci = identity_interval(values["original_difference"], np.ones(len(query), bool),
                                        ann.query_ids[query], bootstrap, seed)
        rows = dict(a_fixed_misses=policy_summary(values["draws"], ann.query_ids[query], "a_fixed_misses", bootstrap, seed),
                    b_common_valid_queries=policy_summary(values["draws"], ann.query_ids[query], "b_common_valid_queries", bootstrap, seed))
        preserving = []
        try:
            for draw_seed in seeds:
                rng = np.random.default_rng(draw_seed)
                same = at_least_two_gallery(ann, query, galleries[source], values["target"]+1, rng, True)
                cross = at_least_two_gallery(ann, query, galleries[1-source], values["target"], rng, False)
                detail, delta, valid = pair_outcomes(sim, ann, query, same, cross)
                for side in ("same", "cross"):
                    actual = detail[side]["metrics"]
                    if actual["gallery_size_min"] != values["target"] or actual["gallery_size_max"] != values["target"]:
                        raise AssertionError("positive-retention policy violated effective-gallery-size matching")
                preserving.append(dict(seed=draw_seed, detail=detail, difference=delta, valid=valid))
            rows["c_at_least_two_positives"] = policy_summary(preserving, ann.query_ids[query], "c_at_least_two_positives", bootstrap, seed)
            rows["c_at_least_two_positives"]["trials"] = [dict(seed=d["seed"], **d["detail"]) for d in preserving]
        except ValueError as exc:
            rows["c_at_least_two_positives"] = dict(status="infeasible", reason=str(exc),
                                                  completed_draws=len(preserving), rule_not_relaxed=True)
        same_key, cross_key = name+"->"+name, name+"->"+("G" if name == "A" else "A")
        reference_original_delta = s11["A"]["original"][cross_key]["R1"] - s11["A"]["original"][same_key]["R1"]
        reference_control_delta = np.mean([trial["cross"]["R1"] - trial["same"]["R1"]
                                           for trial in s11["A"]["gallery_control"][name]["trials"]])
        exact_original = values["original"]["cross"]["metrics"]["R1"] - values["original"]["same"]["metrics"]["R1"]
        exact_control = np.mean([d["detail"]["cross"]["metrics"]["R1"] - d["detail"]["same"]["metrics"]["R1"] for d in values["draws"]])
        exact_gate(checks, f"Table5.{name}.original_metric_delta", float(exact_original), float(reference_original_delta))
        exact_gate(checks, f"Table5.{name}.controlled_metric_delta", float(exact_control), float(reference_control_delta))
        result["groups"][name] = dict(**query_distribution(ann, query), query_indices=query.tolist(),
                                     effective_controlled_gallery_size=values["target"],
                                     original=dict(**values["original"], identity_bootstrap=original_ci),
                                     authoritative_trials=[dict(seed=d["seed"], index_source=d["index_source"], **d["detail"]) for d in values["draws"]],
                                     zero_positive_policies=rows,
                                     table5_reference=dict(source="S11.A original and gallery_control trials, not hardcoded paper numbers",
                                         original_cross_minus_same_pp=float(reference_original_delta),
                                         controlled_cross_minus_same_pp=float(reference_control_delta),
                                         exact_metric_reproduction_passed=True,
                                         fixed_miss_delta_minus_S11_original_pp=original_ci["observed_delta_pp"]-float(reference_original_delta),
                                         fixed_miss_delta_minus_S11_controlled_pp=rows["a_fixed_misses"]["mean_of_per_draw_paired_deltas_pp"]-float(reference_control_delta),
                                         difference_explanation="per-query/identity inference uses float64 ratios; S11 displayed R1 uses float32 percentages; genuine zero-positive denominator differences are separately enumerated"))
        print_table(["Side", "Policy", "Delta(pp)", "Identity 95%", "Two-stage 95%", "Draw std(pp)"],
                    [[name, policy, row["identity_bootstrap_conditional_on_fixed_draws"]["observed_delta_pp"],
                      row["identity_bootstrap_conditional_on_fixed_draws"]["ci95_percentile_pp"],
                      row["two_stage_identity_and_gallery_bootstrap"]["ci95_percentile_pp"],
                      row["std_of_per_draw_paired_deltas_pp"]]
                     if row["status"] == "complete" else [name, policy, "infeasible", row["reason"], "-", "-"]
                     for policy, row in rows.items()])
    result["unique_identity_union"] = np.union1d(*query_id_sets).tolist()
    result["n_unique_identity_union"] = len(result["unique_identity_union"])
    result["unique_identity_counts"] = dict(n_A=result["groups"]["A"]["n_identities"],
                                             n_G=result["groups"]["G"]["n_identities"],
                                             union=result["n_unique_identity_union"])
    result["gallery_seeds"] = seeds
    result["bootstrap_seed"] = seed
    result["bootstrap_repeats"] = bootstrap
    result["elapsed_sec"] = time.perf_counter()-start
    result["status"] = "complete"
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="read-only raw archived sim_qg.npy")
    parser.add_argument("--anno", required=True, help="test_reid.json in original image/caption order")
    parser.add_argument("--s11-json", required=True, help="matching authoritative S11 JSON")
    parser.add_argument("--out", required=True, help="new output JSON or directory; existing files never overwritten")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from .R1_joint_enrichment import write_audited_report
    except ImportError:
        from R1_joint_enrichment import write_audited_report
    inputs = dict(similarity=args.sim, annotation=args.anno, authoritative_S11=args.s11_json)
    seeds = dict(identity_bootstrap=args.seed, two_stage_identity_and_gallery=args.seed)
    log = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(log):
            if args.seed < 0 or args.bootstrap < 1:
                raise ValueError("--seed must be nonnegative and --bootstrap positive")
            with Path(args.s11_json).open(encoding="utf-8-sig") as handle:
                s11 = json.load(handle)
            seeds["gallery"] = s11["seeds"]["gallery"]
            ann = load_annotations(args.anno)
            sim = load_similarity(args.sim, ann)
            report = analyze(sim, ann, s11, args.seed, args.bootstrap)
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError, TypeError) as exc:
        code = 2
        report = dict(status="stopped", error=str(exc))
        print(f"R1_table5_clusters STOP: {exc}", file=log)
    report["readable_log"] = log.getvalue()
    print(report["readable_log"], end="")
    try:
        json_path, log_path = write_audited_report(args, "R1_table5_clusters", report, inputs, seeds, argv)
        print(f"JSON: {json_path}\nLOG: {log_path}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"R1_table5_clusters output error: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())

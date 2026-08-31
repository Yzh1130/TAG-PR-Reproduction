"""S14: audit archived protocols, then add identity-clustered inference.

Only NumPy/stdlib and read-only offline helpers are imported. No model,
training or online evaluation path is changed. Existing S11 folds and gallery
draws are authoritative; replay is gated by exact archived metrics, never by
an expected accuracy constant or a floating-point tolerance.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

try:
    from ._analysis_audit import Audit
    from ._offline_common import (compact_metrics, evaluate, indices,
                                  load_annotations, load_similarity, print_table,
                                  topk_mean)
    from ._s11_groups import balanced_group_halves, _counts
    from ._s11_descriptive import error_decomposition, paired_test
    from .S11_offline_stats import common_query_sets, sample_gallery
except ImportError:
    from _analysis_audit import Audit
    from _offline_common import (compact_metrics, evaluate, indices,
                                 load_annotations, load_similarity, print_table,
                                 topk_mean)
    from _s11_groups import balanced_group_halves, _counts
    from _s11_descriptive import error_decomposition, paired_test
    from S11_offline_stats import common_query_sets, sample_gallery


METRICS = ("R1", "R5", "R10", "mAP")
ENRICHMENT_DEFINITION = (
    "P(predicted source equals true source | top-1 error, query view) / "
    "mean(same-source gallery fraction after removing ALL true-identity images "
    "| the same top-1 error queries and query view)"
)
CELLS = ("same_source_same_view", "same_source_cross_view",
         "cross_source_same_view", "cross_source_cross_view")


def metric_gate(audit, name, actual, expected, keys=METRICS):
    for key in keys:
        audit.check(f"{name}.{key}", actual[key], expected[key])


def restore_identity_folds(sim, ann, s11, audit):
    """Replay exactly S11's split code, verify the OLD full-gallery protocol.

    The narrower evaluation-ID gallery is introduced only after every gate
    here passes. Thus a different protocol cannot accidentally validate a
    guessed/recreated split. Both folds must pass before writing their IDs.
    """
    source = s11["F"]
    seed = source["split_seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("S11 F.split_seed must be an integer")
    expected_folds = source["schemes"]["identity_group"]["folds"]
    if len(expected_folds) != 2 or [x["fold"] for x in expected_folds] != [0, 1]:
        raise ValueError("S11 identity_group must contain ordered folds 0 and 1")
    print("AUTHORITATIVE S11 identity_group " + json.dumps(expected_folds, ensure_ascii=False))
    halves = balanced_group_halves(ann.query_ids, seed=seed)
    k = source["k"]
    records, penalties = [], []
    for fold in (0, 1):
        calibration, evaluation = halves[fold], halves[1 - fold]
        expected = expected_folds[fold]
        counts = {name: _counts(ann, values) for name, values in
                  (("calibration", calibration), ("evaluation", evaluation))}
        for name in counts:
            for key in ("identities", "images", "texts", "queries"):
                audit.check(f"F.fold{fold}.{name}.{key}", counts[name][key], expected[name][key])
        overlap = {
            "queries": int(np.intersect1d(calibration, evaluation).size),
            "images": int(np.intersect1d(ann.query_source_indices[calibration],
                                        ann.query_source_indices[evaluation]).size),
            "identities": int(np.intersect1d(ann.query_ids[calibration], ann.query_ids[evaluation]).size),
        }
        audit.check(f"F.fold{fold}.overlap", overlap, dict(queries=0, images=0, identities=0))
        penalty = topk_mean(sim, k, rows=calibration)
        raw = evaluate(sim, ann, queries=evaluation)
        corrected = evaluate(sim, ann, queries=evaluation, penalty=penalty, alpha=0.5)
        metric_gate(audit, f"F.fold{fold}.uncorrected", raw, expected["metrics"]["uncorrected"])
        metric_gate(audit, f"F.fold{fold}.crossfit_csls", corrected,
                    expected["metrics"]["crossfit_csls"], ("R1",))
        record = dict(fold=fold, **counts, overlap=overlap,
                      calibration_identity_ids=np.unique(ann.query_ids[calibration]).tolist(),
                      evaluation_identity_ids=np.unique(ann.query_ids[evaluation]).tolist(),
                      calibration_query_indices=calibration.tolist(),
                      evaluation_query_indices=evaluation.tolist(),
                      calibration_image_indices=np.unique(ann.query_source_indices[calibration]).tolist(),
                      evaluation_image_indices=np.unique(ann.query_source_indices[evaluation]).tolist(),
                      original_protocol_metrics=dict(uncorrected=compact_metrics(raw),
                                                     crossfit_csls=compact_metrics(corrected)))
        records.append(record)
        penalties.append(penalty)
        print(f"RESTORED fold={fold} counts={counts}; all full-gallery metric gates PASS")
    trace = dict(split_seed=seed, source="S11 F.schemes.identity_group",
                 replay="_s11_groups.balanced_group_halves(ann.query_ids, seed=F.split_seed)",
                 orientation="fold i: calibration=halves[i], evaluation=halves[1-i]",
                 folds=records, checks=list(audit.report.get("checks", [])))
    path = audit.write_json(f"identity_folds_seed{seed}.json", trace)
    return halves, penalties, dict(**trace, trace_file=str(path))


def identity_gallery_crossfit(sim, ann, s11, halves, penalties, audit,
                              scan_sizes=(100, 500, 1000, 5000)):
    """A: all three settings use only evaluation identities in the gallery."""
    seed_list = s11["seeds"]["scans"]
    if not seed_list or not all(isinstance(seed, int) for seed in seed_list):
        raise ValueError("S11 seeds.scans must be nonempty integer seeds")
    k = 10
    folds = []
    for fold in (0, 1):
        cal, query = halves[fold], halves[1 - fold]
        eval_ids = np.unique(ann.query_ids[query])
        gallery = np.flatnonzero(np.isin(ann.gallery_ids, eval_ids))
        audit.check(f"A.fold{fold}.calibration_source_overlap_evaluation_gallery",
                    int(np.intersect1d(ann.query_source_indices[cal], gallery).size), 0)
        audit.check(f"A.fold{fold}.identities_disjoint",
                    int(np.intersect1d(ann.query_ids[cal], ann.gallery_ids[gallery]).size), 0)
        audit.check(f"A.fold{fold}.evaluation_gallery_all_id_images",
                    np.unique(ann.query_source_indices[query]).tolist(), gallery.tolist())
        # Fixed k=10 for the NEW protocol, even if an archived F used another k.
        cross_penalty = penalties[fold] if s11["F"]["k"] == k else topk_mean(sim, k, rows=cal)
        trans_penalty = topk_mean(sim, k, rows=query)
        values = {
            "uncorrected": evaluate(sim, ann, queries=query, gallery=gallery),
            "crossfit_csls": evaluate(sim, ann, queries=query, gallery=gallery,
                                      penalty=cross_penalty, alpha=0.5),
            "evaluation_query_transductive_csls": evaluate(sim, ann, queries=query, gallery=gallery,
                                                           penalty=trans_penalty, alpha=0.5),
        }
        for name, metrics in values.items():
            audit.check(f"A.fold{fold}.{name}.same_queries",
                        metrics["details"]["query_indices"].tolist(), query.tolist())
            audit.check(f"A.fold{fold}.{name}.no_skipped", metrics["skipped_no_positive"], 0)
        scan = []
        for requested in [*scan_sizes, "full"]:
            budget = len(cal) if requested == "full" else requested
            if not k <= budget <= len(cal):
                row = dict(calibration_texts=requested, status="unavailable",
                           reason=f"need {k} <= budget <= {len(cal)} calibration descriptions")
                scan.append(row)
                print(f"A fold={fold} scan={requested}: {row['reason']}")
                continue
            repeats = []
            for seed in seed_list:
                selected = cal if budget == len(cal) else np.random.default_rng(seed).choice(cal, budget, replace=False)
                if np.intersect1d(ann.query_source_indices[selected], gallery).size:
                    raise AssertionError("calibration scan contains an evaluation-gallery image description")
                penalty = cross_penalty if budget == len(cal) else topk_mean(sim, k, rows=selected)
                result = evaluate(sim, ann, queries=query, gallery=gallery, penalty=penalty, alpha=0.5)
                if budget == len(cal):
                    metric_gate(audit, f"A.fold{fold}.full_scan.seed{seed}", result, values["crossfit_csls"])
                repeats.append(dict(seed=seed, calibration_texts=len(selected),
                                    query_indices=selected.tolist(), metrics=compact_metrics(result)))
            summary = {m: dict(mean=float(np.mean([r["metrics"][m] for r in repeats])),
                               std=float(np.std([r["metrics"][m] for r in repeats], ddof=0))) for m in METRICS}
            scan.append(dict(calibration_texts=requested, status="complete", repetitions=repeats,
                             metrics_mean_std=summary))
            print(f"A fold={fold} scan={requested}: {json.dumps(summary)}")
        folds.append(dict(fold=fold, evaluation_identities=len(eval_ids), evaluation_images=len(gallery),
                          evaluation_queries=len(query), calibration_descriptions=len(cal),
                          gallery_indices=gallery.tolist(), query_indices=query.tolist(),
                          metrics={n: compact_metrics(v) for n, v in values.items()},
                          delta_vs_uncorrected={n: {m: values[n][m] - values["uncorrected"][m] for m in METRICS}
                                                for n in values if n != "uncorrected"},
                          calibration_scan=scan))
        print_table(["A fold", "setting", *METRICS],
                    [[fold, name, *[value[m] for m in METRICS]] for name, value in values.items()])
    return dict(alpha=0.5, k=k, folds=folds, scan_seeds=seed_list,
                scan_sampling="individual calibration descriptions, without replacement; no evaluation IDs",
                protocol="evaluation-ID-only gallery, same ordered queries/gallery for all three settings; source image retained",
                penalty_note="column top-k is independent of other columns; only evaluation-gallery columns are used",
                std_ddof=0)


def ratio_bootstrap(numerators, denominators, groups=None, universe=None, repeats=10000, seed=2026):
    """Cluster or query bootstrap of ratios of sums; no empty-draw retries.

    A supplied universe retains identities with zero selected errors. Each
    replicate draws ALL identity clusters, then computes the chosen view/error
    ratios. Undefined draws are counted and that interval is null, not silently
    conditioned on nonempty errors.
    """
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    if num.ndim != 2 or num.shape != den.shape or not len(num) or num.shape[1] == 0:
        raise ValueError("bootstrap requires matching nonempty [rows, estimands] arrays")
    if not np.isfinite(num).all() or not np.isfinite(den).all() or np.any(den < 0):
        raise ValueError("bootstrap counts/weights must be finite; denominators nonnegative")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("bootstrap repeats must be a positive integer")
    if groups is None:
        un, ud = num, den
        unit = "query"
    else:
        groups = np.asarray(groups)
        if groups.shape != (len(num),):
            raise ValueError("one identity is required per bootstrap row")
        labels = np.unique(groups) if universe is None else np.asarray(universe)
        if labels.ndim != 1 or not len(labels) or len(np.unique(labels)) != len(labels):
            raise ValueError("cluster universe must have unique identities")
        labels = np.sort(labels)
        inverse = np.searchsorted(labels, groups)
        if np.any(inverse >= len(labels)) or not np.array_equal(labels[inverse], groups):
            raise ValueError("bootstrap row identity absent from cluster universe")
        un = np.zeros((len(labels), num.shape[1]), np.float64)
        ud = np.zeros_like(un)
        for j in range(num.shape[1]):
            un[:, j] = np.bincount(inverse, weights=num[:, j], minlength=len(labels))
            ud[:, j] = np.bincount(inverse, weights=den[:, j], minlength=len(labels))
        unit = "identity"
    rng = np.random.default_rng(seed)
    values = np.empty((repeats, num.shape[1]), np.float64)
    for start in range(0, repeats, 32):
        end = min(start + 32, repeats)
        selected = rng.integers(0, len(un), size=(end - start, len(un)), dtype=np.int64)
        # Reduce contiguous single-estimand rows, matching S9's query-level
        # summation order; don't materialize [draws, units, estimands].
        for j in range(num.shape[1]):
            nsum, dsum = un[:, j][selected].sum(1), ud[:, j][selected].sum(1)
            values[start:end, j] = np.divide(nsum, dsum, out=np.full_like(nsum, np.nan), where=dsum > 0)
    results = []
    for j in range(num.shape[1]):
        invalid = int(np.count_nonzero(~np.isfinite(values[:, j])))
        total = float(den[:, j].sum())
        results.append(dict(observed=float(num[:, j].sum() / total) if total else None,
                            bootstrap_mean=float(values[:, j].mean()) if not invalid else None,
                            ci95_percentile=np.percentile(values[:, j], [2.5, 97.5]).tolist() if not invalid else None,
                            undefined_replicates=invalid))
    return dict(unit=unit, n_units=len(un), repeats=repeats, seed=seed,
                estimator="ratio of query-weighted sums; whole sampled identities retain all member queries",
                empty_policy="no redraw/discard; any undefined replicate gives a null interval for that estimand",
                results=results)


def _error_arrays(ann, raw):
    details = raw["details"]
    queries = np.asarray(details["query_indices"], dtype=np.int64)
    if not np.array_equal(queries, np.arange(len(ann.query_ids))):
        raise ValueError("error statistics require the full ordered query set")
    predicted = np.asarray(details["top1"], dtype=np.int64)
    errors = ~np.asarray(details["correct"], dtype=bool)
    return errors, ann.query_domains == ann.gallery_domains[predicted], ann.query_views == ann.gallery_views[predicted]


def enrichment_statistics(ann, raw, s9, audit, repeats=10000, seed=2026):
    definition = s9["enrichment_definition"]
    print("S9 enrichment_definition: " + str(definition))
    # Fail closed for a different definition rather than silently interpreting
    # free text. This is the exact definition in the supplied S9_fk.json.
    audit.check("B.S9.enrichment_definition", definition, ENRICHMENT_DEFINITION)
    errors, same_source, _ = _error_arrays(ann, raw)
    error_q = np.flatnonzero(errors)
    id_counts = Counter(ann.gallery_ids.tolist())
    domain_counts = Counter(ann.gallery_domains.tolist())
    id_domain_counts = Counter(zip(ann.gallery_ids.tolist(), ann.gallery_domains.tolist()))
    priors = np.empty(len(error_q), np.float64)
    for offset, q in enumerate(error_q):
        pid, domain = int(ann.query_ids[q]), str(ann.query_domains[q])
        remaining = len(ann.gallery_ids) - id_counts[pid]
        if remaining <= 0:
            raise ValueError("no gallery remains after removing all true-identity images")
        priors[offset] = (domain_counts[domain] - id_domain_counts[(pid, domain)]) / remaining
    keys = ("same_source_error_probability", "mean_conditional_prior", "enrichment_ratio")
    points, arrays = {}, {}
    for view, name in ((0, "aerial"), (1, "ground")):
        selected = ann.query_views[error_q] == view
        q = error_q[selected]
        indicators, prior = same_source[q], priors[selected]
        probability = float(indicators.mean()) if len(q) else None
        mean_prior = float(prior.mean()) if len(q) else None
        ratio = probability / mean_prior if mean_prior else None
        point = dict(zip(keys, (probability, mean_prior, ratio)))
        for key, value in point.items():
            audit.check(f"B.{name}.{key}", value, s9["groups"][name][key])
        audit.check(f"B.{name}.n_errors", len(q), s9["groups"][name]["n_errors"])
        points[name] = point
        arrays[name] = (q, indicators, prior)
    # All two-view point gates pass BEFORE generating new CIs.
    result = {}
    all_ids = np.unique(ann.query_ids)
    for name, (q, indicator, prior) in arrays.items():
        if not len(q):
            result[name] = dict(point=points[name], n_errors=0, reason="no error queries", identity=None, query=None)
            continue
        numerators = np.column_stack((indicator, prior, indicator))
        denominators = np.column_stack((np.ones(len(q)), np.ones(len(q)), prior))
        clustered = ratio_bootstrap(numerators, denominators, ann.query_ids[q], all_ids, repeats, seed)
        query = ratio_bootstrap(numerators, denominators, repeats=repeats, seed=seed)
        for row in (clustered, query):
            row["results"] = dict(zip(keys, row["results"]))
        result[name] = dict(point=points[name], n_errors=len(q), identity=clustered, query=query)
        print_table(["B view", "quantity", "point", "identity 95%", "query 95%"],
                    [[name, key, points[name][key], clustered["results"][key]["ci95_percentile"],
                      query["results"][key]["ci95_percentile"]] for key in keys])
    return dict(definition=definition, groups=result, identity_universe=all_ids.tolist(),
                scope="conditional on top-1 errors within each query view; prior averages ONLY those same errors")


def category_statistics(ann, raw, s11, audit, repeats=10000, seed=2026):
    actual = error_decomposition(ann, raw)["strata"]["all"]
    expected = s11["B_error_decomposition"]["raw"]["strata"]["all"]
    audit.check("C.error_count", actual["n_errors"], expected["n_errors"])
    for cell in CELLS:
        audit.check(f"C.{cell}.count", actual["cells"][cell]["count"], expected["cells"][cell]["count"])
    errors, source, view = _error_arrays(ann, raw)
    cell_rows = np.column_stack((errors & source & view, errors & source & ~view,
                                errors & ~source & view, errors & ~source & ~view)).astype(np.float64)
    if not np.array_equal(cell_rows.sum(1), errors.astype(np.float64)):
        raise AssertionError("four error cells do not partition each query's error event")
    numerators = np.column_stack((cell_rows, cell_rows[:, 1], cell_rows[:, 3]))
    denominators = np.column_stack((np.tile(errors[:, None], (1, 4)), errors & source, errors & ~source))
    bootstrap = ratio_bootstrap(numerators, denominators, ann.query_ids, np.unique(ann.query_ids), repeats, seed)
    keys = [*CELLS, "P(cross_view|same_source,error)", "P(cross_view|cross_source,error)"]
    bootstrap["results"] = dict(zip(keys, bootstrap["results"]))
    print_table(["C category", "proportion", "identity 95%"],
                [[key, row["observed"], row["ci95_percentile"]] for key, row in bootstrap["results"].items()])
    return dict(counts={cell: actual["cells"][cell]["count"] for cell in CELLS},
                n_errors=actual["n_errors"], identity=bootstrap,
                denominator="top-1 errors for four cells; same/cross-source errors for the two conditionals")


def controlled_identity_statistics(sim, ann, s11, audit, repeats=10000, seed=2026):
    """D: use saved draws if present, verify replay and R1, never McNemar."""
    saved = s11["A"]
    if saved["mode"] != "common-min":
        raise ValueError("S14-D requires the authoritative S11 common-min gallery control")
    sets, galleries, counts = common_query_sets(ann)
    gallery_seeds = s11["seeds"]["gallery"]
    for name in ("A", "G"):
        audit.check(f"D.{name}.common_queries", sets[name].tolist(),
                    saved["common_query_sets"][name]["query_indices"])
        audit.check(f"D.{name}.common_count", len(sets[name]),
                    saved["common_query_sets"][name]["common_queries"])
    # Reproduce every original/drawn uncorrected score before ANY paired CI.
    verified = {}
    for name, source in (("A", 0), ("G", 1)):
        query = sets[name]
        if not len(query):
            raise ValueError(f"no common legal queries for {name}")
        same_key, cross_key = name + "->" + name, name + "->" + ("G" if source == 0 else "A")
        same = evaluate(sim, ann, queries=query, gallery=galleries[source], exclude_source=True)
        cross = evaluate(sim, ann, queries=query, gallery=galleries[1 - source], exclude_source=True)
        metric_gate(audit, f"D.original.{same_key}", same, saved["original"][same_key], ("R1",))
        metric_gate(audit, f"D.original.{cross_key}", cross, saved["original"][cross_key], ("R1",))
        control = saved["gallery_control"][name]
        if control["status"] != "complete":
            raise ValueError(f"authoritative S11 gallery control {name} is incomplete")
        trials = control["trials"]
        audit.check(f"D.{name}.gallery_seeds", [t["seed"] for t in trials], gallery_seeds)
        target = min(len(galleries[source]) - 1, len(galleries[1 - source]))
        audit.check(f"D.{name}.effective_gallery_size", target, control["effective_gallery_size"])
        draws = []
        for trial in trials:
            draw_seed = trial["seed"]
            rng = np.random.default_rng(draw_seed)
            replay_same = (galleries[source] if target == len(galleries[source]) - 1 else
                           sample_gallery(ann, query, galleries[source], target + 1, rng, True))
            replay_cross = sample_gallery(ann, query, galleries[1 - source], target, rng, False)
            chosen = []
            for side, replay in (("same", replay_same), ("cross", replay_cross)):
                key = side + "_gallery_indices"
                if key in trial:
                    values = indices(trial[key], len(ann.gallery_ids))
                    audit.check(f"D.{name}.seed{draw_seed}.{key}.replay", replay.tolist(), values.tolist())
                else:
                    values = replay
                evaluated = evaluate(sim, ann, queries=query, gallery=values, exclude_source=True)
                metric_gate(audit, f"D.{name}.seed{draw_seed}.{side}", evaluated, trial[side], ("R1",))
                audit.check(f"D.{name}.seed{draw_seed}.{side}.effective_min", evaluated["gallery_size_min"], target)
                audit.check(f"D.{name}.seed{draw_seed}.{side}.effective_max", evaluated["gallery_size_max"], target)
                audit.check(f"D.{name}.seed{draw_seed}.{side}.query_set",
                            evaluated["details"]["query_indices"].tolist(), query.tolist())
                chosen.append((values, evaluated, "saved_and_replay_verified" if key in trial else "seed_replay"))
            draws.append((draw_seed, chosen))
        verified[name] = (query, same, cross, draws, target)
    result = {}
    for name, (query, same, cross, draws, target) in verified.items():
        groups = ann.query_ids[query]
        original_ci = paired_test(same["details"]["correct"], cross["details"]["correct"],
                                  groups=groups, repeats=repeats, seed=seed)
        draw_results = []
        for draw_seed, chosen in draws:
            sg, se, ss = chosen[0]
            cg, ce, cs = chosen[1]
            ci = paired_test(se["details"]["correct"], ce["details"]["correct"],
                             groups=groups, repeats=repeats, seed=seed)
            draw_results.append(dict(gallery_seed=draw_seed, bootstrap_seed=seed,
                                     same=compact_metrics(se), cross=compact_metrics(ce),
                                     same_gallery_indices=sg.tolist(), cross_gallery_indices=cg.tolist(),
                                     gallery_index_source=dict(same=ss, cross=cs), identity_ci=ci))
        intervals = np.asarray([row["identity_ci"]["ci95_percentile_pp"] for row in draw_results])
        observed = np.asarray([row["identity_ci"]["observed_delta_pp"] for row in draw_results])
        result[name] = dict(n_queries=len(query), n_identities=len(np.unique(groups)),
                            original=dict(same=compact_metrics(same), cross=compact_metrics(cross), identity_ci=original_ci),
                            controlled=dict(effective_gallery_size=target, trials=draw_results,
                                            observed_delta_mean_pp=float(observed.mean()),
                                            ci95_endpoint_mean_pp=intervals.mean(0).tolist(),
                                            ci95_lower_range_pp=[float(intervals[:, 0].min()), float(intervals[:, 0].max())],
                                            ci95_upper_range_pp=[float(intervals[:, 1].min()), float(intervals[:, 1].max())]))
        print(f"D {name}: original identity CI={original_ci['ci95_percentile_pp']}; "
              f"controlled mean endpoints={intervals.mean(0).tolist()}, "
              f"range lower={result[name]['controlled']['ci95_lower_range_pp']}, "
              f"upper={result[name]['controlled']['ci95_upper_range_pp']}")
    return dict(groups=result, bootstrap_seed=seed, gallery_seeds=gallery_seeds,
                paired_difference="cross-view R1 minus same-view R1, percentage points",
                sampling="whole identities among each side's common queries; preserve query weights",
                ci_summary_note="mean/range of separate per-gallery-draw confidence intervals; not a pooled confidence interval",
                mcnemar_computed=False)


def run_analysis(args, audit):
    start = time.perf_counter()
    s11 = audit.read_json(args.s11_json, "authoritative_S11")
    s9 = audit.read_json(args.s9_json, "authoritative_S9")
    print("S9 enrichment_definition: " + str(s9["enrichment_definition"]))
    audit.report["seeds"].update(split=s11["F"]["split_seed"], gallery=s11["seeds"]["gallery"], scans=s11["seeds"]["scans"])
    audit.record_input(args.anno, "test_annotation")
    ann = load_annotations(args.anno)
    audit.record_input(args.sim, "archived_similarity")
    sim = load_similarity(args.sim, ann)
    audit.report["annotation"] = ann.summary
    raw = evaluate(sim, ann)
    metric_gate(audit, "full_gallery.uncorrected", raw, s11["raw"])
    audit.report["raw"] = compact_metrics(raw)
    halves, penalties, trace = restore_identity_folds(sim, ann, s11, audit)
    audit.report["identity_fold_recovery"] = trace
    audit.report["A_identity_gallery_crossfit"] = identity_gallery_crossfit(sim, ann, s11, halves, penalties, audit)
    audit.report["B_source_enrichment"] = enrichment_statistics(ann, raw, s9, audit, args.bootstrap, args.seed)
    audit.report["C_error_categories"] = category_statistics(ann, raw, s11, audit, args.bootstrap, args.seed)
    audit.report["D_controlled_paired"] = controlled_identity_statistics(sim, ann, s11, audit, args.bootstrap, args.seed)
    audit.report["elapsed_sec"] = time.perf_counter() - start
    return audit.report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="read-only archived sim_qg.npy")
    parser.add_argument("--anno", required=True, help="test_reid.json in original image/caption order")
    parser.add_argument("--s11-json", required=True, help="authoritative S11 fk32 report with A/B/F")
    parser.add_argument("--s9-json", required=True, help="authoritative S9_fk.json")
    parser.add_argument("--out", required=True, help="new _e1_s14 output directory; never overwrites")
    parser.add_argument("--seed", type=int, default=2026, help="identity/query bootstrap seed; S11 replay seeds come from JSON")
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    audit = None
    try:
        audit = Audit("S14", args, argv=argv, seeds=dict(bootstrap=args.seed))
        with audit.capture():
            try:
                if args.bootstrap < 1 or args.seed < 0:
                    raise ValueError("--bootstrap must be positive and --seed nonnegative")
                run_analysis(args, audit)
            except Exception as exc:
                print(f"S14 STOP: {exc}; no seed changes or tolerance fallback permitted")
                audit.finish(status="stopped", error=str(exc))
                return 2
            audit.finish(status="complete")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"S14 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

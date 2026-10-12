"""Joint source/view enrichment, conditional on the SAME top-1 error queries.

Read-only offline analysis. Gallery labels define metrics/priors only, never a
post-processing penalty. Whole identities (including zero-error identities)
are resampled, retaining every query with its multiplicity.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

try:
    from ._analysis_audit import _exact, git_head, read_json, sha256_16, unique_path
    from ._offline_common import (compact_metrics, evaluate, json_ready,
                                  load_annotations, load_similarity, print_table)
    from ._s11_descriptive import error_decomposition
except ImportError:
    from _analysis_audit import _exact, git_head, read_json, sha256_16, unique_path
    from _offline_common import (compact_metrics, evaluate, json_ready,
                                 load_annotations, load_similarity, print_table)
    from _s11_descriptive import error_decomposition


CELLS = ("same_source_same_view", "same_source_cross_view",
         "cross_source_same_view", "cross_source_cross_view")
PURE = ("Market", "PRAI-1581", "uavhuman")
MIXED = ("AG-ReID.v2", "G2APS")


def write_audited_report(args, tool, report, inputs, seeds, argv=None):
    """Shared Round B I/O: exclusive JSON/log and exact input hashes/byte sizes.

    --out can name a directory or a JSON file. Collisions add timestamps;
    existing inputs and archive evidence are never overwritten.
    """
    out = Path(args.out).expanduser().resolve()
    directory = out.parent if out.suffix.lower() == ".json" else out
    if any(p.lower() in ("_archive_evidence", "_dump") for p in directory.parts):
        raise ValueError("choose a new output directory outside immutable archives")
    for source in inputs.values():
        parent = Path(source).expanduser().resolve().parent
        if parent == directory or parent in directory.parents:
            raise ValueError(f"output must not be within input directory: {parent}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for role, source in inputs.items():
        source = Path(source).expanduser().resolve()
        manifest[role] = dict(path=str(source), available=source.is_file())
        if source.is_file():
            manifest[role].update(bytes=source.stat().st_size, sha256_16=sha256_16(source))
    report = dict(report)
    entrypoint = "S13_inductive_bootstrap" if tool == "S13_summary_table" else tool
    report["provenance"] = dict(
        argv=list(sys.argv if argv is None else [str(Path(__file__).with_name(entrypoint + ".py")), *argv]),
        arguments=vars(args), seeds=seeds, inputs=manifest, git_HEAD=git_head(),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    content = json.dumps(json_ready(report), indent=2, ensure_ascii=False, allow_nan=False)
    name = out.name if out.suffix.lower() == ".json" else tool + ".json"
    while True:
        path = unique_path(directory, name)
        log_path = path.with_suffix(".log")
        if log_path.exists():
            name = f"{Path(name).stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
            continue
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content + "\n")
            with log_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(str(report.get("readable_log", content)) + "\n")
            break
        except FileExistsError:
            continue
    print(f"JSON: {path}\nLOG: {log_path}")
    return path, log_path


def joint_error_inputs(ann, raw):
    """One-hot observed cells and per-error conditional negative-cell priors."""
    details = raw["details"]
    query = np.asarray(details["query_indices"])
    if not np.array_equal(query, np.arange(len(ann.query_ids))):
        raise ValueError("joint enrichment requires all ordered queries, without exclusions")
    pred = np.asarray(details["top1"])
    errors = ~np.asarray(details["correct"])
    same_source = ann.query_domains == ann.gallery_domains[pred]
    same_view = ann.query_views == ann.gallery_views[pred]
    observed = np.column_stack((same_source & same_view, same_source & ~same_view,
                                ~same_source & same_view, ~same_source & ~same_view)).astype(np.float64)
    observed *= errors[:, None]
    domains = np.unique(ann.gallery_domains)
    counts = {(str(d), v): int(((ann.gallery_domains == d) & (ann.gallery_views == v)).sum())
              for d in domains for v in (0, 1)}
    identity_counts = {}
    for pid in np.unique(ann.gallery_ids):
        rows = np.flatnonzero(ann.gallery_ids == pid)
        identity_counts[int(pid)] = {v: int((ann.gallery_views[rows] == v).sum()) for v in (0, 1)}
    priors = np.zeros_like(observed)
    for q in np.flatnonzero(errors):
        domain, view, pid = str(ann.query_domains[q]), int(ann.query_views[q]), int(ann.query_ids[q])
        own = identity_counts[pid]
        ss = counts[(domain, view)] - own[view]
        sc = counts[(domain, 1 - view)] - own[1 - view]
        cs = sum(counts[(str(d), view)] for d in domains if str(d) != domain)
        cc = sum(counts[(str(d), 1 - view)] for d in domains if str(d) != domain)
        denominator = len(ann.gallery_ids) - own[0] - own[1]
        if denominator <= 0:
            raise ValueError(f"identity {pid}: no negative gallery images")
        if ss + sc + cs + cc != denominator:
            raise AssertionError("negative-cell counts must conserve the negative gallery")
        priors[q] = np.asarray((ss, sc, cs, cc), np.float64) / denominator
    if not np.allclose(priors[errors].sum(1), 1.0, atol=1e-15, rtol=0):
        raise AssertionError("negative priors must sum to one")
    if not np.array_equal(observed.sum(1), errors.astype(np.float64)):
        raise AssertionError("observed cells must partition exactly the top-1 errors")
    return errors, observed, priors


def _quantities(sums):
    """Ratios from resampled sums, not averages of per-identity ratios."""
    sums = np.asarray(sums, np.float64)
    denominator = sums[..., :1]
    p = np.divide(sums[..., 1:5], denominator, out=np.full_like(sums[..., 1:5], np.nan), where=denominator > 0)
    prior = np.divide(sums[..., 5:9], denominator, out=np.full_like(sums[..., 5:9], np.nan), where=denominator > 0)
    ratio = np.divide(p, prior, out=np.full_like(p, np.nan), where=prior > 0)
    obs_same = np.divide(p[..., 0], p[..., 0] + p[..., 1],
                         out=np.full(p.shape[:-1], np.nan), where=(p[..., 0] + p[..., 1]) > 0)
    prior_same = np.divide(prior[..., 0], prior[..., 0] + prior[..., 1],
                           out=np.full(p.shape[:-1], np.nan), where=(prior[..., 0] + prior[..., 1]) > 0)
    conditional_ratio = np.divide(obs_same, prior_same, out=np.full_like(obs_same, np.nan), where=prior_same > 0)
    cross_ratio = np.divide(1 - obs_same, 1 - prior_same, out=np.full_like(obs_same, np.nan), where=(1 - prior_same) > 0)
    observed_view = p[..., 0] + p[..., 2]
    expected_view = prior[..., 0] + prior[..., 2]
    view_ratio = np.divide(observed_view, expected_view, out=np.full_like(observed_view, np.nan), where=expected_view > 0)
    cross_view_ratio = np.divide(1 - observed_view, 1 - expected_view, out=np.full_like(observed_view, np.nan), where=(1 - expected_view) > 0)
    return np.concatenate((p, prior, p - prior, ratio,
                           np.stack((obs_same, prior_same, obs_same - prior_same, conditional_ratio,
                                     1 - obs_same, 1 - prior_same, prior_same - obs_same, cross_ratio,
                                     observed_view, expected_view, observed_view - expected_view, view_ratio,
                                     1 - observed_view, 1 - expected_view, expected_view - observed_view,
                                     cross_view_ratio), axis=-1)), axis=-1)


def _interval(values):
    finite = np.isfinite(values)
    # No redraw or silent dropping of undefined replicates.
    return dict(ci95_percentile=np.percentile(values, [2.5, 97.5]).tolist() if finite.all() else None,
                undefined_replicates=int((~finite).sum()),
                ci95_valid_only_descriptive=(np.percentile(values[finite], [2.5, 97.5]).tolist()
                                            if finite.any() and not finite.all() else None))


def enrichment_stratum(ann, selected, errors, observed, priors, repeats, seed, pure=False):
    selected = np.asarray(selected, bool)
    values = np.column_stack((errors.astype(float), observed, priors)) * selected[:, None]
    point = _quantities(values.sum(0))
    ids, inverse = np.unique(ann.query_ids, return_inverse=True)
    clusters = np.zeros((len(ids), 9), np.float64)
    np.add.at(clusters, inverse, values)
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, 32), np.float64)
    for start in range(0, repeats, 64):
        stop = min(start + 64, repeats)
        sample = rng.integers(0, len(ids), (stop - start, len(ids)))
        draws[start:stop] = _quantities(clusters[sample].sum(1))
    result = dict(n_queries=int(selected.sum()), n_errors=int(values[:, 0].sum()),
                  n_identity_clusters=len(ids), identity_universe="all query identities, including zero-error clusters",
                  bootstrap=repeats, seed=seed, cells={})
    for i, cell in enumerate(CELLS):
        if pure and cell == "same_source_cross_view":
            continue
        item = {}
        for key, position in (("observed_share", i), ("expected_share", 4 + i),
                              ("absolute_excess", 8 + i), ("enrichment_ratio", 12 + i)):
            value = float(point[position]) if np.isfinite(point[position]) else None
            item[key] = dict(estimate=value, **_interval(draws[:, position]))
        if point[4 + i] < 0.01 or not np.isfinite(point[4 + i]):
            item["enrichment_ratio"] = dict(estimate=None, ci95_percentile=None,
                                           reason="expected share below 1% or undefined; report excess only")
        result["cells"][cell] = item
    if pure:
        result["inapplicable_cells"] = {"same_source_cross_view": "pure-view source; no such same-source negatives"}
    result["same_view_given_same_source"] = {
        key: dict(estimate=float(point[position]) if np.isfinite(point[position]) else None,
                  **_interval(draws[:, position]))
        for key, position in (("observed", 16), ("expected", 17), ("absolute_excess", 18), ("enrichment_ratio", 19))
    }
    result["cross_view_given_same_source"] = {
        key: dict(estimate=float(point[position]) if np.isfinite(point[position]) else None,
                  **_interval(draws[:, position]))
        for key, position in (("observed", 20), ("expected", 21), ("absolute_excess", 22), ("enrichment_ratio", 23))
    }
    result["view_marginals_among_all_errors"] = {
        side: {key: dict(estimate=float(point[offset + i]) if np.isfinite(point[offset + i]) else None,
                         **_interval(draws[:, offset + i]))
               for i, key in enumerate(("observed", "expected", "absolute_excess", "enrichment_ratio"))}
        for side, offset in (("same_view", 24), ("cross_view", 28))
    }
    return result


def run_analysis(sim, ann, s11, bootstrap=10000, seed=2026):
    if bootstrap < 1 or seed < 0:
        raise ValueError("bootstrap must be positive and seed nonnegative")
    raw = evaluate(sim, ann)
    decomposition = error_decomposition(ann, raw)
    expected = s11["B_error_decomposition"]["raw"]["strata"]["all"]
    counts = {cell: decomposition["strata"]["all"]["cells"][cell]["count"] for cell in CELLS}
    reference = {cell: expected["cells"][cell]["count"] for cell in CELLS}
    if not _exact(counts, reference):
        raise RuntimeError(f"S11 four-cell exact gate FAILED: {counts} != {reference}")
    errors, observed, priors = joint_error_inputs(ann, raw)
    groups = {"all": np.ones(len(ann.query_ids), bool),
              "aerial": ann.query_views == 0, "ground": ann.query_views == 1}
    groups.update({str(domain): ann.query_domains == domain for domain in np.unique(ann.query_domains)})
    strata = {name: enrichment_stratum(ann, mask, errors, observed, priors, bootstrap, seed, name in PURE)
              for name, mask in groups.items()}
    mixed = {}
    for domain in MIXED:
        if domain in strata:
            mixed[domain] = {"same_view": strata[domain]["same_view_given_same_source"],
                             "cross_view": strata[domain]["cross_view_given_same_source"],
                             "all_error_view_marginals": strata[domain]["view_marginals_among_all_errors"],
                             "conditioning": "query source is this mixed source AND predicted/negative source is the same"}
    return dict(status="complete", raw=compact_metrics(raw), counts=counts, s11_four_cell_gate="PASS",
                strata=strata, mixed_source_conditionals=mixed,
                definitions={
                    "negative_set": "all gallery images excluding ALL images of the true query identity",
                    "observed": "one-hot error cell mean over the selected top-1 ERROR queries",
                    "expected": "per-query negative-cell fractions averaged equally over the SAME error queries",
                    "conditional_expected": "mean prior(same_source_same_view) / mean prior(same_source_any_view), same error queries",
                    "bootstrap": "uniform whole-identity resampling, retain every query and multiplicity; numerator AND denominator reformed each replicate",
                    "small_prior": "expected <1%: ratio suppressed; share and absolute excess retained",
                    "undefined": "no redraw; undefined count recorded and unconditional CI null; valid-only CI explicitly descriptive",
                })


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("sim", "anno", "s11-json", "out"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = dict(sim=args.sim, anno=args.anno, s11_json=args.s11_json)
    try:
        ann = load_annotations(args.anno)
        report = run_analysis(load_similarity(args.sim, ann), ann, read_json(args.s11_json), args.bootstrap, args.seed)
        print("S11 four-cell gate PASS: " + json.dumps(report["counts"]))
        rows = [(name, cell, item["observed_share"]["estimate"], item["expected_share"]["estimate"],
                 item["absolute_excess"]["estimate"], item["enrichment_ratio"]["estimate"])
                for name, group in report["strata"].items() for cell, item in group["cells"].items()]
        print_table(("group", "cell", "observed", "expected", "excess", "ratio"), rows)
        write_audited_report(args, "R1_joint_enrichment", report, inputs, {"bootstrap": args.seed}, argv)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as exc:
        print(f"R1 joint STOPPED: {exc}", file=sys.stderr)
        try:
            write_audited_report(args, "R1_joint_enrichment", dict(status="stopped", error=str(exc)),
                                 inputs, {"bootstrap": args.seed}, argv)
        except (OSError, ValueError, RuntimeError) as report_error:
            print(f"Stopped-report write unavailable: {report_error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

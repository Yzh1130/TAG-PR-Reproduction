"""CPU-only descriptive and paired statistics for S11 (NumPy + stdlib).

No function changes a similarity matrix, estimates correction terms inside a
bootstrap, or imports the project's model/training/evaluation entry points.
"""

import math

import numpy as np


QB_NORM_REFERENCE = "https://arxiv.org/html/2112.12777v2"


def wilson_interval(successes, total, z=1.959963984540054):
    """Binomial proportion and two-sided 95% Wilson interval; null if empty."""
    successes, total = int(successes), int(total)
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    if not total:
        return {"count": successes, "denominator": total,
                "proportion": None, "ci95": [None, None]}
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {"count": successes, "denominator": total, "proportion": p,
            "ci95": [max(0.0, center - radius), min(1.0, center + radius)]}


def _binary_pair(first_correct, second_correct):
    first = np.asarray(first_correct)
    second = np.asarray(second_correct)
    if first.ndim != 1 or first.shape != second.shape or not first.size:
        raise ValueError("paired outcomes must be equally sized, nonempty 1-D arrays")
    if not np.all((first == 0) | (first == 1)) or not np.all((second == 0) | (second == 1)):
        raise ValueError("paired R@1 outcomes must be binary (0/1)")
    return first.astype(np.int8), second.astype(np.int8)


def mcnemar_exact(first_correct, second_correct):
    """Exact two-sided conditional binomial McNemar test, no SciPy required."""
    first, second = _binary_pair(first_correct, second_correct)
    lost = int(np.count_nonzero((first == 1) & (second == 0)))
    gained = int(np.count_nonzero((first == 0) & (second == 1)))
    discordant = lost + gained
    if not discordant:
        p_value, log_p = 1.0, 0.0
    else:
        lower = min(lost, gained)
        log_terms = np.fromiter(
            (math.lgamma(discordant + 1) - math.lgamma(j + 1)
             - math.lgamma(discordant - j + 1) - discordant * math.log(2.0)
             for j in range(lower + 1)), dtype=np.float64, count=lower + 1)
        peak = float(log_terms.max())
        log_p = min(0.0, math.log(2.0) + peak + math.log(float(np.exp(log_terms - peak).sum())))
        p_value = min(1.0, math.exp(log_p))
    return {"test": "exact two-sided McNemar (conditional Binomial(n_discordant, 0.5))",
            "first_only_correct": lost, "second_only_correct": gained,
            "n_discordant": discordant, "p_two_sided": p_value,
            "log_p_two_sided": log_p,
            "scope": "query-level paired test; does not account for within-image/identity dependence"}


def paired_test(first_correct, second_correct, groups=None, repeats=10000, seed=2026):
    """Paired percentile bootstrap of R@1(second) - R@1(first), in pp.

    With groups, sample whole clusters uniformly with replacement, retaining
    every query (and its multiplicity). The estimator remains query-weighted:
    sum(cluster correct-count differences) / sum(cluster query counts), NOT
    the unweighted mean of per-cluster accuracies.
    """
    first, second = _binary_pair(first_correct, second_correct)
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
        raise ValueError("bootstrap repeats must be a positive integer")
    repeats = int(repeats)
    difference = second.astype(np.float64) - first
    if groups is None:
        unit_difference = difference
        unit_size = np.ones(first.size, dtype=np.int64)
        unit = "query"
    else:
        labels = np.asarray(groups)
        if labels.ndim != 1 or labels.shape != first.shape:
            raise ValueError("one cluster label is required for each paired query")
        _, inverse = np.unique(labels, return_inverse=True)
        unit_difference = np.bincount(inverse, weights=difference)
        unit_size = np.bincount(inverse)
        unit = "cluster"
    n_units = len(unit_size)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    # At most 64 x n_units indices at once: <8.1 MiB for 16,504 queries.
    for start in range(0, repeats, 64):
        stop = min(start + 64, repeats)
        draw = rng.integers(0, n_units, size=(stop - start, n_units), dtype=np.int64)
        estimates[start:stop] = (unit_difference[draw].sum(axis=1)
                                / unit_size[draw].sum(axis=1)) * 100.0
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    # Report the commonly used two-sided bootstrap sign-tail probability.
    # This is NOT a bootstrap null-distribution test; label it explicitly.
    negative_tail = (np.count_nonzero(estimates <= 0) + 1) / (repeats + 1)
    positive_tail = (np.count_nonzero(estimates >= 0) + 1) / (repeats + 1)
    result = {
        "delta_definition": "R1(second) - R1(first), percentage points",
        "observed_delta_pp": float(difference.mean() * 100.0),
        "bootstrap_mean_delta_pp": float(estimates.mean()),
        "bootstrap_std_pp": float(estimates.std(ddof=1)) if repeats > 1 else 0.0,
        "ci95_percentile_pp": [float(lower), float(upper)],
        "p_two_sided": float(min(1.0, 2.0 * min(negative_tail, positive_tail))),
        "p_method": "two-sided percentile-bootstrap sign tails, +1 Monte Carlo correction (not a null bootstrap)",
        "repeats": repeats, "seed": int(seed), "unit": unit,
        "n_units": int(n_units), "n_queries": int(first.size),
        "estimator": "query-weighted accuracy; entire sampled clusters retain all member queries",
    }
    if groups is None:
        result["mcnemar"] = mcnemar_exact(first, second)
    return result


def error_decomposition(ann, evaluation):
    """Four source x view error cells, conditional on an incorrect top-1."""
    details = evaluation["details"]
    query_indices = np.asarray(details["query_indices"], dtype=np.int64)
    predicted = np.asarray(details["top1"], dtype=np.int64)
    correct = np.asarray(details["correct"], dtype=bool)
    if query_indices.shape != predicted.shape or correct.shape != predicted.shape:
        raise ValueError("evaluation details must contain aligned query/top1/correct arrays")
    if np.any(predicted < 0) or np.any(predicted >= len(ann.gallery_ids)):
        raise ValueError("top1 predictions must be valid gallery indices")
    same_source = ann.query_domains[query_indices] == ann.gallery_domains[predicted]
    same_view = ann.query_views[query_indices] == ann.gallery_views[predicted]
    results = {}
    strata = {"all": np.ones(len(query_indices), dtype=bool),
              "aerial": ann.query_views[query_indices] == 0,
              "ground": ann.query_views[query_indices] == 1}
    for name, selected in strata.items():
        errors = selected & ~correct
        total = int(errors.sum())
        cells = {}
        for source_name, source_mask in (("same_source", same_source), ("cross_source", ~same_source)):
            for view_name, view_mask in (("same_view", same_view), ("cross_view", ~same_view)):
                cells[source_name + "_" + view_name] = wilson_interval(
                    np.count_nonzero(errors & source_mask & view_mask), total)
        if sum(item["count"] for item in cells.values()) != total:
            raise AssertionError("the four error categories must partition all top-1 errors")
        conditionals = {}
        for source_name, source_mask in (("same_source", same_source), ("cross_source", ~same_source)):
            denominator = np.count_nonzero(errors & source_mask)
            for view_name, view_mask in (("same_view", same_view), ("cross_view", ~same_view)):
                conditionals["P(" + view_name + "|" + source_name + ",error)"] = wilson_interval(
                    np.count_nonzero(errors & source_mask & view_mask), denominator)
        results[name] = {"n_query": int(selected.sum()), "n_errors": total,
                         "error_fraction": wilson_interval(total, np.count_nonzero(selected)),
                         "cells": cells, "conditional_probabilities": conditionals}
    return {"denominator_note": "cell proportions are conditional on top-1 being incorrect; zero denominators return null",
            "interval": "two-sided 95% Wilson, z=1.959963984540054",
            "strata": results}


def _stable_topk_rows(values, k):
    """Partial selection with exact ties resolved by ascending candidate index."""
    values = np.asarray(values)
    if values.ndim != 2 or not 1 <= k <= values.shape[1]:
        raise ValueError("top-k requires a matrix and 1 <= k <= candidate count")
    boundary = np.partition(values, values.shape[1] - k, axis=1)[:, values.shape[1] - k]
    output = np.empty((values.shape[0], k), dtype=np.int64)
    for i, row in enumerate(values):
        above = np.flatnonzero(row > boundary[i])
        tied = np.flatnonzero(row == boundary[i])[:k - len(above)]
        candidates = np.concatenate((above, tied))
        order = np.lexsort((candidates, -row[candidates]))
        output[i] = candidates[order]
    return output


def reciprocal_statistics(sim, ks=(5, 10, 20, 50), block_size=128):
    """Cross-modal reciprocal edges; neither labels nor a q-q matrix is used."""
    if len(sim.shape) != 2 or min(sim.shape) < 1:
        raise ValueError("reciprocal statistics require a nonempty query-gallery matrix")
    if block_size < 1 or not ks or any(int(k) != k or k < 1 for k in ks):
        raise ValueError("reciprocal k values and block_size must be positive integers")
    n_query, n_gallery = sim.shape
    if max(ks) > min(n_query, n_gallery):
        raise ValueError("reciprocal k cannot exceed either query or gallery count")
    max_k = int(max(ks))
    row_neighbors = np.empty((n_query, max_k), dtype=np.int64)
    column_neighbors = np.empty((n_gallery, max_k), dtype=np.int64)
    for start in range(0, n_query, block_size):
        stop = min(start + block_size, n_query)
        block = np.asarray(sim[start:stop])
        if not np.isfinite(block).all():
            raise ValueError("similarity matrix contains non-finite entries")
        row_neighbors[start:stop] = _stable_topk_rows(block, max_k)
    for start in range(0, n_gallery, block_size):
        stop = min(start + block_size, n_gallery)
        column_neighbors[start:stop] = _stable_topk_rows(np.asarray(sim[:, start:stop]).T, max_k)
    by_k = {}
    for k in ks:
        k = int(k)
        reciprocal_per_query = np.zeros(n_query, dtype=np.int64)
        for start in range(0, n_query, block_size):
            stop = min(start + block_size, n_query)
            candidates = row_neighbors[start:stop, :k]
            # [chunk, k, k], bounded independently of the full matrix size.
            reverse_queries = column_neighbors[candidates, :k]
            query_index = np.arange(start, stop)[:, None, None]
            reciprocal_per_query[start:stop] = np.any(reverse_queries == query_index, axis=2).sum(axis=1)
        count = int(reciprocal_per_query.sum())
        if not 0 <= count <= min(n_query * k, n_gallery * k):
            raise AssertionError("reciprocal pair count violates edge conservation")
        histogram = np.bincount(reciprocal_per_query, minlength=k + 1)
        if int(histogram.sum()) != n_query:
            raise AssertionError("reciprocal histogram must contain every query")
        by_k[str(k)] = {
            "k": k, "reciprocal_pairs": count,
            "query_topk_candidate_pairs": int(n_query * k),
            "fraction_of_query_topk_candidates": float(count / (n_query * k)),
            "gallery_topk_candidate_pairs": int(n_gallery * k),
            "fraction_of_gallery_topk_candidates": float(count / (n_gallery * k)),
            "mean_reciprocal_per_query": float(reciprocal_per_query.mean()),
            "std_reciprocal_per_query": float(reciprocal_per_query.std()),
            "per_query_distribution": {str(i): int(n) for i, n in enumerate(histogram)},
        }
    return {"definition": "(t,g) is reciprocal iff g is in row-top-k(t) and t is in column-top-k(g)",
            "tie_break": "descending score, then ascending original index",
            "candidate_note": "cross-modal neighbors; source images are not excluded here",
            "block_size": int(block_size), "results": by_k}


def _occurrence_stats(top1, n_gallery):
    top1 = np.asarray(top1, dtype=np.int64)
    if top1.ndim != 1 or n_gallery < 1 or np.any(top1 < 0) or np.any(top1 >= n_gallery):
        raise ValueError("N1 statistics require valid gallery indices and a positive gallery count")
    counts = np.bincount(top1, minlength=n_gallery)
    if int(counts.sum()) != len(top1):
        raise AssertionError("N1 counts must sum to the number of included queries")
    mean = float(counts.mean())
    centered = counts.astype(np.float64) - mean
    variance = float(np.mean(centered * centered))
    skew = float(np.mean(centered ** 3) / variance ** 1.5) if variance > 0 else None
    ascending = np.sort(counts)
    total = int(counts.sum())
    gini = (float(2 * np.dot(np.arange(1, n_gallery + 1, dtype=np.float64), ascending)
                  / (n_gallery * total) - (n_gallery + 1) / n_gallery)
            if total else None)
    if gini is not None:
        gini = max(0.0, min(1.0, gini))
    unique, frequency = np.unique(counts, return_counts=True)
    result = {"n_gallery": int(n_gallery), "n_query": len(top1),
              "n_selected_gallery": int(np.count_nonzero(counts)),
              "mean_N1": mean, "std_N1": math.sqrt(variance),
              "skewness_N1": skew, "gini_N1": gini,
              "N1_counts_in_gallery_order": counts.tolist(),
              "N1_histogram": {str(int(x)): int(n) for x, n in zip(unique, frequency)},
              "top10_counts": ascending[::-1][:10].tolist(),
              "head_rounding": "ceil(fraction * total gallery size), including zero-count gallery images"}
    for fraction, name in ((0.01, "head_1pct"), (0.05, "head_5pct")):
        size = int(math.ceil(n_gallery * fraction))
        head_queries = int(ascending[-size:].sum())
        result[name] = {"n_gallery": size, "n_query": head_queries,
                        "query_share": float(head_queries / total) if total else None}
    return result


def hubness_comparison(ann, raw, corrected):
    before = np.asarray(raw["details"]["query_indices"], dtype=np.int64)
    after = np.asarray(corrected["details"]["query_indices"], dtype=np.int64)
    if not np.array_equal(before, after):
        raise ValueError("hubness comparison must evaluate the same query indices in the same order")
    result = {
        "skewness_convention": "E[(N1 - mean(N1))^3] / std(N1)^3; population moments; ALL gallery items including zeros",
        "reference": QB_NORM_REFERENCE,
        "reference_note": "QB-Norm uses the same standardized-third-moment definition for Nk (normally k=10); this task explicitly requests N1",
        "undefined_note": "constant N1 has null skewness; empty query subsets have null Gini/head shares",
    }
    for name, evaluation in (("raw", raw), ("csls_k10", corrected)):
        predicted = np.asarray(evaluation["details"]["top1"], dtype=np.int64)
        errors = ~np.asarray(evaluation["details"]["correct"], dtype=bool)
        result[name] = {"R1": float(evaluation["R1"]),
                        "all_queries": _occurrence_stats(predicted, len(ann.gallery_ids)),
                        "errors_only": _occurrence_stats(predicted[errors], len(ann.gallery_ids))}
    result["delta_R1_pp"] = float(corrected["R1"] - raw["R1"])
    for key in ("all_queries", "errors_only"):
        old = result["raw"][key]["skewness_N1"]
        new = result["csls_k10"][key]["skewness_N1"]
        result["delta_skewness_" + key] = new - old if old is not None and new is not None else None
    return result


def clustered_correction_bootstrap(ann, raw, corrected, repeats=10000, seed=2026):
    queries = np.asarray(raw["details"]["query_indices"], dtype=np.int64)
    if not np.array_equal(queries, np.asarray(corrected["details"]["query_indices"])):
        raise ValueError("paired bootstrap requires the same ordered query set in both evaluations")
    first, second = raw["details"]["correct"], corrected["details"]["correct"]
    results = {}
    for name, groups in (("query_level", None),
                         ("source_image_clustered", ann.query_source_indices[queries]),
                         ("identity_clustered", ann.query_ids[queries])):
        results[name] = paired_test(first, second, groups=groups, repeats=repeats, seed=seed)
    return {"correction": "CSLS k=10; r(g) estimated ONCE from all test queries, held fixed during every bootstrap",
            "resampling_note": "resample evaluation units only; paired raw/corrected outcomes stay together",
            "results": results}


def run_descriptive(sim, ann, raw, corrected, penalties, bootstrap=10000, seed=2026):
    """S11 B-E. ``corrected`` must be the once-computed full-query CSLS k10."""
    penalties = np.asarray(penalties)
    if penalties.shape != (sim.shape[1],) or not np.isfinite(penalties).all():
        raise ValueError("full-query CSLS penalties must be a finite gallery-length vector")
    if sim.shape != (len(ann.query_ids), len(ann.gallery_ids)):
        raise ValueError("similarity shape does not match annotation query/gallery counts")
    return {
        "B_error_decomposition": {"raw": error_decomposition(ann, raw),
                                  "csls_k10": error_decomposition(ann, corrected)},
        "C_reciprocal": reciprocal_statistics(sim),
        "D_hubness": hubness_comparison(ann, raw, corrected),
        "E_bootstrap": clustered_correction_bootstrap(ann, raw, corrected, repeats=bootstrap, seed=seed),
    }

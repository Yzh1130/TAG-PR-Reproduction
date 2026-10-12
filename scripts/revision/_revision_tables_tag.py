"""Render recorded TAG-PEDES analysis results, without evaluating retrieval.

Producer schemas are R1_table5_clusters, R1_joint_enrichment and
R3_finegrained_causes. Only the explicitly requested ten stored Table-5
gallery trials are summarized; missing metrics/intervals are never inferred.
"""

from __future__ import annotations

from numbers import Real
import math
import statistics


CELLS = (
    "same_source_same_view", "same_source_cross_view",
    "cross_source_same_view", "cross_source_cross_view",
)
CELL_LABELS = (
    "Same source / same view", "Same source / cross view",
    "Cross source / same view", "Cross source / cross view",
)
POLICIES = ("a_fixed_misses", "b_common_valid_queries", "c_at_least_two_positives")


def _dict(value, context):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected an object in the producer JSON")
    return value


def _obj(data, *path):
    result = _dict(data, "root")
    for key in path:
        result = _dict(result.get(key), ".".join(map(str, path)))
    return result


def _title(name, label):
    return f"## {name}" + (f" — {label}" if label else "") + "\n\n"


def _gallery(metrics, fmt, context):
    metrics = _dict(metrics, context)
    return (fmt.count(metrics.get("gallery_size_min"), context + ".gallery_size_min")
            + "–" + fmt.count(metrics.get("gallery_size_max"), context + ".gallery_size_max"))


def _ten_trial_summary(trials, direction, metric, fmt, context):
    if not isinstance(trials, list) or len(trials) != 10:
        fmt.reason(f"{context}: requires exactly 10 recorded authoritative_trials; no missing draw is filled")
        return "n/a"
    values = [_obj(row, direction, "metrics").get(metric) for row in trials]
    if any(value is None for value in values):
        fmt.reason(f"{context}: at least one of the 10 recorded trial metrics is missing/null")
        return "n/a"
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{context}: all ten trial metrics must be finite numbers")
    return fmt.mean_std(statistics.fmean(values), statistics.pstdev(values), context)


def _bootstrap(row, fmt, context):
    row = _dict(row, context)
    return fmt.with_ci(row.get("observed_delta_pp"), row.get("ci95_percentile_pp"), context, signed=True)


def render_table5(data, fmt, label=None):
    """Six protocol rows plus explicit rule, identity and zero-positive evidence."""
    data = _dict(data, "table5")
    groups = _obj(data, "groups")
    main_rows, side_rows, policy_rows = [], [], []
    for side, cross in (("A", "G"), ("G", "A")):
        context = f"groups.{side}"
        group = _dict(groups.get(side), context)
        original = _obj(group, "original")
        trials = group.get("authoritative_trials")
        policies = _obj(group, "zero_positive_policies")
        reference = _obj(group, "table5_reference")
        for direction, protocol in (("same", f"{side}→{side}"), ("cross", f"{side}→{cross}")):
            metrics = _obj(original, direction, "metrics")
            main_rows.append([
                protocol,
                fmt.count(group.get("n_queries"), context + ".n_queries"),
                _gallery(metrics, fmt, context + f".original.{direction}.metrics"),
                fmt.count(group.get("effective_controlled_gallery_size"), context + ".effective_controlled_gallery_size"),
                fmt.number(metrics.get("R1"), context + f".original.{direction}.metrics.R1"),
                fmt.number(metrics.get("mAP"), context + f".original.{direction}.metrics.mAP"),
                _ten_trial_summary(trials, direction, "R1", fmt, context + f".authoritative_trials.{direction}.R1"),
                _ten_trial_summary(trials, direction, "mAP", fmt, context + f".authoritative_trials.{direction}.mAP"),
                "—", "—", "—", "—", "—",
            ])
        a_rule = _dict(policies.get("a_fixed_misses"), context + ".zero_positive_policies.a_fixed_misses")
        main_rows.append([
            f"{side} paired cross−same",
            fmt.count(group.get("n_queries"), context + ".n_queries"), "—",
            fmt.count(group.get("effective_controlled_gallery_size"), context + ".effective_controlled_gallery_size"),
            "—", "—", "—", "—",
            fmt.number(reference.get("original_cross_minus_same_pp"), context + ".table5_reference.original_cross_minus_same_pp", signed=True),
            _bootstrap(original.get("identity_bootstrap"), fmt, context + ".original.identity_bootstrap"),
            fmt.number(reference.get("controlled_cross_minus_same_pp"), context + ".table5_reference.controlled_cross_minus_same_pp", signed=True),
            _bootstrap(a_rule.get("identity_bootstrap_conditional_on_fixed_draws"), fmt, context + ".a.identity_bootstrap_conditional_on_fixed_draws"),
            _bootstrap(a_rule.get("two_stage_identity_and_gallery_bootstrap"), fmt, context + ".a.two_stage_identity_and_gallery_bootstrap"),
        ])
        same, other = _obj(original, "same"), _obj(original, "cross")
        zero_draws = []
        if isinstance(trials, list) and trials:
            for index, trial in enumerate(trials):
                trial = _dict(trial, context + f".authoritative_trials[{index}]")
                zeros = []
                for direction in ("same", "cross"):
                    detail = _obj(trial, direction)
                    zeros.append(fmt.count(detail.get("zero_positive_queries"), context + f".trial{index}.{direction}.zero_positive_queries")
                                 + "/" + fmt.count(detail.get("zero_positive_identities"), context + f".trial{index}.{direction}.zero_positive_identities"))
                zero_draws.append(fmt.text(trial.get("seed"), context + f".trial{index}.seed") + ": " + ", ".join(zeros))
            zeros_text = "; ".join(zero_draws)
        else:
            fmt.reason(f"{context}.authoritative_trials: controlled zero-positive counts not recorded")
            zeros_text = "n/a"
        side_rows.append([
            side,
            fmt.count(group.get("n_identities"), context + ".n_identities"),
            fmt.count(group.get("n_queries"), context + ".n_queries"),
            fmt.count(same.get("zero_positive_queries"), context + ".original.same.zero_positive_queries") + "/"
            + fmt.count(other.get("zero_positive_queries"), context + ".original.cross.zero_positive_queries"),
            fmt.count(same.get("zero_positive_identities"), context + ".original.same.zero_positive_identities") + "/"
            + fmt.count(other.get("zero_positive_identities"), context + ".original.cross.zero_positive_identities"),
            zeros_text,
        ])
        for policy in POLICIES:
            row = _dict(policies.get(policy), context + ".zero_positive_policies." + policy)
            pcontext = context + ".zero_positive_policies." + policy
            status = fmt.text(row.get("status"), pcontext + ".status")
            if row.get("status") == "infeasible":
                reason = row.get("reason")
                fmt.reason(f"{pcontext}: {reason if reason is not None else 'infeasible, no reason recorded'}")
                policy_rows.append([side, policy, status, "n/a", "n/a", "n/a", fmt.text(reason, pcontext + ".reason")])
            else:
                policy_rows.append([
                    side, policy, status,
                    fmt.mean_std(row.get("mean_of_per_draw_paired_deltas_pp"), row.get("std_of_per_draw_paired_deltas_pp"), pcontext + ".per_draw_delta_pp", signed=True),
                    _bootstrap(row.get("identity_bootstrap_conditional_on_fixed_draws"), fmt, pcontext + ".identity_bootstrap_conditional_on_fixed_draws"),
                    _bootstrap(row.get("two_stage_identity_and_gallery_bootstrap"), fmt, pcontext + ".two_stage_identity_and_gallery_bootstrap"),
                    fmt.text(row.get("per_query_average_policy"), pcontext + ".per_query_average_policy"),
                ])
    body = _title("Table 5 — common-query controlled evaluation", label)
    body += fmt.table([
        "Protocol", "Common queries", "Original gallery min–max", "Controlled gallery",
        "Original R@1 (%)", "Original mAP (%)", "Controlled R@1 mean ± std (%)",
        "Controlled mAP mean ± std (%)", "Original displayed Δ (pp)",
        "Original bootstrap Δ [identity CI] (pp)", "Controlled displayed Δ (pp)",
        "Controlled bootstrap Δ [identity CI] (pp)", "Controlled bootstrap Δ [two-stage CI] (pp)",
    ], main_rows)
    body += ("\nControlled metric mean ± std summarizes exactly ten stored "
             "`groups.{A,G}.authoritative_trials[*].{same,cross}.metrics`; std is population (ddof=0). "
             "Displayed paired differences use `table5_reference`; bootstrap differences use the "
             "producer's float64 per-query estimator, not rounded/displayed metric subtraction. "
             "The primary controlled intervals are rule a; rules a/b/c are kept separate below. "
             "A dash is a structurally inapplicable cell, not a missing value.\n\n")
    body += "### Identity and zero-positive accounting\n\n"
    body += fmt.table(["Side", "n_A / n_G (side identities)", "Common queries", "Original zero-positive queries same/cross",
                       "Original zero-positive identities same/cross", "Controlled per draw: seed: same queries/IDs, cross queries/IDs"], side_rows)
    counts = _obj(data, "unique_identity_counts")
    body += "\n" + fmt.table(["n_A", "n_G", "Identity union"], [[
        fmt.count(counts.get("n_A"), "unique_identity_counts.n_A"),
        fmt.count(counts.get("n_G"), "unique_identity_counts.n_G"),
        fmt.count(counts.get("union"), "unique_identity_counts.union"),
    ]])
    body += "\n### Zero-positive rules a/b/c\n\n"
    body += fmt.table(["Side", "Rule", "Status", "Per-draw paired Δ mean ± std (pp)",
                       "Δ [identity CI] (pp)", "Δ [two-stage CI] (pp)", "Recorded rule / reason"], policy_rows)
    return body


def _record(item, key):
    return _dict(_dict(item, "record").get(key), key)


def _estimate_ci(item, key, fmt, context, percent=False, signed=False):
    item = _record(item, key)
    if item.get("estimate") is None and item.get("reason"):
        fmt.reason(f"{context}: {item['reason']}")
    return fmt.with_ci(item.get("estimate"), item.get("ci95_percentile"), context, percent=percent, signed=signed)


def _conditional_rows(group, fmt, context, prefix):
    result = []
    for key, description in (("same_view_given_same_source", "P(same view | same source)"),
                             ("cross_view_given_same_source", "P(cross view | same source)")):
        item = _obj(group, key)
        result.append([
            prefix + description,
            _estimate_ci(item, "observed", fmt, context + "." + key + ".observed", percent=True),
            fmt.number(_record(item, "expected").get("estimate"), context + "." + key + ".expected", percent=True),
            _estimate_ci(item, "enrichment_ratio", fmt, context + "." + key + ".enrichment_ratio"),
            fmt.number(_record(item, "absolute_excess").get("estimate"), context + "." + key + ".absolute_excess", signed=True, percent=True),
        ])
    return result


def render_table3(data, fmt, label=None):
    """Joint four-cell enrichment and the producer's source-conditioned results."""
    data = _dict(data, "table3")
    all_group = _obj(data, "strata", "all")
    cells = _obj(all_group, "cells")
    counts = _obj(data, "counts")
    rows = []
    for key, name in zip(CELLS, CELL_LABELS):
        context = "strata.all.cells." + key
        item = _dict(cells.get(key), context)
        rows.append([
            name, fmt.count(counts.get(key), "counts." + key),
            _estimate_ci(item, "observed_share", fmt, context + ".observed_share", percent=True),
            fmt.number(_record(item, "expected_share").get("estimate"), context + ".expected_share", percent=True),
            _estimate_ci(item, "enrichment_ratio", fmt, context + ".enrichment_ratio"),
            fmt.number(_record(item, "absolute_excess").get("estimate"), context + ".absolute_excess", signed=True, percent=True),
        ])
    body = _title("Table 3 — joint source/view enrichment", label)
    body += fmt.table(["Error cell", "Count", "Observed share (%) [identity CI]", "Conditional-negative prior share (%)",
                       "Enrichment ratio [identity CI]", "Absolute excess (pp)"], rows)
    body += "\n### Conditional view enrichment\n\n"
    conditionals = _conditional_rows(all_group, fmt, "strata.all", "All: ")
    mixed = _obj(data, "mixed_source_conditionals")
    for source in ("AG-ReID.v2", "G2APS"):
        item = _dict(mixed.get(source), "mixed_source_conditionals." + source)
        # Producer uses same_view/cross_view here, not the strata key names.
        group = {"same_view_given_same_source": item.get("same_view"),
                 "cross_view_given_same_source": item.get("cross_view")}
        conditionals.extend(_conditional_rows(group, fmt, "mixed_source_conditionals." + source, source + ": "))
    body += fmt.table(["Condition", "Observed (%) [identity CI]", "Conditional prior (%)",
                       "Enrichment ratio [identity CI]", "Absolute excess (pp)"], conditionals)
    body += "\n### Pure-view source applicability\n\n"
    pure_rows = []
    for source in ("Market", "PRAI-1581", "uavhuman"):
        group = _obj(data, "strata", source)
        explanation = _obj(group, "inapplicable_cells").get("same_source_cross_view")
        if explanation is None:
            pure_rows.append([source, "n/a", fmt.text(None, f"strata.{source}.inapplicable_cells.same_source_cross_view")])
        else:
            pure_rows.append([source, "inapplicable", fmt.text(explanation, f"strata.{source}.inapplicable_cells.same_source_cross_view")])
    body += fmt.table(["Source", "Same-source cross-view condition", "Recorded reason"], pure_rows)
    definitions = _obj(data, "definitions")
    body += "\nDefinitions from input:\n\n" + fmt.table(["Field", "Definition"], [
        [name, fmt.text(definitions.get(name), "definitions." + name)]
        for name in ("negative_set", "observed", "expected", "conditional_expected", "bootstrap", "small_prior")
    ])
    return body


def _mean_ci(item, fmt, context, percent=False, signed=False):
    item = _dict(item, context)
    return fmt.with_ci(item.get("mean"), item.get("ci95"), context, percent=percent, signed=signed)


def render_r3(data, fmt, label=None):
    """Area-quartile, lexical overlap, similarity-margin and truth-rank tables."""
    data = _dict(data, "table-r3")
    quartiles = _obj(data, "image_size", "source_quartiles")
    rows = []
    for source, source_data in quartiles.items():
        source_data = _dict(source_data, f"image_size.source_quartiles.{source}")
        groups = source_data.get("groups")
        if groups is None:
            fmt.reason(f"image_size.source_quartiles.{source}.groups: missing/null")
            rows.append([source, "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        if not isinstance(groups, list):
            raise ValueError(f"image_size.source_quartiles.{source}.groups: expected a list")
        for index, group in enumerate(groups):
            context = f"image_size.source_quartiles.{source}.groups[{index}]"
            group = _dict(group, context)
            metrics = _obj(group, "metrics")
            interval = _obj(group, "R1_identity_bootstrap_proportion").get("ci95")
            # R1 metric is recorded in percent; bootstrap interval in proportion.
            converted = None if interval is None else [None if x is None else x * 100 for x in interval]
            rows.append([
                source, fmt.count(group.get("quartile"), context + ".quartile"),
                fmt.count(group.get("n_source_images"), context + ".n_source_images"),
                fmt.count(group.get("n_queries"), context + ".n_queries"),
                fmt.with_ci(metrics.get("R1"), converted, context + ".R1_percent"),
                _mean_ci(group.get("same_source_same_view_error_share_among_errors"), fmt, context + ".SSSV_given_error", percent=True),
            ])
    if not quartiles:
        fmt.reason("image_size.source_quartiles: missing/empty; no source rows invented")
    body = _title("R3 — descriptive fine-grained error associations", label)
    body += fmt.table(["Source", "Area quartile", "Source images", "n queries", "R@1 (%) [identity CI]",
                       "SSSV | error (%) [identity CI]"], rows)
    image_size = _obj(data, "image_size")
    body += "\n" + fmt.text(image_size.get("quartile_basis"), "image_size.quartile_basis") + "\n"
    body += "\n### Attribute overlap (Jaccard, unscaled)\n\n"
    attributes = _obj(data, "attribute_overlap")
    attribute_rows = []
    for key, name, signed in (
        ("error_top1", "Error top-1, all errors", False),
        ("error_top1_paired_subset", "Error top-1, paired subset", False),
        ("random_same_source_negative", "Random same-source negative, paired subset", False),
        ("paired_error_minus_random", "Paired error minus random", True),
    ):
        context = "attribute_overlap." + key
        item = _dict(attributes.get(key), context)
        attribute_rows.append([name, fmt.count(item.get("n_queries"), context + ".n_queries"),
                               fmt.count(item.get("n_identities"), context + ".n_identities"),
                               _mean_ci(item, fmt, context, signed=signed)])
    body += fmt.table(["Comparison", "n queries", "n identities", "Mean [identity CI]"], attribute_rows)
    body += "\nQueries with no same-source negative: " + fmt.count(attributes.get("queries_with_no_same_source_negative"), "attribute_overlap.queries_with_no_same_source_negative") + ".\n"
    body += "\n### Error similarity margins and best-positive ranks\n\n"
    distributions = []
    quantiles = []
    for key, description in (("similarity_margin", "Error-top1 minus best-positive similarity"),
                             ("truth_rank", "Best-positive rank among full gallery")):
        distribution = _obj(data, key, "distribution")
        distributions.append([
            description, fmt.count(distribution.get("count"), key + ".distribution.count"),
            *[fmt.number(distribution.get(stat), key + ".distribution." + stat) for stat in ("mean", "std", "min", "max")],
        ])
        recorded = _obj(distribution, "quantiles")
        if not recorded:
            fmt.reason(f"{key}.distribution.quantiles: missing/empty")
            quantiles.append([description, "n/a", "n/a"])
        else:
            for quantile, value in sorted(recorded.items(), key=lambda pair: float(pair[0])):
                quantiles.append([description, quantile, fmt.number(value, key + ".distribution.quantiles." + quantile)])
    body += fmt.table(["Distribution", "n errors", "Mean", "Population std", "Minimum", "Maximum"], distributions)
    body += "\n" + fmt.table(["Distribution", "Recorded quantile", "Value"], quantiles)
    body += "\nSimilarity-margin definition: " + fmt.text(_obj(data, "similarity_margin").get("definition"), "similarity_margin.definition") + ".\n"
    return body

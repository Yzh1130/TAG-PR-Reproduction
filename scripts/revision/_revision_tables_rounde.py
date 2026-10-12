"""Render recorded Round-E evidence; never run producers, models, or metrics."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean, pstdev

from scripts.revision._revision_tables_ext import S13_ROWS, _get, _mapping


METRICS = ("r1", "r5", "r10", "mAP")
SOURCES = ("AG-ReID.v2", "G2APS", "Market", "PRAI-1581", "uavhuman")
PURE_CAM = {"Market": 1, "PRAI-1581": 0, "uavhuman": 0}
PROTOCOLS = (("Official aerial", "T2A_official"),
             ("Controlled aerial (5 gallery draws)", "T2A_size_controlled_4220"),
             ("Ground only", "T2G_ground_only"), ("Mixed", "T2M_mixed"))
PAIRS = (("Controlled aerial − official aerial", "size_controlled_aerial_minus_official_aerial"),
         ("Ground − official aerial", "ground_minus_official_aerial"),
         ("Mixed − official aerial", "mixed_minus_official_aerial"))


def _list(value, context):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected an array")
    return value


def _unique_labels(documents, context):
    labels = [label for label, _ in documents]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{context}: model labels must be unique")


def _recall(stats, index, context):
    recalls = _list(stats.get("class_recall"), context + ".class_recall")
    if recalls and len(recalls) != 2:
        raise ValueError(context + ".class_recall: expected aerial/ground entries")
    return recalls[index] if recalls else None


def _pure_result(counts, expected, context, fmt):
    """Recheck the producer's pure-source condition from saved counts only."""
    opposite, correct = counts.get(str(1 - expected)), counts.get(str(expected))
    if opposite is None or correct is None:
        return fmt.text(None, context + ".pure_view_check")
    fmt.count(opposite, context + ".opposite_count")
    fmt.count(correct, context + ".expected_count")
    if opposite < 0 or correct < 0:
        raise ValueError(context + ": camera counts must be nonnegative")
    if correct + opposite == 0:
        return "not present / not asserted"
    return "PASS" if opposite == 0 else "FAIL"


def render_e5(documents, fmt):
    """Router statistics are fractions in the producer; display percentages."""
    _unique_labels(documents, "table-e5")
    rows, source_rows, cam_rows = [], [], []
    for label, data in documents:
        data = _mapping(data, label)
        blocks = _mapping(data.get("blocks"), label + ".blocks")
        order = _list(data.get("block_order"), label + ".block_order")
        if order and (len(set(order)) != len(order) or set(order) != set(blocks)):
            raise ValueError(f"{label}: block_order must name every recorded block exactly once")
        names = order or list(blocks)
        if not names:
            fmt.reason(label + ".blocks: no recorded router blocks")
            names = [None]
        for name in names:
            block = _mapping(blocks.get(name), f"{label}.blocks.{name}")
            prefix = f"{label}.blocks.{name}"
            det = _mapping(block.get("deterministic_argmax"), prefix + ".deterministic_argmax")
            stochastic = _mapping(block.get("gumbel"), prefix + ".gumbel")
            main = data.get("main_report_block")
            title = fmt.text(name, prefix + ".name") + (" (main report)" if name is not None and name == main else "")
            rows.append([label, title, fmt.count(data.get("n_images"), label + ".n_images"),
                         fmt.number(det.get("accuracy"), prefix + ".deterministic_argmax.accuracy", percent=True),
                         fmt.number(_recall(det, 0, prefix + ".deterministic_argmax"), prefix + ".recall_aerial", percent=True),
                         fmt.number(_recall(det, 1, prefix + ".deterministic_argmax"), prefix + ".recall_ground", percent=True),
                         fmt.number(det.get("balanced_accuracy"), prefix + ".balanced_accuracy", percent=True),
                         fmt.count(data.get("n_gumbel"), label + ".n_gumbel"),
                         fmt.mean_std(stochastic.get("accuracy_mean"), stochastic.get("accuracy_std"), prefix + ".gumbel.accuracy", percent=True),
                         fmt.number(stochastic.get("mean_per_image_flip_rate"), prefix + ".gumbel.mean_per_image_flip_rate", percent=True),
                         fmt.number(stochastic.get("fraction_images_ever_flipped"), prefix + ".gumbel.fraction_images_ever_flipped", percent=True),
                         *[fmt.number(_get(block, "logit_margin_quantiles", str(q)), prefix + f".logit_margin_quantiles.{q}")
                           for q in (.005, .01, .05, .5)]])
            by_source = _mapping(block.get("by_source"), prefix + ".by_source")
            for source in SOURCES:
                record = _mapping(by_source.get(source), prefix + ".by_source." + source)
                source_rows.append([label, title, source,
                                    fmt.number(record.get("accuracy"), prefix + ".by_source." + source + ".accuracy", percent=True)])
        table = _mapping(data.get("cam_source_table"), label + ".cam_source_table")
        for source in SOURCES:
            counts = _mapping(table.get(source), label + ".cam_source_table." + source)
            context = label + ".cam_source_table." + source
            if source in PURE_CAM:
                check = _pure_result(counts, PURE_CAM[source], context, fmt)
            else:
                check = "inapplicable (mixed-view source)"
            cam_rows.append([label, source, fmt.count(counts.get("0"), context + ".0"),
                             fmt.count(counts.get("1"), context + ".1"), check])
    return ("## E5 — image-level router evaluation\n\n"
            + fmt.table(["Model", "Router block", "Gallery images", "Argmax accuracy (%)", "Aerial recall (%)",
                         "Ground recall (%)", "Balanced accuracy (%)", "Gumbel draws", "Gumbel accuracy mean ± population std (%)",
                         "Mean per-image modal flip rate (%)", "Images ever flipped (%)",
                         "Margin q0.005", "Margin q0.01", "Margin q0.05", "Margin q0.5"], rows)
            + "\nPer-image flip is the saved '1 - max(class_count_0,class_count_1)/n_gumbel'; the table reports its saved mean, not disagreement with argmax. Complete per-image rates remain in the input JSON. Router margins are unscaled logit differences.\n\n"
            + "### cam_id × source and pure-source consistency\n\n"
            + fmt.table(["Model", "Source", "cam_id=0 (aerial)", "cam_id=1 (ground)", "Pure-view check (rechecked from saved counts)"], cam_rows)
            + "\nThe producer does not store a separate assertion-result boolean. This table rechecks its Market=ground / PRAI-1581=aerial / uavhuman=aerial conditions from saved cam_source_table, without inferring missing counts.\n\n"
            + "### Argmax accuracy by source\n\n"
            + fmt.table(["Model", "Router block", "Source", "Argmax accuracy (%)"], source_rows))


def collect_e6_inputs(root):
    """One explicit E6 RUN_ROOT only; do not guess modes from files or logs."""
    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"table-e6 run directory does not exist: {directory}")
    path = directory / "router_randomness.json"
    if not path.is_file():
        raise ValueError(f"table-e6 requires the exact E6 RUN_ROOT containing router_randomness.json: {directory}")
    return path


def render_e6(documents, fmt):
    """Read exact metrics/deltas/maxima saved by the E6 shell aggregator."""
    _unique_labels(documents, "table-e6")
    rows, maximum_rows = [], []
    for label, data in documents:
        context = f"e6.{label}"
        baseline = _mapping(data.get("argmax"), context + ".argmax")
        if baseline.get("eval_router_mode") not in (None, "argmax"):
            raise ValueError(context + ".argmax: saved router mode is not argmax")
        rows.append([label, "argmax", fmt.count(baseline.get("eval_seed"), context + ".argmax.eval_seed"),
                     *[fmt.number(_get(baseline, "metrics", metric), context + ".argmax.metrics." + metric) for metric in METRICS],
                     *["n/a (reference)"] * 4])
        repetitions = _list(data.get("gumbel"), context + ".gumbel")
        known_seeds = []
        for index, run in enumerate(repetitions):
            run = _mapping(run, context + f".gumbel[{index}]")
            prefix = context + f".gumbel[{index}]"
            if run.get("eval_router_mode") not in (None, "gumbel"):
                raise ValueError(prefix + ": saved router mode is not gumbel")
            if run.get("eval_seed") is not None:
                known_seeds.append(run["eval_seed"])
            rows.append([label, "gumbel", fmt.count(run.get("eval_seed"), prefix + ".eval_seed"),
                         *[fmt.number(_get(run, "metrics", metric), prefix + ".metrics." + metric) for metric in METRICS],
                         *[fmt.number(_get(run, "minus_argmax_pp", metric), prefix + ".minus_argmax_pp." + metric, signed=True) for metric in METRICS]])
        if len(set(known_seeds)) != len(known_seeds):
            raise ValueError(context + ": duplicate recorded Gumbel eval_seed")
        count = data.get("n_repetitions")
        if count is not None:
            fmt.count(count, context + ".n_repetitions")
            if count != len(repetitions):
                raise ValueError(context + ": n_repetitions disagrees with the saved Gumbel run count")
        if not repetitions:
            fmt.reason(context + ".gumbel: no recorded repetition rows")
        maximum_rows.append([label, fmt.count(count, context + ".n_repetitions"),
                             *[fmt.number(_get(data, "observed_maximum_pairwise_difference_pp", metric),
                                          context + ".observed_maximum_pairwise_difference_pp." + metric) for metric in METRICS],
                             fmt.text(data.get("maximum_scope"), context + ".maximum_scope")])
    return ("## E6 — argmax and Gumbel retrieval repetitions\n\n"
            + fmt.table(["Model", "Router", "eval_seed", "R@1 (%)", "R@5 (%)", "R@10 (%)", "mAP (%)",
                         "ΔR@1 vs argmax (pp)", "ΔR@5 vs argmax (pp)", "ΔR@10 vs argmax (pp)", "ΔmAP vs argmax (pp)"], rows)
            + "\nAll metrics and deltas are read from router_randomness.json; evaluation files and checkpoints are not re-evaluated. The argmax row is the reference, not a sampled repetition.\n\n"
            + "### These N repetitions: observed maximum pairwise difference\n\n"
            + fmt.table(["Model", "N", "Observed max R@1 difference (pp)", "Observed max R@5 difference (pp)",
                         "Observed max R@10 difference (pp)", "Observed max mAP difference (pp)", "Recorded scope"], maximum_rows)
            + "\n这 N 次中观测到的最大差异；只描述已保存的这 N 次重复，不是其他种子或总体的差异界限。Maxima are the producer's recorded descriptive values, not recomputed or filled.\n")


def _draw_stats(runs, path, context, fmt):
    """Requested descriptive mean/std over five saved draws, never a CI."""
    if len(runs) != 5:
        fmt.reason(context + ": exactly five complete saved gallery draws required for descriptive mean/std")
        return None, None
    values = [_get(run, *path) for run in runs]
    if any(value is None for value in values):
        fmt.reason(context + ": a draw value is missing; five-draw mean/std is not filled")
        return None, None
    for value in values:
        fmt.number(value, context + ".draw_value")
    return fmean(values), pstdev(values)


def _common_ng(runs, context, fmt):
    values = [run.get("ng") for run in runs]
    if not values or any(value is None for value in values):
        return fmt.count(None, context + ".ng")
    if len(set(values)) != 1:
        raise ValueError(context + ": controlled gallery draw sizes disagree")
    return fmt.count(values[0], context + ".ng")


def render_s7d(documents, fmt):
    """AERI compositions, saved identity CIs, and explicit five-draw summaries."""
    rows, paired_rows, draw_rows, seen = [], [], [], set()
    for label, data in documents:
        seed = _get(data, "_meta", "provenance", "seeds", "run_seed")
        context = f"s7d.{label}.seed={seed}"
        if seed is not None:
            fmt.count(seed, context + ".run_seed")
            key = (label, seed)
            if key in seen:
                raise ValueError(f"table-s7d: duplicate model/training seed {label!r}/{seed}")
            seen.add(key)
        protocols = _mapping(data.get("protocols"), context + ".protocols")
        for title, name in PROTOCOLS:
            p = _mapping(protocols.get(name), context + ".protocols." + name)
            prefix = context + ".protocols." + name
            cells = [label, fmt.count(seed, context + ".run_seed"), title]
            if name == "T2A_size_controlled_4220":
                runs = [_mapping(run, prefix + ".runs") for run in _list(p.get("runs"), prefix + ".runs")]
                seeds = _list(p.get("seeds"), prefix + ".seeds")
                if seeds and len(seeds) != len(runs):
                    raise ValueError(prefix + ": gallery seeds and runs have different counts")
                cells.extend([_common_ng(runs, prefix, fmt), fmt.mean_std(p.get("mean_r1"), p.get("sd_r1"), prefix + ".baseline_R1")])
                for path in (("baseline", "mAP"), ("excess_skew_N10",)):
                    mean, std = _draw_stats(runs, path, prefix + "." + ".".join(path), fmt)
                    cells.append(fmt.mean_std(mean, std, prefix + "." + ".".join(path)))
                for mode, key, mean_key in (("transductive", "trans_delta_r1", "mean_trans_delta"),
                                            ("inductive", "ind_delta_r1", "mean_ind_delta")):
                    _, std = _draw_stats(runs, (key,), prefix + "." + key, fmt)
                    cells.append(fmt.mean_std(p.get(mean_key), std, prefix + "." + mean_key, signed=True) + " [n/a]")
                    fmt.reason(prefix + "." + mode + ": producer stores per-draw identity CIs, not an identity CI for this mean correction gain; CIs are not averaged")
                for index, run in enumerate(runs):
                    draw_context = prefix + f".runs[{index}]"
                    draw_rows.append([label, fmt.count(seed, context + ".run_seed"),
                                      fmt.count(seeds[index] if seeds else None, draw_context + ".gallery_seed"),
                                      fmt.count(run.get("ng"), draw_context + ".ng"),
                                      fmt.number(_get(run, "baseline", "r1"), draw_context + ".baseline.r1"),
                                      fmt.number(_get(run, "baseline", "mAP"), draw_context + ".baseline.mAP"),
                                      fmt.number(run.get("excess_skew_N10"), draw_context + ".excess_skew_N10"),
                                      fmt.with_ci(run.get("trans_delta_r1"), run.get("trans_ci"), draw_context + ".trans", signed=True),
                                      fmt.with_ci(run.get("ind_delta_r1"), run.get("ind_ci"), draw_context + ".ind", signed=True)])
            else:
                cells.extend([fmt.count(p.get("ng"), prefix + ".ng"),
                              fmt.number(_get(p, "baseline", "r1"), prefix + ".baseline.r1"),
                              fmt.number(_get(p, "baseline", "mAP"), prefix + ".baseline.mAP"),
                              fmt.number(p.get("excess_skew_N10"), prefix + ".excess_skew_N10"),
                              fmt.with_ci(p.get("trans_delta_r1"), p.get("trans_ci"), prefix + ".trans", signed=True),
                              fmt.with_ci(p.get("ind_delta_r1"), p.get("ind_ci"), prefix + ".ind", signed=True)])
            rows.append(cells)
        pairs = _mapping(data.get("paired_gallery_differences"), context + ".paired_gallery_differences")
        for title, key in PAIRS:
            record = _mapping(pairs.get(key), context + ".paired_gallery_differences." + key)
            prefix = context + ".paired_gallery_differences." + key
            controlled = key.startswith("size_controlled")
            value_key = "mean_delta_r1_pp" if controlled else "delta_r1_pp"
            ci_key = "identity_ci95_conditional_on_five_draws" if controlled else "identity_ci95"
            cells = [label, fmt.count(seed, context + ".run_seed"), title,
                     fmt.with_ci(record.get(value_key), record.get(ci_key), prefix + ".baseline", signed=True)]
            for mode in ("transductive", "inductive"):
                gain = _mapping(_get(record, "correction_gain_difference", mode), prefix + ".correction_gain_difference." + mode)
                value_key = "mean_delta_gain_r1_pp" if controlled else "delta_gain_r1_pp"
                cells.append(fmt.with_ci(gain.get(value_key), gain.get(ci_key), prefix + ".correction_gain_difference." + mode, signed=True))
            cells.extend([fmt.count(record.get("nq"), prefix + ".nq"),
                          fmt.count(record.get("n_identity_clusters"), prefix + ".n_identity_clusters"),
                          "conditional on five gallery draws" if controlled else "identity resampling"])
            paired_rows.append(cells)
    return ("## S7d — AERI gallery compositions by training seed\n\n"
            + fmt.table(["Model", "Training seed", "Gallery composition", "Gallery images", "Baseline R@1 (%)",
                         "Baseline mAP (%)", "Hub excess skew (N10)", "Transductive ΔR@1 (pp) [identity CI95]",
                         "Inductive ΔR@1 (pp) [identity CI95]"], rows)
            + "\nControlled entries are means ± population std over five gallery draws, not five training seeds. R@1 mean/std and correction-gain means use saved aggregate fields; mAP/hub mean/std and gain std are the explicitly requested descriptive reductions of all five saved run values (ddof=0). Missing draws are not filled. CIs are never averaged.\n\n"
            + "### Controlled gallery: each saved draw and its own identity CI\n\n"
            + fmt.table(["Model", "Training seed", "Gallery draw seed", "Gallery images", "Baseline R@1 (%)", "Baseline mAP (%)",
                         "Hub excess skew (N10)", "Transductive ΔR@1 (pp) [identity CI95]", "Inductive ΔR@1 (pp) [identity CI95]"], draw_rows)
            + "\n### Paired gallery differences relative to official aerial\n\n"
            + fmt.table(["Model", "Training seed", "Composition − official", "Baseline ΔR@1 (pp) [identity CI95]",
                         "Transductive gain difference (pp) [identity CI95]", "Inductive gain difference (pp) [identity CI95]",
                         "Paired queries", "Identity clusters", "CI scope"], paired_rows)
            + "\nPaired correction differences are read from correction_gain_difference, not reconstructed by subtracting independently rounded gains. Training seed comes only from _meta.provenance.seeds.run_seed.\n")


def render_s13_compare(documents, fmt):
    """Compact model × four-setting wide table; leave the old S13 renderer alone."""
    _unique_labels(documents, "s13 --compare")
    headers = ["Model"]
    for _, title, _, _, _ in S13_ROWS:
        headers.extend([title + " R@1 (%)", title + " ΔR@1 (pp) [identity CI95]"])
    rows = []
    for label, data in documents:
        records = _list(data.get("rows"), f"s13.{label}.rows")
        indexed = {}
        for record in records:
            record = _mapping(record, f"s13.{label}.rows")
            key = record.get("setting")
            if key in indexed:
                raise ValueError(f"s13.{label}: duplicate summary setting {key!r}")
            indexed[key] = record
        cells = [label]
        for setting, _, bank, alpha, k in S13_ROWS:
            record = indexed.get(setting, {})
            prefix = f"s13.{label}.{setting}"
            if not record:
                fmt.reason(prefix + ": missing summary-table row; legacy a/b/c is not substituted")
            for key, expected in (("bank", bank), ("alpha", alpha), ("k", k)):
                if key in record and record[key] != expected:
                    raise ValueError(f"{prefix}.{key}: setting differs from the requested fixed summary")
            cells.append(fmt.number(_get(record, "metrics", "R1"), prefix + ".metrics.R1"))
            if setting == "uncorrected":
                cells.append(fmt.number(record.get("delta_R1_pp"), prefix + ".delta_R1_pp", signed=True) + " [n/a]")
                fmt.reason(prefix + ".identity_bootstrap: baseline row has no paired-gain interval")
            else:
                cells.append(fmt.with_ci(record.get("delta_R1_pp"), _get(record, "identity_bootstrap", "ci95_percentile_pp"), prefix + ".delta_R1_pp", signed=True))
        rows.append(cells)
    return ("## S13 — compact model × correction comparison\n\n" + fmt.table(headers, rows)
            + "\nUses only the four --summary-table rows. Baseline identity-gain CI is inapplicable, not zero-filled; the original full-metric S13 rendering remains the default.\n")

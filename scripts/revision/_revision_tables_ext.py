"""Read recorded external/S13 results and render Markdown; no metric computation."""

from __future__ import annotations


DATASETS = ("cuhkpedes", "icfgpedes", "aeri", "tbapr")
NAMES = {"cuhkpedes": "CUHK-PEDES", "icfgpedes": "ICFG-PEDES",
         "aeri": "AERI-PEDES", "tbapr": "TBAPR"}
MODES = ("transductive", "inductive")
FITS = ("trans_cohortQ_delta", "trans_fullQ_delta", "ind_delta")
S13_ROWS = (("uncorrected", "Uncorrected", None, None, None),
            ("inductive_default", "train_full (α=0.5, k=10)", "train_full", 0.5, 10),
            ("inductive_075_30", "train_full (α=0.75, k=30)", "train_full", 0.75, 30),
            ("transductive_k10", "test_full (α=0.5, k=10)", "test_full", 0.5, 10))


def _mapping(value, context):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected an object")
    return value


def _get(value, *keys):
    """Missing recorded fields stay None, never reconstructed from other fields."""
    for key in keys:
        value = _mapping(value, ".".join(map(str, keys))).get(key)
    return value


def render_table11(data, fmt):
    """Per-training-seed R1 and archived population summaries, all in percent/pp."""
    entries = _mapping(data.get("datasets"), "datasets")
    rows = []
    for dataset in DATASETS:
        entry = _mapping(entries.get(dataset), f"datasets.{dataset}")
        per_seed = entry.get("per_seed", [])
        if not isinstance(per_seed, list):
            raise ValueError(f"{dataset}.per_seed: expected an array")
        per_seed = [_mapping(row, f"{dataset}.per_seed[{index}]")
                    for index, row in enumerate(per_seed)]
        known_seeds = [row.get("seed") for row in per_seed if row.get("seed") is not None]
        if len(set(known_seeds)) != len(known_seeds):
            raise ValueError(f"{dataset}: duplicate recorded training seed")
        if not per_seed:
            fmt.reason(f"{dataset}.per_seed: no recorded seed rows")
            per_seed = [{}]
        for index, row in enumerate(per_seed):
            row = _mapping(row, f"{dataset}.per_seed[{index}]")
            context = f"{dataset}.per_seed[{index}]"
            cells = [NAMES[dataset], fmt.count(row.get("seed"), context + ".seed"),
                     fmt.number(_get(row, "baseline", "r1"), context + ".baseline.r1")]
            for mode in MODES:
                cells.append(fmt.with_ci(_get(row, mode, "delta", "r1"),
                                         _get(row, mode, "identity_ci95"),
                                         context + "." + mode, signed=True))
            rows.append(cells)
        summary = _mapping(entry.get("summary"), dataset + ".summary")
        for statistic, title in (("mean", "Seed mean"), ("std", "Population std (ddof=0)")):
            cells = [NAMES[dataset], title,
                     fmt.number(_get(summary, "baseline", "r1", statistic),
                                f"{dataset}.summary.baseline.r1.{statistic}")]
            for mode in MODES:
                cells.append(fmt.number(_get(summary, mode, "delta", "r1", statistic),
                                        f"{dataset}.summary.{mode}.delta.r1.{statistic}",
                                        signed=statistic == "mean"))
            rows.append(cells)
        cells = [NAMES[dataset], "Range [min, max]"]
        for mode in ("baseline", *MODES):
            path = (mode, "r1") if mode == "baseline" else (mode, "delta", "r1")
            lower = _get(summary, *path, "min")
            upper = _get(summary, *path, "max")
            interval = None if lower is None or upper is None else [lower, upper]
            cells.append(fmt.interval(interval, f"{dataset}.summary.{mode}.range",
                                      signed=mode != "baseline"))
        rows.append(cells)
        cells = [NAMES[dataset], "Signs (+ / 0 / −; total)", "n/a"]
        # There is no meaningful sign statistic for baseline accuracy, and the
        # producer only records signs of the correction gains.
        fmt.reason(f"{dataset}.summary.baseline.sign_counts: not applicable to baseline accuracy; producer records correction-gain signs only")
        for mode in MODES:
            signs = _mapping(_get(summary, mode, "sign_counts"), f"{dataset}.{mode}.sign_counts")
            context = f"{dataset}.summary.{mode}.sign_counts"
            cells.append(" / ".join(fmt.count(signs.get(key), context + "." + key)
                                    for key in ("positive", "zero", "negative"))
                         + "; " + fmt.count(signs.get("total"), context + ".total"))
        rows.append(cells)
    return ("## Table 11 — external multi-seed R@1\n\n"
            + fmt.table(["Dataset", "Training seed / statistic", "Baseline R@1 (%)",
                         "Transductive ΔR@1 (pp) [identity CI95]",
                         "Inductive ΔR@1 (pp) [identity CI95]"], rows)
            + "\nAll seed summaries are read from the archived `summary`; no CI or missing statistic is recomputed. Range denotes the recorded [minimum, maximum].\n")


def render_table12(documents, fmt):
    """Recorded distractor slopes/correlations, grouped by actual training seed."""
    rows, seen = [], set()
    for source, data in documents:
        seeds = _mapping(_get(data, "_meta", "provenance", "seeds", "run_seeds"),
                         f"{source}._meta.provenance.seeds.run_seeds")
        for dataset in DATASETS:
            entry = _mapping(data.get(dataset), f"{source}.{dataset}")
            seed = seeds.get(dataset)
            if seed is not None:
                key = (dataset, seed)
                if key in seen:
                    raise ValueError(f"table12: duplicate recorded {dataset} training seed {seed}")
                seen.add(key)
            context = f"{source}.{dataset}"
            for setting in FITS:
                fit = _mapping(_get(entry, "fits", setting), context + ".fits." + setting)
                prefix = context + ".fits." + setting
                rows.append([NAMES[dataset], fmt.count(seed, context + ".training_seed"), setting,
                             fmt.with_ci(fit.get("slope_vs_log10D"), fit.get("slope_ci95"),
                                         prefix + ".slope", signed=True),
                             fmt.number(fit.get("spearman_vs_D"), prefix + ".spearman_vs_D"),
                             fmt.number(entry.get("spearman_excess_skew_vs_D"),
                                        context + ".spearman_excess_skew_vs_D"),
                             fmt.number(fit.get("spearman_vs_excess_skew"),
                                        prefix + ".spearman_vs_excess_skew")])
    return ("## Table 12 — distractor scaling\n\n"
            + fmt.table(["Dataset", "Training seed", "Correction", "Slope vs log10(D) [CI95]",
                         "Spearman(Δ, D)", "Spearman(excess skew, D)",
                         "Spearman(Δ, excess skew)"], rows)
            + "\nTraining seeds come only from `_meta.provenance.seeds.run_seeds[dataset]`; gallery sampling seeds are not training seeds. The excess-skew-vs-D correlation is dataset-level, while the other correlations are fit-level.\n")


def render_s13(documents, fmt):
    """Four S13 --summary-table rows; explicit model labels form parallel columns."""
    by_model, labels = {}, []
    for label, data in documents:
        if label in by_model:
            raise ValueError(f"s13: duplicate explicit model label {label!r}")
        labels.append(label)
        source_rows = data.get("rows", [])
        if not isinstance(source_rows, list):
            raise ValueError(f"s13 {label}.rows: expected an array")
        indexed = {}
        for row in source_rows:
            row = _mapping(row, f"s13 {label}.rows")
            setting = row.get("setting")
            if setting in indexed:
                raise ValueError(f"s13 {label}: duplicate summary setting {setting!r}")
            indexed[setting] = row
        by_model[label] = indexed
    headers = ["Setting"]
    for label in labels:
        headers.extend([f"{label} R@1 (%)", f"{label} R@5 (%)", f"{label} R@10 (%)",
                        f"{label} mAP (%)", f"{label} ΔR@1 (pp) [identity CI95]"])
    output_rows = []
    for setting, title, bank, alpha, k in S13_ROWS:
        cells = [title]
        for label in labels:
            row = by_model[label].get(setting, {})
            context = f"s13.{label}.{setting}"
            if row:
                for key, expected in (("bank", bank), ("alpha", alpha), ("k", k)):
                    if key in row and row[key] != expected:
                        raise ValueError(f"{context}.{key}: recorded setting {row[key]!r} disagrees with {expected!r}")
            else:
                fmt.reason(context + ": summary row is missing; legacy a/b/c output is not substituted")
            for metric in ("R1", "R5", "R10", "mAP"):
                cells.append(fmt.number(_get(row, "metrics", metric), context + ".metrics." + metric))
            if setting == "uncorrected":
                cells.append(fmt.number(row.get("delta_R1_pp"), context + ".delta_R1_pp", signed=True)
                             + " [n/a]")
                fmt.reason(context + ".identity_bootstrap: baseline row has no paired-gain interval")
            else:
                cells.append(fmt.with_ci(row.get("delta_R1_pp"),
                                         _get(row, "identity_bootstrap", "ci95_percentile_pp"),
                                         context + ".delta_R1_pp", signed=True))
        output_rows.append(cells)
    return ("## S13 — fixed-bank correction summary\n\n" + fmt.table(headers, output_rows)
            + "\nRead from S13 `--summary-table` rows, not the legacy a/b/c configurations. Identity intervals are recorded paired-gain percentile intervals; no interval is inferred for the uncorrected row.\n")

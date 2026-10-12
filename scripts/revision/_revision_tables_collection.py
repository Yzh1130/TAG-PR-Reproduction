"""Render collector archives without reading runs or unpickling checkpoints.

Only explicitly requested descriptive aggregation of recorded paired CSV values
is performed. Missing metrics are not reconstructed from other columns.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from scripts.revision.collect_revision_runs import describe, matches_prefix


def _csv_number(value: Any, context: str, fmt):
    if value is None or value == "" or value == "n/a":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{context}: a boolean is not a recorded CSV metric")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{context}: invalid recorded CSV number {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: non-finite recorded CSV metric {value!r}")
    return number


def _seed_sort(seed):
    text = "" if seed is None else str(seed)
    return (0, int(text)) if re.fullmatch(r"[-+]?\d+", text) else (1, text)


def _summary_cells(values, fmt, context, *, signed=False):
    summary = describe(values)
    if not values:
        fmt.reason(f"{context}: no paired row has this recorded metric; not deriving missing values")
        return ["n/a"] * 4
    signs = summary["sign_counts"]
    return [
        fmt.number(summary["mean"], context + ".mean", signed=signed),
        fmt.number(summary["population_std"], context + ".population_std"),
        fmt.number(summary["range"], context + ".range") + " "
        + fmt.interval([summary["min"], summary["max"]], context + ".min_max", signed=signed),
        f"−/{0}/+: {signs['negative']}/{signs['zero']}/{signs['positive']}",
    ]


def render_pairs(documents, fmt, metric="R1"):
    """Merge exact duplicate rows; stop on conflicting pair/scope/seed records.

    ``documents`` is a list of (source path/label, DictReader rows). Collector's
    ``metric`` column is the checkpoint scope (last/best), not the retrieval
    metric selected for this table. Distinct scopes are always separate tables.
    """
    if metric not in ("R1", "R5", "R10", "mAP"):
        raise ValueError("--metric must be R1, R5, R10, or mAP")
    merged = {}
    source_membership = defaultdict(list)
    for source, rows in documents:
        if not isinstance(rows, list):
            raise ValueError(f"{source}: expected a list of CSV rows")
        for line, row in enumerate(rows, 2):
            if not isinstance(row, dict) or not row.get("pair"):
                raise ValueError(f"{source}:{line}: missing nonempty collector pair column")
            scope = row.get("metric") or None
            if scope not in (None, "last", "best"):
                raise ValueError(f"{source}:{line}: unknown collector checkpoint metric scope {scope!r}")
            seed = row.get("seed") or None
            key = (row["pair"], scope, seed)
            if key in merged and merged[key] != row:
                raise ValueError(
                    f"conflicting duplicate pair/scope/seed {key!r}: "
                    f"{source_membership[key][0]} and {source}:{line}; "
                    "not selecting by filename, order, or rounded value")
            merged[key] = row
            source_membership[key].append(f"{source}:{line}")
    if not merged:
        fmt.reason("pairs: input CSVs contain no collector pair rows")
        return "## Seed-paired configurations\n\nNo recorded pair rows.\n"
    groups = defaultdict(list)
    for (pair, scope, seed), row in merged.items():
        groups[(pair, scope)].append((seed, row))
    parts = ["## Seed-paired configurations\n\n"
             f"Retrieval metric: {metric}; checkpoint scopes remain separate. "
             "Statistics use only recorded paired values; population std uses ddof=0.\n"]
    for (pair, scope), items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        context = f"pairs.{pair}.{scope or 'unknown'}"
        parts.append(f"\n### {fmt.text(pair, context + '.pair')} ({fmt.text(scope, context + '.scope')})\n")
        display_rows = []
        values = {side: [] for side in ("A", "B", "delta")}
        for seed, row in sorted(items, key=lambda item: _seed_sort(item[0])):
            row_context = context + f".seed={seed}"
            status = row.get("status") or None
            if status not in ("paired", "unpaired", None):
                raise ValueError(f"{row_context}: unknown collector pairing status {status!r}")
            cells = [fmt.text(seed, row_context + ".seed")]
            if status != "paired":
                reason = row.get("reason") or "pairing status missing; no paired values used"
                fmt.reason(f"{row_context}: {reason}")
                cells.extend(["n/a"] * 3)
            else:
                for side in ("A", "B", "delta"):
                    column = f"{side}_{metric}"
                    value = _csv_number(row.get(column), row_context + "." + column, fmt)
                    cells.append(fmt.number(value, row_context + "." + column, signed=side == "delta"))
                    if value is not None:
                        values[side].append(value)
            cells.extend([fmt.text(status, row_context + ".status"), row.get("reason") or ""])
            display_rows.append(cells)
        parts.append(fmt.table(["seed", "A", "B", "Δ(B−A)", "status", "reason"], display_rows))
        summaries = [_summary_cells(values[side], fmt, context + f".{side}", signed=side == "delta")
                     for side in ("A", "B", "delta")]
        statistic_names = ["mean", "population std (ddof=0)", "range [min, max]", "sign counts (−/0/+)"]
        parts.append(fmt.table(["statistic", "A", "B", "Δ(B−A)"],
                               [[name, *(summary[index] for summary in summaries)]
                                for index, name in enumerate(statistic_names)]))
    return "\n".join(parts)


def _dict(value):
    return value if isinstance(value, dict) else {}


def _epoch_iterations(run, fmt, context):
    epochs = run.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        return fmt.text(None, context + ".epochs.actual_iterations")
    result = []
    for index, epoch in enumerate(epochs):
        if not isinstance(epoch, dict):
            raise ValueError(f"{context}.epochs[{index}]: expected an object")
        entry = context + f".epochs[{index}]"
        result.append("epoch " + fmt.count(epoch.get("epoch"), entry + ".epoch") + ": "
                      + fmt.count(epoch.get("actual_iterations"), entry + ".actual_iterations"))
    return "; ".join(result)


def _checkpoint_listing(run, fmt, context):
    checkpoints = run.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        return fmt.text(None, context + ".checkpoints")
    result = []
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{context}.checkpoints[{index}]: expected an object")
        entry = context + f".checkpoints[{index}]"
        result.append(fmt.text(checkpoint.get("name"), entry + ".name") + " / "
                      + fmt.text(checkpoint.get("sha256_16"), entry + ".sha256_16") + " / "
                      + fmt.count(checkpoint.get("bytes"), entry + ".bytes") + " bytes")
    return "\n".join(result)


def render_runs(data, fmt, include=None):
    """Render exact archived fields; prefix semantics reuse the collector."""
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError("runs JSON must be the collector object containing a runs list")
    prefixes = include or []
    if isinstance(prefixes, str) or not all(isinstance(item, str) and item for item in prefixes):
        raise ValueError("--include must be repeated nonempty prefixes")
    rows = []
    for index, run in enumerate(data["runs"]):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}]: expected an object")
        # directory_name is saved by the collector; do not derive an exp_name
        # or a seed from a directory filename when fields are absent.
        if prefixes:
            if run.get("directory_name") is None:
                raise ValueError(f"runs[{index}]: directory_name needed for collector prefix filtering")
            if not any(matches_prefix(run, prefix) for prefix in prefixes):
                continue
        context = f"runs[{index}]"
        best = _dict(run.get("best"))
        last = _dict(run.get("last"))
        missing = run.get("missing_fields")
        if missing is None:
            missing_cell = fmt.text(None, context + ".missing_fields")
        elif not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
            raise ValueError(f"{context}.missing_fields: expected a list of field names")
        else:
            missing_cell = "; ".join(missing) if missing else "none (recorded empty list)"
            for field in missing:
                fmt.reason(f"{context}.{field}: recorded in collector missing_fields")
        rows.append([
            fmt.text(run.get("directory"), context + ".directory"),
            fmt.text(run.get("exp_name"), context + ".exp_name"),
            fmt.count(run.get("seed"), context + ".seed"),
            fmt.text(run.get("training_git_commit"), context + ".training_git_commit"),
            fmt.count(run.get("batch_size"), context + ".batch_size"),
            fmt.count(run.get("itc_block_size"), context + ".itc_block_size"),
            fmt.count(run.get("completed_epochs"), context + ".completed_epochs"),
            _epoch_iterations(run, fmt, context),
            fmt.count(best.get("epoch"), context + ".best.epoch"),
            fmt.number(_dict(best.get("metrics")).get("R1"), context + ".best.metrics.R1"),
            fmt.number(_dict(last.get("metrics")).get("R1"), context + ".last.metrics.R1"),
            _checkpoint_listing(run, fmt, context),
            missing_cell,
        ])
    if not rows:
        fmt.reason("runs: no archived run matches the requested prefixes")
    return "## Archived checkpoint inventory\n\n" + fmt.table(
        ["directory", "exp_name", "seed", "training commit", "batch_size", "itc_block_size",
         "completed epochs", "actual iterations by epoch", "best epoch", "best R@1", "last R@1",
         "checkpoint / sha256_16 / bytes", "missing_fields"], rows)

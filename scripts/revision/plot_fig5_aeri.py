"""Plot AERI gallery-composition results from explicit S7d/aggregate JSONs.

No features, scores, labels, or retrieval metrics are recomputed. Controlled
gallery summaries are conditional on the five draws saved by S7d; those draws
are never presented as independent training seeds. Input CIs are not pooled.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


COMPOSITIONS = (
    ("official_aerial", "T2A_official"),
    ("controlled_aerial", "T2A_size_controlled_4220"),
    ("ground_only", "T2G_ground_only"),
    ("mixed", "T2M_mixed"),
)
PAIRED = {
    "controlled_aerial": "size_controlled_aerial_minus_official_aerial",
    "ground_only": "ground_minus_official_aerial",
    "mixed": "mixed_minus_official_aerial",
}
COLORS = {"baseline_r1": "#353535", "trans_delta_r1": "#2070b4", "ind_delta_r1": "#db7c19"}


def _load(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context}: expected a finite recorded number, got {value!r}")
    return float(value)


def _ci(value, context):
    if value is None:
        return None, None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context}: expected a recorded two-element CI")
    low, high = [_number(item, context) for item in value]
    if low > high:
        raise ValueError(f"{context}: CI lower endpoint exceeds upper endpoint")
    return low, high


def _evidence(path):
    path = Path(path).expanduser().resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256_16": digest.hexdigest()[:16]}


def _seed(result, path):
    try:
        value = result["_meta"]["provenance"]["seeds"]["run_seed"]
    except KeyError as error:
        raise ValueError(f"{path}: missing _meta.provenance.seeds.run_seed; filename is not a seed source") from error
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: run_seed must be an integer")
    return value


def prepare_data(paths, aggregate_path):
    """Return recorded plotting points; only controlled hub means are derived."""
    if len(paths) != 3:
        raise ValueError("exactly three --s7d-json inputs are required")
    aggregate = _load(aggregate_path)
    try:
        aggregate_rows = aggregate["datasets"]["aeri"]["per_seed"]
    except KeyError as error:
        raise ValueError("aggregate JSON is missing datasets.aeri.per_seed") from error
    aggregate_by_seed = {}
    for row in aggregate_rows:
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("aggregate: per_seed seed must be a recorded integer")
        if seed in aggregate_by_seed:
            raise ValueError(f"aggregate contains duplicate AERI seed {seed}")
        aggregate_by_seed[seed] = row
    results = []
    for path in paths:
        result = _load(path)
        results.append((_seed(result, path), str(Path(path).resolve()), result))
    seeds = [entry[0] for entry in results]
    if len(set(seeds)) != 3:
        raise ValueError("S7d inputs must have three distinct recorded training seeds")
    if set(seeds) != set(aggregate_by_seed):
        raise ValueError(f"S7d/aggregate AERI seeds differ: {sorted(seeds)} versus {sorted(aggregate_by_seed)}")
    points, paired_points, notices, comparisons, sources = [], [], [], [], []
    for seed, path, result in sorted(results):
        protocols = result.get("protocols")
        if not isinstance(protocols, dict):
            raise ValueError(f"{path}: missing protocols object")
        sources.append({"seed": seed, "path": path,
                        "recorded_analysis_seeds": result["_meta"]["provenance"]["seeds"]})
        official = protocols["T2A_official"]["baseline"]["r1"]
        aggregate_official = aggregate_by_seed[seed].get("baseline", {}).get("r1")
        comparisons.append({"seed": seed, "S7d_official_r1": official,
                            "aggregate_official_r1": aggregate_official})
        if aggregate_official != official:
            notices.append(f"seed {seed}: aggregate/S7d official R@1 differs ({aggregate_official!r} / {official!r}); plotting uses S7d only")
        for composition, key in COMPOSITIONS:
            protocol = protocols[key]
            prefix = f"protocols.{key}"
            if composition == "controlled_aerial":
                runs = protocol.get("runs", [])
                draw_seeds = protocol.get("seeds", [])
                if not runs or len(runs) != len(draw_seeds):
                    raise ValueError(f"{path}: controlled runs and saved gallery seeds must align and be nonempty")
                hub = float(np.mean([_number(run["excess_skew_N10"], prefix) for run in runs]))
                values = (protocol["mean_r1"], protocol["mean_trans_delta"], protocol["mean_ind_delta"])
                fields = ("mean_r1", "mean_trans_delta", "mean_ind_delta")
                cis = (None, None, None)
                hub_source = prefix + ".runs[*].excess_skew_N10 (arithmetic mean over saved gallery draws)"
                notices.append(f"seed {seed}: controlled composition uses saved mean fields; hub is the mean over {len(runs)} gallery draws, not additional training seeds; no per-draw CIs were averaged")
            else:
                hub = _number(protocol["excess_skew_N10"], prefix)
                values = (protocol["baseline"]["r1"], protocol["trans_delta_r1"], protocol["ind_delta_r1"])
                fields = ("baseline.r1", "trans_delta_r1", "ind_delta_r1")
                cis = (None, protocol.get("trans_ci"), protocol.get("ind_ci"))
                hub_source = prefix + ".excess_skew_N10"
            for metric, value, field, interval in zip(COLORS, values, fields, cis):
                low, high = _ci(interval, prefix + "." + field)
                points.append({"composition": composition, "seed": seed, "metric": metric,
                               "value": _number(value, prefix + "." + field), "ci_low": low, "ci_high": high,
                               "hub_skew": hub, "source_json": path, "source_field": prefix + "." + field,
                               "hub_source_field": hub_source,
                               "gallery_draws": len(protocol["runs"]) if composition == "controlled_aerial" else None})
        for composition, key in PAIRED.items():
            entry = result.get("paired_gallery_differences", {}).get(key)
            if entry is None:
                notices.append(f"seed {seed}: no paired_gallery_differences.{key}; no paired values or CIs inferred")
                continue
            controlled = composition == "controlled_aerial"
            value_field = "mean_delta_r1_pp" if controlled else "delta_r1_pp"
            ci_field = "identity_ci95_conditional_on_five_draws" if controlled else "identity_ci95"
            low, high = _ci(entry.get(ci_field), key)
            paired_points.append({"composition": composition, "seed": seed, "metric": "baseline_difference_vs_official",
                                  "value": _number(entry[value_field], key), "ci_low": low, "ci_high": high,
                                  "source_json": path, "source_field": f"paired_gallery_differences.{key}.{value_field}",
                                  "ci_source_field": ci_field})
            for mode in ("transductive", "inductive"):
                gain = entry.get("correction_gain_difference", {}).get(mode)
                if gain is None:
                    notices.append(f"seed {seed}: {key} has no {mode} paired correction-gain difference; not inferred")
                    continue
                field = "mean_delta_gain_r1_pp" if controlled else "delta_gain_r1_pp"
                low, high = _ci(gain.get(ci_field), key + "." + mode)
                paired_points.append({"composition": composition, "seed": seed, "metric": mode + "_gain_difference_vs_official",
                                      "value": _number(gain[field], key + "." + mode), "ci_low": low, "ci_high": high,
                                      "source_json": path, "source_field": f"paired_gallery_differences.{key}.correction_gain_difference.{mode}.{field}",
                                      "ci_source_field": ci_field})
    return {"points": points, "paired_points": paired_points, "warnings": notices,
            "training_seeds": sorted(seeds), "sources": sources,
            "aggregate_official_comparison": comparisons,
            "mean_convention": "Arithmetic mean across three recorded training seeds; saved controlled-gallery means remain conditional on their saved gallery draws.",
            "CI_convention": "Input identity CIs only; no averaging or pooling of CI endpoints."}


def _font(path):
    from matplotlib.font_manager import FontProperties
    if path:
        source = Path(path).expanduser()
        if source.is_file():
            try:
                from matplotlib.ft2font import FT2Font
                FT2Font(str(source))
                return FontProperties(fname=str(source.resolve())), True
            except (OSError, RuntimeError, ValueError):
                pass
    print(f"WARNING: Chinese font unavailable ({path!r}); using English figure labels.", flush=True)
    return FontProperties(family="DejaVu Sans"), False


def _text_font(figure, font):
    from matplotlib.text import Text
    for item in figure.findobj(Text):
        size = item.get_fontsize()
        item.set_fontproperties(font)
        item.set_fontsize(size)


def _save_figure(figure, directory, stem, font):
    _text_font(figure, font)
    result = {}
    for suffix in ("png", "pdf"):
        path = directory / f"{stem}.{suffix}"
        with path.open("xb") as handle:
            figure.savefig(handle, format=suffix, dpi=300, bbox_inches="tight")
        result[suffix] = str(path)
    return result


def plot_data(data, directory, tag, font_path=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    font, chinese = _font(font_path)
    labels = (["官方纯空中", "规模受控纯空中", "纯地面", "空地混合"] if chinese else
              ["Official aerial", "Size-controlled aerial", "Ground only", "Aerial + ground"])
    metric_labels = ({"baseline_r1": "原始 R@1", "trans_delta_r1": "转导式 ΔR@1", "ind_delta_r1": "归纳式 ΔR@1"} if chinese else
                     {"baseline_r1": "Baseline R@1", "trans_delta_r1": "Transductive delta R@1", "ind_delta_r1": "Inductive delta R@1"})
    seeds = data["training_seeds"]
    composition_index = {name: index for index, (name, _) in enumerate(COMPOSITIONS)}
    jitter = {seed: (index - 1) * .035 for index, seed in enumerate(seeds)}
    markers = ("o", "s", "^")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))
    gains = left.twinx()
    handles = []
    offsets = {"baseline_r1": -.22, "trans_delta_r1": 0, "ind_delta_r1": .22}
    for metric, color in COLORS.items():
        axis = left if metric == "baseline_r1" else gains
        for point in (row for row in data["points"] if row["metric"] == metric):
            x = composition_index[point["composition"]] + offsets[metric]
            axis.scatter(x + jitter[point["seed"]], point["value"], color=color,
                         marker=markers[seeds.index(point["seed"])], s=38, zorder=3)
        for name, _ in COMPOSITIONS:
            values = [row["value"] for row in data["points"] if row["composition"] == name and row["metric"] == metric]
            x = composition_index[name] + offsets[metric]
            axis.hlines(float(np.mean(values)), x - .075, x + .075, color=color, linewidth=2.2)
        handles.append(plt.Line2D([], [], color=color, marker="o", linestyle="none", label=metric_labels[metric]))
    left.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    left.set_ylabel("原始 R@1 (%)" if chinese else "Baseline R@1 (%)")
    gains.set_ylabel("校正 ΔR@1 (百分点)" if chinese else "Correction delta R@1 (percentage points)")
    left.set_title("图库构成：逐种子点与种子均值" if chinese else "Gallery composition: seed points and seed means")
    # Keep the metric key outside the data area rather than hiding seed points.
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.025), ncol=3, fontsize=9)
    left.grid(axis="x", alpha=.15)
    scatter_handles = []
    for metric in ("trans_delta_r1", "ind_delta_r1"):
        color = COLORS[metric]
        for point in (row for row in data["points"] if row["metric"] == metric):
            right.scatter(point["hub_skew"], point["value"], color=color,
                          marker=markers[seeds.index(point["seed"])], s=42, alpha=.85)
        scatter_handles.append(plt.Line2D([], [], color=color, marker="o", linestyle="none", label=metric_labels[metric]))
    scatter_handles += [plt.Line2D([], [], color="#666666", marker=markers[index], linestyle="none", label=f"seed {seed}") for index, seed in enumerate(seeds)]
    right.legend(handles=scatter_handles, fontsize=8)
    right.set_xlabel("Hub 超额偏度 (N10)" if chinese else "Hub excess skewness (N10)")
    right.set_ylabel("ΔR@1 (百分点)" if chinese else "Delta R@1 (percentage points)")
    right.set_title("Hub 偏度与校正增益：描述性散点" if chinese else "Hub skewness and correction gain: descriptive scatter")
    right.grid(alpha=.15)
    fig.tight_layout()
    artifacts = {"main": _save_figure(fig, directory, f"fig5_aeri_{tag}", font)}
    plt.close(fig)
    if data["paired_points"]:
        paired_metrics = ("baseline_difference_vs_official", "transductive_gain_difference_vs_official", "inductive_gain_difference_vs_official")
        titles = (["原始 R@1 相对官方的配对差", "转导增益相对官方的配对差", "归纳增益相对官方的配对差"] if chinese else
                  ["Baseline R@1 difference vs official", "Transductive gain difference vs official", "Inductive gain difference vs official"])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        seed_colors = ("#353535", "#2070b4", "#db7c19")
        categories = list(PAIRED)
        category_labels = [labels[composition_index[name]] for name in categories]
        for axis, metric, title in zip(axes, paired_metrics, titles):
            for row in (point for point in data["paired_points"] if point["metric"] == metric):
                x = categories.index(row["composition"]) + jitter[row["seed"]] * 2
                # Draw endpoints directly: a percentile CI need not contain the point estimate.
                if row["ci_low"] is not None:
                    axis.vlines(x, row["ci_low"], row["ci_high"], color="#666666", linewidth=1.4)
                    axis.hlines([row["ci_low"], row["ci_high"]], x - .025, x + .025, color="#666666")
                axis.scatter(x, row["value"], marker=markers[seeds.index(row["seed"])],
                             color=seed_colors[seeds.index(row["seed"])], s=34,
                             label=f"seed {row['seed']}", zorder=3)
            axis.axhline(0, color="#888888", linestyle="--", linewidth=.8)
            axis.set_xticks(range(len(categories)), category_labels, rotation=20, ha="right")
            axis.set_title(title, fontsize=10)
            axis.set_ylabel("配对差 (百分点)" if chinese else "Paired difference (percentage points)")
        fig.legend(handles=[plt.Line2D([], [], color=seed_colors[index], marker=markers[index],
                                      linestyle="none", label=f"seed {seed}") for index, seed in enumerate(seeds)],
                   loc="upper center", bbox_to_anchor=(.5, -.025), ncol=3, fontsize=9)
        fig.tight_layout()
        artifacts["paired"] = _save_figure(fig, directory, f"fig5_aeri_paired_{tag}", font)
        plt.close(fig)
    return artifacts, chinese


def _directory(root, tag, inputs):
    root = Path(root).expanduser().resolve()
    if any(root == path.parent for path in inputs):
        raise ValueError("--out must not be an input JSON directory")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"fig5_aeri_{tag}"
    while True:
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = root / f"fig5_aeri_{tag}_{stamp}"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s7d-json", action="append", required=True, help="Explicit S7d AERI JSON; repeat exactly three times")
    parser.add_argument("--aggregate-json", required=True, help="S7_aggregate_seeds JSON with the same three AERI seeds")
    parser.add_argument("--out", required=True, help="Parent for a fresh output directory")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--font", default=None, help="Chinese font file; missing/unreadable font falls back to English")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
            raise ValueError("--tag accepts letters, digits, _, ., - only")
        input_paths = [Path(path).expanduser().resolve() for path in args.s7d_json + [args.aggregate_json]]
        data = prepare_data(args.s7d_json, args.aggregate_json)
        evidence = [_evidence(path) for path in input_paths]
        if args.font and Path(args.font).expanduser().is_file():
            evidence.append(_evidence(args.font))
        directory = _directory(args.out, args.tag, input_paths)
        for warning in data["warnings"]:
            print("WARNING:", warning, flush=True)
        artifacts, chinese = plot_data(data, directory, args.tag, args.font)
        csv_path = directory / f"fig5_aeri_data_{args.tag}.csv"
        fields = ("plot", "composition", "seed", "metric", "value", "ci_low", "ci_high", "hub_skew", "source_json", "source_field", "hub_source_field", "ci_source_field", "gallery_draws")
        with csv_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in data["points"]:
                writer.writerow({"plot": "gallery_composition", **row})
                if row["metric"] != "baseline_r1":
                    writer.writerow({"plot": "hub_scatter", **row})
            for row in data["paired_points"]:
                writer.writerow({"plot": "paired_difference", **row})
        artifacts["data_csv"] = str(csv_path)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
        data["provenance"] = {"argv": [str(Path(__file__).resolve())] + list(sys.argv[1:] if argv is None else argv),
                              "arguments": vars(args), "git_HEAD": head, "inputs": evidence,
                              "random_sampling": False, "training_seeds": data["training_seeds"],
                              "created_utc": datetime.now(timezone.utc).isoformat()}
        data["artifacts"] = artifacts
        data["chinese_font_used"] = chinese
        data["font_status"] = "Chinese labels with supplied font" if chinese else "WARN: English labels; Chinese font unavailable"
        path = directory / f"fig5_aeri_plot_{args.tag}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        with (directory / f"fig5_aeri_{args.tag}.log").open("x", encoding="utf-8") as handle:
            handle.write(data["font_status"] + "\n")
            handle.write("\n".join(data["warnings"]) + "\n")
            handle.write(json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n")
        print("wrote", path, flush=True)
        return data
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

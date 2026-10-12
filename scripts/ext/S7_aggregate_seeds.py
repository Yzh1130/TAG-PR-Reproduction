"""Aggregate explicit S7a seed outputs; shared S7 parameter/I/O helpers.

The helpers below do not implement or change any S7 statistical calculation.
Run discovery rejects ambiguous dataset/split selections rather than guessing
the latest training/evaluation run. Importing this module performs no I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_ROOT = "/root/autodl-tmp/TAG-PR/experiments_ext"
DATASETS = ("cuhkpedes", "icfgpedes", "aeri", "tbapr")


def common_parser(name, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", default=DEFAULT_ROOT, help="External run parent directory")
    parser.add_argument("--run-dir", action="append", default=[], metavar="[DATASET[:SPLIT]=]PATH",
                        help="Explicit run; repeat for datasets/splits, never choose latest")
    parser.add_argument("--tag", required=True, help="Safe tag included in every output filename")
    parser.add_argument("--out", required=True, help="Parent for a fresh, exclusive result directory")
    parser.set_defaults(script_name=name)
    return parser


def parse_common(parser, argv=None):
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("--tag must contain only letters, digits, _, ., -")
    args.recorded_argv = [parser.prog] + list(sys.argv[1:] if argv is None else argv)
    return args


def _run_meta(path):
    directory = Path(path).expanduser().resolve()
    with (directory / "meta.json").open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if meta.get("dataset") not in DATASETS:
        raise ValueError(f"{directory}: unknown external dataset {meta.get('dataset')!r}")
    if meta.get("epoch") != 10 or meta.get("split") not in ("test", "val"):
        raise ValueError(f"{directory}: expected epoch=10 and split=test/val")
    sim = np.load(directory / "sim_qg.npy", mmap_mode="r", allow_pickle=False)
    if sim.ndim != 2 or sim.dtype != np.float32:
        raise ValueError(f"{directory}: sim_qg must be a 2-D float32 matrix")
    if sim.shape != (len(meta["query_pid"]), len(meta["gallery_pid"])):
        raise ValueError(f"{directory}: score shape and query/gallery PID lists disagree")
    return str(directory), meta


def select_runs(args, required=None):
    selected = []
    if args.run_dir:
        for spec in args.run_dir:
            expected, path = (spec.split("=", 1) if "=" in spec else (None, spec))
            directory, meta = _run_meta(path)
            if expected:
                pieces = expected.split(":", 1)
                if pieces[0] != meta["dataset"] or (len(pieces) == 2 and pieces[1] != meta["split"]):
                    raise ValueError(f"--run-dir {spec!r}: dataset/split disagrees with meta.json")
            selected.append((directory, meta))
    else:
        root = Path(args.root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"--root does not exist: {root}")
        for directory in sorted(root.iterdir()):
            if not (directory / "meta.json").is_file():
                continue
            with (directory / "meta.json").open(encoding="utf-8") as stream:
                meta = json.load(stream)
            if meta.get("epoch") != 10 or meta.get("split") not in ("test", "val"):
                continue
            key = (meta.get("dataset"), meta.get("split"))
            if required is None and key[1] != "test":
                continue
            if required is not None and key not in required:
                continue
            selected.append(_run_meta(directory))
    if not selected:
        raise ValueError("no eligible run: supply --run-dir explicitly")
    by_key = {}
    for directory, meta in selected:
        key = (meta["dataset"], meta["split"])
        if required is None and key[1] != "test":
            raise ValueError(f"{directory}: this analysis requires a test run")
        if required is not None and key not in required:
            raise ValueError(f"{directory}: unexpected dataset/split {key}")
        if key in by_key:
            raise ValueError(f"ambiguous runs for {key}: {by_key[key][0]} and {directory}; "
                             "use one invocation/tag per seed and explicit --run-dir")
        by_key[key] = (directory, meta)
    if required is not None and set(by_key) != set(required):
        raise ValueError(f"missing required runs: {sorted(set(required) - set(by_key))}")
    return [by_key[key] for key in sorted(by_key)]


def result_directory(args):
    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{args.script_name}_{args.tag}"
    while True:
        try:
            path.mkdir(exist_ok=False)
            return str(path)
        except FileExistsError:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = root / f"{args.script_name}_{args.tag}_{stamp}"


def output_name(directory, stem, tag, suffix):
    return str(Path(directory) / f"{stem}_{tag}{suffix}")


def save_npy(path, array):
    with Path(path).open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def save_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def file_evidence(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256_16": digest.hexdigest()[:16], "bytes": path.stat().st_size}


def provenance(args, paths, seeds):
    paths = sorted({str(Path(path).expanduser().resolve()) for path in paths})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=Path(__file__).resolve().parents[2], text=True).strip()
    return {"argv": args.recorded_argv, "seeds": seeds, "git_HEAD": head,
            "inputs": [file_evidence(path) for path in paths], "tag": args.tag}


def run_inputs(directory, names):
    return [str(Path(directory) / name) for name in names]


def peak_rss_gb():
    # resource is unavailable on Windows: this I/O-only diagnostic never changes
    # statistical computations or supplies a fabricated memory measurement.
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def logged_rss(value):
    return float("nan") if value is None else value


def load_s7a_artifact(summary_path, directory, dataset, mode):
    with Path(summary_path).open(encoding="utf-8") as stream:
        summary = json.load(stream)
    entry = summary["datasets"][dataset]
    if Path(entry.get("run_dir", "")).resolve() != Path(directory).resolve():
        raise ValueError(f"S7a summary/run mismatch for {dataset}: explicit matching --s7a-json required")
    recorded = {item["path"]: item for item in summary["provenance"]["inputs"]}
    for name in ("meta.json", "sim_qg.npy"):
        evidence = file_evidence(Path(directory) / name)
        if evidence != recorded.get(evidence["path"]):
            raise ValueError(f"S7a input evidence changed or missing: {evidence['path']}")
    path = Path(entry["artifacts"][f"r_{mode}"])
    if not path.is_absolute():
        path = Path(summary_path).resolve().parent / path
    values = np.load(path, allow_pickle=False)
    if values.shape != (len(entry.get("gallery_pid", [])) if "gallery_pid" in entry else entry["ng"],):
        raise ValueError(f"S7a penalty shape mismatch: {path}")
    return values, str(path.resolve())


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def aggregate(paths):
    datasets = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            result = json.load(stream)
        for dataset, item in result["datasets"].items():
            if "seed" not in item:
                raise ValueError(f"{path}: missing recorded run seed for {dataset}")
            row = {"seed": item["seed"], "run": item["run"], "input": str(Path(path).resolve()),
                   "baseline": item["baseline"], "analysis_seeds": result["provenance"]["seeds"]}
            for mode in ("transductive", "inductive"):
                row[mode] = {"delta": item[mode]["delta"],
                             "identity_ci95": item[mode]["boot_identity"]["ci95"]}
            datasets.setdefault(dataset, []).append(row)
    output = {}
    for dataset, rows in sorted(datasets.items()):
        rows.sort(key=lambda row: row["seed"])
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError(f"{dataset}: duplicate seed inputs, cannot treat repeated runs as independent seeds")
        summary = {"baseline": {key: _stats([row["baseline"][key] for row in rows]) for key in ("r1", "mAP")}}
        for mode in ("transductive", "inductive"):
            deltas = [row[mode]["delta"]["r1"] for row in rows]
            summary[mode] = {"delta": {key: _stats([row[mode]["delta"][key] for row in rows]) for key in ("r1", "mAP")},
                             "sign_counts": {"positive": sum(value > 0 for value in deltas),
                                             "zero": sum(value == 0 for value in deltas),
                                             "negative": sum(value < 0 for value in deltas), "total": len(rows)},
                             "positive_fraction_text": f"{sum(value > 0 for value in deltas)}/{len(rows)} positive"}
        output[dataset] = {"per_seed": rows, "summary": summary}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s7a-json", "--input", dest="inputs", action="append", required=True,
                        help="Explicit S7a summary JSON; repeat for seed 0/1/2")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True, help="Parent for an exclusive result directory")
    parser.set_defaults(script_name="S7_aggregate_seeds")
    args = parse_common(parser, argv)
    try:
        datasets = aggregate(args.inputs)
        seeds = {"new_random_sampling": False,
                 "run_seeds": {dataset: [row["seed"] for row in entry["per_seed"]]
                               for dataset, entry in datasets.items()},
                 "input_analysis_seeds": [{"input": row["input"], "seeds": row["analysis_seeds"]}
                                          for entry in datasets.values() for row in entry["per_seed"]]}
        result = {"datasets": datasets, "standard_deviation": "population, ddof=0",
                  "provenance": provenance(args, args.inputs, seeds)}
        directory = result_directory(args)
        path = output_name(directory, "S7_aggregate_seeds", args.tag, ".json")
        write_json(path, result)
        for dataset, entry in result["datasets"].items():
            for row in entry["per_seed"]:
                print(dataset, "seed", row["seed"], "baseline", row["baseline"],
                      "transductive", row["transductive"], "inductive", row["inductive"])
            print(dataset, "seed summaries", entry["summary"])
        print("wrote", path)
        return result
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

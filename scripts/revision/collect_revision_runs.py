"""Read-only experiment collection and descriptive, seed-paired comparisons.

No checkpoint is deserialized. A filename, current repository default, or a
configured epoch/iteration budget is never substituted for missing run evidence.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


METRICS = ("R1", "R5", "R10", "mAP")
RUN_MARKERS = {
    "config_used.yaml", "run_command.txt", "git_commit.txt", "train.log",
    "train_log.txt", "metrics_by_epoch.csv", "final_results.txt",
}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ITERATION = re.compile(r"Epoch\[(\d+)\]\s+Iteration\[(\d+)/(\d+)\]")
DONE = re.compile(r"\bEpoch\s+(\d+)\s+done\.")
RAW_METRICS = re.compile(
    rf"Acc@1\s+({NUMBER})\s+Acc@5\s+({NUMBER})\s+"
    rf"Acc@10\s+({NUMBER})\s+mAP\s+({NUMBER})")
CONFIG_KEYS = {
    "seed": ("misc", "seed"), "batch_size": ("data", "batch_size"),
    "itc_block_size": ("experiment", "itc_block_size"),
    "grad_checkpoint": ("experiment", "grad_checkpoint"),
    "eval_router_mode": ("experiment", "eval_router_mode"),
    "eval_seed": ("experiment", "eval_seed"),
}


def file_evidence(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing; collect only completed, stable runs: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size,
            "sha256_16": digest.hexdigest()[:16]}


def git_head() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _literal(value: str) -> Any:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        return parsed if str(parsed) == str(value).strip() else None
    except (ValueError, TypeError):
        return None


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (ValueError, TypeError):
        return None


def _metric_dict(values: Any) -> dict[str, float | None]:
    aliases = {"R1": ("R1", "r1", "R@1"), "R5": ("R5", "r5", "R@5"),
               "R10": ("R10", "r10", "R@10"), "mAP": ("mAP", "map")}
    return {key: next((_float(values[name]) for name in names if name in values), None)
            for key, names in aliases.items()}


def _kv(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def _command_evidence(text: str, warnings: list[str]) -> dict[str, Any]:
    values = _kv(text)
    argv = _literal(values["sys.argv"]) if "sys.argv" in values else None
    if argv is not None and (not isinstance(argv, list) or
                             not all(isinstance(x, str) for x in argv)):
        warnings.append("run_command sys.argv is not a string list")
        argv = None
    args = _literal(values["args"]) if "args" in values else None
    if not isinstance(args, dict):
        args = None
    # The POSIX shell string is kept verbatim; sys.argv remains authoritative.
    if argv is None and values.get("command") and not values.get("cwd", "").startswith("C:\\"):
        try:
            argv = shlex.split(values["command"], posix=True)
        except ValueError as exc:
            warnings.append(f"command tokenization unavailable: {exc}")
    return {"command": values.get("command"), "argv": argv, "args": args,
            "cwd": values.get("cwd"), "config_source_path": values.get("config_source"),
            "config_source_sha256_16": (values.get("config_source_sha256_16") or
                                        values.get("config_source_sha256", "")[:16] or None)}


def _argv_options(argv: list[str] | None) -> dict[str, Any]:
    if argv is None:
        return {}
    result = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            key, separator, value = token[2:].partition("=")
            if separator:
                result[key] = _literal(value)
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                index += 1
                result[key] = _literal(argv[index])
            else:
                result[key] = True
        index += 1
    return result


def parse_log(text: str) -> dict[str, Any]:
    """Parse sampled progress without calling max observed i a completed epoch."""
    epochs: dict[int, dict[str, Any]] = {}
    completed = set()
    metrics: dict[int, list[dict[str, float]]] = {}
    effective = {}
    current_epoch = None
    best_line = None
    args = None
    for line_number, line in enumerate(text.splitlines(), 1):
        match = ITERATION.search(line)
        if match:
            epoch, index, denominator = map(int, match.groups())
            current_epoch = epoch
            item = epochs.setdefault(epoch, {"indices": set(), "denominators": set(),
                                             "lines": []})
            item["indices"].add(index)
            item["denominators"].add(denominator)
            item["lines"].append(line_number)
        match = DONE.search(line)
        if match:
            current_epoch = int(match.group(1))
            completed.add(current_epoch)
        match = RAW_METRICS.search(line)
        if match and not re.search(r"\b(?:CSLS|RERANK)\b", line[:match.start()]):
            if current_epoch is not None:
                metrics.setdefault(current_epoch, []).append(
                    dict(zip(METRICS, map(float, match.groups()))))
        match = re.search(r"(experiment\.\w+) effective:\s*(.*)$", line)
        if match:
            effective[match.group(1)] = _literal(match.group(2))
        match = re.search(r"DataLoader batch_size verified: expected=(\d+), actual=(\d+)", line)
        if match:
            effective["data.batch_size"] = int(match.group(2))
        match = re.search(rf"best Acc@1:\s*({NUMBER})\s+at epoch\s+(\d+)", line)
        if match:
            best_line = {"epoch": int(match.group(2)),
                         "metrics": {"R1": float(match.group(1)),
                                     "R5": None, "R10": None, "mAP": None}}
        match = re.search(r"\bNamespace\((.*)\)\s*$", line)
        if match:
            try:
                expression = ast.parse("f(" + match.group(1) + ")", mode="eval").body
                args = {keyword.arg: ast.literal_eval(keyword.value)
                        for keyword in expression.keywords if keyword.arg}
            except (SyntaxError, ValueError):
                pass
    return {"epochs": epochs, "completed": completed, "metrics": metrics,
            "effective": effective, "best_line": best_line, "args": args}


def _lookup(config: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, Any]:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _logged_configuration(text: str) -> dict[str, Any]:
    """main logs EasyDict as a Python dictionary on its own line."""
    for line in text.splitlines():
        candidate = re.sub(r"^.*?\b(?:INFO|DEBUG):\s*", "", line).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            parsed = _literal(candidate)
            if isinstance(parsed, dict) and any(key in parsed for key in ("misc", "data", "experiment")):
                return parsed
    return {}


def parse_run(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    warnings: list[str] = []
    missing: list[str] = []
    evidence = {}
    config = {}
    config_path = directory / "config_used.yaml"
    if config_path.is_file():
        evidence[config_path.name] = file_evidence(config_path)
        try:
            config = yaml.safe_load(_read(config_path)) or {}
            if not isinstance(config, dict):
                warnings.append("config_used.yaml is not a mapping")
                config = {}
        except yaml.YAMLError as exc:
            warnings.append(f"config_used.yaml cannot be parsed: {exc}")
    command_path = directory / "run_command.txt"
    command = _command_evidence(_read(command_path), warnings) if command_path.is_file() else {}
    if command_path.is_file():
        evidence[command_path.name] = file_evidence(command_path)
    logs = []
    for name in ("train.log", "train_log.txt"):
        if (directory / name).is_file():
            logs = [directory / name]
            break
    if not logs:
        logs = sorted(path for path in directory.glob("*.log") if path.is_file())
    log_text = "\n".join(_read(path) for path in logs)
    for path in logs:
        evidence[path.name] = file_evidence(path)
    log = parse_log(log_text)
    configuration_source = "config_used.yaml" if config else None
    if not config:
        config = _logged_configuration(log_text)
        if config:
            configuration_source = "training_log_literal_configuration"
    args = command.get("args") or log["args"]
    cli = _argv_options(command.get("argv"))
    if args:
        cli = {**args, **cli}
    result: dict[str, Any] = {
        "directory": str(directory), "directory_name": directory.name,
        "exp_name": cli.get("exp_name"), "command": command.get("command"),
        "argv": command.get("argv"), "args": args,
        "config_source_path": command.get("config_source_path") or cli.get("config"),
        "config_source_sha256_16": command.get("config_source_sha256_16"),
        "config_used_path": str(config_path) if config_path.is_file() else None,
        "config_used_sha256_16": evidence.get(config_path.name, {}).get("sha256_16"),
        "effective_configuration_source": configuration_source,
        "config_hash_note": "Only archived config_used bytes are hashed here; a current source config is not evidence of its training-time bytes.",
    }
    for name, keys in CONFIG_KEYS.items():
        present, value = _lookup(config, keys)
        if not present and ".".join(keys) in log["effective"]:
            present, value = True, log["effective"][".".join(keys)]
        if not present and name == "grad_checkpoint" and cli.get("no_grad_checkpoint"):
            present, value = True, False
        if not present and name in cli and cli[name] is not None:
            present, value = True, cli[name]
        result[name] = value
        if not present:
            missing.append(name)
    # No use of this checkout's HEAD as a substitute for the run's commit.
    training_commit = None
    git_path = directory / "git_commit.txt"
    if git_path.is_file():
        text = _read(git_path)
        match = re.search(r"git rev-parse HEAD:\s*\n\s*([0-9a-fA-F]{40})\b", text)
        if not match:
            match = re.search(r"^\s*([0-9a-fA-F]{40})\s*$", text, re.MULTILINE)
        training_commit = match.group(1).lower() if match else None
        evidence[git_path.name] = file_evidence(git_path)
    if training_commit is None:
        for key in ("git_commit", "git_HEAD", "training_git_commit"):
            present, value = _lookup(config, (key,))
            if present and re.fullmatch(r"[0-9a-fA-F]{40}", str(value)):
                training_commit = str(value).lower()
                break
    if training_commit is None:
        environment_path = directory / "environment_summary.txt"
        archived_text = log_text
        if environment_path.is_file():
            archived_text += "\n" + _read(environment_path)
            evidence[environment_path.name] = file_evidence(environment_path)
        match = re.search(r"\b(?:training_git_commit|git_HEAD|git_commit|git commit)\s*[:=]\s*([0-9a-fA-F]{40})\b", archived_text)
        training_commit = match.group(1).lower() if match else None
    result["training_git_commit"] = training_commit
    final_path = directory / "final_results.txt"
    final = _kv(_read(final_path)) if final_path.is_file() else {}
    if final_path.is_file():
        evidence[final_path.name] = file_evidence(final_path)
    csv_path = directory / "metrics_by_epoch.csv"
    csv_metrics: dict[int, list[dict[str, float | None]]] = {}
    if csv_path.is_file():
        evidence[csv_path.name] = file_evidence(csv_path)
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 2):
                epoch = _integer(row.get("epoch"))
                if epoch is None:
                    warnings.append(f"metrics_by_epoch.csv line {row_number}: invalid epoch")
                    continue
                if epoch < 1:
                    warnings.append(f"metrics_by_epoch.csv line {row_number}: epoch={epoch} is not a one-based training epoch; ignored for training completion")
                    continue
                csv_metrics.setdefault(epoch, []).append(_metric_dict(row))
    result["evaluation_only"] = cli.get("eval_only") is True
    if result["evaluation_only"]:
        if csv_metrics or log["completed"] or log["epochs"]:
            warnings.append("archived eval_only invocation with training-looking records; excluded from training epoch/last/best collection")
        csv_metrics = {}
        log["completed"] = set()
        log["epochs"] = {}
        log["metrics"] = {}
        final = {}
    completed = log["completed"] | set(csv_metrics)
    declared_completed = _integer(final.get("completed_epochs"))
    result["completed_epochs"] = (declared_completed if declared_completed is not None
                                   else len(completed) if completed else None)
    result["completed_epoch_numbers_observed"] = sorted(completed)
    result["run_status"] = final.get("run_status")
    epoch_numbers = set(log["epochs"]) | set(log["metrics"]) | set(csv_metrics) | completed
    epoch_rows = []
    for epoch in sorted(epoch_numbers):
        item = log["epochs"].get(epoch, {"indices": set(), "denominators": set(), "lines": []})
        denominators = sorted(item["denominators"])
        measured_n = denominators[0] if len(denominators) == 1 else None
        actual_n = measured_n if epoch in completed else None
        if len(denominators) > 1:
            warnings.append(f"epoch {epoch}: multiple logged Iteration denominators; actual_iterations unknown")
        if measured_n is not None and measured_n in item["indices"]:
            actual_n = measured_n
        rows = csv_metrics.get(epoch, log["metrics"].get(epoch, []))
        source = "metrics_by_epoch.csv" if epoch in csv_metrics else "training_log_5_decimal_display"
        unique = {json.dumps(row, sort_keys=True) for row in rows}
        values = rows[0] if len(unique) == 1 else dict.fromkeys(METRICS)
        if len(unique) > 1:
            warnings.append(f"epoch {epoch}: conflicting {source} metric records; not selecting a record")
        entry = {"epoch": epoch, "completed_evidence": epoch in completed,
                 "actual_iterations": actual_n, "measured_loader_iterations": measured_n,
                 "observed_iteration_indices": sorted(item["indices"]),
                 "observed_iteration_max": max(item["indices"]) if item["indices"] else None,
                 "log_denominators": denominators, "metrics": values,
                 "metrics_source": source if rows else None}
        epoch_rows.append(entry)
        if actual_n is None:
            missing.append(f"epochs.{epoch}.actual_iterations")
        for key in METRICS:
            if values.get(key) is None:
                missing.append(f"epochs.{epoch}.metrics.{key}")
    result["epochs"] = epoch_rows
    best_epoch = _integer(final.get("best_epoch"))
    if best_epoch is not None:
        best_metrics = _metric_dict(final)
        result["best"] = {"epoch": best_epoch, "metrics": best_metrics,
                          "source": "final_results.txt"}
    else:
        valid = [row for row in epoch_rows if row["metrics"].get("R1") is not None]
        if valid:
            # main updates best only for strictly greater R1: retain earliest tie.
            best = max(valid, key=lambda row: row["metrics"]["R1"])
            result["best"] = {"epoch": best["epoch"], "metrics": best["metrics"],
                              "source": best["metrics_source"]}
        else:
            result["best"] = log["best_line"]
    last_epoch = declared_completed if declared_completed and declared_completed > 0 else max(completed, default=None)
    last_row = next((row for row in epoch_rows if row["epoch"] == last_epoch), None)
    result["last"] = ({"epoch": last_epoch,
                       "metrics": last_row["metrics"] if last_row else dict.fromkeys(METRICS),
                       "source": last_row["metrics_source"] if last_row else None}
                      if last_epoch is not None else None)
    result["checkpoints"] = [dict(name=path.name, **file_evidence(path))
                              for path in sorted(directory.glob("checkpoint_*.pth")) if path.is_file()]
    for field in ("exp_name", "training_git_commit", "config_source_path",
                  "config_source_sha256_16", "config_used_path", "config_used_sha256_16",
                  "completed_epochs", "best", "last"):
        if result[field] is None:
            missing.append(field)
    if not result["command"] and not result["argv"] and not result["args"]:
        missing.append("command_or_args")
    for scope in ("best", "last"):
        if result[scope] is not None:
            for key in METRICS:
                if result[scope]["metrics"].get(key) is None:
                    missing.append(f"{scope}.metrics.{key}")
    if not result["checkpoints"]:
        missing.append("checkpoints")
    result.update(missing_fields=sorted(set(missing)), warnings=warnings, input_files=evidence)
    return result


def matches_prefix(run: dict[str, Any], prefix: str) -> bool:
    name = run["directory_name"]
    dated_name = re.sub(r"^\d{8}_", "", name)
    exp_name = run.get("exp_name")
    return any(value.startswith(prefix) for value in (name, dated_name,
                                                     exp_name if isinstance(exp_name, str) else ""))


def _included_before_checkpoint_hash(directory: Path, include: list[str]) -> bool:
    if not include or any(matches_prefix({"directory_name": directory.name}, prefix) for prefix in include):
        return True
    # Name filtering is not an exp_name inference. Read inexpensive saved
    # arguments as a second route, before opening potentially multi-GB weights.
    command_path = directory / "run_command.txt"
    command = _command_evidence(_read(command_path), []) if command_path.is_file() else {}
    args = command.get("args") or {}
    args.update(_argv_options(command.get("argv")))
    exp_name = args.get("exp_name")
    if exp_name is None:
        logs = [directory / name for name in ("train.log", "train_log.txt")
                if (directory / name).is_file()]
        if not logs:
            logs = sorted(directory.glob("*.log"))
        for path in logs:
            parsed_args = parse_log(_read(path))["args"]
            if parsed_args and parsed_args.get("exp_name") is not None:
                exp_name = parsed_args["exp_name"]
                break
    return any(matches_prefix({"directory_name": directory.name, "exp_name": exp_name}, prefix)
               for prefix in include)


def _training_log_present(directory: Path, filenames: Any) -> bool:
    for name in filenames:
        if name.endswith(".log"):
            text = _read(directory / name)
            if ITERATION.search(text) or DONE.search(text):
                return True
    return False


def discover_runs(roots: list[Path], include: list[str]) -> list[dict[str, Any]]:
    directories = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"--root is not a directory: {root}")
        for folder, _subdirectories, filenames in os.walk(root, followlinks=False):
            if RUN_MARKERS.intersection(filenames) or any(
                    name.startswith("checkpoint_") and name.endswith(".pth") for name in filenames):
                directories.add(Path(folder).resolve())
            elif _training_log_present(Path(folder), filenames):
                directories.add(Path(folder).resolve())
    return [parse_run(directory) for directory in sorted(directories)
            if _included_before_checkpoint_hash(directory, include)]


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "population_std": None, "min": None, "max": None,
                "range": None, "sign_counts": {"negative": 0, "zero": 0, "positive": 0}}
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {"n": len(values), "mean": mean, "population_std": math.sqrt(variance),
            "min": min(values), "max": max(values), "range": max(values) - min(values),
            "sign_counts": {"negative": sum(value < 0 for value in values),
                            "zero": sum(value == 0 for value in values),
                            "positive": sum(value > 0 for value in values)}}


def pair_runs(runs: list[dict[str, Any]], specification: str, metric: str) -> dict[str, Any]:
    parts = specification.split("|")
    if len(parts) != 2 or not all(parts):
        raise ValueError("--pair must be two nonempty prefixes separated by |")
    prefix_a, prefix_b = parts
    group_a = [run for run in runs if matches_prefix(run, prefix_a)]
    group_b = [run for run in runs if matches_prefix(run, prefix_b)]
    seeds = sorted({run["seed"] for run in group_a + group_b if run["seed"] is not None}, key=str)
    rows = []
    for seed in seeds:
        candidates_a = [run for run in group_a if run["seed"] == seed]
        candidates_b = [run for run in group_b if run["seed"] == seed]
        row = {"seed": seed, "A_directories": [run["directory"] for run in candidates_a],
               "B_directories": [run["directory"] for run in candidates_b],
               "status": "unpaired", "reason": None, "A": None, "B": None,
               "delta_B_minus_A": dict.fromkeys(METRICS)}
        if len(candidates_a) != 1 or len(candidates_b) != 1:
            row["reason"] = f"requires exactly one directory per side; A={len(candidates_a)}, B={len(candidates_b)}"
        elif candidates_a[0]["directory"] == candidates_b[0]["directory"]:
            row["reason"] = "the same directory matches both prefixes; not pairing it to itself"
        else:
            a, b = candidates_a[0].get(metric), candidates_b[0].get(metric)
            row["A"], row["B"] = a, b
            if a is None or b is None or any(a["metrics"].get(key) is None or
                                            b["metrics"].get(key) is None for key in METRICS):
                row["reason"] = f"incomplete {metric} metrics; not guessing missing values"
            else:
                row["status"] = "paired"
                row["delta_B_minus_A"] = {key: b["metrics"][key] - a["metrics"][key]
                                           for key in METRICS}
        rows.append(row)
    unknown_seed = [{"side": side, "directory": run["directory"], "reason": "seed missing"}
                    for side, group in (("A", group_a), ("B", group_b))
                    for run in group if run["seed"] is None]
    return {"specification": specification, "by": "seed", "metric": metric,
            "prefix_A": prefix_a, "prefix_B": prefix_b, "rows": rows,
            "unknown_seed_unpaired": unknown_seed,
            "summary_A": {key: describe([row["A"]["metrics"][key] for row in rows
                                         if row["status"] == "paired"]) for key in METRICS},
            "summary_B": {key: describe([row["B"]["metrics"][key] for row in rows
                                         if row["status"] == "paired"]) for key in METRICS},
            "summary": {key: describe([row["delta_B_minus_A"][key] for row in rows
                                       if row["status"] == "paired"]) for key in METRICS},
            "description": "Descriptive paired differences only; no statistical test or additive decomposition."}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True, help="read-only experiment root; repeatable")
    parser.add_argument("--include", action="append", default=[], help="directory/archived exp_name prefix; repeatable")
    parser.add_argument("--out", type=Path, required=True, help="new output directory, never a scanned run directory")
    parser.add_argument("--tag", required=True, help="output tag; collision gets UTC suffix")
    parser.add_argument("--pair", action="append", default=[], help="A_PREFIX|B_PREFIX; repeatable")
    parser.add_argument("--by", choices=("seed",), default="seed")
    parser.add_argument("--metric", choices=("last", "best"), default="last", help="one checkpoint metric scope per output table")
    return parser


def _csv_value(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
            raise ValueError("--tag must contain only letters, digits, underscore, dot, or hyphen")
        roots = [path.expanduser().resolve() for path in args.root]
        runs = discover_runs(roots, args.include)
        pairs = [pair_runs(runs, specification, args.metric) for specification in args.pair]
        output = args.out.expanduser().resolve()
        for run in runs:
            directory = Path(run["directory"])
            if output == directory or directory in output.parents:
                raise ValueError(f"--out would write into a scanned experiment directory: {directory}")
        for directory in (output, *output.parents):
            if directory.is_dir() and any(root == directory or root in directory.parents for root in roots):
                filenames = {path.name for path in directory.iterdir() if path.is_file()}
                if RUN_MARKERS.intersection(filenames) or any(
                        name.startswith("checkpoint_") and name.endswith(".pth") for name in filenames) or _training_log_present(directory, filenames):
                    raise ValueError(f"--out would write into an experiment directory (even if excluded): {directory}")
        output.mkdir(parents=True, exist_ok=True)
        tag = args.tag
        names = lambda current: [output / f"runs_{current}.json", output / f"runs_{current}.csv"] + (
            [output / f"pairs_{args.metric}_{current}.csv"] if pairs else [])
        if any(path.exists() for path in names(tag)):
            tag += "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        report = {"provenance": {"argv": [str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
                                 "roots": list(map(str, roots)),
                                 "scanned_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "git_HEAD": git_head(), "seeds": [],
                                 "read_only_scan": True},
                  "runs": runs, "paired_comparisons": pairs, "metric_scope": args.metric}
        json_path, csv_path, *pair_paths = names(tag)
        with json_path.open("x", encoding="utf-8", newline="") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        fields = list(runs[0]) if runs else ["directory", "exp_name", "seed"]
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: _csv_value(run[key]) for key in fields} for run in runs)
        if pair_paths:
            fields = ["pair", "metric", "seed", "status", "reason", "A_directories", "B_directories"]
            fields += [f"{side}_{key}" for key in METRICS for side in ("A", "B", "delta")]
            with pair_paths[0].open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for pair in pairs:
                    for row in pair["rows"]:
                        values = {key: _csv_value(row[key]) for key in fields if key in row}
                        values.update(pair=pair["specification"], metric=args.metric)
                        for key in METRICS:
                            values[f"A_{key}"] = row["A"]["metrics"].get(key) if row["A"] else None
                            values[f"B_{key}"] = row["B"]["metrics"].get(key) if row["B"] else None
                            values[f"delta_{key}"] = row["delta_B_minus_A"][key]
                        writer.writerow(values)
        print(f"Collected {len(runs)} runs; JSON: {json_path}; CSV: {csv_path}")
        for pair in pairs:
            print(f"PAIR {pair['specification']} metric={args.metric}: "
                  f"paired={sum(row['status'] == 'paired' for row in pair['rows'])}, "
                  f"unpaired={sum(row['status'] != 'paired' for row in pair['rows']) + len(pair['unknown_seed_unpaired'])}")
            print(json.dumps(pair["summary"], ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, csv.Error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

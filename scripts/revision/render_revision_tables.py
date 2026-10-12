"""Render archived Round-D tables, without evaluating models or recomputing CIs.

Only explicitly requested descriptive reductions of complete stored draws or
paired CSV values are permitted. Missing values stay n/a, with an explanation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.revision._revision_table_format import TableFormatter
from scripts.revision._revision_tables_collection import render_pairs, render_runs
from scripts.revision._revision_tables_ext import render_s13, render_table11, render_table12
from scripts.revision._revision_tables_tag import render_r3, render_table3, render_table5


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("table5", "Common-query, controlled-gallery and identity-CI Table 5"),
        ("table3", "Joint source/view enrichment Table 3"),
        ("table11", "External per-seed and population-summary Table 11"),
        ("table12", "Distractor-scaling Table 12"),
        ("pairs", "Collector seed-paired CSVs, merged without conflict resolution"),
        ("table-r3", "Descriptive fine-grained cause tables"),
        ("runs", "Collector checkpoint inventory"),
        ("s13", "S13 --summary-table four-row comparison by model"),
    ):
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--out", required=True, help="New analysis output directory, never an input archive directory")
        sub.add_argument("--decimals", type=int, default=2, help="Decimal places (default: 2)")
        if name in ("table5", "table3", "table-r3", "runs"):
            sub.add_argument("--json", required=True, help="Explicit archived JSON path")
        if name in ("table5", "table3", "table-r3"):
            sub.add_argument("--label", default=None, help="Optional model label (default: None)")
        if name == "table11":
            sub.add_argument("--aggregate", required=True)
        if name == "table12":
            sub.add_argument("--s7b", required=True, action="append", help="Explicit S7b JSON, repeatable")
        if name == "pairs":
            sub.add_argument("--csv", required=True, action="append", help="Explicit collector pair CSV, repeatable")
            sub.add_argument("--metric", choices=("R1", "R5", "R10", "mAP"), default="R1")
        if name == "runs":
            sub.add_argument("--include", action="append", default=[], help="Collector prefix filter, repeatable")
        if name == "s13":
            sub.add_argument("--summary", required=True, action="append", help="Explicit label=PATH, repeatable")
    return parser


def _path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"input file does not exist: {path}")
    return path


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _json(path):
    def invalid_constant(value):
        raise ValueError(f"{path}: non-standard non-finite JSON constant {value}")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, not {type(data).__name__}")
    return data


def _csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        needed = {"pair", "metric", "seed", "status"}
        if reader.fieldnames is None or not needed.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing collector CSV headers {sorted(needed - set(reader.fieldnames or []))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate CSV headers would silently overwrite recorded values")
        result = list(reader)
    if any(None in row for row in result):
        raise ValueError(f"{path}: CSV row has more values than headers")
    return result


def _head():
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False)
    if process.returncode:
        raise ValueError(f"cannot record script git HEAD: {process.stderr.strip()}")
    return process.stdout.strip()


def _prepare(args, fmt):
    if args.command == "table11":
        paths = [_path(args.aggregate)]
    elif args.command == "table12":
        paths = [_path(value) for value in args.s7b]
    elif args.command == "pairs":
        paths = [_path(value) for value in args.csv]
    elif args.command == "s13":
        labels, paths = [], []
        for value in args.summary:
            if "=" not in value:
                raise ValueError("--summary must be explicit label=PATH")
            label, value = value.split("=", 1)
            if not label.strip() or not value or label in labels:
                raise ValueError("--summary labels must be nonempty and unique, with a nonempty path")
            labels.append(label)
            paths.append(_path(value))
    else:
        paths = [_path(args.json)]
    if len(set(paths)) != len(paths):
        raise ValueError("the same input path was provided more than once")
    evidence = [{"path": str(path), "bytes": path.stat().st_size, "sha256_16": _sha(path)} for path in paths]
    if args.command == "pairs":
        body = render_pairs([(str(path), _csv(path)) for path in paths], fmt, args.metric)
    else:
        data = [_json(path) for path in paths]
        if args.command == "table5":
            body = render_table5(data[0], fmt, args.label)
        elif args.command == "table3":
            body = render_table3(data[0], fmt, args.label)
        elif args.command == "table-r3":
            body = render_r3(data[0], fmt, args.label)
        elif args.command == "table11":
            body = render_table11(data[0], fmt)
        elif args.command == "table12":
            body = render_table12([(str(path), value) for path, value in zip(paths, data)], fmt)
        elif args.command == "runs":
            body = render_runs(data[0], fmt, args.include)
        else:
            body = render_s13(list(zip(labels, data)), fmt)
    for path, record in zip(paths, evidence):
        if path.stat().st_size != record["bytes"] or _sha(path) != record["sha256_16"]:
            raise ValueError(f"input changed during rendering: {path}")
    return body, paths, evidence


def _write(args, body, paths, evidence, fmt, argv):
    root = Path(args.out).expanduser().resolve()
    if any(root == path.parent or path.parent in root.parents for path in paths):
        raise ValueError("--out must not be inside a read-only input archive directory")
    if "_archive_evidence" in {part.lower() for part in root.parts} or "_dump" in {part.lower() for part in root.parts}:
        raise ValueError("archive/dump directories are read-only; choose a new analysis output directory")
    head = _head()
    manifest_path = root / "render_manifest.json"
    combined = root / "all_tables.md"
    entries = []
    if root.exists():
        if manifest_path.exists():
            manifest = _json(manifest_path)
            if manifest.get("tool") != "render_revision_tables" or not isinstance(manifest.get("entries"), list):
                raise ValueError(f"unrecognized output manifest: {manifest_path}")
            entries = manifest["entries"]
            if not combined.is_file() or _sha(combined) != manifest.get("all_tables_sha256_16"):
                raise ValueError("existing all_tables.md does not match this renderer's manifest; not modifying it")
        elif combined.exists():
            raise ValueError("existing all_tables.md has no renderer manifest; not modifying it")
    prior = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("output manifest entries must be objects; not modifying this collection")
        name = entry.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("invalid table path in output manifest")
        path = root / name
        if not path.is_file() or _sha(path) != entry.get("sha256_16"):
            raise ValueError(f"existing generated table changed; not modifying collection: {path}")
        prior.append(path.read_text(encoding="utf-8"))
    if fmt.missing:
        body += "\nMissing / inapplicable fields (not filled):\n\n" + "\n".join("- " + reason for reason in fmt.missing) + "\n"
    inputs = "; ".join(f"{record['path']} (sha256_16={record['sha256_16']})" for record in evidence)
    body += f"\nProvenance: inputs: {inputs}; script git HEAD={head}.\n"
    suffix = getattr(args, "label", None) or (args.metric if args.command == "pairs" else "")
    stem = args.command.replace("-", "_")
    if suffix:
        stem += "_" + re.sub(r"[^\w.-]+", "_", suffix, flags=re.UNICODE).strip("._")
    root.mkdir(parents=True, exist_ok=True)
    path = root / (stem + ".md")
    while True:
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
            break
        except FileExistsError:
            path = root / (stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".md")
    record = {"file": path.name, "sha256_16": _sha(path), "command": args.command,
              "argv": list(argv), "created_utc": datetime.now(timezone.utc).isoformat(),
              "git_HEAD": head, "inputs": evidence, "missing_fields": list(fmt.missing)}
    entries.append(record)
    # Update only this new analysis collection, after validating all its stored
    # hashes. Never modify an experiment directory or any original result JSON.
    combined.write_text("\n---\n\n".join(prior + [body]), encoding="utf-8", newline="\n")
    manifest_path.write_text(json.dumps({"tool": "render_revision_tables", "entries": entries,
                                        "all_tables_sha256_16": _sha(combined)}, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    print(f"updated {combined}")
    if fmt.missing:
        print(f"n/a reasons: {len(fmt.missing)} (listed in table and manifest)")
    return {"table": str(path), "all_tables": str(combined), "manifest": str(manifest_path),
            "provenance": record}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    actual_argv = list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv])
    try:
        fmt = TableFormatter(args.decimals)
        body, paths, evidence = _prepare(args, fmt)
        return _write(args, body, paths, evidence, fmt, actual_argv)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

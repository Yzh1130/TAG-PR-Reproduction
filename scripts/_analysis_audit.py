"""Exclusive, provenance-bearing outputs for the standalone S14/S15 tools.

Constructing an Audit never opens an input. In particular S15 can postpone all
test provenance reads until after its proxy-only parameter lock is on disk.
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import uuid
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _utc():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _ready(value):
    if isinstance(value, argparse.Namespace):
        return _ready(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _ready(value.tolist())
    if isinstance(value, np.generic):
        return _ready(value.item())
    if isinstance(value, dict):
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value cannot be written as audit JSON")
    return value


def git_head(repo=None):
    root = Path(repo) if repo else Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True, check=True)
    head = result.stdout.strip()
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise RuntimeError(f"invalid git HEAD in {root}: {head!r}")
    return head


def sha256_16(path):
    import hashlib
    source = Path(path)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {source}")
    return digest.hexdigest()[:16]


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def unique_path(directory, name):
    """Choose a new name; writers must still use exclusive creation (mode x)."""
    directory = Path(directory)
    name = Path(name)
    if name.name != str(name) or name.name in ("", ".", ".."):
        raise ValueError("output filename must be a single path component")
    candidate = directory / name
    while candidate.exists():
        candidate = directory / f"{name.stem}_{_utc()}_{uuid.uuid4().hex[:8]}{name.suffix}"
    return candidate


def _exact(first, second):
    first, second = _ready(first), _ready(second)
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_exact(first[k], second[k]) for k in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(_exact(a, b) for a, b in zip(first, second))
    if isinstance(first, float) and isinstance(second, float):
        return struct.pack("!d", first) == struct.pack("!d", second)
    return type(first) is type(second) and first == second


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()


class Audit:
    def __init__(self, tool, args, argv=None, seeds=None, repo=None):
        self.tool = tool
        self.out = Path(args.out).expanduser().resolve()
        expected = "_e1_s14" if str(tool).upper().startswith("S14") else "_e1_s15"
        if self.out.name != expected:
            raise ValueError(f"--out must name a dedicated {expected} directory: {self.out}")
        for part in self.out.parent.parts:
            if part.lower() in ("_archive_evidence", "_dump") or part.lower().startswith("_e1"):
                raise ValueError(f"refusing to place outputs inside existing evidence tree: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.report = dict(task=tool, argv=list(sys.argv if argv is None else argv),
                           arguments=_ready(args), seeds=_ready(seeds or {}),
                           git_HEAD=git_head(repo), inputs={}, checks=[], started_utc=_utc())
        self.log_path, self.log_handle = self._open_new(f"{tool}.log")
        self._finished = False
        self._capturing = False

    def _open_new(self, name):
        while True:
            path = unique_path(self.out, name)
            try:
                return path, path.open("x", encoding="utf-8", newline="\n")
            except FileExistsError:
                continue

    @contextmanager
    def capture(self):
        self._capturing = True
        try:
            with redirect_stdout(_Tee(sys.stdout, self.log_handle)), redirect_stderr(_Tee(sys.stderr, self.log_handle)):
                yield self
        finally:
            self._capturing = False
            if self._finished:
                self.log_handle.close()

    def record_input(self, path, role):
        source = Path(path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"{role}: input is not a file: {source}")
        info = dict(path=str(source), bytes=source.stat().st_size, sha256_16=sha256_16(source))
        previous = self.report["inputs"].get(role)
        if previous is not None and previous != info:
            raise RuntimeError(f"input role {role!r} changed during this run")
        self.report["inputs"][role] = info
        return info

    def read_json(self, path, role):
        self.record_input(path, role)
        return read_json(path)

    def check(self, name, actual, expected):
        passed = _exact(actual, expected)
        self.report["checks"].append(dict(name=name, passed=passed,
                                          actual=_ready(actual), expected=_ready(expected)))
        if not passed:
            raise RuntimeError(f"exact check FAILED: {name}: actual={actual!r}; expected={expected!r}")
        print(f"PASS exact: {name}")

    def write_json(self, name, payload):
        content = json.dumps(_ready(payload), ensure_ascii=False, indent=2, allow_nan=False)
        path, handle = self._open_new(name)
        with handle:
            handle.write(content + "\n")
        return path

    def new_run_dir(self):
        while True:
            path = self.out / f"{self.tool}_{_utc()}_{uuid.uuid4().hex[:8]}"
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue

    def finish(self, status="complete", error=None):
        if self._finished:
            raise RuntimeError("audit already finished")
        self.report.update(status=status, finished_utc=_utc(), log=str(self.log_path))
        if error is not None:
            self.report["error"] = str(error)
        report_path = self.write_json(f"{self.tool}.json", self.report)
        message = f"{self.tool} status={status}\nJSON: {report_path}\nLOG: {self.log_path}\n"
        print(message, end="")
        # finish can be called inside or outside capture; avoid duplicating it.
        if not isinstance(sys.stdout, _Tee) or sys.stdout.log is not self.log_handle:
            self.log_handle.write(message)
        self.log_handle.flush()
        self._finished = True
        if not self._capturing:
            self.log_handle.close()
        return report_path, self.log_path

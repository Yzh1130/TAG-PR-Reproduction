"""Immutable external-evaluation feature and score dump contract.

The external adapters deliberately retain both retrieval branches.  Scores are
materialised from the persisted feature files, then independently recomputed
from those same files and compared bit-for-bit before metadata is committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from misc.eval import compose_global_local_similarity


_PROTECTED_PARTS = {"_archive_evidence", "_dump"}
_CORE_FEATURES = (
    "feats_text_global.npy",
    "feats_text_local.npy",
    "feats_img_global.npy",
    "feats_img_local.npy",
)


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_16(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def text_sha256_16(texts: Iterable[str]):
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def gallery_identity_image_index(gallery_pid):
    """Map each remapped identity to its ordered global gallery row indices."""
    result = {}
    for gallery_index, pid in enumerate(gallery_pid):
        pid = int(pid)
        result.setdefault(pid, []).append(int(gallery_index))
    flattened = [index for indices in result.values() for index in indices]
    expected = list(range(len(gallery_pid)))
    if sorted(flattened) != expected:
        raise RuntimeError("gallery identity index does not conserve gallery rows")
    for pid, indices in result.items():
        if any(int(gallery_pid[index]) != pid for index in indices):
            raise RuntimeError("gallery identity index contains a mismatched pid")
    return result


def git_head(repository=None):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repository or Path.cwd()),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"cannot record git HEAD: {detail}")
    value = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError(f"invalid git HEAD returned by git: {value!r}")
    return value


def _reject_protected(path):
    parts = {part.lower() for part in Path(path).expanduser().resolve().parts}
    if parts & _PROTECTED_PARTS or any(part.startswith("_e1") for part in parts):
        raise ValueError(
            "external results may not be written under _archive_evidence, "
            "_dump, or an _e1* evidence directory"
        )


def _slug(value):
    value = re.sub(r"[^A-Za-z0-9.-]+", "-", str(value)).strip("-.")
    if not value:
        raise ValueError("empty output-directory name component")
    return value


def create_output_directory(
    output_root,
    dataset,
    backbone,
    split,
    seed,
    *,
    timestamp=None,
):
    """Create one non-overwriting ``experiments_ext`` run directory."""
    root = Path(output_root).expanduser().resolve()
    _reject_protected(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = "_".join((
        _slug(dataset),
        _slug(backbone),
        _slug(split),
        f"seed{int(seed)}",
        _slug(stamp),
    ))
    destination = root / name
    destination.mkdir(mode=0o755, exist_ok=False)
    return destination


def _exclusive_json(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        json.dump(_plain(payload), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _exclusive_npy(path, values):
    values = np.asarray(values)
    with Path(path).open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def validate_features(values, name, expected_rows=None):
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(f"{name}: expected nonempty 2-D features, got {values.shape}")
    if values.dtype != np.float32:
        raise ValueError(f"{name}: expected float32, got {values.dtype}")
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise ValueError(
            f"{name}: expected {expected_rows} rows, got {values.shape[0]}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: contains non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=2e-5):
        raise ValueError(
            f"{name}: rows are not L2 normalized; "
            f"norm range={norms.min()}..{norms.max()}"
        )
    return values


def _feature_paths(directory):
    directory = Path(directory)
    return {
        "text_global": directory / _CORE_FEATURES[0],
        "text_local": directory / _CORE_FEATURES[1],
        "image_global": directory / _CORE_FEATURES[2],
        "image_local": directory / _CORE_FEATURES[3],
    }


def write_core_features(
    directory,
    text_global,
    text_local,
    image_global,
    image_local,
):
    directory = Path(directory)
    arrays = {
        "text_global": validate_features(text_global, "text_global"),
        "text_local": validate_features(text_local, "text_local"),
        "image_global": validate_features(image_global, "image_global"),
        "image_local": validate_features(image_local, "image_local"),
    }
    nq = arrays["text_global"].shape[0]
    ng = arrays["image_global"].shape[0]
    validate_features(arrays["text_local"], "text_local", nq)
    validate_features(arrays["image_local"], "image_local", ng)
    if arrays["text_global"].shape[1] != arrays["image_global"].shape[1]:
        raise ValueError("global text/image dimensions differ")
    if arrays["text_local"].shape[1] != arrays["image_local"].shape[1]:
        raise ValueError("local text/image dimensions differ")
    if arrays["text_global"].shape[1] != arrays["text_local"].shape[1]:
        raise ValueError("external dump contract requires a common feature D")
    paths = _feature_paths(directory)
    for key, path in paths.items():
        _exclusive_npy(path, arrays[key])
    return paths


def _score_block(text_global, text_local, image_global, image_local, device):
    text_global = torch.from_numpy(np.asarray(text_global)).to(device)
    text_local = torch.from_numpy(np.asarray(text_local)).to(device)
    with torch.no_grad():
        result = compose_global_local_similarity(
            text_global,
            text_local,
            image_global,
            image_local,
        )
    return result.cpu().numpy().astype(np.float32, copy=False)


def write_and_verify_similarity(directory, block_size=128, device=None):
    """Write scores from saved features, then recompute every value exactly."""
    if block_size < 1:
        raise ValueError("score block size must be positive")
    directory = Path(directory)
    paths = _feature_paths(directory)
    arrays = {
        # Copy-on-write mappings are writable from NumPy/PyTorch's perspective
        # but can never alter the immutable feature files.
        name: np.load(path, mmap_mode="c", allow_pickle=False)
        for name, path in paths.items()
    }
    device = torch.device("cpu" if device is None else device)
    image_global = torch.from_numpy(np.asarray(arrays["image_global"])).to(device)
    image_local = torch.from_numpy(np.asarray(arrays["image_local"])).to(device)
    nq, ng = arrays["text_global"].shape[0], arrays["image_global"].shape[0]
    sim_path = directory / "sim_qg.npy"
    if sim_path.exists():
        raise FileExistsError(f"refusing to overwrite {sim_path}")
    scores = np.lib.format.open_memmap(
        sim_path,
        mode="w+",
        dtype=np.float32,
        shape=(nq, ng),
    )
    for start in range(0, nq, block_size):
        stop = min(start + block_size, nq)
        scores[start:stop] = _score_block(
            arrays["text_global"][start:stop],
            arrays["text_local"][start:stop],
            image_global,
            image_local,
            device,
        )
    scores.flush()
    del scores

    persisted = np.load(sim_path, mmap_mode="r", allow_pickle=False)
    compared = 0
    for start in range(0, nq, block_size):
        stop = min(start + block_size, nq)
        recomputed = _score_block(
            arrays["text_global"][start:stop],
            arrays["text_local"][start:stop],
            image_global,
            image_local,
            device,
        )
        if not np.array_equal(recomputed, persisted[start:stop]):
            mismatch = np.argwhere(recomputed != persisted[start:stop])[0]
            raise RuntimeError(
                "persisted sim_qg failed strict feature recomputation at "
                f"[{start + int(mismatch[0])}, {int(mismatch[1])}]"
            )
        compared += recomputed.size
    if compared != persisted.size:
        raise RuntimeError("strict score check did not visit every matrix element")
    return sim_path, {
        "array_equal": True,
        "compared_elements": int(compared),
        "shape": [int(nq), int(ng)],
    }


def evaluate_scores_and_write_output(
    sim_path,
    query_pid,
    gallery_pid,
    output_path,
    *,
    block_size=64,
    topk=50,
):
    """Evaluate external scores in bounded memory and write ranked rows."""
    scores = np.load(sim_path, mmap_mode="r", allow_pickle=False)
    query_pid = np.asarray(query_pid, dtype=np.int64)
    gallery_pid = np.asarray(gallery_pid, dtype=np.int64)
    if scores.shape != (len(query_pid), len(gallery_pid)):
        raise ValueError("score/label shape mismatch")
    if block_size < 1 or topk < 1:
        raise ValueError("evaluation block size and topk must be positive")
    hit1 = hit5 = hit10 = 0
    ap_total = 0.0
    rows = []
    keep = min(topk, len(gallery_pid))
    for start in range(0, len(query_pid), block_size):
        stop = min(start + block_size, len(query_pid))
        order = np.argsort(-np.asarray(scores[start:stop]), axis=1)
        matches = gallery_pid[order] == query_pid[start:stop, None]
        positives = matches.sum(axis=1)
        if np.any(positives == 0):
            bad = start + int(np.flatnonzero(positives == 0)[0])
            raise ValueError(f"query {bad} has no positive gallery image")
        hit1 += int(np.any(matches[:, :1], axis=1).sum())
        hit5 += int(np.any(matches[:, :5], axis=1).sum())
        hit10 += int(np.any(matches[:, :10], axis=1).sum())
        ranks = np.arange(1, matches.shape[1] + 1, dtype=np.float64)
        precision = np.cumsum(matches, axis=1) / ranks
        ap_total += float((precision * matches).sum(axis=1).dot(1.0 / positives))
        for offset in range(stop - start):
            chosen = order[offset, :keep]
            rows.append({
                str(start + offset): {
                    "pred": chosen.astype(int).tolist(),
                    "rank": list(range(1, keep + 1)),
                    "match": matches[offset, :keep].astype(int).tolist(),
                }
            })
    _exclusive_json(output_path, rows)
    count = len(query_pid)
    return {
        "r1": 100.0 * hit1 / count,
        "r5": 100.0 * hit5 / count,
        "r10": 100.0 * hit10 / count,
        "mAP": 100.0 * ap_total / count,
    }


def write_optional_features(
    directory,
    *,
    bank_global=None,
    bank_local=None,
    extra_global=None,
    extra_local=None,
    extra_meta=None,
):
    directory = Path(directory)
    written = {}
    if (bank_global is None) != (bank_local is None):
        raise ValueError("bank global/local features must be provided together")
    if bank_global is not None:
        bank_global = validate_features(bank_global, "bank_text_global")
        bank_local = validate_features(
            bank_local, "bank_text_local", bank_global.shape[0]
        )
        for name, values in (
            ("feats_bank_text_global.npy", bank_global),
            ("feats_bank_text_local.npy", bank_local),
        ):
            _exclusive_npy(directory / name, values)
            written[name] = list(values.shape)
    if (extra_global is None) != (extra_local is None):
        raise ValueError("extra image global/local features must be provided together")
    if extra_global is not None:
        extra_global = validate_features(extra_global, "extra_img_global")
        extra_local = validate_features(
            extra_local, "extra_img_local", extra_global.shape[0]
        )
        if extra_meta is None:
            raise ValueError("extra image features require extra_meta")
        for name, values in (
            ("feats_extra_img_global.npy", extra_global),
            ("feats_extra_img_local.npy", extra_local),
        ):
            _exclusive_npy(directory / name, values)
            written[name] = list(values.shape)
        _exclusive_json(directory / "extra_meta.json", extra_meta)
        written["extra_meta.json"] = True
    return written


def write_pid_remap(directory, train_remap, evaluation_remap):
    path = Path(directory) / "pid_remap_table.json"
    _exclusive_json(path, {
        "train_original_to_contiguous": {
            str(key): int(value) for key, value in train_remap.items()
        },
        "evaluation_original_to_contiguous": {
            str(key): int(value) for key, value in evaluation_remap.items()
        },
    })
    return path


def write_meta(directory, payload):
    path = Path(directory) / "meta.json"
    _exclusive_json(path, payload)
    return path


def annotation_provenance(paths):
    return [
        {
            "path": str(Path(path).expanduser().resolve()),
            "sha256_16": sha256_16(path),
        }
        for path in paths
    ]


__all__ = [
    "annotation_provenance",
    "create_output_directory",
    "evaluate_scores_and_write_output",
    "gallery_identity_image_index",
    "git_head",
    "sha256_16",
    "text_sha256_16",
    "validate_features",
    "write_and_verify_similarity",
    "write_core_features",
    "write_meta",
    "write_optional_features",
    "write_pid_remap",
]

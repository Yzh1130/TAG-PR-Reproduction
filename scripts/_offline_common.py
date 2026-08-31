"""Read-only NumPy primitives shared by S11/S12; no model imports.

Annotation order is preserved. No expected retrieval score or dataset size is
hardcoded. Exact ties use ascending original gallery index, explicitly unlike
the unspecified tie ordering of a GPU sort.
"""

import json
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class Annotation:
    paths: list
    gallery_ids: np.ndarray
    gallery_views: np.ndarray
    gallery_domains: np.ndarray
    query_source_indices: np.ndarray
    query_ids: np.ndarray
    query_views: np.ndarray
    query_domains: np.ndarray
    dual_ids: np.ndarray
    summary: dict


def annotation_from_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("annotation must be a nonempty list of image records")
    aliases = {"agreidv2": "AG-ReID.v2", "g2aps": "G2APS", "market": "Market",
               "market1501": "Market", "prai1581": "PRAI-1581", "uavhuman": "uavhuman"}
    paths, ids, views, domains, sources = [], [], [], [], []
    seen, id_domains = set(), {}
    for i, row in enumerate(records):
        if not isinstance(row, dict) or not {"file_path", "id", "captions"} <= row.keys():
            raise ValueError(f"record {i}: need file_path, id, captions")
        if not isinstance(row["file_path"], str) or not row["file_path"].strip():
            raise ValueError(f"record {i}: invalid file_path")
        path = row["file_path"].replace("\\", "/").removeprefix("./")
        if path in seen:
            raise ValueError(f"duplicate gallery image: {path}")
        seen.add(path)
        key = re.sub(r"[^a-z0-9]", "", path.split("/")[0].lower())
        if key not in aliases:
            raise ValueError(f"unknown source dataset in {path!r}")
        domain = aliases[key]
        inferred = set()
        if domain == "Market":
            inferred.add(1)
        elif domain in ("PRAI-1581", "uavhuman"):
            inferred.add(0)
        elif domain == "G2APS":
            match = re.match(r"([DS])\d+_", path.rsplit("/", 1)[-1], re.I)
            if match:
                inferred.add(0 if match[1].upper() == "D" else 1)
        else:
            match = re.search(r"C(0|3)F\d+", path.rsplit("/", 1)[-1], re.I)
            if match:
                inferred.add(0 if match[1] == "0" else 1)
        for part in path.lower().split("/")[1:-1]:
            if part in ("aerial", "air", "uav", "drone"):
                inferred.add(0)
            if part in ("ground", "terrestrial"):
                inferred.add(1)
        if "cam_id" in row:
            cam = row["cam_id"]
            if isinstance(cam, bool) or cam not in (0, 1):
                raise ValueError(f"record {i}: cam_id must be 0 or 1")
            inferred.add(int(cam))
        if len(inferred) != 1:
            raise ValueError(f"record {i}: missing or conflicting view for {path!r}")
        pid, captions = row["id"], row["captions"]
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise ValueError(f"record {i}: id must be an integer")
        if not isinstance(captions, list) or not captions or not all(isinstance(c, str) for c in captions):
            raise ValueError(f"record {i}: captions must be a nonempty list of strings")
        if pid in id_domains and id_domains[pid] != domain:
            raise ValueError(f"identity {pid} occurs in multiple source datasets")
        id_domains[pid] = domain
        paths.append(path)
        ids.append(pid)
        views.append(inferred.pop())
        domains.append(domain)
        sources.extend([i] * len(captions))
    ids, views = np.asarray(ids, dtype=np.int64), np.asarray(views, dtype=np.int8)
    domains, sources = np.asarray(domains), np.asarray(sources, dtype=np.int64)
    dual = np.intersect1d(ids[views == 0], ids[views == 1])
    summary = dict(n_query=len(sources), n_gallery=len(ids), aerial=int((views == 0).sum()),
                   ground=int((views == 1).sum()), identities=len(np.unique(ids)),
                   dual_view_identities=len(dual),
                   domains={str(d): int((domains == d).sum()) for d in np.unique(domains)},
                   query_order="annotation image order, then caption order; never sorted")
    return Annotation(paths, ids, views, domains, sources, ids[sources], views[sources],
                      domains[sources], dual, summary)


def load_annotations(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return annotation_from_records(json.load(handle))


def load_matrix(path, shape=None):
    try:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"cannot read NumPy matrix {path}: {exc}") from exc
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError(f"{path}: expected a two-dimensional .npy array")
    if shape is not None and matrix.shape != tuple(shape):
        raise ValueError(f"{path}: expected shape {tuple(shape)}, got {matrix.shape}")
    if not all(matrix.shape) or matrix.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError(f"{path}: need nonempty float16/float32 matrix, got {matrix.shape}/{matrix.dtype}")
    for start in range(0, len(matrix), 256):
        if not np.isfinite(matrix[start:start + 256]).all():
            raise ValueError(f"{path}: NaN/infinity near row {start}")
    return matrix


def load_similarity(path, ann):
    return load_matrix(path, (len(ann.query_ids), len(ann.gallery_ids)))


def indices(values, size):
    result = np.arange(size) if values is None else np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if len(result) and (result.min() < 0 or result.max() >= size):
        raise ValueError("index out of range")
    if len(np.unique(result)) != len(result):
        raise ValueError("duplicate indices are not permitted")
    return result.astype(np.int64, copy=False)


def eligible_queries(ann, queries, gallery, exclude_source=True):
    queries = indices(queries, len(ann.query_ids))
    gallery = indices(gallery, len(ann.gallery_ids))
    pids, counts = np.unique(ann.gallery_ids[gallery], return_counts=True)
    lookup = dict(zip(pids.tolist(), counts.tolist()))
    n = np.asarray([lookup.get(int(pid), 0) for pid in ann.query_ids[queries]])
    if exclude_source:
        n -= np.isin(ann.query_source_indices[queries], gallery)
    return queries[n > 0], int((n == 0).sum())


def topk_values(sim, k, rows=None, block_size=256):
    """Sorted column top-k, exact and independent of calibration row order."""
    rows = indices(rows, sim.shape[0])
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or not 1 <= k <= len(rows):
        raise ValueError(f"k must lie in [1, calibration rows={len(rows)}], got {k}")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    best = np.empty((0, sim.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), block_size):
        block = np.asarray(sim[rows[start:start + block_size]], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("top-k input contains NaN/infinity")
        candidates = np.concatenate((best, block), axis=0)
        n = min(k, len(candidates))
        best = np.partition(candidates, len(candidates) - n, axis=0)[-n:].copy()
    return np.sort(best, axis=0)[::-1].copy()


def topk_mean(sim, k, rows=None):
    return topk_values(sim, k, rows).mean(axis=0, dtype=np.float32)


def evaluate(sim, ann, queries=None, gallery=None, exclude_source=False,
             penalty=None, alpha=1.0, apply_mask=None, block_size=64):
    """Exact ID retrieval metrics plus per-query outcomes (percent units).

    Raw/CSLS scores are float32. A float64 IS penalty retains float64 arithmetic;
    correction is s - alpha * penalty (no unstable exponential ranking).
    Source-image exclusion occurs BEFORE sorting and removes only that image.
    Sorting is blockwise, never allocating a full query-by-gallery rank array.
    """
    if sim.shape != (len(ann.query_ids), len(ann.gallery_ids)) or block_size < 1:
        raise ValueError("similarity shape/annotation mismatch or invalid block size")
    queries = indices(queries, len(ann.query_ids))
    gallery = np.sort(indices(gallery, len(ann.gallery_ids)))
    if not len(gallery):
        raise ValueError("gallery is empty")
    n_input = len(queries)
    queries, skipped = eligible_queries(ann, queries, gallery, exclude_source)
    if penalty is not None:
        penalty = np.asarray(penalty)
        if penalty.shape != (len(ann.gallery_ids),) or not np.isfinite(penalty).all():
            raise ValueError("penalty must be a finite vector over the FULL gallery")
        if penalty.dtype not in (np.dtype("float32"), np.dtype("float64")):
            penalty = penalty.astype(np.float32)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if apply_mask is not None:
        apply_mask = np.asarray(apply_mask)
        if apply_mask.shape != (len(ann.query_ids),) or apply_mask.dtype != bool:
            raise ValueError("apply_mask must be a boolean vector over all queries")
    n = len(queries)
    top1, correct, aps = np.empty(n, np.int64), np.empty(n, bool), np.empty(n, np.float64)
    hits = np.zeros((n, 3), dtype=bool)
    sizes = len(gallery) - (np.isin(ann.query_source_indices[queries], gallery) if exclude_source
                            else np.zeros(n, bool))
    ties = 0
    for start in range(0, n, block_size):
        q = queries[start:start + block_size]
        dtype = np.result_type(sim.dtype, np.float32, penalty.dtype if penalty is not None else np.float32)
        values = np.array(sim[np.ix_(q, gallery)], dtype=dtype, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("evaluation input contains NaN/infinity")
        if penalty is not None:
            p = dtype.type(alpha) * penalty[gallery]
            values -= p[None, :] if apply_mask is None else apply_mask[q, None] * p[None, :]
        allowed = (ann.query_source_indices[q, None] != gallery[None, :] if exclude_source
                   else np.ones(values.shape, dtype=bool))
        values[~allowed] = -np.inf
        pos = (ann.query_ids[q, None] == ann.gallery_ids[gallery][None, :]) & allowed
        ties += int(((values == values.max(1, keepdims=True)).sum(1) > 1).sum())
        end = start + len(q)
        first = values.argmax(axis=1)
        top1[start:end] = gallery[first]
        correct[start:end] = pos[np.arange(len(q)), first]
        if pos.sum() <= 16 * len(q):
            # ReID has few positives. Count the exact rank of each positive
            # instead of sorting every negative for every hyperparameter. A
            # tied earlier gallery index also precedes that positive. This is
            # algebraically identical to stable sorting, not approximate top-k.
            rp, cp = np.nonzero(pos)
            ranks = np.empty(len(rp), dtype=np.int64)
            for ps in range(0, len(rp), 64):
                r, c = rp[ps:ps + 64], cp[ps:ps + 64]
                candidate = values[r]
                score = values[r, c, None]
                ranks[ps:ps + len(r)] = 1 + (candidate > score).sum(1) + (
                    (candidate == score) & (np.arange(len(gallery))[None, :] < c[:, None])).sum(1)
            for row in range(len(q)):
                ordered = np.sort(ranks[rp == row])
                hits[start + row] = ordered[0] <= np.asarray([1, 5, 10])
                aps[start + row] = np.mean(np.arange(1, len(ordered) + 1) / ordered)
        else:
            order = np.argsort(-values, axis=1, kind="stable")
            matches = np.take_along_axis(pos, order, axis=1)
            for j, k in enumerate((1, 5, 10)):
                hits[start:end, j] = matches[:, :k].any(axis=1)
            precision = matches.cumsum(1, dtype=np.int64) / np.arange(1, len(gallery) + 1)
            aps[start:end] = (precision * matches).sum(1) / pos.sum(1)
    result = dict(n_query=n, n_query_input=n_input, skipped_no_positive=skipped,
                  gallery_size=len(gallery), gallery_size_min=int(sizes.min()) if n else 0,
                  gallery_size_max=int(sizes.max()) if n else 0, exclude_source=exclude_source,
                  top1_tied_queries=ties, tie_policy="score descending, gallery index ascending")
    # Online R@k uses float32 percentages. mAP deliberately accumulates in float64.
    result.update({key: float(np.float32(hits[:, i].sum()) * np.float32(100) / np.float32(n))
                   if n else None for i, key in enumerate(("R1", "R5", "R10"))})
    result["mAP"] = float(aps.mean() * 100) if n else None
    result["details"] = dict(query_indices=queries, top1=top1, correct=correct, ap=aps)
    return result


def compact_metrics(result):
    return {key: value for key, value in result.items() if key != "details"}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite number in output JSON")
    return value


def safe_output_dir(path):
    path = Path(path).expanduser().resolve()
    if "_archive_evidence" in {part.lower() for part in path.parts}:
        raise ValueError("_archive_evidence is read-only; choose a new output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_stem(out, tag):
    out = safe_output_dir(out)
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(tag)).strip(".") or "analysis"
    return out / f"{safe_tag}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}"


def write_report(out, tag, report, logtext):
    stem = report_stem(out, tag)
    json_path, log_path = Path(str(stem) + ".json"), Path(str(stem) + ".log")
    with json_path.open("x", encoding="utf-8") as handle:
        json.dump(json_ready(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write(logtext)
    return json_path, log_path


@contextmanager
def tee_log(path):
    """Mirror stdout to a new log; flush each write, including early failures."""
    path = Path(path)
    safe_output_dir(path.parent)
    original = sys.stdout
    with path.open("x", encoding="utf-8") as handle:
        class Tee:
            def write(self, text):
                original.write(text)
                handle.write(text)
                handle.flush()
            def flush(self):
                original.flush()
                handle.flush()
        sys.stdout = Tee()
        try:
            yield
        finally:
            sys.stdout = original


def print_table(headers, rows):
    rows = [[str(v) for v in row] for row in rows]
    widths = [max(len(str(h)), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))


def provenance(sim_path):
    path = Path(sim_path).resolve()
    info = dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    meta = path.parent / "sim_meta.json"
    if meta.is_file():
        with meta.open(encoding="utf-8-sig") as handle:
            info["sim_meta"] = json.load(handle)
    return info

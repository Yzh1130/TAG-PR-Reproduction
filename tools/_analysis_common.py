"""NumPy-only, read-only TAG-PEDES annotation and retrieval utilities.

No training/evaluation modules are imported. Normal command-line entry points
enforce the published TAG-PEDES anchors; only synthetic self-tests relax them.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DOMAIN_NAMES = ("AG-ReID.v2", "G2APS", "Market", "PRAI-1581", "uavhuman")
ANCHORS = dict(n_query=16504, n_gallery=8252, aerial=4498, ground=3754,
               identities=2000, dual_view_identities=800)
_DOMAIN_ALIASES = {
    "agreidv2": "AG-ReID.v2", "g2aps": "G2APS", "market": "Market",
    "market1501": "Market", "prai1581": "PRAI-1581", "uavhuman": "uavhuman",
}


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


def _domain_name(value):
    key = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if key not in _DOMAIN_ALIASES:
        raise ValueError(f"unknown source dataset {value!r}; expected one of {DOMAIN_NAMES}")
    return _DOMAIN_ALIASES[key]


def _path_view(path, domain):
    """Conservative rules cross-checked against this repository's annotations."""
    if domain == "Market":
        return 1
    if domain in ("PRAI-1581", "uavhuman"):
        return 0
    basename = path.rsplit("/", 1)[-1]
    candidates = set()
    if domain == "G2APS":
        match = re.match(r"([DS])\d+_", basename, flags=re.IGNORECASE)
        if match:
            candidates.add(0 if match[1].upper() == "D" else 1)
    elif domain == "AG-ReID.v2":
        match = re.search(r"C(\d+)F\d+", basename, flags=re.IGNORECASE)
        if match and match[1] in ("0", "3"):
            candidates.add(0 if match[1] == "0" else 1)
    for component in path.lower().split("/")[1:-1]:
        if component in ("aerial", "air", "drone", "uav"):
            candidates.add(0)
        elif component in ("ground", "terrestrial"):
            candidates.add(1)
    if len(candidates) > 1:
        raise ValueError(f"conflicting camera rules in path {path!r}")
    return next(iter(candidates)) if candidates else None


def _view_value(value):
    if isinstance(value, str):
        value = {"aerial": 0, "air": 0, "uav": 0, "drone": 0,
                 "ground": 1, "terrestrial": 1, "0": 0, "1": 1}.get(value.lower(), value)
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError(f"view must be 0/aerial or 1/ground, got {value!r}")
    return int(value)


def annotation_from_records(records, enforce_anchors=True):
    if not isinstance(records, list) or not records:
        raise ValueError("annotation must be a nonempty JSON list of gallery records")
    paths, ids, views, domains, query_sources = [], [], [], [], []
    explicit_count = 0
    seen_paths, id_domains, id_views = set(), {}, {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"annotation record {index} is not an object")
        for key in ("file_path", "id", "captions"):
            if key not in record:
                raise ValueError(f"annotation record {index} missing {key!r}")
        path = record["file_path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"annotation record {index} has an invalid file_path")
        path = path.replace("\\", "/").removeprefix("./")
        if path in seen_paths:
            raise ValueError(f"duplicate gallery image path: {path}")
        seen_paths.add(path)
        domain = _domain_name(path.split("/")[0])
        for field in ("source", "dataset", "domain"):
            if field in record and _domain_name(record[field]) != domain:
                raise ValueError(f"record {index}: explicit {field} disagrees with path")
        derived_view = _path_view(path, domain)
        explicit_views = [_view_value(record[key]) for key in ("cam_id", "view", "view_id")
                          if key in record]
        if explicit_views:
            explicit_count += 1
            view = explicit_views[0]
            if any(item != view for item in explicit_views):
                raise ValueError(f"conflicting explicit camera fields for {path}")
            if derived_view is not None and derived_view != view:
                raise ValueError(f"cam_id disagrees with verified path rule for {path}")
        elif derived_view is not None:
            view = derived_view
        else:
            raise ValueError(f"cannot infer aerial/ground view for {path!r}; add explicit cam_id")
        person_id = record["id"]
        if isinstance(person_id, bool) or not isinstance(person_id, int):
            raise ValueError(f"identity id must be an integer at record {index}")
        if not np.iinfo(np.int64).min <= person_id <= np.iinfo(np.int64).max:
            raise ValueError(f"identity id outside int64 range at record {index}")
        if person_id in id_domains and id_domains[person_id] != domain:
            raise ValueError(f"identity {person_id} spans different source datasets")
        id_domains[person_id] = domain
        id_views.setdefault(person_id, set()).add(view)
        captions = record["captions"]
        if not isinstance(captions, list) or not captions or not all(isinstance(c, str) for c in captions):
            raise ValueError(f"captions must be a nonempty list of strings for {path}")
        paths.append(path)
        ids.append(person_id)
        views.append(view)
        domains.append(domain)
        query_sources.extend([index] * len(captions))
    ids = np.asarray(ids, dtype=np.int64)
    views = np.asarray(views, dtype=np.int8)
    domains = np.asarray(domains)
    sources = np.asarray(query_sources, dtype=np.int64)
    dual_ids = np.asarray(sorted(pid for pid, v in id_views.items() if len(v) == 2), dtype=np.int64)
    summary = dict(n_query=len(sources), n_gallery=len(ids), aerial=int((views == 0).sum()),
                   ground=int((views == 1).sum()), identities=len(id_views),
                   dual_view_identities=len(dual_ids), explicit_view_records=explicit_count,
                   inferred_view_records=len(ids) - explicit_count,
                   query_order="gallery-record order, then caption order within each record",
                   domains={name: int((domains == name).sum()) for name in DOMAIN_NAMES})
    if enforce_anchors:
        mismatches = [f"{key}: expected {expected}, got {summary[key]}"
                      for key, expected in ANCHORS.items() if summary[key] != expected]
        if mismatches:
            raise ValueError("TAG-PEDES anchor assertion failed: " + "; ".join(mismatches))
    return Annotation(paths, ids, views, domains, sources, ids[sources], views[sources],
                      domains[sources], dual_ids, summary)


def load_annotations(path, enforce_anchors=True):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return annotation_from_records(json.load(handle), enforce_anchors)


def load_similarity(path, ann):
    try:
        scores = np.load(path, mmap_mode="r", allow_pickle=False)
    except EOFError as exc:
        raise ValueError(f"empty or truncated NumPy matrix file: {path}") from exc
    expected = (len(ann.query_ids), len(ann.gallery_ids))
    if not isinstance(scores, np.ndarray) or scores.shape != expected:
        raise ValueError(f"sim_qg must have shape {expected}, got {getattr(scores, 'shape', None)}")
    if scores.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError(f"sim_qg dtype must be float16 or float32, got {scores.dtype}")
    for start in range(0, len(scores), 256):
        if not np.isfinite(scores[start:start + 256]).all():
            raise ValueError(f"sim_qg contains NaN or infinity near query {start}")
    return scores


def _indices(indices, size, label):
    result = np.arange(size) if indices is None else np.asarray(indices)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{label} indices must be a one-dimensional integer array")
    if len(result) and (result.min() < 0 or result.max() >= size):
        raise ValueError(f"{label} index out of range")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"duplicate {label} indices")
    return result.astype(np.int64, copy=False)


def _score_block(scores, queries, galleries):
    # Explicit float32 avoids half-precision arithmetic on float16 dumps.
    return np.asarray(scores[np.ix_(queries, galleries)], dtype=np.float32)


def column_topk_mean(scores, k, query_indices=None, gallery_indices=None,
                     source_indices=None, block_size=256):
    """Exact streaming column top-k; optionally remove text-source self pairs.

    If a column has fewer than k eligible queries, use its available count.
    No eligible query is an error, never an infinity-valued penalty.
    """
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("CSLS k must be a positive integer")
    queries = _indices(query_indices, scores.shape[0], "query")
    galleries = _indices(gallery_indices, scores.shape[1], "gallery")
    if len(queries) == 0 or len(galleries) == 0 or block_size < 1:
        raise ValueError("CSLS requires nonempty query/gallery sets and positive block_size")
    if k > len(queries):
        raise ValueError(f"CSLS k={k} exceeds calibration query count {len(queries)}")
    running = None
    for start in range(0, len(queries), block_size):
        q = queries[start:start + block_size]
        block = _score_block(scores, q, galleries)
        if source_indices is not None:
            block[np.asarray(source_indices)[q, None] == galleries[None, :]] = -np.inf
        candidates = block if running is None else np.concatenate((running, block), axis=0)
        count = min(k, candidates.shape[0])
        running = np.partition(candidates, candidates.shape[0] - count, axis=0)[-count:]
    # Sorting makes the mean's floating point summation independent of chunking
    # and query order, including full-pool calibration scans.
    running = np.sort(running, axis=0)[::-1]
    valid = np.isfinite(running)
    counts = valid.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("CSLS has a gallery column with no eligible calibration query")
    return (np.where(valid, running, 0).sum(axis=0, dtype=np.float32)
            / counts.astype(np.float32)).astype(np.float32)


def eligible_queries(ann, query_indices, gallery_indices, exclude_source):
    queries = _indices(query_indices, len(ann.query_ids), "query")
    galleries = _indices(gallery_indices, len(ann.gallery_ids), "gallery")
    unique_ids, counts = np.unique(ann.gallery_ids[galleries], return_counts=True)
    positive_counts = dict(zip(unique_ids.tolist(), counts.tolist()))
    counts = np.asarray([positive_counts.get(int(pid), 0) for pid in ann.query_ids[queries]])
    if exclude_source:
        counts -= np.isin(ann.query_source_indices[queries], galleries).astype(np.int64)
    return queries[counts > 0], int((counts == 0).sum())


def _online_percent(count, total):
    # Match misc.eval.acc_k's float32 multiply then divide (not Python float64).
    return float(np.float32(np.float32(count) * np.float32(100)) / np.float32(total))


def evaluate(scores, ann, query_indices=None, gallery_indices=None, exclude_source=False,
             penalties=None, block_size=128, r1_only=False):
    queries = _indices(query_indices, len(ann.query_ids), "query")
    galleries = _indices(gallery_indices, len(ann.gallery_ids), "gallery")
    if block_size < 1 or len(galleries) == 0:
        raise ValueError("positive block_size and nonempty gallery are required")
    original_count = len(queries)
    queries, removed = eligible_queries(ann, queries, galleries, exclude_source)
    if penalties is not None:
        penalties = np.asarray(penalties, dtype=np.float32)
        if penalties.shape != (len(galleries),) or not np.isfinite(penalties).all():
            raise ValueError("CSLS penalty must be a finite vector matching the selected gallery")
    sizes = len(galleries) - (np.isin(ann.query_source_indices[queries], galleries)
                             if exclude_source else np.zeros(len(queries), dtype=bool))
    result = dict(R1=None, R5=None, R10=None, mAP=None, n_query=len(queries),
                  n_query_input=original_count, n_query_excluded_no_positive=removed,
                  gallery_size=len(galleries), gallery_size_min=int(sizes.min()) if len(sizes) else 0,
                  gallery_size_max=int(sizes.max()) if len(sizes) else 0,
                  exclude_source_image=bool(exclude_source), top1_tied_queries=0,
                  tie_policy="descending score; tied candidates keep selected gallery index order")
    if not len(queries):
        return result
    hits = {1: 0, 5: 0, 10: 0}
    ap_sum = 0.0
    gallery_ids = ann.gallery_ids[galleries]
    for start in range(0, len(queries), block_size):
        q = queries[start:start + block_size]
        values = _score_block(scores, q, galleries)
        if penalties is not None:
            values *= 2
            values -= penalties[None, :]
        allowed = np.ones(values.shape, dtype=bool)
        if exclude_source:
            allowed = ann.query_source_indices[q, None] != galleries[None, :]
            values[~allowed] = -np.inf
        positives = (ann.query_ids[q, None] == gallery_ids[None, :]) & allowed
        maxima = values.max(axis=1)
        result["top1_tied_queries"] += int(((values == maxima[:, None]).sum(1) > 1).sum())
        if r1_only:
            hits[1] += int(positives[np.arange(len(q)), values.argmax(axis=1)].sum())
            continue
        order = np.argsort(-values, axis=1, kind="stable")
        matches = np.take_along_axis(positives, order, axis=1)
        for k in hits:
            hits[k] += int(matches[:, :k].any(axis=1).sum())
        precision = matches.cumsum(axis=1, dtype=np.int64) / np.arange(1, len(galleries) + 1)
        ap = (precision * matches).sum(axis=1) / positives.sum(axis=1)
        ap_sum += float(ap.sum(dtype=np.float64))
    result["R1"] = _online_percent(hits[1], len(queries))
    result["R1_correct"] = hits[1]
    if not r1_only:
        result.update(R5=_online_percent(hits[5], len(queries)),
                      R10=_online_percent(hits[10], len(queries)),
                      mAP=ap_sum / len(queries) * 100)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def print_table(headers, rows):
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [max(len(str(header)), *(len(row[i]) for row in rows))
              for i, header in enumerate(headers)] if rows else list(map(len, headers))
    print("  ".join(str(header).ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def make_mock_records():
    paths = ["AG-ReID.v2/P0000C0F0001.jpg", "AG-ReID.v2/P0000C3F0002.jpg",
             "AG-ReID.v2/P0000C0F0003.jpg", "G2APS/D1_1.jpg", "G2APS/S1_1.jpg",
             "G2APS/S2_1.jpg", "Market/0002_c1.jpg", "PRAI-1581/0003.jpg",
             "uavhuman/0004_0.jpg", "uavhuman/0004_1.jpg"]
    ids = [0, 0, 0, 1, 1, 1, 2, 3, 4, 4]
    return [dict(file_path=path, id=pid, captions=[f"caption {i}a", f"caption {i}b"])
            for i, (path, pid) in enumerate(zip(paths, ids))]


def make_mock_annotation():
    return annotation_from_records(make_mock_records(), enforce_anchors=False)


def make_mock_similarity(ann):
    rng = np.random.default_rng(2026)
    scores = rng.uniform(-0.5, 0.5, (len(ann.query_ids), len(ann.gallery_ids))).astype(np.float32)
    scores += (ann.query_ids[:, None] == ann.gallery_ids[None, :]) * np.float32(1)
    scores[np.arange(len(scores)), ann.query_source_indices] = 3
    return scores

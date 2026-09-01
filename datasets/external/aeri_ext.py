"""AERI-PEDES external-dataset adapter.

The adapter is deliberately label-only: it parses the official annotations and
returns paths/captions/labels, while image decoding and model execution remain in
the external runtime.  In particular, ``ground_img`` is never used to construct
a training sample (the official W/O Ground setting).
"""

import json
from pathlib import Path

from .common import (
    AdapterBundle,
    EvaluationProtocol,
    ExtraImageProtocol,
    TrainSample,
    contiguous_pid_remap,
    resolve_path,
)


DATASET_NAME = "aeri"

_EXPECTED = {
    "train_samples": 76007,
    "train_ids": 3522,
    "test_records": 1000,
    "test_queries": 6141,
    "test_gallery": 5525,
    "test_ids": 1000,
    "extra_ground": 4220,
}


def _read_list(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON list")
    return value


def _require_keys(record, keys, annotation, row):
    missing = sorted(set(keys) - set(record))
    if missing:
        raise ValueError(f"{annotation}: row {row} is missing keys {missing}")


def _append_unique(paths, pids, seen, path, pid):
    """Append the first occurrence, rejecting contradictory duplicate labels."""
    if path in seen:
        if seen[path] != pid:
            raise ValueError(f"image {path!r} has conflicting pids {seen[path]} and {pid}")
        return
    seen[path] = pid
    paths.append(path)
    pids.append(pid)


def load_adapter(data_root, strict=True):
    """Load AERI-PEDES using its official aerial-only retrieval protocol.

    ``strict=False`` exists solely for small contract tests.  Production callers
    use the default, which enforces every published dataset cardinality.
    """
    data_root = str(Path(data_root).resolve())
    train_annotation = resolve_path(data_root, "AERI-PEDES", "train_caption.json")
    test_annotation = resolve_path(data_root, "AERI-PEDES", "test_caption.json")
    train_rows = _read_list(train_annotation)
    test_rows = _read_list(test_annotation)

    train_original_pids = []
    for row_index, row in enumerate(train_rows):
        _require_keys(
            row,
            {"pid", "split", "image_id", "aerial_img", "ground_img", "caption"},
            train_annotation,
            row_index,
        )
        if str(row["split"]).lower() != "train":
            raise ValueError(f"{train_annotation}: row {row_index} is not split=train")
        train_original_pids.append(int(row["pid"]))
    train_pid_remap = contiguous_pid_remap(train_original_pids)

    # Official W/O Ground protocol: ground_img is intentionally not resolved or
    # exposed as a training path.  Merely parsing it above catches schema damage.
    train_samples = [
        TrainSample(
            image_path=resolve_path(data_root, "", row["aerial_img"]),
            caption=str(row["caption"]),
            pid=train_pid_remap[int(row["pid"])],
            original_pid=int(row["pid"]),
            cam_id=0,
        )
        for row in train_rows
    ]

    test_original_pids = []
    for row_index, row in enumerate(test_rows):
        _require_keys(
            row,
            {"pid", "split", "captions", "file_paths", "ground_file_paths"},
            test_annotation,
            row_index,
        )
        if str(row["split"]).lower() != "test":
            raise ValueError(f"{test_annotation}: row {row_index} is not split=test")
        if not isinstance(row["captions"], list) or not isinstance(row["file_paths"], list):
            raise ValueError(f"{test_annotation}: row {row_index} has non-list test fields")
        if not isinstance(row["ground_file_paths"], list):
            raise ValueError(f"{test_annotation}: row {row_index} has non-list ground paths")
        test_original_pids.append(int(row["pid"]))
    test_pid_remap = contiguous_pid_remap(test_original_pids)

    query_texts = []
    query_pid = []
    gallery_paths = []
    gallery_pid = []
    ground_paths = []
    ground_pid = []
    aerial_seen = {}
    # Dict insertion order deliberately preserves the first occurrence while
    # expanding test_rows and each row's ground_file_paths in annotation order.
    # Do not replace this with a set: extra feature row order is part of the
    # persisted analysis contract and must be reproducible across runs.
    ground_seen = {}
    for row in test_rows:
        mapped_pid = test_pid_remap[int(row["pid"])]
        for caption in row["captions"]:
            query_texts.append(str(caption))
            query_pid.append(mapped_pid)
        for relative in row["file_paths"]:
            path = resolve_path(data_root, "AERI-PEDES", relative)
            _append_unique(gallery_paths, gallery_pid, aerial_seen, path, mapped_pid)
        for relative in row["ground_file_paths"]:
            path = resolve_path(data_root, "AERI-PEDES", relative)
            _append_unique(ground_paths, ground_pid, ground_seen, path, mapped_pid)

    overlap = set(gallery_paths) & set(ground_paths)
    if overlap:
        example = next(iter(overlap))
        raise ValueError(f"aerial and ground test path sets overlap, e.g. {example}")

    evaluation = EvaluationProtocol(
        split="test",
        protocol="official_aerial_gallery",
        query_texts=query_texts,
        query_pid=query_pid,
        gallery_paths=gallery_paths,
        gallery_pid=gallery_pid,
        # AERI test captions are identity-level lists and do not identify one
        # unique source image within the aerial gallery.
        query_source_gallery_idx=[-1] * len(query_texts),
        pid_remap=test_pid_remap,
        query_view=["aerial"] * len(query_texts),
        gallery_view=["aerial"] * len(gallery_paths),
    )
    extra = ExtraImageProtocol(
        split="test",
        paths=ground_paths,
        pid=ground_pid,
        view=["ground"] * len(ground_paths),
        pid_remap=test_pid_remap,
    )

    bundle = AdapterBundle(
        dataset=DATASET_NAME,
        data_root=data_root,
        annotation_paths=[train_annotation, test_annotation],
        train_samples=train_samples,
        train_pid_remap=train_pid_remap,
        evaluations={"test": evaluation},
        extras={"ground": extra},
        notes=[
            "Official W/O Ground training: ground_img is parsed but never used.",
            "Test captions are identity-level, so query_source_gallery_idx is -1.",
        ],
    ).validate()

    if strict:
        actual = {
            "train_samples": len(bundle.train_samples),
            "train_ids": bundle.num_classes,
            "test_records": len(test_rows),
            "test_queries": len(evaluation.query_texts),
            "test_gallery": len(evaluation.gallery_paths),
            "test_ids": len(test_pid_remap),
            "extra_ground": len(extra.paths),
        }
        if actual != _EXPECTED:
            raise ValueError(f"AERI-PEDES cardinality mismatch: expected {_EXPECTED}, got {actual}")
    return bundle


__all__ = ["DATASET_NAME", "load_adapter"]

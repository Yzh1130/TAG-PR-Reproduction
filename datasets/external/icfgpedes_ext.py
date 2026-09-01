"""ICFG-PEDES annotation adapter for the external evaluation runtime.

The JSON ``split`` value is authoritative.  In particular, ``train/`` and
``test/`` inside an image path are inherited MSMT17 directory names and must
never be used to partition ICFG-PEDES.
"""

import json
from collections import defaultdict
from pathlib import Path

from .common import (
    AdapterBundle,
    EvaluationProtocol,
    TrainSample,
    contiguous_pid_remap,
    resolve_path,
)


DATASET_NAME = "icfgpedes"
_ANNOTATION = "ICFG-PEDES/ICFG-PEDES.json"
_IMAGE_PREFIX = "ICFG-PEDES/imgs"
_KNOWN_DUPLICATE = "test/2868/2868_005_05_0114afternoon_0833_1_ex.jpg"
_EXPECTED = {
    "train_rows": 34674,
    "train_samples": 34674,
    "train_ids": 3102,
    "test_queries": 19848,
    "test_gallery": 19848,
}


def _read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON list")
    return records


def _captions(record, row):
    captions = record.get("captions")
    if not isinstance(captions, list) or not captions:
        raise ValueError(f"ICFG-PEDES row {row}: captions must be a nonempty list")
    if any(not isinstance(caption, str) for caption in captions):
        raise ValueError(f"ICFG-PEDES row {row}: every caption must be a string")
    return captions


def _normalise_relative_path(value):
    return str(value).replace("\\", "/").lstrip("/")


def _assert_known_duplicate(records):
    occurrences = defaultdict(list)
    for row, record in enumerate(records):
        occurrences[_normalise_relative_path(record["file_path"])].append((row, record))
    duplicates = {path: rows for path, rows in occurrences.items() if len(rows) > 1}
    if set(duplicates) != {_KNOWN_DUPLICATE}:
        raise ValueError(
            "ICFG-PEDES duplicate-image invariant failed: expected exactly "
            f"{_KNOWN_DUPLICATE!r}, observed {sorted(duplicates)!r}"
        )
    duplicate_rows = duplicates[_KNOWN_DUPLICATE]
    duplicate_ids = sorted(int(record["id"]) for _, record in duplicate_rows)
    duplicate_splits = [record.get("split") for _, record in duplicate_rows]
    if len(duplicate_rows) != 2 or duplicate_ids != [1, 5] or duplicate_splits != ["train", "train"]:
        raise ValueError(
            "ICFG-PEDES known duplicate must occur in exactly two train rows with ids 1 and 5; "
            f"observed rows={len(duplicate_rows)}, ids={duplicate_ids}, splits={duplicate_splits}"
        )
    print("ICFG DUPLICATE SUMMARY")
    print(f"path: {_KNOWN_DUPLICATE}")
    print(f"rows: {[row for row, _ in duplicate_rows]}")
    print(f"ids: {duplicate_ids}")
    print("preserved_as_independent_training_samples: true")
    print("END ICFG DUPLICATE SUMMARY")


def _build_test_evaluation(records, data_root):
    selected = [(row, record) for row, record in enumerate(records)
                if record.get("split") == "test"]
    if not selected:
        raise ValueError("ICFG-PEDES: test split is empty")
    pid_remap = contiguous_pid_remap(record["id"] for _, record in selected)
    gallery_paths = []
    gallery_pid = []
    path_to_index = {}
    query_texts = []
    query_pid = []
    query_source_gallery_idx = []

    for row, record in selected:
        original_pid = int(record["id"])
        pid = pid_remap[original_pid]
        path = resolve_path(data_root, _IMAGE_PREFIX, record["file_path"])
        if path not in path_to_index:
            path_to_index[path] = len(gallery_paths)
            gallery_paths.append(path)
            gallery_pid.append(pid)
        elif gallery_pid[path_to_index[path]] != pid:
            raise ValueError(
                f"ICFG-PEDES test image {record['file_path']!r} has conflicting ids"
            )
        source_index = path_to_index[path]
        for caption in _captions(record, row):
            query_texts.append(caption)
            query_pid.append(pid)
            query_source_gallery_idx.append(source_index)

    return EvaluationProtocol(
        split="test",
        protocol="identity_retrieval_unique_image_gallery",
        query_texts=query_texts,
        query_pid=query_pid,
        gallery_paths=gallery_paths,
        gallery_pid=gallery_pid,
        query_source_gallery_idx=query_source_gallery_idx,
        pid_remap=pid_remap,
    ).validate()


def load_adapter(data_root, strict=True):
    """Parse ICFG-PEDES while preserving its two conflicting train rows.

    ``strict=False`` is intended only for compact schema/contract tests.  The
    known duplicate invariant is always enforced; production additionally
    checks every published dataset cardinality.
    """

    data_root = str(Path(data_root).resolve())
    annotation_path = Path(data_root) / _ANNOTATION
    records = _read_json(annotation_path)
    allowed_splits = {"train", "test"}
    observed_splits = {record.get("split") for record in records}
    unexpected = observed_splits - allowed_splits
    if unexpected:
        raise ValueError(f"ICFG-PEDES: unsupported split values: {sorted(unexpected)!r}")
    _assert_known_duplicate(records)

    train_records = [(row, record) for row, record in enumerate(records)
                     if record.get("split") == "train"]
    if not train_records:
        raise ValueError("ICFG-PEDES: train split is empty")
    if strict:
        non_single_caption_rows = [
            row for row, record in train_records
            if len(_captions(record, row)) != 1
        ]
        if non_single_caption_rows:
            raise ValueError(
                "ICFG-PEDES: every production train row must contain exactly "
                f"one caption; offending rows: {non_single_caption_rows[:10]}"
            )
    train_pid_remap = contiguous_pid_remap(record["id"] for _, record in train_records)
    train_samples = []
    for row, record in train_records:
        original_pid = int(record["id"])
        path = resolve_path(data_root, _IMAGE_PREFIX, record["file_path"])
        for caption in _captions(record, row):
            train_samples.append(
                TrainSample(
                    image_path=path,
                    caption=caption,
                    pid=train_pid_remap[original_pid],
                    original_pid=original_pid,
                )
            )

    evaluation = _build_test_evaluation(records, data_root)
    bundle = AdapterBundle(
        dataset=DATASET_NAME,
        data_root=data_root,
        annotation_paths=[str(annotation_path.resolve())],
        train_samples=train_samples,
        train_pid_remap=train_pid_remap,
        evaluations={"test": evaluation},
        notes=[
            "split selection uses only the JSON split field, never path directories",
            "the known conflicting image appears as two independent train samples with ids 1 and 5",
            "the test gallery contains unique images in annotation order",
        ],
    ).validate()
    if strict:
        actual = {
            "train_rows": len(train_records),
            "train_samples": len(bundle.train_samples),
            "train_ids": bundle.num_classes,
            "test_queries": len(evaluation.query_texts),
            "test_gallery": len(evaluation.gallery_paths),
        }
        if actual != _EXPECTED:
            raise ValueError(f"ICFG-PEDES cardinality mismatch: expected {_EXPECTED}, got {actual}")
    return bundle


__all__ = ["DATASET_NAME", "load_adapter"]

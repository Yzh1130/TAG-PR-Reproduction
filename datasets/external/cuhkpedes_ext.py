"""CUHK-PEDES annotation adapter for the external evaluation runtime.

The adapter is deliberately label-only: it does not import or alter the
TAG-PEDES loader, transforms, samplers, or evaluation code.
"""

import json
from pathlib import Path

from .common import (
    AdapterBundle,
    EvaluationProtocol,
    TrainSample,
    contiguous_pid_remap,
    resolve_path,
)


DATASET_NAME = "cuhkpedes"
_ANNOTATION = "CUHK-PEDES/reid_raw.json"
_IMAGE_PREFIX = "CUHK-PEDES/imgs"
_EXPECTED = {
    "train_samples": 68126,
    "train_ids": 11003,
    "val_queries": 6158,
    "val_gallery": 3078,
    "val_ids": 1000,
    "test_queries": 6156,
    "test_gallery": 3074,
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
        raise ValueError(f"CUHK-PEDES row {row}: captions must be a nonempty list")
    if any(not isinstance(caption, str) for caption in captions):
        raise ValueError(f"CUHK-PEDES row {row}: every caption must be a string")
    return captions


def _build_evaluation(records, data_root, split):
    selected = [(row, record) for row, record in enumerate(records)
                if record.get("split") == split]
    if not selected:
        raise ValueError(f"CUHK-PEDES: split {split!r} is empty")

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
                f"CUHK-PEDES {split}: image {record['file_path']!r} has conflicting ids"
            )
        source_index = path_to_index[path]
        for caption in _captions(record, row):
            query_texts.append(caption)
            query_pid.append(pid)
            query_source_gallery_idx.append(source_index)

    return EvaluationProtocol(
        split=split,
        protocol="identity_retrieval_unique_image_gallery",
        query_texts=query_texts,
        query_pid=query_pid,
        gallery_paths=gallery_paths,
        gallery_pid=gallery_pid,
        query_source_gallery_idx=query_source_gallery_idx,
        pid_remap=pid_remap,
    ).validate()


def load_adapter(data_root, strict=True):
    """Parse CUHK-PEDES using only the JSON ``split`` field.

    ``strict=False`` is intended only for compact schema/contract tests.  The
    production default checks the published split cardinalities.
    """

    data_root = str(Path(data_root).resolve())
    annotation_path = Path(data_root) / _ANNOTATION
    records = _read_json(annotation_path)
    allowed_splits = {"train", "val", "test"}
    observed_splits = {record.get("split") for record in records}
    unexpected = observed_splits - allowed_splits
    if unexpected:
        raise ValueError(f"CUHK-PEDES: unsupported split values: {sorted(unexpected)!r}")

    train_records = [(row, record) for row, record in enumerate(records)
                     if record.get("split") == "train"]
    if not train_records:
        raise ValueError("CUHK-PEDES: train split is empty")
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

    evaluations = {
        "val": _build_evaluation(records, data_root, "val"),
        "test": _build_evaluation(records, data_root, "test"),
    }
    bundle = AdapterBundle(
        dataset=DATASET_NAME,
        data_root=data_root,
        annotation_paths=[str(annotation_path.resolve())],
        train_samples=train_samples,
        train_pid_remap=train_pid_remap,
        evaluations=evaluations,
        notes=[
            "train expands every train-record caption into one sample",
            "val/test galleries contain unique images in annotation order",
            "all splits are selected exclusively from the JSON split field",
        ],
    ).validate()
    if strict:
        actual = {
            "train_samples": len(bundle.train_samples),
            "train_ids": bundle.num_classes,
            "val_queries": len(evaluations["val"].query_texts),
            "val_gallery": len(evaluations["val"].gallery_paths),
            "val_ids": len(evaluations["val"].pid_remap),
            "test_queries": len(evaluations["test"].query_texts),
            "test_gallery": len(evaluations["test"].gallery_paths),
        }
        if actual != _EXPECTED:
            raise ValueError(f"CUHK-PEDES cardinality mismatch: expected {_EXPECTED}, got {actual}")
    return bundle


__all__ = ["DATASET_NAME", "load_adapter"]

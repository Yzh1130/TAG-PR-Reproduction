"""TBAPR/TBAGPR adapter matching the official AEA-FIRM merge protocol."""

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


DATASET_NAME = "tbapr"

_EXPECTED = {
    "ag_train_rows": 8154,
    "ag_train_ids": 199,
    "train_samples": 31340,
    "train_ids": 1180,
    "ag_test_ids": 189,
    "test_queries": 3200,
    "test_gallery_rows": 3200,
    "test_ids": 529,
}


def _read_json(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _relative_path(value, annotation, field, row):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return "/".join(str(part).strip("/\\") for part in value)
    raise ValueError(f"{annotation}: {field} row {row} must be a path or two path components")


def _validate_train_row(row, annotation, index):
    if not isinstance(row, (list, tuple)) or len(row) != 5:
        raise ValueError(f"{annotation}: train row {index} must contain five fields")
    pid, image_id, image, pair_image, caption = row
    # Training paths are specified as exactly two components by AEA-FIRM.
    if not isinstance(image, (list, tuple)) or len(image) != 2:
        raise ValueError(f"{annotation}: img row {index} is not a two-item path")
    if not isinstance(pair_image, (list, tuple)) or len(pair_image) != 2:
        raise ValueError(f"{annotation}: pair_img row {index} is not a two-item path")
    return int(pid), int(image_id), image, pair_image, str(caption)


def _validate_test_dict(value, annotation):
    required = {"image_pids", "img_paths", "pair_img_paths", "caption_pids", "captions"}
    if not isinstance(value, dict):
        raise ValueError(f"{annotation}: expected a JSON object")
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{annotation}: missing keys {missing}")
    lengths = {key: len(value[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{annotation}: test arrays have unequal lengths {lengths}")
    return lengths["captions"]


def load_adapter(data_root, strict=True):
    """Load TBAPR with official AG-ReID + G2APS-AG offsets and row ordering.

    ``strict=False`` is provided for compact mock tests only.  In production the
    default strict mode checks the official cardinalities and offset bases.
    """
    data_root = str(Path(data_root).resolve())
    annotations = {
        "ag_train": resolve_path(data_root, "TBAGPR/AG-ReID", "agtext_train.json"),
        "g2_train": resolve_path(data_root, "TBAGPR/G2APS-AG", "g2aps_train.json"),
        "ag_test": resolve_path(data_root, "TBAGPR/AG-ReID", "agtext_test.json"),
        "g2_test": resolve_path(data_root, "TBAGPR/G2APS-AG", "g2aps_test.json"),
    }
    ag_train = _read_json(annotations["ag_train"])
    g2_train = _read_json(annotations["g2_train"])
    if not isinstance(ag_train, list) or not isinstance(g2_train, list):
        raise ValueError("TBAPR training annotations must both be JSON lists")

    parsed_ag = [
        _validate_train_row(row, annotations["ag_train"], index)
        for index, row in enumerate(ag_train)
    ]
    parsed_g2 = [
        _validate_train_row(row, annotations["g2_train"], index)
        for index, row in enumerate(g2_train)
    ]
    ag_train_id_count = len({row[0] for row in parsed_ag})
    ag_train_row_count = len(parsed_ag)

    # Apply both official offsets before the final contiguous classifier remap.
    merged_train = []
    for pid, image_id, image, _pair_image, caption in parsed_ag:
        merged_train.append((pid, image_id, image, caption, "AG-ReID"))
    for pid, image_id, image, _pair_image, caption in parsed_g2:
        merged_train.append(
            (pid + ag_train_id_count, image_id + ag_train_row_count, image, caption, "G2APS-AG")
        )
    train_pid_remap = contiguous_pid_remap(row[0] for row in merged_train)
    train_samples = []
    for merged_pid, _merged_image_id, image, caption, subset in merged_train:
        train_samples.append(
            TrainSample(
                image_path=resolve_path(
                    data_root,
                    f"TBAGPR/{subset}",
                    _relative_path(image, "TBAPR train", "img", len(train_samples)),
                ),
                caption=caption,
                pid=train_pid_remap[merged_pid],
                original_pid=merged_pid,
                source_subset=subset,
            )
        )

    ag_test = _read_json(annotations["ag_test"])
    g2_test = _read_json(annotations["g2_test"])
    ag_test_count = _validate_test_dict(ag_test, annotations["ag_test"])
    g2_test_count = _validate_test_dict(g2_test, annotations["g2_test"])
    ag_test_id_count = len({int(pid) for pid in ag_test["image_pids"]})

    merged_gallery_original_pid = []
    merged_query_original_pid = []
    gallery_paths = []
    pair_paths = []
    gallery_subset = []
    query_subset = []
    query_texts = []
    query_source_gallery_idx = []
    for subset, value, pid_offset in (
        ("AG-ReID", ag_test, 0),
        ("G2APS-AG", g2_test, ag_test_id_count),
    ):
        # The official arrays are row-aligned within each source dataset.  Keep
        # that relationship explicit instead of relying on the two merged
        # arrays coincidentally having the same global order.
        gallery_offset = len(gallery_paths)
        for index, raw_pid in enumerate(value["image_pids"]):
            merged_gallery_original_pid.append(int(raw_pid) + pid_offset)
            gallery_paths.append(
                resolve_path(
                    data_root,
                    f"TBAGPR/{subset}",
                    _relative_path(value["img_paths"][index], annotations["ag_test" if subset == "AG-ReID" else "g2_test"], "img_paths", index),
                )
            )
            pair_paths.append(
                resolve_path(
                    data_root,
                    f"TBAGPR/{subset}",
                    _relative_path(value["pair_img_paths"][index], annotations["ag_test" if subset == "AG-ReID" else "g2_test"], "pair_img_paths", index),
                )
            )
            gallery_subset.append(subset)
        for index, (raw_pid, caption) in enumerate(
            zip(value["caption_pids"], value["captions"])
        ):
            merged_query_original_pid.append(int(raw_pid) + pid_offset)
            query_texts.append(str(caption))
            query_subset.append(subset)
            query_source_gallery_idx.append(gallery_offset + index)

    test_pid_remap = contiguous_pid_remap(
        merged_gallery_original_pid + merged_query_original_pid
    )
    gallery_pid = [test_pid_remap[pid] for pid in merged_gallery_original_pid]
    query_pid = [test_pid_remap[pid] for pid in merged_query_original_pid]
    for query_index, gallery_index in enumerate(query_source_gallery_idx):
        if gallery_pid[gallery_index] != query_pid[query_index]:
            raise ValueError(
                "TBAPR official row alignment maps query "
                f"{query_index} (pid={query_pid[query_index]}) to gallery "
                f"{gallery_index} (pid={gallery_pid[gallery_index]})"
            )

    evaluation = EvaluationProtocol(
        split="test",
        protocol="official_row_gallery",
        query_texts=query_texts,
        query_pid=query_pid,
        # Do not deduplicate: official G2APS evaluation ranks all annotation rows.
        gallery_paths=gallery_paths,
        gallery_pid=gallery_pid,
        query_source_gallery_idx=query_source_gallery_idx,
        pid_remap=test_pid_remap,
        query_view=["img"] * len(query_texts),
        gallery_view=["img"] * len(gallery_paths),
        query_source_subset=query_subset,
        gallery_source_subset=gallery_subset,
    )
    extra = ExtraImageProtocol(
        split="test",
        paths=pair_paths,
        pid=gallery_pid.copy(),
        view=["pair"] * len(pair_paths),
        pid_remap=test_pid_remap,
        source_subset=gallery_subset.copy(),
    )
    bundle = AdapterBundle(
        dataset=DATASET_NAME,
        data_root=data_root,
        annotation_paths=list(annotations.values()),
        train_samples=train_samples,
        train_pid_remap=train_pid_remap,
        evaluations={"test": evaluation},
        extras={"pair": extra},
        notes=[
            f"Official G2APS train offsets: pid +{ag_train_id_count}, image_id +{ag_train_row_count}.",
            f"Official G2APS test pid offset: +{ag_test_id_count}.",
            "pair_img is excluded from training and the official gallery.",
            "Official gallery rows are preserved without image-path deduplication.",
        ],
    ).validate()

    if len(query_texts) != len(gallery_paths):
        raise ValueError(
            f"TBAPR official query/gallery row alignment failed: {len(query_texts)} vs {len(gallery_paths)}"
        )
    if strict:
        actual = {
            "ag_train_rows": ag_train_row_count,
            "ag_train_ids": ag_train_id_count,
            "train_samples": len(bundle.train_samples),
            "train_ids": bundle.num_classes,
            "ag_test_ids": ag_test_id_count,
            "test_queries": len(evaluation.query_texts),
            "test_gallery_rows": len(evaluation.gallery_paths),
            "test_ids": len(test_pid_remap),
        }
        if actual != _EXPECTED:
            raise ValueError(f"TBAPR cardinality mismatch: expected {_EXPECTED}, got {actual}")
    # Silence unused variables while retaining explicit per-source array checks.
    if ag_test_count + g2_test_count != len(query_texts):
        raise AssertionError("TBAPR merged test row conservation failed")
    return bundle


__all__ = ["DATASET_NAME", "load_adapter"]

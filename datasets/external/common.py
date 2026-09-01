"""Typed, label-only contract shared by independent external adapters.

This module has no Torch/PIL import. Adapters only parse annotations and resolve
paths; image decoding, transforms, training and evaluation live in the external
runtime and never enter the TAG-PEDES loader.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TrainSample:
    image_path: str
    caption: str
    pid: int
    original_pid: int
    cam_id: int = 1
    source_subset: Optional[str] = None


@dataclass
class EvaluationProtocol:
    split: str
    protocol: str
    query_texts: List[str]
    query_pid: List[int]
    gallery_paths: List[str]
    gallery_pid: List[int]
    query_source_gallery_idx: List[int]
    pid_remap: Dict[int, int]
    query_view: Optional[List[str]] = None
    gallery_view: Optional[List[str]] = None
    query_source_subset: Optional[List[str]] = None
    gallery_source_subset: Optional[List[str]] = None

    def validate(self):
        nq, ng = len(self.query_texts), len(self.gallery_paths)
        fields = {
            "query_pid": len(self.query_pid),
            "query_source_gallery_idx": len(self.query_source_gallery_idx),
        }
        if self.query_view is not None:
            fields["query_view"] = len(self.query_view)
        if self.query_source_subset is not None:
            fields["query_source_subset"] = len(self.query_source_subset)
        for name, size in fields.items():
            if size != nq:
                raise ValueError(f"{name} has {size} rows, expected {nq}")
        gallery_fields = {"gallery_pid": len(self.gallery_pid)}
        if self.gallery_view is not None:
            gallery_fields["gallery_view"] = len(self.gallery_view)
        if self.gallery_source_subset is not None:
            gallery_fields["gallery_source_subset"] = len(self.gallery_source_subset)
        for name, size in gallery_fields.items():
            if size != ng:
                raise ValueError(f"{name} has {size} rows, expected {ng}")
        if nq < 1 or ng < 1:
            raise ValueError("evaluation query and gallery must both be nonempty")
        if any(index < -1 or index >= ng for index in self.query_source_gallery_idx):
            raise ValueError("query_source_gallery_idx contains an invalid gallery row")
        known = set(self.pid_remap.values())
        if known != set(range(len(self.pid_remap))):
            raise ValueError("evaluation pid_remap is not contiguous from zero")
        if set(self.query_pid) - known or set(self.gallery_pid) - known:
            raise ValueError("evaluation pid is absent from pid_remap")
        for query_index, gallery_index in enumerate(self.query_source_gallery_idx):
            if gallery_index >= 0 and self.gallery_pid[gallery_index] != self.query_pid[query_index]:
                raise ValueError(
                    "query_source_gallery_idx points to a gallery row with a "
                    "different identity"
                )
        return self


@dataclass
class ExtraImageProtocol:
    split: str
    paths: List[str]
    pid: List[int]
    view: List[str]
    pid_remap: Dict[int, int]
    source_subset: Optional[List[str]] = None

    def validate(self):
        count = len(self.paths)
        if count < 1 or len(self.pid) != count or len(self.view) != count:
            raise ValueError("extra image paths/pids/views must have equal nonzero length")
        if self.source_subset is not None and len(self.source_subset) != count:
            raise ValueError("extra source_subset length mismatch")
        known = set(self.pid_remap.values())
        if known != set(range(len(self.pid_remap))):
            raise ValueError("extra-image pid_remap is not contiguous from zero")
        if set(self.pid) - known:
            raise ValueError("extra-image pid is absent from pid_remap")
        return self


@dataclass
class AdapterBundle:
    dataset: str
    data_root: str
    annotation_paths: List[str]
    train_samples: List[TrainSample]
    train_pid_remap: Dict[int, int]
    evaluations: Dict[str, EvaluationProtocol]
    extras: Dict[str, ExtraImageProtocol] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def num_classes(self):
        return len(self.train_pid_remap)

    def validate(self):
        if not self.train_samples or not self.train_pid_remap:
            raise ValueError(f"{self.dataset}: empty training samples or pid map")
        expected = set(range(self.num_classes))
        if set(self.train_pid_remap.values()) != expected:
            raise ValueError(f"{self.dataset}: train pids are not mapped to 0..C-1")
        if any(sample.pid not in expected for sample in self.train_samples):
            raise ValueError(f"{self.dataset}: a training pid is outside the classifier")
        for protocol in self.evaluations.values():
            protocol.validate()
        for protocol in self.extras.values():
            protocol.validate()
        return self

    def all_image_paths(self):
        paths = [sample.image_path for sample in self.train_samples]
        for protocol in self.evaluations.values():
            paths.extend(protocol.gallery_paths)
        for protocol in self.extras.values():
            paths.extend(protocol.paths)
        return paths


def contiguous_pid_remap(values):
    unique = sorted(set(int(value) for value in values))
    return {pid: index for index, pid in enumerate(unique)}


def resolve_path(data_root, prefix, relative):
    relative = str(relative).replace("\\", "/").lstrip("/")
    prefix = str(prefix).replace("\\", "/").strip("/")
    return str((Path(data_root) / prefix / Path(relative)).resolve())


def path_resolution(paths, limit=None):
    paths = list(paths)
    if limit is not None and len(paths) > limit:
        # Stable coverage across the complete ordered list, not random retries.
        positions = [round(i * (len(paths) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
        paths = [paths[position] for position in positions]
    resolved = sum(Path(path).is_file() for path in paths)
    return {"checked": len(paths), "resolved": resolved,
            "ratio": resolved / len(paths) if paths else 0.0}

"""S12 feature extraction; imports the unmodified repository only on demand."""

import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np


def _read_records(path, split=None):
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        if split and isinstance(payload.get(split), list):
            payload = payload[split]
        else:
            choices = [key for key in ("annotations", "records", "data") if isinstance(payload.get(key), list)]
            if len(choices) != 1:
                raise ValueError(f"{path}: expected record list or one records/annotations/data list; keys={list(payload)}")
            payload = payload[choices[0]]
    if not isinstance(payload, list) or not payload or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{path}: annotation must contain a nonempty list of object records")
    fields = sorted(set().union(*(row.keys() for row in payload)))
    print(f"ANNOTATION INSPECTION path={path} records={len(payload)} fields={fields}")
    for index, row in enumerate(payload):
        if not {"file_path", "id", "captions"}.issubset(row):
            raise ValueError(f"{path}: record {index} lacks file_path/id/captions; fields={list(row)}")
        captions = row["captions"]
        if not isinstance(captions, list) or not captions or not all(isinstance(text, str) for text in captions):
            raise ValueError(f"{path}: record {index} captions must be a nonempty string list")
    return payload, {"path": str(Path(path).resolve()), "records": len(payload), "fields": fields,
                     "captions": sum(len(row["captions"]) for row in payload),
                     "identities": len({row["id"] for row in payload})}


def discover_train_annotations(test_path, explicit=None):
    """Inspect candidate contents instead of assuming train_reid.json exists."""
    if explicit:
        candidates = [Path(explicit)]
    else:
        parent = Path(test_path).resolve().parent
        search_dirs = [parent] + [parent / name for name in ("anno", "annotations") if (parent / name).is_dir()]
        candidates = sorted({path for directory in search_dirs for path in directory.glob("*.json")
                             if path.resolve() != Path(test_path).resolve()})
    found, rejected = [], []
    for path in candidates:
        try:
            with path.open(encoding="utf-8") as stream:
                probe = json.load(stream)
            has_train_key = isinstance(probe, dict) and isinstance(probe.get("train"), list)
            probe_rows = probe.get("train", probe) if isinstance(probe, dict) else probe
            splits = {str(row.get("split", "")).lower() for row in probe_rows if isinstance(row, dict)} if isinstance(probe_rows, list) else set()
            is_train = explicit or has_train_key or (splits and splits <= {"train", "training"}) or "train" in path.stem.lower()
            if not is_train:
                continue
            records, info = _read_records(path, split="train")
            declared = {str(row["split"]).lower() for row in records if "split" in row}
            if declared and not declared <= {"train", "training"}:
                raise ValueError(f"candidate contains non-training splits: {sorted(declared)}")
            found.append((path, records, info))
        except (OSError, ValueError, TypeError) as exc:
            rejected.append(f"{path}: {exc}")
    if len(found) != 1:
        raise ValueError("Training annotation discovery requires exactly one valid candidate; "
                         f"found={[str(item[0]) for item in found]}; rejected={rejected}. "
                         "Use --train-anno to disambiguate; no annotation is guessed.")
    return found[0]


def add_repository_to_path(repo=None):
    choices = [Path(repo)] if repo else [Path(__file__).resolve().parents[1], Path.cwd(),
                                        Path("/root/autodl-tmp/TAG-PR-Reproduction"), Path("/root/autodl-tmp/TAG-PR")]
    for path in choices:
        if (path / "model/tbps_model_MoE.py").is_file() and (path / "main.py").is_file():
            sys.path.insert(0, str(path.resolve()))
            return path.resolve()
    raise FileNotFoundError("Cannot locate repository model/main.py; pass --repo /path/to/TAG-PR-Reproduction")


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidate = path / "checkpoint_best.pth"
        if not candidate.is_file():
            raise FileNotFoundError(f"No checkpoint_best.pth under {path}; pass --resume exact-file")
        path = candidate
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_model(config_path, checkpoint, backbone, device, num_classes, seed, repo=None):
    """Use existing config defaults, architecture inference and strict resume loader."""
    root = add_repository_to_path(repo)
    import torch
    import main as training_main
    from model.tbps_model_MoE import configure_backbone_from_checkpoint, clip_vitb

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; S12 feature extraction needs the training GPU environment")
    argv = ["--config", str(config_path), "--eval_only", "--resume", str(checkpoint)]
    if backbone:
        argv.extend(["--backbone", backbone])
    args = training_main.build_parser().parse_args(argv)
    config, _ = training_main.resolve_runtime_config(args)
    configure_backbone_from_checkpoint(config, checkpoint_path=str(checkpoint))
    seed = int(config.misc.seed if seed is None else seed)
    # One initial seed only. In particular it is NOT reset between extractions.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    config.device, config.num_classes = str(device), num_classes
    model = clip_vitb(config, num_classes)

    class ResumeLog:
        state_name = None

        def info(self, message, *values):
            print(message % values if values else message)
            if message.startswith("Loaded %s weights"):
                self.state_name = values[0]

        def warning(self, message, *values):
            print("WARNING: " + (message % values if values else message))

    resume_log = ResumeLog()
    training_main.load_resume_weights(model, str(checkpoint), config.experiment.use_ema, resume_log)
    model.to(device).eval()
    if model.dtype != torch.float32:
        raise ValueError(f"Expected unchanged float32 checkpoint evaluation, got model dtype {model.dtype}")
    return model, config, {"repository": str(root), "checkpoint": str(checkpoint), "config": str(Path(config_path).resolve()),
                           "backbone": config.model.backbone, "use_ema_requested": bool(config.experiment.use_ema),
                           "loaded_state": resume_log.state_name,
                           "seed": seed, "device": str(device), "torch": torch.__version__,
                           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                           "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                           "model_parameters": sum(parameter.numel() for parameter in model.parameters())}


def preprocess_texts(records, max_words):
    from misc.caption_dataset import pre_caption
    return [pre_caption(text, max_words=max_words) for row in records for text in row["captions"]]


def extract_text_features(model, texts, max_length, device, view_prompt=False, batch_size=256, tokenizer=None):
    """Exact eval text path: global and TSE independently normalized, no image calls."""
    import torch
    import torch.nn.functional as F
    if tokenizer is None:
        from text_utils.tokenizer import tokenize as tokenizer
    if batch_size < 1 or not texts:
        raise ValueError("Text extraction needs nonempty texts and a positive batch size")
    result = {}
    model.eval()
    with torch.no_grad():
        for view in ((0, 1) if view_prompt else (None,)):
            global_parts, tse_parts = [], []
            for start in range(0, len(texts), batch_size):
                tokens = tokenizer(texts[start:start + batch_size], context_length=max_length - int(view_prompt)).to(device)
                if view_prompt:
                    cams = torch.full((tokens.shape[0],), view, dtype=torch.long, device=device)
                    global_feat, dense, attention = model.encode_text_with_view_prompt(tokens, cams, return_dense=True)
                    tse_tokens = model.view_prompt_token_ids(tokens)
                else:
                    global_feat, dense, attention = model.encode_text(tokens, return_dense=True)
                    tse_tokens = tokens
                tse_feat = model.textual_emb_layer(attention, tse_tokens, dense, training=False)
                global_parts.append(F.normalize(global_feat, dim=-1).cpu().numpy())
                tse_parts.append(F.normalize(tse_feat, dim=-1).cpu().numpy())
            suffix = f"_view{view}" if view_prompt else ""
            result[f"text_feat{suffix}"] = np.concatenate(global_parts)
            result[f"text_tse_feat{suffix}"] = np.concatenate(tse_parts)
    validate_features(result)
    return result


def validate_features(features):
    counts = set()
    for name, values in features.items():
        if values.ndim != 2 or min(values.shape) < 1 or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected finite two-dimensional float32 features, got {values.shape}/{values.dtype}")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-5, rtol=0):
            raise ValueError(f"{name}: features are not L2 normalized; norm range={norms.min(), norms.max()}")
        counts.add(len(values))
    if len(counts) != 1:
        raise ValueError(f"Feature sample counts differ: {counts}")


def assert_identical_features(first, second):
    if first.keys() != second.keys():
        raise RuntimeError("Consecutive extraction feature keys differ")
    report = {}
    for key in first:
        left, right = first[key], second[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"Consecutive extraction shape/dtype mismatch in {key}")
        left_bits, right_bits = left.view(np.uint32).reshape(-1), right.view(np.uint32).reshape(-1)
        count, positions, maximum = 0, [], 0.0
        # Compare bits (including signed zero), with bounded diagnostic memory.
        for start in range(0, left_bits.size, 65536):
            stop = min(start + 65536, left_bits.size)
            mismatch = left_bits[start:stop] != right_bits[start:stop]
            count += int(mismatch.sum())
            if len(positions) < 10 and mismatch.any():
                for offset in np.flatnonzero(mismatch)[:10 - len(positions)]:
                    positions.append([int(part) for part in np.unravel_index(start + offset, left.shape)])
            maximum = max(maximum, float(np.max(np.abs(left.reshape(-1)[start:stop] - right.reshape(-1)[start:stop]))))
        report[key] = {"shape": list(left.shape), "max_abs_error": maximum,
                       "different_elements": count, "bitwise_equal": count == 0}
        if not report[key]["bitwise_equal"]:
            raise RuntimeError(f"NONDETERMINISTIC TEXT FEATURES {key}: max_abs_error={maximum}, "
                               f"different_elements={count}, first_positions={positions}; no seed reset/retry performed")
    print("CONSECUTIVE TEXT EXTRACTION " + json.dumps(report, sort_keys=True))
    return report


def save_training_features(path, features, metadata):
    path = Path(path)
    if "_archive_evidence" in {part.lower() for part in path.resolve().parts}:
        raise ValueError("Refusing to write into read-only _archive_evidence")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if "metadata_json" not in saved:
                raise FileExistsError(f"{path} exists without S12 provenance; choose a fresh --out directory")
            old_meta = json.loads(str(saved["metadata_json"].item()))
            if old_meta != metadata or set(saved.files) != set(features) | {"metadata_json"}:
                raise FileExistsError(f"{path} exists with different provenance/features; choose a fresh --out directory")
            assert_identical_features(features, {key: saved[key] for key in features})
        print(f"Existing identical training features verified, not overwritten: {path}")
    else:
        with path.open("xb") as stream:
            np.savez(stream, **features, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    return str(path)


def discover_gallery_features(directory, explicit=None):
    candidates = [Path(explicit)] if explicit else [Path(directory) / name for name in
                    ("gallery_feats.npz", "gallery_features.npz", "image_feats.npz", "image_features.npz")]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"Multiple archived gallery caches found: {existing}; select with --gallery-features")
    if explicit and not existing:
        raise FileNotFoundError(f"Gallery feature archive does not exist: {explicit}")
    if not existing:
        return None, None
    path = existing[0]
    with np.load(path, allow_pickle=False) as saved:
        aliases = [("image_feat", "image_tse_feat"), ("image_feats", "image_tse_feats"), ("global_features", "tse_features")]
        pairs = [(left, right) for left, right in aliases if left in saved and right in saved]
        if len(pairs) != 1:
            raise ValueError(f"{path}: expected exactly one global/TSE feature pair; available keys={saved.files}")
        left, right = pairs[0]
        features = {"image_feat": saved[left], "image_tse_feat": saved[right]}
    validate_features(features)
    print(f"Archived gallery cache loaded read-only: {path}; mandatory qg parity guard will validate row order/checkpoint")
    return features, str(path)


def extract_gallery_features(model, config, ann_path, image_root, device, batch_size=32, num_workers=2):
    """Reuse ps_eval_dataset and the exact existing validation transform, no train loader."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from misc.caption_dataset import ps_eval_dataset
    path = Path(ann_path)
    if not path.name.endswith("_reid.json"):
        raise ValueError("Repository ps_eval_dataset requires a <split>_reid.json annotation filename")
    size = config.experiment.input_resolution
    if isinstance(size, int):
        size = (size, size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ps_eval_dataset(str(path.parent), str(image_root), transform, path.name[:-len("_reid.json")],
                              max_words=config.experiment.max_words, text_length=config.experiment.text_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    global_parts, tse_parts = [], []
    model.eval()
    with torch.no_grad():
        for images in loader:
            global_feat, dense, attention = model.encode_image(images.to(device), return_dense=True)
            tse_feat = model.visual_emb_layer(attention, dense, training=False)
            global_parts.append(F.normalize(global_feat, dim=-1).cpu().numpy())
            tse_parts.append(F.normalize(tse_feat, dim=-1).cpu().numpy())
    result = {"image_feat": np.concatenate(global_parts), "image_tse_feat": np.concatenate(tse_parts)}
    validate_features(result)
    return result


def iter_similarity_blocks(text, gallery, gallery_views, device, block_size=256):
    """FP32 GEMMs; same 0.5 global + 0.5 TSE composition as online eval."""
    import torch
    if block_size < 1:
        raise ValueError("Similarity block size must be positive")
    images = torch.from_numpy(gallery["image_feat"]).to(device)
    image_tse = torch.from_numpy(gallery["image_tse_feat"]).to(device)
    prompted = "text_feat_view0" in text
    count = len(next(iter(text.values())))
    gallery_views = np.asarray(gallery_views)
    if len(gallery_views) != len(images) or not np.isin(gallery_views, [0, 1]).all():
        raise ValueError("Invalid gallery views/count for score construction")
    with torch.no_grad():
        for start in range(0, count, block_size):
            stop = min(start + block_size, count)
            block = np.empty((stop - start, len(images)), dtype=np.float32)
            for view in ((0, 1) if prompted else (None,)):
                suffix = f"_view{view}" if prompted else ""
                selected = gallery_views == view if prompted else np.ones(len(images), dtype=bool)
                if not selected.any():
                    continue
                index = torch.from_numpy(np.flatnonzero(selected)).to(device)
                global_text = torch.from_numpy(text[f"text_feat{suffix}"][start:stop]).to(device)
                tse_text = torch.from_numpy(text[f"text_tse_feat{suffix}"][start:stop]).to(device)
                scores = (global_text @ images[index].t() + tse_text @ image_tse[index].t()) / 2
                block[:, selected] = scores.cpu().numpy()
            yield start, stop, block


def verify_query_gallery(text, gallery, gallery_views, archived, device, block_size=256, tolerance=1e-5):
    validate_features(text)
    validate_features(gallery)
    expected = (len(next(iter(text.values()))), len(gallery["image_feat"]))
    if archived.ndim != 2 or archived.shape != expected:
        raise ValueError(f"Full recomputed/archived qg shape mismatch: expected {expected}, archive {archived.shape}")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("qg parity tolerance must be positive and finite")
    worst, different, total_error, worst_position = 0.0, 0, 0.0, None
    visited_rows, visited_elements = 0, 0
    for start, stop, block in iter_similarity_blocks(text, gallery, gallery_views, device, block_size):
        reference = np.asarray(archived[start:stop])
        if reference.shape != block.shape:
            raise ValueError(f"Recomputed/archived qg shape mismatch: {block.shape}/{reference.shape}")
        if start != visited_rows or not np.isfinite(reference).all() or not np.isfinite(block).all():
            raise ValueError("qg parity comparison encountered skipped rows or non-finite scores")
        delta = np.abs(block.astype(np.float64) - reference.astype(np.float64))
        local = float(delta.max())
        if local > worst:
            row, col = np.unravel_index(delta.argmax(), delta.shape)
            worst, worst_position = local, [int(start + row), int(col)]
        different += int(np.count_nonzero(delta))
        total_error += float(delta.sum())
        visited_rows = stop
        visited_elements += block.size
    if visited_rows != archived.shape[0] or visited_elements != archived.size:
        raise RuntimeError("qg parity guard did not inspect every archived matrix element")
    report = {"max_abs_error": worst, "mean_abs_error": total_error / archived.size,
              "different_elements": different, "max_error_position": worst_position,
              "compared_shape": list(archived.shape), "compared_elements": visited_elements,
              "tolerance_strict_less_than": tolerance, "passed": worst < tolerance}
    print("ARCHIVED QG PARITY " + json.dumps(report, sort_keys=True))
    if not report["passed"]:
        error = RuntimeError("STOP: recomputed qg differs from read-only archive; possible stochastic image-side "
                             f"Gumbel/MoE routing or config/checkpoint/batching mismatch. {report}. "
                             "No RNG searching, retry, archive overwrite, or replacement is permitted.")
        error.parity_report = report
        raise error
    return report


def write_bank_similarity(path, text, gallery, gallery_views, device, block_size=256):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite bank matrix {path}; choose a fresh --out directory")
    if "_archive_evidence" in {part.lower() for part in path.resolve().parts}:
        raise ValueError("Refusing to write into read-only _archive_evidence")
    shape = (len(next(iter(text.values()))), len(gallery["image_feat"]))
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    try:
        for start, stop, block in iter_similarity_blocks(text, gallery, gallery_views, device, block_size):
            output[start:stop] = block
        output.flush()
    finally:
        del output
    return np.load(path, mmap_mode="r", allow_pickle=False)


def text_fingerprint(texts):
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def feature_self_test():
    """Small actual-torch CPU test of extraction, parity and immutable caches."""
    import tempfile
    import torch
    from torch import nn
    import torch.nn.functional as F

    class MockTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 5)
            self.tse = nn.Linear(5, 9, bias=False)
            self.prompts = nn.Embedding(2, 5)
            self.image_calls = 0

        def encode_text(self, tokens, return_dense=False):
            assert return_dense
            dense = self.embedding(tokens)
            return dense.mean(1), dense, torch.zeros(tokens.shape[0], tokens.shape[1], tokens.shape[1])

        def encode_text_with_view_prompt(self, tokens, cams, return_dense=False):
            assert return_dense
            dense = self.embedding(tokens)
            dense = torch.cat((dense[:, :1], self.prompts(cams).unsqueeze(1), dense[:, 1:]), dim=1)
            return dense.mean(1), dense, torch.zeros(dense.shape[0], dense.shape[1], dense.shape[1])

        def view_prompt_token_ids(self, tokens):
            return torch.cat((tokens[:, :1], torch.zeros_like(tokens[:, :1]), tokens[:, 1:]), dim=1)

        def textual_emb_layer(self, attention, tokens, dense, training=False):
            assert not training and tokens.shape[:2] == dense.shape[:2]
            return self.tse(dense.mean(1))

        def encode_image(self, *args, **kwargs):
            self.image_calls += 1
            raise AssertionError("text feature extraction must not call encode_image")

    def tokenizer(texts, context_length):
        result = torch.zeros((len(texts), context_length), dtype=torch.long)
        for index, text in enumerate(texts):
            result[index, :3] = torch.tensor([1, 3 + int(text), 2])
        return result

    torch.manual_seed(2026)
    model = MockTextModel()
    texts = [str(index) for index in range(13)]
    results = {}
    for prompted in (False, True):
        before = torch.get_rng_state().clone()
        first = extract_text_features(model, texts, 8, "cpu", prompted, 4, tokenizer)
        second = extract_text_features(model, texts, 8, "cpu", prompted, 4, tokenizer)
        results[str(prompted)] = assert_identical_features(first, second)
        assert torch.equal(before, torch.get_rng_state())
        assert model.image_calls == 0
        if prompted:
            assert not np.array_equal(first["text_feat_view0"], first["text_feat_view1"])
        gallery = {"image_feat": F.normalize(torch.randn(7, 5), dim=-1).numpy(),
                   "image_tse_feat": F.normalize(torch.randn(7, 9), dim=-1).numpy()}
        views = np.arange(7) % 2
        reference = np.concatenate([block for _, _, block in iter_similarity_blocks(first, gallery, views, "cpu", 4)])
        parity = verify_query_gallery(first, gallery, views, reference, "cpu", 4)
        assert parity["max_abs_error"] == 0
        for incomplete in ({key: value[:-1] for key, value in first.items()},
                           {key: value[:1] for key, value in first.items()}):
            try:
                verify_query_gallery(incomplete, gallery, views, reference, "cpu", 4)
            except ValueError as exc:
                assert "Full recomputed/archived qg shape mismatch" in str(exc)
            else:
                raise AssertionError("Truncated text features passed full qg parity guard")
        nonfinite_reference = reference.copy()
        nonfinite_reference[-1, -1] = np.nan
        try:
            verify_query_gallery(first, gallery, views, nonfinite_reference, "cpu", 4)
        except ValueError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError("NaN archived score passed parity guard")
        corrupt = reference.copy()
        corrupt[2, 3] += 0.01
        try:
            verify_query_gallery(first, gallery, views, corrupt, "cpu", 4)
        except RuntimeError as exc:
            assert not exc.parity_report["passed"]
            results[str(prompted)]["negative_parity_test"] = exc.parity_report
        else:
            raise AssertionError("Mismatching archived matrix was not rejected")
    left, right = np.array([[0.0]], np.float32), np.array([[-0.0]], np.float32)
    try:
        assert_identical_features({"signed_zero": left}, {"signed_zero": right})
    except RuntimeError as exc:
        assert "first_positions=[[0, 0]]" in str(exc)
    else:
        raise AssertionError("Signed-zero bit mismatch was not detected")
    with tempfile.TemporaryDirectory(prefix="tagpr-s12-features-") as directory:
        folder = Path(directory)
        training_rows = [{"file_path": "G2APS/D1_0.jpg", "id": 0, "captions": ["one", "two"], "split": "train"}]
        (folder / "train_custom.json").write_text(json.dumps(training_rows), encoding="utf-8")
        (folder / "test_reid.json").write_text(json.dumps([{**training_rows[0], "split": "test"}]), encoding="utf-8")
        path, rows, info = discover_train_annotations(folder / "test_reid.json")
        assert path.name == "train_custom.json" and info["captions"] == 2 and len(rows) == 1
        saved = folder / "train_text_feats.npz"
        save_training_features(saved, first, {"test": 1})
        before = saved.read_bytes()
        save_training_features(saved, first, {"test": 1})
        assert before == saved.read_bytes()
        try:
            save_training_features(saved, first, {"test": 2})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Changed feature provenance overwrote saved cache")
        with (folder / "gallery_feats.npz").open("xb") as stream:
            np.savez(stream, **gallery)
        cached, cache_path = discover_gallery_features(folder)
        assert cache_path and all(np.array_equal(gallery[key], cached[key]) for key in gallery)
        bank = write_bank_similarity(folder / "bank.npy", first, gallery, views, "cpu", 4)
        assert np.array_equal(bank, reference)
        del bank
    summary = {"status": "PASS", "device": "cpu", "seed": 2026,
               "image_encoder_calls_during_text_extraction": model.image_calls,
               "consecutive_max_abs_error": 0.0, "prompted_and_unprompted": True,
               "qg_parity_max_abs_error": 0.0, "corrupt_qg_rejected": True,
               "truncated_text_features_rejected": True, "nan_archived_score_rejected": True,
               "signed_zero_mismatch_rejected": True, "existing_feature_cache_not_overwritten": True}
    print("S12 FEATURE MOCK SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary

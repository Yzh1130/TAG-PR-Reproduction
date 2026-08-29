import json
import subprocess
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
import torch.nn.functional as F

# import clip
from misc.utils import get_rank
from text_utils.tokenizer import tokenize


SIMILARITY_DUMP_BLOCK_SIZE = 1024
_ACTIVE_SIMILARITY_DUMP = ContextVar(
    "active_similarity_dump",
    default=None,
)


class _StreamingStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, values):
        import numpy as np

        values = np.asarray(values, dtype=np.float64)
        self.count += values.size
        self.total += values.sum(dtype=np.float64)
        self.total_square += np.square(values).sum(dtype=np.float64)
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))

    def result(self):
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": variance ** 0.5,
            "min": self.minimum,
            "max": self.maximum,
        }


def _dump_dtype(dtype_name):
    import numpy as np

    if dtype_name == "float16":
        return np.float16, torch.float16
    if dtype_name == "float32":
        return np.float32, torch.float32
    raise ValueError("dump_sim_dtype must be 'float16' or 'float32'")


def _write_tensor_npy(path, tensor, dtype_name):
    import numpy as np

    numpy_dtype, torch_dtype = _dump_dtype(dtype_name)
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=numpy_dtype,
        shape=tuple(tensor.shape),
    )
    stats = _StreamingStats()
    for start in range(0, tensor.shape[0], SIMILARITY_DUMP_BLOCK_SIZE):
        end = min(start + SIMILARITY_DUMP_BLOCK_SIZE, tensor.shape[0])
        block = tensor[start:end].detach().to(
            device="cpu",
            dtype=torch_dtype,
        ).numpy()
        output[start:end] = block
        stats.update(block)
    output.flush()
    del output
    return stats.result()


def _write_gallery_similarity_npy(
    path,
    global_features,
    tse_features,
    dtype_name,
):
    import numpy as np

    if global_features.ndim != 2:
        raise ValueError("global gallery features must be two-dimensional")
    if tse_features is not None:
        if tse_features.ndim != 2:
            raise ValueError("TSE gallery features must be two-dimensional")
        if global_features.shape[0] != tse_features.shape[0]:
            raise ValueError(
                "global and TSE features must contain the same number of "
                "gallery samples"
            )

    numpy_dtype, torch_dtype = _dump_dtype(dtype_name)
    gallery_size = global_features.shape[0]
    output = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=numpy_dtype,
        shape=(gallery_size, gallery_size),
    )
    stats = _StreamingStats()
    global_transposed = global_features.t()
    tse_transposed = tse_features.t() if tse_features is not None else None
    for start in range(0, gallery_size, SIMILARITY_DUMP_BLOCK_SIZE):
        end = min(start + SIMILARITY_DUMP_BLOCK_SIZE, gallery_size)
        block = global_features[start:end] @ global_transposed
        if tse_features is not None:
            block.add_(tse_features[start:end] @ tse_transposed).mul_(0.5)
        cpu_block = block.detach().to(
            device="cpu",
            dtype=torch_dtype,
        ).numpy()
        output[start:end] = cpu_block
        stats.update(cpu_block)
        del block, cpu_block
    output.flush()
    del output
    return stats.result()


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        return f"unavailable: {detail}"
    return result.stdout.strip()


class _SimilarityDumpSession:
    def __init__(
        self,
        model,
        output_dir,
        dtype_name,
        score_description,
        checkpoint_path,
        exp_name,
        logger,
    ):
        self.model = model.module if hasattr(model, "module") else model
        if output_dir is None:
            warnings.warn(
                "experiment directory unavailable; writing similarity dumps "
                "to the current working directory",
                RuntimeWarning,
                stacklevel=2,
            )
            self.output_dir = Path.cwd()
        else:
            self.output_dir = Path(output_dir)
        self.dtype_name = dtype_name
        self.score_description = score_description
        self.checkpoint_path = checkpoint_path
        self.exp_name = exp_name
        self.logger = logger
        self.global_features = []
        self.tse_features = []
        self.hooks = []
        self.dumped = False

    def start(self):
        visual = getattr(self.model, "visual", None)
        visual_tse = getattr(self.model, "visual_emb_layer", None)
        if not isinstance(visual, torch.nn.Module):
            raise RuntimeError("similarity dumping requires model.visual")
        if not isinstance(visual_tse, torch.nn.Module):
            raise RuntimeError(
                "similarity dumping requires model.visual_emb_layer"
            )
        self.hooks.append(visual.register_forward_hook(self._capture_global))
        self.hooks.append(visual_tse.register_forward_hook(self._capture_tse))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _capture_global(self, module, inputs, output):
        del module, inputs
        features = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(features) or features.ndim != 2:
            raise RuntimeError("model.visual did not return 2D global features")
        self.global_features.append(F.normalize(features.detach(), dim=-1))

    def _capture_tse(self, module, inputs, output):
        del module, inputs
        if not torch.is_tensor(output) or output.ndim != 2:
            raise RuntimeError(
                "model.visual_emb_layer did not return 2D TSE features"
            )
        self.tse_features.append(F.normalize(output.detach(), dim=-1))

    def dump(self, similarities, raw_result):
        if self.dumped:
            return
        self.dumped = True
        if not self.global_features:
            raise RuntimeError("evaluation produced no gallery global features")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        global_features = torch.cat(self.global_features, dim=0)
        self.global_features.clear()
        tse_features = (
            torch.cat(self.tse_features, dim=0)
            if self.tse_features
            else None
        )
        self.tse_features.clear()
        gallery_size = global_features.shape[0]
        if similarities.ndim != 2 or similarities.shape[1] != gallery_size:
            raise RuntimeError(
                "similarity matrix gallery dimension does not match captured "
                "gallery features"
            )

        composition_matches = tse_features is not None
        if not composition_matches:
            warning = (
                "WARNING: sim_gg uses only the global image branch and does "
                "not match sim_qg because no gallery TSE features were produced"
            )
            warnings.warn(warning, RuntimeWarning, stacklevel=2)
            if self.logger is not None:
                self.logger.warning(warning)

        started = time.perf_counter()
        qg_stats = _write_tensor_npy(
            self.output_dir / "sim_qg.npy",
            similarities,
            self.dtype_name,
        )
        gg_stats = _write_gallery_similarity_npy(
            self.output_dir / "sim_gg.npy",
            global_features,
            tse_features,
            self.dtype_name,
        )
        elapsed = time.perf_counter() - started

        if composition_matches:
            gg_description = (
                "sim_gg = 0.5 * cosine(image_global, image_global) + "
                "0.5 * cosine(image_TSE, image_TSE); the two image branches "
                "are L2-normalized and use the same weights as sim_qg."
            )
        else:
            gg_description = (
                "sim_gg uses only the global image branch and differs from "
                "the sim_qg composition because gallery TSE features were "
                "not available."
            )
        checkpoint = (
            str(Path(self.checkpoint_path).expanduser().resolve())
            if self.checkpoint_path is not None
            else None
        )
        metadata = {
            "n_query": int(similarities.shape[0]),
            "n_gallery": int(similarities.shape[1]),
            "dtype": self.dtype_name,
            "sim_qg_scoring": self.score_description,
            "sim_gg_scoring": gg_description,
            "sim_gg_composition_matches_sim_qg": composition_matches,
            "checkpoint_path": checkpoint,
            "checkpoint_note": (
                "loaded from --resume"
                if checkpoint is not None
                else "in-memory training weights; no checkpoint path"
            ),
            "exp_name": self.exp_name,
            "git_commit": _git_commit(),
            "raw_metrics": {
                "R@1": raw_result["r1"],
                "R@5": raw_result["r5"],
                "R@10": raw_result["r10"],
                "mAP": raw_result["mAP"],
            },
            "sim_qg_stats": qg_stats,
            "sim_gg_stats": gg_stats,
            "dump_elapsed_sec": elapsed,
        }
        with (self.output_dir / "sim_meta.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

        if self.logger is not None:
            self.logger.info(
                "SIMILARITY DUMP dtype=%s qg_shape=%s gg_shape=%s "
                "elapsed=%.3fs output_dir=%s",
                self.dtype_name,
                tuple(similarities.shape),
                (gallery_size, gallery_size),
                elapsed,
                self.output_dir,
            )
            self.logger.info(
                "SIM_QG STATS mean=%.6f std=%.6f min=%.6f max=%.6f",
                qg_stats["mean"],
                qg_stats["std"],
                qg_stats["min"],
                qg_stats["max"],
            )
            self.logger.info(
                "SIM_GG STATS mean=%.6f std=%.6f min=%.6f max=%.6f",
                gg_stats["mean"],
                gg_stats["std"],
                gg_stats["min"],
                gg_stats["max"],
            )


@contextmanager
def similarity_dump_context(
    model,
    output_dir,
    dtype_name,
    score_description,
    checkpoint_path=None,
    exp_name=None,
    logger=None,
):
    """Capture gallery representations and dump the first raw score matrix."""
    if get_rank() != 0:
        yield
        return
    session = _SimilarityDumpSession(
        model,
        output_dir,
        dtype_name,
        score_description,
        checkpoint_path,
        exp_name,
        logger,
    )
    session.start()
    token = _ACTIVE_SIMILARITY_DUMP.set(session)
    try:
        yield
    finally:
        _ACTIVE_SIMILARITY_DUMP.reset(token)
        session.close()


@torch.no_grad()
def test(model, data_loader, max_length, device, output_dir=None):
    # switch to evaluate mode
    model.eval()

    dataset = data_loader.dataset
    texts = dataset.text
    num_text = len(texts)
    text_bs = 256

    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text = tokenize(text, context_length=max_length).to(device)
        text_feat = F.normalize(model.encode_text(text), dim=-1)
        text_feats.append(text_feat)
    text_feats = torch.cat(text_feats, dim=0)

    image_feats = []
    for image in data_loader:
        image = image.to(device)
        image_feat = F.normalize(model.encode_image(image), dim=-1)
        image_feats.append(image_feat)
    image_feats = torch.cat(image_feats, dim=0)

    sims_matrix = text_feats @ image_feats.t()
    eval_result = metric_eval(
        sims_matrix,
        dataset.img2person,
        dataset.txt2person,
        output_dir=output_dir,
    )

    return eval_result

# TSE evaluation
@torch.no_grad()
def test_tse(model, data_loader, max_length, device, output_dir=None):
    # switch to evaluate mode
    model.eval()

    dataset = data_loader.dataset
    texts = dataset.text
    num_text = len(texts)
    text_bs = 256

    text_feats = []
    text_tse_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text = tokenize(text, context_length=max_length).to(device)
        text_feat, text_feat_dense, attn_t = model.encode_text(text, return_dense=True)
        text_feat = F.normalize(text_feat, dim=-1)
        text_feats.append(text_feat)
        text_tse_feat = F.normalize(model.textual_emb_layer(attn_t, text, text_feat_dense, training=False), dim=-1)
        text_tse_feats.append(text_tse_feat)
    text_feats = torch.cat(text_feats, dim=0)
    text_tse_feats = torch.cat(text_tse_feats, dim=0)

    image_feats = []
    image_tse_feats = []
    for image in data_loader:
        image = image.to(device)
        image_feat, image_feat_dense, attn_i = model.encode_image(image, return_dense=True)
        image_feat = F.normalize(image_feat, dim=-1)
        image_feats.append(image_feat)
        image_tse_feat = F.normalize(model.visual_emb_layer(attn_i, image_feat_dense, training=False), dim=-1)
        image_tse_feats.append(image_tse_feat)
    image_feats = torch.cat(image_feats, dim=0)
    image_tse_feats = torch.cat(image_tse_feats, dim=0)

    sims_matrix = (text_feats @ image_feats.t() + text_tse_feats @ image_tse_feats.t()) / 2
    eval_result = metric_eval(
        sims_matrix,
        dataset.img2person,
        dataset.txt2person,
        output_dir=output_dir,
    )

    return eval_result


@torch.no_grad()
def metric_eval(scores_t2i, img2person, txt2person, output_dir=None):
    device = scores_t2i.device
    img2person = img2person.to(device)
    txt2person = txt2person.to(device)

    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    if get_rank() == 0:
        import json
        output_file = []
        index_output = index.cpu().numpy()[:, :10]
        for i, match in enumerate(matches):
            match = match.cpu().numpy().tolist()
            output_file.append({
                i: {
                    'match': match[:10],
                    'pred': index_output[i].tolist(),
                }
            })
        output = json.dumps(output_file, indent=4)
        if output_dir is None:
            warnings.warn(
                "experiment directory unavailable; writing output.json "
                "to the current working directory",
                RuntimeWarning,
                stacklevel=2,
            )
            output_path = Path('./output.json')
        else:
            output_path = Path(output_dir) / 'output.json'
        with open(output_path, 'w') as f:
            f.write(output)

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    real_num = matches.sum(dim=-1)
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long).to(device)
    tmp_cmc /= order
    tmp_cmc *= matches
    AP = tmp_cmc.sum(dim=-1) / real_num
    mAP = AP.mean() * 100.0

    eval_result = {'r1': ir1,
                   'r5': ir5,
                   'r10': ir10,
                   'r_mean': ir_mean,
                   'mAP': mAP.item()
                   }

    dump_session = _ACTIVE_SIMILARITY_DUMP.get()
    if dump_session is not None:
        # The full-rank tensors are no longer needed once raw metrics exist.
        # Release them before constructing gallery-gallery blocks so dumping
        # does not stack its workspace on top of metric_eval's peak memory.
        del index, pred_person, matches, real_num, tmp_cmc, order, AP
        dump_session.dump(scores_t2i, eval_result)

    return eval_result

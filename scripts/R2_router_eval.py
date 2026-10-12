"""Standalone image-level router evaluation; no retrieval metrics or training."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._analysis_audit import git_head, sha256_16, unique_path
from scripts._s12_features import (
    _read_records, extract_gallery_features, load_model, resolve_checkpoint,
)
from tools._analysis_common import DOMAIN_NAMES, annotation_from_records


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Existing model YAML configuration')
    parser.add_argument('--backbone', choices=('ViT-B/16', 'ViT-L/14'))
    parser.add_argument('--resume', help='Existing trained checkpoint')
    parser.add_argument('--anno', help='Image-record annotation: <split>_reid.json')
    parser.add_argument('--image-root', help='Image root; defaults to config.image_dir')
    parser.add_argument('--split', choices=('gallery', 'train'), default='gallery')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--n-gumbel', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', help='New result JSON filename or output directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-logits', action='store_true', default=False)
    parser.add_argument('--self-test', action='store_true', help='200 images x 100 dimensions; CPU mock only')
    return parser


def classification_stats(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.ndim != 1 or labels.shape != predictions.shape or not len(labels):
        raise ValueError('classification labels/predictions must be matching nonempty vectors')
    if not np.isin(labels, (0, 1)).all() or not np.isin(predictions, (0, 1)).all():
        raise ValueError('router classification requires labels/predictions in {0,1}')
    matrix = np.zeros((2, 2), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    counts = matrix.sum(axis=1)
    recall = [float(matrix[i, i] / counts[i]) if counts[i] else None for i in range(2)]
    return dict(accuracy=float(np.mean(labels == predictions)),
                confusion_matrix=matrix.tolist(), confusion_axes='row=true, column=predicted',
                class_names=['aerial', 'ground'], class_counts=counts.tolist(),
                class_recall=recall,
                balanced_accuracy=float(np.mean([v for v in recall if v is not None])))


def verify_annotation(records):
    ann = annotation_from_records(records, enforce_anchors=False)
    table = {domain: {str(cam): int(np.sum((ann.gallery_domains == domain) &
                                          (ann.gallery_views == cam))) for cam in (0, 1)}
             for domain in DOMAIN_NAMES}
    print('CAM_ID x SOURCE (0=aerial, 1=ground)')
    for domain, counts in table.items():
        print(f"  {domain}: cam_id=0 {counts['0']}, cam_id=1 {counts['1']}")
    for domain, expected in (('Market', 1), ('PRAI-1581', 0), ('uavhuman', 0)):
        values = ann.gallery_views[ann.gallery_domains == domain]
        if len(values) and not np.all(values == expected):
            raise ValueError(f'pure-view source {domain} must have cam_id={expected}')
    # The model uses the dataset's cam_id (default 1), not an independently
    # inferred label. Require the repository dataset and verified path rule to agree.
    actual_cams = np.array([record.get('cam_id', 1) for record in records])
    if not np.array_equal(actual_cams, ann.gallery_views):
        raise ValueError('annotation cam_id/default disagrees with source/path view semantics')
    source = ROOT / 'misc' / 'caption_dataset.py'
    lines = source.read_text(encoding='utf-8').splitlines()
    evidence = [{'file': 'misc/caption_dataset.py', 'line': i + 1, 'code': line.strip()}
                for i, line in enumerate(lines)
                if 'aerial pairs (cam_id==0)' in line or 'ground pairs (cam_id==1)' in line
                or "self.img_cam_ids.append(ann.get('cam_id', 1))" in line]
    if len(evidence) != 3:
        raise RuntimeError('dataset cam_id evidence changed; inspect semantics before evaluating')
    return ann, table, evidence


class RouterCapture:
    """Capture the original Gumbel call's output; never draw a second sample."""

    def __init__(self, model):
        self.blocks = [(name, module) for name, module in model.named_modules()
                       if hasattr(module, 'img_router') and hasattr(module, 'eval_router_mode')]
        if not self.blocks:
            raise ValueError('model has no image-level MoE routers (disable_moe may be true)')
        self.logits = {name: [] for name, _ in self.blocks}
        self.predictions = {name: [] for name, _ in self.blocks}
        self.pending = {}
        self.handles = []
        self.mode = None
        self.calls = 0

    def _hook(self, name):
        def hook(module, inputs, output):
            if output.ndim != 2 or output.shape[1] != 2:
                raise RuntimeError(f'{name}: expected [B,2] router logits, got {tuple(output.shape)}')
            array = output.detach().float().cpu().numpy().copy()
            if not np.isfinite(array).all():
                raise RuntimeError(f'{name}: nonfinite router logits')
            self.logits[name].append(array)
            if self.mode == 'argmax':
                self.predictions[name].append(array.argmax(axis=1))
            else:
                self.pending[id(output)] = name
        return hook

    @contextmanager
    def capture(self, mode):
        import torch.nn.functional as functional
        if mode not in ('gumbel', 'argmax'):
            raise ValueError('router capture mode must be gumbel/argmax')
        old_modes = [block.eval_router_mode for _, block in self.blocks]
        self.mode = mode
        self.logits = {name: [] for name, _ in self.blocks}
        self.predictions = {name: [] for name, _ in self.blocks}
        self.pending, self.calls = {}, 0
        original = functional.gumbel_softmax

        def observed(logits, *args, **kwargs):
            result = original(logits, *args, **kwargs)
            name = self.pending.pop(id(logits), None)
            if name is None:
                raise RuntimeError('unassociated Gumbel call during image router evaluation')
            if not kwargs.get('hard', False):
                raise RuntimeError('image evaluation unexpectedly used soft Gumbel routing')
            self.predictions[name].append(result.detach().argmax(-1).cpu().numpy().copy())
            self.calls += 1
            return result

        try:
            for name, block in self.blocks:
                block.eval_router_mode = mode
                self.handles.append(block.img_router.register_forward_hook(self._hook(name)))
            with patch.object(functional, 'gumbel_softmax', side_effect=observed):
                yield self
            if self.pending:
                raise RuntimeError('router logits were captured without their original Gumbel call')
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles = []
            for (_, block), old_mode in zip(self.blocks, old_modes):
                block.eval_router_mode = old_mode

    def arrays(self, count):
        result = {}
        for name, _ in self.blocks:
            if not self.logits[name] or not self.predictions[name]:
                raise RuntimeError(f'{name}: router did not execute')
            logits = np.concatenate(self.logits[name])
            predictions = np.concatenate(self.predictions[name])
            if logits.shape != (count, 2) or predictions.shape != (count,):
                raise RuntimeError(f'{name}: captured sample count/order does not cover annotation')
            result[name] = dict(logits=logits, predictions=predictions)
        return result


def evaluate_router_passes(model, extract, ann, n_gumbel):
    """One deterministic full image pass, then n complete stochastic passes."""
    if n_gumbel < 1:
        raise ValueError('n-gumbel must be positive')
    model.eval()
    capture = RouterCapture(model)
    with capture.capture('argmax'):
        extract()
    deterministic = capture.arrays(len(ann.gallery_views))
    if capture.calls:
        raise RuntimeError('argmax image pass called Gumbel-Softmax')
    sampled = {name: [] for name in deterministic}
    call_counts = []
    for _ in range(n_gumbel):
        with capture.capture('gumbel'):
            extract()
        values = capture.arrays(len(ann.gallery_views))
        call_counts.append(capture.calls)
        for name in sampled:
            sampled[name].append(values[name]['predictions'])
    blocks = {}
    for name, values in deterministic.items():
        draws = np.stack(sampled[name])
        by_source = {}
        for domain in DOMAIN_NAMES:
            mask = ann.gallery_domains == domain
            if mask.any():
                by_source[domain] = classification_stats(ann.gallery_views[mask], values['predictions'][mask])
        counts0 = (draws == 0).sum(axis=0)
        modal_disagreement = 1.0 - np.maximum(counts0, n_gumbel - counts0) / n_gumbel
        any_flip = np.any(draws != draws[0], axis=0)
        per_draw = [classification_stats(ann.gallery_views, draw) for draw in draws]
        accuracies = [result['accuracy'] for result in per_draw]
        sorted_logits = np.sort(values['logits'], axis=1)
        margins = sorted_logits[:, 1] - sorted_logits[:, 0]
        blocks[name] = dict(
            deterministic_argmax=classification_stats(ann.gallery_views, values['predictions']),
            by_source=by_source,
            per_image_logits=values['logits'].tolist(),
            logit_margin_quantiles={str(q): float(np.quantile(margins, q)) for q in (.005, .01, .05, .5)},
            gumbel=dict(accuracy_mean=float(np.mean(accuracies)), accuracy_std=float(np.std(accuracies)),
                        per_draw_metrics=per_draw,
                        predictions_by_draw=draws.tolist(),
                        per_image_flip_rate=modal_disagreement.tolist(),
                        flip_rate_definition='1 - max(class_count_0,class_count_1)/n_gumbel per image',
                        mean_per_image_flip_rate=float(modal_disagreement.mean()),
                        per_image_any_flip=any_flip.tolist(), fraction_images_ever_flipped=float(any_flip.mean()),
                        disagreement_with_argmax=float(np.mean(draws != values['predictions'][None, :]))))
    names = list(blocks)
    return dict(main_report_block=names[0], block_order=names, blocks=blocks,
                n_images=len(ann.gallery_views), n_gumbel=n_gumbel,
                extraction_passes=1 + n_gumbel, gumbel_calls_per_pass=call_counts,
                sampling_protocol='complete image forwards; original Gumbel call observed once; no extra sampling'), deterministic


def _output_path(out):
    path = Path(out).expanduser().resolve()
    if any(part.lower() in ('_archive_evidence', '_dump') or part.lower().startswith('_e1')
           for part in path.parts):
        raise ValueError('router outputs must use a new directory outside existing evidence trees')
    if path.suffix.lower() == '.json':
        directory, name = path.parent, path.name
    else:
        directory, name = path, 'R2_router_eval.json'
    directory.mkdir(parents=True, exist_ok=True)
    return unique_path(directory, name)


def save_report(report, out, deterministic, save_logits=False):
    path = _output_path(out)
    if save_logits:
        arrays = np.stack([deterministic[name]['logits'] for name in report['block_order']])
        logits_path = unique_path(path.parent, path.stem + '_logits.npy')
        with logits_path.open('xb') as stream:
            np.save(stream, arrays, allow_pickle=False)
        report['logits_npy'] = dict(path=str(logits_path), shape=list(arrays.shape),
                                  dtype=str(arrays.dtype), axes=['MoE block', 'image', 'class'])
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(f'ROUTER EVALUATION COMPLETE JSON={path}')
    return path


def mock_run(seed=0, n_gumbel=20, batch_size=32):
    """All synthetic values are explicitly labelled mock; no accuracy target."""
    import torch
    from torch import nn
    import torch.nn.functional as functional
    torch.manual_seed(seed)
    rows = []
    for i in range(200):
        domain = DOMAIN_NAMES[i % len(DOMAIN_NAMES)]
        cam = 1 if domain == 'Market' else 0 if domain in ('PRAI-1581', 'uavhuman') else (i // 5) % 2
        prefix = 'D' if cam == 0 else 'S'
        filename = f'{prefix}{i}_mock.jpg' if domain == 'G2APS' else f'mock_{i}.jpg'
        rows.append(dict(file_path=f'{domain}/{filename}', id=i, cam_id=cam, captions=[f'mock {i}']))
    ann, table, evidence = verify_annotation(rows)

    class MockBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_router = nn.Linear(100, 2)
            self.eval_router_mode = 'gumbel'

        def forward(self, x):
            logits = self.img_router(x)
            if not self.training and self.eval_router_mode == 'argmax':
                probs = functional.one_hot(logits.argmax(-1), 2).to(logits.dtype)
            else:
                probs = functional.gumbel_softmax(logits, hard=not self.training)
            return x + (probs[:, :1] - .5) * .25

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([MockBlock(), MockBlock()])
            self.passes = 0

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = MockModel().eval()
    images = torch.randn(200, 100)

    def extract():
        model.passes += 1
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                model(images[start:start + batch_size])

    report, deterministic = evaluate_router_passes(model, extract, ann, n_gumbel)
    if model.passes != 1 + n_gumbel:
        raise AssertionError('mock did not execute every full image pass')
    expected_calls = 2 * ((len(images) + batch_size - 1) // batch_size)
    if report['gumbel_calls_per_pass'] != [expected_calls] * n_gumbel:
        raise AssertionError('mock captured extra/missing Gumbel calls')
    report.update(mock=True, mock_image_shape=list(images.shape), cam_source_table=table,
                  cam_id_evidence=evidence, seed=seed)
    print(f'MOCK ROUTER PASS images={len(images)} dimensions=100 routers=2 full_passes={model.passes}')
    return report, deterministic


def run_real(args):
    import torch
    checkpoint = resolve_checkpoint(args.resume)
    records, annotation_info = _read_records(args.anno, split='train' if args.split == 'train' else 'test')
    ann, table, evidence = verify_annotation(records)
    anno_path = Path(args.anno)
    expected_name = 'train_reid.json' if args.split == 'train' else 'test_reid.json'
    if anno_path.name != expected_name:
        raise ValueError(f'--split {args.split} requires {expected_name} for repository eval dataset reuse')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get('model'), dict):
        raise ValueError('resume checkpoint must contain a model state dict')
    classifier = state['model'].get('classifier.weight')
    num_classes = int(classifier.shape[0]) if classifier is not None else 1
    del state
    model, config, provenance = load_model(args.config, checkpoint, args.backbone, args.device,
                                          num_classes, args.seed)
    started = time.perf_counter()

    def extract():
        extract_gallery_features(model, config, args.anno, args.image_root or config.image_dir,
                                 args.device, args.batch_size, args.num_workers)

    report, deterministic = evaluate_router_passes(model, extract, ann, args.n_gumbel)
    report.update(mock=False, feature_provenance=provenance, annotation=annotation_info,
                  cam_source_table=table, cam_id_evidence=evidence,
                  elapsed_sec=time.perf_counter() - started,
                  inputs={role: dict(path=str(Path(path).resolve()), sha256_16=sha256_16(path))
                          for role, path in (('checkpoint', checkpoint), ('annotation', args.anno), ('config', args.config))})
    return report, deterministic


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1 or args.num_workers < 0 or args.n_gumbel < 1:
            raise ValueError('batch-size and n-gumbel must be positive; num-workers must be nonnegative')
        if args.self_test:
            report, deterministic = mock_run(args.seed, args.n_gumbel, args.batch_size)
        else:
            missing = [name for name in ('config', 'resume', 'anno', 'out') if not getattr(args, name)]
            if missing:
                raise ValueError('required arguments: ' + ', '.join('--' + name for name in missing))
            report, deterministic = run_real(args)
        report.update(argv=list(sys.argv if argv is None else [str(Path(__file__)), *argv]),
                      arguments=vars(args), seed=args.seed, git_HEAD=git_head(ROOT))
        for name in report['block_order']:
            block = report['blocks'][name]
            print(f"ROUTER {name} argmax_accuracy={block['deterministic_argmax']['accuracy']:.9f} "
                  f"gumbel_accuracy_mean={block['gumbel']['accuracy_mean']:.9f} "
                  f"mean_flip_rate={block['gumbel']['mean_per_image_flip_rate']:.9f}")
        if args.out:
            save_report(report, args.out, deterministic, args.save_logits)
        elif args.save_logits:
            raise ValueError('--save-logits requires --out')
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f'R2_router_eval error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

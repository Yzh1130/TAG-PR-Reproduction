"""CPU checks run the real base_transformer source (unused einops import omitted)."""

import ast
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import importlib.util
import io
import json
import logging
from pathlib import Path
import subprocess
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import R2_router_eval as router_eval


def load_base_source(original=False):
    """Execute all real source classes; this is not a substitute router formula.

    The reference is read verbatim from the frozen 8bf2616 Git object. The
    unused einops import is the only dependency omitted in either version.
    """
    path = ROOT / 'model' / 'base_transformer.py'
    source = subprocess.check_output(
        ['git', 'show', '8bf2616403145f5a0e64747ddf9914e0a9e0fcdf:model/base_transformer.py'],
        cwd=ROOT).decode('utf-8') if original else path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ImportFrom) or node.module != 'einops']
    ast.fix_missing_locations(tree)
    module = types.ModuleType('base_transformer_reference' if original else 'base_transformer_under_test')
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


class RouterEvalModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.reference = load_base_source(), load_base_source(original=True)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def block(self, source=None, mode='gumbel'):
        kwargs = {} if source is self.reference else {'eval_router_mode': mode}
        return (source or self.base).ResidualAttentionBlockMoE(
            8, 2, num_of_experts=3, num_of_selected_experts=2, **kwargs)

    def compare_reference(self, training):
        torch.manual_seed(72)
        current = self.block().train(training)
        reference = self.block(self.reference).train(training)
        reference.load_state_dict(current.state_dict())
        x = torch.randn(5, 7, 8)
        cams = torch.tensor([0, 1, 0, 1, 0, 1, 0])
        state = torch.get_rng_state()
        original = F.gumbel_softmax
        records = []

        def observed(logits, *args, **kwargs):
            records.append((logits.detach().clone(), kwargs.copy()))
            return original(logits, *args, **kwargs)

        with patch.object(F, 'gumbel_softmax', side_effect=observed):
            expected = reference(x, cam_id=cams, training=training)
        expected_state, expected_records = torch.get_rng_state(), records[:]
        records.clear()
        torch.set_rng_state(state)
        with patch.object(F, 'gumbel_softmax', side_effect=observed):
            actual = current(x, cam_id=cams, training=training)
        self.assertEqual(len(records), len(expected_records))
        self.assertEqual(len(records), 1)
        self.assertTrue(torch.equal(records[0][0], expected_records[0][0]))
        self.assertEqual(records[0][1], expected_records[0][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_state))
        for left, right in zip(actual, expected):
            self.assertTrue(torch.equal(left, right) if torch.is_tensor(left) else left == right)
        print(f'F3 reference training={training}: Gumbel calls=1, inputs/output/RNG torch.equal=True')

    def test_default_gumbel_matches_original_eval_and_training(self):
        self.compare_reference(False)
        self.compare_reference(True)

    def test_argmax_uses_matching_onehot_dtype_shape_and_no_gumbel(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(39)
            block = self.block(mode='argmax').to(dtype).eval()
            x = torch.randn(5, 7, 8, dtype=dtype)
            logits, probs = [], []
            hook = block.img_router.register_forward_hook(lambda m, i, o: logits.append(o.detach().clone()))
            original = torch.argmax

            def observed(values, *args, **kwargs):
                if values.ndim == 2 and values.shape == (7, 2):
                    probs.append(values.detach().clone())
                return original(values, *args, **kwargs)

            state = torch.get_rng_state()
            try:
                with patch.object(F, 'gumbel_softmax', side_effect=AssertionError('argmax sampled')):
                    with patch.object(torch, 'argmax', side_effect=observed):
                        block(x, training=False)
            finally:
                hook.remove()
            expected = F.one_hot(logits[0].argmax(-1), 2).to(dtype)
            self.assertEqual(probs[0].dtype, dtype)
            self.assertEqual(probs[0].shape, (7, 2))
            self.assertTrue(torch.equal(probs[0], expected))
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
        print('F3 argmax: onehot torch.equal=True, dtype/shape preserved, Gumbel calls=0, RNG unchanged')

    def test_training_modes_are_bitwise_identical(self):
        torch.manual_seed(105)
        gumbel = self.block().train()
        argmax = copy.deepcopy(gumbel)
        argmax.eval_router_mode = 'argmax'
        x = torch.randn(5, 7, 8)
        cams = torch.arange(7) % 2
        state = torch.get_rng_state()
        expected = gumbel(x, cam_id=cams, training=True)
        expected_state = torch.get_rng_state()
        torch.set_rng_state(state)
        actual = argmax(x, cam_id=cams, training=True)
        for left, right in zip(actual, expected):
            self.assertTrue(torch.equal(left, right) if torch.is_tensor(left) else left == right)
        self.assertTrue(torch.equal(expected_state, torch.get_rng_state()))
        print('F3 training gumbel/argmax attributes: all outputs/RNG torch.equal=True')

    def test_module_training_with_false_explicit_flag_still_uses_gumbel(self):
        block = self.block(mode='argmax').train()
        with patch.object(F, 'gumbel_softmax', wraps=F.gumbel_softmax) as call:
            block(torch.randn(5, 7, 8), training=False)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.kwargs, {'hard': True})

    def test_constructor_mode_passes_to_each_real_moe_block(self):
        for mode in ('gumbel', 'argmax'):
            config = types.SimpleNamespace(expert_blocks=[0, 2], num_of_experts=3,
                num_of_selected_experts=2, experiment=types.SimpleNamespace(
                disable_moe=False, view_ratio=1., ortho_ratio=1., eval_router_mode=mode))
            transformer = self.base.Transformer_MoE(config, width=8, layers=3, heads=2).eval()
            self.assertEqual(transformer.resblocks[0].eval_router_mode, mode)
            self.assertEqual(transformer.resblocks[2].eval_router_mode, mode)
            self.assertFalse(hasattr(transformer.resblocks[1], 'eval_router_mode'))
            with torch.no_grad():
                result = transformer(torch.randn(5, 7, 8), training=False)
            self.assertEqual(result[0].shape, (4, 7, 8))
        with self.assertRaisesRegex(ValueError, 'eval_router_mode'):
            self.block(mode='invalid')

    def test_f5_mock_200_by_100_all_blocks_stats_and_save(self):
        with redirect_stdout(io.StringIO()):
            report, deterministic = router_eval.mock_run(seed=123, n_gumbel=20, batch_size=32)
        self.assertEqual(report['mock_image_shape'], [200, 100])
        self.assertEqual(report['extraction_passes'], 21)
        self.assertEqual(report['gumbel_calls_per_pass'], [14] * 20)
        self.assertEqual(len(report['blocks']), 2)
        for name, block in report['blocks'].items():
            preds = deterministic[name]['predictions']
            matrix = np.asarray(block['deterministic_argmax']['confusion_matrix'])
            self.assertEqual(int(matrix.sum()), 200)
            self.assertEqual(sum(block['deterministic_argmax']['class_counts']), 200)
            draws = np.asarray(block['gumbel']['predictions_by_draw'])
            self.assertEqual(draws.shape, (20, 200))
            counts = (draws == 0).sum(0)
            expected = 1 - np.maximum(counts, 20-counts) / 20
            np.testing.assert_array_equal(expected, block['gumbel']['per_image_flip_rate'])
            self.assertTrue(all(0 <= value <= 1 for value in expected))
            self.assertEqual(set(block['by_source']), set(router_eval.DOMAIN_NAMES))
            logits = np.asarray(block['per_image_logits'])
            np.testing.assert_array_equal(logits.argmax(-1), preds)
            margins = np.diff(np.sort(logits, axis=1), axis=1)[:, 0]
            self.assertEqual(block['logit_margin_quantiles']['0.005'], float(np.quantile(margins, .005)))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = router_eval.save_report(report, Path(directory)/'result.json', deterministic, True)
            saved = json.loads(path.read_text(encoding='utf-8'))
            values = np.load(saved['logits_npy']['path'], allow_pickle=False)
            self.assertEqual(values.shape, (2, 200, 2))
            self.assertEqual(values.dtype, np.float32)
            second = router_eval.save_report(report, path, deterministic, False)
            self.assertNotEqual(second, path)
        print('F5 MOCK 200x100: 2 routers, 21 full forwards, 14 original Gumbel calls/draw; stats/source/save/no-overwrite PASS')

    def test_capture_is_bitwise_rng_neutral(self):
        block = self.block().eval()
        x = torch.randn(5, 7, 8)
        state = torch.get_rng_state()
        expected = block(x, training=False)[0]
        expected_state = torch.get_rng_state()
        torch.set_rng_state(state)
        capture = router_eval.RouterCapture(block)
        with capture.capture('gumbel'):
            actual = block(x, training=False)[0]
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_state, torch.get_rng_state()))
        self.assertEqual(capture.calls, 1)
        self.assertEqual(capture.arrays(7)['']['logits'].shape, (7, 2))

    def test_missing_class_and_bad_input_errors(self):
        result = router_eval.classification_stats([0, 0], [0, 1])
        self.assertIsNone(result['class_recall'][1])
        self.assertEqual(result['balanced_accuracy'], result['class_recall'][0])
        with self.assertRaisesRegex(ValueError, 'labels/predictions'):
            router_eval.classification_stats([0], [0, 1])
        with self.assertRaisesRegex(ValueError, 'no image-level'):
            router_eval.RouterCapture(torch.nn.Linear(100, 2))
        with self.assertRaisesRegex(ValueError, 'cam_id disagrees'):
            router_eval.verify_annotation([dict(file_path='Market/x.jpg', cam_id=0, id=0, captions=['x'])])
        with self.assertRaisesRegex(ValueError, 'cam_id/default'):
            router_eval.verify_annotation([dict(file_path='PRAI-1581/x.jpg', id=0, captions=['x'])])
        with self.assertRaisesRegex(ValueError, 'outside existing evidence'):
            router_eval._output_path(ROOT/'_dump'/'router.json')

    def test_help_and_cli_validation(self):
        script = ROOT/'scripts'/'R2_router_eval.py'
        result = subprocess.run([sys.executable, '-B', str(script), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--config', '--backbone', '--resume', '--anno', '--image-root', '--split',
                     '--batch-size', '--num-workers', '--n-gumbel', '--seed', '--out', '--device', '--save-logits'):
            self.assertIn(flag, result.stdout)
        for argv, expected in (([], 'required arguments'), (['--n-gumbel', '0'], 'must be positive'),
                               (['--num-workers', '-1'], 'nonnegative')):
            result = subprocess.run([sys.executable, '-B', str(script), *argv], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn(expected, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', str(script), '--self-test', '--n-gumbel', '1',
                                     '--out', directory, '--save-logits'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('MOCK ROUTER PASS', result.stdout)

    def test_f5_run_real_orchestration_200_by_100_both_splits(self):
        # Only I/O/model-loading/extraction dependencies are patched. Every
        # captured forward executes two real repository MoE blocks on CPU.
        rows = []
        for i in range(200):
            domain = router_eval.DOMAIN_NAMES[i % 5]
            cam = 1 if domain == 'Market' else 0 if domain in ('PRAI-1581', 'uavhuman') else (i // 5) % 2
            prefix = 'D' if cam == 0 else 'S'
            filename = f'{prefix}{i}_mock.jpg' if domain == 'G2APS' else f'mock_{i}.jpg'
            rows.append(dict(file_path=f'{domain}/{filename}', id=i, cam_id=cam, captions=[f'mock {i}']))
        base = self.base

        class ImageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([
                    base.ResidualAttentionBlockMoE(100, 4, num_of_experts=3,
                                                   num_of_selected_experts=2) for _ in range(2)])
                self.classifier = torch.nn.Linear(100, 11)

            def encode_image(self, images, return_dense=False):
                x = torch.stack((images, images * .5, images * -.25))
                for block in self.blocks:
                    x = block(x, training=False)[0]
                return x[0], x.permute(1, 0, 2), None

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint, config_path = directory/'mock.pth', directory/'config.yaml'
            config_path.write_text('mock: true\n', encoding='utf-8')
            torch.save({'model': {'classifier.weight': torch.zeros(11, 100)}}, checkpoint)
            for split, filename in (('gallery', 'test_reid.json'), ('train', 'train_reid.json')):
                with self.subTest(split=split):
                    anno = directory/filename
                    anno.write_text(json.dumps(rows), encoding='utf-8')
                    args = router_eval.build_parser().parse_args([
                        '--config', str(config_path), '--resume', str(checkpoint), '--anno', str(anno),
                        '--image-root', str(directory/'images'), '--split', split, '--backbone', 'ViT-B/16',
                        '--device', 'cpu', '--n-gumbel', '20', '--seed', '2026', '--batch-size', '32',
                        '--num-workers', '2', '--out', str(directory/'result.json')])
                    model = ImageModel().eval()
                    images = torch.randn(200, 100)
                    config = types.SimpleNamespace(image_dir=str(directory/'default-images'))
                    extractions = []

                    def load(config_file, ckpt, backbone, device, classes, seed):
                        self.assertEqual(classes, 11)  # checkpoint head, not 200 test identities
                        self.assertEqual(seed, 2026)
                        self.assertEqual(backbone, 'ViT-B/16')
                        return model, config, dict(loaded_state='model', seed=seed, mock=True)

                    def extract(loaded, cfg, ann_path, image_root, device, batch_size, workers):
                        self.assertIs(loaded, model)
                        self.assertFalse(model.training)
                        self.assertEqual(Path(ann_path), anno)
                        self.assertEqual(image_root, str(directory/'images'))
                        self.assertEqual(workers, 2)
                        extractions.append((ann_path, batch_size))
                        with torch.no_grad():
                            for start in range(0, len(images), batch_size):
                                model.encode_image(images[start:start+batch_size], return_dense=True)

                    with patch.object(router_eval, 'load_model', side_effect=load) as load_spy, \
                            patch.object(router_eval, 'extract_gallery_features', side_effect=extract), \
                            redirect_stdout(io.StringIO()):
                        report, deterministic = router_eval.run_real(args)
                    self.assertEqual(load_spy.call_count, 1)
                    self.assertEqual(len(extractions), 21)
                    self.assertEqual(report['n_images'], 200)
                    self.assertEqual(report['gumbel_calls_per_pass'], [14] * 20)
                    self.assertEqual(report['annotation']['identities'], 200)
                    self.assertEqual(report['feature_provenance']['loaded_state'], 'model')
                    self.assertEqual(report['main_report_block'], 'blocks.0')
                    self.assertEqual(set(report['inputs']), {'checkpoint', 'annotation', 'config'})
                    for role, path in (('checkpoint', checkpoint), ('annotation', anno), ('config', config_path)):
                        self.assertEqual(report['inputs'][role]['sha256_16'], router_eval.sha256_16(path))
                    self.assertEqual(deterministic['blocks.0']['logits'].shape, (200, 2))
                    self.assertTrue(all(block.eval_router_mode == 'gumbel' for block in model.blocks))
                    wrong = copy.copy(args)
                    wrong.split = 'train' if split == 'gallery' else 'gallery'
                    with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'requires .*_reid.json'):
                        router_eval.run_real(wrong)
            bad_checkpoint = directory/'bad.pth'
            torch.save({'unexpected': {}}, bad_checkpoint)
            args.resume = str(bad_checkpoint)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'model state dict'):
                router_eval.run_real(args)
        print('F5 MOCK run_real: gallery/train 200x100, 2 actual MoE blocks, 21 full passes; checkpoint head=11, hashes/source/provenance/errors PASS')


class EvaluationSeedAndDebugTests(unittest.TestCase):
    """CPU toys exercise the actual main entry/loop; no model loss is substituted
    for F2. These are orchestration tests, not full TAG/CUDA training evidence.
    """

    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main
        source = subprocess.check_output(
            ['git', 'show', '8bf2616403145f5a0e64747ddf9914e0a9e0fcdf:main.py'],
            cwd=ROOT).decode('utf-8')
        cls.baseline = types.ModuleType('round_a_baseline_main')
        exec(compile(source, '8bf2616:main.py', 'exec'), cls.baseline.__dict__)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    class ToyModel(torch.nn.Module):
        def __init__(self, config, num_classes):
            super().__init__()
            self.classifier = torch.nn.Linear(3, num_classes)

        def forward(self, batch, alpha, training=True, **kwargs):
            return {'itc_loss': self.classifier(batch['image']).square().mean(),
                    'view_loss': 0, 'ortho_loss': 0}

    class ToyDataset(torch.utils.data.Dataset):
        person2text = {0: [], 1: []}

        def __len__(self):
            return 16

        def __getitem__(self, index):
            return {'image': torch.tensor([index / 16, 0.2, -0.1])}

    def toy_run(self, directory, module=None, eval_only=False, eval_seed=None,
                debug_max_iters=0, save_last=False, use_ema=False, overflow_first=False):
        module = module or self.main
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stream = io.StringIO()
        logger = logging.getLogger('round-a-toy-' + directory.name)
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler(stream))
        events, scores, learning_rates, outcomes = [], [], [], []
        checkpoint = directory / 'input.pth'
        if eval_only:
            torch.save({'model': self.ToyModel(None, 2).state_dict()}, checkpoint)
        argv = ['--batch_size', '2', '--epoch', '2', '--exp_name', 'toy',
                '--output_dir', str(directory)]
        if eval_only:
            argv += ['--eval_only', '--resume', str(checkpoint)]
        if eval_seed is not None:
            argv += ['--eval_seed', str(eval_seed)]
        if debug_max_iters:
            argv += ['--debug_max_iters', str(debug_max_iters)]
        if save_last:
            argv += ['--save_last']
        if use_ema:
            argv += ['--use_ema']
        resolve = module.resolve_runtime_config

        def resolve_cpu(args):
            config, overrides = resolve(args)
            config.device = 'cpu'
            config.data.num_workers = 0
            return config, overrides

        fake_model_module = types.ModuleType('model.tbps_model_MoE')

        def model_factory(config, classes):
            events.append('model')
            return self.ToyModel(config, classes)

        fake_model_module.clip_vitb = model_factory
        fake_model_module.configure_backbone_from_checkpoint = lambda *a, **k: None

        def loaders(config):
            events.append('loader')
            return {'train_loader': torch.utils.data.DataLoader(
                        self.ToyDataset(), batch_size=config.data.batch_size),
                    'test_loader': types.SimpleNamespace(batch_size=32)}

        def evaluate(model, loader, config, logger, **kwargs):
            model.eval()
            # Explicit mock stochastic eval; metrics are computed, not invented.
            matrix = F.gumbel_softmax(torch.randn(12, 12), hard=True)
            matrix += float(random.random()) + float(np.random.random())
            scores.append(matrix.clone())
            labels = torch.arange(12) // 2
            result = module.metric_eval(matrix, labels, labels, output_dir=kwargs['output_dir'])
            return result, None, None

        build_optimizer = module.build_optimizer

        def optimizer_factory(config, model):
            optimizer = build_optimizer(config, model)
            original_step = optimizer.step

            def step(*a, **kw):
                learning_rates.append([group['lr'] for group in optimizer.param_groups])
                return original_step(*a, **kw)

            optimizer.step = step
            return optimizer

        class CPUScaler:
            def __init__(self):
                self.value, self.calls = 1.0, 0

            def scale(self, value):
                return value

            def unscale_(self, optimizer):
                pass  # already unscaled CPU toy, no AMP claim

            def get_scale(self):
                return self.value

            def step(self, optimizer):
                self.calls += 1
                if not (overflow_first and self.calls == 1):
                    optimizer.step()

            def update(self):
                if overflow_first and self.calls == 1:
                    self.value /= 2

        import model as model_package
        original_run = module.run

        def observed_run(*a, **kw):
            result = original_run(*a, **kw)
            outcomes.append(result)
            return result

        counts = {}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'model.tbps_model_MoE': fake_model_module}))
            stack.enter_context(patch.object(model_package, 'tbps_model_MoE', fake_model_module, create=True))
            for key, value in {
                'resolve_runtime_config': resolve_cpu, 'build_pedes_data': loaders,
                'setup_logger': lambda *a, **kw: logger, 'add_file_handler': lambda *a: None,
                'init_distributed_mode': lambda config: None, 'is_using_distributed': lambda: False,
                'is_master': lambda: True, 'get_rank': lambda: 0,
                'load_checkpoint': lambda model, config: (model, None),
                'validate_checkpoint_mapping': lambda *a: None,
                'evaluate_model': evaluate, 'build_optimizer': optimizer_factory,
                'run': observed_run,
            }.items():
                stack.enter_context(patch.object(module, key, value))
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
            stack.enter_context(patch.object(torch, 'autocast', side_effect=lambda *a, **kw: nullcontext()))
            stack.enter_context(patch.object(torch.cuda.amp, 'GradScaler', side_effect=CPUScaler))
            for owner, name, key in ((random, 'seed', 'python'), (np.random, 'seed', 'numpy'),
                                     (torch, 'manual_seed', 'torch'),
                                     (torch.cuda, 'manual_seed_all', 'cuda')):
                counts[key] = stack.enter_context(patch.object(owner, name, wraps=getattr(owner, name)))
            with redirect_stdout(io.StringIO()):
                exit_code = module.main(argv)
            counts = {key: observer.call_count for key, observer in counts.items()}
        run_directory = next(path.parent for path in directory.glob('*/config_used.yaml'))
        traces = [json.loads(line.split('DEBUG ITER ', 1)[1]) for line in
                  stream.getvalue().splitlines() if 'DEBUG ITER ' in line]
        return dict(exit_code=exit_code, events=events, scores=scores, lr=learning_rates,
                    counts=counts, outcomes=outcomes, directory=run_directory, traces=traces)

    def test_eval_seed_two_actual_eval_only_toys_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.toy_run(Path(directory)/'first', eval_only=True, eval_seed=917)
            second = self.toy_run(Path(directory)/'second', eval_only=True, eval_seed=917)
            self.assertTrue(torch.equal(first['scores'][0], second['scores'][0]))
            self.assertEqual(first['outcomes'], second['outcomes'])
            self.assertEqual(first['events'], ['model', 'loader'])
            meta = json.loads((first['directory']/'evaluation_revision.json').read_text())
            self.assertEqual(meta['eval_seed'], 917)
            self.assertEqual(meta['eval_router_mode'], 'gumbel')
            print('F4 eval_only seed=917: two scores/results torch.equal=True; load -> seed -> loader')

    def test_no_seed_actual_default_call_counts_equal_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            before = self.toy_run(Path(directory)/'before', module=self.baseline, eval_only=True)
            after = self.toy_run(Path(directory)/'after', eval_only=True)
            self.assertEqual(before['counts'], after['counts'])
            self.assertEqual(before['events'], after['events'])
            self.assertEqual(after['events'], ['loader', 'model'])
            self.assertTrue(torch.equal(before['scores'][0], after['scores'][0]))
            self.assertEqual(before['outcomes'], after['outcomes'])
            print(f"F4 default vs 8bf2616: seed calls={after['counts']}; scores/results exact")

    def test_training_eval_seed_does_not_reset_any_rng(self):
        random.seed(83)
        np.random.seed(83)
        torch.manual_seed(83)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with patch.object(random, 'seed') as py, patch.object(np.random, 'seed') as np_seed, \
                patch.object(torch, 'manual_seed') as cpu, patch.object(torch.cuda, 'manual_seed_all') as cuda:
            self.assertFalse(self.main.set_evaluation_seed(False, 98))
            for spy in (py, np_seed, cpu, cuda):
                spy.assert_not_called()
        self.assertEqual(python_state, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], after_numpy[0])
        self.assertTrue(np.array_equal(numpy_state[1], after_numpy[1]))
        self.assertEqual(numpy_state[2:], after_numpy[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        with tempfile.TemporaryDirectory() as directory:
            without = self.toy_run(Path(directory)/'without', debug_max_iters=3)
            with_seed = self.toy_run(Path(directory)/'with', debug_max_iters=3, eval_seed=98)
            self.assertEqual(without['counts'], with_seed['counts'])
            self.assertEqual(without['traces'], with_seed['traces'])
        print('F4 training provided eval_seed: Python/NumPy/torch states unchanged; no extra seed calls')

    def test_debug_three_updates_real_loop_and_formal_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            formal = self.toy_run(Path(directory)/'formal')
            debug = self.toy_run(Path(directory)/'debug', debug_max_iters=3)
            self.assertEqual(debug['exit_code'], 0)
            self.assertEqual(len(debug['lr']), 3)
            self.assertEqual(debug['lr'], formal['lr'][:3])
            self.assertEqual([trace['update'] for trace in debug['traces']], [1, 2, 3])
            self.assertEqual([trace['niter_per_ep'] for trace in debug['traces']], [8, 8, 8])
            self.assertTrue(all(trace['finite'] for trace in debug['traces']))
            self.assertEqual(debug['scores'], [])  # no partial epoch evaluation
            final = (debug['directory']/'final_results.txt').read_text()
            self.assertIn('run_status: debug_stopped', final)
            self.assertIn('debug_max_iters: 3', final)
            self.assertFalse((debug['directory']/'checkpoint_best.pth').exists())
            print(f"F4b CPU actual main loop: exit=0, updates=3, LRs={debug['lr']}, formal[:3] exact, debug_stopped")

    def test_overflow_not_counted_and_save_last_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            debug = self.toy_run(Path(directory)/'overflow', debug_max_iters=3, overflow_first=True)
            self.assertEqual(len(debug['lr']), 3)
            self.assertEqual([trace['it'] for trace in debug['traces']], [1, 2, 3])
            formal = self.toy_run(Path(directory)/'last', save_last=True, use_ema=True)
            checkpoint = torch.load(formal['directory']/'checkpoint_last.pth', weights_only=False)
            self.assertEqual(set(checkpoint), {'model', 'optimizer', 'config', 'epoch', 'model_ema'})
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertIn('last_checkpoint_sha256:', (formal['directory']/'final_results.txt').read_text())
            print('D4 CPU: last checkpoint model/optimizer/config/epoch/model_ema; SHA256 recorded; debug overflow not counted')

    def test_multiprocess_guard_and_cli_config_archival(self):
        args = self.main.build_parser().parse_args([
            '--itc_block_size', '80', '--eval_router_mode', 'argmax', '--eval_seed', '7',
            '--save_last', '--debug_max_iters', '3'])
        config, overrides = self.main.resolve_runtime_config(args)
        with patch.object(self.main, 'get_world_size', return_value=2):
            with self.assertRaisesRegex(ValueError, '多进程路径未验证'):
                self.main.validate_itc_multiprocess(config)
            config.experiment.allow_unverified_multiproc = True
            self.main.validate_itc_multiprocess(config)
        stream = io.StringIO()
        logger = logging.getLogger('round-a-summary-test')
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler(stream))
        self.main.log_cli_overrides(logger, overrides, config)
        for key in self.main.REVISION_CONFIG_DEFAULTS:
            self.assertIn('experiment.' + key, stream.getvalue())


if __name__ == '__main__':
    unittest.main()

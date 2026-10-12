"""CPU fixed-feature parity against verbatim 8bf2616 candidate losses.

Production methods are compiled from the repository AST to avoid importing
unrelated optional vision dependencies. The exercised functions are the actual
production source, not a rewritten implementation. Projectors are deterministic
toy modules; these tests do not claim full-model forward or DDP equivalence.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

try:
    from tests.reference_losses import ReferenceLosses
except ModuleNotFoundError:
    from reference_losses import ReferenceLosses


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'model' / 'tbps_model_MoE.py'
METHODS = {
    '_original_candidate_loss', 'compute_itc_in_blocks',
    'compute_bridge_in_blocks', 'compute_bridge_losses',
    'bidirectional_soft_target_loss', 'compute_ritc', 'compute_sdm',
    'compute_tal', '_compute_tal_direction', 'build_dhn_hard_mask',
    'apply_dhn_margin', 'calc_contrastive', 'calc_contrastive_weighted',
    'forward', 'compute_nrw_weights', '_nrw_batch_moments',
    '_nrw_sigmoid_weights', '_masked_mean',
}
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
clip_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == 'CLIP')
method_class = ast.ClassDef(
    name='ProductionLosses', bases=[], keywords=[], decorator_list=[],
    body=[copy.deepcopy(node) for node in clip_class.body
          if isinstance(node, ast.FunctionDef) and node.name in METHODS],
)
namespace = {'torch': torch, 'F': F, 'tokenize': lambda texts, **kwargs: texts}
exec(compile(ast.fix_missing_locations(ast.Module(
    body=[method_class], type_ignores=[],
)), str(SOURCE), 'exec'), namespace)
ProductionLosses = namespace['ProductionLosses']


def _config(block):
    return SimpleNamespace(experiment=SimpleNamespace(
        itc_block_size=block, allow_unverified_multiproc=False,
        tal_margin=0.1, tal_tau=0.02,
    ))


def _targets(ids):
    positive = ids[:, None].eq(ids[None, :]).float()
    return positive / positive.sum(1, keepdim=True)


class ToyFeatures(nn.Module):
    """Explicit trainable paths for global projection, local/TSE and scale."""
    def __init__(self):
        super().__init__()
        self.image_projection = nn.Linear(12, 8)
        self.text_projection = nn.Linear(10, 8)
        self.visual_emb_layer = nn.Linear(8, 16)
        self.textual_emb_layer = nn.Linear(8, 16)
        self.logit_scale = nn.Parameter(torch.tensor(2.659260036932778))

    def forward(self, image, text):
        visual = self.image_projection(image)
        textual = self.text_projection(text)
        return (
            F.normalize(visual, dim=-1), F.normalize(textual, dim=-1),
            F.normalize(self.visual_emb_layer(visual), dim=-1),
            F.normalize(self.textual_emb_layer(textual), dim=-1),
            self.logit_scale.exp(),
        )


PARAMETER_NAMES = [
    'logit_scale',
    'image_projection.weight', 'image_projection.bias',
    'text_projection.weight', 'text_projection.bias',
    'visual_emb_layer.weight', 'visual_emb_layer.bias',
    'textual_emb_layer.weight', 'textual_emb_layer.bias',
]


def _reference_call(reference, kind, visual, textual, ids, scale,
                    domains, weights, alpha):
    targets = _targets(ids)
    if kind in ('nitc', 'weighted_nitc'):
        hard_mask = reference.build_dhn_hard_mask(ids, ids, domains, domains)
        arguments = (
            visual, textual, visual.detach(), textual.detach(),
            visual, textual, visual.detach(), textual.detach(),
            targets, alpha, scale,
        )
        if kind == 'weighted_nitc':
            return reference.calc_contrastive_weighted(
                *arguments, weights, hard_mask, 0.2,
            )
        return reference.calc_contrastive(*arguments, hard_mask, 0.2)
    if kind == 'ritc':
        return reference.compute_ritc(
            visual, textual, visual, textual, scale, targets,
        )
    if kind == 'sdm':
        return reference.compute_sdm(
            visual, textual, visual, textual, targets, scale,
        )
    if kind == 'tal':
        return reference.compute_tal(
            visual, textual, visual, textual, ids, ids, 0.1, 0.02,
        )
    raise AssertionError(kind)


def _independent_reference(reference, kind, features, ids, scale,
                           domains, weights, alpha, block):
    visual, textual = features
    n = len(ids)
    if block <= 0 or block >= n:
        return _reference_call(
            reference, kind, visual, textual, ids, scale, domains, weights, alpha,
        )
    loss = 0
    for start in range(0, n, block):
        stop = min(start + block, n)
        loss = loss + ((stop - start) / n) * _reference_call(
            reference, kind, visual[start:stop], textual[start:stop],
            ids[start:stop], scale, domains[start:stop],
            weights[start:stop], alpha,
        )
    return loss


class ForwardHarness(ProductionLosses, nn.Module):
    """Exercise the actual CLIP.forward branches with toy feature encoders."""
    def __init__(self, block, nrw='off', bridge=False):
        nn.Module.__init__(self)
        self.features = ToyFeatures()
        self.logit_scale = self.features.logit_scale
        self.use_gather = False
        self.eps = 0.01
        self.config = _config(block)
        self.config.device = 'cpu'
        self.config.experiment.__dict__.update(
            view_prompt=False, text_length=77, view_ratio=1.0, ortho_ratio=1.0,
            nrw_mode=nrw, dhn_margin=0.2, nitc_ratio=1.0,
            sdm_ratio=1.0, tal_ratio=1.0, ritc=True, ritc_ratio=1.0,
            id=False, id_ratio=0.5, use_bridge=bridge,
            bridge_cv_ratio=0.3 if bridge else 0.0,
            bridge_tg_ratio=0.4 if bridge else 0.0,
        )
        self.nrw_calls = 0
        self.bridge_calls = 0

    def all_gather(self, value):
        return value

    def encode_image(self, image, **kwargs):
        features = self.features.image_projection(image)
        return features, features.unsqueeze(1), {}, None

    def encode_text(self, text, **kwargs):
        features = self.features.text_projection(text)
        return features, features.unsqueeze(1), None

    def visual_emb_layer(self, attention, dense):
        return self.features.visual_emb_layer(dense[:, 0])

    def textual_emb_layer(self, attention, tokens, dense):
        return self.features.textual_emb_layer(dense[:, 0])

    def compute_nrw_weights(self, visual, textual, cam, epoch):
        self.nrw_calls += 1
        return torch.linspace(0.2, 1.0, len(cam)), {}

    def encode_bridge_image(self, image, cam_id, training):
        self.bridge_calls += 1
        self.encoded_bridge_count = len(image)
        return self.features.image_projection(image)


class ITCBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        print('F2 trainable parameter list: ' + ', '.join(PARAMETER_NAMES))
        print('F2 fixed features: D_global=8 D_TSE=16 alpha=0.5 ritc_eps=0.01 '
              'sdm_epsilon=1e-8 tal_margin=0.1 tal_tau=0.02 DHN_margin=0.2')
        print('F2 TAL does not use logit_scale: its gradient is None in both implementations')

    def _compare(self, n, block, kind):
        torch.manual_seed(2026 + n)
        actual_network = ToyFeatures()
        expected_network = copy.deepcopy(actual_network)
        image = torch.randn(n, 12)
        text = torch.randn(n, 10)
        # IDs repeat within each full block and across boundaries.
        ids = torch.arange(n) % 17
        domains = torch.arange(n) % 5
        weights = torch.linspace(0.2, 1.0, n).detach()
        actual_features = actual_network(image, text)
        expected_features = expected_network(image, text)
        actual = ProductionLosses()
        actual.config = _config(block)
        actual.eps = 0.01
        actual.use_gather = False
        reference = ReferenceLosses(eps=0.01)
        actual_loss = 0
        expected_loss = 0
        for route in (0, 2):
            visual, textual = actual_features[route:route + 2]
            actual_loss = actual_loss + actual.compute_itc_in_blocks(
                'nitc' if kind == 'weighted_nitc' else kind,
                visual, textual, visual, textual, ids, ids, _targets(ids),
                actual_features[-1], alpha=0.5,
                sample_weights=weights if kind == 'weighted_nitc' else None,
                domain_ids=domains, gathered_domain_ids=domains,
                dhn_margin=0.2 if kind in ('nitc', 'weighted_nitc') else 0.0,
            )
            expected_loss = expected_loss + _independent_reference(
                reference, kind, expected_features[route:route + 2],
                ids, expected_features[-1], domains, weights, 0.5, block,
            )
        self.assertTrue(torch.isfinite(actual_loss))
        self.assertTrue(torch.isfinite(expected_loss))
        actual_loss.backward()
        expected_loss.backward()
        exact = block <= 0 or block >= n
        tolerance = 0.0 if exact else 1e-6
        loss_error = (actual_loss - expected_loss).abs().item()
        self.assertLessEqual(loss_error, tolerance)
        if exact:
            self.assertTrue(torch.equal(actual_loss, expected_loss))
        max_gradient_error = 0.0
        actual_parameters = dict(actual_network.named_parameters())
        expected_parameters = dict(expected_network.named_parameters())
        self.assertEqual(set(actual_parameters), set(PARAMETER_NAMES))
        for name in PARAMETER_NAMES:
            actual_grad = actual_parameters[name].grad
            expected_grad = expected_parameters[name].grad
            if kind == 'tal' and name == 'logit_scale':
                self.assertIsNone(actual_grad)
                self.assertIsNone(expected_grad)
                continue  # The original TAL uses tau, not CLIP logit_scale.
            self.assertIsNotNone(actual_grad, name)
            self.assertIsNotNone(expected_grad, name)
            self.assertTrue(torch.isfinite(actual_grad).all(), name)
            self.assertTrue(torch.isfinite(expected_grad).all(), name)
            error = (actual_grad - expected_grad).abs().max().item()
            max_gradient_error = max(max_gradient_error, error)
            self.assertLessEqual(error, tolerance, name)
            if exact:
                self.assertTrue(torch.equal(actual_grad, expected_grad), name)
        print(f'F2 {kind} global+TSE N={n} B={block}: '
              f'loss_error={loss_error:.12g} all_gradient_error={max_gradient_error:.12g} '
              f'finite=True exact={exact}')

    def test_reference_parity_all_candidate_losses_and_boundary_cases(self):
        for n, block in ((320, 0), (320, 320), (320, 80),
                         (330, 80), (79, 80)):
            for kind in ('nitc', 'weighted_nitc', 'ritc', 'sdm', 'tal'):
                with self.subTest(n=n, block=block, kind=kind):
                    self._compare(n, block, kind)

    def test_original_loss_function_sources_are_unchanged(self):
        reference_tree = ast.parse((Path(__file__).parent / 'reference_losses.py').read_text(encoding='utf-8'))
        reference_class = next(n for n in reference_tree.body if isinstance(n, ast.ClassDef))
        original_methods = {
            n.name: n for n in reference_class.body if isinstance(n, ast.FunctionDef)
        }
        production_methods = {
            n.name: n for n in clip_class.body if isinstance(n, ast.FunctionDef)
        }
        unchanged = METHODS - {
            '_original_candidate_loss', 'compute_itc_in_blocks',
            'compute_bridge_in_blocks', 'compute_bridge_losses', 'forward',
            'compute_nrw_weights', '_nrw_batch_moments',
            '_nrw_sigmoid_weights', '_masked_mean',
        }
        for name in unchanged:
            self.assertEqual(
                ast.dump(original_methods[name], include_attributes=False),
                ast.dump(production_methods[name], include_attributes=False),
                name,
            )
        print('F2 original loss formulas/reductions: AST equal to verbatim 8bf2616 reference')

    def test_default_actual_forward_matches_baseline_forward_and_gradients(self):
        for nrw in ('off', 'fixed'):
            torch.manual_seed(19)
            current = ForwardHarness(0, nrw=nrw, bridge=True)
            baseline = ForwardHarness(0, nrw=nrw, bridge=True)
            baseline.load_state_dict(current.state_dict())
            # Bind the exact old methods; feature encoders remain identical toys.
            baseline.compute_bridge_losses = ReferenceLosses.compute_bridge_losses.__get__(baseline)
            batch = {
                'image': torch.randn(17, 12), 'caption': torch.randn(17, 10),
                'cam_id': torch.arange(17) % 2, 'id': torch.arange(17) % 3,
                'domain_id': torch.arange(17) % 2,
                'bridge_image': torch.randn(17, 12),
                'has_bridge': torch.arange(17) % 2,
            }
            current_result = current.forward(batch, 0.5, training=True)
            with mock.patch.dict(ReferenceLosses.forward.__globals__, {
                'tokenize': lambda texts, **kwargs: texts,
            }):
                old_result = ReferenceLosses.forward(baseline, batch, 0.5, training=True)
            self.assertEqual(set(current_result), set(old_result))
            for key in current_result:
                if torch.is_tensor(current_result[key]):
                    self.assertTrue(torch.equal(current_result[key], old_result[key]), key)
                else:
                    self.assertEqual(current_result[key], old_result[key])
            current_loss = sum(v for k, v in current_result.items() if 'loss' in k)
            old_loss = sum(v for k, v in old_result.items() if 'loss' in k)
            current_loss.backward()
            old_loss.backward()
            maximum = 0.0
            for (name, p), (old_name, q) in zip(current.named_parameters(), baseline.named_parameters()):
                self.assertEqual(name, old_name)
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.equal(p.grad, q.grad), name)
                maximum = max(maximum, (p.grad - q.grad).abs().max().item())
            print(f'F2 actual default forward versus 8bf2616 forward NRW={nrw}: '
                  f'loss_error={(current_loss - old_loss).abs().item():.12g} '
                  f'all_gradient_error={maximum:.12g} torch.equal=True')

    def test_tal_all_same_id_and_single_row_blocks_are_finite(self):
        reference = ReferenceLosses()
        model = ProductionLosses()
        model.config = _config(1)
        model.eps = 0.01
        model.use_gather = False
        for n in (1, 3):
            images = F.normalize(torch.randn(n, 8), dim=-1).requires_grad_(True)
            texts = F.normalize(torch.randn(n, 8), dim=-1).requires_grad_(True)
            ids = torch.zeros(n, dtype=torch.long)
            loss = model.compute_itc_in_blocks(
                'tal', images, texts, images, texts, ids, ids,
                _targets(ids), torch.tensor(14.0),
            )
            expected = reference.compute_tal(images, texts, images, texts, ids, ids, 0.1, 0.02)
            self.assertTrue(torch.equal(loss, expected))
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(torch.isfinite(images.grad).all())
            self.assertTrue(torch.isfinite(texts.grad).all())

    def test_real_nrw_full_batch_ema_updates_once_and_weights_are_reused(self):
        torch.manual_seed(17)
        batch = {
            'image': torch.randn(17, 12), 'caption': torch.randn(17, 10),
            'cam_id': torch.arange(17) % 2, 'id': torch.arange(17) % 3,
            'domain_id': torch.arange(17) % 2,
        }
        for mode in ('view_norm', 'global_norm'):
            full = ForwardHarness(0, nrw=mode)
            blocked = ForwardHarness(4, nrw=mode)
            blocked.load_state_dict(full.state_dict())
            for model in (full, blocked):
                model.config.experiment.__dict__.update(
                    nrw_ema=0.99, nrw_warmup_epochs=3,
                    nrw_min_weight=0.2, nrw_alpha=2.0,
                )
                if mode == 'view_norm':
                    model.register_buffer('nrw_view_mu', torch.zeros(2))
                    model.register_buffer('nrw_view_var', torch.ones(2))
                else:
                    model.register_buffer('nrw_global_mu', torch.zeros(()))
                    model.register_buffer('nrw_global_var', torch.ones(()))
                model.compute_nrw_weights = ProductionLosses.compute_nrw_weights.__get__(model)
            full.forward(batch, 0.5, training=True, epoch=3)
            with mock.patch.object(blocked, 'compute_nrw_weights', wraps=blocked.compute_nrw_weights) as compute:
                with mock.patch.object(blocked, '_nrw_batch_moments', wraps=blocked._nrw_batch_moments) as moments:
                    result = blocked.forward(batch, 0.5, training=True, epoch=3)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(moments.call_count, 1)
            for name, expected in full.named_buffers():
                self.assertTrue(torch.equal(dict(blocked.named_buffers())[name], expected), name)
            self.assertTrue(all(torch.isfinite(value).all() for key, value in result.items()
                                if 'loss' in key and torch.is_tensor(value)))
            print(f'F2 real NRW={mode}: weights/statistics called once; '
                  'buffers torch.equal to unblocked full-batch update')

    def test_bridge_original_row_boundaries_and_gradient_parity(self):
        for block in (0, 80, 330):
            torch.manual_seed(330)
            n = 330
            has_bridge = (torch.arange(n) % 3 == 0)
            has_bridge[80:160] = False  # Covers an empty original block.
            positions = has_bridge.nonzero().view(-1)
            ids = (torch.arange(n) % 17)[has_bridge]
            a = torch.randn(len(ids), 8, requires_grad=True)
            b = torch.randn(len(ids), 8, requires_grad=True)
            scale = torch.tensor(14.0, requires_grad=True)
            ar = a.detach().clone().requires_grad_(True)
            br = b.detach().clone().requires_grad_(True)
            sr = scale.detach().clone().requires_grad_(True)
            actual = ProductionLosses()
            actual.config = _config(block)
            reference = ReferenceLosses()
            loss = actual.compute_bridge_in_blocks(a, b, ids, has_bridge, _targets(ids), scale)
            if block <= 0 or block >= n:
                expected = reference.bidirectional_soft_target_loss(ar, br, _targets(ids), sr)
            else:
                expected = 0
                for start in range(0, n, block):
                    mask = (positions >= start) & (positions < min(start + block, n))
                    count = int(mask.sum())
                    if count:
                        expected = expected + (count / len(ids)) * reference.bidirectional_soft_target_loss(
                            ar[mask], br[mask], _targets(ids[mask]), sr,
                        )
            loss.backward()
            expected.backward()
            errors = [(x.grad - y.grad).abs().max().item()
                      for x, y in ((a, ar), (b, br), (scale, sr))]
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in (a, b, scale)))
            self.assertLessEqual((loss - expected).abs().item(), 1e-6)
            self.assertLessEqual(max(errors), 1e-6)
            print(f'F2 bridge original indices N={n} M={len(ids)} B={block}: '
                  f'loss_error={(loss - expected).abs().item():.12g} '
                  f'all_gradient_error={max(errors):.12g} finite=True')

    def test_real_forward_wiring_preserves_off_and_nrw_update_count(self):
        torch.manual_seed(3)
        batch = {
            'image': torch.randn(10, 12), 'caption': torch.randn(10, 10),
            'cam_id': torch.arange(10) % 2, 'id': torch.arange(10) % 3,
            'domain_id': torch.arange(10) % 2,
            'bridge_image': torch.randn(10, 12),
            'has_bridge': torch.arange(10) % 2,
        }
        for nrw in ('off', 'fixed'):
            zero = ForwardHarness(0, nrw=nrw, bridge=True)
            full = ForwardHarness(10, nrw=nrw, bridge=True)
            full.load_state_dict(zero.state_dict())
            # Default branch must not call any new candidate helper.
            with mock.patch.object(zero, 'compute_itc_in_blocks', side_effect=AssertionError('off helper')):
                with mock.patch.object(zero, 'compute_bridge_in_blocks', side_effect=AssertionError('off bridge helper')):
                    raw = zero.forward(batch, 0.5, training=True)
            entire = full.forward(batch, 0.5, training=True)
            self.assertEqual(set(raw), set(entire))
            for key in raw:
                if torch.is_tensor(raw[key]):
                    self.assertTrue(torch.equal(raw[key], entire[key]), key)
                else:
                    self.assertEqual(raw[key], entire[key])
            short = ForwardHarness(4, nrw=nrw, bridge=True)
            short.load_state_dict(zero.state_dict())
            output = short.forward(batch, 0.5, training=True)
            self.assertTrue(all(torch.isfinite(v).all() for k, v in output.items()
                                if 'loss' in k and torch.is_tensor(v)))
            self.assertEqual(short.nrw_calls, 0 if nrw == 'off' else 1)
            self.assertEqual(short.bridge_calls, 1)
            self.assertEqual(short.encoded_bridge_count, int(batch['has_bridge'].sum()))
        print('F2 actual CLIP.forward AST: off helpers=0; full-block outputs=torch.equal; '
              'NRW computed once; bridge encoded once; zero rows not encoded')

    def test_all_zero_bridge_returns_differentiable_zero(self):
        model = ForwardHarness(4, bridge=True)
        batch = {
            'has_bridge': torch.zeros(10, dtype=torch.long),
            'bridge_image': torch.randn(10, 12),
        }
        features = torch.randn(10, 8, requires_grad=True)
        result = model.compute_bridge_losses(
            batch, features, features, torch.arange(10), torch.tensor(14.0),
            0.3, 0.4, True,
        )
        self.assertEqual(model.bridge_calls, 0)
        self.assertEqual(result['bridge_cv_loss'].item(), 0.0)
        self.assertEqual(result['bridge_tg_loss'].item(), 0.0)
        sum(result.values()).backward()
        self.assertTrue(torch.equal(features.grad, torch.zeros_like(features)))

    def test_unverified_multiprocess_requires_opt_in(self):
        model = ProductionLosses()
        model.config = _config(80)
        model.use_gather = True
        image = torch.randn(2, 4)
        ids = torch.arange(2)
        with mock.patch('torch.distributed.is_initialized', return_value=True):
            with mock.patch('torch.distributed.get_world_size', return_value=2):
                with self.assertRaisesRegex(RuntimeError, '多进程路径未验证'):
                    model.compute_itc_in_blocks(
                        'nitc', image, image, image, image, ids, ids,
                        _targets(ids), torch.tensor(14.0),
                    )


if __name__ == '__main__':
    unittest.main()

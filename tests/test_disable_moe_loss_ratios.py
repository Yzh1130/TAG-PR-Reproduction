"""Numerical guard for paper-compatible ratios when HR-MoE is disabled."""

import copy
import unittest
from types import SimpleNamespace

import torch

try:
    from model.base_transformer import ResidualAttentionBlock, Transformer_MoE
except ModuleNotFoundError as exc:
    if exc.name != "einops":
        raise
    ResidualAttentionBlock = None
    Transformer_MoE = None


def _config(view_ratio, ortho_ratio):
    return SimpleNamespace(
        expert_blocks=[0, 1],
        num_of_experts=6,
        num_of_selected_experts=5,
        experiment=SimpleNamespace(
            disable_moe=True,
            view_ratio=float(view_ratio),
            ortho_ratio=float(ortho_ratio),
        ),
    )


@unittest.skipIf(Transformer_MoE is None, "optional training dependency einops is unavailable")
class DisableMoeLossRatioTests(unittest.TestCase):
    def test_one_and_zero_ratios_have_identical_forward_and_backward(self):
        torch.manual_seed(2026)
        ratios_one = Transformer_MoE(
            _config(1.0, 1.0), width=8, layers=2, heads=2,
        )
        ratios_zero = Transformer_MoE(
            _config(0.0, 0.0), width=8, layers=2, heads=2,
        )
        ratios_zero.load_state_dict(copy.deepcopy(ratios_one.state_dict()))

        self.assertEqual(ratios_one.expert_blocks, [])
        self.assertEqual(ratios_zero.expert_blocks, [])
        self.assertTrue(all(
            isinstance(block, ResidualAttentionBlock)
            for block in ratios_one.resblocks
        ))

        # Sequence layout is [CLS + patches + appended view token, batch, D].
        # disable_moe removes the final row before either standard ViT stack.
        base = torch.randn(7, 3, 8)
        input_one = base.clone().requires_grad_(True)
        input_zero = base.clone().requires_grad_(True)
        cam_id = torch.tensor([0, 1, 0])

        output_one, losses_one, attention_one = ratios_one(
            input_one, cam_id=cam_id, training=True,
        )
        output_zero, losses_zero, attention_zero = ratios_zero(
            input_zero, cam_id=cam_id, training=True,
        )

        self.assertEqual(losses_one, {})
        self.assertEqual(losses_zero, {})
        self.assertTrue(torch.equal(output_one, output_zero))
        self.assertTrue(torch.equal(attention_one, attention_zero))

        output_one.square().sum().backward()
        output_zero.square().sum().backward()
        self.assertTrue(torch.equal(input_one.grad, input_zero.grad))
        for (name_one, parameter_one), (name_zero, parameter_zero) in zip(
            ratios_one.named_parameters(), ratios_zero.named_parameters()
        ):
            self.assertEqual(name_one, name_zero)
            self.assertIsNotNone(parameter_one.grad, name_one)
            self.assertTrue(
                torch.equal(parameter_one.grad, parameter_zero.grad),
                name_one,
            )


if __name__ == "__main__":
    unittest.main()

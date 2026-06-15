from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.training.losses import apply_water_vs_other_loss


class WaterVsOtherLossTests(unittest.TestCase):
    def test_collapses_multiclass_logits_into_water_vs_other(self) -> None:
        action_logits = torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
            ],
            requires_grad=True,
        )
        action_targets = torch.tensor([0, 2])
        base_loss = action_logits.sum() * 0.0

        total_loss, loss_value = apply_water_vs_other_loss(
            total_loss=base_loss,
            action_logits=action_logits,
            action_targets=action_targets,
            weight=0.3,
            water_class_index=0,
        )

        expected_logits = torch.stack(
            [action_logits[:, 0], torch.logsumexp(action_logits[:, 1:], dim=1)],
            dim=1,
        )
        expected_loss = F.cross_entropy(expected_logits, torch.tensor([0, 1]))
        self.assertAlmostEqual(loss_value, float(expected_loss.item()))
        self.assertTrue(torch.allclose(total_loss, 0.3 * expected_loss))

        total_loss.backward()
        self.assertIsNotNone(action_logits.grad)

    def test_is_disabled_for_binary_action_classification(self) -> None:
        action_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
        base_loss = action_logits.sum() * 0.0

        total_loss, loss_value = apply_water_vs_other_loss(
            total_loss=base_loss,
            action_logits=action_logits,
            action_targets=torch.tensor([0]),
            weight=0.3,
        )

        self.assertIs(total_loss, base_loss)
        self.assertEqual(loss_value, 0.0)


if __name__ == "__main__":
    unittest.main()

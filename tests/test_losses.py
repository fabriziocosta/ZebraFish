from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from src.training.losses import (
    apply_water_vs_other_loss,
    compute_hierarchical_action_losses,
    prototype_consistency_loss,
)


class WaterVsOtherLossTests(unittest.TestCase):
    def test_prototype_consistency_loss_backpropagates_to_both_branches(self) -> None:
        st_logits = torch.tensor([[2.0, 0.0, -1.0]], requires_grad=True)
        ts_logits = torch.tensor([[0.5, 1.0, -0.5]], requires_grad=True)

        loss = prototype_consistency_loss(st_logits, ts_logits, temperature=0.2)

        self.assertGreater(float(loss.item()), 0.0)
        loss.backward()
        self.assertIsNotNone(st_logits.grad)
        self.assertIsNotNone(ts_logits.grad)

    def test_prototype_consistency_loss_rejects_non_positive_temperature(self) -> None:
        with self.assertRaises(ValueError):
            prototype_consistency_loss(torch.zeros(1, 2), torch.zeros(1, 2), temperature=0.0)

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

    def test_hierarchical_loss_uses_water_head_and_drug_conditional_classes(self) -> None:
        action_logits = torch.tensor(
            [
                [7.0, 1.0, 0.0],
                [6.0, 2.0, 0.5],
                [8.0, -1.0, 3.0],
            ],
            requires_grad=True,
        )
        water_logits = torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 3.0],
                [-1.0, 4.0],
            ],
            requires_grad=True,
        )
        action_targets = torch.tensor([0, 1, 2])
        action_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([10.0, 2.0, 3.0]))

        action_loss, water_loss, total_loss = compute_hierarchical_action_losses(
            action_logits=action_logits,
            water_logits=water_logits,
            action_targets=action_targets,
            action_criterion=action_criterion,
            water_weight=0.4,
            water_class_index=0,
        )

        expected_action_loss = F.cross_entropy(
            action_logits[1:, 1:],
            torch.tensor([0, 1]),
            weight=torch.tensor([2.0, 3.0]),
        )
        expected_water_loss = F.cross_entropy(water_logits, torch.tensor([0, 1, 1]))
        self.assertTrue(torch.allclose(action_loss, expected_action_loss))
        self.assertTrue(torch.allclose(water_loss, expected_water_loss))
        self.assertTrue(torch.allclose(total_loss, expected_action_loss + 0.4 * expected_water_loss))

        total_loss.backward()
        self.assertIsNotNone(action_logits.grad)
        self.assertIsNotNone(water_logits.grad)


if __name__ == "__main__":
    unittest.main()

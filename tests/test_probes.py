from __future__ import annotations

import unittest

import torch

from src.models.probes import (
    PROBE_TYPES,
    ProbeSpec,
    build_probe_masks,
    build_probe_targets,
    masked_probe_loss,
)


class ProbeTests(unittest.TestCase):
    def test_probe_targets_and_masks_keep_fixed_ontology(self) -> None:
        X = torch.arange(2 * 4 * 2 * 6 * 6, dtype=torch.float32).reshape(2, 4, 2, 6, 6)
        spec = ProbeSpec(region_grid=(1, 2, 2), time_bins=3, frequency_bins=2)

        targets = build_probe_targets(X, spec)
        masks = build_probe_masks(targets, observe_probability=0.5)

        self.assertEqual(tuple(targets), PROBE_TYPES)
        self.assertEqual(tuple(masks), PROBE_TYPES)
        self.assertEqual(targets["local"].shape, (2, 4, 3, 2))
        self.assertEqual(targets["region_time"].shape, (2, 4, 3))
        self.assertEqual(targets["derivative"].shape, (2, 4, 3))
        self.assertEqual(targets["frequency"].shape, (2, 4, 2))
        self.assertEqual(targets["correlation"].shape, (2, 6))
        for probe_type in PROBE_TYPES:
            self.assertEqual(masks[probe_type].shape, targets[probe_type].shape)
            self.assertGreaterEqual(float(masks[probe_type].sum().item()), 1.0)

    def test_probe_targets_use_coarse_spatiotemporal_statistics(self) -> None:
        X = torch.empty(1, 4, 1, 2, 4)
        X[:, 0, :, :, :2] = 1.0
        X[:, 0, :, :, 2:] = 2.0
        X[:, 1, :, :, :2] = 3.0
        X[:, 1, :, :, 2:] = 4.0
        X[:, 2, :, :, :2] = 5.0
        X[:, 2, :, :, 2:] = 8.0
        X[:, 3, :, :, :2] = 7.0
        X[:, 3, :, :, 2:] = 10.0
        spec = ProbeSpec(region_grid=(1, 1, 2), time_bins=2, frequency_bins=2)

        targets = build_probe_targets(X, spec)

        expected_local = torch.tensor([[[[2.0, 1.0], [6.0, 1.0]], [[3.0, 1.0], [9.0, 1.0]]]])
        expected_region_time = expected_local[..., 0]
        expected_derivative = torch.tensor([[[0.0, 4.0], [0.0, 6.0]]])
        expected_frequency = torch.tensor([[[0.0, 4.0], [0.0, 6.0]]])
        expected_correlation = torch.tensor([[1.0]])

        self.assertEqual(targets["local"].shape, (1, 2, 2, 2))
        self.assertTrue(torch.allclose(targets["local"], expected_local))
        self.assertTrue(torch.equal(targets["region_time"], targets["local"][..., 0]))
        self.assertTrue(torch.allclose(targets["region_time"], expected_region_time))
        self.assertTrue(torch.allclose(targets["derivative"], expected_derivative))
        self.assertTrue(torch.allclose(targets["frequency"], expected_frequency))
        self.assertTrue(torch.allclose(targets["correlation"], expected_correlation, atol=1e-5))

    def test_probe_targets_reject_empty_coarse_bins(self) -> None:
        X = torch.zeros(1, 2, 1, 2, 2)
        with self.assertRaisesRegex(ValueError, "time_bins"):
            build_probe_targets(X, ProbeSpec(region_grid=(1, 1, 1), time_bins=3))
        with self.assertRaisesRegex(ValueError, "region_grid"):
            build_probe_targets(X, ProbeSpec(region_grid=(2, 1, 1), time_bins=2))

    def test_probe_masks_can_use_full_validation_mask(self) -> None:
        targets = {
            "local": torch.zeros(2, 4, 3, 2),
            "region_time": torch.zeros(2, 4, 3),
        }
        masks = build_probe_masks(targets, observe_probability=0.25, full=True)

        for probe_type, target in targets.items():
            self.assertTrue(torch.equal(masks[probe_type], torch.ones_like(target)))

    def test_masked_probe_loss_normalizes_by_observed_entries(self) -> None:
        targets = {"local": torch.tensor([[1.0, 3.0]])}
        predictions = {"local": torch.tensor([[2.0, 5.0]])}
        masks = {"local": torch.tensor([[1.0, 0.0]])}
        for probe_type in PROBE_TYPES:
            targets.setdefault(probe_type, torch.zeros(1, 1))
            predictions.setdefault(probe_type, torch.zeros(1, 1))
            masks.setdefault(probe_type, torch.zeros(1, 1))

        alpha_weights = {probe_type: 0.0 for probe_type in PROBE_TYPES}
        alpha_weights["local"] = 1.0
        total, per_probe = masked_probe_loss(
            predictions,
            targets,
            masks,
            alpha_weights=alpha_weights,
        )

        self.assertAlmostEqual(float(total.item()), 1.0, places=5)
        self.assertAlmostEqual(float(per_probe["local"].item()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

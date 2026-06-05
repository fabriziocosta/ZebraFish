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
        spec = ProbeSpec(local_count=5, region_grid=(1, 2, 2), time_bins=3, frequency_bins=2)

        targets = build_probe_targets(X, spec)
        masks = build_probe_masks(targets, observe_probability=0.5)

        self.assertEqual(tuple(targets), PROBE_TYPES)
        self.assertEqual(tuple(masks), PROBE_TYPES)
        self.assertEqual(targets["local"].shape, (2, 5))
        self.assertEqual(targets["region_time"].shape, (2, 4, 3))
        self.assertEqual(targets["derivative"].shape, (2, 4, 3))
        self.assertEqual(targets["frequency"].shape, (2, 4, 2))
        self.assertEqual(targets["correlation"].shape, (2, 6))
        for probe_type in PROBE_TYPES:
            self.assertEqual(masks[probe_type].shape, targets[probe_type].shape)

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

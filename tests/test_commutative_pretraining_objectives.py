from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from src.ml import (
    CommutativeCNNClassifier,
    CommutativeCNNConfig,
    CommutativeTransformerClassifier,
    CommutativeTransformerConfig,
    LossWeightConfig,
    OptimizationConfig,
)
from src.models.probes import ProbeSpec, build_probe_targets
from src.training.pretraining import (
    _compute_commutative_pretraining_loss,
    _cross_probe_teacher_targets,
    _cross_weight_for_epoch,
    _prototype_weight_for_epoch,
    linear_ramp,
)


class CommutativePretrainingObjectiveTests(unittest.TestCase):
    def _build_cnn_model(self, *, num_prototypes: int = 7):
        estimator = CommutativeCNNClassifier(
            model_config=CommutativeCNNConfig(
                spatial_conv_channels=(4,),
                temporal_st_channels=(4,),
                temporal_ts_channels=(4,),
                spatial_agg_channels=(4,),
                patch_size_z=1,
                patch_size_xy=8,
                embedding_dim=6,
                num_prototypes=num_prototypes,
                verbose=False,
            ),
            optimization_config=OptimizationConfig(verbose=False),
        )
        estimator.compound_classes_ = None
        estimator.concentration_classes_ = None
        return estimator._build_model(num_classes=2)

    def _build_transformer_model(self, *, num_prototypes: int = 5):
        estimator = CommutativeTransformerClassifier(
            model_config=CommutativeTransformerConfig(
                spatial_patch_size_st=(1, 8, 8),
                spatial_patch_size_ts=(1, 8, 8),
                temporal_patch_size_ts=2,
                embed_dim=8,
                num_heads=2,
                st_spatial_depth=1,
                st_temporal_depth=1,
                ts_temporal_depth=1,
                ts_spatial_depth=1,
                embedding_dim=6,
                num_prototypes=num_prototypes,
                verbose=False,
            ),
            optimization_config=OptimizationConfig(verbose=False),
        )
        estimator.compound_classes_ = None
        estimator.concentration_classes_ = None
        return estimator._build_model(num_classes=2)

    def test_commutative_cnn_outputs_branch_prototypes(self) -> None:
        model = self._build_cnn_model(num_prototypes=7)
        outputs = model(torch.randn(2, 4, 2, 16, 16))

        self.assertEqual(tuple(outputs["st_prototypes"].shape), (2, 7))
        self.assertEqual(tuple(outputs["ts_prototypes"].shape), (2, 7))

    def test_commutative_transformer_outputs_branch_prototypes(self) -> None:
        model = self._build_transformer_model(num_prototypes=5)
        outputs = model(torch.randn(2, 4, 2, 16, 16))

        self.assertEqual(tuple(outputs["st_prototypes"].shape), (2, 5))
        self.assertEqual(tuple(outputs["ts_prototypes"].shape), (2, 5))

    def test_linear_ramp_honors_warmup_ramp_and_target(self) -> None:
        self.assertEqual(linear_ramp(epoch=2, target_weight=0.8, warmup_epochs=2, ramp_epochs=4), 0.0)
        self.assertAlmostEqual(linear_ramp(epoch=3, target_weight=0.8, warmup_epochs=2, ramp_epochs=4), 0.2)
        self.assertAlmostEqual(linear_ramp(epoch=4, target_weight=0.8, warmup_epochs=2, ramp_epochs=4), 0.4)
        self.assertAlmostEqual(linear_ramp(epoch=6, target_weight=0.8, warmup_epochs=2, ramp_epochs=4), 0.8)
        self.assertAlmostEqual(linear_ramp(epoch=20, target_weight=0.8, warmup_epochs=2, ramp_epochs=4), 0.8)

    def test_cross_and_prototype_zero_ramp_activate_after_warmup(self) -> None:
        estimator = SimpleNamespace(
            lambda_cross=0.35,
            cross_warmup_epochs=3,
            cross_ramp_epochs=0,
            prototype_alignment_weight=0.6,
            prototype_warmup_epochs=2,
            prototype_ramp_epochs=0,
        )

        self.assertEqual(_cross_weight_for_epoch(estimator, 3), 0.0)
        self.assertEqual(_prototype_weight_for_epoch(estimator, 2), 0.0)
        self.assertAlmostEqual(_cross_weight_for_epoch(estimator, 4), 0.35)
        self.assertAlmostEqual(_prototype_weight_for_epoch(estimator, 3), 0.6)

    def test_prototype_schedule_uses_linear_ramp(self) -> None:
        estimator = SimpleNamespace(
            prototype_alignment_weight=1.2,
            prototype_warmup_epochs=2,
            prototype_ramp_epochs=3,
        )

        self.assertEqual(_prototype_weight_for_epoch(estimator, 2), 0.0)
        self.assertAlmostEqual(_prototype_weight_for_epoch(estimator, 3), 0.4)
        self.assertAlmostEqual(_prototype_weight_for_epoch(estimator, 4), 0.8)
        self.assertAlmostEqual(_prototype_weight_for_epoch(estimator, 5), 1.2)
        self.assertAlmostEqual(_prototype_weight_for_epoch(estimator, 8), 1.2)

    def test_cross_probe_teacher_targets_are_detached_opposite_self_predictions(self) -> None:
        outputs = {
            "pred_A_self": {"local": torch.ones(2, 3, requires_grad=True)},
            "pred_B_self": {"local": torch.full((2, 3), 2.0, requires_grad=True)},
        }

        targets_A_to_B, targets_B_to_A = _cross_probe_teacher_targets(outputs)

        self.assertTrue(torch.equal(targets_A_to_B["local"], outputs["pred_B_self"]["local"]))
        self.assertTrue(torch.equal(targets_B_to_A["local"], outputs["pred_A_self"]["local"]))
        self.assertFalse(targets_A_to_B["local"].requires_grad)
        self.assertFalse(targets_B_to_A["local"].requires_grad)

    def test_cross_probe_targets_do_not_switch_back_to_raw_probe_targets_after_warmup(self) -> None:
        X = torch.randn(2, 4, 1, 8, 8)
        spec = ProbeSpec(local_count=3, region_grid=(1, 1, 1), time_bins=2, frequency_bins=2)
        probe_targets = build_probe_targets(X, spec)
        pred_A_self = {key: target + 1.0 for key, target in probe_targets.items()}
        pred_B_self = {key: target + 2.0 for key, target in probe_targets.items()}
        outputs = {
            "pred_A_self": pred_A_self,
            "pred_B_self": pred_B_self,
            "pred_A_to_B": {key: value.detach().clone() for key, value in pred_B_self.items()},
            "pred_B_to_A": {key: value.detach().clone() for key, value in pred_A_self.items()},
            "st_embedding": torch.zeros(2, 4),
            "ts_embedding": torch.zeros(2, 4),
            "st_prototypes": torch.zeros(2, 3),
            "ts_prototypes": torch.zeros(2, 3),
        }
        estimator = SimpleNamespace(
            model_=SimpleNamespace(probe_spec=spec),
            probe_mask_probability=1.0,
            lambda_cross=1.0,
            cross_warmup_epochs=1,
            cross_ramp_epochs=2,
            prototype_alignment_weight=0.0,
            prototype_warmup_epochs=0,
            prototype_ramp_epochs=0,
            prototype_temperature=0.1,
            latent_alignment_weight=0.0,
            lambda_align=0.0,
        )

        _, components = _compute_commutative_pretraining_loss(
            estimator,
            X,
            outputs,
            epoch=20,
            full_probe_mask=True,
        )

        self.assertEqual(components["cross_probe_loss"], 0.0)

    def test_latent_alignment_is_off_by_default_and_optional(self) -> None:
        self.assertEqual(LossWeightConfig().latent_alignment_weight, 0.0)

        X = torch.randn(2, 4, 1, 8, 8)
        spec = ProbeSpec(local_count=3, region_grid=(1, 1, 1), time_bins=2, frequency_bins=2)
        probe_targets = build_probe_targets(X, spec)
        outputs = {
            "pred_A_self": {key: value.clone() for key, value in probe_targets.items()},
            "pred_B_self": {key: value.clone() for key, value in probe_targets.items()},
            "pred_A_to_B": {key: value.clone() for key, value in probe_targets.items()},
            "pred_B_to_A": {key: value.clone() for key, value in probe_targets.items()},
            "st_embedding": torch.zeros(2, 4),
            "ts_embedding": torch.ones(2, 4),
            "st_prototypes": torch.zeros(2, 3),
            "ts_prototypes": torch.zeros(2, 3),
        }
        estimator = SimpleNamespace(
            model_=SimpleNamespace(probe_spec=spec),
            probe_mask_probability=1.0,
            lambda_cross=1.0,
            cross_warmup_epochs=0,
            cross_ramp_epochs=0,
            prototype_alignment_weight=0.0,
            prototype_warmup_epochs=0,
            prototype_ramp_epochs=0,
            prototype_temperature=0.1,
            latent_alignment_weight=0.0,
            lambda_align=0.0,
        )

        default_loss, default_components = _compute_commutative_pretraining_loss(
            estimator,
            X,
            outputs,
            epoch=1,
            full_probe_mask=True,
        )
        estimator.latent_alignment_weight = 0.25
        weighted_loss, weighted_components = _compute_commutative_pretraining_loss(
            estimator,
            X,
            outputs,
            epoch=1,
            full_probe_mask=True,
        )

        self.assertEqual(float(default_components["latent_alignment_weight"]), 0.0)
        self.assertAlmostEqual(float(default_loss.item()), 0.0)
        self.assertEqual(float(weighted_components["latent_alignment_weight"]), 0.25)
        self.assertAlmostEqual(float(weighted_loss.item()), 0.25)


if __name__ == "__main__":
    unittest.main()

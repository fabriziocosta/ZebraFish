from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from src.ml import (
    CommutativeCNNClassifier,
    CommutativeCNNConfig,
    CommutativeTransformerClassifier,
    CommutativeTransformerConfig,
    LossWeightConfig,
    OptimizationConfig,
    TimeChannel3DCNNClassifier,
    TimeChannel3DCNNConfig,
)


class EstimatorSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(8, 4, 2, 16, 16)).astype("float32")
        self.y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.compound = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.concentration = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.optimization = OptimizationConfig(
            batch_size=4,
            epochs=1,
            validation_split=0.25,
            verbose=False,
        )
        self.losses = LossWeightConfig(
            action_weight=1.0,
            compound_weight=0.1,
            concentration_weight=0.1,
            consistency_weight=0.2,
            feature_weight=0.05,
        )

    def _run_estimator(self, estimator) -> None:
        estimator.fit(
            self.X,
            self.y,
            compound_y=self.compound,
            concentration_y=self.concentration,
        )
        preds = estimator.predict(self.X)
        probs = estimator.predict_proba(self.X)
        embeddings = estimator.transform(self.X)
        self.assertEqual(sorted(preds.keys()), ["action", "compound", "concentration"])
        self.assertEqual(sorted(probs.keys()), ["action", "compound", "concentration"])
        self.assertEqual(embeddings.shape[0], len(self.X))
        self.assertTrue(hasattr(estimator, "history_"))
        self.assertGreaterEqual(estimator.best_epoch_, 1)

    def test_time_channel_estimator_with_configs(self) -> None:
        estimator = TimeChannel3DCNNClassifier(
            model_config=TimeChannel3DCNNConfig(conv_channels=(4, 8), embedding_dim=8),
            optimization_config=self.optimization,
            loss_weight_config=self.losses,
        )
        params = estimator.get_params(deep=False)
        self.assertIsInstance(params["loss_weight_config"], LossWeightConfig)
        self.assertIsInstance(params["optimization_config"], OptimizationConfig)
        self.assertIsInstance(params["model_config"], TimeChannel3DCNNConfig)
        self._run_estimator(estimator)

    def test_commutative_cnn_estimator_with_configs(self) -> None:
        estimator = CommutativeCNNClassifier(
            model_config=CommutativeCNNConfig(
                spatial_conv_channels=(4, 8),
                temporal_st_channels=(8,),
                temporal_ts_channels=(8,),
                spatial_agg_channels=(8,),
                patch_size_z=1,
                patch_size_xy=8,
                embedding_dim=8,
                num_prototypes=4,
            ),
            optimization_config=self.optimization,
            loss_weight_config=self.losses,
        )
        params = estimator.get_params(deep=False)
        self.assertIsInstance(params["loss_weight_config"], LossWeightConfig)
        self.assertIsInstance(params["optimization_config"], OptimizationConfig)
        self.assertIsInstance(params["model_config"], CommutativeCNNConfig)
        self._run_estimator(estimator)

    def test_commutative_cnn_pretrain_save_load_and_frozen_head_fit(self) -> None:
        model_config = CommutativeCNNConfig(
            spatial_conv_channels=(4,),
            temporal_st_channels=(4,),
            temporal_ts_channels=(4,),
            spatial_agg_channels=(4,),
            patch_size_z=1,
            patch_size_xy=8,
            embedding_dim=4,
            num_prototypes=4,
        )
        estimator = CommutativeCNNClassifier(
            model_config=model_config,
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
            loss_weight_config=LossWeightConfig(consistency_weight=0.2, feature_weight=0.05),
        )
        estimator.pretrain(self.X, epochs=1)
        self.assertTrue(hasattr(estimator, "pretrain_history_"))
        self.assertTrue(hasattr(estimator, "pretrained_encoder_state_dict_"))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = estimator.save_pretrained_encoder(Path(tmpdir) / "encoder.pt")
            fine_tune_estimator = CommutativeCNNClassifier(
                model_config=model_config,
                optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
                loss_weight_config=LossWeightConfig(consistency_weight=0.2, feature_weight=0.0),
                pretrained_state_path=checkpoint_path,
                freeze_backbone=True,
            )
            fine_tune_estimator.fit(self.X, self.y)
            self.assertTrue(hasattr(fine_tune_estimator, "pretrained_loaded_keys_"))
            self.assertGreater(len(fine_tune_estimator.pretrained_loaded_keys_), 0)
            frozen_parameters = [
                parameter.requires_grad
                for name, parameter in fine_tune_estimator.model_.named_parameters()
                if not name.startswith(("classifier.", "compound_classifier.", "concentration_classifier."))
            ]
            self.assertTrue(frozen_parameters)
            self.assertFalse(any(frozen_parameters))

    def test_commutative_transformer_estimator_with_configs(self) -> None:
        estimator = CommutativeTransformerClassifier(
            model_config=CommutativeTransformerConfig(
                spatial_patch_size_st=(1, 8, 8),
                spatial_patch_size_ts=(1, 8, 8),
                temporal_patch_size_ts=2,
                embed_dim=16,
                num_heads=4,
                st_spatial_depth=1,
                st_temporal_depth=1,
                ts_temporal_depth=1,
                ts_spatial_depth=1,
                embedding_dim=8,
                num_prototypes=4,
            ),
            optimization_config=self.optimization,
            loss_weight_config=self.losses,
        )
        params = estimator.get_params(deep=False)
        self.assertIsInstance(params["loss_weight_config"], LossWeightConfig)
        self.assertIsInstance(params["optimization_config"], OptimizationConfig)
        self.assertIsInstance(params["model_config"], CommutativeTransformerConfig)
        self._run_estimator(estimator)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import io
import unittest
import tempfile
from pathlib import Path

import numpy as np
from torch import nn

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


COMMUTATIVE_HEAD_PREFIXES = (
    "classifier.",
    "water_classifier.",
    "compound_classifier.",
    "concentration_classifier.",
    "st_self_probe_decoder.",
    "ts_self_probe_decoder.",
    "st_cross_probe_decoder.",
    "ts_cross_probe_decoder.",
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

    def test_hierarchical_water_head_composes_action_probabilities(self) -> None:
        estimator = TimeChannel3DCNNClassifier(
            model_config=TimeChannel3DCNNConfig(conv_channels=(4,), embedding_dim=6),
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
            loss_weight_config=LossWeightConfig(water_vs_other_weight=1.0),
        )
        estimator.fit(self.X, np.array([0, 1, 2, 0, 1, 2, 0, 1]))

        probs = estimator.predict_proba(self.X)["action"]

        self.assertEqual(probs.shape, (len(self.X), 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue(np.all((0.0 <= probs) & (probs <= 1.0)))

    def test_transformer_hierarchical_water_head_composes_action_probabilities(self) -> None:
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
            ),
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
            loss_weight_config=LossWeightConfig(water_vs_other_weight=1.0),
        )
        estimator.fit(self.X, np.array([0, 1, 2, 0, 1, 2, 0, 1]))

        probs = estimator.predict_proba(self.X)["action"]

        self.assertEqual(probs.shape, (len(self.X), 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue(np.all((0.0 <= probs) & (probs <= 1.0)))

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
            ),
            optimization_config=self.optimization,
            loss_weight_config=self.losses,
        )
        params = estimator.get_params(deep=False)
        self.assertIsInstance(params["loss_weight_config"], LossWeightConfig)
        self.assertIsInstance(params["optimization_config"], OptimizationConfig)
        self.assertIsInstance(params["model_config"], CommutativeCNNConfig)
        self._run_estimator(estimator)

    def test_commutative_cnn_config_can_select_group_normalization(self) -> None:
        estimator = CommutativeCNNClassifier(
            model_config=CommutativeCNNConfig(
                spatial_conv_channels=(4,),
                temporal_st_channels=(4,),
                temporal_ts_channels=(4,),
                spatial_agg_channels=(4,),
                patch_size_z=1,
                patch_size_xy=8,
                embedding_dim=4,
                normalization="group",
                verbose=False,
            ),
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
        )

        estimator.fit(self.X, self.y)

        normalization_layers = [
            module for module in estimator.model_.modules() if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm3d, nn.GroupNorm))
        ]
        self.assertTrue(normalization_layers)
        self.assertTrue(all(isinstance(module, nn.GroupNorm) for module in normalization_layers))

    def test_commutative_cnn_fit_logs_selected_best_epoch_when_not_early_stopped(self) -> None:
        estimator = CommutativeCNNClassifier(
            model_config=CommutativeCNNConfig(
                spatial_conv_channels=(4,),
                temporal_st_channels=(4,),
                temporal_ts_channels=(4,),
                spatial_agg_channels=(4,),
                patch_size_z=1,
                patch_size_xy=8,
                embedding_dim=4,
                verbose=False,
            ),
            optimization_config=OptimizationConfig(
                batch_size=4,
                epochs=1,
                validation_split=0.0,
                verbose=True,
                early_stopping_patience=None,
                scheduler_patience=None,
            ),
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            estimator.fit(self.X, self.y)

        self.assertIn("select_best epoch=001 best_epoch=001", stdout.getvalue())

    def test_commutative_cnn_pretrain_save_load_and_frozen_head_fit(self) -> None:
        model_config = CommutativeCNNConfig(
            spatial_conv_channels=(4,),
            temporal_st_channels=(4,),
            temporal_ts_channels=(4,),
            spatial_agg_channels=(4,),
            patch_size_z=1,
            patch_size_xy=8,
            embedding_dim=4,
            probe_time_bins=4,
        )
        estimator = CommutativeCNNClassifier(
            model_config=model_config,
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
        )
        estimator.pretrain(self.X, epochs=1)
        self.assertTrue(hasattr(estimator, "pretrain_history_"))
        self.assertTrue(hasattr(estimator, "pretrained_encoder_state_dict_"))
        self.assertIn("train_self_probe_loss", estimator.pretrain_history_.columns)
        self.assertIn("train_cross_probe_loss", estimator.pretrain_history_.columns)
        self.assertIn("train_self_probe_local_loss", estimator.pretrain_history_.columns)
        self.assertIn("train_cross_probe_correlation_loss", estimator.pretrain_history_.columns)
        self.assertEqual(float(estimator.pretrain_history_["train_lambda_cross"].iloc[0]), 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = estimator.save_pretrained_encoder(Path(tmpdir) / "encoder.pt")
            fine_tune_estimator = CommutativeCNNClassifier(
                model_config=model_config,
                optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
                pretrained_state_path=checkpoint_path,
                freeze_backbone=True,
            )
            fine_tune_estimator.fit(self.X, self.y)
            self.assertTrue(hasattr(fine_tune_estimator, "pretrained_loaded_keys_"))
            self.assertGreater(len(fine_tune_estimator.pretrained_loaded_keys_), 0)
            frozen_parameters = [
                parameter.requires_grad
                for name, parameter in fine_tune_estimator.model_.named_parameters()
                if not name.startswith(COMMUTATIVE_HEAD_PREFIXES)
            ]
            self.assertTrue(frozen_parameters)
            self.assertFalse(any(frozen_parameters))

    def test_commutative_cnn_pretrain_loads_pretrained_state_path_for_continuation(self) -> None:
        model_config = CommutativeCNNConfig(
            spatial_conv_channels=(4,),
            temporal_st_channels=(4,),
            temporal_ts_channels=(4,),
            spatial_agg_channels=(4,),
            patch_size_z=1,
            patch_size_xy=8,
            embedding_dim=4,
            probe_time_bins=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            first = CommutativeCNNClassifier(
                model_config=model_config,
                optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
            )
            first.pretrain(self.X, epochs=1)
            checkpoint_path = first.save_pretrained_encoder(Path(tmpdir) / "encoder.pt")

            continuation = CommutativeCNNClassifier(
                model_config=model_config,
                optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
                pretrained_state_path=checkpoint_path,
            )
            continuation.pretrain(self.X, epochs=1)

        self.assertTrue(hasattr(continuation, "pretrained_loaded_keys_"))
        self.assertGreater(len(continuation.pretrained_loaded_keys_), 0)

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
            ),
            optimization_config=self.optimization,
            loss_weight_config=self.losses,
        )
        params = estimator.get_params(deep=False)
        self.assertIsInstance(params["loss_weight_config"], LossWeightConfig)
        self.assertIsInstance(params["optimization_config"], OptimizationConfig)
        self.assertIsInstance(params["model_config"], CommutativeTransformerConfig)
        self._run_estimator(estimator)

    def _assert_commutative_hot_start_reuses_compatible_weights(self, estimator) -> None:
        binary_y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        multiclass_y = np.array([0, 1, 2, 0, 1, 2, 0, 1])

        estimator.fit(self.X, binary_y)
        self.assertEqual(len(estimator.history_), 1)

        estimator.fit(
            self.X,
            multiclass_y,
            compound_y=self.compound,
            concentration_y=self.concentration,
        )

        self.assertEqual(len(estimator.history_), 1)
        self.assertIn("classifier.weight", estimator.hot_start_skipped_keys_)
        self.assertIn("classifier.bias", estimator.hot_start_skipped_keys_)
        self.assertGreater(len(estimator.hot_start_loaded_keys_), 0)
        loaded_non_head_keys = [
            key
            for key in estimator.hot_start_loaded_keys_
            if not key.startswith(COMMUTATIVE_HEAD_PREFIXES)
        ]
        self.assertTrue(loaded_non_head_keys)

    def test_commutative_cnn_hot_start_reuses_compatible_weights(self) -> None:
        estimator = CommutativeCNNClassifier(
            model_config=CommutativeCNNConfig(
                spatial_conv_channels=(4, 8),
                temporal_st_channels=(8,),
                temporal_ts_channels=(8,),
                spatial_agg_channels=(8,),
                patch_size_z=1,
                patch_size_xy=8,
                embedding_dim=8,
            ),
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
            loss_weight_config=self.losses,
            hot_start=True,
        )

        self._assert_commutative_hot_start_reuses_compatible_weights(estimator)

    def test_commutative_transformer_hot_start_reuses_compatible_weights(self) -> None:
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
            ),
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
            loss_weight_config=self.losses,
            hot_start=True,
        )

        self._assert_commutative_hot_start_reuses_compatible_weights(estimator)


if __name__ == "__main__":
    unittest.main()

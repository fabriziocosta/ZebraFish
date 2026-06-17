from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ml import (
    CommutativeCNNConfig,
    CommutativeCNNPretrainingConfig,
    LossWeightConfig,
    OptimizationConfig,
    load_commutative_cnn_pretraining_config,
    write_commutative_cnn_pretraining_config,
)


class CommutativeCNNPretrainingConfigTests(unittest.TestCase):
    def test_write_and_load_pretraining_config(self) -> None:
        config = CommutativeCNNPretrainingConfig(
            unlabeled_dataset_path=Path(".dataset_cache/unlabeled"),
            pretrained_encoder_path=Path("artifacts/pretrained_commutative_cnn/encoder_state_v2.pt"),
            validation_fraction=0.1,
            train_num_random_rotations=1,
            rotation_range_degrees=8.0,
            model_config=CommutativeCNNConfig(
                spatial_conv_channels=(12, 24),
                spatial_kernel_size_z=(3, 3),
                temporal_st_channels=(32, 32),
                temporal_st_kernel_sizes=(5, 3),
                temporal_ts_channels=(24, 32),
                temporal_ts_kernel_sizes=(7, 5),
                spatial_agg_channels=(24, 32),
                spatial_agg_kernel_size_z=(3, 3),
                spatial_agg_stride_z=(1, 1),
                embedding_dim=48,
            ),
            optimization_config=OptimizationConfig(
                batch_size=8,
                epochs=60,
                learning_rate=3e-4,
                early_stopping_patience=10,
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            written_path = write_commutative_cnn_pretraining_config(config, config_path)
            loaded_config = load_commutative_cnn_pretraining_config(written_path)

        self.assertEqual(loaded_config, config)
        self.assertIsInstance(loaded_config.model_config.spatial_conv_channels, tuple)
        self.assertIsInstance(loaded_config.model_config.spatial_kernel_size_z, tuple)

    def test_load_rejects_removed_probe_local_count(self) -> None:
        payload = {
            "unlabeled_dataset_path": ".dataset_cache/unlabeled",
            "pretrained_encoder_path": "artifacts/pretrained_commutative_cnn/encoder_state.pt",
            "validation_fraction": 0.1,
            "train_num_random_rotations": 0,
            "rotation_range_degrees": 0.0,
            "model_config": {"probe_local_count": 32},
            "optimization_config": {},
            "loss_weight_config": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "probe_local_count"):
                load_commutative_cnn_pretraining_config(config_path)


if __name__ == "__main__":
    unittest.main()

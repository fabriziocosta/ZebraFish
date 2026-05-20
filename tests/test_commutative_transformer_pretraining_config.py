from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ml import (
    CommutativeTransformerConfig,
    CommutativeTransformerPretrainingConfig,
    LossWeightConfig,
    OptimizationConfig,
    load_commutative_transformer_pretraining_config,
    write_commutative_transformer_pretraining_config,
)


class CommutativeTransformerPretrainingConfigTests(unittest.TestCase):
    def test_write_and_load_pretraining_config(self) -> None:
        config = CommutativeTransformerPretrainingConfig(
            unlabeled_dataset_path=Path(".dataset_cache/unlabeled"),
            pretrained_encoder_path=Path("artifacts/pretrained_commutative_transformer/encoder_state.pt"),
            model_config=CommutativeTransformerConfig(
                spatial_patch_size_st=(1, 32, 32),
                spatial_patch_size_ts=(1, 32, 32),
                temporal_patch_size_ts=2,
                embed_dim=32,
                num_heads=2,
                embedding_dim=16,
                num_prototypes=8,
            ),
            optimization_config=OptimizationConfig(batch_size=8, epochs=75, learning_rate=1e-4),
            loss_weight_config=LossWeightConfig(consistency_weight=1.0, feature_weight=0.05),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            written_path = write_commutative_transformer_pretraining_config(config, config_path)
            loaded_config = load_commutative_transformer_pretraining_config(written_path)

        self.assertEqual(loaded_config, config)
        self.assertIsInstance(loaded_config.model_config.spatial_patch_size_st, tuple)
        self.assertIsInstance(loaded_config.model_config.spatial_patch_size_ts, tuple)


if __name__ == "__main__":
    unittest.main()

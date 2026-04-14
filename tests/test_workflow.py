from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd
import torch

from src.ml import (
    LossWeightConfig,
    OptimizationConfig,
    TimeChannel3DCNNClassifier,
    TimeChannel3DCNNConfig,
    evaluate_multitask_estimator,
    persist_experiment_artifacts,
    prepare_multitask_experiment_data,
)


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        labels = []
        compound_labels = []
        concentration_labels = []
        for instance_id in range(12):
            label = instance_id % 3
            for replica in range(2):
                rows.append(
                    {
                        "original_instance_id": instance_id,
                        "label": label,
                        "image_condition_dir": f"/tmp/cond_{instance_id}_{replica}",
                        "is_control": bool(label == 0),
                    }
                )
                labels.append(label)
                compound_labels.append(label)
                concentration_labels.append(replica % 2)
        self.dataset = {
            "tensors": torch.randn(len(rows), 4, 2, 8, 8),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "compound_labels": torch.tensor(compound_labels, dtype=torch.int64),
            "concentration_labels": torch.tensor(concentration_labels, dtype=torch.int64),
            "metadata": pd.DataFrame(rows),
            "label_map": {0: "control", 1: "a", 2: "b"},
            "compound_label_map": {0: "control", 1: "ca", 2: "cb"},
            "concentration_label_map": {0: "low", 1: "high"},
        }

    def test_prepare_evaluate_and_persist_workflow(self) -> None:
        experiment = prepare_multitask_experiment_data(
            self.dataset,
            holdout_fraction=0.25,
            validation_fraction_within_train=0.25,
            train_num_random_rotations=1,
            rotation_range_degrees=5.0,
            random_state=0,
        )
        estimator = TimeChannel3DCNNClassifier(
            model_config=TimeChannel3DCNNConfig(conv_channels=(4, 8), embedding_dim=8),
            optimization_config=OptimizationConfig(batch_size=4, epochs=1, validation_split=0.0, verbose=False),
            loss_weight_config=LossWeightConfig(compound_weight=0.1, concentration_weight=0.1),
        )
        estimator.fit(
            experiment.X_train,
            experiment.y_train.to_numpy(),
            validation_data=(experiment.splits.X_val, experiment.splits.y_val),
            compound_y=None if experiment.compound_train is None else experiment.compound_train.to_numpy(),
            concentration_y=None if experiment.concentration_train is None else experiment.concentration_train.to_numpy(),
            validation_compound_y=experiment.splits.compound_val,
            validation_concentration_y=experiment.splits.concentration_val,
        )
        reports = evaluate_multitask_estimator(
            estimator,
            experiment.splits.X_holdout,
            experiment.y_true_holdout,
            label_maps=experiment.label_maps,
            class_labels=experiment.class_labels,
        )
        self.assertIn("action", reports)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = persist_experiment_artifacts(
                output_dir=tmpdir,
                estimator=estimator,
                reports=reports,
                config={
                    "model": "TimeChannel3DCNNClassifier",
                    "dataset_artifact_path": Path("/tmp/example-dataset.pt"),
                },
            )
            self.assertTrue(Path(artifacts.config_path).exists())
            self.assertTrue(Path(artifacts.history_path).exists())
            self.assertTrue(Path(artifacts.summary_metrics_path).exists())
            self.assertTrue(Path(artifacts.checkpoint_path).exists())
            persisted_config = json.loads(Path(artifacts.config_path).read_text(encoding="utf-8"))
            self.assertEqual(persisted_config["dataset_artifact_path"], "/tmp/example-dataset.pt")

    def test_prepare_multitask_experiment_accepts_serialized_metadata_records_payload(self) -> None:
        raw_dataset = {
            key: value for key, value in self.dataset.items() if key != "metadata"
        }
        raw_dataset["metadata_records"] = self.dataset["metadata"].to_dict(orient="records")

        experiment = prepare_multitask_experiment_data(
            raw_dataset,
            holdout_fraction=0.25,
            validation_fraction_within_train=0.25,
            train_num_random_rotations=1,
            rotation_range_degrees=5.0,
            random_state=0,
        )

        self.assertIsNotNone(experiment.train_metadata)
        assert experiment.train_metadata is not None
        self.assertIn("original_instance_id", experiment.train_metadata.columns)


if __name__ == "__main__":
    unittest.main()

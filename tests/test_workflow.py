from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd
import torch

from src.ml import (
    LossWeightConfig,
    OptimizationConfig,
    CommutativeCNNClassifier,
    CommutativeCNNConfig,
    TimeChannel3DCNNClassifier,
    TimeChannel3DCNNConfig,
    evaluate_multitask_estimator,
    fit_chunked_water_vs_other_hot_start,
    persist_experiment_artifacts,
    prepare_multitask_experiment_data,
    prepare_water_vs_other_pretraining_data,
)
from src.training.workflow import build_reports_excluding_control


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

    def test_build_reports_excluding_control_renormalizes_probabilities(self) -> None:
        reports, y_true, y_pred, probabilities = build_reports_excluding_control(
            y_true={"action": [0, 1, 2, 1]},
            probabilities={
                "action": [
                    [0.90, 0.05, 0.05],
                    [0.60, 0.30, 0.10],
                    [0.10, 0.20, 0.70],
                    [0.05, 0.80, 0.15],
                ]
            },
            class_labels={"action": [0, 1, 2]},
            label_maps={"action": {0: "Water", 1: "A", 2: "B"}},
        )

        self.assertEqual(y_true["action"].tolist(), [1, 2, 1])
        self.assertEqual(y_pred["action"].tolist(), [1, 2, 1])
        self.assertTrue(torch.allclose(torch.as_tensor(probabilities["action"].sum(axis=1)), torch.ones(3, dtype=torch.float64)))
        self.assertNotIn("Water", reports["action"][0].index)
        self.assertEqual(float(reports["action"][1].loc["accuracy", "value"]), 1.0)

    def test_prepare_water_vs_other_pretraining_data_excludes_holdout_and_augments_train_only(self) -> None:
        rows = []
        labels = []
        for index in range(10):
            condition_kind = "control" if index % 2 == 0 else "treatment"
            rows.append(
                {
                    "condition_kind": condition_kind,
                    "image_condition_dir": f"/tmp/pretrain_{index}",
                }
            )
            labels.append(0 if condition_kind == "control" else 1)
        unlabeled_dataset = {
            "tensors": torch.randn(10, 4, 2, 8, 8),
            "metadata": pd.DataFrame(rows),
        }
        holdout_metadata = pd.DataFrame(
            {
                "image_condition_dir": [
                    "/tmp/pretrain_0",
                    "/tmp/pretrain_1",
                    "/tmp/supervised_only",
                ]
            }
        )

        binary_data = prepare_water_vs_other_pretraining_data(
            unlabeled_dataset,
            holdout_metadata=holdout_metadata,
            validation_fraction=0.25,
            train_num_random_rotations=1,
            rotation_range_degrees=5.0,
            random_state=0,
        )

        self.assertEqual(binary_data.excluded_holdout_count, 2)
        self.assertEqual(binary_data.label_map, {0: "Water", 1: "Other"})
        self.assertEqual(len(binary_data.X_val), 2)
        self.assertEqual(len(binary_data.X_train), 12)
        self.assertEqual(sorted(binary_data.y_val.tolist()), [0, 1])
        self.assertEqual(sorted(binary_data.y_train.value_counts().tolist()), [6, 6])
        excluded_dirs = set(holdout_metadata["image_condition_dir"])
        self.assertFalse(set(binary_data.train_metadata["image_condition_dir"]).intersection(excluded_dirs))
        self.assertFalse(set(binary_data.val_metadata["image_condition_dir"]).intersection(excluded_dirs))
        self.assertIn("augmentation_index", binary_data.train_metadata.columns)
        self.assertNotIn("augmentation_index", binary_data.val_metadata.columns)

    def test_fit_chunked_water_vs_other_hot_start_uses_chunks_without_concat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            chunk_dir = Path(tmpdir)
            chunk_names = []
            for chunk_index in range(2):
                rows = []
                tensors = []
                for local_index in range(4):
                    global_index = chunk_index * 4 + local_index
                    condition_kind = "control" if global_index % 2 == 0 else "treatment"
                    rows.append(
                        {
                            "condition_kind": condition_kind,
                            "image_condition_dir": f"/tmp/chunked_{global_index}",
                        }
                    )
                    tensors.append(torch.randn(4, 2, 8, 8))
                chunk_name = f"chunk_{chunk_index:02d}.pt"
                torch.save(
                    {
                        "tensors": torch.stack(tensors),
                        "metadata_records": rows,
                    },
                    chunk_dir / chunk_name,
                )
                chunk_names.append(chunk_name)
            (chunk_dir / "manifest.json").write_text(json.dumps({"chunks": chunk_names}), encoding="utf-8")

            estimator = CommutativeCNNClassifier(
                model_config=CommutativeCNNConfig(
                    spatial_conv_channels=(2,),
                    temporal_st_channels=(2,),
                    temporal_ts_channels=(2,),
                    spatial_agg_channels=(2,),
                    embedding_dim=4,
                    patch_size_xy=4,
                    dropout=0.0,
                    verbose=False,
                ),
                optimization_config=OptimizationConfig(
                    batch_size=2,
                    epochs=1,
                    validation_split=0.0,
                    verbose=True,
                    early_stopping_start_epoch=8,
                    early_stopping_patience=None,
                    scheduler_patience=None,
                ),
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = fit_chunked_water_vs_other_hot_start(
                    estimator,
                    chunk_dir,
                    holdout_metadata=pd.DataFrame({"image_condition_dir": ["/tmp/chunked_0"]}),
                    validation_fraction=0.25,
                    epochs=1,
                    random_state=0,
                )

            self.assertEqual(result.excluded_holdout_count, 1)
            self.assertEqual(result.label_map, {0: "Water", 1: "Other"})
            self.assertEqual(result.train_count + result.val_count, 7)
            self.assertTrue(hasattr(estimator, "model_"))
            self.assertEqual(len(estimator.history_), 1)
            self.assertEqual(estimator.best_epoch_, 1)
            self.assertLess(estimator.best_metric_, float("inf"))
            self.assertIn("select_best epoch=001 best_epoch=001", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

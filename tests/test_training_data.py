from __future__ import annotations

import unittest

import pandas as pd
import torch

from src.training.data import split_labeled_tensor_dataset_by_instance


class SplitByInstanceTests(unittest.TestCase):
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
        }

    def test_split_keeps_instances_disjoint(self) -> None:
        splits = split_labeled_tensor_dataset_by_instance(
            self.dataset,
            holdout_fraction=0.25,
            validation_fraction_within_train=0.25,
            random_state=0,
        )
        train_ids = set(map(int, splits.train_instance_ids))
        val_ids = set(map(int, splits.val_instance_ids))
        holdout_ids = set(map(int, splits.holdout_instance_ids))
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & holdout_ids)
        self.assertFalse(val_ids & holdout_ids)
        self.assertEqual(len(splits.X_train_base), len(splits.y_train_base))
        self.assertEqual(len(splits.X_val), len(splits.y_val))
        self.assertEqual(len(splits.X_holdout), len(splits.y_holdout))

    def test_split_accepts_serialized_metadata_records_payload(self) -> None:
        raw_dataset = {
            key: value for key, value in self.dataset.items() if key != "metadata"
        }
        raw_dataset["metadata_records"] = self.dataset["metadata"].to_dict(orient="records")

        splits = split_labeled_tensor_dataset_by_instance(
            raw_dataset,
            holdout_fraction=0.25,
            validation_fraction_within_train=0.25,
            random_state=0,
        )

        self.assertEqual(len(splits.metadata_all), len(self.dataset["metadata"]))
        self.assertIn("original_instance_id", splits.metadata_all.columns)


if __name__ == "__main__":
    unittest.main()

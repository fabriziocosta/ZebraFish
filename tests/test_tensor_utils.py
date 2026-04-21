from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from src import tensor_utils
from src.dataset_config import write_current_dataset_config


class CacheRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.tensor_cache_dir = self.root / ".tensor_cache"
        self.tiff_cache_dir = self.root / ".tiff_cache"
        self.dataset_cache_dir = self.root / ".dataset_cache"
        self.tensor_cache_dir.mkdir()
        self.tiff_cache_dir.mkdir()
        self.dataset_cache_dir.mkdir()

        self.original_project_root = tensor_utils.PROJECT_ROOT
        self.original_tensor_cache_dir = tensor_utils.TENSOR_CACHE_DIR
        self.original_tiff_cache_dir = tensor_utils.TIFF_CACHE_DIR
        self.original_dataset_cache_dir = tensor_utils.DATASET_CACHE_DIR
        self.original_default_cache_budgets = dict(tensor_utils.DEFAULT_CACHE_BUDGETS)
        self.original_cwd = Path.cwd()
        self.original_env = os.environ.copy()

        tensor_utils.PROJECT_ROOT = self.root
        tensor_utils.TENSOR_CACHE_DIR = self.tensor_cache_dir
        tensor_utils.TIFF_CACHE_DIR = self.tiff_cache_dir
        tensor_utils.DATASET_CACHE_DIR = self.dataset_cache_dir
        tensor_utils.DEFAULT_CACHE_BUDGETS = {}
        tensor_utils._CACHE_MAINTENANCE_LAST_RUN.clear()

        os.chdir(self.root)
        os.environ["ZF_TENSOR_CACHE_MAX_BYTES"] = "40"
        os.environ["ZF_TIFF_CACHE_MAX_BYTES"] = "40"
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "40"
        os.environ["ZF_CACHE_MIN_FREE_BYTES"] = "0"
        os.environ["ZF_CACHE_MAX_AGE_SECONDS"] = "3600"
        os.environ["ZF_CACHE_MAINTENANCE_INTERVAL_SECONDS"] = "0"

        self.addCleanup(self._restore_state)

    def _restore_state(self) -> None:
        os.chdir(self.original_cwd)
        os.environ.clear()
        os.environ.update(self.original_env)
        tensor_utils.PROJECT_ROOT = self.original_project_root
        tensor_utils.TENSOR_CACHE_DIR = self.original_tensor_cache_dir
        tensor_utils.TIFF_CACHE_DIR = self.original_tiff_cache_dir
        tensor_utils.DATASET_CACHE_DIR = self.original_dataset_cache_dir
        tensor_utils.DEFAULT_CACHE_BUDGETS = self.original_default_cache_budgets
        tensor_utils._CACHE_MAINTENANCE_LAST_RUN.clear()

    def _write_bytes(self, path: Path, size: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)

    def _build_dataset(self) -> dict[str, object]:
        return {
            "tensors": torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32),
            "labels": torch.tensor([0], dtype=torch.int64),
            "metadata": pd.DataFrame(
                [{"original_instance_id": 0, "image_condition_dir": "/tmp/example", "label": 0}]
            ),
            "label_map": {0: "Water"},
        }

    def test_prune_cache_entries_removes_oldest_unpinned_files_first(self) -> None:
        older_file = self.tensor_cache_dir / "older.pt"
        newer_file = self.tensor_cache_dir / "newer.pt"
        self._write_bytes(older_file, 20)
        self._write_bytes(newer_file, 20)
        now_ns = time.time_ns()
        tensor_utils._write_cache_index(
            self.tensor_cache_dir,
            {
                "older.pt": {"size": 20, "last_used_ns": now_ns - 2_000_000_000},
                "newer.pt": {"size": 20, "last_used_ns": now_ns - 1_000_000_000},
            },
        )

        tensor_utils._prune_cache_entries(self.tensor_cache_dir, incoming_bytes=1, force=True)

        self.assertFalse(older_file.exists())
        self.assertTrue(newer_file.exists())

    def test_prune_cache_entries_respects_max_age(self) -> None:
        os.environ["ZF_CACHE_MAX_AGE_SECONDS"] = "1"
        stale_file = self.tensor_cache_dir / "stale.pt"
        fresh_file = self.tensor_cache_dir / "fresh.pt"
        self._write_bytes(stale_file, 10)
        self._write_bytes(fresh_file, 10)
        now_ns = time.time_ns()
        tensor_utils._write_cache_index(
            self.tensor_cache_dir,
            {
                "stale.pt": {"size": 10, "last_used_ns": now_ns - 5_000_000_000},
                "fresh.pt": {"size": 10, "last_used_ns": now_ns},
            },
        )

        tensor_utils._prune_cache_entries(self.tensor_cache_dir, force=True)

        self.assertFalse(stale_file.exists())
        self.assertTrue(fresh_file.exists())

    def test_prune_cache_entries_preserves_current_dataset_artifact(self) -> None:
        pinned_file = self.dataset_cache_dir / "current.pt"
        stale_file = self.dataset_cache_dir / "stale.pt"
        self._write_bytes(pinned_file, 20)
        self._write_bytes(stale_file, 20)
        write_current_dataset_config(pinned_file, config_path=self.root / "artifacts" / "current_dataset.json")
        tensor_utils._write_cache_index(
            self.dataset_cache_dir,
            {
                "current.pt": {"size": 20, "last_used_ns": 2},
                "stale.pt": {"size": 20, "last_used_ns": 1},
            },
        )

        tensor_utils._prune_cache_entries(self.dataset_cache_dir, incoming_bytes=10, force=True)

        self.assertTrue(pinned_file.exists())
        self.assertFalse(stale_file.exists())

    def test_save_labeled_tensor_dataset_fails_early_when_dataset_exceeds_budget(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1"

        with self.assertRaisesRegex(RuntimeError, "too large for the configured dataset cache budget"):
            tensor_utils.save_labeled_tensor_dataset(self._build_dataset(), "oversized.pt")

    def test_save_labeled_tensor_dataset_fails_early_when_disk_space_is_insufficient(self) -> None:
        disk_usage_type = type(shutil.disk_usage(self.root))
        fake_disk_usage = disk_usage_type(total=1000, used=995, free=5)

        with patch.object(tensor_utils.shutil, "disk_usage", return_value=fake_disk_usage):
            with self.assertRaisesRegex(RuntimeError, "Insufficient free space to save dataset artifact"):
                tensor_utils.save_labeled_tensor_dataset(self._build_dataset(), self.root / "external.pt")

    def test_build_unlabeled_tensor_dataset_filters_and_loads_selected_rows(self) -> None:
        condition_df = pd.DataFrame(
            [
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "A",
                    "condition_kind": "treatment",
                    "compound": "c1",
                    "concentration_band": "high",
                    "concentration_label": "10 uM",
                    "image_condition_dir": "/tmp/a",
                },
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "B",
                    "condition_kind": "treatment",
                    "compound": "c2",
                    "concentration_band": "low",
                    "concentration_label": "1 uM",
                    "image_condition_dir": "/tmp/b",
                },
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "A",
                    "condition_kind": "control",
                    "compound": "c1",
                    "concentration_band": "control",
                    "concentration_label": "water",
                    "image_condition_dir": "/tmp/c",
                },
            ]
        )

        with patch.object(tensor_utils, "describe_condition_tensor_source", return_value="test"), patch.object(
            tensor_utils,
            "load_image_condition_tensor",
            side_effect=lambda **_: torch.zeros((2, 1, 4, 4), dtype=torch.float32),
        ):
            dataset = tensor_utils.build_unlabeled_tensor_dataset(
                condition_df,
                output_size=(2, 1, 4, 4),
                selected_mechanisms=["A"],
                selected_concentrations=["high"],
                verbose=False,
            )

        self.assertEqual(tuple(dataset["tensors"].shape), (2, 2, 1, 4, 4))
        self.assertEqual(dataset["metadata"]["image_condition_dir"].tolist(), ["/tmp/c", "/tmp/a"])

    def test_save_and_load_unlabeled_tensor_dataset(self) -> None:
        dataset = {
            "tensors": torch.zeros((2, 2, 1, 4, 4), dtype=torch.float32),
            "metadata": pd.DataFrame([{"image_condition_dir": "/tmp/a"}, {"image_condition_dir": "/tmp/b"}]),
        }
        path = tensor_utils.save_unlabeled_tensor_dataset(dataset, self.root / "unlabeled.pt")
        loaded = tensor_utils.load_unlabeled_tensor_dataset(path)
        self.assertEqual(tuple(loaded["tensors"].shape), (2, 2, 1, 4, 4))
        self.assertEqual(loaded["metadata"]["image_condition_dir"].tolist(), ["/tmp/a", "/tmp/b"])


if __name__ == "__main__":
    unittest.main()

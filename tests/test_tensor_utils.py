from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

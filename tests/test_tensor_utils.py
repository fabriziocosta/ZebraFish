from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from matplotlib.colors import to_hex

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

    def test_load_image_condition_tensor_reports_tiff_cause_chain(self) -> None:
        condition_dir = self.root / "condition"
        tiff_path = condition_dir / "spim_TL001_Angle0.ome.tiff"
        self._write_bytes(tiff_path, 20)

        def fail_load(*_, **__):
            try:
                raise ValueError("bad page shape")
            except ValueError as exc:
                raise RuntimeError("reader failed") from exc

        with patch.object(tensor_utils, "load_tiff_as_tzyx", side_effect=fail_load):
            with self.assertRaisesRegex(
                RuntimeError,
                "Failed to load TIFF timepoint 1/1.*source_size=20 B.*RuntimeError: reader failed.*ValueError: bad page shape",
            ):
                tensor_utils.load_image_condition_tensor(
                    condition_dir,
                    output_size=(1, 1, 4, 4),
                    use_cache=False,
                    use_tiff_cache=False,
                )

    def test_list_timepoint_files_ignores_tiff_named_directories(self) -> None:
        condition_dir = self.root / "condition"
        tiff_path = condition_dir / "spim_TL001_Angle0.ome.tiff"
        roi_dir = condition_dir / "spim_TL002_Angle0.ome.tiff_ROIS"
        self._write_bytes(tiff_path, 20)
        roi_dir.mkdir()

        self.assertEqual(tensor_utils.list_timepoint_files(condition_dir), [tiff_path])

    def test_load_image_condition_tensor_downsamples_time_after_concatenating_tiffs(self) -> None:
        condition_dir = self.root / "condition"
        for index in range(20):
            self._write_bytes(condition_dir / f"spim_TL{index:03d}_Angle0.ome.tiff", 20)

        with patch.object(
            tensor_utils,
            "load_tiff_as_tzyx",
            side_effect=lambda *_, **__: torch.zeros((2, 5, 4, 4), dtype=torch.float32),
        ):
            tensor = tensor_utils.load_image_condition_tensor(
                condition_dir,
                output_size=(20, 5, 4, 4),
                use_cache=False,
                use_tiff_cache=False,
            )

        self.assertEqual(tuple(tensor.shape), (20, 5, 4, 4))

    def test_load_image_condition_tensor_ignores_cached_tensor_with_wrong_shape(self) -> None:
        condition_dir = self.root / "condition"
        for index in range(20):
            self._write_bytes(condition_dir / f"spim_TL{index:03d}_Angle0.ome.tiff", 20)
        timepoint_files = tensor_utils.list_timepoint_files(condition_dir)
        output_size = (20, 5, 4, 4)
        cache_key = tensor_utils.build_tensor_cache_key(
            condition_dir=condition_dir,
            timepoint_files=timepoint_files,
            output_size=output_size,
            normalize_global_drift=False,
            loess_frac=0.25,
        )
        bad_cache_path = self.tensor_cache_dir / f"{cache_key}.pt"
        bad_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros((269, 5, 4, 4), dtype=torch.float32), bad_cache_path)

        with patch.object(
            tensor_utils,
            "load_tiff_as_tzyx",
            side_effect=lambda *_, **__: torch.ones((1, 5, 4, 4), dtype=torch.float32),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tensor = tensor_utils.load_image_condition_tensor(
                    condition_dir,
                    output_size=output_size,
                    normalize_global_drift=False,
                    use_cache=True,
                    use_tiff_cache=False,
                )

        self.assertEqual(tuple(tensor.shape), output_size)
        self.assertTrue(any("Ignoring cached tensor with unexpected shape" in str(item.message) for item in caught))

    def test_build_unlabeled_tensor_dataset_warning_reports_cause_chain(self) -> None:
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
                }
            ]
        )

        def fail_load(*_, **__):
            try:
                raise ValueError("bad page shape")
            except ValueError as exc:
                raise RuntimeError("reader failed") from exc

        with patch.object(tensor_utils, "describe_condition_tensor_source", return_value="test"), patch.object(
            tensor_utils,
            "load_image_condition_tensor",
            side_effect=fail_load,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaisesRegex(ValueError, "No unlabeled dataset examples were created"):
                    tensor_utils.build_unlabeled_tensor_dataset(
                        condition_df,
                        output_size=(1, 1, 4, 4),
                        verbose=False,
                    )

        self.assertIn("RuntimeError: reader failed", str(caught[0].message))
        self.assertIn("ValueError: bad page shape", str(caught[0].message))

    def test_save_unlabeled_tensor_dataset_prunes_tensor_cache_before_dataset_save(self) -> None:
        os.environ["ZF_TENSOR_CACHE_MAX_BYTES"] = "1"
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1M"
        stale_tensor_cache_file = self.tensor_cache_dir / "stale.pt"
        self._write_bytes(stale_tensor_cache_file, 20)
        tensor_utils._write_cache_index(
            self.tensor_cache_dir,
            {"stale.pt": {"size": 20, "last_used_ns": 1}},
        )
        dataset = {
            "tensors": torch.zeros((1, 2, 1, 4, 4), dtype=torch.float32),
            "metadata": pd.DataFrame([{"image_condition_dir": "/tmp/a"}]),
        }

        tensor_utils.save_unlabeled_tensor_dataset(dataset, ".dataset_cache/unlabeled.pt")

        self.assertFalse(stale_tensor_cache_file.exists())
        self.assertTrue((self.dataset_cache_dir / "unlabeled.pt").exists())

    def test_build_unlabeled_tensor_dataset_writes_and_loads_chunks(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1M"
        condition_df = pd.DataFrame(
            [
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "A",
                    "condition_kind": "treatment",
                    "compound": f"c{index}",
                    "concentration_band": "high",
                    "concentration_label": "10 uM",
                    "image_condition_dir": f"/tmp/{index}",
                }
                for index in range(5)
            ]
        )

        with patch.object(tensor_utils, "describe_condition_tensor_source", return_value="test"), patch.object(
            tensor_utils,
            "load_image_condition_tensor",
            side_effect=[
                torch.full((2, 1, 4, 4), fill_value=index, dtype=torch.float32)
                for index in range(5)
            ],
        ):
            chunked = tensor_utils.build_unlabeled_tensor_dataset(
                condition_df,
                output_size=(2, 1, 4, 4),
                chunk_output_dir=".dataset_cache/unlabeled_chunks",
                chunk_size=2,
                verbose=False,
            )

        self.assertEqual(len(chunked["chunk_paths"]), 3)
        self.assertTrue((self.dataset_cache_dir / "unlabeled_chunks" / "manifest.json").exists())

        loaded = tensor_utils.load_unlabeled_tensor_dataset(".dataset_cache/unlabeled_chunks")
        self.assertEqual(tuple(loaded["tensors"].shape), (5, 2, 1, 4, 4))
        self.assertEqual(loaded["tensors"][:, 0, 0, 0, 0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(loaded["metadata"]["image_condition_dir"].tolist(), [f"/tmp/{index}" for index in range(5)])

    def test_build_unlabeled_tensor_dataset_persists_and_skips_failed_conditions(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1M"
        condition_df = pd.DataFrame(
            [
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "A",
                    "condition_kind": "treatment",
                    "compound": f"c{index}",
                    "concentration_band": "high",
                    "concentration_label": "10 uM",
                    "image_condition_dir": f"/tmp/{index}",
                }
                for index in range(3)
            ]
        )

        def load_or_fail(*, condition_dir, **_):
            index = int(Path(condition_dir).name)
            if index == 1:
                raise IsADirectoryError("bad roi directory")
            return torch.full((2, 1, 4, 4), fill_value=index, dtype=torch.float32)

        with patch.object(tensor_utils, "describe_condition_tensor_source", return_value="test"), patch.object(
            tensor_utils,
            "load_image_condition_tensor",
            side_effect=load_or_fail,
        ) as first_load:
            chunked = tensor_utils.build_unlabeled_tensor_dataset(
                condition_df,
                output_size=(2, 1, 4, 4),
                chunk_output_dir=".dataset_cache/unlabeled_chunks",
                chunk_size=2,
                verbose=False,
            )

        failure_log_path = self.dataset_cache_dir / "unlabeled_chunks" / "failed_conditions.json"
        self.assertTrue(failure_log_path.exists())
        self.assertEqual(first_load.call_count, 3)
        self.assertEqual(chunked["metadata"]["image_condition_dir"].tolist(), ["/tmp/0", "/tmp/2"])
        (self.dataset_cache_dir / "unlabeled_chunks" / "manifest.json").unlink()

        with patch.object(tensor_utils, "describe_condition_tensor_source", return_value="test"), patch.object(
            tensor_utils,
            "load_image_condition_tensor",
            side_effect=AssertionError("previously failed row should not be retried"),
        ) as second_load:
            resumed = tensor_utils.build_unlabeled_tensor_dataset(
                condition_df,
                output_size=(2, 1, 4, 4),
                chunk_output_dir=".dataset_cache/unlabeled_chunks",
                chunk_size=2,
                verbose=False,
            )

        self.assertEqual(second_load.call_count, 0)
        self.assertEqual(resumed["metadata"]["image_condition_dir"].tolist(), ["/tmp/0", "/tmp/2"])

    def test_build_unlabeled_tensor_dataset_resumes_existing_chunks_without_manifest(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1M"
        condition_df = pd.DataFrame(
            [
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "A",
                    "condition_kind": "treatment",
                    "compound": f"c{index}",
                    "concentration_band": "high",
                    "concentration_label": "10 uM",
                    "image_condition_dir": f"/tmp/{index}",
                }
                for index in range(5)
            ]
        )
        chunk_dir = self.dataset_cache_dir / "unlabeled_chunks"
        chunk_dir.mkdir(parents=True)
        existing_dataset = {
            "tensors": torch.stack(
                [torch.full((2, 1, 4, 4), fill_value=index, dtype=torch.float32) for index in range(2)],
                dim=0,
            ),
            "metadata": pd.DataFrame(
                [
                    {
                        "original_instance_id": index,
                        "mechanism_of_action": "A",
                        "compound": f"c{index}",
                        "condition_kind": "treatment",
                        "concentration_band": "high",
                        "concentration_label": "10 uM",
                        "image_condition_dir": f"/tmp/{index}",
                    }
                    for index in range(2)
                ]
            ),
        }
        tensor_utils.save_unlabeled_tensor_dataset(existing_dataset, chunk_dir / "unlabeled_chunk_0001.pt")

        with patch.object(tensor_utils, "describe_condition_tensor_source", return_value="test"), patch.object(
            tensor_utils,
            "load_image_condition_tensor",
            side_effect=[
                torch.full((2, 1, 4, 4), fill_value=index, dtype=torch.float32)
                for index in range(2, 5)
            ],
        ) as load_tensor:
            chunked = tensor_utils.build_unlabeled_tensor_dataset(
                condition_df,
                output_size=(2, 1, 4, 4),
                chunk_output_dir=".dataset_cache/unlabeled_chunks",
                chunk_size=2,
                verbose=False,
            )

        self.assertEqual(load_tensor.call_count, 3)
        self.assertEqual([path.name for path in chunked["chunk_paths"]], [
            "unlabeled_chunk_0001.pt",
            "unlabeled_chunk_0002.pt",
            "unlabeled_chunk_0003.pt",
        ])
        loaded = tensor_utils.load_unlabeled_tensor_dataset(".dataset_cache/unlabeled_chunks")
        self.assertEqual(loaded["tensors"][:, 0, 0, 0, 0].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(loaded["metadata"]["image_condition_dir"].tolist(), [f"/tmp/{index}" for index in range(5)])

    def test_build_unlabeled_tensor_dataset_fills_missing_chunk_numbers(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1M"
        condition_df = pd.DataFrame(
            [
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "A",
                    "condition_kind": "treatment",
                    "compound": f"c{index}",
                    "concentration_band": "high",
                    "concentration_label": "10 uM",
                    "image_condition_dir": f"/tmp/{index}",
                }
                for index in range(8)
            ]
        )
        chunk_dir = self.dataset_cache_dir / "unlabeled_chunks"
        chunk_dir.mkdir(parents=True)
        existing_dataset = {
            "tensors": torch.stack(
                [torch.full((2, 1, 4, 4), fill_value=index, dtype=torch.float32) for index in range(4, 6)],
                dim=0,
            ),
            "metadata": pd.DataFrame(
                [
                    {
                        "original_instance_id": index,
                        "mechanism_of_action": "A",
                        "compound": f"c{index}",
                        "condition_kind": "treatment",
                        "concentration_band": "high",
                        "concentration_label": "10 uM",
                        "image_condition_dir": f"/tmp/{index}",
                    }
                    for index in range(4, 6)
                ]
            ),
        }
        tensor_utils.save_unlabeled_tensor_dataset(existing_dataset, chunk_dir / "unlabeled_chunk_0003.pt")

        def load_tensor(*, condition_dir, **_):
            index = int(Path(condition_dir).name)
            return torch.full((2, 1, 4, 4), fill_value=index, dtype=torch.float32)

        with patch.object(tensor_utils, "describe_condition_tensor_source", return_value="test"), patch.object(
            tensor_utils,
            "load_image_condition_tensor",
            side_effect=load_tensor,
        ) as load_tensor_mock:
            chunked = tensor_utils.build_unlabeled_tensor_dataset(
                condition_df,
                output_size=(2, 1, 4, 4),
                chunk_output_dir=".dataset_cache/unlabeled_chunks",
                chunk_size=2,
                verbose=False,
            )

        self.assertEqual(load_tensor_mock.call_count, 6)
        self.assertEqual([path.name for path in chunked["chunk_paths"]], [
            "unlabeled_chunk_0001.pt",
            "unlabeled_chunk_0002.pt",
            "unlabeled_chunk_0003.pt",
            "unlabeled_chunk_0004.pt",
        ])
        loaded = tensor_utils.load_unlabeled_tensor_dataset(".dataset_cache/unlabeled_chunks")
        self.assertEqual(loaded["tensors"][:, 0, 0, 0, 0].tolist(), list(range(8)))

    def test_build_unlabeled_tensor_dataset_writes_manifest_when_chunks_are_complete(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1M"
        condition_df = pd.DataFrame(
            [
                {
                    "condition_folder_status": "active",
                    "mechanism_of_action": "A",
                    "condition_kind": "treatment",
                    "compound": f"c{index}",
                    "concentration_band": "high",
                    "concentration_label": "10 uM",
                    "image_condition_dir": f"/tmp/{index}",
                }
                for index in range(2)
            ]
        )
        chunk_dir = self.dataset_cache_dir / "unlabeled_chunks"
        chunk_dir.mkdir(parents=True)
        existing_dataset = {
            "tensors": torch.stack(
                [torch.full((2, 1, 4, 4), fill_value=index, dtype=torch.float32) for index in range(2)],
                dim=0,
            ),
            "metadata": pd.DataFrame(
                [
                    {
                        "original_instance_id": index,
                        "mechanism_of_action": "A",
                        "compound": f"c{index}",
                        "condition_kind": "treatment",
                        "concentration_band": "high",
                        "concentration_label": "10 uM",
                        "image_condition_dir": f"/tmp/{index}",
                    }
                    for index in range(2)
                ]
            ),
        }
        tensor_utils.save_unlabeled_tensor_dataset(existing_dataset, chunk_dir / "unlabeled_chunk_0001.pt")

        with patch.object(tensor_utils, "load_image_condition_tensor") as load_tensor:
            chunked = tensor_utils.build_unlabeled_tensor_dataset(
                condition_df,
                output_size=(2, 1, 4, 4),
                chunk_output_dir=".dataset_cache/unlabeled_chunks",
                chunk_size=2,
                verbose=False,
            )

        self.assertEqual(load_tensor.call_count, 0)
        self.assertEqual(len(chunked["chunk_paths"]), 1)
        self.assertTrue((chunk_dir / "manifest.json").exists())

    def test_dataset_cache_relative_path_is_not_double_prefixed(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "1M"
        dataset = {
            "tensors": torch.zeros((1, 2, 1, 4, 4), dtype=torch.float32),
            "metadata": pd.DataFrame([{"image_condition_dir": "/tmp/a"}]),
        }
        path = tensor_utils.save_unlabeled_tensor_dataset(dataset, ".dataset_cache/unlabeled.pt")
        self.assertEqual(path, self.root / ".dataset_cache" / "unlabeled.pt")

        loaded = tensor_utils.load_unlabeled_tensor_dataset(".dataset_cache/unlabeled.pt")
        self.assertEqual(tuple(loaded["tensors"].shape), (1, 2, 1, 4, 4))
        self.assertEqual(loaded["metadata"]["image_condition_dir"].tolist(), ["/tmp/a"])

    def test_prune_cache_entries_ignores_files_removed_during_scan(self) -> None:
        missing_path = self.tiff_cache_dir / "missing.ome.tiff"
        with patch.object(tensor_utils, "_list_cache_files", return_value=[missing_path]):
            tensor_utils._prune_cache_entries(self.tiff_cache_dir, force=True)

    def test_dataset_cache_directory_pin_keeps_chunk_files(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "10"
        chunk_dir = self.dataset_cache_dir / "unlabeled_chunks"
        pinned_chunk = chunk_dir / "unlabeled_chunk_0001.pt"
        unpinned_file = self.dataset_cache_dir / "old.pt"
        self._write_bytes(pinned_chunk, 30)
        self._write_bytes(unpinned_file, 30)
        os.environ["ZF_PINNED_DATASET_PATHS"] = str(chunk_dir)

        tensor_utils._prune_cache_entries(self.dataset_cache_dir, incoming_bytes=30, force=True)

        self.assertTrue(pinned_chunk.exists())
        self.assertFalse(unpinned_file.exists())

    def test_dataset_cache_manifest_keeps_chunk_files(self) -> None:
        os.environ["ZF_DATASET_CACHE_MAX_BYTES"] = "10"
        chunk_dir = self.dataset_cache_dir / "unlabeled_chunks"
        pinned_chunk = chunk_dir / "unlabeled_chunk_0001.pt"
        unpinned_file = self.dataset_cache_dir / "old.pt"
        self._write_bytes(pinned_chunk, 30)
        self._write_bytes(unpinned_file, 30)
        (chunk_dir / "manifest.json").write_text(
            json.dumps({"chunks": ["unlabeled_chunk_0001.pt"]}),
            encoding="utf-8",
        )

        tensor_utils._prune_cache_entries(self.dataset_cache_dir, incoming_bytes=30, force=True)

        self.assertTrue(pinned_chunk.exists())
        self.assertTrue((chunk_dir / "manifest.json").exists())
        self.assertFalse(unpinned_file.exists())

    def test_plot_tensor_embedding_2d_uses_distinct_current_action_colors(self) -> None:
        embedding_df = pd.DataFrame(
            {
                "embed_x": [0.0, 1.0, 2.0, 3.0, 4.0],
                "embed_y": [0.0, 1.0, 2.0, 3.0, 4.0],
                "label": [0, 1, 2, 3, 4],
                "label_name": [
                    "Water",
                    "GABAAR_Antagonist",
                    "NMDAR_Activation",
                    "AChE_Inhibitor_Reversible",
                    "mAChR_Agonist_NonSelective",
                ],
                "method": ["pca"] * 5,
            }
        )

        fig, ax = tensor_utils.plot_tensor_embedding_2d(
            embedding_df,
            marker_column=None,
        )
        legend = ax.get_legend()
        colors = {
            text.get_text().split(": ", 1)[1]: to_hex(handle.get_markerfacecolor()).upper()
            for text, handle in zip(legend.get_texts(), legend.legend_handles)
        }

        self.assertEqual(colors["NMDAR_Activation"], "#59A14F")
        self.assertEqual(colors["AChE_Inhibitor_Reversible"], "#B07AA1")
        self.assertNotEqual(colors["NMDAR_Activation"], colors["AChE_Inhibitor_Reversible"])
        fig.clf()


if __name__ == "__main__":
    unittest.main()

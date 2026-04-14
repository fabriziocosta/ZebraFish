from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.dataset_config import load_current_dataset_artifact_path, write_current_dataset_config


class DatasetConfigTests(unittest.TestCase):
    def test_write_and_load_current_dataset_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "datasets" / "example.pt"
            config_path = tmp_path / "artifacts" / "current_dataset.json"

            written_config_path = write_current_dataset_config(dataset_path, config_path=config_path)
            loaded_dataset_path = load_current_dataset_artifact_path(config_path=written_config_path)

            self.assertEqual(written_config_path, config_path)
            self.assertEqual(loaded_dataset_path, dataset_path.resolve())


if __name__ == "__main__":
    unittest.main()

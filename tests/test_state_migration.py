from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.state_migration import migrate_campaign, rebuild_compatibility_views
from src.scientific_state import load_state


class StateMigrationTests(unittest.TestCase):
    def test_migration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_root = root / "campaigns"
            trial = campaign_root / "trial_1"
            trial.mkdir(parents=True)
            (trial / "trial_manifest.json").write_text(
                '{"trial_id":"trial_1","stages":["10C","13C"],"trial_configs":{},"stage_runs":{}}',
                encoding="utf-8",
            )
            config = {
                "campaign": {"id": "test", "stages": ["10C", "13C"]},
                "artifacts": {"root": str(campaign_root)},
                "scientific_state": {"path": str(root / "scientific_state.yaml")},
            }
            first = migrate_campaign(config)
            second = migrate_campaign(config)
            self.assertEqual(first["imported_trials"], 1)
            self.assertEqual(second["imported_trials"], 0)
            state = load_state(root / "scientific_state.yaml")
            self.assertEqual(len(state["entities"]["trials"]), 1)
            self.assertEqual(len(state["entities"]["experiments"]), 2)
            views = rebuild_compatibility_views(config)
            self.assertTrue(Path(views["trials_csv"]).exists())
            self.assertTrue(Path(views["leaderboard_csv"]).exists())
            self.assertTrue(Path(views["logbook"]).exists())


if __name__ == "__main__":
    unittest.main()

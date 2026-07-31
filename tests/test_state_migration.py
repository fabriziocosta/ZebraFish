from __future__ import annotations

import json
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
            run_dir = trial / "outputs" / "13C" / "runs" / "run-1"
            (run_dir / "figures").mkdir(parents=True)
            (run_dir / "confusion_matrices").mkdir()
            (run_dir / "figures" / "holdout_umap.csv").write_text("x,y\n0,0\n", encoding="utf-8")
            (run_dir / "confusion_matrices" / "action_confusion_counts.csv").write_text(
                "true,pred,count\nA,A,1\n",
                encoding="utf-8",
            )
            (trial / "trial_manifest.json").write_text(
                json.dumps({
                    "trial_id": "trial_1",
                    "stages": ["10C", "13C"],
                    "trial_configs": {},
                    "stage_runs": {"13C": str(run_dir)},
                }),
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
            legacy = [
                observation
                for observation in state["entities"]["observations"].values()
                if observation.get("type") == "legacy_domain_artifacts"
            ]
            self.assertEqual(len(legacy), 1)
            self.assertFalse(legacy[0]["measurements"]["umap_used_for_decision"])
            views = rebuild_compatibility_views(config)
            self.assertTrue(Path(views["trials_csv"]).exists())
            self.assertTrue(Path(views["leaderboard_csv"]).exists())
            self.assertTrue(Path(views["logbook"]).exists())


if __name__ == "__main__":
    unittest.main()

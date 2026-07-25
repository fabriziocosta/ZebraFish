from __future__ import annotations

import unittest
from pathlib import Path

from src.dashboard_data import build_investigation


class DashboardDataTests(unittest.TestCase):
    def test_live_campaign_is_normalized_without_mutation(self) -> None:
        payload = build_investigation(Path.cwd(), "cnn", level=3, relation_depth=1)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["campaign"]["id"], "cnn_pretrain_finetune")
        self.assertIn(payload["investigation"]["status"], {"running", "stalled", "completed", "awaiting results"})
        self.assertIsNotNone(payload["current_experiment"]["stage"])
        self.assertGreaterEqual(len(payload["current_experiment"]["metric_series"]), 1)
        self.assertIn("supporting", payload["evidence"])
        self.assertIn("contradicting", payload["evidence"])
        self.assertIn("inconclusive", payload["evidence"])
        self.assertIn("nodes", payload["graph"])

    def test_stale_safe_stop_metadata_is_not_reported_as_stopped_when_process_is_live(self) -> None:
        payload = build_investigation(Path.cwd(), "cnn")
        if payload["current_experiment"]["process_running"]:
            self.assertEqual(payload["investigation"]["status"], "running")
            stale = [alert for alert in payload["alerts"] if alert["type"] == "stale_controller_metadata"]
            self.assertTrue(stale)
            self.assertEqual(stale[0]["severity"], "warning")


if __name__ == "__main__":
    unittest.main()

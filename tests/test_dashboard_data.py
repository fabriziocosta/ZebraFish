from __future__ import annotations

import unittest
from pathlib import Path

from src.dashboard_data import (
    _classify_observation,
    _evidence,
    _hypothesis_quality,
    _metric_descriptor,
    build_investigation,
)


class DashboardDataTests(unittest.TestCase):
    def test_live_campaign_is_normalized_without_mutation(self) -> None:
        payload = build_investigation(Path.cwd(), "cnn", level=3, relation_depth=1)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["campaign"]["id"], "cnn_pretrain_finetune")
        self.assertIn(payload["investigation"]["status"], {"running", "terminated", "stalled", "completed", "awaiting results"})
        self.assertIsNotNone(payload["current_experiment"]["stage"])
        self.assertIsInstance(payload["current_experiment"]["metric_series"], list)
        self.assertIn("supporting", payload["evidence"])
        self.assertIn("contradicting", payload["evidence"])
        self.assertIn("inconclusive", payload["evidence"])
        self.assertIn("unclassified", payload["evidence"])
        self.assertIn("metric_display", payload["current_experiment"])
        self.assertIn("metric_plot", payload["current_experiment"])
        self.assertIn("statistics", payload["current_experiment"]["metric_plot"])
        self.assertIn("events", payload["current_experiment"]["metric_plot"])
        self.assertIn("interpretation", payload["current_experiment"]["metric_plot"])
        self.assertIn("data_coverage", payload["diagnostics"])
        if payload["current_experiment"]["stage"] == "10C":
            self.assertIn(payload["current_experiment"]["metric_display"]["role"], {"diagnostic", "unavailable"})
            if payload["current_experiment"]["metric_display"]["role"] == "diagnostic":
                self.assertEqual(payload["current_experiment"]["metric_display"]["display_metric"], "val_loss")
            self.assertTrue(payload["diagnostics"]["data_coverage"]["missing_metric"])
        self.assertIn("nodes", payload["graph"])
        if payload["current_experiment"]["stage"] == "13C":
            self.assertIn(payload["current_experiment"]["eta_status"], {"available", "warming_up", "not_applicable"})
            if not payload["current_experiment"]["process_running"]:
                self.assertEqual(payload["current_experiment"]["eta_status"], "not_applicable")

    def test_stale_safe_stop_metadata_is_not_reported_as_stopped_when_process_is_live(self) -> None:
        payload = build_investigation(Path.cwd(), "cnn")
        if payload["current_experiment"]["process_running"]:
            self.assertEqual(payload["investigation"]["status"], "running")
            stale = [alert for alert in payload["alerts"] if alert["type"] == "stale_controller_metadata"]
            if stale:
                self.assertEqual(stale[0]["severity"], "warning")
                self.assertEqual(payload["health"]["controller_metadata"], "stale")
            self.assertTrue(payload["health"]["process_live"])

    def test_evidence_direction_requires_explicit_classification(self) -> None:
        state = {
            "entities": {"observations": {
                "obs_support": {"statement": "support"},
                "obs_contra": {"statement": "contra"},
                "obs_inconclusive": {"statement": "uncertain", "direction": "inconclusive"},
                "obs_unclassified": {"statement": "unknown", "source_experiments": ["exp-1"]},
            }},
            "relations": [
                {"type": "supports", "source": "obs_support", "target": "hyp-1"},
                {"type": "contradicts", "source": "hyp-1", "target": "obs_contra"},
            ],
        }
        self.assertEqual(_classify_observation(state, "obs_support", "hyp-1")[0], "supporting")
        self.assertEqual(_classify_observation(state, "obs_contra", "hyp-1")[0], "contradicting")
        self.assertEqual(_classify_observation(state, "obs_inconclusive", "hyp-1")[0], "inconclusive")
        self.assertEqual(_classify_observation(state, "obs_unclassified", "hyp-1")[0], "unclassified")
        evidence = _evidence(state, "hyp-1")
        self.assertEqual({item["direction"] for item in evidence}, {"supporting", "contradicting", "inconclusive", "unclassified"})
        self.assertEqual(next(item for item in evidence if item["id"] == "obs_unclassified")["classification_source"], "unavailable")

    def test_generic_seed_and_metric_fallback_are_explicit(self) -> None:
        quality = _hypothesis_quality({"title": "A bounded optimisation change can improve the campaign objective."})
        self.assertEqual(quality["quality"], "generic_seed")
        descriptor = _metric_descriptor([{"val_loss": 1.2, "train_loss": 1.0}], "compound.macro_f1")
        self.assertEqual(descriptor["role"], "diagnostic")
        self.assertEqual(descriptor["display_metric"], "val_loss")
        self.assertIn("macro_f1", descriptor["fallback_reason"])


if __name__ == "__main__":
    unittest.main()

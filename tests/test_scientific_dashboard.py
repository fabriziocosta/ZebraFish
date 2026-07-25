from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.scientific_dashboard import build_dashboard_snapshot, build_reasoning_graph, nx
from src.scientific_state import empty_state, record_entity, save_state


class ScientificDashboardTests(unittest.TestCase):
    def _state(self):
        state = empty_state()
        state["project"] = {"objective": "improve compound macro F1"}
        state = record_entity(
            state,
            "questions",
            "q_1",
            {"question": "Does regularisation improve generalisation?", "status": "open"},
        )
        state = record_entity(
            state,
            "hypotheses",
            "h_1",
            {"statement": "Moderate regularisation reduces the validation gap.", "status": "active"},
        )
        state = record_entity(
            state,
            "trials",
            "trial_1",
            {
                "status": "completed",
                "purpose": {"question_id": "q_1", "hypothesis_ids": ["h_1"]},
                "stage_experiment_ids": ["exp_1"],
                "outcome": {"score": 0.61, "guardrail_passed": True},
            },
        )
        state = record_entity(
            state,
            "experiments",
            "exp_1",
            {"status": "completed", "trial_id": "trial_1", "stage": "10C"},
        )
        state = record_entity(state, "experiments", "exp_10", {"status": "completed", "trial_id": "trial_10", "stage": "10C"})
        state = record_entity(state, "experiments", "exp_2", {"status": "completed", "trial_id": "trial_2", "stage": "10C"})
        state = record_entity(state, "experiments", "exp_unobserved", {"status": "completed", "trial_id": "trial_unobserved", "stage": "10C"})
        state = record_entity(
            state,
            "observations",
            "obs_1",
            {
                "type": "generalisation_gap",
                "statement": "Validation trails training; macro f1=-0.252809...",
                "source_experiments": ["exp_1", "exp_10", "exp_2"],
            },
        )
        state = record_entity(
            state,
            "candidate_experiments",
            "cand_1",
            {"purpose": "test bounded regularisation", "status": "proposed", "question_id": "q_1"},
        )
        state = record_entity(state, "trials", "trial_10", {"status": "completed", "outcome": {"score": 0.98765}})
        state = record_entity(state, "trials", "trial_2", {"status": "completed", "outcome": {"score": 0.12345}})
        state = record_entity(state, "trials", "trial_unobserved", {"status": "completed", "outcome": {"score": 0.88}})
        state["relations"].append({"type": "tests", "source": "h_1", "target": "trial_1"})
        return state

    def test_graph_granularity_controls_evidence(self) -> None:
        if nx is None:
            self.skipTest("NetworkX is not installed")
        state = self._state()
        coarse = build_reasoning_graph(state, level=0)
        detailed = build_reasoning_graph(state, level=5)
        self.assertNotIn("observations:obs_1", coarse.nodes)
        self.assertIn("trials:trial_1", coarse.nodes)
        self.assertNotIn("trials:trial_unobserved", coarse.nodes)
        self.assertNotIn("experiments:exp_unobserved", detailed.nodes)
        self.assertIn("observations:obs_1", detailed.nodes)
        self.assertTrue(any(data["relation"] == "produced" for *_edge, data in detailed.edges(data=True)))
        self.assertIn("TRIAL 1", detailed.nodes["trials:trial_1"]["label"])
        self.assertIn("score: 0.61", detailed.nodes["trials:trial_1"]["label"])
        self.assertIn("TRIAL 2", detailed.nodes["trials:trial_10"]["label"])
        self.assertIn("score: 0.988", detailed.nodes["trials:trial_10"]["label"])
        self.assertIn("TRIAL 3", detailed.nodes["trials:trial_2"]["label"])
        self.assertIn("Validation trails training", detailed.nodes["observations:obs_1"]["label"])
        self.assertIn("-0.253", detailed.nodes["observations:obs_1"]["label"])
        self.assertIn("macro f1=-0.253...", detailed.nodes["observations:obs_1"]["tooltip"])
        self.assertIn("Does regularisation improve generalisation?", detailed.nodes["questions:q_1"]["tooltip"])
        self.assertIn("Moderate regularisation reduces the validation gap.", detailed.nodes["hypotheses:h_1"]["tooltip"])

    def test_snapshot_reads_current_run_and_clamps_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state" / "scientific_state.yaml"
            run_status_path = root / "run_status.json"
            run_status_path.write_text(
                json.dumps({"status": "running", "run_dir": str(root / "run"), "checkpoint_path": "checkpoints/latest.pt"}),
                encoding="utf-8",
            )
            save_state(state_path, self._state())
            campaign_path = root / "campaign_state.json"
            campaign_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "cnn",
                        "status": "running",
                        "phase": "running",
                        "current_trial_id": "trial_1",
                        "current_stage": "10C",
                        "active_launch_state": {
                            "pid": 123,
                            "runner": "pretrain",
                            "run_status_path": str(run_status_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            snapshot = build_dashboard_snapshot(state_path=state_path, campaign_state_path=campaign_path, level=99)
            self.assertEqual(snapshot["level"], 5)
            self.assertEqual(snapshot["current_run"]["campaign_id"], "cnn")
            self.assertEqual(snapshot["current_run"]["run_status"], "running")
            self.assertEqual(snapshot["current_run"]["pid"], 123)
            self.assertEqual(len(snapshot["trial_rows"]), 4)
            self.assertEqual(len(snapshot["observation_rows"]), 1)


if __name__ == "__main__":
    unittest.main()

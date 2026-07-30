from __future__ import annotations

import io
import unittest
from unittest import mock

from src.autonomous_campaign import _apply_decision
from src.autonomous_campaign import _ensure_seed_knowledge
from src.autonomous_campaign import parse_decision
from src.scientific_state import empty_state


class AutonomousCampaignTests(unittest.TestCase):
    def _configs(self):
        loop = {"experiments": {"10C": {"allowed_patch_paths": ["optimization_config"]}}}
        campaign = {
            "campaign": {"id": "test", "stages": ["10C"], "max_patch_leaf_count": 2, "max_single_trial_gpu_hours": 20, "max_decision_retries": 3},
            "objective": {"primary_metric": "macro_f1"},
        }
        return loop, campaign

    def test_valid_candidate_is_launched_without_approval(self) -> None:
        loop, campaign = self._configs()
        scientific = _ensure_seed_knowledge(empty_state(), campaign)
        legacy_state = {"status": "trial_completed", "current_trial_id": "old"}
        candidate = {
            "id": "candidate_1",
            "purpose": "test schedule",
            "question_id": "q_test_objective",
            "hypothesis_ids": ["hyp_test_objective"],
            "configuration_patch": {"10C": {"optimization_config": {"epochs": 100}}},
            "estimated_gpu_hours": 2,
            "expected_outcomes": ["higher compound score"],
            "falsification_criteria": ["lower compound score"],
            "risks": ["runtime"],
        }
        with mock.patch("src.agent_campaign_loop.start_trial", return_value={"current_trial_id": "new", "status": "running"}) as launch:
            updated, next_legacy, reason = _apply_decision(
                campaign,
                loop,
                scientific,
                legacy_state,
                {"decision": "propose_trial", "reason": "test", "operations": [], "candidate": candidate},
                stream=io.StringIO(),
            )
        launch.assert_called_once()
        self.assertEqual(next_legacy["current_trial_id"], "new")
        self.assertIn("launched", reason)
        self.assertIn("candidate_1", updated["entities"]["candidate_experiments"])

    def test_unsafe_candidate_is_rejected_without_launch(self) -> None:
        loop, campaign = self._configs()
        scientific = _ensure_seed_knowledge(empty_state(), campaign)
        candidate = {
            "id": "candidate_bad",
            "purpose": "architecture change",
            "question_id": "q_test_objective",
            "hypothesis_ids": ["hyp_test_objective"],
            "configuration_patch": {"10C": {"model_config": {"embedding_dim": 128}}},
            "estimated_gpu_hours": 2,
            "expected_outcomes": ["higher score"],
            "falsification_criteria": ["lower score"],
            "risks": ["architecture"],
        }
        with mock.patch("src.agent_campaign_loop.start_trial") as launch:
            updated, _legacy_state, reason = _apply_decision(
                campaign,
                loop,
                scientific,
                {"status": "trial_completed"},
                {"decision": "propose_trial", "reason": "test", "operations": [], "candidate": candidate},
                stream=io.StringIO(),
            )
        launch.assert_not_called()
        self.assertIn("rejected candidate", reason)
        self.assertIn("last_rejected_candidate", updated["controller_state"])

    def test_empty_expected_old_is_treated_as_null(self) -> None:
        decision = parse_decision(
            '{"decision":"no_action","reason":"ok","evidence_references":[],'
            '"operations":[{"operation":"update","path":"controller_state.status",'
            '"value":"\\"running\\"","expected_old":""}],"candidate":null}'
        )
        self.assertIsNone(decision["operations"][0]["expected_old"])


if __name__ == "__main__":
    unittest.main()

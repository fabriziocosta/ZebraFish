from __future__ import annotations

import unittest

from src.autonomous_policy import validate_candidate
from src.scientific_state import empty_state, record_entity


class AutonomousPolicyTests(unittest.TestCase):
    def _state(self):
        state = record_entity(empty_state(), "questions", "q_1", {"question": "what helps?"})
        return record_entity(state, "hypotheses", "hyp_1", {"statement": "optimisation helps"})

    def _configs(self):
        loop = {"experiments": {"10C": {"allowed_patch_paths": ["optimization_config"]}}}
        campaign = {"campaign": {"stages": ["10C"], "max_patch_leaf_count": 2, "max_single_trial_gpu_hours": 20}}
        return loop, campaign

    def test_accepts_bounded_candidate(self) -> None:
        loop, campaign = self._configs()
        result = validate_candidate(
            {
                "id": "candidate_1",
                "purpose": "test schedule",
                "question_id": "q_1",
                "hypothesis_ids": ["hyp_1"],
                "configuration_patch": {"10C": {"optimization_config": {"epochs": 100}}},
                "estimated_gpu_hours": 3,
                "expected_outcomes": ["higher score"],
                "falsification_criteria": ["lower score"],
                "risks": ["runtime"],
            },
            loop_config=loop,
            campaign_config=campaign,
            state=self._state(),
        )
        self.assertTrue(result.valid)

    def test_rejects_unallowlisted_candidate(self) -> None:
        loop, campaign = self._configs()
        result = validate_candidate(
            {
                "id": "candidate_1",
                "purpose": "architecture change",
                "question_id": "q_1",
                "hypothesis_ids": ["hyp_1"],
                "configuration_patch": {"10C": {"model_config": {"embedding_dim": 128}}},
                "estimated_gpu_hours": 3,
                "expected_outcomes": ["higher score"],
                "falsification_criteria": ["lower score"],
                "risks": ["runtime"],
            },
            loop_config=loop,
            campaign_config=campaign,
            state=self._state(),
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("non-allowlisted" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()

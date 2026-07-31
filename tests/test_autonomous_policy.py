from __future__ import annotations

import unittest

from src.autonomous_policy import validate_candidate
from src.domain_guidance import load_domain_contract
from src.experiment_protocol import config_hash
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

    def test_rejects_value_outside_campaign_range(self) -> None:
        loop, campaign = self._configs()
        campaign["parameter_ranges"] = {"10C": {"optimization_config": {"epochs": {"min": 20, "max": 200}}}}
        result = validate_candidate(
            {
                "id": "candidate_1",
                "purpose": "too long",
                "question_id": "q_1",
                "hypothesis_ids": ["hyp_1"],
                "configuration_patch": {"10C": {"optimization_config": {"epochs": 1000}}},
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
        self.assertIn("parameter above maximum: 10C.optimization_config.epochs", result.reasons)

    def test_protocol_candidate_must_preregister_every_domain_constraint(self) -> None:
        state = self._state()
        resolved = {"optimization_config": {"epochs": 100}}
        state = record_entity(
            state,
            "experiments",
            "base",
            {
                "configuration": {"resolved_configuration": resolved},
                "provenance": {"created_by": "test"},
            },
        )
        loop = {
            "experiments": {
                "10C": {
                    "allowed_patch_paths": ["optimization_config.epochs"],
                    "parameter_ranges": {"optimization_config": {"epochs": {"min": 20, "max": 200}}},
                }
            }
        }
        campaign = {
            "campaign": {
                "stages": ["10C"],
                "evaluation_protocol": "three_seed_replicate_lockbox_v1",
                "require_protocol_compliance": True,
                "replicate_seeds": [0, 1, 2],
                "minimum_replicates": 3,
                "max_patch_leaf_count": 2,
                "max_single_trial_gpu_hours": 20,
            },
            "domain_guidance": {
                "enabled": True,
                "contract_path": "configs/domain_guidance/cnn_action_domain_v1.yaml",
                "candidate_family_id": "domain-family",
            },
            "parameter_ranges": {"10C": {"optimization_config": {"epochs": {"min": 20, "max": 200}}}},
        }
        candidate = {
            "id": "candidate-domain",
            "candidate_kind": "intervention",
            "purpose": "test a bounded schedule",
            "question_id": "q_1",
            "hypothesis_ids": ["hyp_1"],
            "base_experiment": "base",
            "base_stage": "10C",
            "configuration_patch": {"10C": {"optimization_config": {"epochs": 101}}},
            "fixed_variables": {},
            "varied_variables": {"optimization_config": {"epochs": 101}},
            "resolved_base_configuration_hash": config_hash(resolved),
            "source_checkpoint": "checkpoint-id",
            "baseline": "base",
            "replicate_seeds": [0, 1, 2],
            "estimated_gpu_hours": 3,
            "expected_outcomes": [{"metric": "compound.macro_f1", "comparison": "paired_baseline", "direction": "increase", "minimum_effect": 0.02}],
            "falsification_criteria": [{"metric": "compound.macro_f1", "comparison": "paired_baseline", "direction": "increase", "minimum_effect": 0.02}],
            "risks": ["runtime"],
            "allowed_stages": ["10C"],
        }
        missing = validate_candidate(
            candidate,
            loop_config=loop,
            campaign_config=campaign,
            state=state,
        )
        self.assertFalse(missing.valid)
        self.assertIn("candidate requires domain_expectations for the active domain contract", missing.reasons)
        contract = load_domain_contract(campaign["domain_guidance"]["contract_path"])
        candidate["domain_expectations"] = [
            {
                "constraint_id": constraint["id"],
                "comparison": "paired_baseline",
                "direction": constraint["checks"][0]["direction"],
                "role": constraint["role"],
            }
            for constraint in contract["constraints"]
        ]
        state = record_entity(
            state,
            "domain_calibrations",
            "calibration-1",
            {
                "id": "calibration-1",
                "status": "frozen",
                "contract_hash": contract["_hash"],
                "candidate_family_id": "domain-family",
                "provenance": {"created_by": "test"},
            },
        )
        complete = validate_candidate(
            candidate,
            loop_config=loop,
            campaign_config=campaign,
            state=state,
        )
        self.assertTrue(complete.valid, complete.reasons)


if __name__ == "__main__":
    unittest.main()

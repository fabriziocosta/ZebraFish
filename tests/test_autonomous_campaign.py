from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.autonomous_campaign import _apply_decision
from src.autonomous_campaign import _ensure_seed_knowledge
from src.autonomous_campaign import parse_decision
from src.scientific_state import empty_state, record_entity


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

    def test_single_stage_13c_candidate_reuses_completed_10c_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop = {
                "experiments": {
                    "10C": {"allowed_patch_paths": ["optimization_config"]},
                    "13C": {"allowed_patch_paths": ["train_num_random_rotations", "rotation_range_degrees"]},
                }
            }
            campaign = {
                "campaign": {
                    "id": "test",
                    "stages": ["10C", "13C"],
                    "max_patch_leaf_count": 2,
                    "max_single_trial_gpu_hours": 20,
                    "max_decision_retries": 3,
                },
                "objective": {"primary_metric": "macro_f1"},
            }
            scientific = _ensure_seed_knowledge(empty_state(), campaign)
            source_id = "old_trial:10C"
            run_dir = root / "old_run"
            run_dir.mkdir()
            checkpoint = run_dir / "encoder_state.pt"
            checkpoint.write_bytes(b"checkpoint")
            resolved_config = run_dir / "resolved_config.yaml"
            resolved_config.write_text("pretrained_encoder_path: encoder_state.pt\n", encoding="utf-8")
            scientific = record_entity(
                scientific,
                "experiments",
                source_id,
                {
                    "id": source_id,
                    "trial_id": "old_trial",
                    "stage": "10C",
                    "status": "completed",
                    "configuration": {"resolved_config_paths": [str(resolved_config)]},
                    "execution": {
                        "run_dir": str(run_dir),
                        "completed_at": "2026-07-30T09:00:00",
                        "artifacts": {"checkpoints": [str(checkpoint)]},
                    },
                },
                actor="test",
            )
            candidate = {
                "id": "candidate_13c",
                "purpose": "test rotation augmentation",
                "question_id": "q_test_objective",
                "hypothesis_ids": ["hyp_test_objective"],
                "configuration_patch": {"13C": {"train_num_random_rotations": 8, "rotation_range_degrees": 20}},
                "allowed_stages": ["13C"],
                "estimated_gpu_hours": 2,
                "expected_outcomes": ["higher compound score"],
                "falsification_criteria": ["no improvement"],
                "risks": ["overfitting"],
            }
            with mock.patch("src.agent_campaign_loop.start_trial", return_value={"current_trial_id": "new", "status": "running"}) as launch:
                updated, next_legacy, reason = _apply_decision(
                    campaign,
                    loop,
                    scientific,
                    {"status": "trial_completed", "current_trial_id": "old"},
                    {"decision": "propose_trial", "reason": "test", "operations": [], "candidate": candidate},
                    stream=io.StringIO(),
                )
            self.assertEqual(next_legacy["current_trial_id"], "new")
            self.assertIn("launched candidate candidate_13c", reason)
            kwargs = launch.call_args.kwargs
            self.assertEqual(kwargs["start_stage"], "13C")
            self.assertEqual(kwargs["checkpoint_source_experiment"], source_id)
            self.assertEqual(kwargs["checkpoint_path"], str(checkpoint))
            self.assertIn("candidate_13c", updated["entities"]["candidate_experiments"])

    def test_single_stage_candidate_without_checkpoint_is_rejected(self) -> None:
        loop = {
            "experiments": {
                "10C": {"allowed_patch_paths": ["optimization_config"]},
                "13C": {"allowed_patch_paths": ["train_num_random_rotations"]},
            }
        }
        campaign = {
            "campaign": {"id": "test", "stages": ["10C", "13C"], "max_patch_leaf_count": 1, "max_decision_retries": 3},
        }
        scientific = _ensure_seed_knowledge(empty_state(), campaign)
        candidate = {
            "id": "candidate_no_checkpoint",
            "purpose": "test rotation augmentation",
            "question_id": "q_test_objective",
            "hypothesis_ids": ["hyp_test_objective"],
            "configuration_patch": {"13C": {"train_num_random_rotations": 8}},
            "allowed_stages": ["13C"],
            "estimated_gpu_hours": 2,
            "expected_outcomes": ["higher score"],
            "falsification_criteria": ["lower score"],
            "risks": ["overfitting"],
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
        self.assertIn("no completed compatible 10C checkpoint", reason)
        self.assertIn("last_rejected_candidate", updated["controller_state"])


if __name__ == "__main__":
    unittest.main()

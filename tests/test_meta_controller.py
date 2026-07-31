from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.meta_controller import (
    MetaDecisionError,
    PatchSafetyError,
    collect_snapshot,
    compact_architecture_summary,
    parse_decision,
    read_architecture,
    read_mandate,
    request_run_now,
    run_once,
    _meta_loop_owns_state,
    _validate_decision_evidence,
    _execute_campaign_control,
    validate_patch,
    validate_verifications,
    apply_patch_in_worktree,
)
from src.scientific_state import empty_state, load_state, record_entity, save_state, ImmutableEntityError, apply_operations


def valid_decision(version: str = "sha256:test") -> dict:
    return {
        "mandate_version": version,
        "architecture_version": "sha256:architecture-test",
        "decision": "no_action",
        "diagnosis": {
            "summary": "The controller is healthy.",
            "severity": "info",
            "evidence_references": ["controller_state"],
            "root_causes": [],
        },
        "actions": [],
        "proposal_only_changes": [],
        "rollback_plan": "No patch was applied.",
        "unresolved_risks": [],
    }


class _Response:
    def __init__(self, text: str) -> None:
        self.output_text = text


class _Client:
    class _Responses:
        def __init__(self, text: str) -> None:
            self.text = text

        def create(self, **_: object) -> _Response:
            return _Response(self.text)

    def __init__(self, text: str) -> None:
        self.responses = self._Responses(text)


class MetaControllerTests(unittest.TestCase):
    def test_mandate_is_read_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mandate = root / "docs/meta_controller_mandate.md"
            mandate.parent.mkdir()
            mandate.write_text("# mandate\n", encoding="utf-8")
            content, version, path = read_mandate(root, {"mandate_path": "docs/meta_controller_mandate.md"})
            self.assertEqual(content, "# mandate\n")
            self.assertTrue(version.startswith("sha256:"))
            self.assertEqual(path, str(mandate))

    def test_system_architecture_is_read_hashed_and_summarized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            architecture = root / "docs/system_architecture.md"
            architecture.parent.mkdir()
            architecture.write_text(
                "# Architecture\n\n## Compact supervisory summary\n\n- State is canonical.\n- Artifacts are evidence.\n\n## Other\n\nprivate detail\n",
                encoding="utf-8",
            )
            content, version, path = read_architecture(root, {"architecture_path": "docs/system_architecture.md"})
            self.assertEqual(content, architecture.read_text(encoding="utf-8"))
            self.assertTrue(version.startswith("sha256:"))
            self.assertEqual(path, str(architecture))
            summary = compact_architecture_summary(content)
            self.assertIn("State is canonical", summary)
            self.assertNotIn("private detail", summary)

    def test_strict_decision_rejects_missing_fields(self) -> None:
        with self.assertRaises(MetaDecisionError):
            parse_decision({"decision": "no_action"})

    def test_patch_allowlist_and_limits(self) -> None:
        config = {
            "allowed_paths": ["src/meta_controller.py"],
            "forbidden_prefixes": ["state/"],
            "max_patch_bytes": 2000,
            "max_changed_files": 2,
        }
        patch = "--- a/src/meta_controller.py\n+++ b/src/meta_controller.py\n@@ -1 +1 @@\n-a\n+b\n"
        self.assertEqual(validate_patch(patch, config), ("src/meta_controller.py",))
        with self.assertRaises(PatchSafetyError):
            validate_patch(patch.replace("src/meta_controller.py", "src/training/loop.py"), config)

    def test_domain_contract_and_campaign_guidance_are_protected(self) -> None:
        config = {
            "allowed_paths": ["configs/experiment_campaigns/*.yaml"],
            "forbidden_prefixes": ["configs/domain_guidance/"],
            "max_patch_bytes": 4000,
            "max_changed_files": 2,
        }
        campaign_patch = (
            "--- a/configs/experiment_campaigns/cnn_campaign.yaml\n"
            "+++ b/configs/experiment_campaigns/cnn_campaign.yaml\n"
            "@@ -1 +1,2 @@\n"
            " campaign:\n"
            "+domain_guidance: disabled\n"
        )
        with self.assertRaises(PatchSafetyError):
            validate_patch(campaign_patch, config)
        contract_patch = (
            "--- a/configs/domain_guidance/cnn_action_domain_v1.yaml\n"
            "+++ b/configs/domain_guidance/cnn_action_domain_v1.yaml\n"
            "@@ -1 +1 @@\n"
            "-version: 1\n"
            "+version: 2\n"
        )
        with self.assertRaises(PatchSafetyError):
            validate_patch(contract_patch, config)

    def test_verification_commands_are_allowlisted(self) -> None:
        config = {"verification": [{"name": "tests", "argv": ["python", "-m", "pytest"]}]}
        self.assertEqual(len(validate_verifications(["tests"], config)), 1)
        with self.assertRaises(PatchSafetyError):
            validate_verifications(["rm-all"], config)

    def test_patch_requires_at_least_one_verification(self) -> None:
        config = {"allowed_paths": ["src/meta_controller.py"], "forbidden_prefixes": [], "max_patch_bytes": 2000, "max_changed_files": 2}
        patch = "--- a/src/meta_controller.py\n+++ b/src/meta_controller.py\n@@ -1 +1 @@\n-a\n+b\n"
        with self.assertRaises(PatchSafetyError):
            apply_patch_in_worktree(Path.cwd(), config, patch, "test", verification_names=[])

    def test_meta_runs_are_immutable_state_records(self) -> None:
        state = record_entity(
            empty_state(),
            "meta_controller_runs",
            "meta_1",
            {"status": "completed", "provenance": {"created_by": "test"}},
        )
        with self.assertRaises(ImmutableEntityError):
            apply_operations(state, [{"operation": "update", "path": "entities.meta_controller_runs.meta_1.status", "value": "failed"}])

    def test_run_once_persists_report_without_modifying_campaign_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/meta_controller_mandate.md").write_text("# mandate\n", encoding="utf-8")
            (root / "docs/system_architecture.md").write_text("# architecture\n\n## Compact supervisory summary\n\n- State is canonical.\n", encoding="utf-8")
            (root / "state").mkdir()
            save_state(root / "state/scientific_state.yaml", empty_state())
            config = {
                "campaign": {"id": "test_campaign"},
                "scientific_state": {"path": "state/scientific_state.yaml"},
                "artifacts": {"state_path": "artifacts/campaign_state.json"},
                "agent": {"model": "test"},
                "meta_controller": {"mandate_path": "docs/meta_controller_mandate.md", "report_root": "state/meta_controller/reports"},
            }
            first = valid_decision()
            # The version is computed from the temporary mandate, so make the
            # fake client return it after the controller has loaded it.
            _, version, _ = read_mandate(root, config["meta_controller"])
            first["mandate_version"] = version
            _, architecture_version, _ = read_architecture(root, config["meta_controller"])
            first["architecture_version"] = architecture_version
            report = run_once(root, config, client=_Client(__import__("json").dumps(first)))
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["architecture_version"], architecture_version)
            loaded = load_state(root / "state/scientific_state.yaml")
            self.assertIn(report["id"], loaded["entities"]["meta_controller_runs"])
            self.assertEqual(loaded["controller_state"]["meta_controller"]["last_run_id"], report["id"])

    def test_snapshot_reports_process_and_recent_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_state(root / "state/scientific_state.yaml", empty_state())
            config = {"campaign": {"id": "test"}, "scientific_state": {"path": "state/scientific_state.yaml"}, "artifacts": {"state_path": "artifacts/state.json"}}
            snapshot = collect_snapshot(root, config)
            self.assertFalse(snapshot["process"]["running"])
            self.assertEqual(snapshot["campaign"]["id"], "test")

    def test_evidence_validation_accepts_snapshot_field_assertions(self) -> None:
        snapshot = {
            "campaign": {"status": "running", "stage": "13C"},
            "process": {"running": True, "pid": 1234},
            "recent_meta_reports": [{"controller_status": {"summary": "older"}}, {"controller_status": {"summary": "latest"}}],
            "known_evidence_ids": ["obs_1"],
        }
        decision = {
            "diagnosis": {
                "evidence_references": [
                    "campaign.status=running",
                    "campaign.stage",
                    "process.running=true",
                    "recent_meta_reports[1].controller_status.summary=latest",
                    "obs_1",
                ]
            }
        }
        _validate_decision_evidence(decision, snapshot)

    def test_evidence_validation_rejects_missing_or_mismatched_snapshot_fields(self) -> None:
        snapshot = {"campaign": {"status": "running"}, "known_evidence_ids": []}
        with self.assertRaisesRegex(MetaDecisionError, "unavailable evidence"):
            _validate_decision_evidence({"diagnosis": {"evidence_references": ["campaign.phase"]}}, snapshot)
        with self.assertRaisesRegex(MetaDecisionError, "does not match snapshot"):
            _validate_decision_evidence({"diagnosis": {"evidence_references": ["campaign.status=stopped"]}}, snapshot)

    def test_meta_loop_ownership_follows_current_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "state/scientific_state.yaml"
            state_path.parent.mkdir()
            state = empty_state()
            state["controller_state"]["meta_controller"] = {"pid": 222}
            save_state(state_path, state)
            config = {"scientific_state": {"path": "state/scientific_state.yaml"}}
            self.assertTrue(_meta_loop_owns_state(root, config, 222))
            self.assertFalse(_meta_loop_owns_state(root, config, 111))

    def test_stop_control_requires_replacement_launch(self) -> None:
        import src.agent_campaign_loop as campaign_loop
        import src.autonomous_campaign as autonomous

        config = {"campaign": {"id": "test"}, "meta_controller": {"stop_grace_seconds": 1}}
        with patch.object(campaign_loop, "terminate_campaign", return_value=0), patch.object(
            autonomous, "run_autonomous_campaign", return_value=0
        ) as recovery, patch.object(campaign_loop, "campaign_live_status", return_value={"running": True}):
            result = _execute_campaign_control(Path.cwd(), config, "stop", {"stop_grace_seconds": 1}, client=object())
        recovery.assert_called_once()
        self.assertTrue(result["replacement_running"])
        self.assertEqual(result["returncode"], 0)

    def test_run_now_wakes_scheduler_without_spawning_duplicate_cycle(self) -> None:
        config = {"campaign": {"id": "test"}}
        meta = {
            "pid": 1234,
            "mode": "loop",
            "run_now_supported": True,
            "status": "running",
        }
        with patch("src.meta_controller._meta_state", return_value={"controller_state": {"meta_controller": meta}}), patch(
            "src.meta_controller._pid_running", return_value=True
        ), patch("src.meta_controller.os.kill") as kill, patch("src.meta_controller._set_meta_status") as set_status:
            result = request_run_now(Path.cwd(), config)
        self.assertEqual(result["status"], "requested")
        kill.assert_called_once()
        set_status.assert_called_once()

    def test_run_now_reports_busy_for_existing_one_shot(self) -> None:
        config = {"campaign": {"id": "test"}}
        with patch(
            "src.meta_controller._meta_state",
            return_value={"controller_state": {"meta_controller": {"pid": 1234, "mode": "once"}}},
        ), patch("src.meta_controller._pid_running", return_value=True):
            result = request_run_now(Path.cwd(), config)
        self.assertEqual(result["status"], "already_running")


if __name__ == "__main__":
    unittest.main()

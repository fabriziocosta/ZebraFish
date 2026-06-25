from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.agent_campaign_loop import (
    CampaignDecision,
    _acquire_campaign_lock,
    _release_campaign_lock,
    _render_pdf_preview_png,
    apply_campaign_decision,
    collect_init_snapshot,
    initialize_campaign_trial,
    load_campaign_config,
    parse_campaign_decision,
    run_campaign,
    score_metrics,
    start_trial,
    wire_pretrain_checkpoint_into_finetune_config,
)
from src.agent_experiment_loop import load_loop_config


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse(self.output_text)


class _FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


class AgentCampaignLoopTests(unittest.TestCase):
    def _write_loop_config(self, root: Path) -> Path:
        (root / "configs").mkdir()
        (root / "params10.yaml").write_text("optimization_config:\n  epochs: 10\n", encoding="utf-8")
        (root / "params13.yaml").write_text(
            "pretraining_config_path: old/config.yaml\noptimization_config:\n  epochs: 5\n",
            encoding="utf-8",
        )
        path = root / "loop.yaml"
        path.write_text(
            f"""
agent:
  model: gpt-5.3-codex
  reasoning_effort: medium
  poll_seconds: 18000
  api_key_env: OPENAI_API_KEY
state:
  path: {root / 'single_state.json'}
  log_dir: {root / 'single_logs'}
logbook:
  path: {root / 'logbook.md'}
experiments:
  "10C":
    runner: 10C_pretrain_commutative_cnn_encoder.py
    params_yaml: {root / 'params10.yaml'}
    artifact_root: {root / 'pretrain_artifacts'}
    allowed_patch_paths: [optimization_config]
    required_completion_artifacts: [history, summary_metrics, checkpoint]
    next: "13C"
  "13C":
    runner: 13C_finetune_pretrained_commutative_cnn_classifier.py
    params_yaml: {root / 'params13.yaml'}
    artifact_root: {root / 'finetune_artifacts'}
    allowed_patch_paths: [optimization_config, loss_weight_config]
    required_completion_artifacts: [history, summary_metrics, checkpoint, confusion_matrices, umap_pdf]
    next: null
prompts:
  status_decision: return JSON
""",
            encoding="utf-8",
        )
        return path

    def _write_campaign_config(self, root: Path, loop_config: Path) -> Path:
        path = root / "campaign.yaml"
        path.write_text(
            f"""
campaign:
  id: test_campaign
  loop_config: {loop_config}
  poll_seconds: 18000
  stages: ["10C", "13C"]
objective:
  target: compound
  primary_metric: macro_f1
  fallback_metrics: [accuracy, roc_auc_ovr_macro]
  minimums:
    action.accuracy: 0.30
artifacts:
  root: {root / 'campaign_artifacts'}
logbook:
  path: {root / 'logbook.md'}
prompts:
  analysis_policy: analyze completed trials
  update_logbook: write links
  patch_parameters: small patches only
  decision_schema: return JSON
""",
            encoding="utf-8",
        )
        return path

    def test_load_campaign_config_defaults_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            config = load_campaign_config(campaign_path)
            self.assertEqual(config["campaign"]["poll_seconds"], 18000)
            self.assertTrue(config["artifacts"]["state_path"].endswith("campaign_state.json"))
            self.assertEqual(config["objective"]["target"], "compound")

    def test_score_metrics_prefers_primary_and_checks_guardrail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "summary_metrics.csv"
            metrics_path.write_text(
                "target,metric,value\n"
                "action,accuracy,0.41\n"
                "compound,macro_f1,0.22\n"
                "compound,roc_auc_ovr_macro,0.7\n",
                encoding="utf-8",
            )
            scored = score_metrics(
                metrics_path,
                {
                    "target": "compound",
                    "primary_metric": "macro_f1",
                    "fallback_metrics": ["roc_auc_ovr_macro"],
                    "minimums": {"action.accuracy": 0.30},
                },
            )
            self.assertEqual(scored["score"], 0.22)
            self.assertEqual(scored["ranking_score"], 0.22)
            self.assertEqual(scored["selected_metric"], "compound.macro_f1")
            self.assertTrue(scored["guardrail_passed"])

    def test_score_metrics_excludes_guardrail_failure_from_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "summary_metrics.csv"
            metrics_path.write_text(
                "target,metric,value\n"
                "action,accuracy,0.10\n"
                "compound,macro_f1,0.90\n",
                encoding="utf-8",
            )
            scored = score_metrics(
                metrics_path,
                {
                    "target": "compound",
                    "primary_metric": "macro_f1",
                    "minimums": {"action.accuracy": 0.30},
                },
            )
            self.assertEqual(scored["score"], 0.90)
            self.assertIsNone(scored["ranking_score"])
            self.assertFalse(scored["guardrail_passed"])

    def test_wire_pretrain_checkpoint_into_finetune_config_uses_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pretrain_run = root / "pretrain_run"
            pretrain_run.mkdir()
            pretrain_config = pretrain_run / "10C_config.yaml"
            pretrain_config.write_text("pretrained_encoder_path: ckpt.pt\n", encoding="utf-8")
            finetune_config = root / "13C.yaml"
            finetune_config.write_text("pretraining_config_path: old.yaml\n", encoding="utf-8")
            selected = wire_pretrain_checkpoint_into_finetune_config(finetune_config, pretrain_run)
            self.assertEqual(selected, pretrain_config)
            self.assertIn(str(pretrain_config), finetune_config.read_text(encoding="utf-8"))

    def test_initialize_campaign_calls_model_writes_logbook_and_launches_patched_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            loop_config = load_loop_config(loop_path)
            campaign_config["agent"] = dict(loop_config["agent"])
            output = json.dumps(
                {
                    "decision": "propose_trial",
                    "reason": "lower epochs for smoke",
                    "logbook_markdown": "Try a short first campaign trial.",
                    "trial_patch": {"10C": {"optimization_config": {"epochs": 3}}},
                }
            )
            client = _FakeClient(output)

            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                launch.return_value = {"active_experiment": "10C", "pid": 123}
                stream = io.StringIO()
                state = initialize_campaign_trial(
                    campaign_config,
                    loop_config,
                    start_trial_id="trial_a",
                    client=client,
                    stream=stream,
                )

            self.assertEqual(state["current_trial_id"], "trial_a")
            self.assertEqual(state["current_stage"], "10C")
            patched_config = Path(state["trial_configs"]["10C"]).read_text(encoding="utf-8")
            self.assertIn("epochs: 3", patched_config)
            self.assertIn("Try a short first campaign trial", (root / "logbook.md").read_text(encoding="utf-8"))
            self.assertIn("campaign initialization analysis: trial_a", stream.getvalue())
            self.assertIn("Try a short first campaign trial", stream.getvalue())
            self.assertIn("Initialization snapshot JSON", client.responses.kwargs["input"])

    def test_initialize_campaign_rejects_non_allowlisted_patch_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            loop_config = load_loop_config(loop_path)
            campaign_config["agent"] = dict(loop_config["agent"])
            client = _FakeClient(
                json.dumps(
                    {
                        "decision": "propose_trial",
                        "reason": "unsafe",
                        "trial_patch": {"10C": {"model_config": {"embedding_dim": 128}}},
                    }
                )
            )
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch, self.assertRaisesRegex(
                ValueError, "non-allowlisted"
            ):
                initialize_campaign_trial(campaign_config, loop_config, client=client)
            launch.assert_not_called()

    def test_parse_campaign_decision_rejects_malformed_output(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_campaign_decision("not json")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_campaign_decision(json.dumps({"decision": "run_shell", "reason": "bad"}))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_campaign_decision(json.dumps({"decision": "no_action", "reason": ""}))

    def test_dry_run_once_does_not_call_openai(self) -> None:
        class RaisingClient:
            @property
            def responses(self):
                raise AssertionError("OpenAI should not be called during campaign dry-run")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            stream = io.StringIO()
            code = run_campaign(
                campaign_config,
                once=True,
                dry_run=True,
                start_trial_id="dry_trial",
                client=RaisingClient(),
                stream=stream,
            )
            self.assertEqual(code, 0)
            self.assertEqual(stream.getvalue().count("next poll:"), 1)
            self.assertIn("campaign poll", stream.getvalue())

    def test_collect_init_snapshot_includes_logbook_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            (root / "logbook.md").write_text("old result\nnext idea\n", encoding="utf-8")
            campaign_config = load_campaign_config(campaign_path)
            loop_config = load_loop_config(loop_path)
            snapshot = collect_init_snapshot(campaign_config, loop_config)
            self.assertIn("10C", snapshot["stage_statuses"])
            self.assertIn("next idea", snapshot["logbook_tail"])

    def test_start_trial_respects_campaign_trial_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            campaign_config["campaign"]["trial_budget"] = 1
            trials_csv = Path(campaign_config["artifacts"]["trials_csv"])
            trials_csv.parent.mkdir(parents=True)
            trials_csv.write_text(
                "campaign_id,trial_id,status,score,selected_metric,guardrail_passed,trial_dir,metrics_path,updated_at\n"
                "test_campaign,old,trial_completed,0.1,compound.macro_f1,True,/tmp/old,/tmp/m.csv,now\n",
                encoding="utf-8",
            )
            loop_config = load_loop_config(loop_path)
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch, self.assertRaisesRegex(
                RuntimeError, "budget exhausted"
            ):
                start_trial(campaign_config, loop_config, dry_run=False)
            launch.assert_not_called()

    def test_start_trial_counts_existing_trial_directories_against_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            campaign_config["campaign"]["trial_budget"] = 1
            (Path(campaign_config["artifacts"]["root"]) / "trials" / "existing_trial").mkdir(parents=True)
            loop_config = load_loop_config(loop_path)
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch, self.assertRaisesRegex(
                RuntimeError, "budget exhausted"
            ):
                start_trial(campaign_config, loop_config, dry_run=False)
            launch.assert_not_called()

    def test_apply_campaign_decision_prints_analysis_to_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            loop_config = load_loop_config(loop_path)
            state = {
                "current_trial_id": "trial_result",
                "current_trial_dir": str(root / "trial_result"),
                "stage_runs": {},
                "trial_configs": {},
            }
            stream = io.StringIO()
            result = apply_campaign_decision(
                campaign_config,
                loop_config,
                state,
                CampaignDecision(
                    decision="update_logbook",
                    reason="analysis fallback",
                    logbook_markdown="The result was weak; next try a lower learning rate.",
                ),
                stream=stream,
            )
            self.assertTrue(result["applied"])
            self.assertIn("campaign result analysis: trial_result", stream.getvalue())
            self.assertIn("next try a lower learning rate", stream.getvalue())

    def test_apply_campaign_no_action_still_logs_and_prints_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            loop_config = load_loop_config(loop_path)
            state = {
                "current_trial_id": "trial_no_action",
                "current_trial_dir": str(root / "trial_no_action"),
                "stage_runs": {},
                "trial_configs": {},
            }
            stream = io.StringIO()
            result = apply_campaign_decision(
                campaign_config,
                loop_config,
                state,
                CampaignDecision(decision="no_action", reason="Keep current settings."),
                stream=stream,
            )
            self.assertTrue(result["applied"])
            self.assertIn("Keep current settings.", stream.getvalue())
            self.assertIn("Keep current settings.", (root / "logbook.md").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((Path(campaign_config["artifacts"]["state_path"])).read_text())["status"], "analysis_completed")

    def test_run_campaign_does_not_restart_completed_campaign_without_new_trial(self) -> None:
        class RaisingClient:
            @property
            def responses(self):
                raise AssertionError("OpenAI should not be called for completed campaign status")

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            state_path = Path(campaign_config["artifacts"]["state_path"])
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "test_campaign",
                        "status": "campaign_completed",
                        "current_trial_id": "done",
                        "current_trial_dir": str(root / "done"),
                    }
                ),
                encoding="utf-8",
            )
            stream = io.StringIO()
            code = run_campaign(campaign_config, once=False, client=RaisingClient(), stream=stream)
            self.assertEqual(code, 0)
            self.assertIn("campaign state is campaign_completed", stream.getvalue())

    def test_run_campaign_records_generic_openai_failure_as_retryable_state(self) -> None:
        class FailingResponses:
            def create(self, **kwargs):
                raise RuntimeError("temporary API outage")

        class FailingClient:
            responses = FailingResponses()

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            stream = io.StringIO()
            code = run_campaign(campaign_config, once=True, client=FailingClient(), stream=stream)
            self.assertEqual(code, 0)
            state = json.loads(Path(campaign_config["artifacts"]["state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "agent_decision_failed")
            self.assertIn("temporary API outage", state["agent_decision_error"])

    def test_run_campaign_terminates_explicitly_when_openai_credits_are_exhausted(self) -> None:
        class CreditResponses:
            def create(self, **kwargs):
                raise RuntimeError("insufficient_quota: credits exhausted")

        class CreditClient:
            responses = CreditResponses()

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            stream = io.StringIO()
            code = run_campaign(campaign_config, once=True, client=CreditClient(), stream=stream)
            self.assertEqual(code, 1)
            state = json.loads(Path(campaign_config["artifacts"]["state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "openai_credits_exhausted")
            self.assertIn("credits appear to be exhausted", stream.getvalue())

    def test_campaign_lock_rejects_second_live_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            campaign_path = self._write_campaign_config(root, self._write_loop_config(root))
            campaign_config = load_campaign_config(campaign_path)
            lock = _acquire_campaign_lock(campaign_config)
            try:
                with self.assertRaisesRegex(RuntimeError, "Campaign lock already exists"):
                    _acquire_campaign_lock(campaign_config)
            finally:
                _release_campaign_lock(lock)

    def test_apply_campaign_propose_trial_returns_new_state_for_caller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            loop_config = load_loop_config(loop_path)
            state = {
                "current_trial_id": "completed_trial",
                "current_trial_dir": str(root / "completed_trial"),
                "stage_runs": {},
                "trial_configs": {},
            }
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                launch.return_value = {"active_experiment": "10C", "pid": 123}
                result = apply_campaign_decision(
                    campaign_config,
                    loop_config,
                    state,
                    CampaignDecision(
                        decision="propose_trial",
                        reason="next trial",
                        trial_patch={"10C": {"optimization_config": {"epochs": 4}}},
                    ),
                    stream=io.StringIO(),
                )
            self.assertTrue(result["applied"])
            self.assertIsInstance(result.get("state"), dict)
            self.assertNotEqual(result["state"]["current_trial_id"], "completed_trial")
            self.assertEqual(result["state"]["current_stage"], "10C")

    def test_pdf_preview_png_renderer_writes_png_when_fitz_is_available(self) -> None:
        try:
            import fitz
        except ModuleNotFoundError:
            self.skipTest("fitz is not installed")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pdf_path = root / "plot.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "loss plot")
            document.save(pdf_path)
            document.close()
            preview = _render_pdf_preview_png(pdf_path, root / "previews")
            self.assertIsNotNone(preview)
            self.assertTrue(preview.exists())
            self.assertEqual(preview.suffix, ".png")


if __name__ == "__main__":
    unittest.main()

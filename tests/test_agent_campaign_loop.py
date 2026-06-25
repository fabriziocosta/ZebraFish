from __future__ import annotations

import io
import json
import os
import signal
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.agent_campaign_loop import (
    CampaignDecision,
    _acquire_campaign_lock,
    _campaign_decision_text_format,
    _release_campaign_lock,
    _render_pdf_preview_png,
    _write_trial_outputs,
    apply_campaign_decision,
    campaign_live_status,
    collect_init_snapshot,
    initialize_campaign_trial,
    load_campaign_config,
    parse_campaign_decision,
    run_campaign,
    score_metrics,
    start_trial,
    terminate_campaign,
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


class _SequenceResponses:
    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = list(outputs)
        self.kwargs_history = []

    def create(self, **kwargs):
        self.kwargs_history.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return _FakeResponse(output)


class _SequenceClient:
    def __init__(self, outputs: list[str | Exception]) -> None:
        self.responses = _SequenceResponses(outputs)


class AgentCampaignLoopTests(unittest.TestCase):
    def _write_loop_config(self, root: Path) -> Path:
        (root / "configs").mkdir()
        (root / "params10.yaml").write_text(
            "model_config:\n  normalization: group\noptimization_config:\n  epochs: 10\n",
            encoding="utf-8",
        )
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
  poll_seconds: 3600
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
    allowed_patch_paths: [optimization_config, model_config.normalization]
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
  poll_seconds: 3600
  stages: ["10C", "13C"]
objective:
  target: compound
  primary_metric: macro_f1
  required_primary_metric: true
  fallback_metrics: [accuracy]
  tie_breaker_metrics: [roc_auc_ovr_macro, accuracy]
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
            self.assertEqual(config["campaign"]["poll_seconds"], 3600)
            self.assertEqual(config["campaign"]["max_patch_leaf_count"], 2)
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
                    "fallback_metrics": ["balanced_accuracy"],
                    "tie_breaker_metrics": ["roc_auc_ovr_macro"],
                    "minimums": {"action.accuracy": 0.30},
                },
            )
            self.assertEqual(scored["score"], 0.22)
            self.assertEqual(scored["ranking_score"], 0.22)
            self.assertEqual(scored["ranking_values"], [0.22, 0.7])
            self.assertEqual(scored["ranking_metric_order"], ["compound.macro_f1", "compound.roc_auc_ovr_macro"])
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
            self.assertIn(f"experiment_output_dir: {root / 'campaign_artifacts' / 'trial_a' / 'outputs' / '10C'}", patched_config)
            self.assertIn("Try a short first campaign trial", (root / "logbook.md").read_text(encoding="utf-8"))
            self.assertIn("campaign initialization analysis: trial_a", stream.getvalue())
            self.assertIn("Try a short first campaign trial", stream.getvalue())
            self.assertIn("Initialization snapshot JSON", client.responses.kwargs["input"])
            self.assertEqual(client.responses.kwargs["text"]["format"]["type"], "json_schema")

    def test_initialize_campaign_accepts_four_leaf_schedule_and_loss_weight_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            campaign_config["campaign"]["max_patch_leaf_count"] = 4
            loop_config = load_loop_config(loop_path)
            campaign_config["agent"] = dict(loop_config["agent"])
            output = json.dumps(
                {
                    "decision": "propose_trial",
                    "reason": "review old result and adjust schedule plus loss balance",
                    "logbook_markdown": (
                        "### Previous results reviewed\nOld evidence was weak.\n\n"
                        "### Next experiment to run\nLaunch 10C then 13C with a short schedule and rebalanced losses."
                    ),
                    "trial_patch": {
                        "10C": {"optimization_config": {"epochs": 8, "early_stopping_patience": 3}},
                        "13C": {"loss_weight_config": {"action_weight": 0.6, "compound_weight": 1.4}},
                    },
                }
            )
            client = _FakeClient(output)
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                launch.return_value = {"active_experiment": "10C", "pid": 123}
                stream = io.StringIO()
                state = initialize_campaign_trial(
                    campaign_config,
                    loop_config,
                    start_trial_id="trial_four_leaf",
                    client=client,
                    stream=stream,
                )
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["current_stage"], "10C")
            pretrain_config = Path(state["trial_configs"]["10C"]).read_text(encoding="utf-8")
            finetune_config = Path(state["trial_configs"]["13C"]).read_text(encoding="utf-8")
            self.assertIn("epochs: 8", pretrain_config)
            self.assertIn("early_stopping_patience: 3", pretrain_config)
            self.assertIn("action_weight: 0.6", finetune_config)
            self.assertIn("compound_weight: 1.4", finetune_config)
            self.assertIn("Previous results reviewed", (root / "logbook.md").read_text(encoding="utf-8"))

    def test_initialize_campaign_accepts_10c_normalization_patch(self) -> None:
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
                    "reason": "try batch normalization",
                    "logbook_markdown": "### Previous results reviewed\nPrior runs looked unstable.\n\n### Next experiment to run\nTry batch normalization.",
                    "trial_patch": {"10C": {"model_config": {"normalization": "batch"}}},
                }
            )
            client = _FakeClient(output)
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                launch.return_value = {"active_experiment": "10C", "pid": 123}
                state = initialize_campaign_trial(
                    campaign_config,
                    loop_config,
                    start_trial_id="trial_norm",
                    client=client,
                    stream=io.StringIO(),
                )
            self.assertEqual(state["status"], "running")
            pretrain_config = Path(state["trial_configs"]["10C"]).read_text(encoding="utf-8")
            self.assertIn("normalization: batch", pretrain_config)

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
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                stream = io.StringIO()
                state = initialize_campaign_trial(campaign_config, loop_config, client=client, stream=stream)
            launch.assert_not_called()
            self.assertEqual(state["status"], "agent_decision_failed")
            self.assertEqual(state["phase"], "initializing")
            self.assertIn("non-allowlisted", state["agent_decision_error"])
            self.assertIn("campaign initialization decision rejected", stream.getvalue())
            saved_state = json.loads(Path(campaign_config["artifacts"]["state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(saved_state["status"], "agent_decision_failed")

    def test_initialize_campaign_rejects_too_many_patch_leaves_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            campaign_config["campaign"]["max_patch_leaf_count"] = 1
            loop_config = load_loop_config(loop_path)
            campaign_config["agent"] = dict(loop_config["agent"])
            client = _FakeClient(
                json.dumps(
                    {
                        "decision": "propose_trial",
                        "reason": "too broad",
                        "trial_patch": {"10C": {"optimization_config": {"epochs": 3, "learning_rate": 0.001}}},
                    }
                )
            )
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                state = initialize_campaign_trial(campaign_config, loop_config, client=client, stream=io.StringIO())
            launch.assert_not_called()
            self.assertEqual(state["status"], "agent_decision_failed")
            self.assertIn("max_patch_leaf_count", state["agent_decision_error"])

    def test_parse_campaign_decision_rejects_malformed_output(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_campaign_decision("not json")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_campaign_decision(json.dumps({"decision": "run_shell", "reason": "bad"}))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            parse_campaign_decision(json.dumps({"decision": "no_action", "reason": ""}))
        parsed = parse_campaign_decision(
            json.dumps(
                {
                    "decision": "propose_trial",
                    "reason": "string patch",
                    "trial_patch": "{\"10C\":{\"optimization_config\":{\"epochs\":3}}}",
                }
            )
        )
        self.assertEqual(parsed.trial_patch, {"10C": {"optimization_config": {"epochs": 3}}})

    def test_campaign_decision_schema_is_strict(self) -> None:
        text_format = _campaign_decision_text_format(["10C", "13C"])
        self.assertTrue(text_format["format"]["strict"])
        self.assertFalse(text_format["format"]["schema"]["additionalProperties"])
        self.assertEqual(text_format["format"]["schema"]["properties"]["trial_patch"]["type"], "string")

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
            (Path(campaign_config["artifacts"]["root"]) / "existing_trial" / "configs").mkdir(parents=True)
            loop_config = load_loop_config(loop_path)
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch, self.assertRaisesRegex(
                RuntimeError, "budget exhausted"
            ):
                start_trial(campaign_config, loop_config, dry_run=False)
            launch.assert_not_called()

    def test_start_trial_ignores_prelaunch_orphan_snapshot_for_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            campaign_config["campaign"]["trial_budget"] = 1
            orphan = Path(campaign_config["artifacts"]["root"]) / "orphan_trial"
            orphan.mkdir(parents=True)
            (orphan / "init_snapshot.json").write_text("{}", encoding="utf-8")
            loop_config = load_loop_config(loop_path)
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                launch.return_value = {"active_experiment": "10C", "pid": 123}
                state = start_trial(campaign_config, loop_config, dry_run=False)
            self.assertEqual(state["current_stage"], "10C")

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
            self.assertEqual(state["phase"], "initializing")
            self.assertIn("temporary API outage", state["agent_decision_error"])

    def test_run_campaign_retries_failed_initialization_with_init_prompt(self) -> None:
        output = json.dumps(
            {
                "decision": "propose_trial",
                "reason": "retry init",
                "logbook_markdown": "Retry initialization and launch.",
                "trial_patch": {"10C": {"optimization_config": {"epochs": 3}}},
            }
        )
        client = _SequenceClient([RuntimeError("temporary API outage"), output])
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            stream = io.StringIO()
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch:
                launch.return_value = {"active_experiment": "10C", "pid": 123}
                code = run_campaign(campaign_config, once=True, client=client, stream=stream)
            self.assertEqual(code, 0)
            state = json.loads(Path(campaign_config["artifacts"]["state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "running")
            self.assertEqual(state["current_stage"], "10C")
            self.assertEqual(len(client.responses.kwargs_history), 2)
            self.assertIn("Initialization snapshot JSON", client.responses.kwargs_history[1]["input"])
            launch.assert_called_once()

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

    def test_run_campaign_refuses_new_trial_when_current_stage_is_running(self) -> None:
        class RaisingClient:
            @property
            def responses(self):
                raise AssertionError("OpenAI should not be called when new trial is refused")

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            trial_dir = root / "trial_running"
            stage_state = trial_dir / "stage_state" / "10C.json"
            stage_state.parent.mkdir(parents=True)
            run_status_path = trial_dir / "logs" / "10C" / "run_status.json"
            run_status_path.parent.mkdir(parents=True)
            run_status_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            stage_state.write_text(
                json.dumps(
                    {
                        "active_experiment": "10C",
                        "pid": 999,
                        "status": "running",
                        "run_status_path": str(run_status_path),
                    }
                ),
                encoding="utf-8",
            )
            Path(campaign_config["artifacts"]["state_path"]).parent.mkdir(parents=True)
            Path(campaign_config["artifacts"]["state_path"]).write_text(
                json.dumps(
                    {
                        "campaign_id": "test_campaign",
                        "status": "running",
                        "current_trial_id": "trial_running",
                        "current_trial_dir": str(trial_dir),
                        "current_stage": "10C",
                        "current_stage_index": 0,
                        "stage_state_path": str(stage_state),
                        "trial_configs": {
                            "10C": str(root / "params10.yaml"),
                            "13C": str(root / "params13.yaml"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("src.agent_experiment_loop._is_process_running", return_value=True):
                stream = io.StringIO()
                code = run_campaign(campaign_config, once=True, new_trial=True, client=RaisingClient(), stream=stream)
            self.assertEqual(code, 1)
            self.assertIn("refusing to start a new campaign trial", stream.getvalue())

    def test_running_stale_does_not_call_openai_or_launch_next_trial(self) -> None:
        class RaisingClient:
            @property
            def responses(self):
                raise AssertionError("OpenAI should not be called for live stale process")

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            trial_dir = root / "trial_stale"
            stage_state = trial_dir / "stage_state" / "10C.json"
            stage_state.parent.mkdir(parents=True)
            stage_state.write_text(
                json.dumps(
                    {
                        "active_experiment": "10C",
                        "pid": 999,
                        "status": "running",
                        "stale_polls": 99,
                    }
                ),
                encoding="utf-8",
            )
            state_path = Path(campaign_config["artifacts"]["state_path"])
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "test_campaign",
                        "status": "running",
                        "current_trial_id": "trial_stale",
                        "current_trial_dir": str(trial_dir),
                        "current_stage": "10C",
                        "current_stage_index": 0,
                        "stage_state_path": str(stage_state),
                        "trial_configs": {
                            "10C": str(root / "params10.yaml"),
                            "13C": str(root / "params13.yaml"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("src.agent_experiment_loop._is_process_running", return_value=True), mock.patch(
                "src.agent_experiment_loop.launch_experiment"
            ) as launch:
                stream = io.StringIO()
                code = run_campaign(campaign_config, once=True, client=RaisingClient(), stream=stream)
            self.assertEqual(code, 0)
            launch.assert_not_called()
            self.assertIn("action=wait_stale", stream.getvalue())

    def test_terminate_campaign_marks_campaign_and_stage_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            trial_dir = root / "trial_running"
            stage_state = trial_dir / "stage_state" / "10C.json"
            stage_state.parent.mkdir(parents=True)
            run_status_path = trial_dir / "logs" / "10C" / "run_status.json"
            run_status_path.parent.mkdir(parents=True)
            run_status_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
            stage_state.write_text(
                json.dumps(
                    {
                        "active_experiment": "10C",
                        "pid": 999,
                        "status": "running",
                        "run_status_path": str(run_status_path),
                    }
                ),
                encoding="utf-8",
            )
            state_path = Path(campaign_config["artifacts"]["state_path"])
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "test_campaign",
                        "status": "running",
                        "current_trial_id": "trial_running",
                        "current_trial_dir": str(trial_dir),
                        "current_stage": "10C",
                        "current_stage_index": 0,
                        "stage_state_path": str(stage_state),
                        "trial_configs": {
                            "10C": str(root / "params10.yaml"),
                            "13C": str(root / "params13.yaml"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("src.agent_experiment_loop._is_process_running", return_value=True), mock.patch(
                "os.killpg"
            ) as killpg:
                stream = io.StringIO()
                code = terminate_campaign(campaign_config, reason="stop test", stream=stream)
            self.assertEqual(code, 0)
            killpg.assert_called_once_with(999, signal.SIGTERM)
            campaign_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(campaign_state["status"], "terminated")
            self.assertEqual(campaign_state["termination_reason"], "stop test")
            updated_stage_state = json.loads(stage_state.read_text(encoding="utf-8"))
            self.assertEqual(updated_stage_state["status"], "terminated")
            self.assertEqual(updated_stage_state["terminated_pid"], 999)
            updated_run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_run_status["status"], "terminated")
            self.assertEqual(updated_run_status["terminated_pid"], 999)
            self.assertIn("campaign terminated", stream.getvalue())

    def test_terminate_campaign_escalates_when_force_after_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            trial_dir = root / "trial_running"
            stage_state = trial_dir / "stage_state" / "10C.json"
            stage_state.parent.mkdir(parents=True)
            stage_state.write_text(
                json.dumps({"active_experiment": "10C", "pid": 999, "status": "running"}),
                encoding="utf-8",
            )
            state_path = Path(campaign_config["artifacts"]["state_path"])
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "test_campaign",
                        "status": "running",
                        "current_trial_id": "trial_running",
                        "current_trial_dir": str(trial_dir),
                        "current_stage": "10C",
                        "current_stage_index": 0,
                        "stage_state_path": str(stage_state),
                        "trial_configs": {
                            "10C": str(root / "params10.yaml"),
                            "13C": str(root / "params13.yaml"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("src.agent_experiment_loop._is_process_running", return_value=True), mock.patch(
                "os.killpg"
            ) as killpg:
                code = terminate_campaign(campaign_config, force_after=0, stream=io.StringIO(), sleep_fn=lambda _seconds: None)
            self.assertEqual(code, 0)
            self.assertEqual(killpg.mock_calls, [mock.call(999, signal.SIGTERM), mock.call(999, signal.SIGKILL)])
            campaign_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(campaign_state["termination_signal"], "SIGKILL")

    def test_terminate_campaign_no_running_process_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            Path(campaign_config["artifacts"]["state_path"]).parent.mkdir(parents=True)
            Path(campaign_config["artifacts"]["state_path"]).write_text(
                json.dumps({"campaign_id": "test_campaign", "status": "analysis_completed"}),
                encoding="utf-8",
            )
            with mock.patch("os.killpg") as killpg:
                stream = io.StringIO()
                code = terminate_campaign(campaign_config, stream=stream)
            self.assertEqual(code, 0)
            killpg.assert_not_called()
            self.assertIn("no running campaign found", stream.getvalue())

    def test_terminate_campaign_require_running_noop_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            Path(campaign_config["artifacts"]["state_path"]).parent.mkdir(parents=True)
            Path(campaign_config["artifacts"]["state_path"]).write_text(
                json.dumps({"campaign_id": "test_campaign", "status": "analysis_completed"}),
                encoding="utf-8",
            )
            code = terminate_campaign(campaign_config, stream=io.StringIO(), require_running=True)
            self.assertEqual(code, 1)

    def test_terminate_campaign_marks_race_when_process_stops_before_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            trial_dir = root / "trial_running"
            stage_state = trial_dir / "stage_state" / "10C.json"
            stage_state.parent.mkdir(parents=True)
            stage_state.write_text(
                json.dumps({"active_experiment": "10C", "pid": 999, "status": "running"}),
                encoding="utf-8",
            )
            state_path = Path(campaign_config["artifacts"]["state_path"])
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "test_campaign",
                        "status": "running",
                        "current_trial_id": "trial_running",
                        "current_trial_dir": str(trial_dir),
                        "current_stage": "10C",
                        "current_stage_index": 0,
                        "stage_state_path": str(stage_state),
                        "trial_configs": {
                            "10C": str(root / "params10.yaml"),
                            "13C": str(root / "params13.yaml"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("src.agent_experiment_loop._is_process_running", return_value=True), mock.patch(
                "os.killpg", side_effect=ProcessLookupError
            ):
                stream = io.StringIO()
                code = terminate_campaign(campaign_config, stream=stream)
            self.assertEqual(code, 0)
            campaign_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(campaign_state["status"], "termination_race_stopped")
            updated_stage_state = json.loads(stage_state.read_text(encoding="utf-8"))
            self.assertEqual(updated_stage_state["status"], "termination_race_stopped")
            self.assertIn("termination_race_stopped", stream.getvalue())

    def test_campaign_live_status_reports_running_pid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            trial_dir = root / "trial_running"
            stage_state = trial_dir / "stage_state" / "10C.json"
            stage_state.parent.mkdir(parents=True)
            stage_state.write_text(
                json.dumps({"active_experiment": "10C", "pid": 999, "status": "running"}),
                encoding="utf-8",
            )
            state_path = Path(campaign_config["artifacts"]["state_path"])
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "campaign_id": "test_campaign",
                        "status": "running",
                        "current_trial_id": "trial_running",
                        "current_trial_dir": str(trial_dir),
                        "current_stage": "10C",
                        "current_stage_index": 0,
                        "stage_state_path": str(stage_state),
                        "trial_configs": {
                            "10C": str(root / "params10.yaml"),
                            "13C": str(root / "params13.yaml"),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("src.agent_experiment_loop._is_process_running", return_value=True):
                status = campaign_live_status(campaign_config)
            self.assertTrue(status["running"])
            self.assertEqual(status["pid"], 999)

    def test_apply_campaign_propose_trial_validates_patch_size_before_logbook_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            campaign_config["campaign"]["max_patch_leaf_count"] = 1
            loop_config = load_loop_config(loop_path)
            state = {
                "current_trial_id": "completed_trial",
                "current_trial_dir": str(root / "completed_trial"),
                "stage_runs": {},
                "trial_configs": {},
            }
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch, self.assertRaisesRegex(
                ValueError, "max_patch_leaf_count"
            ):
                apply_campaign_decision(
                    campaign_config,
                    loop_config,
                    state,
                    CampaignDecision(
                        decision="propose_trial",
                        reason="too many changes",
                        logbook_markdown="This should not be written.",
                        trial_patch={"10C": {"optimization_config": {"epochs": 4, "learning_rate": 0.001}}},
                    ),
                    stream=io.StringIO(),
                )
            launch.assert_not_called()
            self.assertFalse((root / "logbook.md").exists())

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

    def test_start_trial_rejects_existing_explicit_trial_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            loop_config = load_loop_config(loop_path)
            (Path(campaign_config["artifacts"]["root"]) / "same_trial").mkdir(parents=True)
            with mock.patch("src.agent_experiment_loop.launch_experiment") as launch, self.assertRaisesRegex(
                FileExistsError, "already exists"
            ):
                start_trial(campaign_config, loop_config, start_trial="same_trial", dry_run=False)
            launch.assert_not_called()

    def test_write_trial_outputs_uses_auc_tie_breaker_in_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            loop_path = self._write_loop_config(root)
            campaign_path = self._write_campaign_config(root, loop_path)
            campaign_config = load_campaign_config(campaign_path)
            for trial_id, auc in [("trial_low_auc", 0.60), ("trial_high_auc", 0.80)]:
                state = {
                    "current_trial_id": trial_id,
                    "current_trial_dir": str(root / trial_id),
                    "status": "trial_completed",
                }
                summary = {
                    "trial_id": trial_id,
                    "trial_dir": str(root / trial_id),
                    "status": "trial_completed",
                    "score": 0.50,
                    "ranking_score": 0.50,
                    "ranking_values": [0.50, auc],
                    "ranking_metric_order": ["compound.macro_f1", "compound.roc_auc_ovr_macro"],
                    "objective_eligible": True,
                    "selected_metric": "compound.macro_f1",
                    "guardrail_passed": True,
                    "guardrail_failures": {},
                    "metrics_path": "",
                    "stage_runs": {},
                    "trial_configs": {},
                    "metrics": {},
                }
                _write_trial_outputs(campaign_config, state, summary)
            leaderboard = (Path(campaign_config["artifacts"]["leaderboard_csv"])).read_text(encoding="utf-8")
            self.assertLess(leaderboard.index("trial_high_auc"), leaderboard.index("trial_low_auc"))

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

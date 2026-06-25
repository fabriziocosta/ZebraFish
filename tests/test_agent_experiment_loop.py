from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.agent_experiment_loop import (
    AgentDecision,
    apply_agent_decision,
    collect_status,
    load_loop_config,
    main,
    parse_agent_decision,
    request_agent_decision,
    run_loop,
)


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse(self.output_text)


class _FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


class AgentExperimentLoopTests(unittest.TestCase):
    def _write_config(self, tmpdir: str) -> Path:
        root = Path(tmpdir)
        config_path = root / "agent.yaml"
        config_path.write_text(
            """
agent:
  model: gpt-5.3-codex
  reasoning_effort: medium
  poll_seconds: 3600
  api_key_env: OPENAI_API_KEY
state:
  path: state.json
  log_dir: logs
logbook:
  path: logbook.md
experiments:
  "10C":
    runner: 10C_pretrain_commutative_cnn_encoder.py
    params_yaml: params.yaml
    artifact_root: artifacts/pretrained_commutative_cnn
    next: "13C"
  "13C":
    runner: 13C_finetune_pretrained_commutative_cnn_classifier.py
    params_yaml: params13.yaml
    artifact_root: artifacts/nb13C_commutative_cnn_full_finetune
    next: null
  "12T":
    runner: 12T_pretrain_commutative_transformer_encoder.py
    params_yaml: params12.yaml
    artifact_root: artifacts/pretrained_commutative_transformer
    next: null
prompts:
  analysis_policy: analyze only completed/stale/failed runs
  status_decision: return JSON
""",
            encoding="utf-8",
        )
        (root / "params.yaml").write_text("optimization_config:\n  epochs: 140\n", encoding="utf-8")
        (root / "params13.yaml").write_text("optimization_config:\n  epochs: 120\n", encoding="utf-8")
        (root / "params12.yaml").write_text("optimization_config:\n  epochs: 90\n", encoding="utf-8")
        return config_path

    def test_load_loop_config_defaults_poll_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(tmpdir)
            config = load_loop_config(config_path)
            self.assertEqual(config["agent"]["poll_seconds"], 3600)
            self.assertEqual(config["agent"]["reasoning_effort"], "medium")

    def test_missing_api_key_errors_before_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {}, clear=True):
            config_path = self._write_config(tmpdir)
            config = load_loop_config(config_path)
            config["state"]["path"] = str(Path(tmpdir) / "state.json")
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                run_loop(config, start_at="10C", once=True, dry_run=False, stream=io.StringIO())
            self.assertFalse(Path(config["state"]["path"]).exists())

    def test_parse_all_allowed_decisions(self) -> None:
        for decision in [
            "patch_next_params",
            "launch_next",
            "no_action",
            "update_logbook",
        ]:
            parsed = parse_agent_decision(
                json.dumps({"decision": decision, "experiment": "10C", "reason": f"{decision} reason"})
            )
            self.assertEqual(parsed.decision, decision)

    def test_malformed_agent_output_raises(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_agent_decision("not json")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_agent_decision(json.dumps({"decision": "run_shell", "experiment": "10C", "reason": "bad"}))

    def test_request_agent_decision_uses_responses_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_loop_config(self._write_config(tmpdir))
            client = _FakeClient(json.dumps({"decision": "no_action", "experiment": "10C", "reason": "no change"}))
            decision = request_agent_decision(config, {"active_experiment": "10C"}, client=client)
            self.assertEqual(decision.decision, "no_action")
            self.assertEqual(client.responses.kwargs["model"], "gpt-5.3-codex")
            self.assertEqual(client.responses.kwargs["reasoning"], {"effort": "medium"})

    def test_dry_run_once_prints_single_next_poll(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_loop_config(self._write_config(tmpdir))
            stream = io.StringIO()
            code = run_loop(config, start_at="10C", once=True, dry_run=True, stream=stream)
            self.assertEqual(code, 0)
            output = stream.getvalue()
            self.assertEqual(output.count("next poll:"), 1)
            self.assertIn("decision=no_action", output)

    def test_running_process_waits_without_openai_call(self) -> None:
        class RaisingClient:
            @property
            def responses(self):
                raise AssertionError("OpenAI client should not be used while deterministic wait applies")

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            root = Path(tmpdir)
            config = load_loop_config(self._write_config(tmpdir))
            state_path = root / "state.json"
            config["state"]["path"] = str(state_path)
            config["experiments"]["10C"]["artifact_root"] = str(root / "missing_artifacts")
            state_path.write_text(
                json.dumps(
                    {
                        "active_experiment": "10C",
                        "pid": os.getpid(),
                        "status": "running",
                        "log_path": str(root / "run.log"),
                    }
                ),
                encoding="utf-8",
            )
            stream = io.StringIO()
            code = run_loop(config, once=True, dry_run=False, client=RaisingClient(), stream=stream)
            self.assertEqual(code, 0)
            self.assertIn("decision=no_action", stream.getvalue())
            self.assertIn("running_wait", state_path.read_text(encoding="utf-8"))

    def test_collect_status_uses_runner_status_run_dir_before_latest_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_loop_config(self._write_config(tmpdir))
            run_dir = root / "exact_run"
            run_dir.mkdir()
            (run_dir / "latest.history.csv").write_text("epoch,loss\n1,1.0\n", encoding="utf-8")
            run_status_path = root / "run_status.json"
            run_status_path.write_text(
                json.dumps({"status": "running", "run_dir": str(run_dir), "experiment_id": "exact"}),
                encoding="utf-8",
            )
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps({"active_experiment": "10C", "run_status_path": str(run_status_path)}),
                encoding="utf-8",
            )
            status = collect_status(config, state_path=state_path)
            self.assertEqual(status["artifacts"]["run_dir"], str(run_dir))
            self.assertEqual(status["artifacts"]["run_dir_source"], "runner_status")
            self.assertEqual(status["run_status"]["experiment_id"], "exact")

    def test_keyboard_interrupt_preserves_child_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_loop_config(self._write_config(tmpdir))

            def raise_interrupt(seconds: float) -> None:
                raise KeyboardInterrupt

            stream = io.StringIO()
            code = run_loop(
                config,
                start_at="10C",
                once=False,
                dry_run=True,
                sleep_fn=raise_interrupt,
                stream=stream,
            )
            self.assertEqual(code, 130)
            self.assertIn("agent loop stopped", stream.getvalue())

    def test_patch_next_params_decision_merges_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_loop_config(self._write_config(tmpdir))
            config["experiments"]["10C"]["params_yaml"] = str(root / "params.yaml")
            decision = AgentDecision(
                decision="patch_next_params",
                experiment="10C",
                reason="lower lr",
                parameters_patch={"optimization_config": {"learning_rate": 1e-5}},
            )
            result = apply_agent_decision(config, decision, {"artifacts": {}}, dry_run=False)
            self.assertTrue(result["applied"])
            self.assertIn("learning_rate", (root / "params.yaml").read_text(encoding="utf-8"))
            self.assertIn("epochs", (root / "params.yaml").read_text(encoding="utf-8"))

    def test_patch_next_params_rejects_non_allowlisted_yaml_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_loop_config(self._write_config(tmpdir))
            config["experiments"]["10C"]["params_yaml"] = str(root / "params.yaml")
            decision = AgentDecision(
                decision="patch_next_params",
                experiment="10C",
                reason="unsafe architecture change",
                parameters_patch={"model_config": {"embed_dim": 128}},
            )
            with self.assertRaisesRegex(ValueError, "non-allowlisted"):
                apply_agent_decision(config, decision, {"artifacts": {}}, dry_run=False)

    def test_launch_next_is_gated_by_controller_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_loop_config(self._write_config(tmpdir))
            decision = AgentDecision(decision="launch_next", experiment="10C", reason="continue")
            result = apply_agent_decision(
                config,
                decision,
                {"active_experiment": "10C", "controller_status": "running_wait"},
                dry_run=False,
            )
            self.assertFalse(result["applied"])
            self.assertIn("completion", result["reason"])

    def test_run_loop_accepts_configured_transformer_experiment_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_loop_config(self._write_config(tmpdir))
            stream = io.StringIO()
            code = run_loop(config, start_at="12T", once=True, dry_run=True, stream=stream)
            self.assertEqual(code, 0)
            self.assertIn("poll experiment=12T", stream.getvalue())

    def test_completed_state_is_replaced_when_start_at_is_given_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = load_loop_config(self._write_config(tmpdir))
            state_path = root / "state.json"
            config["state"]["path"] = str(state_path)
            state_path.write_text(
                json.dumps({"active_experiment": "10C", "status": "completed"}),
                encoding="utf-8",
            )
            stream = io.StringIO()
            code = run_loop(config, start_at="12T", once=True, dry_run=True, stream=stream)
            self.assertEqual(code, 0)
            self.assertIn("poll experiment=12T", stream.getvalue())

    def test_cli_rejects_unknown_start_at_after_loading_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(tmpdir)
            stream = io.StringIO()
            with contextlib.redirect_stderr(stream), self.assertRaises(SystemExit):
                main(["run", "--config", str(config_path), "--dry-run", "--once", "--start-at", "99X"])
            self.assertIn("Unknown experiment", stream.getvalue())


if __name__ == "__main__":
    unittest.main()

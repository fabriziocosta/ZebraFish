# Agent Experiment Loop Infrastructure

## Preface

The experiment loop is a small local orchestration system for long-running scientific model experiments. Its purpose is to separate the mechanical parts of running experiments from the interpretive parts of deciding what to try next.

The mechanical part is deterministic. The controller launches a configured experiment, records the process id, follows a timestamped run folder, checks whether required artifacts exist, and decides whether the run is still progressing, stale, completed, or failed. Waiting, completion, and failure are not delegated to a language model because those decisions should be based on filesystem state, process state, and explicit artifact requirements.

The interpretive part is model-assisted. Once a run has completed, failed, or become stale, the controller can send a compact status snapshot to the OpenAI API. The model can then help write a logbook note, explain likely failure modes, or propose small YAML parameter patches for the next run. Those actions are constrained by the local controller: parameter patches are allowlisted, next-run launches are gated by deterministic completion, and arbitrary shell commands are never accepted from the model.

Each experiment is treated as an immutable attempt. It gets a timestamped folder containing the resolved configuration, live loss histories, live PDFs, final metrics, plots, checkpoints, and status records where available. The logbook is the human-readable index over those attempts. It should explain what happened, point to the actual artifacts, and state the next hypothesis to test.

This architecture is meant to support slow iteration without noisy supervision. A pretraining run may take many hours, so the harness sleeps for the configured poll interval and prints only a bounded status update. If interrupted with `Ctrl-C`, the harness stops; the training process is left running by default unless explicit termination is requested.

This repository can run the 10C pretraining and 13C fine-tuning experiments through a local polling harness:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --config configs/agent_experiment_loop.yaml --start-at 10C
```

The harness is intentionally quiet. It prints startup status, one status line per poll, the next poll time, and shutdown status. It does not print countdowns.

## Components

- `scripts/agent_experiment_loop.py` is the CLI entrypoint.
- `src/agent_experiment_loop.py` implements the controller, polling, OpenAI API call, state tracking, and whitelisted actions.
- `configs/agent_experiment_loop.yaml` configures the model, poll interval, state paths, experiment runners, and analysis prompts.
- `configs/experiments/10C_pretrain_next.yaml` and `configs/experiments/13C_finetune_next.yaml` are the next-run parameter files that the agent may patch.
- `configs/experiments/12T_pretrain_next.yaml` and `configs/experiments/15T_finetune_next.yaml` are the transformer equivalents.
- `10C_pretrain_commutative_cnn_encoder.py` and `13C_finetune_pretrained_commutative_cnn_classifier.py` are code-friendly runner entrypoints equivalent to the notebooks.
- `12T_pretrain_commutative_transformer_encoder.py` and `15T_finetune_pretrained_commutative_transformer_classifier.py` are code-friendly transformer runner entrypoints.
- `src/experiment_runner_cnn.py`, `src/experiment_runner_transformer.py`, and `src/experiment_runner_shared.py` hold architecture-specific runner logic and shared status/YAML helpers. `src/experiment_runners.py` is a compatibility facade.
- `EXPERIMENTS_LOGBOOK.md` remains the human-facing experiment record.

## Control Model

The controller, not the model, decides whether to wait, mark completion, or mark failure.

The model is only called when the controller sees a completed, failed, or stale run. While the process is running normally, the harness waits deterministically until the next configured poll interval.

The model may request only these actions:

- `no_action`
- `update_logbook`
- `patch_next_params`
- `launch_next`

`launch_next` is only honored after deterministic completion of the active run and only for the configured `next` experiment.

Parameter patching is allowlisted per experiment in `configs/agent_experiment_loop.yaml`. This prevents model-proposed patches from changing incompatible architecture keys unless those paths are explicitly listed.

## Run Tracking

When the harness launches a runner, it writes:

- `artifacts/agent_experiment_loop/state.json`
- a per-run log file under `artifacts/agent_experiment_loop/logs/`
- a per-run status JSON under `artifacts/agent_experiment_loop/logs/`

The runner receives the status JSON path through `ZF_AGENT_RUN_STATUS_PATH` and updates it with:

- `status`
- `experiment`
- `experiment_id`
- `run_dir`
- `checkpoint_path` when available
- error text on failure

This avoids guessing the current experiment from the newest artifact folder.

If the controller falls back to inspecting the newest run folder, status includes `run_dir_source: latest_fallback`. Treat this as less reliable than `state` or `runner_status`.

## Artifacts

Each experiment run writes into its timestamped run folder. The training loops write live CSV history files next to the live loss PDFs:

- `latest.history.csv`
- `epoch_XXX.history.csv`
- `latest.loss-curves.pdf`
- `epoch_XXX.loss-curves.pdf`

Final persistence writes resolved config snapshots, history CSVs, summary metrics, checkpoints, confusion matrices, prediction tables, loss PDFs, UMAP CSVs, and UMAP PDFs where applicable.

Completion requirements are declared per experiment in `configs/agent_experiment_loop.yaml`, for example:

```yaml
required_completion_artifacts: [history, summary_metrics, checkpoint, confusion_matrices, umap_pdf]
```

Pretraining runs usually require history, summary metrics, and checkpoint. Fine-tuning runs also require confusion matrices and a UMAP PDF.

## Polling

The poll interval is configured in `configs/agent_experiment_loop.yaml`:

```yaml
agent:
  poll_seconds: 18000
```

This example polls every 5 hours. You can override it at runtime:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --poll-seconds 3600
```

Use `Ctrl-C` to stop the harness. By default this does not terminate the training process. To also terminate a child process launched by the harness:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --terminate-child-on-exit
```

If the state file points to a completed or failed run, `--start-at` launches a fresh run by default. Use `--resume` to inspect the existing state instead. Use `--new-run` to force a fresh run, or `--reset-state` to delete the state file before starting.

## Useful Commands

Run one dry status pass without launching jobs, writing files, or calling OpenAI:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --config configs/agent_experiment_loop.yaml --dry-run --once --start-at 10C
```

Check current harness status without calling OpenAI:

```bash
.venv/bin/python scripts/agent_experiment_loop.py status --config configs/agent_experiment_loop.yaml
```

Run 10C directly:

```bash
.venv/bin/python 10C_pretrain_commutative_cnn_encoder.py --config configs/experiments/10C_pretrain_next.yaml
```

Run 13C directly:

```bash
.venv/bin/python 13C_finetune_pretrained_commutative_cnn_classifier.py --config configs/experiments/13C_finetune_next.yaml
```

Run 12T directly:

```bash
.venv/bin/python 12T_pretrain_commutative_transformer_encoder.py --config configs/experiments/12T_pretrain_next.yaml
```

Run 15T directly:

```bash
.venv/bin/python 15T_finetune_pretrained_commutative_transformer_classifier.py --config configs/experiments/15T_finetune_next.yaml
```

## API Key

The OpenAI-backed loop requires `OPENAI_API_KEY` unless `--dry-run` is used. The status command does not call OpenAI.

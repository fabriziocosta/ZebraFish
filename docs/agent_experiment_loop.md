# Agent Experiment Loop Infrastructure

## Preface

The experiment loop is a small local orchestration system for long-running scientific model experiments. Its purpose is to separate the mechanical parts of running experiments from the interpretive parts of deciding what to try next.

The mechanical part is deterministic. The controller launches a configured experiment, records the process id, follows a timestamped run folder, checks whether required artifacts exist, and decides whether the run is still progressing, stale, completed, or failed. Waiting, completion, and failure are not delegated to a language model because those decisions should be based on filesystem state, process state, and explicit artifact requirements.

The interpretive part is model-assisted. Once a run has completed, failed, or become stale, the controller can send a compact status snapshot to the OpenAI API. The model can then help write a logbook note, explain likely failure modes, or propose small YAML parameter patches for the next run. Those actions are constrained by the local controller: parameter patches are allowlisted, next-run launches are gated by deterministic completion, and arbitrary shell commands are never accepted from the model.

Each experiment is treated as an immutable attempt. It gets a timestamped folder containing the resolved configuration, live loss histories, live PDFs, final metrics, plots, checkpoints, and status records where available. The logbook is the human-readable index over those attempts. It should explain what happened, point to the actual artifacts, and state the next hypothesis to test.

This architecture is meant to support slow iteration without noisy supervision. A pretraining run may take many hours, so the harness sleeps for the configured poll interval and prints only a bounded status update. If interrupted with `Ctrl-C`, the harness stops; the training process is left running by default unless explicit termination is requested through the CLI.

This repository can run the 10C pretraining and 13C fine-tuning experiments through a local polling harness:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --config configs/agent_experiment_loop.yaml --start-at 10C
```

For the normal pretrain-to-finetune optimization loop, start a campaign from the repository root:

```bash
./run_campaign cnn
```

or for the transformer sequence:

```bash
./run_campaign transformer
```

The campaign wrapper performs an initialization phase when there is no active campaign state. It inspects the current experiment statuses, recent logs, existing campaign ledgers, and the tail of `EXPERIMENTS_LOGBOOK.md`; asks the OpenAI model for the next things to try; writes that plan into the logbook; validates any proposed YAML patch against the experiment allowlists and campaign patch-size limit; then launches the first stage.

The harness is intentionally quiet. It prints startup status, one status line per poll, the next poll time, and shutdown status. When the model makes an initialization or result decision, it also prints the analysis/proposal block that is written into the logbook. It does not print countdowns.

## Available Campaigns

| Command name | Campaign id | Sequence | Description | Start command |
| --- | --- | --- | --- | --- |
| `cnn` | `cnn_pretrain_finetune` | `10C -> 13C` | Runs the commutative CNN pretraining stage, then fine-tunes the CNN classifier. The campaign scores trials on downstream 13C compound performance with action accuracy as a guardrail. | `./run_campaign cnn` |
| `transformer` | `transformer_pretrain_finetune` | `12T -> 15T` | Runs the commutative transformer pretraining stage, then fine-tunes the transformer classifier. The campaign uses the same downstream compound objective and action guardrail as the CNN campaign. | `./run_campaign transformer` |

Each command name maps to a YAML file through `run_campaign.py`. You can also pass a campaign YAML path directly:

```bash
./run_campaign configs/experiment_campaigns/cnn_campaign.yaml
```

List campaigns from the root wrapper:

```bash
./run_campaign list
```

## Components

- `scripts/agent_experiment_loop.py` is the CLI entrypoint.
- `src/agent_experiment_loop.py` implements the controller, polling, OpenAI API call, state tracking, and whitelisted actions.
- `run_campaign.py` is the simplified root-level campaign entrypoint.
- `run_campaign` is a root-level shell wrapper that uses `.venv/bin/python` when available.
- `scripts/agent_campaign_loop.py` is the detailed campaign CLI.
- `src/agent_campaign_loop.py` implements campaign initialization, pretrain-to-finetune sequencing, trial ledgers, scoring, and leaderboard updates.
- `configs/experiment_campaigns/cnn_campaign.yaml` defines the `10C -> 13C` optimization campaign.
- `configs/experiment_campaigns/transformer_campaign.yaml` defines the `12T -> 15T` optimization campaign.
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

Parameter patching is allowlisted per experiment in `configs/agent_experiment_loop.yaml`. This prevents model-proposed patches from changing incompatible architecture keys unless those paths are explicitly listed. Campaign configs also set `max_patch_leaf_count`, so the model cannot bundle a broad multi-change sweep into one trial. The current campaign configs allow up to four YAML leaf edits, enough for two schedule controls plus a paired loss-weight change. Proposed patches are validated before the logbook records them as the next trial.

## Campaign Control

The campaign layer treats a `next` chain as one optimization trial. For example, the CNN campaign is a trial sequence of `10C` pretraining followed by `13C` fine-tuning. The transformer campaign is `12T` followed by `15T`.

The `next` fields in `configs/agent_experiment_loop.yaml` remain the local handoff mechanism. The campaign config adds the higher-level meaning: objective metric, trial budget, campaign artifact folder, and the prompt used to propose the next trial.

A campaign has an initialization phase before the first run starts. During init, the campaign controller gathers:

- latest artifact/status snapshots for each stage;
- recent harness and notebook log tails;
- the current logbook tail;
- existing trial and leaderboard CSV tails.

The model may propose only a structured campaign decision:

- `no_action`
- `propose_trial`
- `update_logbook`
- `stop_campaign`

The OpenAI request uses strict structured output. To keep that strict schema compatible with arbitrary YAML patch keys, `trial_patch` is returned as a JSON-encoded string and decoded locally. For initialization, `propose_trial` may include a decoded `trial_patch` keyed by experiment id, such as `10C` or `13C`. The controller writes the plan into the logbook before launching. If the model returns no patch, the campaign launches the copied baseline configs unchanged. If the init model call fails for a transient reason, the campaign persists `phase: initializing` and retries the initialization decision on the next poll instead of pretending an empty trial has results to analyze.

After pretraining completes, the campaign controller wires the generated per-run pretraining config into the fine-tune config. It prefers the resolved config artifact reported by the completed stage status and only falls back to scanning the run folder. Fine-tuning therefore uses the exact checkpoint from the current trial, not whichever checkpoint happens to be latest globally.

The campaign objective is evaluated only after the final fine-tune stage completes. Pretraining losses are diagnostic evidence; they are not the primary score.

The objective separates substitute metrics from tie-breakers. `primary_metric` is the main optimization target. If `required_primary_metric` is true, the trial is not eligible unless that metric is present. `fallback_metrics` are used only when substitution is allowed. `tie_breaker_metrics` rank otherwise comparable eligible trials without changing the primary score.

At campaign decision points, stdout includes a bounded markdown block:

```text
campaign result analysis: <trial_id>
------------------------------------
<analysis and next-step proposal>

proposed trial patch:
{
  ...
}
```

The same analysis is upserted into `EXPERIMENTS_LOGBOOK.md` with artifact links. When PDF plots are linked from a completed campaign trial, the controller also tries to render first-page PNG previews into the trial folder and embeds those PNGs inline in the logbook. If local PDF rendering is unavailable, the PDF links are still written with a preview-unavailable note.

Non-dry campaign runs acquire `artifacts/campaigns/<campaign_id>/campaign.lock`. A second live campaign process for the same campaign is rejected; stale locks are cleared only when the recorded PID no longer exists. `--new-trial` also refuses to start over a campaign state whose current stage still has a running process.

Trial launch is staged through `status: launching` before the child process starts. This means the campaign state and manifest exist before training is handed to the runner. If launch fails, the campaign state is marked failed with the launch error.

Leaderboards use guardrail-aware ranking. The raw objective score is recorded for every completed trial, but `leaderboard.csv` includes only trials that pass configured guardrails such as `action.accuracy`. Eligible trials are sorted by the selected objective metric plus configured tie-breaker metrics, for example macro-F1 first and ROC-AUC as a secondary signal when available.

Stale-but-live processes are not launch gates. If a stage is still running but has not produced new artifacts for the configured stale threshold, the campaign records `running_stale` and keeps monitoring. It does not call the model path that can launch another trial until the process exits, completes, or is explicitly terminated outside the default loop.

Campaign child processes can be terminated explicitly without owning the original terminal. `./run_campaign terminate` terminates the only live known campaign; if multiple campaigns are live, it refuses and prints the candidates so you must choose one explicitly. You can specify a campaign command name, campaign id, or YAML path. Termination first marks campaign and stage state as `terminating`, then sends `SIGTERM` to the active training process group, then marks campaign, stage, and runner status state as `terminated`. Artifacts are preserved. `--force-after <seconds>` escalates to `SIGKILL` if the process is still alive after the positive grace interval. `--require-running` makes the command return nonzero if nothing was actually running.

Terminated or termination-race-stopped campaign states are treated as interrupted runs. A normal `./run_campaign <campaign>` starts a replacement trial from those states after first checking that no child process is still alive. Completed campaigns and completed analyses do not automatically restart; use `--new-trial` when you explicitly want to begin another trial from those states.

OpenAI decision failures are resumable by default: the campaign writes `status: agent_decision_failed` and retries the appropriate decision phase on the next poll. Campaign decisions are requested with a JSON schema and then validated locally before any filesystem mutation or launch. If the API error indicates exhausted credits, quota, or billing exhaustion, the campaign writes `status: openai_credits_exhausted`, prints an explicit message, and terminates instead of retrying.

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

Campaign runs write:

- `artifacts/campaigns/<campaign_id>/campaign.lock` while a non-dry campaign process is active
- `artifacts/campaigns/<campaign_id>/campaign_state.json`
- `artifacts/campaigns/<campaign_id>/trials.csv`
- `artifacts/campaigns/<campaign_id>/trials.jsonl`
- `artifacts/campaigns/<campaign_id>/leaderboard.csv`
- `artifacts/campaigns/<campaign_id>/<trial_id>/trial_manifest.json`
- `artifacts/campaigns/<campaign_id>/<trial_id>/init_snapshot.json` during initialization
- `artifacts/campaigns/<campaign_id>/<trial_id>/trial_summary.json`
- per-stage state and logs under the trial folder.
- per-stage run outputs under `artifacts/campaigns/<campaign_id>/<trial_id>/outputs/<stage>/`.

Campaign trial folders deliberately do not use an extra `trials/` nesting level. A trial id is already globally mnemonic and timestamped, so the direct layout keeps links shorter and makes it clear that the folder is the complete unit of evidence for that trial.

Campaign state and machine-readable ledgers are written through temporary files and atomic replacement to reduce the chance of partial JSON, CSV, or JSONL files after interruption.

## Artifacts

Each campaign-launched experiment run writes into a timestamped run folder under the campaign trial, for example `artifacts/campaigns/cnn_pretrain_finetune/<trial_id>/outputs/10C/runs/<experiment_id>/`. The training loops write live CSV history files next to the live loss PDFs:

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
  poll_seconds: 3600
```

This example polls every 1 hour. You can override it at runtime:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --poll-seconds 3600
```

Use `Ctrl-C` to stop the harness. By default this does not terminate the training process. To also terminate a child process launched by the harness:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --terminate-child-on-exit
```

Terminate the only live known campaign from a separate terminal:

```bash
./run_campaign terminate
```

If both `cnn` and `transformer` are live, this command refuses and prints both candidates instead of guessing.

Terminate a specific campaign:

```bash
./run_campaign terminate cnn
./run_campaign terminate --campaign-id cnn_pretrain_finetune
./run_campaign terminate configs/experiment_campaigns/cnn_campaign.yaml
```

Escalate after a grace period:

```bash
./run_campaign terminate cnn --force-after 30
```

Require an active process for scripting:

```bash
./run_campaign terminate cnn --require-running
```

If the state file points to a completed or failed run, `--start-at` launches a fresh run by default. Use `--resume` to inspect the existing state instead. Use `--new-run` to force a fresh run, or `--reset-state` to delete the state file before starting.

## Useful Commands

Run one dry status pass without launching jobs, writing files, or calling OpenAI:

```bash
.venv/bin/python scripts/agent_experiment_loop.py run --config configs/agent_experiment_loop.yaml --dry-run --once --start-at 10C
```

Start the CNN campaign from the repository root:

```bash
./run_campaign cnn
```

Start the transformer campaign:

```bash
./run_campaign transformer
```

Run one dry campaign pass:

```bash
./run_campaign cnn --dry-run --once
```

Force a new trial from an existing campaign state:

```bash
./run_campaign cnn --new-trial
```

List configured campaign command names:

```bash
./run_campaign list
```

Inspect a campaign through the root wrapper:

```bash
./run_campaign status cnn
```

Inspect a campaign without calling OpenAI:

```bash
.venv/bin/python scripts/agent_campaign_loop.py status --campaign configs/experiment_campaigns/cnn_campaign.yaml
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

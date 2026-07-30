# Autonomous Scientific Experiment Framework

## Purpose

ZebraFish uses a fully autonomous experimental loop for improving the
commutative representation-learning models. The loop combines deterministic
training and measurement with language-model-assisted scientific reasoning.
There is no approval queue: a candidate that passes deterministic policy is
launched automatically, while an unsafe candidate is rejected and fed back to
the controller for correction.

The system separates five things that are often mixed together:

```text
raw measurement
  -> deterministic observation
  -> interpretation
  -> hypothesis/belief
  -> candidate experiment
  -> executed experiment
```

The LLM interprets evidence and proposes state operations. It does not read
epoch curves as a substitute for numerical analysis, execute shell commands,
or directly edit historical records.

## Campaigns and trials

The current campaigns are sequential pretraining-to-fine-tuning chains:

| Campaign | Stages | Primary outcome |
| --- | --- | --- |
| CNN | `10C -> 13C` | downstream compound macro-F1 |
| Transformer | `12T -> 15T` | downstream compound macro-F1 |

A trial is the scientific unit of optimization. It owns the purpose,
hypothesis and question links, parameter intervention, expected outcomes,
stage executions, final score, and decision about what to try next.

Each stage is an immutable experiment record. For example, a CNN trial has a
`<trial-id>:10C` pretraining record and a `<trial-id>:13C` fine-tuning record.
The controller explicitly records the checkpoint reused by the second stage;
it never chooses a globally newest checkpoint by accident.

Trial success is evaluated only after fine-tuning:

- primary metric: `compound.macro_f1`;
- required guardrail: `action.accuracy >= 0.30`;
- tie-breakers: compound ROC-AUC, balanced accuracy, and accuracy.

Pretraining losses are diagnostic evidence. They are not the campaign score.

## Scientific state

The canonical state is [`state/scientific_state.yaml`](../state/scientific_state.yaml).
It is deliberately compact. Histories, PDFs, checkpoints, confusion matrices,
UMAP files, and predictions remain in the existing artifact directories.

The top-level structure is:

```yaml
version: 2
protocol_version: scientific-loop-v3
writer_protocol: scientific-loop-v3
state_revision: 0
project: {}
entities:
  components: {}
  datasets: {}
  trials: {}
  experiments: {}
  observations: {}
  hypotheses: {}
  beliefs: {}
  questions: {}
  candidate_experiments: {}
relations: []
controller_state: {}
audit_log: []
```

All read-modify-write operations use the advisory state lock, revisioned
atomic replacement, and a writer identity. A stale revision is never silently
overwritten. Immutable trials, experiments, observations, replicate groups,
lockbox confirmations, and belief updates are append-only; candidate status
changes after selection are lifecycle events.

## Validity protocol for new candidates

Historical runs are labelled `legacy_single_seed`, `historical_only`, and
`lockbox_status: not_evaluated`. They remain useful context but cannot support
a new confirmatory claim. New candidates require one preregistered
intervention, an exact paired baseline and checkpoint lineage, a fixed split
manifest, three fixed training seeds, structured primary/guardrail/falsification
rules, and a deterministic cost reservation.

The split seed is independent of the training seed. Every replicate therefore
uses the same train/validation/lockbox partition while changing only registered
training randomness. A candidate is at most `replicate_supported` when all
three replicates are valid, the primary metric is present, the action guardrail
passes in every replicate, and the paired Student-t interval clears the
preregistered minimum effect. Loss curves and loss gaps are diagnostic only.

The lockbox is protected from candidate selection and LLM context. It is
evaluated only after the exploratory replicate gate freezes the candidate, and
its result is written to a separate immutable `lockbox_confirmations` record.
Leaderboards distinguish legacy, exploratory, replicate-supported,
statistically unresolved, guardrail-failed, and lockbox-confirmed evidence.

## Recovery and supervision

The low-cost watchdog checks process identity, artifact freshness, history
advancement, checkpoint availability, non-finite metrics, and resource health
without invoking an LLM. Persistent deterministic triggers may invite the
scientific controller to terminate a run, but a healthy run continues when the
LLM is unavailable. A termination finalizes partial evidence, settles its launch
reservation, and searches for the already selected replicate or a new bounded
candidate; it does not restart an unrelated prerequisite or silently fall back
to a fresh checkpoint.

The daily meta-controller reads [`meta_controller_mandate.md`](meta_controller_mandate.md)
and may repair only allowlisted operational code. It works in an isolated git
worktree, runs deterministic verification selected from changed paths, checks
the main revision and dirty targeted paths before application, stores forward
and reverse patches, and rolls back on post-application health failure.
Objective, primary metric, guardrail meaning, architecture, dataset semantics,
the mandate, and verification policy are proposal-only or protected.

### Immutable experiments

An experiment stores the resolved configuration, stage, execution metadata,
artifact references, outcome, and provenance. Once a run reaches a terminal
state, the record cannot be changed or deleted. A later interpretation is a
new state object linked to the old record.

```yaml
entities:
  experiments:
    trial_20260723:10C:
      trial_id: trial_20260723
      stage: 10C
      status: completed
      configuration:
        trial_config: artifacts/.../10C_config.yaml
      execution:
        run_dir: artifacts/.../runs/...
        artifacts:
          latest_history_csvs: []
          latest_metrics: []
      outcome:
        controller_status: completed
      provenance:
        created_by: autonomous_controller
```

### Observations

Observations are reproducible facts detected by Python code. They include the
measurement, detector rule, threshold, source experiment, and reliability.
Typical types are `loss_plateau`, `validation_plateau`,
`generalisation_gap`, `non_finite_metric`, `unstable_metric`,
`regression_against_baseline`, and `anomalous_runtime`.

The LLM may cite, combine, or reinterpret observations, but it must not create
numeric observations from an image or a vague impression of a curve.

### Hypotheses, beliefs, and questions

A hypothesis is a falsifiable mechanism. It contains a statement, scope,
predicted observations, falsification criteria, and a decision-oriented
belief. A question represents an unresolved choice between explanations and
records which observations would resolve it.

Belief values are confidence scores for decision-making. They are not claimed
to be calibrated statistical posteriors.

### Relations

Relations use a controlled vocabulary. Examples include `tests`, `produced`,
`supports`, `contradicts`, `motivates`, `reuses_checkpoint`, and
`evaluated_on`. Arbitrary relation names are rejected so that the ontology
does not drift over time.

## Deterministic observation layer

The observation engine in `src/observation_engine.py` reads the existing CSV
artifacts. It supports both pretraining metrics (`metric,value`) and
fine-tuning metrics (`target,metric,value`).

The configured detectors are:

1. **Plateau detection**: checks the spread over the final history window.
2. **Generalisation gap**: compares paired `train_*` and `val_*` metrics.
3. **Non-finite detection**: finds NaN and infinite values.
4. **Instability detection**: identifies unusually large trajectory changes.
5. **Comparable regression**: compares a final score with comparable-trial
   median performance.
6. **Runtime anomaly**: compares runtime with comparable runs.

Detector thresholds live in configuration, not in the prompt. This makes the
same raw evidence produce the same observation when reanalysed.

## LLM protocol

The autonomous controller sends a bounded context containing:

- project objective and guardrails;
- active hypotheses and beliefs;
- unresolved questions;
- new observations;
- recent comparable trials;
- current stage and process status;
- remaining budget;
- allowlisted interventions.

Raw epoch-level histories are not sent wholesale.

The response is strict JSON with:

```json
{
  "decision": "propose_trial",
  "reason": "...",
  "evidence_references": ["obs_..."],
  "operations": [
    {
      "operation": "create",
      "path": "entities.beliefs.belief_001",
      "value": {},
      "expected_old": null
    }
  ],
  "candidate": {}
}
```

Supported decisions are `continue`, `propose_trial`, `terminate_trial`,
`stop_campaign`, and `no_action`.

Operations are applied transactionally. Updates may include an expected old
value; if it is stale, the complete update is rejected. Immutable experiment
and observation paths cannot be updated. Every accepted operation is copied
to `audit_log` with actor and timestamp.

## Candidate experiments

Version one supports exactly one controlled trial per candidate. A candidate
must identify:

- a question and at least one hypothesis;
- its purpose and base experiment;
- a configuration patch;
- fixed variables;
- expected outcomes;
- falsification criteria;
- estimated GPU and wall-clock cost;
- risks and mitigations;
- permitted stages.

Multi-arm sweeps, architecture changes, unconstrained parameter searches, and
free-form research programs are not valid autonomous candidates.

## Autonomous launch policy

The policy in `src/autonomous_policy.py` validates every candidate before any
configuration is written or process is started.

A candidate is launchable only when:

- it addresses known scientific state;
- it contains one controlled trial;
- every parameter path is allowlisted;
- values are within configured ranges;
- no forbidden architecture path is touched;
- the patch is within the leaf-count limit;
- its cost fits the campaign and single-trial budgets;
- required evidence, dataset, and checkpoint references exist;
- no conflicting process is running.

There is no human approval state. An invalid candidate is recorded with the
deterministic rejection reasons and included in the next LLM context. Repeated
invalid decisions, unavailable API credits, corrupt state, or exhausted budget
produce `autonomous_safe_stop`. The safe stop preserves all evidence and
prevents additional launches.

## Controller lifecycle

```text
initialise
  -> inspect state and historical artifacts
  -> seed objective question/hypothesis if necessary
  -> ask LLM for a bounded first candidate
  -> validate and launch

running
  -> poll process and runner status
  -> detect artifact progress
  -> wait without LLM while progress is healthy

completed / failed / stale
  -> generate deterministic observations
  -> write immutable stage record
  -> advance to the next stage when appropriate

trial completed
  -> score fine-tuning outcome
  -> write immutable trial record
  -> ask LLM for state updates and next candidate
  -> validate and automatically launch or safe-stop

interrupted
  -> inspect process and checkpoint state
  -> resume when possible
  -> preserve all artifacts
```

Live termination is only possible when the LLM requests `terminate_trial` and
the deterministic live-run policy confirms that the run is clearly wasting
compute or cannot produce useful evidence. Staleness alone is not a launch
gate and does not permit a second active trial.

## Storage and compatibility

The new durable source is the scientific YAML file. Existing campaign state,
CSV ledgers, and Markdown logbook remain useful compatibility artifacts.
The migration and view commands preserve their contents and artifact links;
they do not delete or rewrite historical evidence.

The low-level runner entrypoints and existing stage artifact conventions are
unchanged. The root campaign wrapper now routes normal campaign execution to
the autonomous controller.

Useful commands are:

```bash
./run_campaign cnn
./run_campaign transformer
./run_campaign status cnn
./run_campaign migrate-state cnn
./run_campaign state cnn
./run_campaign observations cnn
./run_campaign candidates cnn
./run_campaign rebuild-views cnn
./run_campaign terminate cnn
```

## Dashboard

`scientific_experiment_dashboard.ipynb` is a read-only, runnable view of the
scientific state and the newest campaign artifact. Launch it from the
repository root with:

```bash
.venv/bin/jupyter notebook scientific_experiment_dashboard.ipynb
```

The notebook shows the active process and stage, recent trials, hypotheses,
open questions, deterministic observations, candidate experiments, and a
Graphviz reasoning graph. Its detail slider ranges from 0 (trial-level
overview) to 5 (all available entity metadata and labelled relations). The
graph omits trials and stages that have no deterministic observation link.
Hovering over a node reveals its complete untruncated scientific text.
The dashboard does not mutate state or launch experiments; it is safe to leave
open while a campaign runs.

## Migration

`./run_campaign migrate-state <campaign>` imports campaign manifests, trial
summaries, stage configurations, stage-run paths, and existing outcomes. The
operation is idempotent: records already present by stable trial/stage ID are
not duplicated.

Machine-readable results are imported as deterministic historical facts.
Free-form logbook interpretation is retained as provenance rather than being
treated as a new measured result.

## Failure and recovery

- **Runner failure**: record the failed stage and ask the LLM whether a
  bounded replacement is justified.
- **Stale live process**: continue monitoring; do not launch a second trial.
- **Missing final artifacts**: classify the stage as incomplete and preserve
  its partial evidence.
- **Invalid LLM JSON**: record the failure and retry with bounded retries.
- **Invalid operation**: reject the entire operation transaction.
- **Unsafe candidate**: record policy reasons and request a correction.
- **API quota exhaustion**: persist `autonomous_safe_stop`.
- **Interrupted controller**: reload JSON stage state and scientific YAML;
  recover from the live process or checkpoint where available.
- **State write interruption**: atomic replacement leaves either the previous
  valid state or the complete new state.

## Verification

The framework is tested at four levels:

1. state schema, immutability, relations, atomic writes, and operation
   conflicts;
2. deterministic metric and trajectory detectors;
3. candidate policy, budgets, allowlists, and safe-stop behavior;
4. an end-to-end fake campaign covering completion, observation extraction,
   structured LLM output, state update, and automatic launch.

Acceptance requires that every terminal stage has an immutable record, every
observation is deterministic and traceable, every LLM mutation is audited,
unsafe candidates never launch, historical campaigns can be imported, and CNN
and transformer campaigns remain runnable through the existing wrapper.

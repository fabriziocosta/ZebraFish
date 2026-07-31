# ZebraFish Autonomous Experiment System Architecture

Architecture version: 1

This document describes how the autonomous experiment system works. It is
orientation and change-impact guidance for the meta-controller; it is not a
runtime configuration file and it cannot override code, configuration, or
scientific state. The meta-controller reads and hashes this document before
each supervisory run and records the hash with its report.

The system has three complementary sources of authority:

- `docs/meta_controller_mandate.md` defines what the supervisory controller is
  allowed and required to do.
- This document explains system boundaries, ownership, dependencies, and
  recovery paths.
- `docs/scientific_experiment_framework.md` defines how experiments,
  observations, hypotheses, and candidates are interpreted scientifically.

Operational settings live in YAML under `configs/`. Scientific runtime truth
lives in `state/scientific_state.yaml`. Tests are executable guarantees for
  the behavior described here.

## Compact supervisory summary

The system is a local, autonomous, state-driven experiment loop:

1. Campaign configuration defines a bounded chain, objective, guardrails,
   parameter allowlists, compute budget, and artifact locations.
2. The campaign controller owns launch/reconciliation decisions. It asks the
   scientific LLM for validated semantic operations and bounded candidates;
   deterministic policy decides whether a candidate may launch.
3. Existing training runners own model execution and write histories,
   checkpoints, summaries, status, and evaluation artifacts.
4. The watchdog performs cheap process and artifact checks. It does not make
   scientific claims and does not invoke the LLM by itself.
5. The observation engine converts approved machine-readable artifacts into
   deterministic, versioned observations. It is the only layer that computes
   detector measurements from raw histories for controller decisions.
6. The scientific state manager is the canonical source of scientific truth.
   Locked revisioned transactions, atomic writes, immutable records, typed
   relations, and audit records protect it from concurrent or stale writers.
7. The meta-controller supervises system reliability on a daily schedule or
   by explicit user-triggered run-now invocation. It may repair only
   allowlisted operational code after isolated verification; scientific
   objective and model changes are proposals only.
8. The dashboard and localhost API are read-only for scientific state. Their
   operational controls affect process lifecycle only; they cannot approve a
   candidate or edit evidence.

The normal dependency direction is:

```text
configs + scientific state
          │
          ├── campaign controller ──> launch reservation ──> runner process
          │                                      │
          │                                      └── artifacts/status
          ├── watchdog ─────────────> health observations
          ├── observation engine ────> deterministic observations
          ├── campaign controller ──> candidate validation and next launch
          ├── meta-controller ───────> operational diagnosis and verified repair
          └── dashboard/API ─────────> normalized read-only presentation
```

## System layers and responsibilities

### Configuration and protocol layer

Campaign YAML declares the project objective, metric roles, guardrails,
stage chain, data references, parameter policy, budgets, detector profiles,
domain contract, and recovery settings. Configuration is loaded by the
controller and resolved into per-run files. Resolved configurations are
artifacts and are referenced by hash in stage records.

Configuration does not replace scientific state: it defines what is allowed
and how a run should execute, while the state records what happened.

### Scientific state manager

`src/scientific_state.py` owns `state/scientific_state.yaml` and all
read-modify-write operations. It provides:

- schema validation and protocol migration;
- advisory locking, revision checks, atomic replacement, and fsync;
- immutable experiment, trial, observation, replicate, confirmation, and
  belief-update records;
- candidate freezing and append-only lifecycle events;
- typed relation validation and referential integrity;
- audit records and controlled controller-state updates.

No controller may edit the YAML directly. A writer must use the transaction
layer so concurrent campaign, watchdog, migration, dashboard-control, and
meta-controller writes serialize consistently.

### Campaign controller

`src/autonomous_campaign.py` owns scientific progression. It retrieves a
compact campaign-scoped context, requests a strict LLM decision, validates
operations, evaluates candidates against deterministic policy, reserves
compute, and launches only bounded protocol-compliant trials.

The controller owns scientific decisions such as whether a candidate addresses
the active question. It does not calculate detector values from raw CSV files
and it does not bypass checkpoint, split, seed, guardrail, or budget checks.

### Runner processes

The existing stage runners own model execution. They receive a resolved
configuration, dataset/split references, and checkpoint references. They write
training histories, summaries, checkpoints, runtime metadata, predictions,
latent representations, confusion metrics, and domain-evaluation artifacts.

Runners must not mutate canonical scientific state. The controller imports and
freezes their results after deterministic artifact discovery and validation.

### Watchdog and observation engine

The watchdog polls liveness, process identity, history advancement, checkpoint
availability, non-finite metrics, GPU/disk availability, and artifact
freshness. It updates runtime health and emits registered triggers; it does not
invoke the LLM on every poll.

The observation engine computes metric-specific slopes, plateaus, gaps,
instability, regressions, runtime anomalies, seed sensitivity, trajectory
relations, domain constraints, and insufficient-data results. Every
observation carries a deterministic ID, detector/threshold versions, source
artifact hashes, measurements, and provenance.

### Meta-controller

`src/meta_controller.py` is a reliability supervisor, not a second campaign
planner. It reads and hashes both the mandate and this architecture document,
collects a compact snapshot, asks for a strict diagnosis, and records the
result. It may patch only configured operational paths in an isolated git
worktree, run deterministic verification, and apply a passing patch only when
the main checkout has not changed underneath it.

It may repair orchestration, state reconciliation, observation/monitoring,
dashboard/API reliability, tests, and bounded runtime settings. It must record
proposal-only scientific changes instead of applying them.

### Dashboard and API

`src/dashboard_api.py` adapts YAML and artifacts into a normalized
`InvestigationState`. The React dashboard validates this payload with Zod and
displays current progress, evidence, uncertainty, domain constraints, graph
relations, and meta-controller findings.

The API is localhost-only. Scientific endpoints are read-only. Start, stop,
continue, and run-now controls are operational process controls and are
audited; no dashboard endpoint approves, rejects, launches, or edits a
scientific candidate.

## State and artifact ownership

| Resource | Authoritative owner | Other layers may do |
| --- | --- | --- |
| Campaign YAML | repository/configuration | read and resolve |
| Scientific state YAML | state manager | read through validated loader |
| Campaign/stage runtime JSON | campaign controller and runner | watchdog/dashboard read |
| Histories and summaries | runner | observation engine read |
| Checkpoints and manifests | runner plus controller validation | dashboard link/read |
| Deterministic observations | observation engine | controllers consume |
| Candidates and belief updates | scientific state transactions | campaign LLM proposes, policy validates |
| Meta-controller reports | meta-controller through state manager | dashboard reads |
| Generated logbook/leaderboard | reporting/view builder | users read; never source of truth |
| Dashboard acknowledgement state | browser local storage | no scientific effect |

Artifacts are evidence inputs, not state mutations. A missing, stale, or
incompatible artifact is recorded as missing or invalid; it is never silently
replaced by a nearby file.

## Experiment lifecycle and recovery

The normal lifecycle is:

```text
initialise/reconcile
  → inspect state and legacy history
  → register or select bounded candidate
  → reserve launch atomically
  → launch stage/replicate
  → watchdog polls cheaply
  → runner writes artifacts
  → deterministic observations and evaluation
  → freeze immutable stage/trial record
  → advance chain or aggregate replicate group
  → exploratory gate and optional protected lockbox confirmation
  → update leaderboard/belief events
  → request next bounded candidate
```

Recovery is deterministic and idempotent:

- an interrupted launch is reconciled from reservation and process identity;
- a live process with stale controller metadata is treated as live, with a
  metadata warning;
- a dead process is finalized from available artifacts and marked failed or
  terminated;
- a missing or incompatible checkpoint rejects reuse rather than silently
  restarting a prerequisite;
- a stopped trial releases or settles its reservation and seeks a registered
  replacement or a new valid bounded candidate;
- a state revision conflict retries only non-conflicting controller fields and
  safe-stops on unresolved scientific conflicts;
- repeated malformed LLM responses, quota failures, or failed repairs enter a
  safe-stop while preserving all evidence.

The campaign supervisor and meta-controller use separate process/cycle locks.
The shared scientific-state transaction lock serializes their writes.

## Immutable versus mutable data

Immutable after completion or freeze:

- trials, stage experiments, observations, artifacts/manifests, replicate
  groups, lockbox confirmations, belief updates, preregistered predictions,
  and frozen candidates;
- protocol, split, checkpoint, configuration, dataset, code, and contract
  hashes attached to executed evidence;
- audit events and meta-controller run records.

Mutable under validated transactions:

- active controller lifecycle fields;
- launch reservation status;
- current process health and artifact freshness;
- candidate lifecycle status through append-only events;
- generated compatibility views;
- meta-controller scheduler status.

An update to mutable state must identify the expected old value where a
semantic conflict is possible. An update that would rewrite immutable evidence
is rejected.

## Allowed intervention boundaries

The campaign controller may select and launch only candidates that pass the
allowlist, range, checkpoint, split, seed, budget, protocol, and safety policy.
The watchdog may report health and trigger registered recovery logic. The
meta-controller may patch allowlisted operational code after verification.

No controller may silently alter:

- the active scientific objective, primary metric, direction, guardrail
  meaning, dataset semantics, labels, architecture, or protected lockbox;
- immutable evidence or preregistered candidate predictions;
- the mandate, architecture document, protected policy, secrets, artifacts,
  model implementation, or dataset definitions.

Changes to scientific meaning create a new versioned proposal/campaign family.
They are reviewed outside the autonomous repair authority and never rewrite
historical results.

## Configuration and parameter flow

```text
campaign YAML
  → merged loop/meta/domain configuration
  → deterministic validation and resolved configuration
  → config hash + stage artifact
  → runner command and checkpoint handoff
  → runtime artifacts and observations
```

Every changed parameter must have an explicit allowlist path and range. Broad
prefixes do not authorize unknown leaves. A candidate records fixed and varied
variables, expected outcomes, falsification rules, and cost reservation before
launch. A stage records the exact resolved configuration, code/environment
metadata, split/dataset hashes, seed, checkpoint compatibility signature, and
artifact manifest.

## Common failure modes and diagnostic evidence

| Symptom | First evidence to inspect | Safe response |
| --- | --- | --- |
| Process appears stopped but history is fresh | PID, process identity, history/checkpoint mtimes | reconcile as live; warn on stale metadata |
| History stops advancing | process status, last history row, runner stderr, checkpoint mtime | watchdog trigger; terminate only under registered rule |
| Train/validation gap grows | metric-specific gap observations and domain/class metrics | request bounded controller decision; do not equate gap with objective failure |
| Primary metric is absent | stage metric specification and summary files | label unavailable; never substitute loss as primary |
| Candidate cannot launch | validation reasons, reservation, checkpoint manifest | reject and record exact deterministic reason |
| State write conflicts | state revision, writer identity, lock/audit records | reload/merge safe controller fields or safe-stop |
| Meta repair fails verification | isolated worktree diff and verification matrix | do not apply; record failure and rollback data |
| Domain geometry looks attractive | original latent-space evaluation and bootstrap result | ignore UMAP-only evidence; require registered constraint result |
| API/dashboard disagrees with runner | live process and artifact freshness versus controller metadata | prioritize live evidence and display metadata warning |

The diagnostic order is: process identity and freshness, state revision and
audit, artifact manifest, deterministic observations, protocol eligibility,
then LLM interpretation. The LLM is not the source of raw measurements.

## Change-impact guidance for the meta-controller

Before proposing a repair, classify changed paths:

1. State manager, campaign controller, checkpoint/protocol, watchdog, or
   observation engine changes require the full Python verification set and a
   reconciliation check.
2. Dashboard/API changes require API tests and a production dashboard build;
   they must retain read-only scientific behavior.
3. Test orchestration changes require the mandatory verification matrix and
   cannot remove tests, weaken assertions, or reduce coverage.
4. Runtime polling/retry/timeout changes require watchdog and recovery tests;
   they cannot change scientific parameter ranges or termination meaning.
5. Configuration changes touching objective, metric, guardrails, domain
   contracts, split/lockbox semantics, architecture, or labels are
   proposal-only.

The meta-controller must inspect repository status and targeted dirty paths
before applying a patch. It must use an isolated worktree, deterministic
verification selected from changed paths, a reverse patch, post-application
health checks, and an auditable rollback result. If this document or the
constitutional mandate is missing or cannot be hashed, the supervisory run
fails closed and makes no repair.

## Versioning and precedence

This document is versioned by its explicit `Architecture version` line and
content hash. A hash identifies the exact explanation read by a supervisory
run; it does not make prose executable. Code, validated configuration, and
scientific state remain authoritative when this document is incomplete or
out of date. A mismatch between documented and observed behavior is itself a
diagnostic finding for the meta-controller, not permission to reinterpret the
system silently.

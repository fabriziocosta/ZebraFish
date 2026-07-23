# Experiments Logbook

This file records pretraining and fine-tuning runs by experiment id. Each entry links the timestamped artifacts, summarizes what happened, and states the proposed next round.

## Artifact Cleanup: 2026-06-25

Legacy artifact folders outside `artifacts/campaigns/` were pruned after migrating the campaign harness to campaign-owned trial folders. Older logbook entries below still preserve the historical analysis, but their legacy artifact links should be read as records of the original locations, not as live local files.

Pruned folders:

- `artifacts/pretrained_commutative_cnn`
- `artifacts/pretrained_commutative_transformer`
- `artifacts/nb13C_commutative_cnn_full_finetune`

## 10C_pretrain_commutative_cnn_legacy_20260619_163630

- kind: `pretraining`
- artifact_dir: [pretrained_commutative_cnn](artifacts/pretrained_commutative_cnn)
- config: [config.yaml](artifacts/pretrained_commutative_cnn/config.yaml)
- checkpoint: [encoder_state_v9.pt](artifacts/pretrained_commutative_cnn/encoder_state_v9.pt)
- loss_pdfs: pruned legacy top-level loss plot folder on 2026-06-25 after migration to timestamped run folders
- latest_loss_pdfs: pruned with legacy top-level loss plot folder

### Analysis

The run completed without crashing and restored the best monitored checkpoint before saving. Training self-probe loss fell from `2.5006` to `0.3181`, but validation self-probe loss became unstable immediately after epoch 23. The best region was epoch `22-23` (`val_self_probe_loss=0.8554` at epoch 23), followed by large validation spikes at epochs 24-29. Early stopping fired at epoch 31 with `best_epoch=023`.

This looks like a real pretraining instability, not just a missing artifact. The unlabeled split is broadly balanced by control/treatment and concentration band, so the next suspect is objective or normalization instability. The commutative CNN uses BatchNorm with batch size 8, and the failure is mostly eval/validation-side, which is consistent with unstable running statistics. Cross-probe pressure also starts ramping around the same period.

### Next Round Proposal

Run a diagnostic 10C pretrain that removes the secondary pressures and tests stability first: replace BatchNorm with GroupNorm or InstanceNorm in the commutative CNN, set `lambda_cross=0.0`, keep prototype and latent alignment disabled, use `learning_rate=3e-5`, `weight_decay=1e-3`, and monitor validation self-probe from epoch 8. Reintroduce cross/prototype objectives only after validation self-probe is stable.

## 13C_finetune_commutative_cnn_legacy_20260622_133914

- kind: `fine_tuning`
- artifact_dir: [nb13C_commutative_cnn_full_finetune](artifacts/nb13C_commutative_cnn_full_finetune)
- config: [config.json](artifacts/nb13C_commutative_cnn_full_finetune/config.json)
- history: [history.csv](artifacts/nb13C_commutative_cnn_full_finetune/history.csv)
- summary_metrics: [summary_metrics.csv](artifacts/nb13C_commutative_cnn_full_finetune/summary_metrics.csv)
- checkpoint: [model_state.pt](artifacts/nb13C_commutative_cnn_full_finetune/model_state.pt)
- per_class_reports: [per_class_reports](artifacts/nb13C_commutative_cnn_full_finetune/per_class_reports)
- latest_loss_pdfs: [binary_hot_start latest](artifacts/nb13C_commutative_cnn_full_finetune/loss_plots/binary_hot_start/latest.loss-curves.pdf), [fine_tune latest](artifacts/nb13C_commutative_cnn_full_finetune/loss_plots/fine_tune/latest.loss-curves.pdf)

### Analysis

The v9 fine-tune improved action ranking and drug-only separation but failed calibration across control and drug classes. Action macro-F1 reached `0.4053` and macro ROC-AUC reached `0.7873`, better than the older full fine-tune and the 7C from-scratch run on those metrics. Drug-only action macro-F1 excluding control was `0.5713`.

The decision rule is still poor: the model predicted AChE for `40/71` holdout samples, including `18/25` true Water samples. Water recall collapsed to `0.12`, while AChE recall became `1.0`. Compound and concentration metrics are not meaningful for this run because final-phase `compound_weight=0.0` and `concentration_weight=0.0`, so those heads were reported but not trained by the supervised objective.

### Next Round Proposal

For the next 13C run, restore a small water boundary without letting it dominate: disable binary hot-start or limit it to 2 epochs, set `water_vs_other_weight` around `0.05-0.10`, and consider `class_weighting=None` or capped class weights rather than full balanced weighting. If compound and concentration are reported, train them lightly with `compound_weight=0.03` and `concentration_weight=0.03`; otherwise suppress those reports for this experiment.

## 10C_pretrain_commutative_cnn_20260623_093629

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260623_093629](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629)
- `config`: [10C_pretrain_commutative_cnn_20260623_093629_config.json](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_config.json)
- `history`: [10C_pretrain_commutative_cnn_20260623_093629_history.csv](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_history.csv)
- `summary_metrics`: [10C_pretrain_commutative_cnn_20260623_093629_summary_metrics.csv](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_summary_metrics.csv)
- `checkpoint`: [10C_pretrain_commutative_cnn_20260623_093629_encoder_state.pt](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_encoder_state.pt)
- `loss_pdfs`: [loss_plots](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/loss_plots)
- `latest_loss_pdfs`: [10C_pretrain_commutative_cnn_20260623_093629_latest.loss-curves.pdf](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/loss_plots/10C_pretrain_commutative_cnn_20260623_093629_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/loss_plots/pretraining/latest.loss-curves.pdf)
- `latest_encoder_pointer`: [10C_pretrain_commutative_cnn_20260623_093629_encoder_state.pt](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_encoder_state.pt)

### Analysis

The GroupNorm/no-cross-pressure diagnostic fixed the previous validation instability. The run completed all `70` epochs with `best_epoch=70`, so early stopping never fired. Validation self-probe loss improved monotonically every epoch, from `6.9637` to `2.7334` (`60.7%` drop), while training self-probe loss fell from `2.5463` to `0.9560` (`62.5%` drop). This is a real stabilization versus the prior BatchNorm/cross-pressure run, where validation self-probe spiked after the best epoch.

The checkpoint is usable, but probably undertrained rather than fully converged: the best checkpoint is the final epoch, the last 10 validation points are still descending, and final validation self-probe is still `2.86x` the training self-probe. Local and region-time reconstruction both improved strongly, but correlation reconstruction only dropped about `28.9%`, so the representation may still be weak on temporal relationship structure. Cross-probe loss rose during the run, but `lambda_cross=0.0`, so that is only an unoptimized diagnostic readout, not a trained objective failure.

### Next Round Proposal

Use this checkpoint for the next 13C calibration test, but also run one extended 10C continuation-style experiment before reintroducing auxiliary objectives: keep `normalization="group"`, `lambda_cross=0.0`, prototype/latent alignment disabled, and `learning_rate=3e-5`, but extend to `120-140` epochs with patience around `16`. If validation self-probe plateaus cleanly, then reintroduce cross-probe pressure in a separate run at `lambda_cross=0.005-0.01` with a long warmup/ramp.

### Config Snapshot

```json
{
  "loss_weight_config": {
    "action_weight": 1.0,
    "compound_weight": 0.2,
    "concentration_weight": 0.2,
    "cross_ramp_epochs": 0,
    "cross_warmup_epochs": 0,
    "lambda_align": 0.0,
    "lambda_cross": 0.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 0.05,
    "probe_alpha_derivative": 0.25,
    "probe_alpha_frequency": 0.1,
    "probe_alpha_local": 1.0,
    "probe_alpha_region_time": 1.0,
    "probe_mask_probability": 1.0,
    "prototype_alignment_weight": 0.0,
    "prototype_ramp_epochs": 36,
    "prototype_temperature": 0.25,
    "prototype_warmup_epochs": 16,
    "teacher_student_warmup_epochs": 0,
    "water_vs_other_weight": 0.0
  },
  "model_config": {
    "dropout": 0.25,
    "embedding_dim": 64,
    "normalization": "group",
    "num_prototypes": 64,
  ...
```

## 13C_finetune_commutative_cnn_20260624_162408_incomplete

- kind: `fine_tuning`
- artifact_dir: [13C_finetune_commutative_cnn_20260624_162408](artifacts/nb13C_commutative_cnn_full_finetune/runs/13C_finetune_commutative_cnn_20260624_162408)
- loss_pdfs: [fine_tune](artifacts/nb13C_commutative_cnn_full_finetune/runs/13C_finetune_commutative_cnn_20260624_162408/loss_plots/fine_tune)
- latest_loss_pdfs: [latest.loss-curves.pdf](artifacts/nb13C_commutative_cnn_full_finetune/runs/13C_finetune_commutative_cnn_20260624_162408/loss_plots/fine_tune/latest.loss-curves.pdf)

### Analysis

This 13C attempt did not produce usable evaluation evidence. The run folder contains only `epoch_001`, `epoch_002`, and `latest` fine-tune loss PDFs, with no persisted config, raw history CSV, summary metrics, confusion matrices, prediction tables, checkpoint, or UMAP outputs. Treat it as interrupted before the artifact persistence cell completed, not as a valid hyperparameter result.

The lack of metrics means we cannot judge compound discrimination from this run. The previous complete 13C result still stands as the last measured fine-tune, while the latest 10C GroupNorm/no-cross checkpoint is the best available pretrained encoder.

### Next Round Proposal

Patch 13C explicitly for compound discrimination: keep binary hot-start disabled, monitor `compound_loss`, use balanced criteria, raise `compound_weight` to `0.80`, lower `action_weight` to `0.65`, keep only a small water boundary at `0.03`, and train concentration lightly at `0.05`. Evaluate the next complete run primarily on compound accuracy, compound macro-F1, compound ROC-AUC, and the compound confusion matrix; action metrics should be tracked as a secondary degradation check.

## 10C_pretrain_commutative_cnn_20260624_164816_incomplete

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260624_164816](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260624_164816)
- config: [10C_pretrain_commutative_cnn_20260624_164816_config.yaml](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260624_164816/10C_pretrain_commutative_cnn_20260624_164816_config.yaml)
- run_log: [10C_pretrain_20260624_164809.log](artifacts/notebook_run_logs/10C_pretrain_20260624_164809.log)

### Analysis

This was a failed background launch, not a training result. The notebook created the per-run config folder, then the ipykernel exited with `Parent appears to have exited, shutting down.` No history CSV, summary metrics, checkpoint, or loss PDFs were written. Do not use this run for model selection.

### Next Round Proposal

Relaunch 10C with a fully detached process so nbconvert and the kernel survive the parent shell exiting. Keep the same extended warm-start hyperparameters.

## 10C_pretrain_commutative_cnn_20260624_164930_interrupted

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260624_164930](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260624_164930)
- run_log: [10C_pretrain_20260624_164923.log](artifacts/notebook_run_logs/10C_pretrain_20260624_164923.log)
- pid_file: [10C_pretrain_20260624_164923.pid](artifacts/notebook_run_logs/10C_pretrain_20260624_164923.pid)
- latest_loss_pdf: [latest.loss-curves.pdf](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260624_164930/loss_plots/pretraining/latest.loss-curves.pdf)
- last_epoch_pdf: [epoch_108.loss-curves.pdf](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260624_164930/loss_plots/pretraining/epoch_108.loss-curves.pdf)

### Analysis

Launched as a detached 10C extended warm-start run on June 24, 2026. The run was manually terminated on June 25, 2026 after about `17` hours because it was still running. Terminated process groups: nbconvert parent PID `2595155` and ipykernel PID `2595179`.

The run reached the live PDF for epoch `108` of `140`, but it did not reach the artifact persistence cell. Therefore it has no final raw history CSV, summary metrics, persisted encoder checkpoint, or updated latest checkpoint pointer. Because this run started before live history CSV instrumentation was added, the only usable evidence is the live loss PDFs.

Visual assessment of `epoch_108.loss-curves.pdf`: validation and training self-probe losses were still decreasing at termination, so the run had not clearly plateaued. However, cross-probe losses increased substantially even with `lambda_cross=0`, and validation cross-probe/local losses diverged from the self-probe objective. This is not a clean checkpoint candidate and should not be used as the basis for 13C.

### Next Round Proposal

Rerun 10C through the new `.py` harness so live history CSVs and runner status JSON are written. Keep no-cross pressure for the immediate rerun only if the goal is to obtain a complete baseline checkpoint; otherwise reduce the total epoch target and add explicit decision points from raw CSV history before spending another long run. Do not run 13C from this interrupted run because no checkpoint was persisted.

<!-- campaign:cnn_pretrain_finetune:init:cnn_pretrain_finetune_20260625_115021_08a733:start -->
## Campaign Start: cnn_pretrain_finetune_20260625_115021_08a733

- Campaign: `cnn_pretrain_finetune`
- Trial folder: [artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_115021_08a733](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_115021_08a733)
- Stages: `10C -> 13C`
- Objective: `compound.macro_f1`

### Previous Results And Next Plan

### Previous results reviewed
- Complete 13C evidence used: [20260622 fine-tune run folder](artifacts/nb13C_commutative_cnn_full_finetune), with [config](artifacts/nb13C_commutative_cnn_full_finetune/config.json), [history](artifacts/nb13C_commutative_cnn_full_finetune/history.csv), [summary metrics](artifacts/nb13C_commutative_cnn_full_finetune/summary_metrics.csv), and [loss PDFs](artifacts/nb13C_commutative_cnn_full_finetune/loss_plots/fine_tune/latest.loss-curves.pdf). Key fact: action performed moderately (macro-F1 ~0.405), but compound was not actually optimized in final phase (`compound_weight=0.0`), so compound metrics are not decision-grade for this objective.
- Complete 10C evidence used: [20260623 pretrain run folder](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629), [config](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_config.json), [history](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_history.csv), [summary](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_summary_metrics.csv), [latest PDF](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/loss_plots/pretraining/latest.loss-curves.pdf). This is the last stable completed pretrain.
- Weak/latest-fallback evidence noted: [13C 20260624 incomplete](artifacts/nb13C_commutative_cnn_full_finetune/runs/13C_finetune_commutative_cnn_20260624_162408/loss_plots/fine_tune/latest.loss-curves.pdf) and [10C 20260624 interrupted](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260624_164930/loss_plots/pretraining/latest.loss-curves.pdf) came from latest-folder fallback with low reliability and no persisted metrics/checkpoint; treated only as deterministic interruption/staleness evidence.

### Next experiment to run
Run the proposed full chain `cnn_pretrain_finetune_20260625_115021_08a733` with a compound-focused 13C loss rebalance (keep 10C config unchanged for this first launch).

### Why this should help
- Campaign objective is **13C compound macro-F1** (required primary metric). The only completed 13C run cannot satisfy this objective because compound supervision was effectively off in the decisive phase.
- Reweighting 13C toward compound creates directly attributable pressure on the target head while keeping enough action weight to satisfy the action accuracy guardrail (`>=0.3`).
- Using a minimal patch isolates causality for the first scored trial.

### Patch to apply
- 13C loss-weight rebalance:
  - increase `compound_weight` to `0.80`
  - reduce `action_weight` to `0.65`
  - keep light auxiliary supervision with `concentration_weight=0.05`
  - keep small boundary pressure with `water_vs_other_weight=0.03`

### Monitoring plan
- Primary gate: 13C **compound macro-F1** from summary metrics (required).
- Tie-breakers: `roc_auc_ovr_macro`, `balanced_accuracy`, `accuracy` on compound.
- Guardrail: verify `action.accuracy >= 0.30`; if violated, revert part of action down-weight in next patch.
- Artifact checks before scoring: ensure the new 13C run has persisted config/history/summary/checkpoint/confusion outputs (not just loss PDFs). If artifacts are missing again, classify as deterministic failure and do not use for model selection.

### Initial Trial Patch

```json
{
  "13C": {
    "loss_weight_config": {
      "action_weight": 0.65,
      "compound_weight": 0.8,
      "concentration_weight": 0.05,
      "water_vs_other_weight": 0.03
    }
  }
}
```
<!-- campaign:cnn_pretrain_finetune:init:cnn_pretrain_finetune_20260625_115021_08a733:end -->

<!-- campaign:cnn_pretrain_finetune:init:cnn_pretrain_finetune_20260625_152652_a8e2d2:start -->
## Campaign Start: cnn_pretrain_finetune_20260625_152652_a8e2d2

- Campaign: `cnn_pretrain_finetune`
- Trial folder: [artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2)
- Stages: `10C -> 13C`
- Objective: `compound.macro_f1`

### Previous Results And Next Plan

## Previous results reviewed
- Completed 13C evidence: [run folder](artifacts/nb13C_commutative_cnn_full_finetune), [config](artifacts/nb13C_commutative_cnn_full_finetune/config.json), [history.csv](artifacts/nb13C_commutative_cnn_full_finetune/history.csv), [summary_metrics.csv](artifacts/nb13C_commutative_cnn_full_finetune/summary_metrics.csv), [fine_tune latest PDF](artifacts/nb13C_commutative_cnn_full_finetune/loss_plots/fine_tune/latest.loss-curves.pdf). This run had action signal (macro-F1 ~0.405 in prior notes) but final-phase `compound_weight=0.0`, so compound metrics are not decision-grade for the campaign objective.
- Completed 10C evidence: [run folder](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629), [config](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_config.json), [history.csv](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_history.csv), [summary_metrics.csv](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/10C_pretrain_commutative_cnn_20260623_093629_summary_metrics.csv), [latest PDF](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260623_093629/loss_plots/pretraining/latest.loss-curves.pdf). This is the last complete stable pretrain checkpoint source.
- Weak evidence (latest-folder fallback only): [13C incomplete 20260624](artifacts/nb13C_commutative_cnn_full_finetune/runs/13C_finetune_commutative_cnn_20260624_162408/loss_plots/fine_tune/latest.loss-curves.pdf) and [10C interrupted 20260624](artifacts/pretrained_commutative_cnn/runs/10C_pretrain_commutative_cnn_20260624_164930/loss_plots/pretraining/latest.loss-curves.pdf) had no persisted metrics/checkpoints, so treated as deterministic interruption/staleness, not model-selection evidence.

## Next experiment to run
- Trial: `cnn_pretrain_finetune_20260625_152652_a8e2d2`
- Plan: run full `10C -> 13C` chain, keep 10C unchanged, apply a compound-focused 13C loss rebalance for the first scored campaign trial.

## Why this should help
- Objective is downstream **13C compound macro-F1** (required). The last complete 13C run did not optimize compound in its decisive phase.
- Rebalancing toward compound provides direct training pressure on the target head while retaining enough action supervision to protect the `action.accuracy >= 0.3` guardrail.
- Minimal first patch improves attribution if metric movement occurs.

## Patch to apply
- 13C `loss_weight_config`:
  - `compound_weight: 0.8`
  - `action_weight: 0.65`
  - `concentration_weight: 0.05`
  - `water_vs_other_weight: 0.03`

## Monitoring plan
- Primary score: `compound.macro_f1` (required).
- Tie-breakers: `compound.roc_auc_ovr_macro`, `compound.balanced_accuracy`, `compound.accuracy`.
- Guardrail: verify `action.accuracy >= 0.30`.
- Reliability checks before scoring: confirm persisted 13C [config](artifacts/nb13C_commutative_cnn_full_finetune), history CSV, summary CSV, and loss PDFs exist (not PDF-only incomplete runs). If artifacts are missing again, mark deterministic failure/staleness and exclude from selection.

### Initial Trial Patch

```json
{
  "13C": {
    "loss_weight_config": {
      "action_weight": 0.65,
      "compound_weight": 0.8,
      "concentration_weight": 0.05,
      "water_vs_other_weight": 0.03
    }
  }
}
```
<!-- campaign:cnn_pretrain_finetune:init:cnn_pretrain_finetune_20260625_152652_a8e2d2:end -->

## 10C_pretrain_commutative_cnn_20260625_152717

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260625_152717](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717)
- `config`: [10C_pretrain_commutative_cnn_20260625_152717_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/10C_pretrain_commutative_cnn_20260625_152717_config.json)
- `history`: [10C_pretrain_commutative_cnn_20260625_152717_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/10C_pretrain_commutative_cnn_20260625_152717_history.csv)
- `summary_metrics`: [10C_pretrain_commutative_cnn_20260625_152717_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/10C_pretrain_commutative_cnn_20260625_152717_summary_metrics.csv)
- `checkpoint`: [10C_pretrain_commutative_cnn_20260625_152717_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/10C_pretrain_commutative_cnn_20260625_152717_encoder_state.pt)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/loss_plots)
- `latest_loss_pdfs`: [10C_pretrain_commutative_cnn_20260625_152717_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/loss_plots/10C_pretrain_commutative_cnn_20260625_152717_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/loss_plots/pretraining/latest.loss-curves.pdf)
- `latest_encoder_pointer`: [10C_pretrain_commutative_cnn_20260625_152717_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260625_152652_a8e2d2/outputs/10C/runs/10C_pretrain_commutative_cnn_20260625_152717/10C_pretrain_commutative_cnn_20260625_152717_encoder_state.pt)

### Analysis

Agent-run pretraining: inspect live history CSVs and PDFs for validation self-probe trajectory, train/validation gap, and whether the run plateaued before early stopping.

### Next Round Proposal

If validation self-probe plateaus cleanly, use this checkpoint for 13C; otherwise patch the next pretraining YAML based on the observed failure mode.

### Config Snapshot

```json
{
  "loss_weight_config": {
    "action_weight": 1.0,
    "compound_weight": 0.2,
    "concentration_weight": 0.2,
    "cross_ramp_epochs": 0,
    "cross_warmup_epochs": 0,
    "lambda_align": 0.0,
    "lambda_cross": 0.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 0.05,
    "probe_alpha_derivative": 0.25,
    "probe_alpha_frequency": 0.1,
    "probe_alpha_local": 1.0,
    "probe_alpha_region_time": 1.0,
    "probe_mask_probability": 1.0,
    "prototype_alignment_weight": 0.0,
    "prototype_ramp_epochs": 36,
    "prototype_temperature": 0.25,
    "prototype_warmup_epochs": 16,
    "teacher_student_warmup_epochs": 0,
    "water_vs_other_weight": 0.0
  },
  "model_config": {
    "dropout": 0.25,
    "embedding_dim": 64,
    "normalization": "group",
    "num_prototypes": 64,
  ...
```

<!-- campaign:cnn_pretrain_finetune:init:cnn_pretrain_finetune_20260629_100031_3ff738:start -->
## Campaign Start: cnn_pretrain_finetune_20260629_100031_3ff738

- Campaign: `cnn_pretrain_finetune`
- Trial folder: [artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738)
- Stages: `10C -> 13C`
- Objective: `compound.macro_f1`

### Previous Results And Next Plan

**Previous results reviewed:** The strongest completed evidence is the earlier full 13C fine-tune run (with config/history/summary), which showed usable action performance but had `compound_weight=0.0` in the decisive phase, so its compound macro-F1 is not decision-grade for this campaign objective; the strongest completed 10C evidence is the stable pretrain run with persisted history/summary/checkpoint, and there is also a later completed 10C campaign run with persisted artifacts but no paired completed 13C result yet. The interrupted 10C and incomplete 13C runs from the later date were reviewed only as weak/latest-folder fallback evidence for staleness/failure patterns because they lacked full persisted metrics/checkpoints.

**Next experiment to run:** Launch the proposed full chain `cnn_pretrain_finetune_20260629_100031_3ff738` with 10C unchanged and a compound-focused 13C loss rebalance. **Why this should help:** The objective is downstream compound `macro_f1` (required), so increasing direct compound supervision is the most attributable first move while retaining enough action supervision for the guardrail. **Patch to apply:** set 13C loss weights to `compound_weight=0.80`, `action_weight=0.65`, `concentration_weight=0.05`, `water_vs_other_weight=0.03`. **Monitoring plan:** score by compound macro-F1 first, then tie-break with compound roc-auc/balanced-accuracy/accuracy, and enforce `action.accuracy >= 0.30`; also require persisted config/history/summary/checkpoint artifacts before accepting results for selection.

### Initial Trial Patch

```json
{
  "13C": {
    "loss_weight_config": {
      "action_weight": 0.65,
      "compound_weight": 0.8,
      "concentration_weight": 0.05,
      "water_vs_other_weight": 0.03
    }
  }
}
```
<!-- campaign:cnn_pretrain_finetune:init:cnn_pretrain_finetune_20260629_100031_3ff738:end -->

## 10C_pretrain_commutative_cnn_20260629_100055

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260629_100055](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055)
- `config`: [10C_pretrain_commutative_cnn_20260629_100055_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/10C_pretrain_commutative_cnn_20260629_100055_config.json)
- `history`: [10C_pretrain_commutative_cnn_20260629_100055_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/10C_pretrain_commutative_cnn_20260629_100055_history.csv)
- `summary_metrics`: [10C_pretrain_commutative_cnn_20260629_100055_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/10C_pretrain_commutative_cnn_20260629_100055_summary_metrics.csv)
- `checkpoint`: [10C_pretrain_commutative_cnn_20260629_100055_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/10C_pretrain_commutative_cnn_20260629_100055_encoder_state.pt)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/loss_plots)
- `latest_loss_pdfs`: [10C_pretrain_commutative_cnn_20260629_100055_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/loss_plots/10C_pretrain_commutative_cnn_20260629_100055_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/loss_plots/pretraining/latest.loss-curves.pdf)
- `latest_encoder_pointer`: [10C_pretrain_commutative_cnn_20260629_100055_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/10C_pretrain_commutative_cnn_20260629_100055_encoder_state.pt)

### Analysis

Agent-run pretraining: inspect live history CSVs and PDFs for validation self-probe trajectory, train/validation gap, and whether the run plateaued before early stopping.

### Next Round Proposal

If validation self-probe plateaus cleanly, use this checkpoint for 13C; otherwise patch the next pretraining YAML based on the observed failure mode.

### Config Snapshot

```json
{
  "loss_weight_config": {
    "action_weight": 1.0,
    "compound_weight": 0.2,
    "concentration_weight": 0.2,
    "cross_ramp_epochs": 0,
    "cross_warmup_epochs": 0,
    "lambda_align": 0.0,
    "lambda_cross": 0.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 0.05,
    "probe_alpha_derivative": 0.25,
    "probe_alpha_frequency": 0.1,
    "probe_alpha_local": 1.0,
    "probe_alpha_region_time": 1.0,
    "probe_mask_probability": 1.0,
    "prototype_alignment_weight": 0.0,
    "prototype_ramp_epochs": 36,
    "prototype_temperature": 0.25,
    "prototype_warmup_epochs": 16,
    "teacher_student_warmup_epochs": 0,
    "water_vs_other_weight": 0.0
  },
  "model_config": {
    "dropout": 0.25,
    "embedding_dim": 64,
    "normalization": "group",
    "num_prototypes": 64,
  ...
```

## 13C_finetune_commutative_cnn_20260701_050638

- kind: `fine_tuning`
- artifact_dir: [13C_finetune_commutative_cnn_20260701_050638](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638)
- `config`: [13C_finetune_commutative_cnn_20260701_050638_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/13C_finetune_commutative_cnn_20260701_050638_config.json)
- `history`: [13C_finetune_commutative_cnn_20260701_050638_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/13C_finetune_commutative_cnn_20260701_050638_history.csv)
- `summary_metrics`: [13C_finetune_commutative_cnn_20260701_050638_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/13C_finetune_commutative_cnn_20260701_050638_summary_metrics.csv)
- `checkpoint`: [13C_finetune_commutative_cnn_20260701_050638_model_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/13C_finetune_commutative_cnn_20260701_050638_model_state.pt)
- `per_class_reports`: [per_class_reports](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/per_class_reports)
- `confusion_matrices`: [confusion_matrices](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/confusion_matrices)
- `predictions`: [predictions](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/predictions)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/loss_plots)
- `latest_loss_pdfs`: [13C_finetune_commutative_cnn_20260701_050638_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/loss_plots/13C_finetune_commutative_cnn_20260701_050638_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/loss_plots/fine_tune/latest.loss-curves.pdf)
- `figures`: [figures](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/figures)
- `figure_pdfs`: [13C_finetune_commutative_cnn_20260701_050638_all_labeled_embedding_umap_controls_hidden.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/figures/13C_finetune_commutative_cnn_20260701_050638_all_labeled_embedding_umap_controls_hidden.pdf), [13C_finetune_commutative_cnn_20260701_050638_all_labeled_embedding_umap_with_controls.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/figures/13C_finetune_commutative_cnn_20260701_050638_all_labeled_embedding_umap_with_controls.pdf), [13C_finetune_commutative_cnn_20260701_050638_holdout_embedding_umap.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/figures/13C_finetune_commutative_cnn_20260701_050638_holdout_embedding_umap.pdf)

### Analysis

Agent-run fine-tune: inspect compound metrics, confusion matrices, AUC, action degradation, and UMAP separation.

### Next Round Proposal

Patch the next 10C or 13C YAML according to whether compound discrimination, action stability, or embedding separation remains the limiting failure mode.

### Config Snapshot

```json
{
  "binary_class_weighting": null,
  "binary_learning_rate": 1e-05,
  "binary_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/loss_plots/binary_hot_start",
  "binary_pretraining_epochs": 2,
  "binary_pretraining_excluded_holdout_count": null,
  "binary_pretraining_label_map": null,
  "binary_pretraining_train_count": null,
  "binary_pretraining_val_count": null,
  "binary_weight_decay": 0.00075,
  "dataset_artifact_path": "/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/ZebraFish/.dataset_cache/moa_GA_An_NM_Ac_AC_In_mA_Ag_c2_mca3_mtc16_t20_z5_y96_x96.pt",
  "experiment_id": "13C_finetune_commutative_cnn_20260701_050638",
  "experiment_output_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C",
  "experiment_run_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638",
  "fine_tune_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/loss_plots/fine_tune",
  "freeze_backbone": false,
  "holdout_fraction": 0.25,
  "hot_start": true,
  "loss_weight_config": {
    "action_weight": 0.65,
    "compound_weight": 0.8,
    "concentration_weight": 0.05,
    "cross_ramp_epochs": 5,
    "cross_warmup_epochs": 5,
    "lambda_align": 0.0,
    "lambda_cross": 1.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 1.0,
  ...
```

<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260629_100031_3ff738:start -->
## Campaign Trial: cnn_pretrain_finetune_20260629_100031_3ff738

| Field | Value |
|---|---|
| Campaign | `cnn_pretrain_finetune` |
| Status | `trial_completed` |
| Objective score | 0.2358 |
| Selected metric | `compound.macro_f1` |
| Objective eligible | `True` |
| Guardrail passed | `True` |

### Metrics

| Metric | Value |
|---|---:|
| `compound.macro_f1` | 0.2358 |
| `compound.roc_auc_ovr_macro` | 0.8120 |
| `compound.balanced_accuracy` | missing |
| `compound.accuracy` | 0.4366 |
| `action.accuracy` | 0.4789 |
| `action.average_precision_macro` | 0.4529 |
| `action.macro_f1` | 0.3722 |
| `action.macro_precision` | 0.4869 |
| `action.macro_recall` | 0.3873 |
| `action.n_samples` | 71.0000 |
| `action.roc_auc_ovr_macro` | 0.7052 |
| `action.weighted_f1` | 0.3986 |
| `action.weighted_precision` | 0.4625 |
| `action.weighted_recall` | 0.4789 |
| `compound.average_precision_macro` | 0.3603 |
| `compound.macro_precision` | 0.2350 |
| `compound.macro_recall` | 0.2794 |
| `compound.n_samples` | 71.0000 |
| `compound.weighted_f1` | 0.3224 |
| `compound.weighted_precision` | 0.2989 |
| `compound.weighted_recall` | 0.4366 |

### Files To Inspect

| File | Link |
|---|---|
| Trial folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738) |
| Summary metrics | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/13C_finetune_commutative_cnn_20260701_050638_summary_metrics.csv) |
| 10C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055) |
| 10C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/loss_plots/10C_pretrain_commutative_cnn_20260629_100055_latest.loss-curves.pdf) |
| 10C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/10C/runs/10C_pretrain_commutative_cnn_20260629_100055/loss_plots/pretraining/latest.loss-curves.pdf) |
| 13C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638) |
| 13C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/figures/13C_finetune_commutative_cnn_20260701_050638_all_labeled_embedding_umap_with_controls.pdf) |
| 13C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/outputs/13C/runs/13C_finetune_commutative_cnn_20260701_050638/figures/13C_finetune_commutative_cnn_20260701_050638_all_labeled_embedding_umap_controls_hidden.pdf) |
| 10C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/configs/10C_10C_pretrain_next.yaml) |
| 13C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260629_100031_3ff738/configs/13C_13C_finetune_next.yaml) |

### Conclusion

This trial completed successfully and passed the action-accuracy guardrail, but downstream compound performance is still limited for the campaign objective. Compound macro_f1 reached 0.2358 with decent compound ROC-AUC (0.812), suggesting ranking signal exists but class decision balance is not translating into strong macro-F1.

### Next

Next, run a finetune-focused reweighting trial that increases compound loss emphasis while slightly reducing auxiliary-task pressure so optimization is more aligned to the objective metric. This should test whether the model can convert the existing separability signal into better per-class compound F1 without sacrificing the action guardrail.
<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260629_100031_3ff738:end -->

## 10C_pretrain_commutative_cnn_20260701_130752

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260701_130752](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752)
- `config`: [10C_pretrain_commutative_cnn_20260701_130752_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/10C_pretrain_commutative_cnn_20260701_130752_config.json)
- `history`: [10C_pretrain_commutative_cnn_20260701_130752_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/10C_pretrain_commutative_cnn_20260701_130752_history.csv)
- `summary_metrics`: [10C_pretrain_commutative_cnn_20260701_130752_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/10C_pretrain_commutative_cnn_20260701_130752_summary_metrics.csv)
- `checkpoint`: [10C_pretrain_commutative_cnn_20260701_130752_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/10C_pretrain_commutative_cnn_20260701_130752_encoder_state.pt)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/loss_plots)
- `latest_loss_pdfs`: [10C_pretrain_commutative_cnn_20260701_130752_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/loss_plots/10C_pretrain_commutative_cnn_20260701_130752_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/loss_plots/pretraining/latest.loss-curves.pdf)
- `latest_encoder_pointer`: [10C_pretrain_commutative_cnn_20260701_130752_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/10C_pretrain_commutative_cnn_20260701_130752_encoder_state.pt)

### Analysis

Agent-run pretraining: inspect live history CSVs and PDFs for validation self-probe trajectory, train/validation gap, and whether the run plateaued before early stopping.

### Next Round Proposal

If validation self-probe plateaus cleanly, use this checkpoint for 13C; otherwise patch the next pretraining YAML based on the observed failure mode.

### Config Snapshot

```json
{
  "loss_weight_config": {
    "action_weight": 1.0,
    "compound_weight": 0.2,
    "concentration_weight": 0.2,
    "cross_ramp_epochs": 0,
    "cross_warmup_epochs": 0,
    "lambda_align": 0.0,
    "lambda_cross": 0.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 0.05,
    "probe_alpha_derivative": 0.25,
    "probe_alpha_frequency": 0.1,
    "probe_alpha_local": 1.0,
    "probe_alpha_region_time": 1.0,
    "probe_mask_probability": 1.0,
    "prototype_alignment_weight": 0.0,
    "prototype_ramp_epochs": 36,
    "prototype_temperature": 0.25,
    "prototype_warmup_epochs": 16,
    "teacher_student_warmup_epochs": 0,
    "water_vs_other_weight": 0.0
  },
  "model_config": {
    "dropout": 0.25,
    "embedding_dim": 64,
    "normalization": "group",
    "num_prototypes": 64,
  ...
```

## 13C_finetune_commutative_cnn_20260702_151104

- kind: `fine_tuning`
- artifact_dir: [13C_finetune_commutative_cnn_20260702_151104](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104)
- `config`: [13C_finetune_commutative_cnn_20260702_151104_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/13C_finetune_commutative_cnn_20260702_151104_config.json)
- `history`: [13C_finetune_commutative_cnn_20260702_151104_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/13C_finetune_commutative_cnn_20260702_151104_history.csv)
- `summary_metrics`: [13C_finetune_commutative_cnn_20260702_151104_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/13C_finetune_commutative_cnn_20260702_151104_summary_metrics.csv)
- `checkpoint`: [13C_finetune_commutative_cnn_20260702_151104_model_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/13C_finetune_commutative_cnn_20260702_151104_model_state.pt)
- `per_class_reports`: [per_class_reports](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/per_class_reports)
- `confusion_matrices`: [confusion_matrices](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/confusion_matrices)
- `predictions`: [predictions](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/predictions)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/loss_plots)
- `latest_loss_pdfs`: [13C_finetune_commutative_cnn_20260702_151104_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/loss_plots/13C_finetune_commutative_cnn_20260702_151104_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/loss_plots/fine_tune/latest.loss-curves.pdf)
- `figures`: [figures](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/figures)
- `figure_pdfs`: [13C_finetune_commutative_cnn_20260702_151104_all_labeled_embedding_umap_controls_hidden.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/figures/13C_finetune_commutative_cnn_20260702_151104_all_labeled_embedding_umap_controls_hidden.pdf), [13C_finetune_commutative_cnn_20260702_151104_all_labeled_embedding_umap_with_controls.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/figures/13C_finetune_commutative_cnn_20260702_151104_all_labeled_embedding_umap_with_controls.pdf), [13C_finetune_commutative_cnn_20260702_151104_holdout_embedding_umap.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/figures/13C_finetune_commutative_cnn_20260702_151104_holdout_embedding_umap.pdf)

### Analysis

Agent-run fine-tune: inspect compound metrics, confusion matrices, AUC, action degradation, and UMAP separation.

### Next Round Proposal

Patch the next 10C or 13C YAML according to whether compound discrimination, action stability, or embedding separation remains the limiting failure mode.

### Config Snapshot

```json
{
  "binary_class_weighting": null,
  "binary_learning_rate": 1e-05,
  "binary_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/loss_plots/binary_hot_start",
  "binary_pretraining_epochs": 2,
  "binary_pretraining_excluded_holdout_count": null,
  "binary_pretraining_label_map": null,
  "binary_pretraining_train_count": null,
  "binary_pretraining_val_count": null,
  "binary_weight_decay": 0.00075,
  "dataset_artifact_path": "/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/ZebraFish/.dataset_cache/moa_GA_An_NM_Ac_AC_In_mA_Ag_c2_mca3_mtc16_t20_z5_y96_x96.pt",
  "experiment_id": "13C_finetune_commutative_cnn_20260702_151104",
  "experiment_output_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C",
  "experiment_run_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104",
  "fine_tune_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/loss_plots/fine_tune",
  "freeze_backbone": false,
  "holdout_fraction": 0.25,
  "hot_start": true,
  "loss_weight_config": {
    "action_weight": 0.5,
    "compound_weight": 1.0,
    "concentration_weight": 0.03,
    "cross_ramp_epochs": 5,
    "cross_warmup_epochs": 5,
    "lambda_align": 0.0,
    "lambda_cross": 1.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 1.0,
  ...
```

<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260701_130749_69a864:start -->
## Campaign Trial: cnn_pretrain_finetune_20260701_130749_69a864

| Field | Value |
|---|---|
| Campaign | `cnn_pretrain_finetune` |
| Status | `trial_completed` |
| Objective score | 0.2332 |
| Selected metric | `compound.macro_f1` |
| Objective eligible | `True` |
| Guardrail passed | `True` |

### Metrics

| Metric | Value |
|---|---:|
| `compound.macro_f1` | 0.2332 |
| `compound.roc_auc_ovr_macro` | 0.7633 |
| `compound.balanced_accuracy` | missing |
| `compound.accuracy` | 0.3944 |
| `action.accuracy` | 0.4789 |
| `action.average_precision_macro` | 0.4872 |
| `action.macro_f1` | 0.3678 |
| `action.macro_precision` | 0.4453 |
| `action.macro_recall` | 0.3907 |
| `action.n_samples` | 71.0000 |
| `action.roc_auc_ovr_macro` | 0.7294 |
| `action.weighted_f1` | 0.3942 |
| `action.weighted_precision` | 0.4282 |
| `action.weighted_recall` | 0.4789 |
| `compound.average_precision_macro` | 0.3658 |
| `compound.macro_precision` | 0.2237 |
| `compound.macro_recall` | 0.2765 |
| `compound.n_samples` | 71.0000 |
| `compound.weighted_f1` | 0.3058 |
| `compound.weighted_precision` | 0.2766 |
| `compound.weighted_recall` | 0.3944 |

### Files To Inspect

| File | Link |
|---|---|
| Trial folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864) |
| Summary metrics | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/13C_finetune_commutative_cnn_20260702_151104_summary_metrics.csv) |
| 10C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752) |
| 10C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/loss_plots/10C_pretrain_commutative_cnn_20260701_130752_latest.loss-curves.pdf) |
| 10C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/10C/runs/10C_pretrain_commutative_cnn_20260701_130752/loss_plots/pretraining/latest.loss-curves.pdf) |
| 13C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104) |
| 13C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/figures/13C_finetune_commutative_cnn_20260702_151104_all_labeled_embedding_umap_with_controls.pdf) |
| 13C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/outputs/13C/runs/13C_finetune_commutative_cnn_20260702_151104/figures/13C_finetune_commutative_cnn_20260702_151104_all_labeled_embedding_umap_controls_hidden.pdf) |
| 10C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/configs/10C_10C_pretrain_next.yaml) |
| 13C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260701_130749_69a864/configs/13C_13C_finetune_next.yaml) |

### Conclusion

This trial completed successfully and met the action accuracy guardrail, but downstream compound performance is still modest: compound macro_f1 is 0.233 with accuracy 0.394. The relatively stronger compound ROC-AUC versus macro_f1 suggests the model has some class-separation signal but is not translating it into strong balanced class decisions at inference time.

### Next

Next, run a follow-up trial that rebalances finetune loss toward the compound objective by reducing auxiliary-task influence. This should help the shared representation and gradients prioritize the target labels directly, which is the most attributable single change given current evidence.
<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260701_130749_69a864:end -->

## 10C_pretrain_commutative_cnn_20260703_031245

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260703_031245](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245)
- `config`: [10C_pretrain_commutative_cnn_20260703_031245_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/10C_pretrain_commutative_cnn_20260703_031245_config.json)
- `history`: [10C_pretrain_commutative_cnn_20260703_031245_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/10C_pretrain_commutative_cnn_20260703_031245_history.csv)
- `summary_metrics`: [10C_pretrain_commutative_cnn_20260703_031245_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/10C_pretrain_commutative_cnn_20260703_031245_summary_metrics.csv)
- `checkpoint`: [10C_pretrain_commutative_cnn_20260703_031245_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/10C_pretrain_commutative_cnn_20260703_031245_encoder_state.pt)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/loss_plots)
- `latest_loss_pdfs`: [10C_pretrain_commutative_cnn_20260703_031245_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/loss_plots/10C_pretrain_commutative_cnn_20260703_031245_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/loss_plots/pretraining/latest.loss-curves.pdf)
- `latest_encoder_pointer`: [10C_pretrain_commutative_cnn_20260703_031245_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/10C_pretrain_commutative_cnn_20260703_031245_encoder_state.pt)

### Analysis

Agent-run pretraining: inspect live history CSVs and PDFs for validation self-probe trajectory, train/validation gap, and whether the run plateaued before early stopping.

### Next Round Proposal

If validation self-probe plateaus cleanly, use this checkpoint for 13C; otherwise patch the next pretraining YAML based on the observed failure mode.

### Config Snapshot

```json
{
  "loss_weight_config": {
    "action_weight": 1.0,
    "compound_weight": 0.2,
    "concentration_weight": 0.2,
    "cross_ramp_epochs": 0,
    "cross_warmup_epochs": 0,
    "lambda_align": 0.0,
    "lambda_cross": 0.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 0.05,
    "probe_alpha_derivative": 0.25,
    "probe_alpha_frequency": 0.1,
    "probe_alpha_local": 1.0,
    "probe_alpha_region_time": 1.0,
    "probe_mask_probability": 1.0,
    "prototype_alignment_weight": 0.0,
    "prototype_ramp_epochs": 36,
    "prototype_temperature": 0.25,
    "prototype_warmup_epochs": 16,
    "teacher_student_warmup_epochs": 0,
    "water_vs_other_weight": 0.0
  },
  "model_config": {
    "dropout": 0.25,
    "embedding_dim": 64,
    "normalization": "group",
    "num_prototypes": 64,
  ...
```

## 13C_finetune_commutative_cnn_20260704_051554

- kind: `fine_tuning`
- artifact_dir: [13C_finetune_commutative_cnn_20260704_051554](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554)
- `config`: [13C_finetune_commutative_cnn_20260704_051554_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/13C_finetune_commutative_cnn_20260704_051554_config.json)
- `history`: [13C_finetune_commutative_cnn_20260704_051554_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/13C_finetune_commutative_cnn_20260704_051554_history.csv)
- `summary_metrics`: [13C_finetune_commutative_cnn_20260704_051554_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/13C_finetune_commutative_cnn_20260704_051554_summary_metrics.csv)
- `checkpoint`: [13C_finetune_commutative_cnn_20260704_051554_model_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/13C_finetune_commutative_cnn_20260704_051554_model_state.pt)
- `per_class_reports`: [per_class_reports](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/per_class_reports)
- `confusion_matrices`: [confusion_matrices](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/confusion_matrices)
- `predictions`: [predictions](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/predictions)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/loss_plots)
- `latest_loss_pdfs`: [13C_finetune_commutative_cnn_20260704_051554_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/loss_plots/13C_finetune_commutative_cnn_20260704_051554_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/loss_plots/fine_tune/latest.loss-curves.pdf)
- `figures`: [figures](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/figures)
- `figure_pdfs`: [13C_finetune_commutative_cnn_20260704_051554_all_labeled_embedding_umap_controls_hidden.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/figures/13C_finetune_commutative_cnn_20260704_051554_all_labeled_embedding_umap_controls_hidden.pdf), [13C_finetune_commutative_cnn_20260704_051554_all_labeled_embedding_umap_with_controls.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/figures/13C_finetune_commutative_cnn_20260704_051554_all_labeled_embedding_umap_with_controls.pdf), [13C_finetune_commutative_cnn_20260704_051554_holdout_embedding_umap.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/figures/13C_finetune_commutative_cnn_20260704_051554_holdout_embedding_umap.pdf)

### Analysis

Agent-run fine-tune: inspect compound metrics, confusion matrices, AUC, action degradation, and UMAP separation.

### Next Round Proposal

Patch the next 10C or 13C YAML according to whether compound discrimination, action stability, or embedding separation remains the limiting failure mode.

### Config Snapshot

```json
{
  "binary_class_weighting": null,
  "binary_learning_rate": 1e-05,
  "binary_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/loss_plots/binary_hot_start",
  "binary_pretraining_epochs": 2,
  "binary_pretraining_excluded_holdout_count": null,
  "binary_pretraining_label_map": null,
  "binary_pretraining_train_count": null,
  "binary_pretraining_val_count": null,
  "binary_weight_decay": 0.00075,
  "dataset_artifact_path": "/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/ZebraFish/.dataset_cache/moa_GA_An_NM_Ac_AC_In_mA_Ag_c2_mca3_mtc16_t20_z5_y96_x96.pt",
  "experiment_id": "13C_finetune_commutative_cnn_20260704_051554",
  "experiment_output_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C",
  "experiment_run_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554",
  "fine_tune_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/loss_plots/fine_tune",
  "freeze_backbone": false,
  "holdout_fraction": 0.25,
  "hot_start": true,
  "loss_weight_config": {
    "action_weight": 0.35,
    "compound_weight": 1.3,
    "concentration_weight": 0.02,
    "cross_ramp_epochs": 5,
    "cross_warmup_epochs": 5,
    "lambda_align": 0.0,
    "lambda_cross": 1.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 1.0,
  ...
```

<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260703_031242_495f28:start -->
## Campaign Trial: cnn_pretrain_finetune_20260703_031242_495f28

| Field | Value |
|---|---|
| Campaign | `cnn_pretrain_finetune` |
| Status | `trial_completed` |
| Objective score | 0.1297 |
| Selected metric | `compound.macro_f1` |
| Objective eligible | `True` |
| Guardrail passed | `True` |

### Metrics

| Metric | Value |
|---|---:|
| `compound.macro_f1` | 0.1297 |
| `compound.roc_auc_ovr_macro` | 0.7246 |
| `compound.balanced_accuracy` | missing |
| `compound.accuracy` | 0.1690 |
| `action.accuracy` | 0.4085 |
| `action.average_precision_macro` | 0.4011 |
| `action.macro_f1` | 0.2355 |
| `action.macro_precision` | 0.3689 |
| `action.macro_recall` | 0.3007 |
| `action.n_samples` | 71.0000 |
| `action.roc_auc_ovr_macro` | 0.7005 |
| `action.weighted_f1` | 0.2859 |
| `action.weighted_precision` | 0.3703 |
| `action.weighted_recall` | 0.4085 |
| `compound.average_precision_macro` | 0.2674 |
| `compound.macro_precision` | 0.1269 |
| `compound.macro_recall` | 0.1887 |
| `compound.n_samples` | 71.0000 |
| `compound.weighted_f1` | 0.1509 |
| `compound.weighted_precision` | 0.1753 |
| `compound.weighted_recall` | 0.1690 |

### Files To Inspect

| File | Link |
|---|---|
| Trial folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28) |
| Summary metrics | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/13C_finetune_commutative_cnn_20260704_051554_summary_metrics.csv) |
| 10C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245) |
| 10C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/loss_plots/10C_pretrain_commutative_cnn_20260703_031245_latest.loss-curves.pdf) |
| 10C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/10C/runs/10C_pretrain_commutative_cnn_20260703_031245/loss_plots/pretraining/latest.loss-curves.pdf) |
| 13C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554) |
| 13C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/figures/13C_finetune_commutative_cnn_20260704_051554_all_labeled_embedding_umap_with_controls.pdf) |
| 13C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/outputs/13C/runs/13C_finetune_commutative_cnn_20260704_051554/figures/13C_finetune_commutative_cnn_20260704_051554_all_labeled_embedding_umap_controls_hidden.pdf) |
| 10C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/configs/10C_10C_pretrain_next.yaml) |
| 13C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260703_031242_495f28/configs/13C_13C_finetune_next.yaml) |

### Conclusion

This trial completed successfully and met the action accuracy guardrail, but downstream compound performance remains weak on the primary objective (macro_f1). The pattern of low compound macro_f1 with comparatively stronger compound ROC-AUC suggests the model has partial class-separation signal but is not converting it into good final class decisions.

### Next

Next, run another trial that increases emphasis on the compound objective by rebalancing finetune loss weights toward compound and away from auxiliary heads. This is a targeted change to improve the primary metric without broad architectural churn, and it is justified because current results indicate optimization focus is likely diluted across tasks.
<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260703_031242_495f28:end -->

## 10C_pretrain_commutative_cnn_20260704_141703

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260704_141703](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703)
- `config`: [10C_pretrain_commutative_cnn_20260704_141703_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/10C_pretrain_commutative_cnn_20260704_141703_config.json)
- `history`: [10C_pretrain_commutative_cnn_20260704_141703_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/10C_pretrain_commutative_cnn_20260704_141703_history.csv)
- `summary_metrics`: [10C_pretrain_commutative_cnn_20260704_141703_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/10C_pretrain_commutative_cnn_20260704_141703_summary_metrics.csv)
- `checkpoint`: [10C_pretrain_commutative_cnn_20260704_141703_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/10C_pretrain_commutative_cnn_20260704_141703_encoder_state.pt)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/loss_plots)
- `latest_loss_pdfs`: [10C_pretrain_commutative_cnn_20260704_141703_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/loss_plots/10C_pretrain_commutative_cnn_20260704_141703_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/loss_plots/pretraining/latest.loss-curves.pdf)
- `latest_encoder_pointer`: [10C_pretrain_commutative_cnn_20260704_141703_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/10C_pretrain_commutative_cnn_20260704_141703_encoder_state.pt)

### Analysis

Agent-run pretraining: inspect live history CSVs and PDFs for validation self-probe trajectory, train/validation gap, and whether the run plateaued before early stopping.

### Next Round Proposal

If validation self-probe plateaus cleanly, use this checkpoint for 13C; otherwise patch the next pretraining YAML based on the observed failure mode.

### Config Snapshot

```json
{
  "loss_weight_config": {
    "action_weight": 1.0,
    "compound_weight": 0.2,
    "concentration_weight": 0.2,
    "cross_ramp_epochs": 0,
    "cross_warmup_epochs": 0,
    "lambda_align": 0.0,
    "lambda_cross": 0.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 0.05,
    "probe_alpha_derivative": 0.25,
    "probe_alpha_frequency": 0.1,
    "probe_alpha_local": 1.0,
    "probe_alpha_region_time": 1.0,
    "probe_mask_probability": 1.0,
    "prototype_alignment_weight": 0.0,
    "prototype_ramp_epochs": 36,
    "prototype_temperature": 0.25,
    "prototype_warmup_epochs": 16,
    "teacher_student_warmup_epochs": 0,
    "water_vs_other_weight": 0.0
  },
  "model_config": {
    "dropout": 0.25,
    "embedding_dim": 64,
    "normalization": "group",
    "num_prototypes": 64,
  ...
```

## 13C_finetune_commutative_cnn_20260705_131931

- kind: `fine_tuning`
- artifact_dir: [13C_finetune_commutative_cnn_20260705_131931](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931)
- `config`: [13C_finetune_commutative_cnn_20260705_131931_config.json](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/13C_finetune_commutative_cnn_20260705_131931_config.json)
- `history`: [13C_finetune_commutative_cnn_20260705_131931_history.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/13C_finetune_commutative_cnn_20260705_131931_history.csv)
- `summary_metrics`: [13C_finetune_commutative_cnn_20260705_131931_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/13C_finetune_commutative_cnn_20260705_131931_summary_metrics.csv)
- `checkpoint`: [13C_finetune_commutative_cnn_20260705_131931_model_state.pt](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/13C_finetune_commutative_cnn_20260705_131931_model_state.pt)
- `per_class_reports`: [per_class_reports](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/per_class_reports)
- `confusion_matrices`: [confusion_matrices](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/confusion_matrices)
- `predictions`: [predictions](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/predictions)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/loss_plots)
- `latest_loss_pdfs`: [13C_finetune_commutative_cnn_20260705_131931_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/loss_plots/13C_finetune_commutative_cnn_20260705_131931_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/loss_plots/fine_tune/latest.loss-curves.pdf)
- `figures`: [figures](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/figures)
- `figure_pdfs`: [13C_finetune_commutative_cnn_20260705_131931_all_labeled_embedding_umap_controls_hidden.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/figures/13C_finetune_commutative_cnn_20260705_131931_all_labeled_embedding_umap_controls_hidden.pdf), [13C_finetune_commutative_cnn_20260705_131931_all_labeled_embedding_umap_with_controls.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/figures/13C_finetune_commutative_cnn_20260705_131931_all_labeled_embedding_umap_with_controls.pdf), [13C_finetune_commutative_cnn_20260705_131931_holdout_embedding_umap.pdf](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/figures/13C_finetune_commutative_cnn_20260705_131931_holdout_embedding_umap.pdf)

### Analysis

Agent-run fine-tune: inspect compound metrics, confusion matrices, AUC, action degradation, and UMAP separation.

### Next Round Proposal

Patch the next 10C or 13C YAML according to whether compound discrimination, action stability, or embedding separation remains the limiting failure mode.

### Config Snapshot

```json
{
  "binary_class_weighting": null,
  "binary_learning_rate": 1e-05,
  "binary_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/loss_plots/binary_hot_start",
  "binary_pretraining_epochs": 2,
  "binary_pretraining_excluded_holdout_count": null,
  "binary_pretraining_label_map": null,
  "binary_pretraining_train_count": null,
  "binary_pretraining_val_count": null,
  "binary_weight_decay": 0.00075,
  "dataset_artifact_path": "/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/ZebraFish/.dataset_cache/moa_GA_An_NM_Ac_AC_In_mA_Ag_c2_mca3_mtc16_t20_z5_y96_x96.pt",
  "experiment_id": "13C_finetune_commutative_cnn_20260705_131931",
  "experiment_output_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C",
  "experiment_run_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931",
  "fine_tune_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/loss_plots/fine_tune",
  "freeze_backbone": false,
  "holdout_fraction": 0.25,
  "hot_start": true,
  "loss_weight_config": {
    "action_weight": 0.25,
    "compound_weight": 1.6,
    "concentration_weight": 0.01,
    "cross_ramp_epochs": 5,
    "cross_warmup_epochs": 5,
    "lambda_align": 0.0,
    "lambda_cross": 1.0,
    "latent_alignment_weight": 0.0,
    "probe_alpha_correlation": 1.0,
  ...
```

<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260704_141700_197670:start -->
## Campaign Trial: cnn_pretrain_finetune_20260704_141700_197670

| Field | Value |
|---|---|
| Campaign | `cnn_pretrain_finetune` |
| Status | `trial_completed` |
| Objective score | 0.2528 |
| Selected metric | `compound.macro_f1` |
| Objective eligible | `True` |
| Guardrail passed | `True` |

### Metrics

| Metric | Value |
|---|---:|
| `compound.macro_f1` | 0.2528 |
| `compound.roc_auc_ovr_macro` | 0.8124 |
| `compound.balanced_accuracy` | missing |
| `compound.accuracy` | 0.2535 |
| `action.accuracy` | 0.3944 |
| `action.average_precision_macro` | 0.4456 |
| `action.macro_f1` | 0.3915 |
| `action.macro_precision` | 0.4088 |
| `action.macro_recall` | 0.3893 |
| `action.n_samples` | 71.0000 |
| `action.roc_auc_ovr_macro` | 0.6725 |
| `action.weighted_f1` | 0.3867 |
| `action.weighted_precision` | 0.3938 |
| `action.weighted_recall` | 0.3944 |
| `compound.average_precision_macro` | 0.4451 |
| `compound.macro_precision` | 0.2201 |
| `compound.macro_recall` | 0.3846 |
| `compound.n_samples` | 71.0000 |
| `compound.weighted_f1` | 0.1803 |
| `compound.weighted_precision` | 0.1670 |
| `compound.weighted_recall` | 0.2535 |

### Files To Inspect

| File | Link |
|---|---|
| Trial folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670) |
| Summary metrics | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/13C_finetune_commutative_cnn_20260705_131931_summary_metrics.csv) |
| 10C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703) |
| 10C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/loss_plots/10C_pretrain_commutative_cnn_20260704_141703_latest.loss-curves.pdf) |
| 10C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/10C/runs/10C_pretrain_commutative_cnn_20260704_141703/loss_plots/pretraining/latest.loss-curves.pdf) |
| 13C run folder | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931) |
| 13C loss PDF 1 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/figures/13C_finetune_commutative_cnn_20260705_131931_all_labeled_embedding_umap_with_controls.pdf) |
| 13C loss PDF 2 | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/outputs/13C/runs/13C_finetune_commutative_cnn_20260705_131931/figures/13C_finetune_commutative_cnn_20260705_131931_all_labeled_embedding_umap_controls_hidden.pdf) |
| 10C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/configs/10C_10C_pretrain_next.yaml) |
| 13C config | [open](artifacts/campaigns/cnn_pretrain_finetune/cnn_pretrain_finetune_20260704_141700_197670/configs/13C_13C_finetune_next.yaml) |

### Conclusion

This run completed successfully and met the action accuracy guardrail, but downstream compound performance is still weak on the primary objective. Compound macro-F1 is only 0.253 with similarly low compound accuracy, so the chain produced usable evidence but not a competitive result yet.

### Next

Next, we should run a follow-up finetune that further concentrates learning on the compound head by increasing compound loss emphasis and reducing auxiliary-task influence. The current pattern of relatively strong compound ROC-AUC but low macro-F1 suggests ranking signal exists but class decision quality is under-optimized, so this targeted reweighting is a direct and attributable next test.
<!-- campaign:cnn_pretrain_finetune:cnn_pretrain_finetune_20260704_141700_197670:end -->

# Experiments Logbook

This file records pretraining and fine-tuning runs by experiment id. Each entry links the timestamped artifacts, summarizes what happened, and states the proposed next round.

## 10C_pretrain_commutative_cnn_legacy_20260619_163630

- kind: `pretraining`
- artifact_dir: [pretrained_commutative_cnn](artifacts/pretrained_commutative_cnn)
- config: [config.yaml](artifacts/pretrained_commutative_cnn/config.yaml)
- checkpoint: [encoder_state_v9.pt](artifacts/pretrained_commutative_cnn/encoder_state_v9.pt)
- loss_pdfs: [loss_plots](artifacts/pretrained_commutative_cnn/loss_plots)
- latest_loss_pdfs: [latest.loss-curves.pdf](artifacts/pretrained_commutative_cnn/loss_plots/latest.loss-curves.pdf)

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
- Trial folder: [artifacts/campaigns/cnn_pretrain_finetune/trials/cnn_pretrain_finetune_20260625_115021_08a733](artifacts/campaigns/cnn_pretrain_finetune/trials/cnn_pretrain_finetune_20260625_115021_08a733)
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

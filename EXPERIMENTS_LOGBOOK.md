# Experiments Logbook

This file records pretraining and fine-tuning runs by experiment id. Each entry links the timestamped artifacts, summarizes what happened, and states the proposed next round.

## 10C_pretrain_commutative_cnn_20260731_133224

- kind: `pretraining`
- artifact_dir: [10C_pretrain_commutative_cnn_20260731_133224](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224)
- `config`: [10C_pretrain_commutative_cnn_20260731_133224_config.json](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/10C_pretrain_commutative_cnn_20260731_133224_config.json)
- `history`: [10C_pretrain_commutative_cnn_20260731_133224_history.csv](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/10C_pretrain_commutative_cnn_20260731_133224_history.csv)
- `summary_metrics`: [10C_pretrain_commutative_cnn_20260731_133224_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/10C_pretrain_commutative_cnn_20260731_133224_summary_metrics.csv)
- `checkpoint`: [10C_pretrain_commutative_cnn_20260731_133224_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/10C_pretrain_commutative_cnn_20260731_133224_encoder_state.pt)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/loss_plots)
- `latest_loss_pdfs`: [10C_pretrain_commutative_cnn_20260731_133224_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/loss_plots/10C_pretrain_commutative_cnn_20260731_133224_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/loss_plots/pretraining/latest.loss-curves.pdf)
- `latest_encoder_pointer`: [10C_pretrain_commutative_cnn_20260731_133224_encoder_state.pt](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/10C/runs/10C_pretrain_commutative_cnn_20260731_133224/10C_pretrain_commutative_cnn_20260731_133224_encoder_state.pt)

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

## 13C_finetune_commutative_cnn_20260802_112726

- kind: `fine_tuning`
- artifact_dir: [13C_finetune_commutative_cnn_20260802_112726](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726)
- `config`: [13C_finetune_commutative_cnn_20260802_112726_config.json](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/13C_finetune_commutative_cnn_20260802_112726_config.json)
- `summary_metrics`: [13C_finetune_commutative_cnn_20260802_112726_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/13C_finetune_commutative_cnn_20260802_112726_summary_metrics.csv)
- `checkpoint`: [13C_finetune_commutative_cnn_20260802_112726_model_state.pt](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/13C_finetune_commutative_cnn_20260802_112726_model_state.pt)
- `per_class_reports`: [per_class_reports](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/per_class_reports)
- `confusion_matrices`: [confusion_matrices](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/confusion_matrices)
- `predictions`: [predictions](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/predictions)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/loss_plots)
- `latest_loss_pdfs`: [13C_finetune_commutative_cnn_20260802_112726_fine_tune_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/loss_plots/13C_finetune_commutative_cnn_20260802_112726_fine_tune_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/loss_plots/fine_tune/latest.loss-curves.pdf)
- `figures`: [figures](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/figures)
- `figure_pdfs`: [13C_finetune_commutative_cnn_20260802_112726_all_labeled_embedding_umap_controls_hidden.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/figures/13C_finetune_commutative_cnn_20260802_112726_all_labeled_embedding_umap_controls_hidden.pdf), [13C_finetune_commutative_cnn_20260802_112726_all_labeled_embedding_umap_with_controls.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/figures/13C_finetune_commutative_cnn_20260802_112726_all_labeled_embedding_umap_with_controls.pdf), [13C_finetune_commutative_cnn_20260802_112726_holdout_embedding_umap.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/figures/13C_finetune_commutative_cnn_20260802_112726_holdout_embedding_umap.pdf)

### Analysis

Agent-run fine-tune: inspect compound metrics, confusion matrices, AUC, action degradation, and UMAP separation.

### Next Round Proposal

Patch the next 10C or 13C YAML according to whether compound discrimination, action stability, or embedding separation remains the limiting failure mode.

### Config Snapshot

```json
{
  "binary_class_weighting": null,
  "binary_learning_rate": 1e-05,
  "binary_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/loss_plots/binary_hot_start",
  "binary_pretraining_epochs": 2,
  "binary_pretraining_excluded_holdout_count": null,
  "binary_pretraining_label_map": null,
  "binary_pretraining_train_count": null,
  "binary_pretraining_val_count": null,
  "binary_weight_decay": 0.00075,
  "dataset_artifact_path": "/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/ZebraFish/.dataset_cache/moa_GA_An_NM_Ac_AC_In_mA_Ag_c2_mca3_mtc16_t20_z5_y96_x96.pt",
  "domain_guidance": {
    "contract_hash": "aad78e5eb5cfee2cb1847257f2a8f4d47ea319c64d879b000b4e6d08fcd7ec96",
    "objective_eligibility": "hard_guardrail_failed",
    "report_hash": "c9ecac9e22d55b917c0434bfe999857a53c56e15fd5b023fefff4fb151680f2c",
    "report_path": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/domain_guidance/domain_evaluation.json"
  },
  "domain_guidance_calibration": "state/campaigns/cnn_pretrain_finetune_protocol_v2/domain_calibration.json",
  "domain_guidance_contract": "configs/domain_guidance/cnn_action_domain_v1.yaml",
  "domain_guidance_live_enabled": true,
  "evaluation_protocol": "three_seed_replicate_lockbox_v1",
  "experiment_id": "13C_finetune_commutative_cnn_20260802_112726",
  "experiment_output_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C",
  "experiment_run_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726",
  "fine_tune_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112726/loss_plots/fine_tune",
  "freeze_backbone": false,
  "holdout_fraction": 0.25,
  "hot_start": true,
  ...
```

## 13C_finetune_commutative_cnn_20260802_112927

- kind: `fine_tuning`
- artifact_dir: [13C_finetune_commutative_cnn_20260802_112927](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927)
- `config`: [13C_finetune_commutative_cnn_20260802_112927_config.json](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/13C_finetune_commutative_cnn_20260802_112927_config.json)
- `summary_metrics`: [13C_finetune_commutative_cnn_20260802_112927_summary_metrics.csv](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/13C_finetune_commutative_cnn_20260802_112927_summary_metrics.csv)
- `checkpoint`: [13C_finetune_commutative_cnn_20260802_112927_model_state.pt](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/13C_finetune_commutative_cnn_20260802_112927_model_state.pt)
- `per_class_reports`: [per_class_reports](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/per_class_reports)
- `confusion_matrices`: [confusion_matrices](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/confusion_matrices)
- `predictions`: [predictions](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/predictions)
- `loss_pdfs`: [loss_plots](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/loss_plots)
- `latest_loss_pdfs`: [13C_finetune_commutative_cnn_20260802_112927_fine_tune_latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/loss_plots/13C_finetune_commutative_cnn_20260802_112927_fine_tune_latest.loss-curves.pdf), [latest.loss-curves.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/loss_plots/fine_tune/latest.loss-curves.pdf)
- `figures`: [figures](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/figures)
- `figure_pdfs`: [13C_finetune_commutative_cnn_20260802_112927_all_labeled_embedding_umap_controls_hidden.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/figures/13C_finetune_commutative_cnn_20260802_112927_all_labeled_embedding_umap_controls_hidden.pdf), [13C_finetune_commutative_cnn_20260802_112927_all_labeled_embedding_umap_with_controls.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/figures/13C_finetune_commutative_cnn_20260802_112927_all_labeled_embedding_umap_with_controls.pdf), [13C_finetune_commutative_cnn_20260802_112927_holdout_embedding_umap.pdf](artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/figures/13C_finetune_commutative_cnn_20260802_112927_holdout_embedding_umap.pdf)

### Analysis

Agent-run fine-tune: inspect compound metrics, confusion matrices, AUC, action degradation, and UMAP separation.

### Next Round Proposal

Patch the next 10C or 13C YAML according to whether compound discrimination, action stability, or embedding separation remains the limiting failure mode.

### Config Snapshot

```json
{
  "binary_class_weighting": null,
  "binary_learning_rate": 1e-05,
  "binary_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/loss_plots/binary_hot_start",
  "binary_pretraining_epochs": 2,
  "binary_pretraining_excluded_holdout_count": null,
  "binary_pretraining_label_map": null,
  "binary_pretraining_train_count": null,
  "binary_pretraining_val_count": null,
  "binary_weight_decay": 0.00075,
  "dataset_artifact_path": "/run/media/fabrizio/06bb7271-2161-43a4-91f1-98f9b67e9ab2/home/fabrizio/code/ZebraFish/.dataset_cache/moa_GA_An_NM_Ac_AC_In_mA_Ag_c2_mca3_mtc16_t20_z5_y96_x96.pt",
  "domain_guidance": {
    "contract_hash": "aad78e5eb5cfee2cb1847257f2a8f4d47ea319c64d879b000b4e6d08fcd7ec96",
    "objective_eligibility": "hard_guardrail_failed",
    "report_hash": "a2b0c4adf25b5ef66b10f0f4d84e981e9f5a1b3ce7b33df9f5700c5ea45dee82",
    "report_path": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/domain_guidance/domain_evaluation.json"
  },
  "domain_guidance_calibration": "state/campaigns/cnn_pretrain_finetune_protocol_v2/domain_calibration.json",
  "domain_guidance_contract": "configs/domain_guidance/cnn_action_domain_v1.yaml",
  "domain_guidance_live_enabled": true,
  "evaluation_protocol": "three_seed_replicate_lockbox_v1",
  "experiment_id": "13C_finetune_commutative_cnn_20260802_112927",
  "experiment_output_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C",
  "experiment_run_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927",
  "fine_tune_loss_plot_dir": "artifacts/campaigns/cnn_pretrain_finetune_protocol_v2/cnn_pretrain_finetune_protocol_v2_20260731_133221_4918be/outputs/13C/runs/13C_finetune_commutative_cnn_20260802_112927/loss_plots/fine_tune",
  "freeze_backbone": false,
  "holdout_fraction": 0.25,
  "hot_start": true,
  ...
```

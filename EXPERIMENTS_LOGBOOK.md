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

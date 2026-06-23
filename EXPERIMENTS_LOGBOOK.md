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

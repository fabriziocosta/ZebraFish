from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.commutative_transformer_pretraining_config import (
    CommutativeTransformerPretrainingConfig,
    load_commutative_transformer_pretraining_config,
    write_commutative_transformer_pretraining_config,
)
from src.dataset_config import load_current_dataset_artifact_path
from src.experiment_runner_shared import merge_dicts, read_yaml_mapping, to_yamlable, update_agent_run_status, write_yaml_mapping
from src.models.configs import CommutativeTransformerConfig, LossWeightConfig, OptimizationConfig
from src.models.estimators import CommutativeTransformerClassifier
from src.tensor_utils import build_tensor_embedding_2d, load_labeled_tensor_dataset, load_unlabeled_tensor_dataset, plot_tensor_embedding_2d
from src.training.checkpointing import TrainingSuspended, install_training_signal_handlers
from src.training.workflow import (
    ExperimentRun,
    MultitaskEvaluationResult,
    build_reports_excluding_control,
    create_experiment_run,
    evaluate_multitask_estimator,
    fit_estimator_on_experiment,
    persist_experiment_artifacts,
    persist_pretraining_artifacts,
    prepare_multitask_experiment_data,
)


DEFAULT_12T_CONFIG_PATH = Path("configs/experiments/12T_pretrain_next.yaml")
DEFAULT_15T_CONFIG_PATH = Path("configs/experiments/15T_finetune_next.yaml")


def _experiment_run_from_existing_dir(run_dir: Path, experiment_id: str) -> ExperimentRun:
    (run_dir / "loss_plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return ExperimentRun(
        experiment_id=experiment_id,
        run_dir=str(run_dir),
        loss_plot_dir=str(run_dir / "loss_plots"),
        figure_dir=str(run_dir / "figures"),
    )


def _experiment_run_from_config_path(config_path: Path, experiment_prefix: str) -> ExperimentRun | None:
    if not config_path.exists():
        return None
    run_dir = config_path.parent
    if run_dir.parent.name != "runs":
        return None
    suffix = "_config.yaml"
    if not config_path.name.endswith(suffix):
        return None
    experiment_id = config_path.name.removesuffix(suffix)
    if not experiment_id.startswith(experiment_prefix):
        return None
    return _experiment_run_from_existing_dir(run_dir, experiment_id)


def default_12t_pretraining_config() -> CommutativeTransformerPretrainingConfig:
    return CommutativeTransformerPretrainingConfig(
        unlabeled_dataset_path=Path(".dataset_cache/unlabeled_active_high_mid_low_t20_z5_y96_x96_chunks"),
        pretrained_encoder_path=Path("artifacts/pretrained_commutative_transformer/runs/PENDING/PENDING_encoder_state.pt"),
        model_config=CommutativeTransformerConfig(
            spatial_patch_size_st=(1, 32, 32),
            spatial_patch_size_ts=(1, 32, 32),
            temporal_patch_size_ts=2,
            embed_dim=32,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.3,
            attention_dropout=0.1,
            st_spatial_depth=1,
            st_temporal_depth=1,
            ts_temporal_depth=1,
            ts_spatial_depth=1,
            embedding_dim=16,
            probe_region_grid=(1, 2, 2),
            probe_time_bins=8,
            probe_frequency_bins=4,
        ),
        optimization_config=OptimizationConfig(
            batch_size=8,
            epochs=90,
            learning_rate=7.5e-5,
            weight_decay=3e-3,
            early_stopping_patience=14,
            early_stopping_min_delta=0.0,
            early_stopping_start_epoch=24,
            early_stopping_monitor="loss",
            early_stopping_smoothing="median",
            early_stopping_smoothing_window=5,
            training_plot_dir=None,
            training_plot_every_n_epochs=1,
            training_plot_smoothing_window=5,
            scheduler_patience=5,
            scheduler_factor=0.85,
            scheduler_min_lr=1e-6,
            validation_split=0.0,
            random_state=0,
            standardize=True,
            device=None,
            verbose=True,
        ),
        loss_weight_config=LossWeightConfig(
            lambda_cross=0.35,
            lambda_align=0.0,
            cross_warmup_epochs=12,
            cross_ramp_epochs=24,
            probe_mask_probability=0.5,
            probe_alpha_local=1.0,
            probe_alpha_region_time=1.0,
            probe_alpha_derivative=1.0,
            probe_alpha_frequency=1.0,
            probe_alpha_correlation=1.0,
        ),
    )


def write_default_12t_config(path: str | Path = DEFAULT_12T_CONFIG_PATH) -> Path:
    return write_commutative_transformer_pretraining_config(default_12t_pretraining_config(), path)


def default_15t_config() -> dict[str, Any]:
    return {
        "dataset_artifact_path": None,
        "pretraining_config_path": "artifacts/pretrained_commutative_transformer/config.yaml",
        "experiment_output_dir": "artifacts/nb15T_commutative_transformer_full_finetune",
        "holdout_fraction": 0.25,
        "validation_fraction_within_train": 0.20,
        "train_num_random_rotations": 6,
        "rotation_range_degrees": 12.0,
        "freeze_backbone": False,
        "optimization_config": asdict(
            OptimizationConfig(
                batch_size=8,
                epochs=100,
                learning_rate=5e-5,
                weight_decay=3e-3,
                early_stopping_patience=8,
                early_stopping_min_delta=0.0,
                training_plot_every_n_epochs=1,
                scheduler_patience=2,
                scheduler_factor=0.7,
                scheduler_min_lr=1e-6,
                validation_split=0.0,
                random_state=0,
                standardize=True,
                device=None,
                verbose=True,
            )
        ),
        "loss_weight_config": asdict(
            LossWeightConfig(
                action_weight=1.0,
                compound_weight=0.05,
                concentration_weight=0.05,
                lambda_align=0.0,
            )
        ),
    }


def write_default_15t_config(path: str | Path = DEFAULT_15T_CONFIG_PATH) -> Path:
    return write_yaml_mapping(path, default_15t_config())


def ensure_default_transformer_configs() -> None:
    if not DEFAULT_12T_CONFIG_PATH.exists():
        write_default_12t_config(DEFAULT_12T_CONFIG_PATH)
    if not DEFAULT_15T_CONFIG_PATH.exists():
        write_default_15t_config(DEFAULT_15T_CONFIG_PATH)


def run_12t_pretraining(config_path: str | Path = DEFAULT_12T_CONFIG_PATH) -> Path:
    config_path = Path(config_path)
    config = load_commutative_transformer_pretraining_config(config_path) if config_path.exists() else default_12t_pretraining_config()
    raw_config = read_yaml_mapping(config_path) if config_path.exists() else {}
    experiment_output_dir = Path(raw_config.get("experiment_output_dir", "artifacts/pretrained_commutative_transformer"))
    experiment_run = _experiment_run_from_config_path(config_path, "12T_pretrain_commutative_transformer") or create_experiment_run(
        experiment_output_dir,
        "12T_pretrain_commutative_transformer",
    )
    run_dir = Path(experiment_run.run_dir)
    pretrained_encoder_path = run_dir / f"{experiment_run.experiment_id}_encoder_state.pt"
    resume_checkpoint_path = run_dir / "resume" / f"{experiment_run.experiment_id}_training_resume.pt"
    suspend_marker_path = run_dir / "control" / f"{experiment_run.experiment_id}.suspend"
    update_agent_run_status(
        status="running",
        experiment="12T",
        experiment_id=experiment_run.experiment_id,
        run_dir=run_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        suspend_marker_path=suspend_marker_path,
    )
    optimization_config = replace(
        config.optimization_config,
        training_plot_dir=str(Path(experiment_run.loss_plot_dir) / "pretraining"),
        resume_checkpoint_path=str(resume_checkpoint_path),
        suspend_marker_path=str(suspend_marker_path),
    )
    resolved_config = replace(config, pretrained_encoder_path=pretrained_encoder_path, optimization_config=optimization_config)
    pretraining_config_path = write_commutative_transformer_pretraining_config(
        resolved_config,
        run_dir / f"{experiment_run.experiment_id}_config.yaml",
    )

    print(f"Experiment id: {experiment_run.experiment_id}", flush=True)
    print(f"Experiment run folder: {run_dir.resolve()}", flush=True)
    print(f"Per-run commutative transformer pretraining config: {pretraining_config_path.resolve()}", flush=True)

    unlabeled_dataset = load_unlabeled_tensor_dataset(resolved_config.unlabeled_dataset_path)
    print(
        {
            "all_tensors": tuple(unlabeled_dataset["tensors"].shape),
            "all_metadata": tuple(unlabeled_dataset["metadata"].shape),
        },
        flush=True,
    )

    model = CommutativeTransformerClassifier(
        model_config=resolved_config.model_config,
        optimization_config=optimization_config,
        loss_weight_config=resolved_config.loss_weight_config,
    )
    install_training_signal_handlers(model)
    try:
        model.pretrain(unlabeled_dataset["tensors"])
    except TrainingSuspended as exc:
        update_agent_run_status(
            status="suspended",
            experiment="12T",
            experiment_id=experiment_run.experiment_id,
            run_dir=run_dir,
            resume_checkpoint_path=exc.checkpoint_path,
            suspend_marker_path=suspend_marker_path,
        )
        print(f"Training suspended. Remove {suspend_marker_path} and rerun this config to resume.", flush=True)
        return run_dir
    pretrained_encoder_path = model.save_pretrained_encoder(pretrained_encoder_path)
    latest_pretraining_config_path = write_commutative_transformer_pretraining_config(
        resolved_config,
        experiment_output_dir / "config.yaml",
    )
    print(f"Updated latest commutative transformer pretraining config at {latest_pretraining_config_path}", flush=True)
    persist_pretraining_artifacts(
        output_dir=experiment_output_dir,
        estimator=model,
        config=resolved_config,
        experiment_prefix="12T_pretrain_commutative_transformer",
        experiment_id=experiment_run.experiment_id,
        pretrained_encoder_path=pretrained_encoder_path,
        loss_plot_dirs=[optimization_config.training_plot_dir],
        analysis="Agent-run transformer pretraining: inspect loss curves and checkpoint readiness for 15T.",
        next_round_proposal="If the transformer pretraining loss is unstable, reduce cross pressure or learning rate before 15T.",
    )
    update_agent_run_status(
        status="completed",
        experiment="12T",
        experiment_id=experiment_run.experiment_id,
        run_dir=run_dir,
        checkpoint_path=pretrained_encoder_path,
    )
    return run_dir


def run_15t_finetune(config_path: str | Path = DEFAULT_15T_CONFIG_PATH) -> Path:
    raw_config = default_15t_config()
    config_path = Path(config_path)
    if config_path.exists():
        raw_config = merge_dicts(raw_config, read_yaml_mapping(config_path))

    dataset_artifact_path = Path(raw_config["dataset_artifact_path"]) if raw_config.get("dataset_artifact_path") else load_current_dataset_artifact_path()
    pretraining_config_path = Path(raw_config["pretraining_config_path"])
    pretraining_config = load_commutative_transformer_pretraining_config(pretraining_config_path)
    pretrained_encoder_path = pretraining_config.pretrained_encoder_path
    if not pretrained_encoder_path.exists():
        raise FileNotFoundError(
            f"Pretrained transformer encoder not found at {pretrained_encoder_path}. Run 12T first."
        )

    experiment_output_dir = Path(raw_config["experiment_output_dir"])
    experiment_run = (
        _experiment_run_from_config_path(config_path, "15T_finetune_commutative_transformer")
        or (
            _experiment_run_from_existing_dir(Path(raw_config["experiment_run_dir"]), str(raw_config["experiment_id"]))
            if raw_config.get("experiment_run_dir") and raw_config.get("experiment_id")
            else None
        )
        or create_experiment_run(experiment_output_dir, "15T_finetune_commutative_transformer")
    )
    run_dir = Path(experiment_run.run_dir)
    loss_plot_dir = Path(experiment_run.loss_plot_dir) / "training"
    figure_dir = Path(experiment_run.figure_dir)
    resume_checkpoint_path = run_dir / "resume" / f"{experiment_run.experiment_id}_training_resume.pt"
    suspend_marker_path = run_dir / "control" / f"{experiment_run.experiment_id}.suspend"
    update_agent_run_status(
        status="running",
        experiment="15T",
        experiment_id=experiment_run.experiment_id,
        run_dir=run_dir,
        resume_checkpoint_path=resume_checkpoint_path,
        suspend_marker_path=suspend_marker_path,
    )

    optimization_config = OptimizationConfig(
        **{
            **dict(raw_config["optimization_config"]),
            "training_plot_dir": str(loss_plot_dir),
            "resume_checkpoint_path": str(resume_checkpoint_path),
            "suspend_marker_path": str(suspend_marker_path),
        }
    )
    loss_weight_config = LossWeightConfig(**dict(raw_config["loss_weight_config"]))
    resolved_yaml_config = {
        **raw_config,
        "dataset_artifact_path": str(dataset_artifact_path),
        "pretrained_encoder_path": str(pretrained_encoder_path),
        "experiment_id": experiment_run.experiment_id,
        "experiment_run_dir": str(run_dir),
        "loss_plot_dir": str(loss_plot_dir),
        "model_config": asdict(pretraining_config.model_config),
        "optimization_config": asdict(optimization_config),
        "loss_weight_config": asdict(loss_weight_config),
    }
    write_yaml_mapping(run_dir / f"{experiment_run.experiment_id}_config.yaml", to_yamlable(resolved_yaml_config))

    print(f"Experiment id: {experiment_run.experiment_id}", flush=True)
    print(f"Experiment run folder: {run_dir.resolve()}", flush=True)
    print(f"Using pretrained encoder checkpoint: {pretrained_encoder_path.resolve()}", flush=True)
    print(f"Training loss PDFs: {loss_plot_dir.resolve()}", flush=True)

    dataset = load_labeled_tensor_dataset(dataset_artifact_path)
    experiment = prepare_multitask_experiment_data(
        dataset,
        holdout_fraction=float(raw_config["holdout_fraction"]),
        validation_fraction_within_train=float(raw_config["validation_fraction_within_train"]),
        train_num_random_rotations=int(raw_config["train_num_random_rotations"]),
        rotation_range_degrees=float(raw_config["rotation_range_degrees"]),
        random_state=optimization_config.random_state,
    )

    model = CommutativeTransformerClassifier(
        model_config=pretraining_config.model_config,
        optimization_config=optimization_config,
        loss_weight_config=loss_weight_config,
        pretrained_state_path=pretrained_encoder_path,
        freeze_backbone=bool(raw_config["freeze_backbone"]),
    )
    model.live_checkpoint_path = run_dir / f"{experiment_run.experiment_id}_model_state.pt"
    install_training_signal_handlers(model)
    try:
        fit_estimator_on_experiment(model, experiment)
    except TrainingSuspended as exc:
        update_agent_run_status(
            status="suspended",
            experiment="15T",
            experiment_id=experiment_run.experiment_id,
            run_dir=run_dir,
            resume_checkpoint_path=exc.checkpoint_path,
            suspend_marker_path=suspend_marker_path,
        )
        print(f"Training suspended. Remove {suspend_marker_path} and rerun this config to resume.", flush=True)
        return run_dir

    predictions = model.predict(experiment.splits.X_holdout)
    probabilities = model.predict_proba(experiment.splits.X_holdout)
    reports = evaluate_multitask_estimator(
        model,
        experiment.splits.X_holdout,
        experiment.y_true_holdout,
        label_maps=experiment.label_maps,
        class_labels=experiment.class_labels,
    )
    reports_excluding_control, y_true_excluding_control, predictions_excluding_control, probabilities_excluding_control = build_reports_excluding_control(
        y_true=experiment.y_true_holdout,
        probabilities=probabilities,
        class_labels=experiment.class_labels,
        label_maps=experiment.label_maps,
    )
    holdout_evaluation = MultitaskEvaluationResult(
        predictions=predictions,
        probabilities=probabilities,
        reports=reports,
        reports_excluding_control=reports_excluding_control,
        predictions_excluding_control=predictions_excluding_control,
        probabilities_excluding_control=probabilities_excluding_control,
        y_true_excluding_control=y_true_excluding_control,
    )

    holdout_embedding_projection = build_tensor_embedding_2d(
        model.transform(experiment.splits.X_holdout),
        experiment.y_true_holdout["action"],
        label_map=experiment.label_maps["action"],
        metadata=experiment.splits.metadata_holdout,
        method="umap",
        random_state=optimization_config.random_state,
    )
    holdout_embedding_projection.to_csv(figure_dir / f"{experiment_run.experiment_id}_holdout_embedding_umap.csv", index=False)
    plot_tensor_embedding_2d(
        holdout_embedding_projection,
        title="Holdout embedding projection by action",
        marker_column="compound",
        output_path=figure_dir / f"{experiment_run.experiment_id}_holdout_embedding_umap.pdf",
    )

    persist_experiment_artifacts(
        output_dir=experiment_output_dir,
        estimator=model,
        reports=reports,
        config=resolved_yaml_config,
        experiment_prefix="15T_finetune_commutative_transformer",
        experiment_id=experiment_run.experiment_id,
        evaluation=holdout_evaluation,
        experiment=experiment,
        loss_plot_dirs=[loss_plot_dir],
        analysis="Agent-run transformer fine-tune: inspect holdout metrics and UMAP separation.",
        next_round_proposal="Patch 12T or 15T according to transformer-specific failure modes.",
    )
    update_agent_run_status(status="completed", experiment="15T", experiment_id=experiment_run.experiment_id, run_dir=run_dir)
    return run_dir

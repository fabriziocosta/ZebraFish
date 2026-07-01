from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.commutative_cnn_pretraining_config import (
    CommutativeCNNPretrainingConfig,
    load_commutative_cnn_pretraining_config,
    write_commutative_cnn_pretraining_config,
)
from src.dataset_config import load_current_dataset_artifact_path
from src.experiment_runner_shared import merge_dicts, read_yaml_mapping, to_yamlable, update_agent_run_status, write_yaml_mapping
from src.models.configs import CommutativeCNNConfig, LossWeightConfig, OptimizationConfig
from src.models.estimators import CommutativeCNNClassifier
from src.tensor_utils import build_tensor_embedding_2d, load_labeled_tensor_dataset, load_unlabeled_tensor_dataset, plot_tensor_embedding_2d
from src.training.data import augment_training_tensors_with_rotations
from src.training.workflow import (
    MultitaskEvaluationResult,
    build_reports_excluding_control,
    create_experiment_run,
    evaluate_multitask_estimator,
    fit_chunked_water_vs_other_hot_start,
    fit_estimator_on_experiment,
    persist_experiment_artifacts,
    persist_pretraining_artifacts,
    prepare_multitask_experiment_data,
)


DEFAULT_10C_CONFIG_PATH = Path("configs/experiments/10C_pretrain_next.yaml")
DEFAULT_13C_CONFIG_PATH = Path("configs/experiments/13C_finetune_next.yaml")


def default_10c_pretraining_config() -> CommutativeCNNPretrainingConfig:
    return CommutativeCNNPretrainingConfig(
        unlabeled_dataset_path=Path(".dataset_cache/unlabeled_active_high_mid_low_t20_z5_y96_x96_chunks"),
        pretrained_encoder_path=Path("artifacts/pretrained_commutative_cnn/runs/PENDING/PENDING_encoder_state.pt"),
        validation_fraction=0.15,
        train_num_random_rotations=0,
        rotation_range_degrees=0.0,
        model_config=CommutativeCNNConfig(
            spatial_conv_channels=(16, 32),
            spatial_kernel_size_z=(3, 1),
            spatial_kernel_size_xy=(5, 3),
            spatial_stride_z=(1, 1),
            spatial_stride_xy=(1, 1),
            spatial_pool_kernel_z=(1, 1),
            spatial_pool_kernel_xy=(2, 2),
            spatial_pool_stride_z=(1, 1),
            spatial_pool_stride_xy=(2, 2),
            temporal_st_channels=(48, 64),
            temporal_st_kernel_sizes=(5, 3),
            temporal_ts_channels=(32, 48, 64),
            temporal_ts_kernel_sizes=(7, 5, 3),
            spatial_agg_channels=(32, 64),
            spatial_agg_kernel_size_z=(3, 1),
            spatial_agg_kernel_size_xy=(3, 3),
            spatial_agg_stride_z=(1, 1),
            spatial_agg_stride_xy=(1, 1),
            spatial_agg_pool_kernel_z=(1, 1),
            spatial_agg_pool_kernel_xy=(1, 2),
            spatial_agg_pool_stride_z=(1, 1),
            spatial_agg_pool_stride_xy=(1, 2),
            patch_size_z=1,
            patch_size_xy=16,
            embedding_dim=64,
            num_prototypes=64,
            probe_region_grid=(1, 2, 2),
            probe_time_bins=8,
            probe_frequency_bins=4,
            dropout=0.25,
            normalization="group",
        ),
        optimization_config=OptimizationConfig(
            batch_size=8,
            epochs=140,
            learning_rate=3e-5,
            weight_decay=1e-3,
            early_stopping_patience=16,
            early_stopping_min_delta=5e-5,
            early_stopping_start_epoch=20,
            early_stopping_monitor="self_probe_loss",
            early_stopping_smoothing="median",
            early_stopping_smoothing_window=3,
            training_plot_dir=None,
            training_plot_every_n_epochs=2,
            training_plot_smoothing_window=5,
            scheduler_patience=3,
            scheduler_factor=0.5,
            scheduler_min_lr=1e-6,
            validation_split=0.0,
            random_state=0,
            standardize=True,
            device=None,
            verbose=True,
        ),
        loss_weight_config=LossWeightConfig(
            lambda_cross=0.0,
            cross_warmup_epochs=0,
            cross_ramp_epochs=0,
            prototype_temperature=0.25,
            prototype_alignment_weight=0.0,
            prototype_warmup_epochs=16,
            prototype_ramp_epochs=36,
            latent_alignment_weight=0.0,
            lambda_align=0.0,
            probe_mask_probability=1.0,
            probe_alpha_local=1.0,
            probe_alpha_region_time=1.0,
            probe_alpha_derivative=0.25,
            probe_alpha_frequency=0.10,
            probe_alpha_correlation=0.05,
        ),
    )


def write_default_10c_config(path: str | Path = DEFAULT_10C_CONFIG_PATH) -> Path:
    return write_commutative_cnn_pretraining_config(default_10c_pretraining_config(), path)


def default_13c_config() -> dict[str, Any]:
    return {
        "dataset_artifact_path": None,
        "unlabeled_dataset_path": ".dataset_cache/unlabeled_active_high_mid_low_t20_z5_y96_x96_chunks",
        "pretraining_config_path": "artifacts/pretrained_commutative_cnn/config.yaml",
        "experiment_output_dir": "artifacts/nb13C_commutative_cnn_full_finetune",
        "holdout_fraction": 0.25,
        "validation_fraction_within_train": 0.20,
        "train_num_random_rotations": 8,
        "rotation_range_degrees": 10.0,
        "freeze_backbone": False,
        "hot_start": True,
        "run_binary_hot_start": False,
        "binary_pretraining_epochs": 2,
        "binary_learning_rate": 1e-5,
        "binary_weight_decay": 7.5e-4,
        "binary_class_weighting": None,
        "optimization_config": asdict(
            OptimizationConfig(
                batch_size=8,
                epochs=120,
                learning_rate=8e-6,
                weight_decay=1e-3,
                early_stopping_patience=14,
                early_stopping_min_delta=5e-5,
                early_stopping_start_epoch=12,
                early_stopping_monitor="compound_loss",
                early_stopping_smoothing="median",
                early_stopping_smoothing_window=5,
                scheduler_patience=3,
                scheduler_factor=0.5,
                scheduler_min_lr=1e-6,
                validation_split=0.0,
                random_state=0,
                standardize=True,
                device=None,
                verbose=True,
                class_weighting="balanced",
            )
        ),
        "loss_weight_config": asdict(
            LossWeightConfig(
                action_weight=0.65,
                water_vs_other_weight=0.03,
                compound_weight=0.80,
                concentration_weight=0.05,
                latent_alignment_weight=0.0,
                lambda_align=0.0,
            )
        ),
    }


def write_default_13c_config(path: str | Path = DEFAULT_13C_CONFIG_PATH) -> Path:
    return write_yaml_mapping(path, default_13c_config())


def ensure_default_cnn_configs() -> None:
    if not DEFAULT_10C_CONFIG_PATH.exists():
        write_default_10c_config(DEFAULT_10C_CONFIG_PATH)
    if not DEFAULT_13C_CONFIG_PATH.exists():
        write_default_13c_config(DEFAULT_13C_CONFIG_PATH)


def run_10c_pretraining(config_path: str | Path = DEFAULT_10C_CONFIG_PATH) -> Path:
    config_path = Path(config_path)
    config = load_commutative_cnn_pretraining_config(config_path) if config_path.exists() else default_10c_pretraining_config()
    raw_config = read_yaml_mapping(config_path) if config_path.exists() else {}
    experiment_output_dir = Path(raw_config.get("experiment_output_dir", "artifacts/pretrained_commutative_cnn"))
    experiment_run = create_experiment_run(experiment_output_dir, "10C_pretrain_commutative_cnn")
    run_dir = Path(experiment_run.run_dir)
    update_agent_run_status(status="running", experiment="10C", experiment_id=experiment_run.experiment_id, run_dir=run_dir)
    pretrained_encoder_path = run_dir / f"{experiment_run.experiment_id}_encoder_state.pt"
    optimization_config = replace(
        config.optimization_config,
        training_plot_dir=str(Path(experiment_run.loss_plot_dir) / "pretraining"),
    )
    resolved_config = replace(config, pretrained_encoder_path=pretrained_encoder_path, optimization_config=optimization_config)
    pretraining_config_path = write_commutative_cnn_pretraining_config(
        resolved_config,
        run_dir / f"{experiment_run.experiment_id}_config.yaml",
    )

    previous_pretrained_encoder_path = None
    previous_pretraining_config_path = experiment_output_dir / "config.yaml"
    if previous_pretraining_config_path.exists():
        previous_pretraining_config = load_commutative_cnn_pretraining_config(previous_pretraining_config_path)
        previous_pretrained_encoder_path = previous_pretraining_config.pretrained_encoder_path
        if not previous_pretrained_encoder_path.exists():
            previous_pretrained_encoder_path = None

    print(f"Experiment id: {experiment_run.experiment_id}", flush=True)
    print(f"Experiment run folder: {run_dir.resolve()}", flush=True)
    print(f"Per-run commutative CNN pretraining config: {pretraining_config_path.resolve()}", flush=True)

    unlabeled_dataset = load_unlabeled_tensor_dataset(resolved_config.unlabeled_dataset_path)
    train_indices, val_indices = train_test_split(
        range(len(unlabeled_dataset["tensors"])),
        test_size=resolved_config.validation_fraction,
        random_state=optimization_config.random_state,
        shuffle=True,
    )
    X_train_base = unlabeled_dataset["tensors"][train_indices]
    X_val = unlabeled_dataset["tensors"][val_indices]
    metadata_train_base = unlabeled_dataset["metadata"].iloc[train_indices].reset_index(drop=True)
    X_train, _, _ = augment_training_tensors_with_rotations(
        X_train_base,
        [0] * len(X_train_base),
        metadata=metadata_train_base,
        num_random_rotations=resolved_config.train_num_random_rotations,
        rotation_range_degrees=resolved_config.rotation_range_degrees,
        random_state=optimization_config.random_state,
    )
    print(
        {
            "all_tensors": tuple(unlabeled_dataset["tensors"].shape),
            "train_base_tensors": tuple(X_train_base.shape),
            "train_tensors": tuple(X_train.shape),
            "val_tensors": tuple(X_val.shape),
        },
        flush=True,
    )

    model = CommutativeCNNClassifier(
        model_config=resolved_config.model_config,
        optimization_config=optimization_config,
        loss_weight_config=resolved_config.loss_weight_config,
        pretrained_state_path=previous_pretrained_encoder_path,
    )
    model.pretrain(X_train, validation_data=X_val)
    pretrained_encoder_path = model.save_pretrained_encoder(pretrained_encoder_path)
    latest_pretraining_config_path = write_commutative_cnn_pretraining_config(
        resolved_config,
        experiment_output_dir / "config.yaml",
    )
    print(f"Updated latest commutative CNN pretraining config at {latest_pretraining_config_path}", flush=True)
    persist_pretraining_artifacts(
        output_dir=experiment_output_dir,
        estimator=model,
        config=resolved_config,
        experiment_prefix="10C_pretrain_commutative_cnn",
        experiment_id=experiment_run.experiment_id,
        pretrained_encoder_path=pretrained_encoder_path,
        loss_plot_dirs=[optimization_config.training_plot_dir],
        analysis=(
            "Agent-run pretraining: inspect live history CSVs and PDFs for validation self-probe trajectory, "
            "train/validation gap, and whether the run plateaued before early stopping."
        ),
        next_round_proposal=(
            "If validation self-probe plateaus cleanly, use this checkpoint for 13C; otherwise patch the next "
            "pretraining YAML based on the observed failure mode."
        ),
    )
    update_agent_run_status(
        status="completed",
        experiment="10C",
        experiment_id=experiment_run.experiment_id,
        run_dir=run_dir,
        checkpoint_path=pretrained_encoder_path,
    )
    return run_dir


def run_13c_finetune(config_path: str | Path = DEFAULT_13C_CONFIG_PATH) -> Path:
    raw_config = default_13c_config()
    config_path = Path(config_path)
    if config_path.exists():
        raw_config = merge_dicts(raw_config, read_yaml_mapping(config_path))

    dataset_artifact_path = Path(raw_config["dataset_artifact_path"]) if raw_config.get("dataset_artifact_path") else load_current_dataset_artifact_path()
    pretraining_config_path = Path(raw_config["pretraining_config_path"])
    pretraining_config = load_commutative_cnn_pretraining_config(pretraining_config_path)
    pretrained_encoder_path = pretraining_config.pretrained_encoder_path
    if not pretrained_encoder_path.exists():
        raise FileNotFoundError(
            f"Pretrained CNN encoder not found at {pretrained_encoder_path}. "
            "Run 10C first, or update artifacts/pretrained_commutative_cnn/config.yaml."
        )

    experiment_output_dir = Path(raw_config["experiment_output_dir"])
    experiment_run = create_experiment_run(experiment_output_dir, "13C_finetune_commutative_cnn")
    run_dir = Path(experiment_run.run_dir)
    update_agent_run_status(status="running", experiment="13C", experiment_id=experiment_run.experiment_id, run_dir=run_dir)
    loss_plot_root = Path(experiment_run.loss_plot_dir)
    figure_dir = Path(experiment_run.figure_dir)
    binary_loss_plot_dir = loss_plot_root / "binary_hot_start"
    fine_tune_loss_plot_dir = loss_plot_root / "fine_tune"

    optimization_config = OptimizationConfig(**{**dict(raw_config["optimization_config"]), "training_plot_dir": str(fine_tune_loss_plot_dir)})
    loss_weight_config = LossWeightConfig(**dict(raw_config["loss_weight_config"]))
    resolved_yaml_config = {
        **raw_config,
        "dataset_artifact_path": str(dataset_artifact_path),
        "pretrained_encoder_path": str(pretrained_encoder_path),
        "experiment_id": experiment_run.experiment_id,
        "experiment_run_dir": str(run_dir),
        "binary_loss_plot_dir": str(binary_loss_plot_dir),
        "fine_tune_loss_plot_dir": str(fine_tune_loss_plot_dir),
        "optimization_config": asdict(optimization_config),
        "loss_weight_config": asdict(loss_weight_config),
        "model_config": asdict(pretraining_config.model_config),
    }
    write_yaml_mapping(run_dir / f"{experiment_run.experiment_id}_config.yaml", to_yamlable(resolved_yaml_config))

    print(f"Experiment id: {experiment_run.experiment_id}", flush=True)
    print(f"Experiment run folder: {run_dir.resolve()}", flush=True)
    print(f"Using pretrained encoder checkpoint: {pretrained_encoder_path.resolve()}", flush=True)
    print(f"Fine-tune loss PDFs: {fine_tune_loss_plot_dir.resolve()}", flush=True)

    dataset = load_labeled_tensor_dataset(dataset_artifact_path)
    experiment = prepare_multitask_experiment_data(
        dataset,
        holdout_fraction=float(raw_config["holdout_fraction"]),
        validation_fraction_within_train=float(raw_config["validation_fraction_within_train"]),
        train_num_random_rotations=int(raw_config["train_num_random_rotations"]),
        rotation_range_degrees=float(raw_config["rotation_range_degrees"]),
        random_state=optimization_config.random_state,
    )

    model = CommutativeCNNClassifier(
        model_config=pretraining_config.model_config,
        optimization_config=optimization_config,
        loss_weight_config=loss_weight_config,
        pretrained_state_path=pretrained_encoder_path,
        freeze_backbone=bool(raw_config["freeze_backbone"]),
        hot_start=bool(raw_config["hot_start"]),
    )
    model.live_checkpoint_path = run_dir / f"{experiment_run.experiment_id}_model_state.pt"

    binary_pretraining_data = None
    binary_pretraining_history = None
    if bool(raw_config["run_binary_hot_start"]) and int(raw_config["binary_pretraining_epochs"]) > 0:
        final_epochs = model.epochs
        final_learning_rate = model.learning_rate
        final_weight_decay = model.weight_decay
        model.training_plot_dir = str(binary_loss_plot_dir)
        model.training_plot_title = "Water-vs-other hot-start loss curves"
        model.learning_rate = float(raw_config["binary_learning_rate"])
        model.weight_decay = float(raw_config["binary_weight_decay"])
        binary_pretraining_data = fit_chunked_water_vs_other_hot_start(
            model,
            raw_config["unlabeled_dataset_path"],
            holdout_metadata=experiment.splits.metadata_holdout,
            validation_fraction=float(raw_config["validation_fraction_within_train"]),
            epochs=int(raw_config["binary_pretraining_epochs"]),
            random_state=optimization_config.random_state,
            class_weighting=raw_config["binary_class_weighting"],
        )
        binary_pretraining_history = model.history_.copy()
        model.epochs = final_epochs
        model.learning_rate = final_learning_rate
        model.weight_decay = final_weight_decay

    model.training_plot_dir = str(fine_tune_loss_plot_dir)
    model.training_plot_title = "Pretrained commutative CNN full fine-tune loss curves"
    fit_estimator_on_experiment(model, experiment)

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

    all_labeled_tensors = torch.cat([experiment.splits.X_train_base, experiment.splits.X_val, experiment.splits.X_holdout], dim=0)
    all_action_labels = torch.cat(
        [
            experiment.splits.y_train_base,
            torch.as_tensor(experiment.splits.y_val),
            torch.as_tensor(experiment.splits.y_holdout),
        ]
    ).numpy()
    all_labeled_metadata = pd.concat(
        [
            experiment.splits.metadata_train_base.assign(dataset_split="train"),
            experiment.splits.metadata_val.assign(dataset_split="validation"),
            experiment.splits.metadata_holdout.assign(dataset_split="holdout"),
        ],
        ignore_index=True,
    )
    all_embedding_projection = build_tensor_embedding_2d(
        model.transform(all_labeled_tensors),
        all_action_labels,
        label_map=experiment.label_maps["action"],
        metadata=all_labeled_metadata,
        method="umap",
        random_state=optimization_config.random_state,
    )
    all_embedding_projection.to_csv(figure_dir / f"{experiment_run.experiment_id}_all_labeled_embedding_umap.csv", index=False)
    plot_tensor_embedding_2d(
        all_embedding_projection,
        title="All labeled data embedding projection by action (with controls)",
        marker_column="compound",
        edge_color_column="dataset_split",
        edge_color_map={"train": "white", "validation": "black", "holdout": "black"},
        display_control=True,
        output_path=figure_dir / f"{experiment_run.experiment_id}_all_labeled_embedding_umap_with_controls.pdf",
    )
    plot_tensor_embedding_2d(
        all_embedding_projection,
        title="All labeled data embedding projection by action (controls hidden)",
        marker_column="compound",
        edge_color_column="dataset_split",
        edge_color_map={"train": "white", "validation": "black", "holdout": "black"},
        display_control=False,
        output_path=figure_dir / f"{experiment_run.experiment_id}_all_labeled_embedding_umap_controls_hidden.pdf",
    )

    run_config = {
        **resolved_yaml_config,
        "binary_pretraining_excluded_holdout_count": None if binary_pretraining_data is None else binary_pretraining_data.excluded_holdout_count,
        "binary_pretraining_label_map": None if binary_pretraining_data is None else binary_pretraining_data.label_map,
        "binary_pretraining_train_count": None if binary_pretraining_data is None else binary_pretraining_data.train_count,
        "binary_pretraining_val_count": None if binary_pretraining_data is None else binary_pretraining_data.val_count,
    }
    persist_experiment_artifacts(
        output_dir=experiment_output_dir,
        estimator=model,
        reports=reports,
        config=run_config,
        experiment_prefix="13C_finetune_commutative_cnn",
        experiment_id=experiment_run.experiment_id,
        evaluation=holdout_evaluation,
        experiment=experiment,
        loss_plot_dirs=[binary_loss_plot_dir, fine_tune_loss_plot_dir],
        analysis="Agent-run fine-tune: inspect compound metrics, confusion matrices, AUC, action degradation, and UMAP separation.",
        next_round_proposal=(
            "Patch the next 10C or 13C YAML according to whether compound discrimination, action stability, or embedding "
            "separation remains the limiting failure mode."
        ),
    )
    if binary_pretraining_history is not None:
        binary_pretraining_history.to_csv(run_dir / f"{experiment_run.experiment_id}_binary_pretraining_history.csv", index=False)
    update_agent_run_status(status="completed", experiment="13C", experiment_id=experiment_run.experiment_id, run_dir=run_dir)
    return run_dir

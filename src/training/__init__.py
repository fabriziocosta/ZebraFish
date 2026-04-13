from src.training.data import TensorDatasetSplits, augment_training_tensors_with_rotations, split_labeled_tensor_dataset_by_instance
from src.training.reporting import build_classification_reports, build_multitask_classification_reports, plot_confusion_matrices, plot_training_history
from src.training.workflow import (
    ExperimentArtifacts,
    MultitaskExperimentData,
    MultitaskEvaluationResult,
    display_experiment_summary,
    display_holdout_evaluation,
    evaluate_multitask_estimator,
    fit_estimator_on_experiment,
    persist_experiment_artifacts,
    plot_holdout_branch_embedding_projections,
    plot_holdout_embedding_projection,
    prepare_multitask_experiment_data,
)

__all__ = [
    "ExperimentArtifacts",
    "MultitaskExperimentData",
    "MultitaskEvaluationResult",
    "TensorDatasetSplits",
    "augment_training_tensors_with_rotations",
    "split_labeled_tensor_dataset_by_instance",
    "build_classification_reports",
    "build_multitask_classification_reports",
    "display_experiment_summary",
    "display_holdout_evaluation",
    "evaluate_multitask_estimator",
    "fit_estimator_on_experiment",
    "persist_experiment_artifacts",
    "plot_confusion_matrices",
    "plot_holdout_branch_embedding_projections",
    "plot_holdout_embedding_projection",
    "plot_training_history",
    "prepare_multitask_experiment_data",
]

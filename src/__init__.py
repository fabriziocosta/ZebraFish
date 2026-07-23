try:
    from src.ml import (
        CommutativeCNNClassifier,
        CommutativeCNNConfig,
        CommutativeTransformerClassifier,
        CommutativeTransformerConfig,
        LossWeightConfig,
        OptimizationConfig,
        TimeChannel3DCNNClassifier,
        TimeChannel3DCNNConfig,
    )
except ModuleNotFoundError as exc:
    # State migration, observation extraction, and campaign inspection are
    # intentionally usable on a controller host without the training stack.
    if exc.name != "torch":
        raise
    CommutativeCNNClassifier = None
    CommutativeCNNConfig = None
    CommutativeTransformerClassifier = None
    CommutativeTransformerConfig = None
    LossWeightConfig = None
    OptimizationConfig = None
    TimeChannel3DCNNClassifier = None
    TimeChannel3DCNNConfig = None

__all__ = [
    "CommutativeCNNConfig",
    "CommutativeCNNClassifier",
    "CommutativeTransformerConfig",
    "CommutativeTransformerClassifier",
    "LossWeightConfig",
    "OptimizationConfig",
    "TimeChannel3DCNNConfig",
    "TimeChannel3DCNNClassifier",
]

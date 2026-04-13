# ZebraFish

This repo contains a small notebook-and-utility workflow for linking a compound workbook to zebrafish imaging folders, deriving concentration-aware condition maps, and inspecting image time series.

Start here: use this README to understand the repository layout, the notebook workflow, and where each detailed document lives.

The workflow is built around:

- a workbook of compounds, classes, and mechanisms of action
- a mounted imaging directory tree under `/mnt/tyler`
- shared utility code in [`src/notebook_utils.py`](src/notebook_utils.py)
- dedicated tensor loading code in [`src/tensor_utils.py`](src/tensor_utils.py)
- dedicated ML/training code in [`src/ml.py`](src/ml.py)
- eight notebooks for exploration, dataset preparation, baseline training, and commutative-model experiments

## Documentation

- [README.md](README.md)
  Repository index and workflow orchestration: inputs, modules, notebooks, artifacts, and execution order.
- [docs/method.md](docs/method.md)
  Proposed commutative-representation method: theory, architecture options, and implementation direction.
- [docs/architecture.md](docs/architecture.md)
  Detailed architecture reference for the baseline 3D CNN, the pure-CNN commutative model, and the transformer commutative model.
- [docs/preprocessing.md](docs/preprocessing.md)
  Tensor-loading, downsampling, normalization, caching, and dataset-preparation conventions.
- [docs/data_artifacts.md](docs/data_artifacts.md)
  Generated CSV artifacts and their column meanings.
- [docs/figures.md](docs/figures.md)
  Standalone figures referenced by the method document.
- [docs/introduction.md](docs/introduction.md)
  Higher-level conceptual introduction to the commutative representation-learning idea.

## Inputs

### Workbook

- [`compounds (MJW V2).xlsx`](compounds%20(MJW%20V2).xlsx)
  Source workbook used to derive compound names, compound classes, mechanisms of action, and exposure-condition text.

### Imaging Mount

- [`mount_command.txt`](mount_command.txt)
  Example CIFS mount command used to expose the imaging tree at `/mnt/tyler`.

The notebooks assume the image data are accessible from the mounted path referenced in the generated CSVs, for example:

- `/mnt/tyler/Matt Winter/BRAIN IMAGES BACKUP/...`

## Naming Conventions

The shared utilities now emit normalized names by default:

- `compound`
  Clean standardized compound name. Marker asterisks from workbook labels are removed.
  Example: `DMCM*` becomes `DMCM`.

- `mechanism_of_action`
  Normalized mnemonic label with a consistent readable convention.
  Example: `GABAAR_Antagonist`, `NMDAR_Activation`, `CYP450_Inhibitor`.

Raw workbook strings are still used internally where needed for folder matching and for building normalization maps, but the main dataframe outputs use normalized values.

## Main Utility Module

- [`src/notebook_utils.py`](src/notebook_utils.py)
  Shared logic for:
  - loading workbook data
  - normalizing compound names
  - mapping mechanisms of action to mnemonic aliases
  - discovering candidate image run folders
  - building run-level and condition-level mapping tables
  - parsing concentration labels from folder names
  - plotting sampled mid-z time slices from a tensor

## Tensor Loading Module

- [`src/tensor_utils.py`](src/tensor_utils.py)
  Dedicated image tensor materialization code.

The detailed preprocessing specification now lives in [docs/preprocessing.md](docs/preprocessing.md).

At a high level, the tensor pipeline handles:

- TIFF loading and axis normalization to `T x Z x Y x X`
- deterministic downsampling across time and space
- optional LOESS-style global intensity-drift normalization
- repo-local tensor and selected-TIFF caching
- labeled dataset assembly for downstream ML workflows

Key public helpers include:

- `load_compound_classification()`
- `build_compound_standardization_map()`
- `build_mechanism_of_action_alias_map()`
- `build_compound_image_run_map()`
- `build_compound_image_index()`
- `build_compound_image_condition_map()`
- `build_compound_image_condition_index()`
- `select_condition_choices()`
- `plot_midz_time_slices_from_tensor()`

Tensor-loading helpers include:

- `load_tiff_as_tzyx()`
- `downsample_tzyx()`
- `load_image_condition_tensor()`
- `rotate_tensor_xy()`
- `build_moa_labeled_tensor_dataset()`
- `build_tensor_embedding_2d()`
- `plot_tensor_embedding_2d()`

`plot_tensor_embedding_2d()` can now also overlay an optional low-alpha class-region background by fitting an RBF-kernel SVM directly on the 2D embedding coordinates. This is useful as a quick visual sanity check for projected class separation.

## ML Module

- [`src/ml.py`](src/ml.py)
  Simple model-training utilities for tensor classification experiments.

`src/ml.py` is now a compatibility façade. The implementation is split internally into:

- [`src/models/common.py`](src/models/common.py)
  shared tensor/label validation helpers, multitask target preparation, and the shared estimator mixin
- [`src/models/configs.py`](src/models/configs.py)
  typed configuration dataclasses for model architecture, optimization, and loss weights
- [`src/models/backbones_cnn.py`](src/models/backbones_cnn.py)
  baseline 3D CNN backbone and pure-CNN commutative backbone modules
- [`src/models/backbones_transformer.py`](src/models/backbones_transformer.py)
  transformer patch/embed/encoder blocks and the commutative transformer backbone
- [`src/models/estimators.py`](src/models/estimators.py)
  the public scikit-style estimator classes only
- [`src/training/data.py`](src/training/data.py)
  train/validation/holdout split and training-only augmentation helpers
- [`src/training/losses.py`](src/training/losses.py)
  shared commutative-consistency and auxiliary multitask loss helpers
- [`src/training/loop.py`](src/training/loop.py)
  shared optimization loop, checkpoint selection, inference batching, and score helpers
- [`src/training/reporting.py`](src/training/reporting.py)
  loss plotting, confusion matrices, and classification-report utilities
- [`src/training/workflow.py`](src/training/workflow.py)
  notebook-thinning helpers for experiment preparation, multitask evaluation, and artifact persistence

The three public estimators now share one internal estimator scaffold for:

- target preparation and encoding
- optimizer / scheduler setup
- early stopping
- epoch logging and history collection
- prediction, probability output, and embedding extraction

The model-specific code is therefore constrained to:

- backbone construction
- architecture-specific forward outputs
- architecture-specific loss terms
- branch-specific diagnostics such as `transform_branches(...)`

Typed configuration helpers are available through:

- `TimeChannel3DCNNConfig`
- `CommutativeCNNConfig`
- `CommutativeTransformerConfig`
- `OptimizationConfig`
- `LossWeightConfig`

Shared experiment helpers are available through:

- `prepare_multitask_experiment_data()`
- `evaluate_multitask_estimator()`
- `persist_experiment_artifacts()`

Current contents:

- `TimeChannel3DCNNClassifier`
  A scikit-style estimator built on a simple 3D CNN that treats time as channels and supports configurable convolution kernel sizes, strides, and pooling separately for `z` and `xy`. It now supports multi-head supervision for action, compound, and concentration targets.
- `CommutativeCNNClassifier`
  A scikit-style estimator for the pure-CNN dual-pathway commutative model, with the same multi-head target support.
- `CommutativeTransformerClassifier`
  A scikit-style estimator for the factorized transformer dual-pathway commutative model, with the same multi-head target support.
- `augment_training_tensors_with_rotations()`
  Training-only XY rotation augmentation helper.
- `split_labeled_tensor_dataset_by_instance()`
  Leakage-safe train / validation / holdout splitter for persisted tensor datasets, grouping by `original_instance_id` and returning the sliced tensors, labels, auxiliary targets, metadata, and split ids needed by notebooks 6-8.
- `build_multitask_classification_reports()`
  Builds separate per-target classification reports for `action`, `compound`, and `concentration`.
- `plot_training_history()`
  Plot train and validation loss by epoch.
- `plot_confusion_matrices()`
  Plot both absolute-count and row-fraction confusion matrices.

Estimator API note:

- `predict(...)` and `predict_proba(...)` for the current classifiers return dictionaries keyed by target:
  `action`, `compound`, and `concentration`
- `transform(...)` still returns the shared embedding used for downstream visualization and analysis

Experiment persistence note:

- `persist_experiment_artifacts()` writes a config snapshot, training history CSV, per-target report CSVs, a summary metrics CSV, and a PyTorch checkpoint for the fitted model

## Testing

The repository now includes a small `unittest` smoke suite under [`tests/`](tests) covering:

- estimator API and config-object construction
- leakage-safe split behavior
- shared experiment preparation, evaluation, and persistence helpers

Run it with:

```bash
python -m unittest discover -s tests -v
```

Important leakage rule:

- split train / validation / holdout on the unaugmented base tensors first
- keep all rotated views of the same source tensor in the same split by grouping on the original pre-rotation instance id
- apply random rotation augmentation only to the training subset afterward

This rule is enforced by `split_labeled_tensor_dataset_by_instance()` in notebook 6 and should be preserved in future model-training workflows.

## Generated CSV Files

Generated mapping and index CSVs are part of the workflow outputs. Detailed artifact descriptions now live in [docs/data_artifacts.md](docs/data_artifacts.md).

## Notebooks

### 1. Compound Class and Mechanism of Action

- [`1_compound_class_moa.ipynb`](1_compound_class_moa.ipynb)

Purpose:

- show the normalization maps first
- load the normalized compound classification table
- summarize compounds by class and normalized mechanism of action

Current flow:

1. display the compound-name normalization map
2. display the mechanism-of-action alias map
3. load `classification_df` with normalized `compound` and `mechanism_of_action`
4. display class counts and mechanism counts

### 2. Compound Image Folder Mapping

- [`2_compound_image_mapping.ipynb`](2_compound_image_mapping.ipynb)

Purpose:

- build the run-level mapping from normalized compounds to matched image run folders
- build a summarized run index for quick inspection

Main outputs in the notebook:

- `compound_image_map_df`
- `compound_image_index_df`

This notebook is the run-folder linkage step.

### 3. Compound Image Concentration Mapping

- [`3_compound_image_concentration_mapping.ipynb`](3_compound_image_concentration_mapping.ipynb)

Purpose:

- expand the run-level map into condition-level folders
- parse concentration values from condition folder names
- build a grouped condition index

Main outputs in the notebook:

- `compound_image_condition_map_df`
- `compound_image_condition_index_df`

This notebook is the concentration/condition parsing step.

### 4. View Mid-Z Time Samples

- [`4_view_midz_time_samples.ipynb`](4_view_midz_time_samples.ipynb)

Purpose:

- select a normalized compound and concentration grouping
- explicitly materialize the image data as a torch tensor
- plot all loaded time slices from the sampled middle z plane
- show the sampled source timepoints and source z index in panel titles
- inspect per-timepoint mean intensity with a line plot

Current flow:

1. build the condition map
2. set the selection inputs:
   `selected_compound`, `selector_column`, `selected_concentration`, `only_active`, `selected_condition_index`
3. set the tensor downsampling target with `tensor_size = (T, Z, Y, X)`
4. load `condition_tensor = load_image_condition_tensor(...)` directly from `condition_df` plus the selection inputs
5. inspect `condition_tensor.shape`
6. derive source-timepoint labels and the sampled middle-z label
7. plot all loaded timepoints with `plot_midz_time_slices_from_tensor(condition_tensor, n_columns=..., time_labels=..., z_label=...)`
8. plot the mean intensity by sampled timepoint with `plot_timepoint_mean_intensity(...)`

The tensor shape is:

- `T x Z x Y x X`

The plotting step shows every loaded timepoint, uses the middle z plane of the loaded tensor, and lays the panels out using `n_columns`.

### 5. Dataset Preparation for ML

- [`5_dataset_preparation_ml.ipynb`](5_dataset_preparation_ml.ipynb)

Purpose:

- build a labeled tensor dataset for downstream ML applications
- select a list of mechanisms of action to include as classes
- use water controls as class `0`
- cap the number of compounds per mechanism and the number of tensors per compound
- persist the prepared dataset artifact for reuse in notebooks 6-8

Current flow:

1. build the condition map
2. set:
   `selected_mechanisms`, `selected_concentrations`, `max_compounds_per_action`, `max_tensors_per_compound`
3. set tensor-loading options:
   `output_size`
4. set `dataset_artifact_path` for the saved tensor dataset artifact
5. build the dataset with `build_moa_labeled_tensor_dataset(...)`
6. save it with `save_labeled_tensor_dataset(...)`
7. inspect tensor shapes, label tensor, label map, and metadata table
8. inspect the persisted auxiliary target tensors and label maps for mechanism, compound, concentration, and control
9. optionally reduce the tensors to 2D with `build_dataset_tensor_embedding_2d(...)`
10. visualize the dataset structure with `plot_tensor_embedding_2d(...)`

Dataset conventions:

- label `0` is always `Water`
- each selected mechanism of action is assigned a distinct positive integer label
- `selected_concentrations` controls which treatment concentration bands are included
- metadata includes `original_instance_id`, which identifies each persisted base tensor
- the persisted dataset artifact also stores explicit `compound_labels`, `concentration_labels`, and `is_control` tensors plus their label maps
- water/control examples are collapsed to a dedicated control class for the compound and concentration auxiliary targets
- cached tensor loading and cached selected-TIFF mirroring are used through the same loader path as notebook 4
- PCA is available by default through scikit-learn
- UMAP is also supported, but requires the optional `umap-learn` package in the active environment
- `plot_tensor_embedding_2d(...)` can optionally show an RBF-SVM decision background on the 2D embedding

For model training, use notebook 5 to persist the unaugmented base dataset artifact, then let notebooks 6-8 split by `original_instance_id` before applying any training-only augmentation.

### 6. Train 3D CNN Classifier

- [`6_train_3d_cnn_classifier.ipynb`](6_train_3d_cnn_classifier.ipynb)

Purpose:

- train a simple 3D CNN classifier on the tensor dataset
- treat time as channels and convolve across `z`, `y`, and `x`
- load the persisted base dataset artifact from notebook 5
- split base tensors into train / validation / holdout before any augmentation
- split on `original_instance_id` so all rotated variants of the same source tensor stay together
- augment only the training subset with random XY rotations
- monitor train and validation loss during training
- evaluate the holdout split separately for action, compound, and concentration with classification metrics and confusion matrices
- visualize learned pre-classifier embeddings in 2D

Current flow:

1. set `dataset_artifact_path` to the saved artifact produced by notebook 5
2. load the base dataset with `load_labeled_tensor_dataset(...)`
3. split into train, validation, and holdout subsets on unique `original_instance_id` groups with `split_labeled_tensor_dataset_by_instance(...)`
4. augment only the training subset with `augment_training_tensors_with_rotations(...)`
5. fit `TimeChannel3DCNNClassifier(...)` with explicit validation data
6. inspect train / validation loss with `plot_training_history(...)`
7. inspect target-wise holdout reports with `build_multitask_classification_reports(...)`
8. inspect target-wise holdout confusion matrices with `plot_confusion_matrices(...)`
9. project learned embeddings with `build_tensor_embedding_2d(...)`
10. visualize those learned embeddings with `plot_tensor_embedding_2d(...)`

### 7. Train Commutative CNN Classifier

- [`7_train_commutative_cnn_classifier.ipynb`](7_train_commutative_cnn_classifier.ipynb)

Purpose:

- train the pure-CNN dual-pathway commutative classifier on the persisted tensor dataset
- keep the same leakage-safe split and training-only augmentation rules as notebook 6
- optimize action, compound, and concentration supervision jointly with prototype-consistency and feature-alignment losses
- inspect branch agreement through branch-specific embeddings and loss components
- visualize fused holdout embeddings in 2D

Current flow:

1. set `dataset_artifact_path` to the saved artifact produced by notebook 5
2. load the base dataset with `load_labeled_tensor_dataset(...)`
3. split into train, validation, and holdout subsets on unique `original_instance_id` groups with `split_labeled_tensor_dataset_by_instance(...)`
4. augment only the training subset with `augment_training_tensors_with_rotations(...)`
5. fit `CommutativeCNNClassifier(...)` with explicit validation data
6. inspect train / validation loss with `plot_training_history(...)`
7. inspect target-wise holdout classification metrics and confusion matrices
8. inspect validation and holdout loss components with `evaluate_loss_components(...)`
9. inspect branch-specific embeddings with `transform_branches(...)`
10. project fused holdout embeddings with `build_tensor_embedding_2d(...)`
11. visualize those learned embeddings with `plot_tensor_embedding_2d(...)`

### 8. Train Commutative Transformer Classifier

- [`8_train_commutative_transformer_classifier.ipynb`](8_train_commutative_transformer_classifier.ipynb)

Purpose:

- train the factorized transformer dual-pathway commutative classifier on the persisted tensor dataset
- keep the same leakage-safe split and training-only augmentation rules as notebooks 6 and 7
- optimize action, compound, and concentration supervision jointly with prototype-consistency and feature-alignment losses
- inspect branch agreement through branch-specific embeddings and loss components
- visualize fused holdout embeddings in 2D

Current flow:

1. set `dataset_artifact_path` to the saved artifact produced by notebook 5
2. load the base dataset with `load_labeled_tensor_dataset(...)`
3. split into train, validation, and holdout subsets on unique `original_instance_id` groups with `split_labeled_tensor_dataset_by_instance(...)`
4. augment only the training subset with `augment_training_tensors_with_rotations(...)`
5. fit `CommutativeTransformerClassifier(...)` with explicit validation data
6. inspect train / validation loss with `plot_training_history(...)`
7. inspect target-wise holdout classification metrics and confusion matrices
8. inspect validation and holdout loss components with `evaluate_loss_components(...)`
9. inspect branch-specific embeddings with `transform_branches(...)`
10. project fused holdout embeddings with `build_tensor_embedding_2d(...)`
11. visualize those learned embeddings with `plot_tensor_embedding_2d(...)`

## Typical Workflow

1. Mount the image share using the command in [`mount_command.txt`](mount_command.txt).
2. Use notebook 1 to inspect normalization maps and the normalized classification table.
3. Use notebook 2 to build or inspect run-folder mappings.
4. Use notebook 3 to inspect condition-folder and concentration mappings.
5. Use notebook 4 to load a condition directory into a tensor and visualize sampled timepoints.
6. Use notebook 5 to prepare and persist a labeled tensor dataset artifact, then inspect class structure in 2D.
7. Use notebook 6 to load that artifact and train the simple 3D CNN baseline with leakage-safe splitting by `original_instance_id`.
8. Use notebook 7 to train and inspect the experimental pure-CNN commutative dual-pathway model on the same persisted dataset artifacts.
9. Use notebook 8 to train and inspect the experimental transformer commutative dual-pathway model on the same persisted dataset artifacts.

## Notes

- The generated CSV files are snapshots of the derived mapping tables.
- Notebook outputs may be stale until re-run.
- The utility module expects the Python environment to provide the imaging and dataframe dependencies used by the notebooks, including `pandas`, `tifffile`, and `torch`.
- Notebook 6 additionally expects `scikit-learn` and the PyTorch training stack already used by the tensor utilities.
- The simple 3D CNN in notebook 6 remains the baseline reference pipeline.
- The commutative dual-pathway method in notebook 7 is an experimental implementation of the design described in [docs/method.md](docs/method.md).
- The transformer commutative dual-pathway method in notebook 8 is a second experimental implementation of that same design family.

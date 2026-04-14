# ZebraFish

ZebraFish is a research codebase for preparing labeled 4D calcium-imaging tensor datasets from zebrafish compound-screen experiments and for studying commutative representation learning on those tensors. The central hypothesis is that learned representations should not depend on the order in which valid spatial and temporal abstraction operators are applied, so the repository combines a practical data-preparation pipeline with baseline and dual-pathway models designed to test that idea.

In practical terms, the repository covers four pieces of work:

- workbook-driven compound and mechanism metadata handling
- condition- and run-level resolution of the imaging tree
- tensor loading, downsampling, normalization, caching, and dataset persistence
- supervised and commutative neural models for multitask prediction and representation analysis

## Scope

The repository is organized around two layers:

- data preparation, which links workbook metadata to image folders, resolves concentration-aware conditions, materializes `T x Z x Y x X` tensors, and persists reusable dataset artifacts
- modeling, which trains multitask classifiers on mechanism, compound, and concentration targets, compares a baseline architecture against experimental commutative models, and evaluates holdout performance and learned embeddings

## Documentation

- Start with [docs/introduction.md](docs/introduction.md) for the conceptual overview, then use the documents below as the implementation reference set.
- [docs/introduction.md](docs/introduction.md)
  High-level introduction to the commutative representation-learning idea.
- [docs/method.md](docs/method.md)
  Method statement, theoretical objective, and implementation status.
- [docs/architecture.md](docs/architecture.md)
  Detailed architecture reference for the three implemented model families.
- [docs/preprocessing.md](docs/preprocessing.md)
  Tensor-loading, normalization, caching, and dataset-preparation conventions.
- [docs/data_artifacts.md](docs/data_artifacts.md)
  Generated CSV artifacts and their meanings.
- [docs/figures.md](docs/figures.md)
  Standalone figures referenced by the method documentation.

## Repository Layout

### Source Code

- [src/notebook_utils.py](src/notebook_utils.py)
  Workbook loading, folder discovery, name normalization, and exploratory notebook utilities.
- [src/tensor_utils.py](src/tensor_utils.py)
  TIFF loading, tensor materialization, downsampling, drift normalization, caching, dataset building, and embedding helpers.
- [src/ml.py](src/ml.py)
  Public façade for the training and evaluation API.

### Internal ML Modules

The public ML surface in [src/ml.py](src/ml.py) is implemented internally as:

- [src/models/common.py](src/models/common.py)
  Shared tensor validation, target preparation, and estimator mixin behavior.
- [src/models/configs.py](src/models/configs.py)
  Typed configuration dataclasses for models, optimization, and loss weights.
- [src/models/backbones_cnn.py](src/models/backbones_cnn.py)
  Baseline 3D CNN and pure-CNN commutative backbones.
- [src/models/backbones_transformer.py](src/models/backbones_transformer.py)
  Transformer patch embedding and commutative transformer backbones.
- [src/models/estimators.py](src/models/estimators.py)
  Public scikit-style estimators.
- [src/training/data.py](src/training/data.py)
  Leakage-safe splitting and training-only augmentation.
- [src/training/losses.py](src/training/losses.py)
  Shared commutative and auxiliary loss helpers.
- [src/training/loop.py](src/training/loop.py)
  Shared optimization, early stopping, batching, and inference utilities.
- [src/training/reporting.py](src/training/reporting.py)
  Classification reports, confusion matrices, loss plotting, and embedding projection utilities.
- [src/training/workflow.py](src/training/workflow.py)
  Thin experiment orchestration helpers for notebooks.

## Data Inputs

### Workbook

- [compounds (MJW V2).xlsx](compounds%20(MJW%20V2).xlsx)
  Source workbook for compound names, compound classes, mechanisms of action, and exposure-condition text.

### Imaging Tree

- [mount_command.txt](mount_command.txt)
  Example command for mounting the imaging share at `/mnt/tyler`.

The notebooks assume the imaging tree is accessible locally through the mounted path referenced in the generated CSVs.

## Naming Conventions

Normalized outputs use:

- `compound`
  standardized compound name with workbook markers removed
- `mechanism_of_action`
  normalized mnemonic label such as `GABAAR_Antagonist` or `NMDAR_Activation`

Raw workbook strings are still retained internally when needed for folder matching.

## Core APIs

### Tensor and Dataset Utilities

Key tensor/data helpers include:

- `load_image_condition_tensor(...)`
- `build_moa_labeled_tensor_dataset(...)`
- `save_labeled_tensor_dataset(...)`
- `load_labeled_tensor_dataset(...)`
- `build_dataset_tensor_embedding_2d(...)`
- `plot_tensor_embedding_2d(...)`

Dataset artifacts persist:

- `tensors`
- `labels`
- `compound_labels`
- `concentration_labels`
- `is_control`
- `metadata`
- `label_map`
- `compound_label_map`
- `concentration_label_map`

### Model APIs

Implemented estimators:

- `TimeChannel3DCNNClassifier`
- `CommutativeCNNClassifier`
- `CommutativeTransformerClassifier`

Common configuration objects:

- `TimeChannel3DCNNConfig`
- `CommutativeCNNConfig`
- `CommutativeTransformerConfig`
- `OptimizationConfig`
- `LossWeightConfig`

Common experiment helpers:

- `prepare_multitask_experiment_data(...)`
- `fit_estimator_on_experiment(...)`
- `display_holdout_evaluation(...)`
- `persist_experiment_artifacts(...)`

Current estimator behavior:

- `predict(...)` and `predict_proba(...)` return dictionaries keyed by target:
  - `action`
  - `compound`
  - `concentration`
- `transform(...)` returns the shared embedding
- commutative models also expose `transform_branches(...)`

## Dataset and Training Conventions

The main training invariants are:

- tensors are handled in canonical `T x Z x Y x X` order
- train, validation, and holdout splits are assigned on unaugmented base tensors
- split grouping uses `original_instance_id` to avoid leakage across derived views
- random rotation augmentation is applied only to the training subset
- water/control examples remain mechanism label `0` and are collapsed to dedicated control classes for the auxiliary heads

Repo-local caches are now retention-managed rather than unbounded:

- `.tiff_cache` uses a default `30G` LRU budget
- `.tensor_cache` uses a default `5G` LRU budget
- `.dataset_cache` uses a default `10G` LRU budget
- cache eviction also enforces a default `15G` filesystem free-space floor
- the dataset referenced by `artifacts/current_dataset.json` is pinned against dataset-cache eviction

See [docs/preprocessing.md](docs/preprocessing.md) for the full data-pipeline specification.

## Notebooks

The repository provides eight notebooks:

1. [1_compound_class_moa.ipynb](1_compound_class_moa.ipynb)
   Inspect normalized compound and mechanism metadata.
2. [2_compound_image_mapping.ipynb](2_compound_image_mapping.ipynb)
   Build run-level mappings from compounds to image folders.
3. [3_compound_image_concentration_mapping.ipynb](3_compound_image_concentration_mapping.ipynb)
   Expand run mappings into condition-level mappings with concentration parsing.
4. [4_view_midz_time_samples.ipynb](4_view_midz_time_samples.ipynb)
   Load and inspect sampled tensor slices from individual conditions.
5. [5_dataset_preparation_ml.ipynb](5_dataset_preparation_ml.ipynb)
   Build and persist labeled tensor datasets for ML.
6. [6_train_3d_cnn_classifier.ipynb](6_train_3d_cnn_classifier.ipynb)
   Train the baseline time-as-channels 3D CNN.
7. [7_train_commutative_cnn_classifier.ipynb](7_train_commutative_cnn_classifier.ipynb)
   Train the experimental pure-CNN commutative model.
8. [8_train_commutative_transformer_classifier.ipynb](8_train_commutative_transformer_classifier.ipynb)
   Train the experimental transformer commutative model.

Notebooks 6 to 8 are intentionally thin and delegate split preparation, fitting, reporting, plotting, and artifact persistence to shared utilities in `src.ml`.

## Typical Workflow

1. Mount the imaging share using [mount_command.txt](mount_command.txt).
2. Use notebooks 1 to 3 to validate metadata normalization and folder mapping.
3. Use notebook 4 to inspect raw condition tensors.
4. Use notebook 5 to build and persist a labeled tensor dataset artifact.
5. Use notebook 6 to establish the supervised baseline.
6. Use notebooks 7 and 8 to compare the experimental commutative models against that baseline.

## Testing

The repository includes a small `unittest` suite in [tests](tests) covering:

- estimator API smoke tests
- leakage-safe split behavior
- experiment preparation, evaluation, and persistence helpers

Run it with:

```bash
python -m unittest discover -s tests -v
```

## Notes

- Notebook outputs may be stale until re-executed.
- The codebase expects the Python environment to provide the imaging and ML dependencies used by the notebooks, including `pandas`, `tifffile`, `torch`, and `scikit-learn`.
- The 3D CNN remains the baseline reference model.
- The two commutative models are implemented end to end but should still be treated as experimental research models.

# ZebraFish

This repo contains a small notebook-and-utility workflow for linking a compound workbook to zebrafish imaging folders, deriving concentration-aware condition maps, and inspecting image time series.

The workflow is built around:

- a workbook of compounds, classes, and mechanisms of action
- a mounted imaging directory tree under `/mnt/tyler`
- shared utility code in [`zebrafish_notebook_utils.py`](/home/fabrizio/code/ZebraFish/zebrafish_notebook_utils.py)
- dedicated tensor loading code in [`zebrafish_tensor_utils.py`](/home/fabrizio/code/ZebraFish/zebrafish_tensor_utils.py)
- dedicated ML/training code in [`zebrafish_ml.py`](/home/fabrizio/code/ZebraFish/zebrafish_ml.py)
- six notebooks for exploration, dataset preparation, and model training

## Inputs

### Workbook

- [`compounds (MJW V2).xlsx`](/home/fabrizio/code/ZebraFish/compounds%20(MJW%20V2).xlsx)
  Source workbook used to derive compound names, compound classes, mechanisms of action, and exposure-condition text.

### Imaging Mount

- [`mount_command.txt`](/home/fabrizio/code/ZebraFish/mount_command.txt)
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

- [`zebrafish_notebook_utils.py`](/home/fabrizio/code/ZebraFish/zebrafish_notebook_utils.py)
  Shared logic for:
  - loading workbook data
  - normalizing compound names
  - mapping mechanisms of action to mnemonic aliases
  - discovering candidate image run folders
  - building run-level and condition-level mapping tables
  - parsing concentration labels from folder names
  - plotting sampled mid-z time slices from a tensor

## Tensor Loading Module

- [`zebrafish_tensor_utils.py`](/home/fabrizio/code/ZebraFish/zebrafish_tensor_utils.py)
  Dedicated image tensor materialization code.

This module is intended to grow with future preprocessing. It currently handles:

- reading TIFF series
- normalizing axes to `T x Z x Y x X`
- stacking all files inside an `image_condition_dir` into one torch tensor
- deterministic downsampling across `T`, `Z`, `Y`, and `X`
- time downsampling before file reads
- z downsampling during per-file load when possible
- optional post-load intensity-drift normalization using a LOESS-style smoother on the per-timepoint global mean
- repo-local tensor caching in `.tensor_cache/`
- repo-local TIFF mirroring for selected timepoints in `.tiff_cache/`
- tensor-level XY rotation augmentation
- labeled dataset assembly for downstream ML workflows

Downsampling uses explicit target sizes via `output_size=(T, Z, Y, X)`.

Examples:

- `(10, 1, 32, 32)`
  Load 10 timepoints, keep only the middle z slice, and keep a `32 x 32` sampled image grid.
- `(3, None, None, None)`
  Load 3 timepoints: first, middle, and last.

When a requested size is odd, the exact middle index is always included. When a requested size is `1`, the middle index is used.

By default, `load_image_condition_tensor()` also applies a global intensity-drift correction:

- it computes the global mean intensity for each loaded timepoint
- fits a LOESS-style smooth trend across time
- subtracts the smoothed drift while preserving the overall mean level

This correction can be disabled with `normalize_global_drift=False`.

Loaded tensors are cached in:

- [`.tensor_cache/`](/home/fabrizio/code/ZebraFish/.tensor_cache)

The cache key includes the selected files, file mtimes and sizes, `output_size`, and normalization settings. The cache directory is ignored by git.

Caching is enabled by default and can be disabled with `use_cache=False`.

Selected TIFF files can also be mirrored locally in:

- [`.tiff_cache/`](/home/fabrizio/code/ZebraFish/.tiff_cache)

Only the timepoint files selected after time downsampling are copied. The mirrored cache preserves the source path structure under the cache root, for example:

- source: `/mnt/tyler/Matt Winter/BRAIN IMAGES BACKUP/.../TL001.ome.tiff`
- cached: `.tiff_cache/mnt/tyler/Matt Winter/BRAIN IMAGES BACKUP/.../TL001.ome.tiff`

This cache is enabled by default and can be disabled with `use_tiff_cache=False`.

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

- [`zebrafish_ml.py`](/home/fabrizio/code/ZebraFish/zebrafish_ml.py)
  Simple model-training utilities for tensor classification experiments.

Current contents:

- `Zebrafish3DCNNClassifier`
  A scikit-style estimator built on a simple 3D CNN that treats time as channels and supports configurable convolution kernel sizes, strides, and pooling separately for `z` and `xy`.
- `augment_training_tensors_with_rotations()`
  Training-only XY rotation augmentation helper.
- `plot_training_history()`
  Plot train and validation loss by epoch.
- `plot_confusion_matrices()`
  Plot both absolute-count and row-fraction confusion matrices.

Important leakage rule:

- split train / validation / holdout on the unaugmented base tensors first
- apply random rotation augmentation only to the training subset afterward

This rule is enforced in notebook 6 and should be preserved in future model-training workflows.

## Generated CSV Files

### `compound_image_run_map.csv`

- [`compound_image_run_map.csv`](/home/fabrizio/code/ZebraFish/compound_image_run_map.csv)
  Run-level mapping between normalized compounds and image run directories.

Meaning of the main columns:

- `compound`: normalized compound name
- `compound_class`: compound class from the workbook
- `mechanism_of_action`: normalized mechanism mnemonic
- `image_run_dir`: absolute path to the matched image run folder
- `image_run_dir_relative`: run path relative to the image root
- `source_batch`: top-level batch folder
- `dir_name`: original run folder name
- `folder_status`: inferred folder usability, typically `active` or `do_not_use`

Each row represents one matched image run directory for one compound.

### `compound_image_index.csv`

- [`compound_image_index.csv`](/home/fabrizio/code/ZebraFish/compound_image_index.csv)
  Compound-level summary of the run map.

Meaning of the main columns:

- `n_image_run_dirs`: total matched run directories
- `n_active_image_run_dirs`: matched run directories not flagged as do-not-use
- `image_run_dirs`: pipe-delimited list of all matched run directories
- `active_image_run_dirs`: pipe-delimited list of active run directories only

Each row summarizes one normalized `compound` / `compound_class` / `mechanism_of_action` combination.

### `compound_image_condition_map.csv`

- [`compound_image_condition_map.csv`](/home/fabrizio/code/ZebraFish/compound_image_condition_map.csv)
  Condition-level mapping derived from the child folders inside each image run directory.

Meaning of the main columns:

- inherited run metadata:
  `compound`, `compound_class`, `mechanism_of_action`, `image_run_dir`, `folder_status`
- condition directory fields:
  `image_condition_dir`, `image_condition_dir_name`, `condition_folder_status`
- concentration/exposure fields:
  `exposure_conditions`, `condition_kind`, `concentration_value`, `concentration_unit`,
  `concentration_value_uM`, `concentration_label`
- ranking/grouping fields:
  `concentration_rank_in_run`, `concentration_band`, `n_treatment_concentrations_in_run`

`concentration_band` is a coarse within-run label such as `control`, `low`, `mid`, or `high`.

Each row represents one condition folder under one run folder.

### `compound_image_condition_index.csv`

- [`compound_image_condition_index.csv`](/home/fabrizio/code/ZebraFish/compound_image_condition_index.csv)
  Summary table of condition folders grouped by normalized compound and concentration band.

Meaning of the main columns:

- `concentration_band`: grouped concentration label within run
- `n_condition_dirs`: total condition folders in the group
- `n_active_condition_dirs`: active condition folders only
- `condition_dirs`: pipe-delimited list of condition directory paths

Each row summarizes a normalized `compound` / `compound_class` / `mechanism_of_action` / `concentration_band` group.

## Notebooks

### 1. Compound Class and Mechanism of Action

- [`1_compound_class_moa.ipynb`](/home/fabrizio/code/ZebraFish/1_compound_class_moa.ipynb)

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

- [`2_compound_image_mapping.ipynb`](/home/fabrizio/code/ZebraFish/2_compound_image_mapping.ipynb)

Purpose:

- build the run-level mapping from normalized compounds to matched image run folders
- build a summarized run index for quick inspection

Main outputs in the notebook:

- `compound_image_map_df`
- `compound_image_index_df`

This notebook is the run-folder linkage step.

### 3. Compound Image Concentration Mapping

- [`3_compound_image_concentration_mapping.ipynb`](/home/fabrizio/code/ZebraFish/3_compound_image_concentration_mapping.ipynb)

Purpose:

- expand the run-level map into condition-level folders
- parse concentration values from condition folder names
- build a grouped condition index

Main outputs in the notebook:

- `compound_image_condition_map_df`
- `compound_image_condition_index_df`

This notebook is the concentration/condition parsing step.

### 4. View Mid-Z Time Samples

- [`4_view_midz_time_samples.ipynb`](/home/fabrizio/code/ZebraFish/4_view_midz_time_samples.ipynb)

Purpose:

- select a normalized compound and concentration grouping
- explicitly materialize the image data as a torch tensor
- plot all loaded time slices from the middle z plane

Current flow:

1. build the condition map
2. set the selection inputs:
   `selected_compound`, `selector_column`, `selected_concentration`, `only_active`, `selected_condition_index`
3. set the tensor downsampling target with `tensor_size = (T, Z, Y, X)`
4. load `condition_tensor = load_image_condition_tensor(...)` directly from `condition_df` plus the selection inputs
5. inspect `condition_tensor.shape`
6. plot all loaded timepoints with `plot_midz_time_slices_from_tensor(condition_tensor, n_columns=...)`

The tensor shape is:

- `T x Z x Y x X`

The plotting step shows every loaded timepoint, uses the middle z plane of the loaded tensor, and lays the panels out using `n_columns`.

### 5. Dataset Preparation for ML

- [`5_dataset_preparation_ml.ipynb`](/home/fabrizio/code/ZebraFish/5_dataset_preparation_ml.ipynb)

Purpose:

- build a labeled tensor dataset for downstream ML applications
- select a list of mechanisms of action to include as classes
- use water controls as class `0`
- cap the number of compounds per mechanism and the number of tensors per compound
- optionally augment examples with random XY rotations for exploratory datasets and embeddings

Current flow:

1. build the condition map
2. set:
   `selected_mechanisms`, `selected_concentrations`, `max_compounds_per_action`, `max_tensors_per_compound`
3. set tensor-loading and augmentation options:
   `output_size`, `num_random_rotations`, `rotation_range_degrees`
4. build the dataset with `build_moa_labeled_tensor_dataset(...)`
5. inspect tensor shapes, label tensor, label map, and metadata table
6. optionally reduce the tensors to 2D with `build_tensor_embedding_2d(...)`
7. visualize the class structure with `plot_tensor_embedding_2d(...)`

Dataset conventions:

- label `0` is always `Water`
- each selected mechanism of action is assigned a distinct positive integer label
- `selected_concentrations` controls which treatment concentration bands are included
- cached tensor loading and cached selected-TIFF mirroring are used through the same loader path as notebook 4
- PCA is available by default through scikit-learn
- UMAP is also supported, but requires the optional `umap-learn` package in the active environment
- `plot_tensor_embedding_2d(...)` can optionally show an RBF-SVM decision background on the 2D embedding

For model training, do not use notebook 5 augmentation before splitting. Augment only after the split to avoid leakage between related rotated views of the same source tensor.

### 6. Train 3D CNN Classifier

- [`6_train_3d_cnn_classifier.ipynb`](/home/fabrizio/code/ZebraFish/6_train_3d_cnn_classifier.ipynb)

Purpose:

- train a simple 3D CNN classifier on the tensor dataset
- treat time as channels and convolve across `z`, `y`, and `x`
- split base tensors into train / validation / holdout before any augmentation
- augment only the training subset with random XY rotations
- monitor train and validation loss during training
- evaluate the holdout split with classification metrics and confusion matrices
- visualize learned pre-classifier embeddings in 2D

Current flow:

1. build the base dataset with `num_random_rotations=0`
2. split into train, validation, and holdout subsets on the unaugmented tensors
3. augment only the training subset with `augment_training_tensors_with_rotations(...)`
4. fit `Zebrafish3DCNNClassifier(...)` with explicit validation data
5. inspect train / validation loss with `plot_training_history(...)`
6. inspect holdout confusion matrices with `plot_confusion_matrices(...)`
7. project learned embeddings with `build_tensor_embedding_2d(...)`
8. visualize those learned embeddings with `plot_tensor_embedding_2d(...)`

## Typical Workflow

1. Mount the image share using the command in [`mount_command.txt`](/home/fabrizio/code/ZebraFish/mount_command.txt).
2. Use notebook 1 to inspect normalization maps and the normalized classification table.
3. Use notebook 2 to build or inspect run-folder mappings.
4. Use notebook 3 to inspect condition-folder and concentration mappings.
5. Use notebook 4 to load a condition directory into a tensor and visualize sampled timepoints.
6. Use notebook 5 to prepare labeled tensor datasets and inspect class structure in 2D.
7. Use notebook 6 to train and evaluate the simple 3D CNN baseline without augmentation leakage.

## Notes

- The generated CSV files are snapshots of the derived mapping tables.
- Notebook outputs may be stale until re-run.
- The utility module expects the Python environment to provide the imaging and dataframe dependencies used by the notebooks, including `pandas`, `tifffile`, and `torch`.
- Notebook 6 additionally expects `scikit-learn` and the PyTorch training stack already used by the tensor utilities.

# ZebraFish

This repo contains a small notebook-and-utility workflow for linking a compound workbook to zebrafish imaging folders, deriving concentration-aware condition maps, and inspecting image time series.

The workflow is built around:

- a workbook of compounds, classes, and mechanisms of action
- a mounted imaging directory tree under `/mnt/tyler`
- shared utility code in [`zebrafish_notebook_utils.py`](/home/fabrizio/code/ZebraFish/zebrafish_notebook_utils.py)
- dedicated tensor loading code in [`zebrafish_tensor_utils.py`](/home/fabrizio/code/ZebraFish/zebrafish_tensor_utils.py)
- four notebooks for exploration and export

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

Downsampling uses explicit target sizes via `output_size=(T, Z, Y, X)`.

Examples:

- `(10, 1, 32, 32)`
  Load 10 timepoints, keep only the middle z slice, and keep a `32 x 32` sampled image grid.
- `(3, None, None, None)`
  Load 3 timepoints: first, middle, and last.

When a requested size is odd, the exact middle index is always included. When a requested size is `1`, the middle index is used.

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

## Typical Workflow

1. Mount the image share using the command in [`mount_command.txt`](/home/fabrizio/code/ZebraFish/mount_command.txt).
2. Use notebook 1 to inspect normalization maps and the normalized classification table.
3. Use notebook 2 to build or inspect run-folder mappings.
4. Use notebook 3 to inspect condition-folder and concentration mappings.
5. Use notebook 4 to load a condition directory into a tensor and visualize sampled timepoints.

## Notes

- The generated CSV files are snapshots of the derived mapping tables.
- Notebook outputs may be stale until re-run.
- The utility module expects the Python environment to provide the imaging and dataframe dependencies used by the notebooks, including `pandas`, `tifffile`, and `torch`.

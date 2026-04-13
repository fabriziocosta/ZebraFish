# Data Artifacts

This document summarizes the generated CSV artifacts produced by the mapping and indexing notebooks. Repository workflow and notebook ordering remain documented in [README.md](../README.md).

## `compound_image_run_map.csv`

- [`compound_image_run_map.csv`](../compound_image_run_map.csv)
  Run-level mapping between normalized compounds and image run directories.

Main columns:

- `compound`: normalized compound name
- `compound_class`: compound class from the workbook
- `mechanism_of_action`: normalized mechanism mnemonic
- `image_run_dir`: absolute path to the matched image run folder
- `image_run_dir_relative`: run path relative to the image root
- `source_batch`: top-level batch folder
- `dir_name`: original run folder name
- `folder_status`: inferred folder usability, typically `active` or `do_not_use`

Each row represents one matched image run directory for one compound.

## `compound_image_index.csv`

- [`compound_image_index.csv`](../compound_image_index.csv)
  Compound-level summary of the run map.

Main columns:

- `n_image_run_dirs`: total matched run directories
- `n_active_image_run_dirs`: matched run directories not flagged as do-not-use
- `image_run_dirs`: pipe-delimited list of all matched run directories
- `active_image_run_dirs`: pipe-delimited list of active run directories only

Each row summarizes one normalized `compound` / `compound_class` / `mechanism_of_action` combination.

## `compound_image_condition_map.csv`

- [`compound_image_condition_map.csv`](../compound_image_condition_map.csv)
  Condition-level mapping derived from the child folders inside each image run directory.

Main columns:

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

## `compound_image_condition_index.csv`

- [`compound_image_condition_index.csv`](../compound_image_condition_index.csv)
  Summary table of condition folders grouped by normalized compound and concentration band.

Main columns:

- `concentration_band`: grouped concentration label within run
- `n_condition_dirs`: total condition folders in the group
- `n_active_condition_dirs`: active condition folders only
- `condition_dirs`: pipe-delimited list of condition directory paths

Each row summarizes a normalized `compound` / `compound_class` / `mechanism_of_action` / `concentration_band` group.

# Preprocessing

This document describes the image tensor materialization and dataset-preparation conventions used by the repository. Workflow orchestration and notebook ordering remain documented in [README.md](../README.md).

## 1. Tensor materialization

The image tensor pipeline is implemented in [`zebrafish_tensor_utils.py`](../zebrafish_tensor_utils.py).

The current preprocessing path handles:

- reading TIFF series
- normalizing axes to `T x Z x Y x X`
- stacking all files inside an `image_condition_dir` into one torch tensor
- deterministic downsampling across `T`, `Z`, `Y`, and `X`
- time downsampling before file reads
- z downsampling during per-file load when possible
- optional post-load intensity-drift normalization using a LOESS-style smoother on the per-timepoint global mean
- repo-local tensor caching in [`.tensor_cache/`](../.tensor_cache)
- repo-local TIFF mirroring for selected timepoints in [`.tiff_cache/`](../.tiff_cache)
- labeled dataset assembly for downstream ML workflows

Downsampling uses explicit target sizes via `output_size=(T, Z, Y, X)`.

Examples:

- `(10, 1, 32, 32)`
  Load 10 timepoints, keep only the middle z slice, and keep a `32 x 32` sampled image grid.
- `(3, None, None, None)`
  Load 3 timepoints: first, middle, and last.

When a requested size is odd, the exact middle index is always included. When a requested size is `1`, the middle index is used.

## 2. Intensity-drift normalization

By default, `load_image_condition_tensor()` applies a global intensity-drift correction:

- it computes the global mean intensity for each loaded timepoint
- fits a LOESS-style smooth trend across time
- subtracts the smoothed drift while preserving the overall mean level

This correction can be disabled with `normalize_global_drift=False`.

## 3. Caching

Loaded tensors are cached in [`.tensor_cache/`](../.tensor_cache).

The cache key includes the selected files, file mtimes and sizes, `output_size`, and normalization settings. The cache directory is ignored by git.

Caching is enabled by default and can be disabled with `use_cache=False`.

Selected TIFF files can also be mirrored locally in [`.tiff_cache/`](../.tiff_cache).

Only the timepoint files selected after time downsampling are copied. The mirrored cache preserves the source path structure under the cache root, for example:

- source: `/mnt/tyler/Matt Winter/BRAIN IMAGES BACKUP/.../TL001.ome.tiff`
- cached: `.tiff_cache/mnt/tyler/Matt Winter/BRAIN IMAGES BACKUP/.../TL001.ome.tiff`

This cache is enabled by default and can be disabled with `use_tiff_cache=False`.

## 4. Tensor and dataset helpers

Tensor materialization helpers:

- `load_tiff_as_tzyx()`
- `downsample_tzyx()`
- `load_image_condition_tensor()`

Dataset-preparation helpers:

- `rotate_tensor_xy()`
- `build_moa_labeled_tensor_dataset()`
- `build_tensor_embedding_2d()`
- `plot_tensor_embedding_2d()`

`plot_tensor_embedding_2d()` can optionally overlay a low-alpha class-region background by fitting an RBF-kernel SVM directly on the 2D embedding coordinates.

## 5. Dataset preparation and split conventions

The repository dataset-preparation flow used by notebooks 5 and 6 assumes:

- label `0` is always `Water`
- each selected mechanism of action is assigned a distinct positive integer label
- `selected_concentrations` controls which treatment concentration bands are included
- metadata includes `original_instance_id`, which is stable across all random rotations derived from the same base tensor
- cached tensor loading and cached selected-TIFF mirroring are used through the same loader path as notebook 4
- exploratory dataset assembly may include random XY rotations

For evaluation and model training:

- the base dataset artifact is prepared first
- train, validation, and holdout splits are then performed on `original_instance_id`
- training-only augmentation is applied only after splitting

This keeps all rotated views of the same source tensor together and avoids leakage.

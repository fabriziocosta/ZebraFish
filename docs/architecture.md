# Architecture

This document describes the three model architectures currently represented in the repository:

1. the baseline 3D CNN classifier
2. the pure-CNN commutative dual-pathway classifier
3. the transformer-based commutative dual-pathway classifier

Theory and objective-level motivation remain in [method.md](method.md). Preprocessing and dataset conventions remain in [preprocessing.md](preprocessing.md).

## 1. Baseline: time-as-channels 3D CNN

Implemented as `TimeChannel3DCNNClassifier` in [`src/ml.py`](../src/ml.py) and used in notebook 6.

### 1.1 Purpose

This is the repository baseline. It collapses the temporal axis into the channel dimension and applies a conventional `Conv3d` stack over the remaining `z`, `y`, and `x` axes.

The baseline is intentionally simple:

- it is cheap relative to the commutative models
- it provides a direct supervised reference point
- it tests whether the task is already learnable without explicit operator-factorization structure

### 1.2 Input and output

The input tensor is

$$
X \in \mathbb{R}^{N \times T' \times Z' \times H' \times W'}.
$$

The model interprets $T'$ as the channel count for `Conv3d`, so the first convolution sees an input with

$$
\text{in\_channels} = T'.
$$

The output is:

- a learned embedding in $\mathbb{R}^d$
- action logits in $\mathbb{R}^C$
- optional compound logits in $\mathbb{R}^{C_{\text{compound}}}$
- optional concentration logits in $\mathbb{R}^{C_{\text{conc}}}$

where $d$ is `embedding_dim` and $C$ is the number of classes.

### 1.3 Computation

The forward path is

$$
X \xrightarrow{\text{3D CNN backbone}} H
\xrightarrow{\text{global average pool}} h
\xrightarrow{\text{projection MLP}} z
\xrightarrow{\text{linear head}} \text{logits}.
$$

In implementation terms:

- `Conv3d` blocks process spatial volumes with time treated as channels
- `AdaptiveAvgPool3d((1,1,1))` collapses the remaining spatial dimensions
- a small linear projection produces the embedding
- one linear head produces action logits
- optional auxiliary heads produce compound and concentration logits from the same embedding

### 1.4 Example tensor flow

Using

$$
(T', Z', H', W') = (10, 3, 128, 128),
$$

an example flow is

```text
Input: (N, 10, 3, 128, 128)
Conv3d stack:
    -> (N, 8, 3, 64, 64)
    -> (N, 16, 3, 32, 32)
    -> (N, 24, 3, 16, 16)
Global average pool:
    -> (N, 24, 1, 1, 1)
Flatten:
    -> (N, 24)
Projection:
    -> (N, d)
Classifier:
    -> (N, C)
```

The exact intermediate shapes depend on the configured pooling and stride values.

### 1.5 Main hyperparameters

- `conv_channels`
- `kernel_size_z`, `kernel_size_xy`
- `stride_z`, `stride_xy`
- `pool_kernel_z`, `pool_kernel_xy`
- `embedding_dim`
- `dropout`

Training controls shared with the other estimators:

- `batch_size`
- `epochs`
- `learning_rate`
- `weight_decay`
- `early_stopping_patience`
- `scheduler_patience`

### 1.6 Strengths and limitations

Strengths:

- simple and stable
- low implementation complexity
- useful baseline for overfitting and data-quality diagnosis

Limitations:

- temporal order is only implicit in the channel layout
- no explicit factorization into spatial-first vs temporal-first views
- cannot express the commutative objective directly

### 1.7 Multi-head supervision

The current implementation can train the shared embedding against three supervised targets at once:

- action
- compound
- concentration

Controls remain action label `0` and are collapsed to dedicated control classes for the auxiliary compound and concentration heads.

At inference time, `predict(...)` and `predict_proba(...)` return dictionaries keyed by those targets rather than a single array.

## 2. Pure-CNN commutative dual-pathway model

Implemented as `CommutativeCNNClassifier` in [`src/ml.py`](../src/ml.py) and used in notebook 7.

### 2.1 Purpose

This is the first explicit implementation of the method in [method.md](method.md). It builds two branch embeddings from the same 4D tensor:

- spatial then temporal
- temporal then spatial

The training objective encourages those two branches to agree semantically.

### 2.2 Branch structure

Let

$$
\tilde X \in \mathbb{R}^{T' \times Z' \times H' \times W'}.
$$

The model computes two embeddings:

$$
z^{ST}, z^{TS} \in \mathbb{R}^d.
$$

The fused representation is

$$
z = \frac{1}{2}(z^{ST} + z^{TS}).
$$

The network then produces:

- branch prototype logits from $z^{ST}$ and $z^{TS}$
- action logits from the fused embedding $z$
- optional compound logits from the fused embedding $z$
- optional concentration logits from the fused embedding $z$

### 2.3 Spatial-then-temporal branch

At each timepoint, the model extracts a 3D volume and applies a shared 3D CNN:

$$
V_t \in \mathbb{R}^{Z' \times H' \times W'}
\xrightarrow{f_S}
e_t^{(S)} \in \mathbb{R}^{d_s}.
$$

Those per-frame embeddings are stacked across time and processed by a temporal 1D CNN:

$$
[e_1^{(S)}, \dots, e_{T'}^{(S)}]
\xrightarrow{f_T}
z^{ST} \in \mathbb{R}^d.
$$

Implementation stages:

1. reshape `(N, T', Z', H', W')` to `(N*T', 1, Z', H', W')`
2. apply a shared 3D CNN frame encoder
3. global-pool each frame to one vector
4. reshape to a temporal sequence
5. run a temporal `Conv1d` stack
6. temporal-average pool and project to `embedding_dim`

### 2.4 Temporal-then-spatial branch

The model first partitions space into coarse patches and extracts one temporal signal per patch. For each patch location $p$:

$$
s_p \in \mathbb{R}^{T'}
\xrightarrow{g_T}
u_p^{(T)} \in \mathbb{R}^{d_t}.
$$

Those patch embeddings are reassembled on a 3D lattice and aggregated by a second 3D CNN:

$$
U^{(T)} \in \mathbb{R}^{Z_p \times H_p \times W_p \times d_t}
\xrightarrow{g_S}
z^{TS} \in \mathbb{R}^d.
$$

Implementation stages:

1. average-pool the input into coarse `(z, y, x)` patches
2. reshape to one temporal sequence per patch
3. run a shared temporal `Conv1d` encoder on each patch sequence
4. pool each patch sequence to one vector
5. reshape those vectors back to a 3D patch lattice
6. run a 3D CNN spatial aggregator
7. global-pool and project to `embedding_dim`

### 2.5 Example tensor flow

Using

$$
(T', Z', H', W') = (125, 8, 100, 150),
$$

with spatial patching

$$
(p_z, p_y, p_x) = (1, 10, 10),
$$

the ST branch can look like

```text
Input: (N, 125, 8, 100, 150)
Reshape: (N*125, 1, 8, 100, 150)
3D CNN frame encoder
Global average pool
Reshape: (N, d_s, 125)
Temporal 1D CNN
Temporal average pool
Projection -> z^(ST): (N, d)
```

The TS branch can look like

```text
Input: (N, 125, 8, 100, 150)
Spatial patch pooling -> (N, 125, 8, 10, 15)
Patch sequences -> (N*1200, 1, 125)
Temporal 1D CNN per patch
Temporal average pool
Reshape to lattice -> (N, d_t, 8, 10, 15)
3D CNN spatial aggregator
Global average pool
Projection -> z^(TS): (N, d)
```

### 2.6 Losses

The model uses three loss components:

1. supervised classification loss on the fused logits
2. prototype-consistency loss between the two branches
3. feature-alignment loss between $z^{ST}$ and $z^{TS}$

The implemented total loss is

$$
\mathcal L =
\mathcal L_{\text{cls}}
 + \alpha \mathcal L_{\text{swap}}
 + \beta \mathcal L_{\text{feat}}.
$$

The weights correspond to:

- `consistency_weight = \alpha`
- `feature_weight = \beta`

In the repository implementation, optional auxiliary cross-entropy losses for compound and concentration classification are added on top of this total objective using the same fused embedding.

At inference time, the estimator returns target-keyed prediction and probability dictionaries for `action`, `compound`, and `concentration`.

### 2.7 Main hyperparameters

Spatial-first branch:

- `spatial_conv_channels`
- `spatial_kernel_size_z`, `spatial_kernel_size_xy`
- `spatial_pool_kernel_z`, `spatial_pool_kernel_xy`
- `temporal_st_channels`
- `temporal_st_kernel_sizes`

Temporal-first branch:

- `patch_size_z`
- `patch_size_xy`
- `temporal_ts_channels`
- `temporal_ts_kernel_sizes`
- `spatial_agg_channels`

Shared heads:

- `embedding_dim`
- `num_prototypes`
- `prototype_temperature`
- `dropout`

### 2.8 Strengths and limitations

Strengths:

- directly implements the commutative idea
- cheaper than a transformer-based factorization
- exposes branch-specific diagnostics through `transform_branches(...)`

Limitations:

- patch pooling in the TS branch is coarse by construction
- long-range interactions are mediated only through convolution and pooling
- branch design choices are hand-structured rather than token-based

## 3. Transformer-based commutative dual-pathway model

Implemented as `CommutativeTransformerClassifier` in [`src/ml.py`](../src/ml.py) and used in notebook 8.

### 3.1 Purpose

This model keeps the same commutative training objective as the pure-CNN version, but replaces the branch aggregators with factorized transformer stacks.

It is a more expressive model family, but also substantially heavier. Token count is the critical engineering constraint.

### 3.2 Spatial-then-temporal transformer branch

For each timepoint, a 3D frame is patchified into non-overlapping volumetric tokens:

$$
V_t \mapsto \{v_{t,1}, \dots, v_{t,M}\}.
$$

Each flattened patch is projected to an embedding token and processed by a spatial transformer:

$$
\{v_{t,m}\}_{m=1}^M
\xrightarrow{\text{3D patch embed}}
\{h_{t,m}\}_{m=1}^M
\xrightarrow{\text{spatial transformer}}
\{\hat h_{t,m}\}_{m=1}^M.
$$

The spatial tokens are mean-pooled to one vector per timepoint, then passed through a temporal transformer:

$$
\bar h_t = \frac{1}{M}\sum_m \hat h_{t,m},
\qquad
[\bar h_1, \dots, \bar h_{T'}]
\xrightarrow{\text{temporal transformer}}
z^{ST}.
$$

Implementation stages:

1. reshape `(N, T', Z', H', W')` to `(N*T', 1, Z', H', W')`
2. 3D patch embedding by strided `Conv3d`
3. add sinusoidal spatial positional encoding
4. apply a spatial transformer encoder
5. mean-pool to one vector per frame
6. reshape to `(N, T', d_model)`
7. add temporal positional encoding
8. apply a temporal transformer encoder
9. mean-pool and project to `embedding_dim`

### 3.3 Temporal-then-spatial transformer branch

The model first coarse-pools the spatial lattice into patch locations and extracts a temporal sequence for each patch location. Each temporal signal is then patchified along time:

$$
s_p \in \mathbb{R}^{T'}
\mapsto
\{s_{p,1}, \dots, s_{p,L}\}.
$$

These temporal tokens are projected and processed by a temporal transformer:

$$
\{s_{p,\ell}\}_{\ell=1}^L
\xrightarrow{\text{1D patch embed}}
\{g_{p,\ell}\}_{\ell=1}^L
\xrightarrow{\text{temporal transformer}}
\{\hat g_{p,\ell}\}_{\ell=1}^L.
$$

The token sequence for each patch location is mean-pooled to one vector per spatial patch. Those patch vectors are then treated as a spatial token sequence and processed by a second transformer:

$$
\bar g_p = \frac{1}{L}\sum_\ell \hat g_{p,\ell},
\qquad
[\bar g_1, \dots, \bar g_P]
\xrightarrow{\text{spatial transformer}}
z^{TS}.
$$

Implementation stages:

1. average-pool the input into coarse spatial patches
2. reshape to one temporal sequence per spatial patch
3. 1D temporal patch embedding by strided `Conv1d`
4. add sinusoidal temporal positional encoding
5. apply a temporal transformer encoder
6. mean-pool to one vector per spatial patch
7. reshape to `(N, P, d_model)`
8. add spatial positional encoding
9. apply a spatial transformer encoder across patch tokens
10. mean-pool and project to `embedding_dim`

### 3.4 Example tensor flow

Using

$$
(T', Z', H', W') = (10, 3, 128, 128),
$$

with

$$
\text{spatial\_patch\_size\_st} = (1, 16, 16),
\qquad
\text{spatial\_patch\_size\_ts} = (1, 16, 16),
\qquad
\text{temporal\_patch\_size\_ts} = 5,
$$

the ST branch has

```text
Input frame: (1, 3, 128, 128)
3D patch grid: (3, 8, 8)
Tokens per frame: 192
Frame tokens: (N*10, 192, d_model)
Spatial transformer
Mean pool per frame -> (N, 10, d_model)
Temporal transformer
Mean pool + projection -> z^(ST): (N, d)
```

and the TS branch has

```text
Input: (N, 10, 3, 128, 128)
Coarse spatial patch grid: (3, 8, 8)
Spatial patch count: 192
Patch sequences: (N*192, 1, 10)
Temporal patch tokens per sequence: 2
Temporal transformer per patch
Mean pool per patch -> (N, 192, d_model)
Spatial transformer over patch tokens
Mean pool + projection -> z^(TS): (N, d)
```

### 3.5 Main hyperparameters

Tokenization:

- `spatial_patch_size_st`
- `spatial_patch_size_ts`
- `temporal_patch_size_ts`

Transformer capacity:

- `embed_dim`
- `num_heads`
- `mlp_ratio`
- `st_spatial_depth`
- `st_temporal_depth`
- `ts_temporal_depth`
- `ts_spatial_depth`
- `dropout`
- `attention_dropout`

Shared heads:

- `embedding_dim`
- `num_prototypes`
- `prototype_temperature`

### 3.6 Engineering constraints

This model requires:

- exact divisibility of the input dimensions by the configured patch sizes
- explicit control of token count to avoid memory blow-up
- more conservative defaults than the CNN variants

The current implementation therefore:

- enforces divisibility checks
- uses non-overlapping patch embeddings
- uses mean pooling rather than CLS tokens
- keeps the transformer factorized rather than building a single full spatiotemporal token sequence

As with the CNN estimators, inference returns target-keyed prediction and probability dictionaries for `action`, `compound`, and `concentration`.

### 3.7 Strengths and limitations

Strengths:

- more expressive token-level modeling within each branch
- cleaner separation between local tokenization and global aggregation
- closer to the abstract operator-factorization view in [method.md](method.md)

Limitations:

- much heavier in memory and runtime
- patch-size choices strongly affect tractability
- more exposed to overfitting on small supervised datasets

### 3.8 Multi-head supervision

Like the pure-CNN commutative model, the transformer variant uses the fused embedding for:

- action classification
- optional compound classification
- optional concentration classification

while keeping the branch-consistency and feature-alignment losses as the commutative part of the training objective.

## 4. Choosing between the three

Use the baseline `TimeChannel3DCNNClassifier` when:

- a stable supervised reference is needed
- the dataset is small
- you want the simplest debugging surface

Use `CommutativeCNNClassifier` when:

- you want to test the commutative objective with moderate complexity
- you need branch-specific diagnostics
- full token-based transformers would be too expensive

Use `CommutativeTransformerClassifier` when:

- you specifically want tokenized spatial and temporal aggregation
- you can afford the extra memory and training time
- you are testing architecture extensions rather than establishing the baseline

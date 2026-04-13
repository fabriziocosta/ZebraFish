# Method: Commutative Representation Learning for 4D Calcium Imaging

This document describes the method itself. Repository structure, notebook flow, and code entry points remain indexed in [README.md](../README.md), preprocessing details are documented in [preprocessing.md](preprocessing.md), and detailed model specifications are collected in [architecture.md](architecture.md).

Status:

- this document specifies the proposed commutative dual-pathway method
- the currently implemented repository baseline is the simpler 3D CNN classifier described in [README.md](../README.md)
- an experimental pure-CNN dual-pathway implementation is now available in `CommutativeCNNClassifier` and notebook 7, but it should still be treated as an early method implementation rather than the established repository baseline
- an experimental transformer dual-pathway implementation is now available in `CommutativeTransformerClassifier` and notebook 8, with the same status
- the current implementations can supervise the shared embedding against mechanism, compound, and concentration targets simultaneously

The raw imaging signal is a 4D tensor

$$
X \in \mathbb{R}^{T \times Z \times Y \times X},
$$

with a representative native resolution

$$
(T, Z, H, W) = (250, 10, 400, 600).
$$

The objective is to learn a representation that is invariant to the order in which valid spatial and temporal abstraction operators are applied.

## 1. Preprocessing assumptions

The method assumes the preprocessing and dataset conventions documented in [preprocessing.md](preprocessing.md):

- tensors are materialized in canonical `T x Z x Y x X` order
- deterministic downsampling is applied through `output_size=(T', Z', H', W')`
- optional LOESS-style drift normalization may be applied during loading
- dataset artifacts preserve `original_instance_id` for leakage-safe split assignment
- augmentation is applied only after splitting, and only to the training subset

The working tensor for the method is therefore

$$
X \mapsto \tilde X \in \mathbb{R}^{T' \times Z' \times H' \times W'}.
$$

Practical default working resolutions are

$$
(T', Z', H', W') = (125, 10, 100, 150)
$$

or

$$
(125, 8, 100, 150).
$$

## 2. Theory of the method

### 2.1 Commutative representation objective

Let $S$ denote a spatial abstraction operator and $T$ denote a temporal abstraction operator. The central principle is that the representation should not depend on whether spatial abstraction is applied before temporal abstraction or vice versa.

Define two pathways:

$$
F_{ST}(X) = T(S(X)), \qquad F_{TS}(X) = S(T(X)).
$$

The target invariance is approximate commutativity:

$$
F_{ST}(X) \approx F_{TS}(X).
$$

This does not require the intermediate computations to be identical. Instead, it requires the two admissible factorizations of the same signal to preserve the same semantic content.

### 2.2 Prototype-consistent semantic alignment

Let

$$
z_i^{ST}, z_i^{TS} \in \mathbb{R}^d
$$

be the embeddings produced by the two pathways for sample $i$, and let

$$
C \in \mathbb{R}^{d \times K}
$$

be a learnable prototype matrix with $K$ prototypes. Following the SwAV paradigm, each embedding is mapped to prototype logits

$$
p_i^{ST} = C^\top z_i^{ST}, \qquad p_i^{TS} = C^\top z_i^{TS},
$$

and then to soft prototype assignments

$$
q_i^{ST} = \operatorname{softmax}(p_i^{ST}/\tau), \qquad
q_i^{TS} = \operatorname{softmax}(p_i^{TS}/\tau).
$$

The commutative training signal is the swapped-assignment consistency loss

$$
\mathcal L_{\text{swap}} = H(q_i^{ST}, \operatorname{softmax}(p_i^{TS}/\tau)) + H(q_i^{TS}, \operatorname{softmax}(p_i^{ST}/\tau)).
$$

This enforces semantic agreement in prototype space rather than exact equality of latent vectors. Equivalently, if $\pi$ denotes the prototype-assignment map, then the method encourages

$$
\pi(ST(X)) \approx \pi(TS(X)).
$$

An optional feature-level alignment term may be added:

$$
\mathcal L_{\text{feat}} = \lVert z_i^{ST} - z_i^{TS} \rVert_2^2.
$$

The total loss is then

$$
\mathcal L = \mathcal L_{\text{swap}} + \lambda \mathcal L_{\text{feat}}.
$$

### 2.3 Interpretation

The method is not an augmentation-invariance objective in the usual sense. Instead, it is an operator-invariance objective: the representation should be stable under alternative valid decompositions of the same spatiotemporal structure. Prototype consistency provides the quotient space in which this relaxed form of commutativity is enforced. ([arXiv][2])

## 3. Practical implementation and architecture options

Detailed architecture descriptions for the three implemented model families now live in [architecture.md](architecture.md). This section only summarizes their methodological role.

### 3.1 Input layout

At dataset level, a subsampled sample may be written as

$$
\tilde X \in \mathbb{R}^{T' \times Z' \times H' \times W'}.
$$

For PyTorch implementation, branch-specific reshaping is applied as needed. `Conv3d` provides the core volumetric primitive for the CNN-based baseline and for the proposed dual-pathway design. ([PyTorch Documentation][1])

### 3.2 First implementation: pure-CNN dual pathway

The recommended first version is fully convolutional:

- spatial-then-temporal branch: shared 3D CNN per timepoint followed by a temporal 1D CNN over the sequence of frame embeddings
- temporal-then-spatial branch: shared temporal 1D CNN per spatial patch followed by a 3D CNN spatial aggregator
- shared projection head to latent dimension $d$
- shared prototype matrix $C$

This design is compatible with the repository preprocessing assumptions and with the practical constraint that full voxelwise tokenization is too expensive at the working resolutions above.

#### Spatial-then-temporal branch

At each timepoint $t$, extract a 3D volume

$$
V_t \in \mathbb{R}^{Z' \times H' \times W'}.
$$

A shared 3D CNN produces per-timepoint embeddings

$$
e_t^{(S)} = f_S(V_t) \in \mathbb{R}^d.
$$

These are stacked into a temporal sequence

$$
E^{(S)} = [e_1^{(S)}, \dots, e_{T'}^{(S)}] \in \mathbb{R}^{T' \times d},
$$

which is processed by a temporal 1D CNN to produce

$$
z^{ST} = f_T(E^{(S)}) \in \mathbb{R}^d.
$$

#### Temporal-then-spatial branch

The spatial lattice is partitioned into patches. For each patch location $p$, extract a temporal signal

$$
s_p \in \mathbb{R}^{T'}.
$$

A shared temporal 1D CNN produces patch embeddings

$$
u_p^{(T)} = g_T(s_p) \in \mathbb{R}^{d_t}.
$$

These embeddings are reassembled on a 3D patch lattice

$$
U^{(T)} \in \mathbb{R}^{Z_p \times H_p \times W_p \times d_t},
$$

and aggregated by a 3D CNN spatial encoder to produce

$$
z^{TS} = g_S(U^{(T)}) \in \mathbb{R}^d.
$$

#### Default working configuration

A practical baseline configuration is

$$
(T', Z', H', W') = (125, 8, 100, 150), \quad d = 256, \quad K = 512.
$$

For the temporal-then-spatial branch, patching the spatial lattice into

$$
(Z_p, H_p, W_p) = (8, 10, 15)
$$

gives

$$
P' = 8 \times 10 \times 15 = 1200
$$

patchwise temporal sequences, which is tractable for a convolutional temporal encoder.

### 3.3 Alternative architecture option: transformer-based variant

A heavier alternative replaces one or both temporal/spatial aggregation stages with transformers:

- spatial-then-temporal branch: spatial patch embedding followed by a temporal transformer
- temporal-then-spatial branch: temporal encoding followed by a spatial transformer over patch tokens

This variant is conceptually attractive but substantially more expensive in memory and implementation complexity. It is better treated as a later ablation or extension than as the first baseline.

### 3.4 High-level architecture diagram

See [Figure 1 in `figures.md`](figures.md).

### 3.5 Example tensor shapes for the pure-CNN baseline

Using

$$
(T', Z', H', W') = (125, 8, 100, 150),
$$

the spatial-then-temporal branch can be instantiated as

```text
Input: (N, 125, 8, 100, 150)
Reshape: (N*125, 1, 8, 100, 150)
3D CNN spatial encoder:
    -> (N*125, 32, 8, 50, 75)
    -> (N*125, 64, 4, 25, 38)
    -> (N*125, 128, 2, 13, 19)
Global average pool
    -> (N*125, 128)
Reshape
    -> (N, 125, 128)
Transpose
    -> (N, 128, 125)
Temporal 1D CNN
    -> (N, 256, 125)
Temporal global average pool
    -> z^(ST): (N, 256)
```

For the temporal-then-spatial branch, patching space into $(8,10,15)$ gives

```text
Input: (N, 125, 8, 100, 150)
Spatial patching
    -> (N, 1200, 125)
Temporal 1D CNN per patch
    -> (N*1200, 125, 128)
Temporal pooling
    -> (N, 1200, 128)
Reshape to spatial lattice
    -> (N, 8, 10, 15, 128)
Permute
    -> (N, 128, 8, 10, 15)
3D CNN spatial aggregator
    -> (N, 128, 4, 5, 8)
    -> (N, 256, 2, 3, 4)
Global average pool
    -> z^(TS): (N, 256)
```

## 4. Compatibility with the repository workflow

The method is designed to sit on top of the existing workflow rather than replace it:

- [README.md](../README.md) remains the orchestration document for notebooks, modules, and generated artifacts
- [preprocessing.md](preprocessing.md) defines the tensor materialization and split conventions assumed here
- notebook 5 persists the labeled tensor dataset artifact together with mechanism, compound, concentration, and control targets
- notebook 6 performs leakage-safe splitting by `original_instance_id`, applies training-only augmentation, and trains the currently implemented 3D CNN baseline
- notebook 7 applies the same split discipline to the pure-CNN commutative model
- notebook 8 applies the same split discipline to the transformer commutative model
- the current estimator API exposes target-wise predictions and probabilities keyed by `action`, `compound`, and `concentration`

In other words, this document specifies the proposed method and its current implementation direction, while the README documents the repository workflow and the baseline-to-experimental progression across notebooks.

[1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html?utm_source=chatgpt.com "Conv3d — PyTorch 2.11 documentation"
[2]: https://arxiv.org/abs/2006.09882?utm_source=chatgpt.com "Unsupervised Learning of Visual Features by Contrasting ..."

# Introduction

This document is the conceptual overview of the project. For preprocessing details, see [preprocessing.md](preprocessing.md). For the formal method specification, see [method.md](method.md).

**Commutative Representation Learning via Prototype-Consistent Dual Pathways**

---

### Problem

Modern representation learning for structured data (e.g., spatiotemporal signals, graphs, multi-scale systems) depends heavily on *architectural factorization*: whether one processes spatial structure before temporal dynamics, or vice versa. These choices are typically arbitrary, yet they can significantly affect learned representations.

This raises a fundamental issue:

> **Representations should not depend on the order in which valid abstraction operators are applied.**

Current self-supervised methods enforce invariance to augmentations, but **do not enforce invariance to factorization order of structure itself**.

---

### Core Idea

We introduce a framework for learning representations that are **invariant to the order of structured operators**, by enforcing **approximate commutativity** between alternative processing pathways.

Given two operators:

* $S$: spatial abstraction
* $T$: temporal abstraction

we construct two pathways:

$$
F_{ST}(x) = T(S(x)), \quad F_{TS}(x) = S(T(x))
$$

and enforce:

$$
F_{ST}(x) \approx F_{TS}(x)
$$

This enforces that spatial-then-temporal and temporal-then-spatial processing produce **consistent semantic representations**.

---

### Method

We instantiate this principle using a **dual-pathway self-supervised architecture** combined with **masked probe prediction**:

* Each pathway produces an embedding of the same input.
* A fixed probe ontology defines structured questions about the input.
* Self heads decode probes from their own pathway embeddings.
* Cross heads decode the same probes using the other pathway's prediction machinery.

The current probe types are:

* `local`
* `region_time`
* `derivative`
* `frequency`
* `correlation`

Every probe type is present in every batch. Random binary masks decide which entries are supervised at a given step.

The loss enforces:

$$
\text{ProbeAnswers}(ST(x)) \approx \text{ProbeAnswers}(TS(x))
$$

This can be interpreted as:

> **Commutativity through a shared structured prediction interface.**

---

### Key Properties

#### 1. Structural Invariance

The learned representation is invariant to the order of abstraction operators, capturing intrinsic structure rather than architectural artifacts.

#### 2. Structured Self-Supervision

Masked probe prediction makes each embedding preserve information needed to answer local, regional, temporal, spectral, and correlation questions about the same input.

#### 3. Operator-Level Regularization

Unlike standard multi-view learning, the method constrains **how representations are formed**, not just their final similarity.

#### 4. Generality

The framework applies to any pair of structured operators, including:

* spatial vs temporal
* local vs global
* node vs graph
* fine vs coarse

---

### Conceptual Interpretation

The method can be viewed as enforcing:

$$
D(ST(x)) \approx D(TS(x))
$$

where $D$ denotes the shared family of structured probe decoders.

Thus, instead of requiring exact commutativity, we require:

> **semantic commutativity under shared probe prediction**

This is a relaxed but powerful structural constraint.

---

### Why It Matters

This approach introduces a new principle for representation learning:

> **Learn representations that are invariant to valid decompositions of structure.**

This is fundamentally different from:

* augmentation invariance (data-level),
* contrastive similarity (instance-level),

and instead operates at the level of **operator algebra over data structure**.

---

### Potential Impact

* More robust representations across architectures
* Reduced sensitivity to modeling choices
* Improved interpretability via consistent semantic structure
* New theoretical connections between representation learning and operator theory

---

### Summary

We propose a new direction in self-supervised learning:

> **Commutative Representation Learning** — enforcing invariance to the order of structured abstraction via masked-probe dual pathways.

This framework provides a principled, general, and extensible approach to learning representations that reflect the intrinsic structure of data rather than the arbitrary order of processing.

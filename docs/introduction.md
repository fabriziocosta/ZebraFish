# Introduction

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

We instantiate this principle using a **dual-pathway self-supervised architecture** combined with **prototype-based consistency learning**:

* Each pathway produces an embedding of the same input.
* Embeddings are mapped to a shared set of prototypes.
* Training enforces **agreement of prototype assignments** across pathways.

This follows the SwAV paradigm, where representations are aligned via clustering rather than direct feature matching.

The loss enforces:

$$
\text{Cluster}(ST(x)) \approx \text{Cluster}(TS(x))
$$

This can be interpreted as:

> **Commutativity in the quotient space defined by learned semantic clusters.**

---

### Key Properties

#### 1. Structural Invariance

The learned representation is invariant to the order of abstraction operators, capturing intrinsic structure rather than architectural artifacts.

#### 2. Non-Collapsing Self-Supervision

Prototype-based alignment avoids trivial solutions and provides meaningful semantic organization.

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
\pi(ST(x)) = \pi(TS(x))
$$

where $\pi$ maps representations to prototype assignments.

Thus, instead of requiring exact commutativity, we require:

> **semantic commutativity under learned clustering**

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

> **Commutative Representation Learning** — enforcing invariance to the order of structured abstraction via prototype-consistent dual pathways.

This framework provides a principled, general, and extensible approach to learning representations that reflect the intrinsic structure of data rather than the arbitrary order of processing.

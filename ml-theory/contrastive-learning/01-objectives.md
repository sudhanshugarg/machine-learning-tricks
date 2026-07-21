# Contrastive Objectives and Loss Functions

## Overview

A contrastive objective’s job is to score the similarity between embeddings and turn that into a training signal. This chapter covers the most common objectives — **InfoNCE**, **NT-Xent (Normalized Temperature-scaled Cross Entropy)**, and **Triplet Loss** — with derivation, intuition, and what exactly they optimize.

---

## 1. InfoNCE

### The Objective

InfoNCE (Noise-Contrastive Estimation) is the workhorse of modern contrastive learning. Given an anchor embedding $z_i$, one positive $z_i^+$, and $N-1$ negatives $\{z_j^-\}_{j=1}^{N-1}$:

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j) / \tau)}
$$

Where:
- $\text{sim}(a, b) = a^\top b$ (dot product) or cosine similarity
- $\tau$ = temperature hyperparameter
- The denominator sums over **all $N$ candidates** (1 positive + $N-1$ negatives)

### What It Optimizes

InfoNCE is a **lower bound on mutual information** between the representations of two views of the same data:

$$
I(f(x_i), f(x_i^+)) \geq \log N - \mathcal{L}_{\text{InfoNCE}}
$$

By minimizing the loss, you maximize a lower bound on the mutual information captured by the encoder.

### Intuition

Think of it as a **$(N+1)$-way classification** problem:

```
Query: z_i
Candidates:  [z_i^+ ,  z_1^- ,  z_2^- ,  ... ,  z_{N-1}^- ]
             ↑positive   negatives

Goal: make the positive's score highest among all candidates
```

It is structurally identical to softmax cross-entropy, except the "classes" are other embeddings in the batch/memory bank.

---

## 2. NT-Xent (Normalized Temperature-scaled Cross Entropy)

### The Objective

NT-Xent is the specific variant used in SimCLR. It adds **L2 normalization** and uses **cosine similarity**:

$$
\text{sim}(z_i, z_j) = \frac{z_i^\top z_j}{\|z_i\| \|z_j\|}
$$

$$
\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2B} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

Where $B$ = batch size, and the denominator sums over **all other samples in the augmented batch** (size $2B$).

### Why Normalize?

L2 normalization constrains embeddings to the unit hypersphere:

$$
\|z\|_2 = 1 \quad \Rightarrow \quad \text{sim}(z_i, z_j) \in [-1, 1]
$$

Benefits:
- Similarity is bounded, preventing extreme logits
- Training is more stable (no runaway magnitudes)
- Encourages the model to focus on **direction** rather than magnitude

### Symmetric Formulation

SimCLR averages the loss over both directions:

$$
\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2B} \sum_{k=1}^{B} \left[ \ell(2k-1, 2k) + \ell(2k, 2k-1) \right]
$$

Each positive pair appears twice (as anchor → positive and positive → anchor).

---

## 3. Triplet Loss

### The Objective

Triplet loss predates InfoNCE and is still widely used in face verification, recommendation, and metric learning:

$$
\mathcal{L}_{\text{triplet}} = \max(0, \, d(z_a, z_p) - d(z_a, z_n) + m)
$$

Where:
- $z_a$ = anchor embedding
- $z_p$ = positive embedding (same identity/class)
- $z_n$ = negative embedding (different identity/class)
- $d(\cdot, \cdot)$ = distance function (usually Euclidean or cosine distance)
- $m$ = margin hyperparameter

### Intuition

```
Embedding Space:

Anchor  Positive      Negative
  ●        ●              ○
  └──────┘
   d(a,p)

Requirement: d(a,p) + m < d(a,n)

If the negative is already farther than d(a,p) + m → loss = 0 (easy triplet)
If the negative is too close                               → loss > 0 (hard triplet)
```

### Triplet Selection Strategies

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Easy** | Random negatives; most triplets already satisfy margin. | Fast sampling | No learning signal (loss = 0) |
| **Hard** | Hardest negative in batch (closest to anchor). | Strong signal | Can destabilize training, collapse |
| **Semi-hard** | Hardest negative that is still farther than positive + margin. | Balanced signal | Requires careful tuning |

---

## 4. Comparison Table

| Aspect | InfoNCE / NT-Xent | Triplet Loss |
|--------|-------------------|--------------|
| **Supervision** | Self-supervised or supervised | Usually supervised (labels for anchor/positive/negative) |
| **Normalization** | Typically L2-normalized embeddings | Optional (Euclidean distance common) |
| **Temperature** | Explicit $\tau$ scales similarity | Implicit via margin $m$ |
| **Number of negatives** | Many ($B-1$ or memory-bank size) | One explicit negative per triplet |
| **Gradient behavior** | Smooth, well-behaved with many negatives | Can collapse to zero with easy triplets |
| **Popular in** | SimCLR, MoCo, CLIP | FaceNet, recommendation, metric learning |

---

## 5. Unified View: All Are Similarity-Ranking Objectives

At their core, all contrastive losses optimize the same thing:

> **Rank the positive higher than all negatives by a comfortable margin.**

- **InfoNCE** uses softmax over a candidate set; the margin is implicit and softened by temperature.
- **Triplet loss** uses a hinge; the margin is explicit and hard.
- **NT-Xent** is InfoNCE + L2 normalization + symmetric formulation.

---

## 6. Practical Training Dynamics

### What Happens If Negatives Are Too Easy?

- **InfoNCE**: Gradient still flows (softmax never truly reaches 0 loss), but learning slows.
- **Triplet**: Loss becomes exactly 0; gradients vanish; model stops learning.

This is why **hard negative mining** and **large batch sizes** are so important for triplet-based methods.

### What Happens If Negatives Are Too Hard?

- The model may **collapse** — learning to map everything to the same vector to minimize distance to all positives.
- Remedies: stop gradients on one branch (e.g., MoCo momentum encoder), use large batches, or add regularization.

Next: [02 — Positives & Negatives](02-positives-negatives.md)

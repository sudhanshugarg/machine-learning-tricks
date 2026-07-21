# Constructing Positives and Negatives

## Overview

The quality of a contrastive model depends entirely on how you define **what is similar** (positives) and **what is dissimilar** (negatives). This chapter covers construction strategies, the dominant "in-batch negatives" paradigm, and the subtle trade-offs of each approach.

---

## 1. Positive Pair Construction

### Definition

A **positive pair** consists of two augmented views of the **same underlying sample**.

### Standard Image Augmentations (SimCLR-style)

For an image $x$, generate two independent augmentations:

```
Original Image x
      ┌─────────┴─────────┐
      ↓                   ↓
   View 1              View 2
  (t_1(x))            (t_2(x))
  - RandomResizedCrop   - RandomResizedCrop
  - ColorJitter         - HorizontalFlip
  - GaussianBlur        - Grayscale (p=0.2)
```

Common augmentation pipeline:

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| RandomResizedCrop | scale=(0.08, 1.0) | Local views, scale invariance |
| ColorJitter | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 | Color invariance |
| GaussianBlur | kernel=23, sigma=(0.1, 2.0) | Texture invariance |
| HorizontalFlip | p=0.5 | Mirroring invariance |
| Grayscale | p=0.2 | Color-agnostic features |

### Text Positives (NLP)

For a sentence $s$:

- **Back-translation**: $s \to$ German $\to s'$ (paraphrase)
- **Dropout sampling**: Forward the same sentence twice through a dropout-enabled encoder; the stochastic masks create two different embeddings.
- **Span corruption**: Mask/delete spans and reconstruct (used in ELECTRA-style approaches).

### Temporal Positives (Video / Audio)

For a video clip:

- **Same clip, different frames**: Sample two non-overlapping frame sets from the same video segment.
- **Nearby clips**: Clips within a short temporal window are positives; clips far apart are negatives.

---

## 2. Negative Construction Strategies

### Strategy A: In-Batch Negatives (SimCLR, CLIP)

Every other sample in the batch (besides the positive pair) is treated as a negative.

**Pros**:
- **Free**: No extra memory or computation — the negatives are already computed for the forward pass.
- **Simple**: No queue, memory bank, or hard-mining logic needed.
- **Effective**: With batch size $B$, each anchor sees $2(B-1)$ negatives.

**Cons**:
- **Batch size dependency**: Smaller batches mean fewer negatives → weaker learning signal.
- **False negatives**: If two augmentations of different images happen to be semantically similar (e.g., two photos of dogs), one is incorrectly treated as a negative.
- **Distribution mismatch**: Negatives are drawn from the same mini-batch distribution, which may not reflect the full data distribution.

### Strategy B: Memory Banks / Queues (MoCo)

Maintain a FIFO queue of recent embeddings from past iterations:

```
Current Batch → Forward → Embeddings
                              ↓
                        Enqueue to Queue (size K)
                              ↓
                    ┌─────────────────┐
                    │  z_1  z_2  ...  │  ← used as negatives
                    │  z_K  z_{K-1}   │     for current anchors
                    └─────────────────┘
                              ↓
                         Dequeue oldest
```

**Pros**:
- **Many negatives** without proportional memory increase (queue size $K$ can be 65k+).
- **Decouples batch size from negative count**.

**Cons**:
- **Stale negatives**: Embeddings in the queue are from older model versions, creating an encoder-负 sample mismatch.
- **Requires momentum encoder** (MoCo v1/v2) or stop-gradient tricks to prevent collapse.

### Strategy C: Global Negative Mining

Mine hard negatives from the **entire dataset** using approximate nearest-neighbor search (FAISS, ScaNN):

```python
# Offline mining every N epochs
all_embeddings = compute_embeddings(dataloader)
negatives = faiss_search(all_embeddings, anchor, k=100)
```

**Pros**:
- **Hardest negatives** from the full distribution.
- Strong signal for fine-grained tasks.

**Cons**:
- **Expensive**: Requires periodic full-dataset passes and index rebuilding.
- **Can destabilize**: Extremely hard negatives may collapse the model.

---

## 3. The False Negative Problem

A **false negative** occurs when a negative sample is actually semantically similar to the anchor (e.g., two different photos of golden retrievers).

### Impact

- The model is pushed to separate embeddings that should actually be close.
- **Capping representation quality**: The loss has a non-zero lower bound due to these contradictions.

### Mitigations

| Method | How It Helps |
|--------|--------------|
| **Large batches** | Law of large numbers reduces the chance that a specific anchor collides with a semantic duplicate |
| **Data cleaning** | Remove near-duplicates from the dataset |
| **Supervised contrastive** | Use class labels to define positives (all same-class samples are positives), eliminating false negatives entirely |
| **Debiased losses** | Estimate and subtract out the expected false-negative contribution |

---

## 4. In-Batch Negatives: A Deeper Look

### Why It Works

In batch size $B$, you have $2B$ augmented samples (2 views per image). For each anchor, there is exactly **1 positive** and **$2(B-1)$ negatives**.

Total negative pairs per batch:

$$
\text{Total negatives} = 2B \times 2(B-1) = 4B(B-1)
$$

For $B = 4096$:

$$
\text{Total negatives} = 4 \times 4096 \times 4095 \approx 67 \text{ million pairs}
$$

This creates a massive implicit comparison set without any extra computation.

### The Catch: GPU Memory

To store $2B$ embeddings and compute the full $2B \times 2B$ similarity matrix:

- $B = 256$: trivial on a single GPU
- $B = 4096$: requires ~16 GB for embeddings alone; often needs 8-16 GPUs or gradient accumulation tricks.

Next: [03 — Temperature, Normalization, and Memory Banks](03-hyperparameters.md)

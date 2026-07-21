# Contrastive Learning: A Deep Dive

## What is This Topic?

**Contrastive Learning** is a self-supervised learning paradigm that learns representations by bringing "similar" samples (positives) closer together in embedding space while pushing "dissimilar" samples (negatives) farther apart. It is the driving force behind powerful vision encoders (SimCLR, MoCo) and sentence embeddings (Sentence-BERT).

---

## Why Does It Matter?

- **Scalability**: Needs no human labels — supervision comes from data itself (e.g., two augmented views of the same image).
- **State-of-the-art pretraining**: Self-supervised contrastive pretraining rivals or exceeds supervised pretraining on ImageNet.
- **Foundation models**: CLIP, DINO, and modern LLM embeddings all rely on contrastive objectives.
- **Transferability**: Contrastive encoders produce representations that transfer exceptionally well to downstream tasks.

---

## Big Picture

Imagine photos of dogs:

```
        Anchor: photo of a golden retriever
        /                  \
  Positive            Negative
  (cropped &          (photo of
   color-jittered)      a cat)
        \                  /
         \                /
          Embedding Space
          [ dog cluster ]      [ cat cluster ]
             ↑  anchor
             ↑  positive
                  ←—— far away ——→  negative
```

The model learns to map semantically similar inputs to nearby vectors and dissimilar inputs to distant vectors — all without ever seeing a class label.

---

## Concrete Example: Image Patches

Take a single image $x$:

1. **Create two augmented views**: $x_1$ (random crop + color jitter) and $x_2$ (horizontal flip + grayscale).
2. **Forward both** through encoder $f$ and projection head $g$:
   $$z_1 = g(f(x_1)), \quad z_2 = g(f(x_2))$$
3. **Treat $(z_1, z_2)$ as a positive pair** — they should be similar.
4. **Treat $z_1$ with embeddings of other images in the batch** as negatives.
5. **Minimize a contrastive loss** (e.g., InfoNCE) that pulls $z_1$ and $z_2$ together while pushing $z_1$ away from negatives.

---

## What You Will Learn

1. **[Objectives & Loss Functions](01-objectives.md)** — InfoNCE, NT-Xent, triplet loss, and what they optimize.
2. **[Positives & Negatives](02-positives-negatives.md)** — How to construct pairs, pros/cons of in-batch negatives, hard negative mining.
3. **[Hyperparameters & Architecture](03-hyperparameters.md)** — Temperature, L2 normalization, projection heads, memory banks, and queues.
4. **[Evaluation](04-evaluation.md)** — Linear probe, kNN classifier, and downstream fine-tuning benchmarks.
5. **[Resource-Constrained Training](05-resource-optimization.md)** — Gradient checkpointing, large-batch simulation, and communication-efficient variants.

---

## Prerequisites

- PyTorch basics (`nn.Module`, autograd)
- Familiarity with backpropagation and multi-class cross-entropy
- Understanding of L2 normalization and dot products
- Basic linear algebra (matrix multiplication, softmax)

---

## Next Steps

→ [01 — Objectives & Loss Functions](01-objectives.md)

# Evaluation of Contrastive Representations

## Overview

A contrastive model is only as good as its representations. Since there is no explicit downstream task during pretraining, evaluation requires diagnostic probes that measure how **transferable** and **linearly separable** the learned features are.

---

## 1. Linear Probe

### Protocol

1. **Freeze** the pretrained encoder $f$.
2. Attach a **randomly initialized linear classifier** on top of $h = f(x)$.
3. Train only the linear layer on labeled data (e.g., ImageNet).
4. Report top-1 / top-5 accuracy.

```python
# Freeze encoder
for p in encoder.parameters():
    p.requires_grad = False

# Train linear probe
classifier = nn.Linear(encoder_dim, num_classes)
optimizer = torch.optim.SGD(classifier.parameters(), lr=30.0, momentum=0.9)
# Note: very high LR is common for linear probes
```

### Why It Matters

- A strong linear probe means the encoder has learned **semantically meaningful clusters** that are linearly separable.
- It approximates how well the representations will transfer to downstream tasks with limited labeled data.

### Typical Numbers (ImageNet-1K)

| Method | Architecture | Pretraining | Linear Probe Top-1 |
|--------|-------------|-------------|-------------------|
| Random init | ResNet-50 | — | 6% |
| Supervised | ResNet-50 | ImageNet labels | 76% |
| SimCLR | ResNet-50 | Unlabeled ImageNet | 76% |
| MoCo v3 | ViT-B/16 | Unlabeled ImageNet | 81% |
| DINO | ViT-B/16 | Unlabeled ImageNet | 78% |

> Self-supervised methods now match or exceed purely supervised pretraining.

---

## 2. k-Nearest Neighbors (kNN) Classifier

### Protocol

1. Pass the entire evaluation dataset through the **frozen encoder** to get embeddings.
2. For each test sample, find its $k$ nearest neighbors in the training set (by cosine similarity).
3. Predict the majority class of those $k$ neighbors.

```python
# Extract embeddings
train_features, train_labels = encode_all(train_loader)
test_features, test_labels = encode_all(test_loader)

# kNN prediction
distances = test_features @ train_features.T  # cosine similarity
_, indices = distances.topk(k=20, dim=1)
preds = mode(train_labels[indices], dim=1).mode
acc = (preds == test_labels).float().mean()
```

### Why It Matters

- **No training at all** on the downstream task.
- Tests whether the embedding space has **naturally separated classes** without any linear adaptation.
- Often used as a quick sanity check during pretraining (faster than full linear probe training).

---

## 3. Downstream Fine-Tuning

### Protocol

1. Initialize a task-specific head on top of the pretrained encoder.
2. **Unfreeze the entire model** (or progressively unfreeze layers).
3. Fine-tune with a small learning rate (e.g., 0.01× the head LR).
4. Evaluate on the target task.

### Transfer Tasks

| Domain | Tasks | Metric |
|--------|-------|--------|
| Vision | COCO detection/segmentation, VOC classification | mAP, accuracy |
| NLP | GLUE, SuperGLUE, sentiment analysis | Accuracy, F1 |
| Multimodal | Image captioning, retrieval | CIDEr, Recall@K |

### Fine-Tuning vs Linear Probe

- **Linear probe**: Tests representation quality under minimal adaptation.
- **Fine-tuning**: Tests representation quality under full adaptation — closer to real deployment.

Both are necessary. A model with a great linear probe but poor fine-tuning may have representations that are "too linear" and lack the nonlinear structure needed for complex downstream heads.

---

## 4. Semantic Retrieval / Ranking

### Protocol

1. Encode a query and a candidate pool into the same embedding space.
2. Rank candidates by cosine similarity.
3. Compute **Recall@K** or **Mean Reciprocal Rank (MRR)**.

```python
# Image-text retrieval (CLIP-style)
image_features = image_encoder(images)       # [N, D]
text_features = text_encoder(text_tokens)    # [N, D]

# Normalize
image_features = F.normalize(image_features, dim=-1)
text_features = F.normalize(text_features, dim=-1)

# Similarity matrix
similarity = image_features @ text_features.T  # [N, N]

# Image-to-text recall@1
i2t_r1 = (similarity.argmax(dim=1) == torch.arange(N)).float().mean()
```

### Why It Matters

Retrieval tests the **structure of the entire embedding space**, not just class boundaries. A good contrastive model should place "cat" closer to "kitten" than to "airplane."

---

## 5. Uniformity vs. Tolerance (Geometric Diagnostics)

Wang & Isola (2020) proposed decomposing representation quality into two geometric properties:

### Alignment

How close are positive pairs?

$$
\mathcal{L}_{\text{align}} = \mathbb{E}_{(x, x^+) \sim p_{\text{pos}}} \|f(x) - f(x^+)\|_2^2
$$

### Uniformity

How uniformly are embeddings distributed on the hypersphere?

$$
\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{x, y \sim p_{\text{data}}} e^{-t \|f(x) - f(y)\|_2^2}
$$

### Interpretation

| Method | Alignment | Uniformity | Linear Probe |
|--------|-----------|------------|--------------|
| Untrained | Bad (random) | Good (random spread) | Bad |
| Overfit to positives only | Good | Bad (all collapsed) | Bad |
| Well-trained contrastive | Good | Good | Good |

A good contrastive model achieves **both**: positives are close, and the overall space is well-utilized.

Next: [05 — Resource-Constrained Training](05-resource-optimization.md)

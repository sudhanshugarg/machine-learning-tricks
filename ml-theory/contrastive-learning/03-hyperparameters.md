# Temperature, Normalization, and Architecture Choices

## Overview

Beyond the loss function itself, three design choices dominate contrastive learning performance: **temperature scaling**, **L2 normalization**, and the use of **projection heads / memory banks**. This chapter explains the mechanics and trade-offs of each.

---

## 1. Temperature ($\tau$)

### What It Does

The temperature parameter scales the logits before softmax:

$$
\text{softmax}_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

### Effect on Gradients

| Temperature | Behavior | Gradient on Hard Negatives | Gradient on Easy Negatives |
|-------------|----------|---------------------------|---------------------------|
| **$\tau \to 0$** (cold) | Softmax becomes one-hot | Very large | Near zero |
| **$\tau \to \infty$** (hot) | Softmax becomes uniform | Small and balanced | Small and balanced |

### Trade-off

- **Low $\tau$** (e.g., 0.05): The model focuses aggressively on separating the hardest negatives. Can lead to unstable training and representation collapse.
- **High $\tau$** (e.g., 1.0): All negatives contribute more evenly. Training is stable but may not push hard negatives far enough.

### Typical Values

| Method | $\tau$ | Notes |
|--------|--------|-------|
| SimCLR | 0.5 | Large batch, strong uniformity |
| MoCo v3 | 0.2 | Momentum encoder, moderate focus |
| CLIP | 0.07 | Image-text, learned or fixed |
| SwAV | 0.1 | Clustering instead of instance |

### Learned Temperature (CLIP-style)

CLIP initializes $\tau$ as a learnable parameter and constrains it:

```python
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
scale = self.logit_scale.exp().clamp(max=100)
```

This lets the model discover the optimal temperature for the image-text distribution.

---

## 2. L2 Normalization (Embedding Constraining)

### What It Does

Before computing similarity, embeddings are divided by their L2 norm:

$$
z_{\text{norm}} = \frac{z}{\|z\|_2}, \quad \|z\|_2 = \sqrt{\sum_i z_i^2}
$$

This maps all embeddings to the **surface of a unit hypersphere**.

### Why It Matters

**Without normalization**:
- Similarity = dot product = $\|z_i\| \|z_j\| \cos\theta$
- The model can increase similarity simply by **growing magnitudes**.
- Two embeddings can be orthogonal ($\cos\theta = 0$) yet have dot product $\gg 0$ due to large norms.

**With normalization**:
- Similarity = $\cos\theta \in [-1, 1]$
- The model must actually **align directions** to increase similarity.
- Prevents the "magnitude hack."

### Geometric Interpretation

```
Without L2 norm:
     ●───→ z_i (large magnitude)
    /
   /
  ●───→ z_j (large magnitude)
  Dot product huge even if angle is 90°

With L2 norm:
       ↗
      /  z_i (on unit sphere)
     ●─────────→ z_j
     cos(θ) = similarity
```

### When to Skip It

Some methods (e.g., certain metric-learning variants) intentionally omit L2 normalization to let the model learn magnitude-based confidence. This is uncommon in standard vision contrastive learning.

---

## 3. Projection Heads

### The Architecture

A typical contrastive encoder has two parts:

```
Input Image
    ↓
[ Encoder f ]   →  h = f(x)       (representation)
    ↓
[ Projection Head g ] →  z = g(h)   (contrastive space)
    ↓
  Compute loss on z
```

Where $g$ is usually:
- MLP with 1-3 layers
- Hidden dim = encoder dim
- Output dim often smaller (e.g., 128)
- No bias in final layer (SimCLR)

### Why a Separate Head?

**The paradox**: The projection head improves the **encoder** $f$ even though the loss is computed on $g(f(x))$.

- $z$-space is optimized for **discrimination between instances**.
- $h$-space retains **more generalizable features** useful for downstream tasks.
- In practice, you **discard $g$ after pretraining** and use $h$ for transfer learning.

### Ablation Insight

SimCLR ablations show:

| Head | Linear Probe Acc | Observation |
|------|-----------------|-------------|
| Linear ($z = Wh$) | 71% | Worse |
| MLP (1 hidden layer) | 76% | Better |
| MLP + nonlinearity | 76.5% | Best |
| No projection head | 65% | Significantly worse |

---

## 4. Memory Banks and Queues

### MoCo (Momentum Contrast)

MoCo addresses the batch-size limitation by maintaining a **queue** of negative samples:

```python
# Key embeddings (negatives) are maintained in a queue
self.queue = torch.randn(dim, K)  # K = 65536 typically
self.queue_ptr = 0

def dequeue_and_enqueue(self, keys):
    # Replace oldest entries with new keys
    batch_size = keys.shape[0]
    ptr = int(self.queue_ptr)
    self.queue[:, ptr:ptr+batch_size] = keys.T
    self.queue_ptr = (ptr + batch_size) % K
```

### Momentum Encoder

To keep the queue consistent, MoCo encodes keys with a **momentum-updated encoder**:

$$
\theta_k \leftarrow m \theta_k + (1-m) \theta_q
$$

Where:
- $\theta_q$ = query encoder (gradients flow here)
- $\theta_k$ = key encoder (no gradients)
- $m$ = momentum coefficient (e.g., 0.999)

The slow-moving key encoder prevents the queue from becoming stale too quickly.

### MoCo v3 Simplification

MoCo v3 drops the queue entirely for vision transformers and instead uses:
- Large batches + stop-gradient on one branch
- Extra predictor head to prevent collapse

This shows that **queues are not strictly necessary** with sufficient batch size and architectural tricks.

---

## 5. Preventing Representation Collapse

**Collapse** = model maps all inputs to the same (or nearly the same) embedding. Loss is minimized, but representations are useless.

| Technique | How It Prevents Collapse |
|-----------|-------------------------|
| **Large batch / many negatives** | The uniformity term in the loss pushes all negatives apart |
| **Stop-gradient + predictor** (BYOL, SimSiam) | One branch has no gradient; the predictor must transform representations non-trivially |
| **Momentum encoder** (MoCo) | Key encoder moves slowly; cannot chase the query into collapse |
| **Clustering constraint** (SwAV) | Assigns prototypes; prevents all samples from converging to one point |

---

## 6. Quick Reference: Hyperparameter Impact

```
Hyperparameter    ↑ Increase Effect                ↓ Decrease Effect
─────────────────────────────────────────────────────────────────
Temperature τ     More focus on hard negatives     More uniform gradient
Batch size B      More negatives, better signal    Less memory, faster iter
Queue size K      More diverse negatives           More stale negatives
Projection dim    Richer contrastive space         More compute
Momentum m        More stable keys                 Keys track queries too closely
```

Next: [04 — Evaluation Protocols](04-evaluation.md)

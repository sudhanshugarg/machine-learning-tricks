# FAQ: Contrastive Learning

### Q: Why is contrastive learning called "self-supervised"?

**A:** No human-provided labels are needed. The "supervision signal" comes from the data itself: two augmented views of the same image are defined as a positive pair. The structure of the data provides the learning objective.

---

### Q: What is the difference between InfoNCE and NT-Xent?

**A:** NT-Xent is a **specific instance** of InfoNCE. InfoNCE is the general framework (softmax over positives vs. negatives). NT-Xent adds:
1. L2 normalization of embeddings
2. Cosine similarity (dot product of normalized vectors)
3. Symmetric formulation (loss computed in both directions)

In practice, the terms are often used interchangeably.

---

### Q: Do I need a projection head for downstream tasks?

**A:** No. The projection head is discarded after pretraining. The representation $h = f(x)$ (before the projection head) is what you use for downstream tasks. The head exists only to make the contrastive objective easier to optimize.

---

### Q: Why does SimCLR need such large batch sizes?

**A:** SimCLR relies on **in-batch negatives**. With batch size $B$, each anchor sees only $2(B-1)$ negatives. For the InfoNCE bound on mutual information to be tight, you need many negatives. Empirically, SimCLR needs $B \geq 4096$ to work well on ImageNet. Smaller batches work with memory banks or queues (MoCo-style).

---

### Q: What is representation collapse, and how do I detect it?

**A:** Collapse is when the encoder maps all inputs to nearly the same embedding. Detection methods:
- **kNN accuracy drops** to random chance.
- **Uniformity loss** becomes very poor (embeddings cluster).
- **Visual inspection**: Plot embeddings with t-SNE; collapsed models show a single dense cluster.
- **Cosine similarity**: Most pairs have similarity ≈ 1.0 (identical direction).

Prevention: use stop-gradient (BYOL), large batches, momentum encoders (MoCo), or clustering (SwAV).

---

### Q: Can I use contrastive learning for tabular data?

**A:** Yes, but augmentation is harder. Common approaches:
- **Feature corruption**: Mask or noise a subset of features.
- **Mixup in feature space**: Interpolate between samples.
- **VAE-style**: Reconstruction + contrastive on latent space.

Tabular data lacks the spatial invariances of images, so random cropping/color jitter do not apply.

---

### Q: What is the difference between MoCo and SimCLR?

**A:**

| Aspect | MoCo | SimCLR |
|--------|------|--------|
| Negatives | Memory bank / queue | In-batch only |
| Batch size requirement | Small batches OK | Needs very large batches |
| Encoder setup | Momentum encoder + query encoder | Single shared encoder |
| Data augmentation | Standard | Stronger (more aggressive) |
| Key trick | Momentum update stabilizes keys | Large batch + strong aug |

MoCo v3 converges to a design closer to SimCLR (large batch, no queue, stop-gradient).

---

### Q: How does temperature affect the loss landscape?

**A:**
- **Low $\tau$**: The loss is dominated by the hardest negative. Gradients are large but noisy. The model may overfit to specific hard cases.
- **High $\tau$**: All negatives contribute more evenly. Training is smoother but may lack the "push" to separate challenging pairs.

CLIP and some methods learn $\tau$ dynamically because different datasets/modalities have different optimal temperatures.

---

### Q: Is supervised contrastive learning better than cross-entropy?

**A:** Often yes, especially with limited labeled data. Supervised contrastive (Khosla et al., 2020) treats all same-class samples as positives. Benefits:
- Better calibration (confidence scores correlate better with accuracy).
- More robust to label noise.
- Better transfer to other tasks.

Trade-off: requires more computation (encoding all samples in a batch for pairwise comparison).

---

### Q: How do I evaluate if my contrastive model is learning anything during training?

**A:** Cheap online diagnostics:
1. **kNN accuracy** on a validation set every N epochs (no training needed).
2. **Loss curve**: Should decrease steadily. Plateaus are normal; sudden spikes indicate instability.
3. **Alignment loss**: Measure average distance between positive pairs; should decrease.
4. **Uniformity loss**: Measure how spread out embeddings are; should improve.
5. **Visualize embeddings**: t-SNE/UMAP every few epochs to check for structure.

---

### Q: Can contrastive learning work with small datasets?

**A:** It struggles. Contrastive learning needs **diversity** to create meaningful negatives. With small datasets:
- Use **transfer learning**: Pretrain on a large dataset (ImageNet), then fine-tune.
- Use **stronger augmentations** to artificially increase diversity.
- Consider **supervised contrastive** if you have labels.

For very small datasets (hundreds of samples), traditional supervised learning with heavy regularization often outperforms self-supervised contrastive methods.

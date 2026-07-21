"""
Runnable Code Examples: Contrastive Learning

Run this file directly to see concrete demonstrations of:
1. NT-Xent loss computation with in-batch negatives.
2. InfoNCE with explicit negatives.
3. Triplet loss with hard negative mining.
4. Effect of temperature on softmax distribution.
5. L2 normalization and cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def demo_nt_xent():
    """Compute NT-Xent loss for a mini-batch."""
    print("=" * 60)
    print("DEMO: NT-Xent Loss")
    print("=" * 60)

    batch_size = 4
    dim = 8
    temperature = 0.5

    # Two augmented views of the same 4 images
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate: [2B, D]
    z = torch.cat([z1, z2], dim=0)  # [8, 8]

    # Similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # [8, 8]

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_sim = torch.cat([
        sim_matrix.diag(batch_size),
        sim_matrix.diag(-batch_size)
    ])  # [8]

    # Denominator: sum over all negatives for each anchor
    # For each row, sum all similarities except self
    denom = torch.logsumexp(sim_matrix, dim=1)

    loss = - (pos_sim - denom).mean()
    print(f"NT-Xent loss: {loss.item():.4f}")
    print(f"Positive similarities: {pos_sim.detach().numpy()}")
    print(f"Log-denominators:      {denom.detach().numpy()}")
    print()


def demo_info_nce_explicit():
    """InfoNCE with explicit anchor, positive, and negative tensors."""
    print("=" * 60)
    print("DEMO: InfoNCE with Explicit Negatives")
    print("=" * 60)

    batch_size = 2
    num_negatives = 8
    dim = 16
    temperature = 0.07

    query = torch.randn(batch_size, dim)
    positive = torch.randn(batch_size, dim)
    negatives = torch.randn(num_negatives, dim)

    # Normalize
    query = F.normalize(query, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)

    # Positive similarities: [B]
    pos_sim = (query * positive).sum(dim=1) / temperature

    # Negative similarities: [B, N]
    neg_sim = torch.mm(query, negatives.t()) / temperature

    # Concatenate and compute log-softmax
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
    labels = torch.zeros(batch_size, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)

    print(f"InfoNCE loss: {loss.item():.4f}")
    print(f"Logits (first sample): {logits[0].detach().numpy()}")
    print()


def demo_triplet_loss():
    """Triplet loss with random and semi-hard mining."""
    print("=" * 60)
    print("DEMO: Triplet Loss")
    print("=" * 60)

    batch_size = 4
    dim = 8
    margin = 1.0

    anchor = torch.randn(batch_size, dim)
    positive = torch.randn(batch_size, dim)
    negative = torch.randn(batch_size, dim)

    # Standard triplet loss
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(d_pos - d_neg + margin).mean()

    print(f"Triplet loss: {loss.item():.4f}")
    print(f"Positive distances: {d_pos.detach().numpy()}")
    print(f"Negative distances: {d_neg.detach().numpy()}")
    print()


def demo_temperature_effect():
    """Show how temperature sharpens/softens the softmax distribution."""
    print("=" * 60)
    print("DEMO: Temperature Effect on Softmax")
    print("=" * 60)

    logits = torch.tensor([2.0, 1.0, 0.5, 0.0])

    for temp in [0.1, 0.5, 1.0, 5.0]:
        probs = F.softmax(logits / temp, dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        print(f"τ={temp:4.1f} → probs={[f'{p:.3f}' for p in probs.numpy()]}, entropy={entropy:.3f}")

    print("\nObservation: low τ → peaked (focus on max); high τ → uniform (all contribute)")
    print()


def demo_l2_normalization():
    """Show that L2 normalization makes dot product = cosine similarity."""
    print("=" * 60)
    print("DEMO: L2 Normalization")
    print("=" * 60)

    a = torch.randn(1, 8)
    b = torch.randn(1, 8)

    # Raw dot product
    raw_dot = torch.mm(a, b.t()).item()

    # L2 normalized dot product (= cosine similarity)
    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    cos_sim = torch.mm(a_norm, b_norm.t()).item()

    # Verify with explicit cosine formula
    explicit_cos = (a * b).sum() / (a.norm() * b.norm())

    print(f"Raw dot product:       {raw_dot:.4f}")
    print(f"Normalized dot product {cos_sim:.4f}")
    print(f"Explicit cosine sim:   {explicit_cos.item():.4f}")
    print(f"All equal? {abs(cos_sim - explicit_cos.item()) < 1e-6}")
    print()


def demo_alignment_uniformity():
    """Compute alignment and uniformity metrics for a batch."""
    print("=" * 60)
    print("DEMO: Alignment & Uniformity Metrics")
    print("=" * 60)

    batch_size = 32
    dim = 64

    # Simulate positive pairs
    base = torch.randn(batch_size, dim)
    z1 = F.normalize(base + 0.1 * torch.randn_like(base), dim=1)
    z2 = F.normalize(base + 0.1 * torch.randn_like(base), dim=1)

    # Alignment: average distance between positive pairs
    alignment = (z1 - z2).norm(dim=1).pow(2).mean()

    # Uniformity: average pairwise Gaussian potential
    z = torch.cat([z1, z2], dim=0)
    sq_dists = torch.cdist(z, z).pow(2)
    uniformity = sq_dists.mul(-2).exp().mean().log()

    print(f"Alignment (lower=better):  {alignment.item():.4f}")
    print(f"Uniformity (lower=better): {uniformity.item():.4f}")
    print()


def main():
    print("RUNNING ALL CONTRASTIVE LEARNING DEMOS\n")
    demo_nt_xent()
    demo_info_nce_explicit()
    demo_triplet_loss()
    demo_temperature_effect()
    demo_l2_normalization()
    demo_alignment_uniformity()


if __name__ == "__main__":
    main()

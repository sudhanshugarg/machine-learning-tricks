# Resource-Constrained Contrastive Training

## Overview

Contrastive learning is notoriously resource-hungry: large batch sizes, duplicate encoders, and full pairwise similarity matrices strain both GPU memory and cross-device communication. This chapter covers techniques to train effectively under memory and communication constraints.

---

## 1. The Memory Bottleneck

### What Consumes Memory?

For a batch size $B$ with $2B$ augmented samples and embedding dimension $D$:

| Component | Memory | Example ($B=2048, D=128$) |
|-----------|--------|---------------------------|
| Input activations | $2B \times C \times H \times W$ | ~1 GB for 224×224 RGB |
| Embeddings $z$ | $2B \times D$ | ~2 MB |
| Similarity matrix $S$ | $2B \times 2B$ | ~32 MB |
| Gradients (2 encoders if not shared) | $2 \times |\theta|$ | ~100 MB (ResNet-50) |

The real killer is the **encoder forward/backward**, not the similarity matrix.

---

## 2. Large-Batch Simulation (Gradient Accumulation)

### Problem

You want $B=4096$ but only have memory for $B=256$.

### Solution: Accumulate Gradients Over Sub-batches

```python
# Effective batch = 4096, physical batch = 256
accumulation_steps = 16

for step in range(accumulation_steps):
    x1, x2 = next(dataloader)  # batch size 256
    loss = nt_xent_loss(encoder, x1, x2)
    loss = loss / accumulation_steps
    loss.backward()              # gradients accumulate

optimizer.step()    # update after full effective batch
optimizer.zero_grad()
```

### Caveat

- The similarity matrix for in-batch negatives is only computed within the **physical batch**.
- With accumulation, each sub-batch sees only $2 \times 256 - 1 = 511$ negatives, not $2 \times 4096 - 1 = 8191$.
- **Fix**: Use a memory bank or queue to retain negatives across sub-batches (MoCo-style), or accept the weaker signal.

---

## 3. Gradient Checkpointing

### Technique

Instead of storing all intermediate activations for backward, recompute them on demand:

```python
from torch.utils.checkpoint import checkpoint

class Encoder(nn.Module):
    def forward(self, x):
        # Only store input; recompute blocks during backward
        x = checkpoint(self.block1, x)
        x = checkpoint(self.block2, x)
        x = checkpoint(self.block3, x)
        return x
```

### Trade-off

- **Memory**: ~50-70% reduction (only stores checkpoint boundaries, not all layers).
- **Speed**: ~20-30% slowdown due to recomputation.

---

## 4. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x1, x2 in dataloader:
    optimizer.zero_grad()
    with autocast():
        z1 = encoder(x1)
        z2 = encoder(x2)
        loss = nt_xent_loss(z1, z2)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Impact

- **Memory**: ~50% reduction for activations and weights (FP16 vs FP32).
- **Speed**: 2-3× faster on modern GPUs (Tensor Cores).
- **Gotcha**: The similarity matrix ($2B \times 2B$) should be computed in FP32 to prevent numeric overflow in softmax.

---

## 5. Cross-GPU Communication: All-Gather for In-Batch Negatives

### The Problem

In distributed training with $G$ GPUs, each GPU processes a local batch $B_{\text{local}}$. Without communication, each anchor only sees negatives from its **local batch**.

### Solution: Gather Embeddings Across GPUs

```python
import torch.distributed as dist

# Each GPU has z1_local [B_local, D] and z2_local [B_local, D]
# Gather from all GPUs to create global [B_total, D]

z1_list = [torch.zeros_like(z1_local) for _ in range(world_size)]
z2_list = [torch.zeros_like(z2_local) for _ in range(world_size)]

dist.all_gather(z1_list, z1_local)
dist.all_gather(z2_list, z2_local)

z1_global = torch.cat(z1_list, dim=0)
z2_global = torch.cat(z2_list, dim=0)

# Now compute NT-Xent with global negatives
loss = nt_xent_loss(z1_local, z2_local, z1_global, z2_global)
```

### Important: Stop-Gradient on Gathered Tensors

If you backprop through gathered tensors from other GPUs, you create cross-GPU gradient communication that defeats the purpose. The standard fix:

```python
# Only the local slice gets gradients
z1_global[rank * B_local : (rank+1) * B_local] = z1_local
# Other slices have no gradient history
```

Or use `torch.no_grad()` + `detach()` for non-local slices.

---

## 6. Model Parallelism & Sharding (FSDP)

For very large encoders (e.g., ViT-G/14):

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

encoder = FSDP(encoder, auto_wrap_policy=...)  # shards params across GPUs
```

FSDP shards model parameters and optimizer states across data-parallel workers. In contrastive learning, both the query and key encoders can be wrapped with FSDP.

---

## 7. Asymmetric Architectures (Smaller Key Encoder)

To save memory without losing signal:

```python
query_encoder = ViTBase()      # full size
key_encoder = ViTSmall()       # smaller, or shared with momentum
```

This is less common in recent work (SimCLR, MoCo v3 use identical architectures), but can be effective for extreme resource constraints.

---

## 8. Summary: Resource vs. Quality Trade-offs

| Constraint | Technique | Negatives Seen | Memory Saved | Quality Impact |
|------------|-----------|----------------|--------------|----------------|
| Small GPU | Gradient accumulation | Physical batch only | High | Moderate (fewer negatives) |
| Small GPU | Gradient checkpointing | Full | Medium | Low (slower training) |
| Small GPU | Mixed precision | Full | High | Very low |
| Multi-GPU | All-gather negatives | Global ($G \times B$) | None | None |
| Multi-GPU | FSDP sharding | Full | Very high | None |
| Any | Smaller key encoder | Full | Medium | Small |

---

## 9. Practical Training Recipe (Resource-Limited)

```python
# Single-GPU, 16 GB VRAM, ResNet-50, ImageNet
batch_size = 512          # physical batch
accumulation_steps = 8    # effective batch = 4096
use_amp = True
use_checkpointing = True

# Training loop
for epoch in range(100):
    for step, (x1, x2) in enumerate(dataloader):
        with autocast(enabled=use_amp):
            with torch.no_grad() if step % accumulation_steps != 0 else nullcontext():
                # ... forward pass ...
                loss = nt_xent_loss(...)

        (loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

Next: [FAQ](FAQ.md)

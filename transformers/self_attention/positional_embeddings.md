# Positional Embeddings: From Absolute to Rotary Position Embeddings (RoPE)

## Overview

Transformers process sequences of tokens in parallel, losing the sequential order information. Positional embeddings solve this by encoding the position of each token in the sequence. We'll explore the evolution from absolute positional embeddings to modern relative methods, with deep focus on **Rotary Position Embeddings (RoPE)**.

---

## 1. Why Positional Information Matters

### The Problem
In self-attention, the attention mechanism operates on token embeddings with no inherent awareness of their positions:

```
Input: "The cat sat on the mat"
Embeddings: [e_0, e_1, e_2, e_3, e_4, e_5]
```

Without position information, the model treats this identically to:
```
"mat the on sat cat The"  → [e_5, e_0, e_3, e_2, e_1, e_4]
```

The attention weights between tokens would be the same, causing loss of sequential structure.

### The Solution
Add position-aware information so the model knows:
- Where each token is in the sequence
- The relative distance between tokens
- How to adjust attention based on position

---

## 2. Absolute Positional Embeddings

### Basic Concept
The simplest approach: assign a unique embedding vector to each position and add it to token embeddings.

$$x_i' = x_i + p_i$$

where:
- $x_i$ = token embedding at position $i$
- $p_i$ = position embedding for position $i$
- $x_i'$ = final input to attention layer

### Sinusoidal Positional Encoding (Original Transformer)

The original "Attention is All You Need" paper introduced sinusoidal positional encodings:

$$PE(pos, 2j) = \sin\left(\frac{pos}{10000^{2j/d}}\right)$$

$$PE(pos, 2j+1) = \cos\left(\frac{pos}{10000^{2j/d}}\right)$$

where:
- $pos$ = position in sequence (0, 1, 2, ...)
- $j$ = dimension index (0 to $d/2$)
- $d$ = embedding dimension

### Why Sinusoidal?

1. **Bounded values**: Between -1 and 1 for numerical stability
2. **Unique per position**: Different frequencies create unique signatures for each position
3. **Relative position awareness**: The model can learn relative distances from the periodic patterns
4. **Extrapolation**: Works for sequences longer than training length (to some degree)

### Example
For dimension 4 and positions 0-3:

```
Position 0: [sin(0/1),    cos(0/1),    sin(0/100),    cos(0/100)]
           = [0.0,        1.0,         0.0,           1.0]

Position 1: [sin(1/1),    cos(1/1),    sin(1/100),    cos(1/100)]
           = [0.841,      0.540,       0.010,         1.0]

Position 2: [sin(2/1),    cos(2/1),    sin(2/100),    cos(2/100)]
           = [0.909,      -0.416,      0.020,         1.0]

Position 3: [sin(3/1),    cos(3/1),    sin(3/100),    cos(3/100)]
           = [0.141,      -0.990,      0.030,         0.9995]
```

### Learned Positional Embeddings

An alternative: treat position embeddings as learnable parameters, like token embeddings.

**Pros:**
- More flexible, can adapt to specific tasks
- Simple to implement

**Cons:**
- Only trained for sequence lengths seen during training
- Poor extrapolation to longer sequences
- Uses more parameters ($d \times \text{max\_seq\_len}$)

---

## 3. Relative Position Embeddings

### The Insight
Instead of encoding absolute position, encode *relative distance* between two positions. The relative position $i - j$ contains the information needed for attention:

$$\text{Attention}(Q_i, K_j) = \text{depends on } (i - j)$$

### T5-style Relative Position Bias
Add relative position bias directly to attention scores:

$$\text{Attention}(Q_i, K_j) = \frac{Q_i K_j^T}{\sqrt{d}} + b_{i-j}$$

where $b_{i-j}$ is a learnable bias for relative distance $i - j$.

**Advantages:**
- Works for sequences longer than training length
- Explicit modeling of relative distances
- Parameter efficient compared to learned absolute embeddings

**Limitations:**
- Loses some absolute position information
- Still requires explicit bias terms

---

## 4. Rotary Position Embeddings (RoPE)

### The Breakthrough Idea

RoPE (introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding") encodes position information by **rotating query and key vectors in a 2D plane**.

Core insight: For a 2D plane parameterized by angle $\theta$, rotating a vector by angle $\theta$ can encode position:

$$\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{pmatrix}$$

### The Formula

For a vector $v = (v_1, v_2, v_3, v_4, ..., v_d)$, apply rotation matrices to consecutive dimension pairs:

$$\text{RoPE}(v, m) = \begin{pmatrix} \cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \ldots \\ \sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \ldots \\ 0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \ldots \\ 0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \ldots \\ \vdots & \vdots & \vdots & \vdots & \ddots \end{pmatrix} v$$

where:
- $m$ = position in sequence
- $\theta_j = 10000^{-2j/d}$ (same frequency formula as sinusoidal PE)
- Rotation happens in pairs: dimensions $(1,2)$, $(3,4)$, $(5,6)$, etc.

### What Actually Happens (Step-by-step)

**Input:** Query vector $q$ at position $m$, Key vector $k$ at position $n$

**Step 1: Apply rotation to query**
```
For each dimension pair (2j, 2j+1):
  θ_j = 10000^(-2j/d)
  angle = m * θ_j
  
  q'[2j]   = q[2j] * cos(angle) - q[2j+1] * sin(angle)
  q'[2j+1] = q[2j] * sin(angle) + q[2j+1] * cos(angle)
```

**Step 2: Apply rotation to key**
```
For each dimension pair (2j, 2j+1):
  θ_j = 10000^(-2j/d)
  angle = n * θ_j
  
  k'[2j]   = k[2j] * cos(angle) - k[2j+1] * sin(angle)
  k'[2j+1] = k[2j] * sin(angle) + k[2j+1] * cos(angle)
```

**Step 3: Compute attention**
```
attention_score = dot_product(q', k') / sqrt(d)
                = dot_product(rotated_query, rotated_key) / sqrt(d)
```

### Why It Works: The Math

When we compute the dot product of two rotated vectors at different positions:

$$q'(m) \cdot k'(n) = q(m) \text{ rotated by } m\theta \cdot k(n) \text{ rotated by } n\theta$$

This **implicitly encodes the relative angle difference** $(m - n)\theta$:

$$= q(m) \cdot k(n) \text{ rotated by } (m-n)\theta$$

The dot product now depends on:
1. The base similarity between $q$ and $k$
2. The **relative position difference** $(m - n)$

This means position information is baked into the attention computation without explicit bias terms!

### Concrete Example

Say we have a 4-dimensional embedding and want to encode positions.

**Parameters:**
- $d = 4$, so 2 dimension pairs
- $\theta_0 = 10000^{0/4} = 1.0$
- $\theta_1 = 10000^{-2/4} = 100^{-0.5} \approx 0.1$

**Position 0 (no rotation):**
```
angle_0 = 0 * 1.0 = 0
angle_1 = 0 * 0.1 = 0

Rotation matrix is identity: no change to embeddings
```

**Position 1:**
```
angle_0 = 1 * 1.0 = 1 radian ≈ 57.3°
angle_1 = 1 * 0.1 = 0.1 radian ≈ 5.7°

Pair 1 (dims 0,1): rotate by 57.3°
Pair 2 (dims 2,3): rotate by 5.7°
```

**Position 2:**
```
angle_0 = 2 * 1.0 = 2 radians ≈ 114.6°
angle_1 = 2 * 0.1 = 0.2 radians ≈ 11.5°

Pair 1 (dims 0,1): rotate by 114.6°
Pair 2 (dims 2,3): rotate by 11.5°
```

**Key insight:** Different frequencies rotate at different rates, creating a unique position signature just like sinusoidal PE, but now encoded as **geometric rotations**.

---

## 5. Implementation Details

### Simple NumPy Implementation

```python
import numpy as np

def apply_rope(embeddings, positions):
    """
    Apply RoPE to embeddings.
    
    Args:
        embeddings: (batch_size, seq_len, d) or (seq_len, d)
        positions: (seq_len,) or scalar
    
    Returns:
        rope_embeddings: same shape as input
    """
    if embeddings.ndim == 2:
        seq_len, d = embeddings.shape
        embeddings = embeddings[np.newaxis, :, :]  # Add batch dim
    else:
        batch_size, seq_len, d = embeddings.shape
    
    # Create frequency bands
    inv_freq = 1.0 / (10000 ** (np.arange(0, d, 2).astype(np.float32) / d))
    
    # Create rotation angles for each position
    # shape: (seq_len, d//2)
    freqs = np.outer(positions, inv_freq)
    
    # Duplicate to match original dimension
    # shape: (seq_len, d)
    emb = np.concatenate([freqs, freqs], axis=-1)
    
    # Create rotation matrices
    cos_emb = np.cos(emb)
    sin_emb = np.sin(emb)
    
    # Apply rotation: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
    # Reshape embeddings for rotation: (batch, seq_len, d)
    out = embeddings.copy()
    
    # Apply rotations to dimension pairs
    for i in range(0, d, 2):
        x = embeddings[:, :, i]
        y = embeddings[:, :, i+1] if i+1 < d else 0
        
        out[:, :, i] = x * cos_emb[:, i] - y * sin_emb[:, i]
        if i+1 < d:
            out[:, :, i+1] = x * sin_emb[:, i] + y * cos_emb[:, i]
    
    return out.squeeze(0) if len(out.shape) == 3 and out.shape[0] == 1 else out
```

### PyTorch Implementation (Efficient)

```python
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x, positions=None):
        """
        Args:
            x: (batch_size, seq_len, d_model) or (seq_len, d_model)
            positions: (seq_len,) position indices. If None, uses 0, 1, 2, ...
        
        Returns:
            x_rotated: same shape as x
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, seq_len, d = x.shape
        device = x.device
        
        # Default positions
        if positions is None:
            positions = torch.arange(seq_len, device=device).float()
        
        # Compute rotation angles
        # (seq_len, 1) @ (1, d//2) -> (seq_len, d//2)
        freqs = torch.einsum("..., f -> ...f", positions, self.inv_freq)
        
        # Duplicate for all dimensions
        # (seq_len, d//2) -> (seq_len, d)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Create rotation matrix effect
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        # Apply rotation to dimension pairs
        # (x, y) -> (x*cos - y*sin, x*sin + y*cos)
        x_rot = torch.zeros_like(x)
        x_rot[:, :, 0::2] = x[:, :, 0::2] * cos_emb - x[:, :, 1::2] * sin_emb
        x_rot[:, :, 1::2] = x[:, :, 0::2] * sin_emb + x[:, :, 1::2] * cos_emb
        
        return x_rot
```

---

## 6. Advantages of RoPE

### Explicit Relative Position Encoding
RoPE directly encodes the relative position $(m - n)$ in the attention computation:
$$\text{Attention}(q_m, k_n) \propto \text{depends on } (m - n)$$

This is more direct than absolute position embeddings.

### Extrapolation
Frequencies are based on sine/cosine, so the model can attend to positions outside the training range (with some degradation).

### No Extra Parameters
Unlike learned positional embeddings, RoPE requires no learnable parameters—just pre-computed rotations.

### Efficiency
Rotations can be applied efficiently without materializing large position embedding matrices.

### Relative Bias Free
No explicit relative position bias terms needed—position information is encoded geometrically.

---

## 7. Comparison Table

| Feature | Absolute PE | Relative PE | RoPE |
|---------|------------|------------|------|
| **Encodes** | Absolute position | Relative distance | Relative distance (geometric) |
| **Parameters** | 0 (sinusoidal) or $O(d \times L)$ (learned) | $O(\text{max\_distance})$ | 0 |
| **Extrapolation** | Good (sinusoidal) | Moderate | Good |
| **Computation** | Additive | Additive bias to scores | Multiplicative (rotation) |
| **Relative info** | Implicit | Explicit | Explicit & geometric |

---

## 8. Edge Cases & Considerations

### Dimension Must Be Even
RoPE applies rotations to dimension pairs. If $d$ is odd, the last dimension is left unchanged (common approach: duplicate the last frequency).

### Sequence Length Generalization
While RoPE generalizes better than learned embeddings, there's still a performance drop for sequences **much longer** than training data. Recent work explores:
- **YaRN (Yet another RoPE extension)** - frequency warping for length extrapolation
- **NTK-aware scaling** - dynamic frequency scaling based on context length

### Batching & Variable Lengths
Position indices must be aligned correctly when processing batches with varying sequence lengths.

---

## 9. When to Use Each

**Use Absolute Positional Embeddings when:**
- Working with fixed sequence lengths
- Need simplicity and established practice
- Have computational constraints

**Use Relative Position Embeddings when:**
- Need variable-length sequences
- Want explicit relative distance modeling
- Have memory for bias terms

**Use RoPE when:**
- Need efficient relative position encoding
- Want good length extrapolation
- Working with modern large models (LLaMA, Qwen, Mistral use RoPE)
- Need to scale to very long contexts

---

## 10. References & Further Reading

1. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
   - Original RoPE paper
   
2. **YaRN: Efficient Context Window Extension of Large Language Models** (Peng et al., 2023)
   - Frequency scaling for length extrapolation

3. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original sinusoidal positional encodings

4. **Self-Attention with Relative Position Representations** (Shaw et al., 2018)
   - Relative position embeddings background

---

## Summary

- **Positional embeddings** solve the problem of encoding sequence order in parallel transformers
- **Absolute PE** (sinusoidal or learned) adds position information to tokens
- **Relative PE** encodes relative distances between token positions
- **RoPE** (Rotary Position Embeddings) encodes relative positions through geometric rotations of query and key vectors
- RoPE's key innovation: position information becomes a rotation angle that modulates attention scores multiplicatively, making relative position implicit in the dot product

The elegance of RoPE lies in its simplicity: by applying position-dependent rotations to embedding pairs, the model naturally learns to use relative distances without explicit bias terms or learnable parameters.

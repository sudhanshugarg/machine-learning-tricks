# Self-Attention Mechanism

## Overview

Self-attention (also called scaled dot-product attention) is the core mechanism that powers transformer models. It allows each position in a sequence to attend to all other positions, computing a weighted sum of values based on learned similarity scores.

## Problem Statement

Given a sequence of tokens (e.g., words), we want to compute a new representation for each token that incorporates information from all tokens in the sequence. The contribution of each token should be weighted by its relevance (learned through attention weights).

## Mathematical Formulation

### Scaled Dot-Product Attention

Given:
- **Query (Q)**: What we're looking for
- **Key (K)**: What each position represents
- **Value (V)**: What to aggregate when matches found

The attention mechanism computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:
- $d_k$ is the dimension of keys (scaling factor for numerical stability)
- $\text{softmax}$ converts scores to probability distribution
- Result is weighted sum of values

### Step-by-Step Computation

**Step 1: Compute scores**
$$\text{scores} = Q \cdot K^T$$

Shape: (sequence_length, sequence_length)

Each entry $(i,j)$ represents how much position $i$ attends to position $j$.

**Step 2: Scale by $\sqrt{d_k}$**
$$\text{scaled\_scores} = \frac{\text{scores}}{\sqrt{d_k}}$$

Prevents scores from becoming too large (which makes softmax flat).

**Step 3: Apply softmax**
$$\text{weights} = \text{softmax}(\text{scaled\_scores})$$

Converts scores to probabilities: each row sums to 1.

**Step 4: Multiply by values**
$$\text{output} = \text{weights} \cdot V$$

Weighted sum of value vectors.

## Multi-Head Attention

Instead of single attention head, use multiple in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

Where:
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

**Intuition:**
- Different heads learn different attention patterns
- Some focus on syntactic relationships, others on semantic
- Concatenate and project to final dimension

**Typical dimensions:**
- Model dimension: 512
- Number of heads: 8
- Head dimension: 512 / 8 = 64

## Attention Patterns

### Self-Attention Types

1. **Bidirectional**: Attend to past and future tokens
   - Used in encoder (BERT, RoBERTa)
   - Input: full sequence

2. **Causal/Autoregressive**: Attend only to past tokens
   - Used in decoder (GPT)
   - Mask future positions with $-\infty$

3. **Local/Windowed**: Attend within fixed window
   - Reduces computational complexity
   - Used in sparse transformers

## Computational Complexity

**Time Complexity:** $O(n^2 \cdot d_k)$
- $n$ = sequence length
- $d_k$ = key dimension
- Bottleneck for long sequences

**Space Complexity:** $O(n^2)$
- Store attention matrix

**Linear Attention Approximations:**
- Use kernel methods to approximate softmax
- Reduce complexity to $O(n \cdot d)$
- Examples: Linformer, Performer

## Key Properties

1. **Permutation Invariant**: Order info lost; added via positional encoding
2. **Fully Differentiable**: Learns end-to-end via backpropagation
3. **Interpretable**: Can visualize which tokens attend to each other
4. **Parallel**: Can compute all positions simultaneously (unlike RNNs)

## Positional Encoding

Without position information, attention is permutation-invariant. Add positional information:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

Where:
- $pos$ = position in sequence
- $i$ = dimension index
- $d$ = model dimension

**Properties:**
- Encodes absolute positions
- Allows model to learn relative positions
- Works for arbitrary sequence lengths

## Advantages Over RNNs

| Aspect | Attention | RNN |
|--------|-----------|-----|
| **Parallelization** | Full | Sequential |
| **Long-range dependencies** | Direct | Vanishing gradient |
| **Interpretability** | Attention weights | Hidden states opaque |
| **Computational cost** | $O(n^2)$ | $O(n)$ |
| **Speed** | Fast (parallel) | Slow (sequential) |

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Collapsed attention | Bad initialization | Layer norm, careful init |
| Head redundancy | Correlated heads | Head dropout, diversity loss |
| Position info lost | No positional encoding | Add position embeddings |
| Quadratic memory | Large sequences | Sparse/linear attention |


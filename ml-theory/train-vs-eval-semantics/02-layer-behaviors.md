# Layer-by-Layer Behavior and Common Pitfalls

## Overview

Different PyTorch layers react differently to the `self.training` flag. This section breaks down the five most important families of layers: Dropout, BatchNorm, LayerNorm, RNNs, and Transformers. For each, we explain the exact math, show the pitfall, and give a concrete symptom.

---

## 1. Dropout

### Behavior

**Training (`self.training = True`)**

For each element $x_i$ in the input tensor:

$$
y_i = \begin{cases}
0 & \text{with probability } p \\
\frac{x_i}{1-p} & \text{with probability } 1-p
\end{cases}
$$

The scaling by $\frac{1}{1-p}$ ensures the **expected value remains unchanged**:

$$
\mathbb{E}[y_i] = p \cdot 0 + (1-p) \cdot \frac{x_i}{1-p} = x_i
$$

**Evaluation (`self.training = False`)**

$$
y_i = x_i \quad \text{(pure identity)}
$$

### Common Pitfall

Forgetting `model.eval()` during inference means **every forward pass is different**:

```python
model.train()
x = torch.randn(1, 10)
out1 = model(x)
out2 = model(x)
print((out1 - out2).abs().max())  # > 0  (non-deterministic!)
```

**Symptom**: Predictions fluctuate wildly across identical inputs; ensemble-like variance without the ensemble.

**Real-world scenario**: A deployed image classifier returns "cat" on the first request and "dog" on the second for the exact same image.

### Monte-Carlo Dropout Note

Researchers sometimes **intentionally** leave dropout on during inference to approximate Bayesian uncertainty. This is called *Monte-Carlo Dropout*. It is a deliberate choice, not a bug—but it must be documented clearly.

---

## 2. BatchNorm

### Behavior

**Training (`self.training = True`)**

Given a mini-batch of activations $X \in \mathbb{R}^{B \times C \times \dots}$:

1. Compute batch statistics:

$$
\mu_B = \frac{1}{B} \sum_{i=1}^{B} x_i, \quad \sigma^2_B = \frac{1}{B} \sum_{i=1}^{B} (x_i - \mu_B)^2
$$

2. Normalize:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}
$$

3. Update **running statistics** with momentum $m$ (default 0.1):

$$
\mu_{\text{run}} \leftarrow (1-m) \cdot \mu_{\text{run}} + m \cdot \mu_B \\
\sigma^2_{\text{run}} \leftarrow (1-m) \cdot \sigma^2_{\text{run}} + m \cdot \sigma^2_B
$$

**Evaluation (`self.training = False`)**

Use frozen running statistics—**no update**:

$$
\hat{x}_i = \frac{x_i - \mu_{\text{run}}}{\sqrt{\sigma^2_{\text{run}} + \epsilon}}
$$

### Common Pitfall #1: Inference in Train Mode

If you forget `model.eval()`, BatchNorm continues updating its running stats every time you run a forward pass. Over thousands of inference requests, the running statistics drift away from their training values.

**Symptom**: Accuracy degrades slowly in production even though the checkpoint was perfect in validation.

### Common Pitfall #2: Batch Size 1 in Train Mode

During training, BatchNorm needs a batch to compute meaningful statistics. With batch size 1:

$$
\mu_B = x_1, \quad \sigma^2_B = 0
$$

This causes division by (near) zero after normalization:

```python
bn = nn.BatchNorm1d(10)
x = torch.randn(1, 10)
out = bn(x)   # ≡ (x - x) / sqrt(0 + ε) ≈ 0  (degenerate!)
```

**Symptom**: Outputs collapse to near-zero; model predicts the same class for every input.

### PyTorch's `track_running_stats`

```python
nn.BatchNorm2d(64, track_running_stats=False)
```

When `False`, there are no running stats. In eval mode, it falls back to **batch statistics**—which is dangerous if your "batch" is a single sample. Always leave this `True` (default) for inference.

---

## 3. LayerNorm

### Behavior

LayerNorm computes statistics **per sample**, across the feature dimension:

$$
\mu = \frac{1}{D} \sum_{j=1}^{D} x_j, \quad \sigma^2 = \frac{1}{D} \sum_{j=1}^{D} (x_j - \mu)^2
$$

Then normalizes:

$$
y_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_j + \beta_j
$$

**Key point**: Because $\mu$ and $\sigma^2$ are computed from the single sample itself, there is **no batch-dependent statistic** to cache or update.

### Train vs Eval

| Mode | Behavior |
|------|----------|
| `train()` | Computes $\mu, \sigma^2$ from input; applies transformation |
| `eval()` | Computes $\mu, \sigma^2$ from input; applies transformation |

**They are identical.**

### Common Pitfall

Developers sometimes assume *all* normalization layers need special handling during inference and write brittle wrappers around LayerNorm. This is unnecessary.

However, there is one subtlety: if you implement a **custom** LayerNorm that caches statistics (e.g., for efficiency in seq2seq), you must manage that cache yourself—PyTorch's built-in `nn.LayerNorm` does not.

---

## 4. RNNs (LSTM / GRU)

### Behavior

PyTorch RNN layers accept a `dropout` argument:

```python
nn.LSTM(input_size=128, hidden_size=256, num_layers=3, dropout=0.3)
```

- **Training**: Dropout is applied to the **outputs of intermediate layers** (between layer 1 → 2 and 2 → 3). The final layer output is never dropped.
- **Evaluation**: All dropout is disabled; outputs flow through unchanged.

### Common Pitfall

If you evaluate in train mode, your sequence model's hidden states become stochastic. This is especially harmful in:
- **Autoregressive generation**: Sampling becomes noisier than intended.
- **Speech recognition**: WER (word error rate) jumps unpredictably.

---

## 5. Transformers

### Behavior

Transformer blocks in PyTorch (`nn.TransformerEncoderLayer`, `nn.MultiheadAttention`) contain two sources of train/eval sensitivity:

1. **Attention Dropout**

   During training, the scaled dot-product attention scores are dropped out before softmax:

   ```python
   attn_output_weights = F.softmax(attn_scores, dim=-1)
   attn_output_weights = F.dropout(attn_output_weights, p=dropout, training=self.training)
   ```

   In eval mode, the dropout is skipped, giving deterministic attention maps.

2. **Feed-Forward Dropout**

   Standard Dropout applied after the first linear layer of the FFN block. Same behavior as regular Dropout.

3. **LayerNorm**

   As discussed above, unchanged between modes.

### Common Pitfall

When exporting a Transformer to ONNX or TorchScript, if the model is still in `train()` mode, the dropout nodes are baked into the graph. The exported model will produce different outputs every run—even though the graph is "static."

**Symptom**: Exported model `model.onnx` gives inconsistent results across inference frameworks.

---

## Comparison Summary

| Layer | Train Behavior | Eval Behavior | Running Stats? | Pitfall Severity |
|-------|---------------|---------------|----------------|------------------|
| **Dropout** | Random zeros | Identity | No | High (silent randomness) |
| **BatchNorm** | Batch stats + update running | Frozen running stats | **Yes** | Critical (stat drift) |
| **LayerNorm** | Per-sample stats | Per-sample stats | No | None (safe to ignore) |
| **RNN dropout** | Inter-layer dropout | Identity | No | Medium (seq noise) |
| **Transformer attention dropout** | Random attention masks | Deterministic | No | High (export bugs) |

Next: [03 — Testing & Deployment](03-testing-deployment.md)

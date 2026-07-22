# Solution: Debugging a Transformer That Won't Learn

This solution is organized as **hints first**, then **systematic diagnosis**, then the **fixes and corrected code**.

> **Note on versions**: The current `code.py`/`problem.md` already have positional encoding, causal masking, `zero_grad()`, and LR warmup correctly wired up (these were the bugs in an earlier version of this exercise — see the bottom of this file for their diagnoses, kept for reference). The **remaining, currently-active bug** is a vanishing-gradient problem introduced by the model's architecture, covered in Bug 1 below.

---

## Part 1: Progressive Hints

<details>
<summary><b>Hint 1 — Look at the gradient norm printout</b> (click to expand)</summary>

After each epoch, the training loop prints the gradient norm of `token_emb.weight` and every layer's `self_attn.in_proj_weight` / `linear1.weight`. Look at how those numbers change as you go from `token_emb.weight` up through `transformer.layers.7`. Is there a trend? What would cause a *systematic, monotonic* trend like that (as opposed to noise)?

</details>

<details>
<summary><b>Hint 2 — The activation function</b> (click to expand)</summary>

`nn.TransformerEncoderLayer` is constructed with `activation=torch.sigmoid`. What is the maximum value of the derivative of the sigmoid function? Now consider what happens when you multiply many numbers, each at most that large, together via the chain rule across 8 stacked layers.

</details>

<details>
<summary><b>Hint 3 — Post-LN vs. Pre-LN</b> (click to expand)</summary>

`nn.TransformerEncoderLayer` defaults to `norm_first=False` (Post-LN): each sub-layer's output is normalized *after* being added to the residual stream, rather than the sub-layer operating on a normalized copy of the stream while the raw residual passes through untouched. Does the gradient have a clean, un-shrunk path back to the input in this configuration, or does it have to pass back through every LayerNorm and sigmoid on the way?

</details>

<details>
<summary><b>Hint 4 — Depth</b> (click to expand)</summary>

The model now has `num_layers=8`. Would this same bug be as visible, or as harmful, with `num_layers=2`? What does that tell you about *when* saturating activations become a real problem in practice?

</details>

---

## Part 2: Systematic Debugging Methodology

When a transformer won't learn, follow this diagnostic order:

```
1. ARCHITECTURE           → Saturating activations? Post-LN with many layers? Is causal mask applied? Is positional encoding added?
2. GRADIENT FLOW          → Do per-layer gradient norms grow/shrink monotonically with depth?
3. DATA / TOKENIZATION    → Is the tokenizer reversible? Are <PAD> tokens leaking?
4. LOSS / TARGETS         → Are labels shifted correctly? Is padding masked in loss?
5. OPTIMIZER              → LR too high? Warmup missing? Gradient clipping?
6. GENERATION             → Does greedy decoding work? Are logits sensible?
7. ATTENTION PATTERNS     → Is attention attending to future tokens? To <PAD>?
```

**Empirical tools to use at each step:**

```python
# 1. Gradient norms by depth — the key diagnostic for vanishing gradients
for name, p in model.named_parameters():
    if ("in_proj_weight" in name or "linear1.weight" in name or name == "token_emb.weight") and p.grad is not None:
        print(f"{name}: {p.grad.norm():.6f}")
# Healthy: roughly the same order of magnitude at every depth.
# Vanishing: norms shrink by an order of magnitude (or more) going from the last
#            layer back toward token_emb.weight.

# 2. Verify tokenization round-trip
sample = "to be or not to be"
toks = tokenize(sample)
print(detokenize(toks) == sample)  # Should be True

# 3. Check if positional encoding is actually added
for name, module in model.named_modules():
    if "pos" in name.lower():
        print(name, module)

# 4. Inspect causal mask
mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
print(mask)  # Upper-triangular True = masked positions

# 5. Check if attention is causal by inspecting weights
with torch.no_grad():
    out = model(input_ids)
    attn = model.transformer.layers[0].self_attn(input_ids)
    # Inspect attention weights here
```

---

## Part 3: Bug-by-Bug Diagnosis

### Bug 1: Vanishing Gradients from a Deep Post-LN Stack + Saturating Activation (Severity: Critical)

**Symptom**: Loss decreases, but slowly and unevenly, plateauing well above what the model's capacity should allow (e.g. ~2.7–2.8 instead of continuing to drop toward <1.0). The per-epoch gradient-norm printout shows a clear, *monotonic* pattern: small near `token_emb.weight`, growing steadily through `transformer.layers.0` up to `transformer.layers.7`.

Running the actual training loop in this repo for 3 short epochs and printing gradient norms confirms this directly:

```
token_emb.weight:                            grad_norm=0.000877
transformer.layers.0.self_attn.in_proj_weight: grad_norm=0.002696
transformer.layers.0.linear1.weight:           grad_norm=0.000859
transformer.layers.1.self_attn.in_proj_weight: grad_norm=0.002645
...
transformer.layers.6.self_attn.in_proj_weight: grad_norm=0.024293
transformer.layers.7.self_attn.in_proj_weight: grad_norm=0.047647
transformer.layers.7.linear1.weight:           grad_norm=0.013487
```

That's a **~54× ratio** between `transformer.layers.7.self_attn.in_proj_weight` (closest to the loss) and `token_emb.weight` (furthest from it, closest to the input). The embedding table — the very first thing the model needs to learn well — is receiving a gradient signal roughly 1/54th the size of the last attention block. Effectively, only the top couple of layers are actually being trained; everything below them updates too slowly to matter within a normal training budget.

**Theory**:

Two compounding causes:

**(a) `activation=torch.sigmoid` in the feedforward block.** Each `TransformerEncoderLayer`'s feedforward sub-layer computes $\text{Linear}_2(\sigma(\text{Linear}_1(x)))$, where $\sigma$ is the sigmoid. The derivative of sigmoid is $\sigma'(z) = \sigma(z)(1-\sigma(z))$, which has a **maximum value of exactly $0.25$** (at $z=0$) and is smaller everywhere else — and once a unit's pre-activation drifts to $|z| \gtrsim 4$, $\sigma'(z)$ is already below $0.02$. By the chain rule, backpropagating through $L$ such activations multiplies the gradient by up to $0.25^L$ — for $L=8$ layers, that's a factor of at most $0.25^8 \approx 1.5\times 10^{-5}$ in the worst case, and typically much smaller in practice since most units aren't sitting exactly at $z=0$. This is the textbook vanishing-gradient mechanism that motivated the field's move from sigmoid/tanh to ReLU/GELU (whose derivatives are $1$ over most of their range) for deep networks in the first place.

**(b) `norm_first=False` (Post-LN, the `nn.TransformerEncoderLayer` default) offers no escape route.** In a Pre-LN block, each sub-layer computes $x + \text{Sublayer}(\text{LayerNorm}(x))$ — gradients have a direct, un-shrunk path straight through the `+` back to every earlier layer via the residual stream, and the sub-layer's (possibly vanishing) gradient is just *added* on top of that clean signal. In a Post-LN block, the computation is $\text{LayerNorm}(x + \text{Sublayer}(x))$ — the residual sum happens *before* normalization, so the gradient has to flow back through the LayerNorm at every single layer, and there's no unimpeded shortcut around the sigmoid-saturated sublayer. This is exactly the failure mode documented in Xiong et al. (2020), *"On Layer Normalization in the Transformer Architecture"*: Post-LN transformers are demonstrably harder to train at depth than Pre-LN ones, and the effect is most visible in exactly the metric shown above — gradient norm growing sharply from the bottom layers to the top.

**(c) Depth amplifies both.** With only `num_layers=2`, $0.25^2 = 0.0625$ is a meaningful but survivable shrinkage. At `num_layers=8`, the same per-layer factor compounds into the ~54× effect measured above. This is why this bug wasn't apparent in earlier, shallower iterations of this exercise — saturating activations are a *latent* bug that only becomes catastrophic once you scale up depth, which is exactly what happens when a toy 2-layer prototype gets scaled toward a real model.

**Fix**: Replace the saturating activation with GELU (or ReLU), and switch to Pre-LN (`norm_first=True`) so the residual stream carries an unshrunk gradient path regardless of what happens inside each sub-layer:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    activation="gelu",     # FIX: was activation=torch.sigmoid
    norm_first=True,       # FIX: was the norm_first=False default (Post-LN)
    batch_first=True,
)
self.transformer = nn.TransformerEncoder(
    encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)  # Pre-LN stacks need a final norm
)
```

Re-running the same 3-epoch measurement with this fix applied:

```
token_emb.weight:                            grad_norm=0.021102
transformer.layers.0.self_attn.in_proj_weight: grad_norm=0.023362
transformer.layers.0.linear1.weight:           grad_norm=0.035998
...
transformer.layers.6.self_attn.in_proj_weight: grad_norm=0.013841
transformer.layers.7.self_attn.in_proj_weight: grad_norm=0.014547
transformer.layers.7.linear1.weight:           grad_norm=0.021856
```

The ratio between the largest and smallest gradient norms drops from ~54× to **~1.6×** — every layer, including the embedding table, now receives a comparably-sized, trainable gradient signal. Loss after the same 3 short epochs also drops further (2.39 vs. 2.76 with the buggy version), confirming the fix isn't just cosmetic.

**Note**: when you only have the freedom to change one of the two, `norm_first=True` alone recovers most of the benefit (the residual path bypasses the saturating activation entirely), but fixing the activation too removes the root cause rather than just routing around it — do both.

---

### Bug 2 (already fixed in the current code — kept for reference): Positional Encoding Never Added (Severity: Critical)

**Symptom**: Model converges to a "bag-of-characters" predictor. Generated text has correct character frequencies but no coherent order (e.g., `"oeb t t..."`).

**Theory**:
Self-attention without position information is **permutation-invariant**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For any permutation matrix $P$:

$$
\text{Attention}(XP, XP, XP) = \text{Attention}(X, X, X) \cdot P
$$

The output at each position depends on the *set* of tokens, not their order. Without positional encodings, `"abc"` and `"cba"` produce identical contextualized representations up to permutation.

**Fix**: Add positional encoding:

```python
out = tok + pos  # was: out = tok
```

**Why learned positional embeddings work**: They break permutation invariance by assigning a unique vector to each absolute position.

---

### Bug 3 (already fixed in the current code — kept for reference): No Causal Mask (Severity: Critical)

**Symptom**: Training loss drops to near-zero quickly (~0.1), but generated text is random gibberish.

**Theory**:
The autoregressive training objective requires that predicting token $t_{i+1}$ only uses $t_1, \dots, t_i$. Without a causal mask, self-attention computes:

$$
\alpha_{ij} = \frac{\exp(q_i^T k_j)}{\sum_l \exp(q_i^T k_l)} \quad \text{for ALL } j \in [1, S]
$$

During training, the model can attend to $t_{i+1}$ when predicting position $i$. This turns next-token prediction into a trivial copying task. The model achieves near-perfect training loss but never learns to **predict the future from the past**.

At inference time, the model is forced to attend only to generated tokens (no future tokens available), creating a severe **train/test distribution mismatch** (exposure bias on steroids).

**Fix**: Generate a causal mask and pass it to `TransformerEncoder`:

```python
def generate_causal_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask  # True = mask out

# In forward:
mask = generate_causal_mask(s, x.device)  # [S, S]
out = self.transformer(out, mask=mask, is_causal=True)  # is_causal=True also works in PyTorch 2.0+
```

**Note**: `nn.TransformerEncoder` with causal mask still processes all positions in parallel — the mask only hides future positions in attention.

---

### Bug 4 (already fixed in the current code — kept for reference): Missing `zero_grad()` (Severity: Critical)

**Symptom**: After ~10 batches, loss becomes `NaN`. Gradient norms explode.

**Theory**:
Identical to the CNN case. Gradients accumulate:

$$
\theta_{t+1} = \theta_t - \eta \sum_{i=1}^{t} g_i
$$

With `lr=1e-2` and transformer gradients (which can be large early in training), this causes immediate divergence.

**Fix**: Call `optimizer.zero_grad()` before each backward.

---

### Bug 5 (already fixed in the current code — kept for reference): No Learning Rate Warmup (Severity: High)

**Symptom**: Even after fixing bugs 1-3, loss plateaus early or oscillates. Adam's adaptive LR is erratic in the first ~1000 steps.

**Theory**:
Adam maintains second-moment estimates:

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

At $t=1$ (cold start), $v_1 = (1-\beta_2) g_1^2$. The adaptive step size is:

$$
\Delta \theta = -\eta \frac{m_1}{\sqrt{v_1} + \epsilon}
$$

Since $v_1$ is small (biased toward 0), the effective step size is:

$$
\|\Delta \theta\| \approx \frac{\eta}{\sqrt{(1-\beta_2)} \|g_1\|} \cdot \|m_1\| \gg \eta
$$

This is typically 10-100× larger than the nominal LR. On transformers, this causes:
- Attention logits to explode
- Weight matrices to diverge
- Loss spikes to infinity

**Warmup** linearly scales LR from 0 to $\eta_{\max}$ over $T_{\text{warmup}}$ steps, giving $v_t$ time to warm up:

```python
# Warmup scheduler
def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step / warmup_steps)
    return max_lr
```

**Fix**: Use a learning rate warmup schedule:

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_schedule(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_schedule(step))
```

---

### Bug 6 (not applicable to this toy dataset — general checklist item): Loss Includes Padding (Severity: Moderate)

**Symptom**: If using batched sequences with padding (not in this toy example, but common in real training), the model learns to predict `<PAD>` very well but performs poorly on real tokens.

**Theory**:
Padding positions carry no semantic information. If they are included in the cross-entropy loss:

$$
\mathcal{L} = -\frac{1}{B \cdot S} \sum_{b=1}^{B} \sum_{i=1}^{S} \log p(y_{bi} | x_{bi})
$$

The loss is dominated by `<PAD>` predictions (often ~30-50% of tokens), diluting gradients for real tokens.

**Fix**: Use `ignore_index` in `CrossEntropyLoss`:

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

This masks out padding positions from both numerator and denominator.

---

### Bug 7 (design caveat, not a hard bug): `TransformerEncoder` Used for Decoder-Only LM (Severity: Moderate)

**Symptom**: Even with causal mask, the architecture is slightly wrong for autoregressive generation. `nn.TransformerEncoder` adds LayerNorm *after* the residual in default settings, which is fine, but more importantly, it lacks the causal mask handling by default.

Actually, the bigger issue: `nn.TransformerEncoderLayer` applies multi-head self-attention correctly. With a causal mask, it functions as a decoder. But the more standard approach for an LM is to use `nn.TransformerDecoderLayer` or implement causal attention manually. Using `TransformerEncoder` + causal mask is acceptable but non-standard.

For this exercise, the **causal mask fix** is sufficient. In production, prefer `TransformerDecoder` or a custom `CausalSelfAttention` module.

---

## Part 4: Corrected Code

```python
"""
Fixed decoder-only transformer for language modeling.
Each fix is marked with a comment referencing the bug number.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------------
vocab = list("abcdefghijklmnopqrstuvwxyz ") + ["<PAD>", "<EOS>"]
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}
PAD_IDX = char2idx["<PAD>"]
EOS_IDX = char2idx["<EOS>"]
VOCAB_SIZE = len(vocab)


def tokenize(text: str):
    return [char2idx.get(c, char2idx[" "]) for c in text.lower()]


def detokenize(tokens):
    return "".join([idx2char[t] for t in tokens])


# ------------------------------------------------------------------
# Fixed Transformer LM
# ------------------------------------------------------------------
class FixedTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=8, dim_feedforward=256, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # FIX 2: Use sinusoidal positional encoding (no learned parameters, generalizes)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

        # FIX 1: Non-saturating activation (GELU, not sigmoid) + Pre-LN (norm_first=True)
        # so gradients have a clean, unshrunk residual path through all 8 layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",     # FIX 1: was activation=torch.sigmoid
            batch_first=True,
            norm_first=True,  # FIX 1: Pre-norm for training stability at depth
        )
        # FIX 1 (cont.): Pre-LN stacks need a final LayerNorm — otherwise the last
        # residual branch is never normalized before hitting the LM head.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Standard GPT-style weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        b, s = x.shape
        tok = self.token_emb(x) * math.sqrt(self.d_model)  # scale embeddings

        # FIX 2: Add positional encoding
        pos = self.pe[:, :s, :]
        out = tok + pos

        # FIX 3: Create causal mask (True = masked)
        causal_mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1).bool()
        out = self.transformer(out, mask=causal_mask, is_causal=True)

        logits = self.lm_head(out)
        return logits


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class CharDataset(Dataset):
    def __init__(self, text, block_size=32):
        self.data = tokenize(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_text = (
        "to be or not to be that is the question " * 200
        + "all the world is a stage and all the men and women merely players " * 200
        + "now is the winter of our discontent made glorious summer " * 200
    )

    dataset = CharDataset(raw_text, block_size=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = FixedTransformerLM(VOCAB_SIZE).to(device)

    # FIX 5: Lower base LR + add warmup
    warmup_steps = 200
    max_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.01)

    step = 0
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # FIX 6: Ignore padding in loss (defensive even if no padding here)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        count = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # FIX 4: Zero gradients
            optimizer.zero_grad()

            logits = model(x)  # [B, S, V]

            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            count += 1
            step += 1

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    model.eval()
    prompt = tokenize("women and men ")
    input_ids = torch.tensor([prompt], dtype=torch.long).to(device)

    for _ in range(50):
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    generated = detokenize(input_ids[0].cpu().tolist())
    print(f"\nGenerated: {generated}")


if __name__ == "__main__":
    train()
```

---

## Part 5: Answers to Discussion Questions

### Q1: Vanishing Gradients in Deep Stacks

Backpropagation through the feedforward sub-layer's sigmoid activation multiplies the gradient by $\sigma'(z) = \sigma(z)(1-\sigma(z))$ at every layer. This function peaks at $z=0$ with value exactly $0.25$, and decays toward $0$ as $|z|$ grows — so **every** pass through a sigmoid can only shrink the gradient, never preserve or amplify it. By the chain rule, stacking $L$ such activations multiplies the gradient reaching the input by a product of $L$ factors each $\le 0.25$:

$$
\left\|\frac{\partial \mathcal{L}}{\partial x_0}\right\| \le \left(\prod_{l=1}^{L} \sigma'(z_l)\right) \left\|\frac{\partial \mathcal{L}}{\partial x_L}\right\| \le 0.25^{L} \left\|\frac{\partial \mathcal{L}}{\partial x_L}\right\|
$$

For $L=8$, the worst-case bound is $0.25^8 \approx 1.5\times10^{-5}$ — and in practice, once activations drift away from $z=0$ (which they do almost immediately after a few optimizer steps), $\sigma'(z)$ is far smaller than $0.25$, so the real shrinkage is even worse. This is the measured ~54× gradient-norm gap between the last transformer layer and the token embedding table (see Bug 1 above).

**Post-LN makes it worse, not better**, because it removes the one thing that could have bypassed this shrinkage: a clean residual gradient path. A Pre-LN block computes $x + \text{Sublayer}(\text{LN}(x))$, so $\partial x_{\text{out}}/\partial x = I + \partial \text{Sublayer}/\partial x$ — the identity term guarantees gradients of magnitude $\ge 1$ flow straight back through every layer regardless of how badly the sublayer's own gradient has vanished. Post-LN instead computes $\text{LN}(x + \text{Sublayer}(x))$: the residual sum happens *before* normalization, so the gradient must pass back through the LayerNorm's own Jacobian at every layer, and there is no unimpeded shortcut around the (possibly near-zero) sublayer gradient. With 8 stacked Post-LN blocks, the sigmoid's vanishing effect compounds layer after layer with nothing to counteract it — exactly the effect Xiong et al. (2020) identify as the reason Post-LN transformers require careful warmup and are harder to train at depth than Pre-LN ones.

---

### Q2: Why Does No Causal Mask Cause Generation Failure Despite Low Training Loss?

This is a classic case of **train/test distribution mismatch**.

During training, the model is allowed to "peek" at the target token $t_{i+1}$ when predicting position $i$. It learns the trivial identity mapping:

$$
p(t_{i+1} | t_1, \dots, t_i, t_{i+1}) \approx \delta_{t_{i+1}}
$$

At inference, the target token is **not available**. The model must predict from $\hat{t}_1, \dots, \hat{t}_i$, which are themselves noisy predictions. This is fundamentally different from the training distribution:

$$
\underbrace{p(t_{i+1} | t_1, \dots, t_i)}_{\text{inference}} \neq \underbrace{p(t_{i+1} | t_1, \dots, t_i, t_{i+1})}_{_{\text{training}}}
$$

This is related to **exposure bias** but much more severe: the model was never trained on its own predictions because it never had to.

---

### Q3: Permutation Invariance Without Positional Encoding

Self-attention computes:

$$
\text{Attn}(X) = \text{softmax}\left(\frac{XW_Q (XW_K)^T}{\sqrt{d_k}}\right) XW_V
$$

For any permutation matrix $P$:

$$
\text{Attn}(XP) = \text{Attn}(X) \cdot P
$$

The output is simply permuted. This means the contextualized representation of the token `"a"` is identical whether it appears at position 0 or position 5. The model can only learn **unigram statistics**, not sequences.

**Positional encodings break this invariance** by injecting position-dependent signals:

$$
X_{\text{final}} = \text{Attn}(X + P)
$$

Now $\text{Attn}(X_{\text{perm}} + P) \neq \text{Attn}(X + P)_{\text{perm}}$ because $P$ is not permuted along with $X$.

---

### Q4: Why Adam Needs Warmup

Adam's update rule at step $t$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

The bias-corrected estimates are:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

At step 1, $\hat{v}_1 = (1-\beta_2) g_1^2 / (1-\beta_2) = g_1^2$. The update is:

$$
\Delta \theta = -\eta \frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon} = -\eta \frac{g_1}{|g_1| + \epsilon} \approx -\eta \cdot \text{sign}(g_1)
$$

If $g_1$ is large (common at init), the step is $\approx \eta$ regardless of gradient magnitude — but in the *cold-start* regime before bias correction, the raw (uncorrected) $v_1$ is tiny, making the step enormous.

Warmup limits the theoretical maximum step size during this cold-start phase, preventing attention logits from exploding and allowing the second-moment accumulator to stabilize.

---

### Q5: Trivial Solution When Not Shifting Labels

If we train the model to predict $t_i$ from $t_i$ (no shift), the optimal network achieves zero loss by simply copying its input. The cross-entropy lower bound is:

$$
\mathcal{L}_{\text{CE}} \geq H(p^*) = 0
$$

where $p^*$ is a Dirac delta at the input token. The network learns the identity function, which is useless for generation because the model is never asked to predict an unseen token.

---

### Q6: Diagnosing Repetitive Generation Loops

Repetitive loops (e.g., `"the the the..."`) indicate **mode collapse in the output distribution**. Diagnostic steps:

1. **Inspect attention patterns**: Use `register_forward_hook` to extract attention weights. Look for attention heads that heavily attend to the most recent token (diagonal dominance with tail bias).

2. **Check output entropy**: If entropy of the output distribution drops to near-zero after a few tokens, the model has become overconfident about repeating.

3. **Logits analysis**: Print top-5 logits at each step. If the gap between top-1 and top-2 is huge (>5 in logit space), the temperature is too low or the model is stuck in a local attractor.

4. **Fixes**:
   - Increase sampling temperature or use top-p/top-k sampling.
   - Add repetition penalty: $\text{logit}'_i = \text{logit}_i - c \cdot \mathbb{1}[i \in \text{generated}]$.
   - Use contrastive decoding or diverse beam search.

---

## Part 6: Prevention Checklist

Before training any transformer LM, verify:

- [ ] **Activation function** is non-saturating (GELU/ReLU, not sigmoid/tanh) in deep stacks
- [ ] **Pre-LN** (`norm_first=True`) is used for stacks deeper than a couple of layers, with a final `LayerNorm` before the output head
- [ ] **Positional encoding** is added to token embeddings
- [ ] **Causal mask** is applied (no future-token leakage)
- [ ] **Labels** are shifted by 1 for next-token prediction
- [ ] **Loss** ignores padding tokens (`ignore_index=PAD_IDX`)
- [ ] **LR warmup** is configured (linear or cosine)
- [ ] **Gradient clipping** is enabled (`max_norm=1.0`)
- [ ] **zero_grad()** is called every batch
- [ ] **Weight initialization** matches architecture (e.g., GPT-style ~N(0, 0.02))
- [ ] **Tokenizer** round-trip is verified (`detokenize(tokenize(text)) == text`)

---

## Summary Table

| Bug | Symptom | Root Cause (Theory) | Fix |
|-----|---------|---------------------|-----|
| **Sigmoid + Post-LN at depth** | Loss plateaus; grad norms shrink ~54× from top layer to embedding | $\sigma'(z)\le0.25$ compounds over $L$ layers; Post-LN has no clean residual gradient path | GELU activation + `norm_first=True` + final `LayerNorm` |
| **No pos enc** | Random character order | Permutation invariance of self-attention | Add `tok + pos` |
| **No causal mask** | Near-zero loss, garbage generation | Train/test distribution mismatch; trivial future-peeking | `torch.triu` mask |
| **No zero_grad** | NaN, explosion | Gradient accumulation $O(t)$ | Call `zero_grad()` |
| **No warmup** | Loss spikes, unstable | Cold-start $v_t$ bias in Adam | Linear LR warmup |
| **No pad ignore** | "Predicts" padding well | Loss dominated by pad tokens | `ignore_index=PAD_IDX` |
| **TransformerEncoder** | Non-standard for LM | Encoder layer defaults | Add causal mask; or use decoder |

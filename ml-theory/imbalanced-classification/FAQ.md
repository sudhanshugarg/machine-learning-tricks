# FAQ: Weight Initialization for `nn.Linear` Layers

Practical questions about initializing the `nn.Linear` layers in the `FraudMLP` architecture (and any ReLU MLP that ends in a sigmoid/BCE head). The numbers below come from actually running the snippets against this exact architecture in PyTorch 2.11.

The architecture in question:

```python
self.model = nn.Sequential(
    nn.Linear(input_dims, 128),
    nn.BatchNorm1d(128),
    nn.Dropout(p=0.2),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.Dropout(p=0.2),
    nn.ReLU(),
    nn.Linear(32, 1),
)
```

---

### Q: How do I initialize the `nn.Linear` layers with a specific mean and variance?

**A:** Use `nn.init.trunc_normal_` (or `nn.init.normal_`) — both take **`mean` and `std`, not variance**. So if you want variance $v$, pass `std = sqrt(v)`. To reach the `nn.Linear` layers *inside* an `nn.Sequential`, recurse with `.apply()` or `.modules()` (a `Sequential` is just a container, so both walk into it):

```python
def _init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

self.model.apply(_init)
```

This hits all three `nn.Linear` layers: `input_dims→128`, `128→32`, `32→1`.

**Two traps people hit:**

1. **`std`, not variance.** `nn.init.normal_(..., std=0.02)` gives variance $0.02^2 = 0.0004$. If you want variance $v$, pass `std=v**0.5`:
   ```python
   var = 0.02
   nn.init.normal_(m.weight, mean=0.0, std=var ** 0.5)
   ```

2. **`trunc_normal_` at ±2σ shrinks the std.** If you ask for `std=1.0` with `a=-2, b=2`, the truncation removes both tails and the *delivered* std is ~0.88, not 1.0 (measured: 0.8796). For a precise std, either widen the bounds or rescale by `1/0.8796`:
   ```python
   nn.init.trunc_normal_(m.weight, mean=0.0, std=sigma, a=-2*sigma, b=2*sigma)
   # actual std ≈ 0.88 * sigma  (because of the ±2σ truncation)
   ```

---

### Q: My loop uses `fan_in = m.weight.shape[0]` — is that actually `fan_in`?

**A:** No — **`shape[0]` is `fan_out`, not `fan_in`.** This is the bug in your snippet. For `nn.Linear(in_features, out_features)` the weight tensor has shape `[out_features, in_features]` because the forward is `y = x Wᵀ + b`. Verified directly:

```
nn.Linear(8, 4).weight.shape = (4, 8)
   shape[0] = 4  = out_features = FAN_OUT
   shape[1] = 8  = in_features  = FAN_IN
   forward == x @ W.T + b : True
```

So you named the variable `fan_in` but assigned it `fan_out`. On your three layers this makes the per-layer σ wildly inconsistent:

| Layer | `shape[0]` you used (fan_out) | correct `fan_in` (`shape[1]`) | your `σ = 1/fan_out` | correct He std `√(2/fan_in)` |
|---|---|---|---|---|
| `Linear(input_dims, 128)` | 128 | `input_dims` (8 here) | 0.0078 | 0.500 |
| `Linear(128, 32)` | 32 | 128 | 0.0312 | 0.125 |
| `Linear(32, 1)` | 1 | 32 | 1.0000 | 0.250 |

The first layer ends up **64× too small**, the second 4× too small, and the output layer **4× too large** (σ=1.0). Use `m.weight.shape[1]` (or equivalently `m.in_features`) for `fan_in`:

```python
fan_in = m.weight.shape[1]   # FIX: was shape[0] (= fan_out)
```

---

### Q: I used `sigma = 1.0 / fan_in` — is the exponent right?

**A:** No — you're off by a factor of `√fan_in` on the std. Standard inits scale the **variance** as `1/fan_in` (LeCun) or `2/fan_in` (He), which means the **std** is `1/√fan_in` or `√(2/fan_in)`. Your `sigma = 1/fan_in` makes the variance `1/fan_in²` — too small by `√fan_in` (≈5.7× at fan_in=32, ≈11× at fan_in=128). You almost certainly meant:

```python
std = (2.0 / fan_in) ** 0.5    # He / ReLU
# or, for tanh/sigmoid:
std = (1.0 / fan_in) ** 0.5    # LeCun / Xavier-ish
```

---

### Q: What does the buggy init actually do to the network? (measured)

**A:** With `shape[0]` (fan_out) and `1/fan_in`, the per-layer scales are `0.0078 / 0.0312 / 1.0` instead of the correct `0.5 / 0.125 / 0.25`. The net effect — **measured by running this exact architecture at init** (4096 samples) — is a **vanishing signal, not saturation**:

| init | output `σ(p)` | fraction saturated (|logit|>4) |
|---|---|---|
| buggy (`shape[0]`, `1/fan_out`) | 0.003 | 0.0% |
| correct He | 0.153 | 0.0% |

The output collapses to `mean ≈ 0.50, std ≈ 0.003` — i.e. the model predicts ~0.5 for *everything* before training even starts. The under-scaled early layers (the first layer is 64× too small) annihilate the signal before it reaches the over-scaled output layer, so the output layer's large weights have ~nothing to amplify. It's a *vanishing-signal-at-the-input* failure, the same family as the `debugging-model-not-learning` zero-init bug — not a saturated sigmoid. (In `eval()` mode this is especially clean: BatchNorm's *untrained* running stats are `mean=0, var=1`, so BN behaves as ~identity and does not rescue the under-scaled first layer.)

---

### Q: What's the right way, then?

**A:** Since this is a **ReLU** network, use **Kaiming/He normal** — it sets the per-layer std *as a function of `fan_in`* so activation variance stays ~constant across depth, which is the whole point:

```python
for m in self.model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   # std = √(2/fan_in)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

That single call gets the fan_in-dependence right automatically — no manual `shape[1]` lookups, no exponent mistakes. Rule of thumb for which to reach for:

| Architecture | Init |
|---|---|
| **ReLU** nets | `kaiming_normal_` / `kaiming_uniform_` (std = √(2/fan_in)) |
| **tanh / sigmoid** | `xavier_uniform_` / `xavier_normal_` (or LeCun `1/√fan_in`) |
| **Transformers / GPT-style** | fixed `normal_(mean=0, std=0.02)` (the `mean`/`std` API from the first Q) |

---

### Q: Only my first `Linear` is followed by BatchNorm — does that change anything?

**A:** Yes, and it's the one subtlety worth getting right on this architecture. BatchNorm after a layer normalizes that layer's output to ~unit variance, so it **papers over** a bad weight scale for the *first* `Linear` — whatever you init it as, BN roughly resets the scale downstream. But the **second `Linear(128, 32)` and the output `Linear(32, 1)` have no BN after them**, so their init scale actually *propagates* to the output. That's why the under-scaled second layer (σ 0.0312 vs 0.125) in the buggy init contributes directly to the vanishing signal — there's no BN to fix it.

Concretely: the first layer's init matters *less* than usual (BN covers it), and the last layer's init matters *more* than usual (it sets the output logit scale directly). Which leads to the next question.

---

### Q: The output goes into a sigmoid/BCE — should I scale the last layer down?

**A:** Yes — this is the GPT-style trick and it's worth doing on the output `Linear(32, 1)`. After Kaiming init, initialize the last layer's weights to a smaller scale (e.g. multiply by 0.02) so the **initial logits stay near 0**, the initial sigmoid outputs stay near 0.5, and your **first BCE loss starts near `ln(2)` instead of spiking** from a near-saturated sigmoid:

```python
for m in self.model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
# GPT-style: shrink the OUTPUT projection so initial logits ≈ 0
self.model[-1].weight.data.mul_(0.02)
```

**Measured**, on this exact architecture at init:

| last-layer scale | output `σ(p)` | fraction saturated |
|---|---|---|
| He only | 0.154 | 0.5% |
| He + last × 0.02 | 0.005 | 0.0% |

With plain He the output is already mostly fine here (only 0.5% saturated), but scaling the last layer down by 0.02 tightens the initial output distribution from `σ ≈ 0.15` to `σ ≈ 0.005` — guaranteeing every initial logit is near 0 and the first loss is a clean `≈ ln 2 ≈ 0.69` rather than a spike. On a deeper net, or a head with more output units, the same trick is what stops the first forward pass from saturating the sigmoid and producing a huge, badly-behaved initial gradient.

**Important:** only shrink the **output** (last) layer — shrinking the hidden layers defeats the Kaiming variance-balance that keeps gradients healthy through depth. The hidden layers want He's `√(2/fan_in)`; only the final projection wants the extra `× 0.02`.

---

## Quick reference

```python
def init_fraud_mlp(model):
    """He init for the hidden layers + GPT-style downscaling on the output."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # std = √(2/fan_in)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # Keep initial logits near 0 so the first BCE loss doesn't spike.
    model[-1].weight.data.mul_(0.02)
```

The three bugs in the original snippet, ranked:

1. **`shape[0]` is `fan_out`, not `fan_in`** — use `shape[1]` / `m.in_features`. Makes the first layer 64× too small and the output 4× too large.
2. **`sigma = 1/fan_in` has the wrong exponent** — std should be `1/√fan_in` (LeCun) or `√(2/fan_in)` (He), not `1/fan_in`.
3. **Fixed mean/std instead of fan_in-adaptive** — a ReLU net wants Kaiming (std depends on fan_in), not a blanket `N(0, σ)`; and the output projection wants the extra `× 0.02` downscaling so the initial sigmoid isn't saturated and the first loss doesn't spike.

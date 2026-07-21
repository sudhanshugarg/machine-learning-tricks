# FAQ: Train vs Eval Semantics

### Q: Does `model.eval()` stop gradient computation?

**A:** No. `model.eval()` only sets `self.training = False`. Gradients are still computed unless you wrap the forward pass in `torch.no_grad()` or `torch.inference_mode()`.

```python
model.eval()
out = model(x)
loss = out.sum()
loss.backward()   # ← gradients ARE computed
```

Always pair eval mode with a no-grad context for inference:

```python
model.eval()
with torch.no_grad():
    out = model(x)
```

---

### Q: Why do my BatchNorm layers produce NaN during inference?

**A:** Most likely you are running in train mode with batch size 1. In train mode, BatchNorm computes variance across the batch; with a single sample, $\sigma^2 = 0$, and the division $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$ can become unstable depending on input scale.

Fix: Call `model.eval()` before inference.

---

### Q: Is LayerNorm affected by `train()` and `eval()`?

**A:** No. `nn.LayerNorm` computes mean and variance per-sample, so there are no running statistics to update or freeze. Its behavior is deterministic and identical in both modes.

---

### Q: Do I need `model.train()` if I already use `torch.no_grad()` during training?

**A:** Yes. `torch.no_grad()` disables gradient computation globally inside its context; it does **not** re-enable Dropout or BatchNorm updates. `model.train()` is still required for training behavior.

---

### Q: Can I set some layers to eval while keeping the rest in train mode?

**A:** Yes. PyTorch allows local overrides:

```python
model.train()          # global train
for m in model.backbone.modules():
    m.eval()           # freeze backbone
for p in model.backbone.parameters():
    p.requires_grad = False   # also disable gradients
```

Remember that local overrides persist; if you later call `model.train()`, the overridden modules stay in eval until you explicitly call `.train()` on them.

---

### Q: What about `nn.Identity()`, activations, or pooling layers?

**A:** Layers that do not reference `self.training` behave identically in both modes. This includes:
- All activations (`ReLU`, `GELU`, `Sigmoid`, ...)
- Pooling (`MaxPool`, `AvgPool`, `AdaptiveAvgPool`)
- `Identity`, `Flatten`, `Upsample`
- Normalizations **without** running stats (`LayerNorm`, `GroupNorm`)

---

### Q: How does BatchNorm behave with `track_running_stats=False`?

**A:** When `track_running_stats=False`, BatchNorm does not maintain running buffers. In eval mode, it falls back to computing statistics from the current input batch. This is dangerous for batch size 1 inference and generally discouraged for production models.

---

### Q: Why do my validation metrics look great but production metrics are terrible?

**A:** Checklist:
1. Did you call `model.eval()` before the production serving loop?
2. Are you loading the EMA weights (if used during training)?
3. Did you export to ONNX/TorchScript in eval mode?
4. Is your preprocessing pipeline identical to validation?

The most common culprit is #1 — serving code that forgets to switch modes.

---

### Q: Should I call `model.eval()` before saving a checkpoint?

**A:** It does not affect the saved parameters or buffers, but it is harmless. The key is to call `model.eval()` **after loading** and before inference. Some teams prefer saving in eval mode as a defensive convention:

```python
model.eval()
torch.save(model.state_dict(), "model.pt")
```

This does not bake eval behavior into the weights; it merely records the mode at save time. You still must call `model.eval()` after `load_state_dict()`.

---

### Q: Does `torch.compile()` or `torch.jit.script()` respect `model.eval()`?

**A:**
- **`torch.compile()`**: Yes. It traces the graph dynamically and respects `self.training`. Recompiling may happen if you switch modes.
- **`torch.jit.script()` / `trace()`**: Yes, but the graph is frozen at capture time. If you capture in train mode, dropout and BatchNorm train branches are baked in. Always call `model.eval()` **before** scripting or tracing.

---

### Q: Are there any layers that behave *worse* in eval mode than train mode?

**A:** No standard PyTorch layer behaves incorrectly in eval mode. The only scenario is when you genuinely need train-mode behavior at inference time (e.g., Monte-Carlo Dropout for uncertainty estimation). In that case, leaving dropout on is intentional, but you must manage it explicitly rather than forgetting `model.eval()`.

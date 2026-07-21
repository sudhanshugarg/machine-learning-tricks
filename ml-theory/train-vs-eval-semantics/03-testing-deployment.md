# Testing, Deployment, and Advanced Considerations

## Overview

Knowing the semantics is half the battle. This section covers:
1. Symptoms of forgetting `eval()` and how to catch them with tests.
2. Correct save/load patterns for models with cached statistics.
3.distributed training and mixed-precision gotchas.

---

## 1. Symptoms of Forgetting `model.eval()`

| Symptom | Likely Cause |
|---------|--------------|
| Predictions differ on identical inputs | Dropout / attention dropout still active |
| Accuracy drops gradually in production | BatchNorm running stats drift from repeated eval-in-train |
| `NaN` or near-zero outputs with batch size 1 | BatchNorm using degenerate batch statistics in train mode |
| Exported model (ONNX/TorchScript) behaves randomly | Dropout nodes baked in because export happened in train mode |
| Validation loss much higher than training loss | Model was not switched to eval during validation |

---

## 2. Catching Mode Bugs with Tests

### Test 1: Deterministic Inference

```python
def test_inference_is_deterministic(model, sample_input):
    model.eval()
    with torch.no_grad():
        out1 = model(sample_input)
        out2 = model(sample_input)
    assert torch.allclose(out1, out2, atol=1e-6), "Inference is non-deterministic!"
```

**Why it catches the bug**: If `model.eval()` is missing, Dropout will make `out1 != out2`.

### Test 2: BatchNorm Running Stats Freeze

```python
def test_bn_running_stats_freeze(model):
    bn = model.bn  # assume one BatchNorm layer is accessible
    prev_mean = bn.running_mean.clone()
    prev_var  = bn.running_var.clone()
    prev_num  = bn.num_batches_tracked.clone()

    model.eval()
    dummy = torch.randn(4, *bn.running_mean.shape)
    with torch.no_grad():
        model(dummy)

    assert torch.equal(bn.running_mean, prev_mean)
    assert torch.equal(bn.running_var, prev_var)
    assert torch.equal(bn.num_batches_tracked, prev_num)
```

### Test 3: Train Mode Actually Updates BN Stats

```python
def test_bn_stats_update_in_train(model, optimizer):
    bn = model.bn
    prev_mean = bn.running_mean.clone()

    model.train()
    dummy = torch.randn(4, *bn.running_mean.shape)
    optimizer.zero_grad()
    loss = model(dummy).sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(bn.running_mean, prev_mean), "BN stats did not update!"
```

### Test 4: Dropout Off in Eval

```python
def test_dropout_inactive_in_eval(model, sample_input):
    model.eval()
    with torch.no_grad():
        out_eval = model(sample_input)

    # manually set to train and run many times; variance should be huge
    model.train()
    with torch.no_grad():
        outs = torch.stack([model(sample_input) for _ in range(50)])
    variance = outs.var(dim=0).mean()

    assert variance > 1e-3, "Dropout seems inactive in train mode"
```

### Test 5: Hook-Based Sanity Check

For complex models, add a forward hook that asserts `self.training` is `False` during inference:

```python
def assert_eval_hook(module, input, output):
    assert not module.training, f"{module} is in train mode during inference!"

# Register on all Dropout and BatchNorm layers
for m in model.modules():
    if isinstance(m, (nn.Dropout, nn.BatchNorm2d)):
        m.register_forward_hook(assert_eval_hook)
```

> **Caution**: Remove hooks before training, or gate them with an environment variable.

---

## 3. Save / Load and Deployment Alignment

### What Gets Saved in `state_dict()`

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3),
    nn.BatchNorm2d(8),
    nn.Dropout(0.5),
)
print(model.state_dict().keys())
```

Output:
```
odict_keys([
  '0.weight', '0.bias',
  '1.weight', '1.bias',
  '1.running_mean', '1.running_var', '1.num_batches_tracked',
])
```

**No Dropout state** is stored (it has no parameters). **BatchNorm stores three extra buffers**: `running_mean`, `running_var`, and `num_batches_tracked`.

### Loading Checklist

```python
checkpoint = torch.load("model.pt")
model.load_state_dict(checkpoint)
model.eval()          # ← ALWAYS call this after loading!
torch.set_grad_enabled(False)
```

Forgetting `model.eval()` here is the #1 deployment bug.

### Models with Exponential Moving Average (EMA)

Many training pipelines maintain an EMA copy of weights for inference:

```python
# Shadow parameters (not part of model.parameters())
ema_shadow = {name: param.clone().detach()
              for name, param in model.named_parameters()}
```

**Save**: Store both the model state_dict and the EMA dictionary.

```python
torch.save({
    "model": model.state_dict(),
    "ema": ema_shadow,
    "optimizer": optimizer.state_dict(),
}, "checkpoint.pt")
```

**Load & Deploy**:

```python
checkpoint = torch.load("checkpoint.pt")

# Option A: Swap EMA weights into model, then save as canonical state_dict
for name, param in model.named_parameters():
    param.data.copy_(checkpoint["ema"][name])
model.eval()

# Option B: Keep a separate EMA model instance
ema_model = create_model()
ema_model.load_state_dict(checkpoint["ema"])
ema_model.eval()
```

**Pitfall**: If you save the model in train mode and load it into eval mode without realizing the EMA weights were never swapped, you serve the non-EMA (noisier) weights.

### Calibration Statistics (Temperature Scaling)

Post-hoc calibration often learns a scalar temperature $T$:

```python
self.temperature = nn.Parameter(torch.ones(1) * 1.5)
```

Because this is a `Parameter`, it is automatically included in `state_dict()`. Ensure you call `model.eval()` after loading so BatchNorm does not drift while the temperature remains fixed.

### TorchScript / ONNX Export

```python
model.eval()          # ← MANDATORY
torch.set_grad_enabled(False)

# Trace the eval graph only
traced = torch.jit.trace(model, example_input)
torch.jit.save(traced, "model.pt")

# ONNX export
torch.onnx.export(model, example_input, "model.onnx",
                  training=torch.onnx.TrainingMode.EVAL)
```

If you export in train mode:
- Dropout becomes a random-weight node in the graph.
- BatchNorm keeps a `training` branch, causing some runtimes to misbehave.

---

## 4. Distributed Training & Mixed Precision

### DistributedDataParallel (DDP)

```python
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DistributedDataParallel(model)
```

- `model.train()` / `model.eval()` must be called on the **wrapper**:
  ```python
  model.train()   # DDP wrapper propagates to inner module
  ```
- `SyncBatchNorm` computes **global** mean and variance across all GPUs during training. In eval mode, each GPU uses the same frozen running stats (no communication needed).

**Pitfall**: Calling `model.module.eval()` instead of `model.eval()` works but bypasses DDP's own state tracking. Prefer the wrapper methods.

### Mixed Precision (`torch.cuda.amp`)

```python
with torch.autocast(device_type="cuda"):
    loss = model(input)
```

- `autocast` is independent of `model.training`.
- Best-practice pairing:
  ```python
  model.eval()
  with torch.no_grad(), torch.autocast(device_type="cuda"):
      output = model(input)
  ```
- BatchNorm running stats are updated in fp32 internally by PyTorch even when weights are fp16, so no extra care is needed.

### `torch.no_grad()` vs `torch.inference_mode()`

| Context manager | Disables gradients | Disables view tracking | Recommended for |
|-----------------|--------------------|------------------------|-----------------|
| `torch.no_grad()` | Yes | No | General inference |
| `torch.inference_mode()` | Yes | Yes | Maximum speed, pure inference |

Neither affects `self.training`. You **must** still call `model.eval()`.

---

## 5. Quick Reference Cheatsheet

```python
# Training loop
model.train()
optimizer.zero_grad()
with torch.autocast(device_type="cuda"):
    loss = criterion(model(input), target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Validation / inference
model.eval()
with torch.inference_mode():
    logits = model(input)

# Loading a checkpoint for serving
checkpoint = torch.load("best.pt")
model.load_state_dict(checkpoint["model"])
model.eval()
model = torch.jit.script(model)   # or ONNX export
```

---

## Summary

1. **Always call `model.eval()`** before inference, validation, or export.
2. **Pair with `torch.no_grad()` or `torch.inference_mode()`** for speed and safety.
3. **Test determinism**: Two forward passes on the same input should be identical in eval mode.
4. **Save EMA / calibration stats explicitly** alongside `state_dict()`.
5. **Export in eval mode** to avoid baking stochastic nodes into deployment graphs.

Next: [FAQ](FAQ.md)

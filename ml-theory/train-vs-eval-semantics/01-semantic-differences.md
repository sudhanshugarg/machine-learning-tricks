# Semantic Differences: `model.train()` vs `model.eval()`

## Overview

PyTorch models inherit from `nn.Module`, which provides two methods:

- `model.train()` — sets `self.training = True`
- `model.eval()` — sets `self.training = False`

These calls recurse through **all child modules**, flipping the flag globally for the entire model graph.

---

## What the Flag Actually Controls

At its core, `self.training` is a boolean state variable. Individual layers query it inside their `forward()` methods to decide behavior:

```python
# Pseudocode inside nn.Dropout.forward()
if self.training:
    # randomly zero out neurons
else:
    # pass input through unchanged
```

**Critical insight**: This is *not* about gradients. `self.training` has **no effect** on autograd or gradient computation. It only controls **layer-specific behavior** such as:

- Stochastic regularization (Dropout, DropPath)
- Running-statistic updates (BatchNorm, InstanceNorm)
- Special train-only modules (e.g., some custom augmentation layers)

---

## Recursive Application

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn  = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 1)
        self.drop = nn.Dropout(0.3)

model = Net()
model.train()   # sets training=True  on Net, fc1, bn, fc2, drop
model.eval()    # sets training=False on Net, fc1, bn, fc2, drop
```

You can verify the state per layer:

```python
for name, module in model.named_modules():
    print(name, module.training)
```

---

## Local Overrides

Although `model.train()` / `model.eval()` are global to the model graph, you can override individual submodules:

```python
model.train()          # global train mode
model.bn.eval()        # but freeze BatchNorm stats for this layer
```

This is useful for:
- **Fine-tuning**: Keep pretrained BN layers in eval while training new heads.
- **Special layers**: Certain auxiliary heads that should behave differently.

**Caution**: Local overrides are fragile. They persist across `model.train()` calls unless you explicitly reset them.

---

## No-Op Modules

Many layers **do not reference `self.training`** at all. Their behavior is identical in both modes:

| Layer / Operation | Affected by `train()`/`eval()`? |
|-------------------|--------------------------------|
| `nn.Linear`       | No |
| `nn.Conv2d`       | No |
| `nn.ReLU`         | No |
| `nn.LayerNorm`    | **No** (computes per-sample stats) |
| `nn.MaxPool2d`    | No |
| `nn.Embedding`    | No |
| `nn.Softmax`      | No |

---

## Summary Table

| Aspect | `model.train()` | `model.eval()` |
|--------|-----------------|----------------|
| `self.training` | `True` | `False` |
| Dropout | Active (stochastic) | Disabled (identity) |
| BatchNorm | Uses batch stats; updates running stats | Uses running stats; frozen |
| LayerNorm | Same as train | Same as train |
| RNN dropout | Active between layers | Disabled |
| Transformer attention dropout | Active | Disabled |
| Gradient computation | Unaffected | Unaffected |

---

## Key Takeaway

`model.train()` / `model.eval()` toggle a boolean that layers consult internally. It controls **stochasticity and statistic updates**, not gradients. Always pair `model.eval()` with `torch.no_grad()` during inference for maximum efficiency and correctness.

Next: [02 — Layer Behaviors](02-layer-behaviors.md)

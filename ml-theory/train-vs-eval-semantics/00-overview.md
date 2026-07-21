# Train vs Eval Semantics in PyTorch

## What is This Topic?

In PyTorch, calling `model.train()` and `model.eval()` toggles a single boolean flag—`self.training`—that fundamentally changes the behavior of many built-in layers. Misusing these modes is one of the **most common sources of silent bugs** in deep learning pipelines.

---

## Why Does It Matter?

- **Silent degradation**: Forgetting `model.eval()` during inference does not crash your code; it merely produces incorrect or stochastic results.
- **Corrupted statistics**: BatchNorm updates its running averages during inference if left in train mode, slowly poisoning your model.
- **Production bugs**: A model that scores 95% accuracy in a notebook can drop to random-guess performance if served without `eval()`.

---

## Big Picture

```
Training Phase                              Inference Phase
─────────────────                           ─────────────────
model.train()                               model.eval()
      ↓                                            ↓
Dropout ON (random zeros)                   Dropout OFF (identity)
BatchNorm uses BATCH stats                  BatchNorm uses RUNNING stats
      ↓                                            ↓
Stochastic, regularized                     deterministic, stable
```

---

## Concrete Example: A Simple Classifier

Imagine a small CNN for MNIST:

```python
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3)
        self.bn   = nn.BatchNorm2d(8)
        self.drop = nn.Dropout(0.5)
        self.fc   = nn.Linear(8*26*26, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)      # behavior changes with mode
        x = F.relu(x)
        x = self.drop(x)    # behavior changes with mode
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

Running the **same image** through the network twice:

| Mode | Output 1 | Output 2 | Identical? |
|------|----------|----------|------------|
| `train()` | Random (dropout active) | Random (dropout active) | **No** |
| `eval()` | Fixed | Fixed | **Yes** |

---

## What You Will Learn

1. **[Semantic Differences](01-semantic-differences.md)** — Exactly what `train()` and `eval()` do at the framework level.
2. **[Layer Behaviors](02-layer-behaviors.md)** — Deep dive into Dropout, BatchNorm, LayerNorm, RNNs, and Transformers.
3. **[Testing & Deployment](03-testing-deployment.md)** — How to detect mode bugs early, plus save/load best practices for models with cached statistics.

---

## Prerequisites

- Basic PyTorch (`nn.Module`, `forward()`)
- Familiarity with Dropout and BatchNorm concepts
- Understanding of `torch.no_grad()` (helpful but not required)

---

## Next Steps

→ [01 — Semantic Differences](01-semantic-differences.md)

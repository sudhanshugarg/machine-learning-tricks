# Solution: Debugging a Model That Won't Learn

This solution is organized as **hints first**, then the **systematic diagnosis**, then the **fixes and corrected code**.

---

## Part 1: Progressive Hints

<details>
<summary><b>Hint 1 — The most obvious bug</b> (click to expand)</summary>

Look at the loss function. The model outputs raw logits of shape `[batch, 10]`, but the labels are integer class indices. What loss function expects integer class labels for multi-class classification?

</details>

<details>
<summary><b>Hint 2 — Initialization</b> (click to expand)</summary>

All weights and biases are initialized to **constant zero**. For a ReLU network, what happens to gradients when two layers with zero-initialized weights multiply? Think about the symmetry-breaking problem.

</details>

<details>
<summary><b>Hint 3 — The loop</b> (click to expand)</summary>

After `loss.backward()`, where do the gradients go? Are they ever reset between batches?

</details>

<details>
<summary><b>Hint 4 — Scale</b> (click to expand)</summary>

The learning rate is `10.0`. For a network with unnormalized inputs in `[0, 1]`, what does theory say about the relationship between input scale, weight scale, and the maximum stable learning rate?

</details>

<details>
<summary><b>Hint 5 — Hidden subtlety</b> (click to expand)</summary>

The model is moved to CUDA *inside* the training loop, on every batch. But what about the very first batch? Trace the actual execution order carefully.

</details>

---

## Part 2: Systematic Debugging Methodology

When a model won't learn, follow this diagnostic order:

```
1. DATA     → Are inputs normalized? Are labels correct? Any NaN/Inf?
2. MODEL    → Is architecture sound? Forward pass produces expected shapes?
3. LOSS     → Is the loss function appropriate for the task?
4. GRADIENTS→ Are gradients flowing? Are their norms reasonable (~1e-3 to 1e1)?
5. OPTIMIZER→ Is LR too high/low? Is momentum behaving?
6. TRAIN LOOP→ Is zero_grad() called? Is model in train mode?
```

**Empirical tools to use at each step**:

```python
# 1. Inspect a single batch
images, labels = next(iter(trainloader))
print(images.shape, images.min(), images.max(), images.mean())
print(labels.shape, labels.min(), labels.max())

# 2. Forward pass sanity check
model.eval()
with torch.no_grad():
    out = model(images)
print(out.shape, out.min(), out.max())  # Should be [B, 10], not all zeros

# 3. Check for NaN/Inf after forward
assert not torch.isnan(out).any()
assert not torch.isinf(out).any()

# 4. Gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.6f}")

# 5. Weight statistics
for name, param in model.named_parameters():
    print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
```

---

## Part 3: Bug-by-Bug Diagnosis

### Bug 1: Zero Initialization (Severity: Critical)

**Symptom**: Output logits are all zeros; loss is constant; accuracy is exactly 10%.

**Theory**:
Every weight is initialized to $w_{ij} = 0$. In a ReLU network:
- Forward pass: all pre-activations are 0, all activations are 0.
- Backward pass: gradients are 0 because $\frac{\partial \text{ReLU}}{\partial z} = 0$ at $z = 0$ (or undefined, treated as 0).

Mathematically, for layer $l$:

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]} = 0, \quad a^{[l]} = \text{ReLU}(z^{[l]}) = 0
$$

Even if gradients for $W^{[l]}$ were non-zero, **all neurons in a layer receive identical gradients** due to symmetry. They update identically forever, so the network behaves like a single neuron per layer — no representation learning occurs.

**Fix**: Use standard initialization (Kaiming/He for ReLU):

```python
nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

---

### Bug 2: Wrong Loss Function (Severity: Critical)

**Symptom**: `MSELoss` between logits `[B, 10]` and integer labels `[B]`. PyTorch broadcasts labels to `[B, 10]`, creating a bizarre regression target. The model tries to minimize squared error against a one-hot-ish broadcasted integer — not a valid classification objective.

**Theory**:
Multi-class classification requires a loss that treats class labels as **categorical**, not continuous values. Cross-entropy is derived from maximum likelihood:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{B} \sum_{i=1}^{B} \log p(y_i | x_i)
$$

where $p(y_i | x_i) = \text{softmax}(z_i)_{y_i}$.

MSELoss on logits vs. class indices has no probabilistic interpretation. The gradients point in the wrong direction.

**Fix**: Use `nn.CrossEntropyLoss()`, which combines `LogSoftmax + NLLLoss`.

---

### Bug 3: Missing `optimizer.zero_grad()` (Severity: Critical)

**Symptom**: After ~50 batches, loss becomes `NaN`. Weight norms explode.

**Theory**:
Gradients accumulate across batches:

$$
g^{(t)}_{\text{total}} = \sum_{i=1}^{t} g^{(i)}_{\text{batch}}
$$

Since each batch gradient is an independent estimate of $\nabla \mathcal{L}$, their sum grows as $O(\sqrt{t})$ in random walk fashion, but with correlated directions it can grow linearly. The parameter update:

$$
\theta_{t+1} = \theta_t - \eta \sum_{i=1}^{t} g_i
$$

becomes an enormous step. With $\eta = 10.0$, this causes immediate divergence.

**Fix**: Call `optimizer.zero_grad()` before each `backward()`.

---

### Bug 4: Learning Rate Too High (Severity: High)

**Symptom**: Even after fixing bugs 1-3, loss oscillates wildly or diverges.

**Theory**:
For convex quadratic $f(x) = \frac{1}{2} x^T A x$, gradient descent converges only if:

$$
\eta < \frac{2}{\lambda_{\text{max}}(A)}
$$

where $\lambda_{\text{max}}$ is the largest eigenvalue of the Hessian. Deep networks are non-convex, but the intuition holds: a large LR exceeds the local curvature budget and overshoots the minimum.

For CIFAR-10 with standard CNNs, stable LR is typically $0.01$–$0.1$ with SGD + momentum. $\eta = 10.0$ is 100× too large.

**Fix**: Use `lr=0.01` (or `0.1` with LR scheduling).

---

### Bug 5: Missing Data Normalization (Severity: Moderate)

**Symptom**: Training is slow and unstable. Validation accuracy lags even when train accuracy improves.

**Theory**:
Input pixels in `[0, 1]` have mean $\approx 0.5$ per channel. For the first convolution:

$$
z = \sum_{i} w_i x_i + b
$$

If $x_i$ has large positive mean, the pre-activations are biased positive, pushing many neurons into the ReLU's linear regime with large activations. This:
1. Increases gradient variance (线性地依赖于 input magnitude).
2. Shifts internal covariate distribution across layers.

Normalizing to zero mean and unit variance (per-channel) centers the input distribution, reducing the condition number of the optimization landscape:

$$
\kappa(H) = \frac{\lambda_{\text{max}}(H)}{\lambda_{\text{min}}(H)}
$$

Better conditioning $\Rightarrow$ larger stable LR $\Rightarrow$ faster convergence.

**Fix**: Add `transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))`.

---

### Bug 6: Target Type/Shape Mismatch with MSELoss (Severity: Moderate)

**Symptom**: MSELoss broadcasts the integer label tensor into the wrong shape, producing 100 regression targets per sample instead of 1 classification target.

**Theory**:
`nn.MSELoss()(outputs, labels)` with `outputs:[B,10]` and `labels:[B]` broadcasts labels to `[B, 10]`. The model is trained to predict the integer class index as a scalar in *all 10 output dimensions*, which is nonsensical.

**Fix**: Use `CrossEntropyLoss` (no broadcasting issues — it expects logits + integer labels).

---

### Bug 7: Model Moved Inside Loop (Severity: Low)

**Symptom**: Slight overhead; first batch may run on CPU if not careful.

**Fix**: Move model to device **once**, before training.

---

## Part 4: Corrected Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class FixedCNN(nn.Module):
    """A CNN that actually learns on CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)           # Added for stability
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)           # Added for stability
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

        # FIX 1: Proper He initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Added BN
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Added BN
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIX 5: Normalize inputs (per-channel mean/std for CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    model = FixedCNN().to(device)  # FIX 7: Move model once

    # FIX 2: Correct loss for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # FIX 4: Reasonable learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # FIX 3: Zero gradients before backward
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)  # FIX 6: CrossEntropy handles shapes correctly

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Train Acc: {train_acc:.2f}%")
        scheduler.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    train()
```

---

## Part 5: Answers to Discussion Questions

### Q1: Vanishing vs. Exploding Gradients in Deep Networks

In a 50-layer network, the gradient of layer $l$ involves a product of Jacobian matrices:

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \prod_{k=l+1}^{L} \frac{\partial a^{[k]}}{\partial z^{[k]}} \frac{\partial z^{[k]}}{\partial a^{[k-1]}}
$$

For ReLU, $\frac{\partial a}{\partial z} \in \{0, 1\}$. If many neurons are dead (0), gradients vanish. If all are active and weight norms $> 1$, gradients explode as $\|W\|^L$.

**BatchNorm prevents both** by normalizing pre-activations to ~N(0,1), keeping them in the stable gradient regime of activations. It also makes the landscape smoother, allowing higher LR without divergence.

### Q2: Dead ReLUs

A ReLU neuron is **dead** if it always outputs 0 for all training inputs, meaning $z < 0$ always. Its gradient is permanently 0, so it never updates.

**Causes**:
- Bad initialization (large negative bias)
- Too-high learning rate pushing weights into negative region
- Excessive L2 regularization

**Mitigations**:
- **LeakyReLU / PReLU**: $\max(\alpha z, z)$ with small $\alpha > 0$ gives non-zero gradient for $z < 0$.
- **He initialization**: Keeps pre-activations near 0 where ReLU is active.
- **BatchNorm**: Normalizes pre-activations, preventing mass negative drift.

### Q3: Blind LR Tuning Heuristics

1. **LR Range Test** (Smith, 2017): Start with very small LR, increase linearly each batch. Plot loss vs. LR. Choose LR just before loss starts to diverge (the "elbow"), then divide by 3–10.

2. **Largest Stable LR**: Binary search. Double LR until loss goes NaN, then back off by half.

3. **Theoretical heuristic**: $\eta \approx \frac{1}{\|H\|}$ where $H$ is the Hessian. In practice, $\eta \approx 0.01$ for Adam, $0.1$ for SGD+momentum on image tasks.

### Q4: Why Data Normalization Helps

Let $\Sigma = \text{Cov}(X)$ be the data covariance. Gradient descent convergence rate depends on $\kappa(\Sigma) = \lambda_{\text{max}} / \lambda_{\text{min}}$.

For raw pixel data, different channels have different means/variances. Normalizing to zero mean, unit variance **sphericizes** the data, making $\Sigma \approx I$ and $\kappa \approx 1$.

This allows:
- Larger stable learning rates ($\eta \propto 1/\lambda_{\text{max}}$)
- Faster convergence (fewer iterations to reach $\epsilon$-suboptimality)
- More stable training (gradient variance reduced)

### Q5: When Loss Goes NaN

**Most likely cause**: Exploding gradients or logits.

**Confirmation**:
```python
for name, p in model.named_parameters():
    if p.grad is not None:
        assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
```

**Three architectural fixes**:
1. **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)`
2. **Add BatchNorm / LayerNorm**: Stabilizes activations and prevents runaway magnitudes.
3. **Use mixed precision carefully**: FP16 can overflow. Scale loss via `GradScaler`, or keep logits computation in FP32.

---

## Part 6: Prevention Checklist

Before starting any training run, verify:

- [ ] **Data**: Visualize a batch. Are labels correct? Are inputs normalized?
- [ ] **Model**: Print output shape. Does it match expectation?
- [ ] **Initialization**: Are weights non-zero and properly scaled?
- [ ] **Loss**: Does the loss function match the task (CE for classification, MSE for regression)?
- [ ] **Gradients**: After first backward, print `param.grad.norm()`. Is it finite and non-zero?
- [ ] **Optimizer**: Is LR reasonable? Try 0.01 as a safe default.
- [ ] **Loop**: Is `zero_grad()` called every iteration?
- [ ] **Device**: Is the model on the same device as the data?

---

## Summary Table

| Bug | Symptom | Root Cause (Theory) | Fix |
|-----|---------|---------------------|-----|
| Zero init | Constant loss, 10% acc | Symmetric gradients, no representation learning | Kaiming init |
| Wrong loss | Nonsensical targets | MSE broadcast creates invalid regression targets | CrossEntropyLoss |
| No zero_grad | NaN, exploding weights | Gradient accumulation causes $O(t)$ step size | Call zero_grad() |
| LR=10.0 | Oscillation, divergence | Exceeds local curvature budget $2/\lambda_{\text{max}}$ | LR=0.01 |
| No normalization | Slow, unstable training | Poor conditioning $\kappa(\Sigma) \gg 1$ | Normalize per-channel |
| Target mismatch | Wrong broadcast shape | MSELoss treats class index as regression target | Use CE loss |
| Model in loop | CPU/CUDA overhead | Unnecessary device transfer | Move once |

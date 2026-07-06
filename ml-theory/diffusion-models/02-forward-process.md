# Forward Process: Adding Noise to Images

## Overview

The forward process is the easy part—it's fully deterministic and requires no neural networks. We simply add Gaussian noise to images according to a pre-defined schedule.

---

## The Mathematical Formula

Given a real image $x_0$ (MNIST digit), we progressively add noise over $T$ timesteps:

$$q(x_t | x_0) = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$$

Where:
- $x_t$ = noisy image at timestep $t$
- $x_0$ = original clean image
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ = **Gaussian noise, independently sampled each time** (mean 0, variance 1)
- $\bar{\alpha}_t$ = cumulative product of noise schedule coefficients
- $1 - \bar{\alpha}_t$ = variance of noise at timestep $t$

**Intuition**: 
- As $t$ increases, $\bar{\alpha}_t$ decreases
- This means: less signal ($x_0$), more noise ($\epsilon$)
- At $t=0$: image is mostly original (very little random noise added)
- At $t=T$: image is nearly pure noise (signal almost completely drowned out)

**Critical: Epsilon is sampled fresh each time**
- Every call to the forward process samples a new $\epsilon$
- Same $x_0$ at same $t$ will produce **different** $x_t$ if epsilon differs
- This is intentional! It provides data augmentation and prevents overfitting

---

## The Noise Schedule

The **noise schedule** controls how quickly we add noise. Common choice: **cosine schedule**

$$\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

Where $s$ is a small offset (e.g., 0.008) to prevent $\bar{\alpha}_T$ from becoming exactly 0.

**Example values for MNIST (T=1000):**
- $t=0$: $\bar{\alpha}_0 \approx 1.0$ (almost no noise)
- $t=250$: $\bar{\alpha}_{250} \approx 0.75$ (some noise)
- $t=500$: $\bar{\alpha}_{500} \approx 0.5$ (equal signal and noise)
- $t=750$: $\bar{\alpha}_{750} \approx 0.25$ (mostly noise)
- $t=1000$: $\bar{\alpha}_{1000} \approx 0.0$ (nearly pure noise)

---

## Important: Epsilon is Randomly Sampled

Before diving into examples, understand this crucial point:

$$\epsilon \text{ is NOT a fixed constant—it's randomly sampled}$$

Each call to `forward_process(x_0, t)` **samples a fresh epsilon**:

```
Call 1: forward_process(x_0, t=5)
  ε_1 ~ N(0, I)  [independent sample]
  x_5 = √(ᾱ_5) * x_0 + √(1-ᾱ_5) * ε_1

Call 2: forward_process(x_0, t=5)  [same x_0, same t!]
  ε_2 ~ N(0, I)  [DIFFERENT independent sample]
  x_5' = √(ᾱ_5) * x_0 + √(1-ᾱ_5) * ε_2

Result: x_5 ≠ x_5' (almost certainly)
```

**Why?** This creates **natural data augmentation**:
- One real image → infinite corrupted versions
- Network never sees the exact same noisy image twice
- Prevents overfitting and memorization

---

## MNIST Example

### Step 1: Start with a Clean Image

```
Input x_0: 28×28 MNIST image of digit "3"
Pixel values: [0, 255] or normalized [0, 1]
```

### Step 2: Add Noise at Different Timesteps

**At t=0:**
```
x_0 = √(0.99) * [original image] + √(0.01) * [tiny noise]
≈ original image (barely noisy)
```

**At t=500 (halfway):**
```
x_500 = √(0.5) * [original image] + √(0.5) * [noise]
= mix of signal and noise
(image is recognizable but corrupted)
```

**At t=1000 (end):**
```
x_1000 = √(0.001) * [original image] + √(0.999) * [noise]
≈ pure Gaussian noise
(no longer looks like a digit)
```

---

## Implementation Details

### Input Dimensions (MNIST)
- Original image: **1 × 28 × 28** (1 channel, 28×28 pixels)
- When flattened: **784** values

### What Gets Added
- Signal part: $\sqrt{\bar{\alpha}_t} \times$ (28×28 image)
- Noise part: $\sqrt{1 - \bar{\alpha}_t} \times$ (28×28 Gaussian noise)
- **Output shape**: **28×28** (same as input)

### Key Property: One-Step Jump

The beauty of this formulation: **you can jump directly to any timestep without computing all intermediate steps**

Instead of: $x_0 \to x_1 \to x_2 \to \cdots \to x_t$

You can directly compute: $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$

This is crucial for training—we can sample random timesteps and compute noisy images on-the-fly.

---

## Code Sketch

```python
import torch
import math

def cosine_schedule(t, T, s=0.008):
    """Compute cumulative alpha at timestep t"""
    # t ∈ [0, T], normalized to [0, 1]
    normalized_t = t / T
    alpha_bar = torch.cos(
        (torch.tensor(normalized_t) + s) / (1 + s) * math.pi / 2
    ) ** 2
    return alpha_bar

def forward_process(x_0, t, T):
    """Add noise to x_0 at timestep t"""
    alpha_bar = cosine_schedule(t, T)
    
    # Sample noise
    epsilon = torch.randn_like(x_0)
    
    # Compute noisy image
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon
    
    return x_t, epsilon

# Example usage
x_0 = torch.randn(1, 1, 28, 28)  # MNIST digit (batch=1, channels=1, H=28, W=28)
x_500, epsilon = forward_process(x_0, t=500, T=1000)

print(f"Input shape: {x_0.shape}")   # torch.Size([1, 1, 28, 28])
print(f"Noisy image shape: {x_500.shape}")  # torch.Size([1, 1, 28, 28])
```

---

## Why This Matters for Training

The forward process defines our **training objective**:
- We start with a real image $x_0$ and a random timestep $t$
- We compute $x_t$ using the forward process
- We ask the neural network: *"Predict the noise $\epsilon$ that was added"*
- We compare its prediction to the ground truth $\epsilon$

The network learns to estimate noise, which is the key to reversing the process.

---

## Summary

| Property | Value |
|----------|-------|
| **Deterministic?** | Yes (no learning required) |
| **Input to forward process** | Clean image + timestep |
| **Output of forward process** | Noisy image + ground truth noise |
| **Shape change (MNIST)** | 28×28 → 28×28 (unchanged) |
| **Can jump directly to timestep t?** | Yes! |

Next: [Backward Process](03-backward-process.md) — How to reverse this and generate images.

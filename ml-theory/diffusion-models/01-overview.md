# Diffusion Models: A Simple Tutorial

## What is a Diffusion Model?

A **diffusion model** is a generative model that learns to generate data by gradually denoising noise into realistic samples. The core idea is elegantly simple:

1. **Forward Process**: Slowly add Gaussian noise to real images until they become pure noise
2. **Backward Process**: Train a neural network to reverse this process—gradually remove noise to reconstruct images

Think of it like learning to reverse a degradation process. If you can learn what "noise removal" looks like at each step, you can chain these steps together to generate entirely new images from scratch.

---

## MNIST Example Overview

We'll use MNIST (handwritten digits 0-9) as our concrete example throughout:

- **Input shape**: 28×28 images (784 pixels when flattened)
- **Forward process**: Add noise to real digits over T timesteps (typically T=1000)
- **Backward process**: Train a U-Net to predict and remove noise
- **Generation**: Start with random noise, apply denoising 1000 times to get a digit

---

## High-Level Flow

```
Real MNIST Digit
       ↓
[Forward Process: Add noise progressively]
       ↓
Pure Noise (after T=1000 steps)

Now reverse it:

Pure Noise (random initialization)
       ↓
[Backward Process: Neural network removes noise at each step]
       ↓
Generated MNIST Digit
```

---

## The Neural Network at a Glance

### Architecture
- **Type**: U-Net (encoder-decoder with skip connections)
- **Input**: Noisy image + timestep information
- **Output**: Predicted noise to subtract from image
- **Why U-Net?** It preserves spatial information through skip connections, crucial for image generation

### Dimensions (MNIST Example)

**Input to Network:**
- Noisy image: 1 × 28 × 28 (grayscale)
- Timestep embedding: scalar t (encoded to ~128 dims)
- **Total input**: effectively 28 × 28 spatial dims + timestep conditioning

**Output from Network:**
- Predicted noise: 1 × 28 × 28 (same shape as input)

The network learns: *"Given this noisy image at timestep t, what noise was added?"*

---

## Key Insight: Why Does This Work?

The magic happens because we can mathematically define:
- How much noise to add at each step: **noise schedule**
- How to predict the noise given a noisy image: **neural network training**
- How to use predictions to denoise: **reverse diffusion formula**

Each piece is learnable or pre-computable, and together they create a powerful generative model.

---

## Next Steps

- **[Forward Process](02-forward-process.md)**: How noise is added mathematically
- **[Backward Process](03-backward-process.md)**: How the network learns to denoise and generate


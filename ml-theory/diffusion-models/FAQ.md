# Diffusion Models FAQ

This FAQ covers common questions about the diffusion models tutorial. If you have a question not answered here, feel free to ask—we'll add it along with the answer!

---

## Conceptual Questions

### Q: Why do we add noise in the forward process if we just want to remove it in the backward process?

**A:** Great question! The key insight is **learning by corruption**. Here's why:

1. **Training Signal**: By adding known amounts of noise, we create a clear learning signal. The network learns "given THIS level of corruption, predict THIS noise."

2. **Distribution Matching**: A denoiser trained on all corruption levels (t=1 to T) can reverse any noise level, even pure random noise (t=T).

3. **Generation from Scratch**: Once trained, we can start with pure random noise (t=T) and iteratively denoise, since the network has learned to handle the full spectrum of corruption.

It's like learning to undo corruptions of increasing severity, then chaining those skills together to build something from nothing.

---

### Q: What's the difference between α_t and ᾱ_t?

**A:** 
- **α_t**: The noise coefficient at a *single* step t
  - Determines how much noise is added in one step
  - Used in the recurrence relation: $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon$

- **ᾱ_t**: The *cumulative* product up to step t
  - $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$
  - Lets you jump directly from $x_0$ to $x_t$ without computing intermediate steps
  - Used in forward process: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

**Why both?** The cumulative version is computationally efficient for training (sample any timestep instantly), while the individual version is needed for theoretical analysis.

---

### Q: Why is the timestep embedding necessary? Can't the network just take t as a scalar input?

**A:** Technically yes, but it would be very inefficient. Here's why:

1. **Scale**: Timestep values range from 0 to T (e.g., 0-1000). Raw scalars don't provide good gradients for the network.

2. **Sinusoidal Encoding**: We use positional encoding (similar to Transformers):
   ```
   emb_even = sin(t / 10000^(i/d))
   emb_odd = cos(t / 10000^((i-1)/d))
   ```
   This creates a rich, learnable representation of "which step am I?"

3. **Learnable Adaptation**: The embedding is then passed through linear layers to get task-specific representations.

Without embedding, the network would struggle to distinguish between nearby timesteps and learn smooth denoising across all corruption levels.

---

### Q: Why U-Net instead of a simple CNN or ResNet?

**A:** 
- **Simple CNN**: Would lose spatial information through downsampling; bad for preserving fine details
- **ResNet**: Designed for classification (outputs fixed-size vectors); doesn't preserve 2D structure
- **U-Net**: 
  - Preserves spatial dimensions (in = out)
  - Skip connections preserve fine details from encoder
  - Multi-scale processing captures both local and global features
  - Encoder-decoder structure is natural for image-to-image tasks

For diffusion, we need to predict noise (a full image), so U-Net's spatial preservation is crucial.

---

### Q: What does the network actually learn? Is it predicting "what the original image was" or "what noise was added"?

**A:** It predicts **the noise**, not the original image. But these are mathematically equivalent in the loss function:

Given $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, we can rearrange to predict:
1. **Noise**: $\epsilon_\theta(x_t, t)$ ← what we use
2. **Original image**: $x_{0,\theta}(x_t, t) = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$

Both are common implementations. Predicting noise is simpler and works better in practice.

---

## Implementation Questions

### Q: In the MNIST example, why is the input 28×28×1 and output 28×28×1? Shouldn't we predict something different?

**A:** No! For image-to-image tasks like denoising:
- **Input**: The noisy image (28×28×1)
- **Output**: The predicted noise to subtract (28×28×1)

Both have the same shape because:
1. Noise is generated from the same distribution as the image (Gaussian)
2. We subtract the prediction from the input: $x_{t-1} = \text{denoise}(x_t) = x_t - \epsilon_{\text{pred}}$

The output is *not* the denoised image directly—it's a residual (the noise) that we subtract.

---

### Q: How do skip connections actually work with concatenation?

**A:** In the decoder:

```
Encoder output at level k: shape (B, C_enc, H, W)
                           ↓ [saved for later]

Decoder upsamples from level k+1: (B, C_dec, H, W)
                                   ↓ [concatenate along channel dimension]

result = cat([decoder_output, encoder_output], dim=1)
         → (B, C_dec + C_enc, H, W)
         ↓ [pass through conv to reduce channels back down]
         → (B, C_out, H, W)
```

**Why concatenate instead of add?**
- **Add** (element-wise): Would require same # channels
- **Concatenate**: Doubles channels, then conv reduces it
- Concatenate preserves information from both paths; add would lose data if channels mismatch

---

### Q: Why T=1000 timesteps? Why not 100 or 10,000?

**A:** Trade-off between quality and speed:

- **T too small** (e.g., 100): 
  - Fast sampling
  - Poor quality (too few denoising steps)

- **T=1000** (standard):
  - ~1000 denoising steps at generation time
  - Good quality-speed balance
  - Noise schedule can be spread smoothly across range

- **T very large** (e.g., 10,000):
  - Better quality
  - Much slower (10× more denoising steps)
  - More computational waste on nearly-clean images

**Empirically**, 1000 steps is found to be sweet spot for most domains. Modern methods use **acceleration techniques** (DDIM, score-based samplers) to get quality with fewer steps.

---

### Q: What's the difference between the loss during training and the procedure during generation?

**A:** 

**Training:**
```
1. Sample real image x_0
2. Sample random t and noise ε
3. Compute x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
4. Predict: ε_pred = network(x_t, t)
5. Loss = ||ε - ε_pred||²
6. Backprop
```

**Generation:**
```
1. Start with x_T ~ N(0, I) (pure noise)
2. For t = T down to 1:
   a. Predict: ε_pred = network(x_t, t)
   b. Update: x_{t-1} = denoise_formula(x_t, ε_pred, t)
3. Return x_0
```

**Key difference**: Training optimizes a supervised loss; generation uses the trained network to *iteratively* clean up an image.

---

### Q: Can you use the trained network on images of different sizes?

**A:** 
- **Yes**, if U-Net is fully convolutional (no dense layers)
- **No**, if timestep embedding adds absolute positional information

For diffusion models, standard U-Nets are fully convolutional, so you can:
- Train on 28×28 MNIST
- Apply to 32×32, 64×64, etc.

**Caveat**: Quality may be worse on very different sizes (training distribution bias). Usually best to train on the target size.

---

### Q: What does the noise schedule control exactly?

**A:** The noise schedule $\bar{\alpha}_t$ controls:

1. **Signal-to-noise ratio at step t**: 
   - $\bar{\alpha}_t$ = proportion of *signal* (original image)
   - $1 - \bar{\alpha}_t$ = proportion of *noise*

2. **Difficulty curriculum for the network**:
   - At t=0: mostly signal (easy—barely noisy)
   - At t=T/2: balanced (medium difficulty)
   - At t=T: mostly noise (hard—denoising from scratch)

3. **Quality of final samples**:
   - Too slow schedule: too much noise remains at t=0
   - Too fast schedule: early timesteps have too much signal (network doesn't learn well)

Common schedules: **cosine**, linear, sqrt. Cosine is preferred because it spreads difficulty evenly.

---

## Comparison Questions

### Q: How is diffusion different from VAE (Variational Autoencoder)?

**A:**

| Aspect | Diffusion | VAE |
|--------|-----------|-----|
| **Generation** | Iterative denoising (1000 steps) | Single decoder pass |
| **Training** | Predict noise from corrupted image | Learn encoder + decoder + latent prior |
| **Speed** | Slow (many steps) | Fast |
| **Quality** | Often better (state-of-the-art) | Good but usually inferior |
| **Complexity** | Simpler conceptually | More complex (KL divergence, reparameterization) |
| **Flexibility** | Easy to add conditions (class, text) | Requires separate architecture |

Both are generative models, but diffusion's iterative approach gives it advantages in quality.

---

### Q: How is diffusion different from GAN (Generative Adversarial Network)?

**A:**

| Aspect | Diffusion | GAN |
|--------|-----------|-----|
| **Training** | Supervised (predict noise) | Adversarial (generator vs discriminator) |
| **Stability** | Stable, easy to train | Unstable, mode collapse issues |
| **Speed** | Slow sampling (iterative) | Fast sampling (single pass) |
| **Quality** | Excellent | Good but less consistent |
| **Convergence** | Guaranteed (like classification) | Not guaranteed |

Diffusion is more reliable but slower. GANs are faster but trickier to train.

---

## Math Questions

### Q: Why is the denoising formula so complicated? Can't we just subtract the predicted noise?

**A:** Good intuition! Mathematically, it's:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta \right) + \text{noise}$$

This looks complicated because:

1. **$\frac{1}{\sqrt{\alpha_t}}$**: Normalization factor (scales the update correctly)
2. **$\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}$**: Converts noise prediction to correct magnitude
3. **$+\text{noise}$**: Small random component (adds exploration)

If you *just* subtracted $\epsilon_\theta$, the values wouldn't be normalized correctly for the next step. This formula ensures $x_{t-1}$ has the right distribution.

---

### Q: What does √(ᾱ_t) actually represent?

**A:** It's the **amplitude of the signal** at step t.

If you think of $x_t$ as a mixture of two components:
$$x_t = \sqrt{\bar{\alpha}_t} \cdot \text{[signal]} + \sqrt{1 - \bar{\alpha}_t} \cdot \text{[noise]}$$

Then:
- $\sqrt{\bar{\alpha}_t}$ = how much of the original image remains
- $\sqrt{1 - \bar{\alpha}_t}$ = how much noise has been added

As $t$ increases:
- $\bar{\alpha}_t$ decreases (signal fades)
- $1 - \bar{\alpha}_t$ increases (noise dominates)

---

## Troubleshooting

### Q: My model training loss isn't decreasing. What's wrong?

**A:** Common issues:

1. **Learning rate too high**: Loss oscillates or explodes. Try 1e-4 or 1e-5
2. **Timestep embedding not working**: Make sure timesteps are properly encoded
3. **Data preprocessing**: Ensure images are normalized (e.g., to [0, 1] or [-1, 1])
4. **Skip connections not concatenating correctly**: Check tensor shapes match at concat
5. **Noise schedule**: Verify $\bar{\alpha}_t$ is in [0, 1] and monotonically decreasing

Start with a simple forward pass (no training) to verify shapes.

---

### Q: Generated images are blurry. How do I improve quality?

**A:**

1. **Train longer**: More epochs → better noise prediction
2. **Better architecture**: More channels, deeper encoder, better skip connections
3. **Better noise schedule**: Experiment with cosine vs linear vs sqrt
4. **Classifier-free guidance**: Add conditional generation (even without labels)
5. **Sampling tricks**: Use DDIMSampler instead of pure denoising (faster + better)
6. **Post-processing**: Can optionally sharpen final images

Training longer is usually the simplest fix.

---

## References & Further Reading

- **Original Diffusion Models Paper**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **U-Net Architecture**: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)
- **Noise Schedules**: "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
- **Classifier-Free Guidance**: "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2021)
- **Fast Sampling (DDIM)**: "Denoising Diffusion Implicit Models" (Song et al., 2020)

---

## How to Contribute

See a question or issue with the tutorial? 
1. Open an issue or discussion
2. We'll update both the FAQ and the relevant tutorial file
3. This keeps knowledge centralized and discoverable


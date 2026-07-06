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

### Q: The label is epsilon (noise per pixel). Is this the cumulative noise added over ALL T steps, or just the noise in a single step?

**A:** Great distinction! It's neither—it's the **equivalent noise in the jump formula**, not cumulative.

**The confusion:**

There are two ways to think about noise:

**Option 1: Step-by-step (iterative)**
$$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1$$
$$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2$$
$$\vdots$$
$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_t$$

Each step has its own $\epsilon_i$. But this is NOT what we use for training.

**Option 2: Direct jump (what we actually use)**
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

There's ONE epsilon, sampled once. This $\epsilon$ is the "equivalent noise" that transforms $x_0$ directly to $x_t$ in one shot.

**What's the label in training?**

The label is **the single $\epsilon$ from the jump formula**, not cumulative:

```python
# Forward process (training):
x_0 = load_image()           # MNIST digit
epsilon = torch.randn_like(x_0)  # Sample ONE noise vector
t = random.randint(0, T)     # Sample ONE timestep

# Jump directly from x_0 to x_t
alpha_bar = get_alpha_bar(t)
x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon

# Train network to predict this SINGLE epsilon
epsilon_pred = network(x_t, t)
loss = mse(epsilon, epsilon_pred)  # Label is this ONE epsilon
```

**Why the jump formula instead of iterative?**

The jump formula is mathematically equivalent to the iterative process, but:
1. **Faster**: Compute $x_t$ in O(1) time, not O(t)
2. **Simpler**: One epsilon, not t different epsilons
3. **Better training**: Uniform sampling of timesteps

**Key insight: The relationship**

The jump formula with one $\epsilon$ is equivalent to iterating $x_0 → x_1 → ... → x_t$ with step-wise epsilons because of how the noise schedules multiply:

$$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$$

The cumulative product combines all the individual steps into one "jump noise."

**Practical implication:**

During generation, you don't need to know the step-wise epsilons. You only need the network's prediction of the single equivalent noise at each timestep:

```python
x_t = pure_noise  # Start here

for t in range(T, 0, -1):
    epsilon_pred = network(x_t, t)  # Predict the equiv. noise
    x_t = denoise_formula(x_t, epsilon_pred, t)  # Remove it
    
# Result: x_0
```

---

### Q: During training, does the U-Net train on all T timesteps sequentially? And do I need to feed it the original image x_0?

**A:** No to both! This is a key efficiency insight.

**Single-step training (not sequential):**

You do NOT iterate through timesteps 1→2→...→T during training. Instead:

```python
# NAIVE (WRONG - don't do this):
for t in range(1, T+1):
  x_t = forward_process(x_0, t)
  noise_pred = network(x_t, t)
  loss += ||noise - noise_pred||²
  # Very slow! T forward passes per image

# EFFICIENT (CORRECT - what we actually do):
t = random.randint(1, T)  # Sample ONE random timestep
x_t = forward_process(x_0, t)  # One forward pass
noise_pred = network(x_t, t)
loss = ||noise - noise_pred||²
# Single pass per image!
```

**Why this works:** By sampling random timesteps uniformly, you ensure the network learns denoising across all corruption levels. Over many training iterations, it sees enough variety of timesteps that it learns the full spectrum.

**Do you need x_0?**

YES, but NOT as an input to the network. Here's the distinction:

```
During training:

Step 1: Sample x_0 from training dataset
        (MNIST image)

Step 2: Use x_0 to CREATE noisy image x_t
        x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
        (You compute this, don't show to network)

Step 3: Feed ONLY x_t and t to network
        network(x_t, t) → predicts noise

Step 4: Compare prediction to actual noise ε
        loss = ||ε - ε_θ(x_t, t)||²

Network NEVER sees x_0 directly!
```

**Training loop (pseudo-code):**

```python
for epoch in range(num_epochs):
  for x_0 in training_dataset:  # x_0 is needed here
    
    # Sample random timestep
    t = random.randint(0, T)
    
    # Sample random noise
    epsilon = torch.randn_like(x_0)
    
    # Create noisy image (x_0 used here)
    alpha_bar = get_alpha_bar(t)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * epsilon
    
    # Network ONLY sees x_t and t, NOT x_0
    epsilon_pred = network(x_t, t)
    
    # Loss: predict the noise
    loss = mse_loss(epsilon, epsilon_pred)
    
    # Backprop
    loss.backward()
    optimizer.step()
```

**Per-pixel loss:**

Yes, the loss is computed per-pixel:

```
x_t shape: (B, 1, 28, 28)
epsilon shape: (B, 1, 28, 28)
epsilon_pred shape: (B, 1, 28, 28)

loss = mean((epsilon - epsilon_pred)²)  # MSE over all pixels
```

**Key efficiency insight:**

This is why diffusion models are practical to train:
- Each training step: one forward pass (not T passes)
- One loss computation (not T loss computations)
- Linear time in batch size, not exponential in T

If you had to train on all T timesteps sequentially, training would be 1000× slower!

---

### Q: What role does the training dataset have? The network is learning to remove noise (which is generic Gaussian noise). Isn't it just learning generic noise removal, independent of what images it sees?

**A:** This is a crucial misconception. The network is **NOT** learning generic Gaussian noise removal. It's learning **domain-specific denoising based on the training distribution**.

**The key insight:**
The network learns to predict noise, but this prediction is deeply conditioned on what images look like in the training set. Different training datasets → completely different learned behavior.

**Mathematical perspective:**

The forward process is generic:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

The noise $\epsilon$ is always standard Gaussian (generic). But the network learns:
$$\epsilon_\theta(x_t, t) = \text{predicted noise}$$

Here's the subtlety: **The network must predict what noise was added, given what it knows about valid images in the training distribution.**

```
If you show the network a very noisy image:
  - If trained on MNIST: "This noisy pattern looks like it could be a digit,
                          so the 'signal' is probably in these pixel clusters"
  - If trained on faces: "This noisy pattern looks like it could be a face,
                          so the 'signal' is probably in the eye/nose region"
  
Same noisy image, different predictions! Because different training distributions
have different notions of "what is likely signal vs noise"
```

**Concrete example:**

Imagine training two networks on 28×28 grayscale images, but:
- Network A: trained on MNIST digits (0-9)
- Network B: trained on random noise (pixels completely random)

At t=500 (halfway corruption), both see similar noisy images. But:

```
MNIST Network (A):
  "I know MNIST has connected strokes. This noisy blob looks like it could be
   a '3' or '5'. The noise is probably the random speckling. Let me predict
   the noise pattern that would most likely corrupt a digit."
   → Denoised output: looks like a digit

Random Noise Network (B):
  "In my training data, all pixels are independent random. Every pattern is
   equally likely. I have no notion of 'connected strokes' or 'digit shapes'."
   → Denoised output: random-looking pixels
```

**Why the training distribution matters:**

The network learns an implicit model of "what images in this distribution look like."

1. **Signal vs Noise**: Learned in context of training data
   - MNIST: "curved strokes + connected regions" = signal
   - Faces: "eyes, noses, symmetry" = signal  
   - Random noise: "any pattern" = signal

2. **Structural priors**: Absorbed from training examples
   - MNIST network learns digits are centered, bounded, have certain stroke widths
   - Face network learns faces have symmetry, proportions, etc.

3. **Feature restoration**: Only restores features seen in training
   - Train on smooth images → denoised output is smooth
   - Train on textured images → denoised output has texture
   - Train on random noise → denoised output is random noise

**This is why generation works:**

When you generate by iterative denoising from pure noise, the network gradually shapes the noise into valid samples from the **training distribution**:

```
Random Noise → [Network trained on MNIST] → MNIST-like digit
Random Noise → [Network trained on faces] → face-like image
Random Noise → [Network trained on random pixels] → random pixels
```

**The mathematics underneath:**

The loss is:
$$\mathcal{L} = \mathbb{E}_{x_0 \sim p_{\text{data}}, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

The key: **$p_{\text{data}}$ is your training distribution**. The network learns:
- "Given this corrupted image at timestep t, and knowing that $x_0$ comes from my training distribution, predict the noise"

This is distribution-specific. If you change $p_{\text{data}}$ (different dataset), you change what the network learns.

**Out-of-distribution behavior:**

If you generate from a network trained on MNIST but accidentally feed it a corrupted **face** image at test time:
```
Network was trained to denoise things that look like digits.
Face image doesn't match that distribution.
→ Network produces garbage or tries to "correct" it toward a digit shape
```

**Summary table:**

| Aspect | Truth | Misconception |
|--------|-------|----------------|
| **What's being learned?** | Domain-specific denoising | Generic Gaussian noise removal |
| **Does training data matter?** | Absolutely! Entirely determines learned behavior | No, only the noise schedule matters |
| **Can same network work on different datasets?** | No (unless trained jointly) | Yes (if just learning noise) |
| **Why does generation produce images "like training data"?** | Network learned to restore training distribution | Network has no preference |
| **Could you train on faces and generate digits?** | No (or very poorly) | Yes (noise removal is generic) |

**Practical implication:**

You can't just train a diffusion model on MNIST and then use it to generate faces. The network has learned "how to denoise things that look like handwritten digits," not "how to remove Gaussian noise in general."

---

### Q: What exactly is epsilon (ε)? If I sample the same x_0 at timestep 5 twice, will the noisy image x_5 be different both times?

**A:** Excellent question! This is crucial to understanding the forward process.

**What is ε?**
- ε is a **random variable** sampled from a standard normal distribution
- Mean: μ = 0
- Variance: σ² = 1
- Notation: $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
- Each time you sample, you get a different realization

**Sampling x_0 twice at t=5:**

Yes, you will get **different noisy images** both times. Here's why:

```
First training step:
  x_0 = [some MNIST digit]
  ε_1 ~ N(0, I)  [sample new random noise]
  x_5^(1) = √(ᾱ_5) * x_0 + √(1-ᾱ_5) * ε_1

Second training step (same x_0, same t):
  x_0 = [same MNIST digit]
  ε_2 ~ N(0, I)  [sample NEW random noise, different from ε_1]
  x_5^(2) = √(ᾱ_5) * x_0 + √(1-ᾱ_5) * ε_2

Result: x_5^(1) ≠ x_5^(2) because ε_1 ≠ ε_2
```

**Why this matters:**
- Every epoch, every batch, you get *different* corrupted versions of the same image
- This prevents the network from memorizing specific noisy patterns
- It creates diverse training data from limited real images
- The network learns to denoise **all possible noise realizations**, not just specific ones

**Code illustration:**
```python
x_0 = torch.randn(1, 1, 28, 28)  # Same MNIST digit

# First sample
epsilon_1 = torch.randn_like(x_0)  # Independent sample
x_5_v1 = torch.sqrt(alpha_bar_5) * x_0 + torch.sqrt(1 - alpha_bar_5) * epsilon_1

# Second sample
epsilon_2 = torch.randn_like(x_0)  # Independent sample
x_5_v2 = torch.sqrt(alpha_bar_5) * x_0 + torch.sqrt(1 - alpha_bar_5) * epsilon_2

# Check: are they different?
print(torch.allclose(x_5_v1, x_5_v2))  # False (almost certainly)
```

**Practical implications:**
- You can see the same image x_0 many times during training
- Each time it gets corrupted differently (different ε)
- Network learns robust denoising, not overfitting to specific noise patterns

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


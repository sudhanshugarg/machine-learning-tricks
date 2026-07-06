# Diffusion Models: Learning Path

Diffusion models are a class of generative models that learn to generate data by reversing a gradual noise corruption process. This learning path provides a structured progression from foundational concepts to advanced understanding.

---

## Learning Prerequisites

Before diving into diffusion models, ensure familiarity with:
- **Probability & Statistics**: Gaussian distributions, probability density functions, KL divergence
- **Deep Learning**: Neural networks, CNNs, attention mechanisms, backpropagation
- **Optimization**: Gradient descent, loss functions, training dynamics
- **Linear Algebra**: Matrix operations, eigenvalues, variance-covariance matrices

---

## Step-by-Step Learning Path

### **Step 1: Understand Generative Models (Foundation)**

**Goal**: Understand what generative models are and how they differ from discriminative models.

**Key Concepts**:
- Generative vs. Discriminative models
- Data distribution and probability modeling
- Common generative approaches: GANs, VAEs, Autoregressive models
- Why generative modeling is challenging

**What to Know**:
- Generative models aim to learn $p(x)$ — the data distribution
- Different architectures make different trade-offs between training stability, sample quality, and computational cost
- Diffusion models offer a more stable alternative to GANs

**Resources to Study**:
- Bayes theorem and conditional probability
- Maximum likelihood estimation (MLE)
- Basic understanding of VAEs and autoregressive models

---

### **Step 2: The Diffusion Process (Forward Process)**

**Goal**: Understand how noise gradually corrupts data in the forward process.

**Key Concepts**:
- Markov chain formulation
- Noise schedule: variance schedule $\{\beta_t\}$
- Cumulative noise: $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$
- The forward process equation:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

**What to Know**:
- The forward process is not trainable — it's fixed
- After sufficient steps, $x_T$ becomes nearly pure Gaussian noise
- The forward process is fully reversible in theory (but only with a learned reverse process)
- Different noise schedules (linear, cosine, learned) affect model performance

**Intuition**:
Think of it as gradually blurring an image — at each step, you add a small amount of Gaussian noise. After many steps, the image becomes indistinguishable from noise.

---

### **Step 3: The Reverse Process (Denoising)**

**Goal**: Understand how a neural network learns to reverse the noise corruption.

**Key Concepts**:
- The reverse process: going from noise back to data
- Bayes theorem for the reverse process:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t, x_0, t), \tilde{\beta}_t I)$$

- The neural network learns $p_\theta(x_{t-1} | x_t)$
- Parameterization options:
  - **Predicting noise**: $\epsilon_\theta(x_t, t)$ (most common, used in DDPM)
  - **Predicting the mean**: $\mu_\theta(x_t, t)$
  - **Predicting the score**: $\nabla_{x_t} \log q(x_t)$ (score-based models)

**What to Know**:
- The reverse process is learned through training
- The network is conditioned on the timestep $t$
- Training uses L2 loss between predicted and actual noise (for noise prediction)
- One forward pass through the network denoises one timestep

**Intuition**:
The neural network learns to gradually sharpen a noisy image back into a realistic image, one step at a time.

---

### **Step 4: The Training Objective**

**Goal**: Understand how diffusion models are trained.

**Key Concepts**:
- **Evidence Lower Bound (ELBO)** decomposition:

$$-\log p(x) \leq L = \mathbb{E}_q[D_{KL}(q(x_T|x_0) \parallel p(x_T))] + \sum_{t=2}^{T} \mathbb{E}_q[D_{KL}(q(x_{t-1}|x_t, x_0) \parallel p_\theta(x_{t-1}|x_t))] + \mathbb{E}_q[-\log p_\theta(x_0|x_1)]$$

- **Simplified objective**: For noise prediction parameterization:

$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

where $\epsilon \sim \mathcal{N}(0, I)$ and $x_t$ is computed from $x_0$ using the forward process

- **Weighted variants**: Different loss weightings for different timesteps
- **Variance weighting**: Whether to weight loss uniformly or by variance

**What to Know**:
- The simple objective is much easier to work with than the full ELBO
- Training samples random timesteps and denoises them
- The network effectively learns a denoising score function
- The loss is uniformly applied across all timesteps (or weighted)

**Training Loop**:
1. Sample a batch of data $x_0$
2. Sample random timesteps $t$
3. Sample noise $\epsilon$
4. Compute $x_t$ from $x_0$ and $\epsilon$
5. Compute network prediction $\hat{\epsilon}_\theta(x_t, t)$
6. Compute L2 loss and backpropagate

---

### **Step 5: Sampling (Reverse Process Execution)**

**Goal**: Understand how to generate new samples using the trained model.

**Key Concepts**:
- **Ancestral sampling**: Starting from $x_T \sim \mathcal{N}(0, I)$, iteratively apply the reverse process
- **Sampling equation**:

$$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} z$$

where $z \sim \mathcal{N}(0, I)$ for $t > 1$, and no noise added for $t=1$

- **Number of steps**: More steps (larger $T$) = better quality but slower; fewer steps = faster but lower quality
- **Guidance techniques**: Using classifiers or scores to guide generation

**What to Know**:
- Sampling is a sequential process that requires $T$ forward passes through the network
- The number of sampling steps is a trade-off between speed and quality
- Early stopping strategies can reduce steps while maintaining reasonable quality

**Intuition**:
Starting from pure noise, the model iteratively denoises, gradually sharpening random noise into a realistic sample.

---

### **Step 6: Conditioning and Guidance**

**Goal**: Learn how to control generation with conditions (class labels, text, images).

**Key Concepts**:
- **Conditional diffusion**: $p_\theta(x | c)$ where $c$ is a condition
- **Classifier-free guidance**: Blending unconditional and conditional predictions:

$$\hat{\epsilon}_\theta = \epsilon_\theta(x_t, t, c) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t))$$

where $s$ is the guidance scale

- **Classifier guidance**: Using a classifier gradient to guide generation
- **Cross-attention mechanisms**: Incorporating text encodings (for text-to-image models like Stable Diffusion)

**What to Know**:
- Guidance improves sample quality and controllability
- Classifier-free guidance is simpler and more effective than classifier guidance
- Guidance scale controls the strength of conditioning
- Text encodings (from CLIP or similar) enable text-to-image generation

---

### **Step 7: Advanced Topics**

**Goal**: Understand modern variants and improvements.

**Key Concepts**:

#### **a) Denoising Diffusion Probabilistic Models (DDPM)**
- The foundational paper that made diffusion models practical
- Fixed variance formulation
- Demonstrates competitive sample quality with GANs

#### **b) Improved DDPM (iDDPM)**
- Learned variance schedules
- Better sample quality and likelihood

#### **c) Denoising Diffusion Implicit Models (DDIM)**
- Deterministic sampling for faster generation
- Reduces required steps from $T$ to 50-100 without quality loss

#### **d) Score-based Generative Models**
- Predicting score functions instead of noise
- Connection to score matching and Langevin dynamics
- $\nabla_{x_t} \log p(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$

#### **e) Consistency Models**
- One-step generation by learning to map any noisy sample to clean data
- Trade-off: simpler training but potentially lower quality

#### **f) Latent Diffusion Models**
- Apply diffusion in a learned latent space (VAE) instead of pixel space
- Major speedup: Stable Diffusion uses this approach
- Encoder-decoder: $z = E(x)$, then apply diffusion on $z$, decode with $D(z')$

#### **g) Mixture of Experts and Guidance Extensions**
- Combining multiple expert denoisers
- Enhanced guidance with semantic or artistic objectives

---

### **Step 8: Connections to Other Frameworks**

**Goal**: Understand how diffusion models relate to other generative paradigms.

**Key Connections**:
- **Score-based models**: Diffusion models are equivalent to score-based generative modeling
- **Stochastic Differential Equations (SDEs)**: Diffusion processes can be formulated as SDEs
  - Forward SDE: $dx = f(x, t)dt + g(t)dw$
  - Reverse SDE: Obtained through Girsanov's theorem
- **Probability Flow ODE**: A deterministic alternative using ODEs instead of SDEs
- **Energy-based models**: Score functions are gradients of energy landscapes
- **Langevin sampling**: The reverse process is similar to Langevin dynamics from sampling

**What to Know**:
- These connections provide alternative perspectives on diffusion models
- They enable new techniques like ODE-based sampling (faster, no variance)
- Understanding multiple viewpoints deepens comprehension

---

## Key Mathematical Concepts to Master

### Gaussian Properties
- $\mathcal{N}(x; \mu, \sigma^2 I) \propto \exp(-\frac{1}{2\sigma^2} \|x - \mu\|^2)$
- Reparameterization trick: $x = \mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$

### KL Divergence
- $D_{KL}(P \| Q) = \mathbb{E}_P[\log P(x) - \log Q(x)]$
- For Gaussians: Closed form available
- Measures distribution mismatch

### Bayes Theorem
- $p(x|y) = \frac{p(y|x)p(x)}{p(y)}$
- Used to derive the reverse process distribution

### Variance Reduction in ELBO
- The ELBO is decomposed into interpretable terms
- Each term has a specific role in the loss

---

## Implementation Checkpoint

Once you understand Steps 1-5, you should be able to:
- ✅ Implement a basic diffusion model from scratch
- ✅ Train it on a simple dataset (MNIST, CIFAR-10)
- ✅ Generate samples using the trained model
- ✅ Understand training dynamics and sample quality evolution

For Steps 6-8:
- ✅ Add conditioning mechanisms
- ✅ Implement DDIM for faster sampling
- ✅ Understand modern architectures (UNets, Transformers for diffusion)

---

## Recommended Paper Reading Order

1. **Denoising Diffusion Probabilistic Models (DDPM)** — Foundational paper
2. **Diffusion Models Beat GANs on Image Synthesis** — Shows competitive quality
3. **Denoising Diffusion Implicit Models (DDIM)** — Fast sampling
4. **Score-Based Generative Modeling through Stochastic Differential Equations** — Theoretical framework
5. **High-Resolution Image Synthesis with Latent Diffusion Models** — Stable Diffusion (practical)
6. **Classifier-Free Diffusion Guidance** — Conditioning technique
7. **Consistency Models** — One-step generation

---

## Summary

| Step | Focus | Key Equation | Output |
|------|-------|--------------|--------|
| 1 | Generative models | - | Conceptual foundation |
| 2 | Forward process | $q(x_t \| x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$ | Noise schedule |
| 3 | Reverse process | $p_\theta(x_{t-1} \| x_t)$ | Network architecture |
| 4 | Training objective | $L = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | Loss function |
| 5 | Sampling | Ancestral sampling from $x_T$ | Generated samples |
| 6 | Conditioning | Classifier-free guidance | Controlled generation |
| 7 | Advanced variants | DDIM, latent diffusion, etc. | Faster/better models |
| 8 | Theoretical connections | SDEs, score functions | Unified understanding |

---

## Next Steps

After completing this learning path:
1. **Implement** basic DDPM from scratch
2. **Experiment** with different architectures (UNet, Transformer)
3. **Explore** variants (DDIM, latent diffusion)
4. **Apply** to downstream tasks (image-to-image, inpainting, super-resolution)
5. **Study** multimodal extensions (text-to-image, audio generation)

---

## Resources

- **Papers**: arXiv (https://arxiv.org)
- **Implementations**: Hugging Face Diffusers library
- **Blogs**: Lil'Log (lilianweng.github.io) has excellent posts
- **Courses**: MIT 6.S192, Stanford CS236 (generative models)

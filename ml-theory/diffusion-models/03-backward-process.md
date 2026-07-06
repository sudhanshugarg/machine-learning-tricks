# Backward Process: Learning to Denoise

## Overview

The backward process is where the neural network comes in. We train a network to **reverse the forward process** by learning to predict and remove noise.

---

## The Denoising Formula

At each backward step, we compute:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

Where:
- $x_{t-1}$ = denoised image at previous step
- $x_t$ = current noisy image
- $\epsilon_\theta(x_t, t)$ = **neural network's prediction of the noise** (this is what we train!)
- $\sigma_t z$ = small random noise (helps exploration, optional in generation)
- $\alpha_t, \bar{\alpha}_t$ = precomputed from noise schedule

**Intuition**: The network learns to estimate *"what noise was added?"* and we subtract it out.

---

## The Neural Network: U-Net Denoiser

### Architecture Overview

The network $\epsilon_\theta$ is a **U-Net** (encoder-decoder with skip connections):

```
Input: Noisy Image + Timestep Embedding
  ↓
[Encoder] Conv → Down → Down → Down (spatial dims decrease)
  ↓ (skip connections)
[Bottleneck] Core processing
  ↓
[Decoder] Up → Up → Up → Conv (spatial dims increase)
  ↓
Output: Predicted Noise
```

### MNIST Concrete Example

**Input to Network:**
- **Noisy image $x_t$**: shape `[B, 1, 28, 28]`
  - B = batch size
  - 1 = grayscale channel
  - 28×28 = spatial dimensions
  
- **Timestep embedding**: scalar $t$ → encoded to shape `[B, 128]`
  - We encode the timestep as learned sinusoidal embeddings
  - This tells the network "which denoising step are we on?"
  - Allows same network to work for all timesteps

**Output from Network:**
- **Predicted noise**: shape `[B, 1, 28, 28]`
  - Same shape as input image
  - Represents the Gaussian noise that was added

### Network Dimensions in Detail

```
Layer Details for MNIST U-Net:

Input: x_t [B, 1, 28, 28] + timestep emb [B, 128]

--- ENCODER (Down-sampling) ---
Conv2d(1, 64, kernel=3) 
  → [B, 64, 28, 28]
  
ResBlock(64, 128) + downsample
  → [B, 128, 14, 14]  (spatial halved)
  
ResBlock(128, 256) + downsample
  → [B, 256, 7, 7]    (spatial halved again)

--- BOTTLENECK (Core) ---
ResBlock(256, 256)
  → [B, 256, 7, 7]

--- DECODER (Up-sampling) ---
ResBlock(256+256, 128) + upsample  [skip connection!]
  → [B, 128, 14, 14]  (spatial doubled)
  
ResBlock(128+128, 64) + upsample   [skip connection!]
  → [B, 64, 28, 28]   (spatial doubled)

Output:
Conv2d(64, 1, kernel=3)
  → [B, 1, 28, 28]    (predicted noise)
```

### Why These Dimensions?

- **Input channels = 1**: MNIST is grayscale
- **Progressive channel growth (64→128→256)**: More capacity as we compress
- **Bottleneck at 7×7**: Enough to capture digit structure, small enough to be efficient
- **Skip connections**: Preserve details from encoder for decoder (crucial for image generation)
- **Output channels = 1**: Same as input (predict noise, not a different modality)

---

## Training the Network

### Training Objective (Loss Function)

We want the network to predict the added noise accurately:

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \| ^2 \right]$$

Where:
- $x_0$ = real MNIST image (from training set)
- $t$ = random timestep (sample uniformly from 1 to T)
- $\epsilon$ = random Gaussian noise
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ = noisy image
- Network must predict $\epsilon$ given $x_t$ and $t$

### Training Algorithm

```
For each training iteration:
  1. Sample real image x_0 from MNIST training set
  2. Sample random timestep t ∈ [1, T]
  3. Sample random noise ε ~ N(0, I)
  4. Compute noisy image: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
  5. Forward pass: ε_pred = network(x_t, t)
  6. Compute loss: L = ||ε - ε_pred||²
  7. Backprop and update network
```

Key insight: **The network sees noisy images at all timesteps equally**, so it learns to denoise at any corruption level.

---

## Generation: Using the Trained Network

Once trained, we can generate new MNIST digits from scratch:

### Generation Algorithm

```
1. Start with pure noise: x_T ~ N(0, I)  [shape: 1×28×28]

2. For t = T, T-1, ..., 1:
   a. Use network to predict noise: ε_pred = network(x_t, t)
   b. Apply denoising step: x_{t-1} = (1/√α_t) * (x_t - [(1-α_t)/√(1-ᾱ_t)] * ε_pred) + noise_term
   c. x_t := x_{t-1}

3. Return x_0 (final denoised image) — should look like a digit!
```

### Generation Shape Flow (MNIST)

```
x_1000: [1, 1, 28, 28] pure noise
  ↓ network(x_1000, 1000)
x_999: [1, 1, 28, 28] slightly denoised
  ↓ network(x_999, 999)
x_998: [1, 1, 28, 28] more denoised
  ...
x_1: [1, 1, 28, 28] nearly clean
  ↓ network(x_1, 1)
x_0: [1, 1, 28, 28] final generated digit
```

**Key**: Shapes stay constant throughout! Only the *content* changes (noise → signal).

---

## Code Sketch

```python
import torch
import torch.nn as nn

class UNetDenoiser(nn.Module):
    """U-Net that predicts noise from noisy images"""
    
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=128):
        super().__init__()
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder (downsampling)
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.down2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoder (upsampling)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x_t, t):
        """
        Args:
            x_t: noisy image [B, 1, 28, 28]
            t: timestep (scalar or [B])
        Returns:
            predicted noise [B, 1, 28, 28]
        """
        # Encode timestep
        t_emb = self.time_emb(torch.tensor(t, dtype=torch.float32).unsqueeze(-1))
        
        # Encoder with skip connections
        enc1 = self.enc1(x_t)  # [B, 64, 28, 28]
        down1 = self.down1(enc1)  # [B, 64, 14, 14]
        
        enc2 = self.enc2(down1)  # [B, 128, 14, 14]
        down2 = self.down2(enc2)  # [B, 128, 7, 7]
        
        # Bottleneck
        bottleneck = self.bottleneck(down2)  # [B, 256, 7, 7]
        
        # Decoder with skip connections
        up2 = self.up2(bottleneck)  # [B, 256, 14, 14]
        dec2_input = torch.cat([up2, enc2], dim=1)  # [B, 256+128, 14, 14]
        dec2 = self.dec2(dec2_input)  # [B, 128, 14, 14]
        
        up1 = self.up1(dec2)  # [B, 128, 28, 28]
        dec1_input = torch.cat([up1, enc1], dim=1)  # [B, 128+64, 28, 28]
        dec1 = self.dec1(dec1_input)  # [B, 64, 28, 28]
        
        # Output (predicted noise)
        out = self.out(dec1)  # [B, 1, 28, 28]
        
        return out

# Training example
model = UNetDenoiser()
x_t = torch.randn(4, 1, 28, 28)  # batch of 4 noisy MNIST images
t = 500  # timestep
noise_pred = model(x_t, t)

print(f"Input shape: {x_t.shape}")      # torch.Size([4, 1, 28, 28])
print(f"Output shape: {noise_pred.shape}")  # torch.Size([4, 1, 28, 28])
```

---

## Summary Table

| Component | Details |
|-----------|---------|
| **Input** | Noisy image [B, 1, 28, 28] + timestep $t$ |
| **Output** | Predicted noise [B, 1, 28, 28] |
| **Architecture** | U-Net (encoder-decoder) |
| **Training Loss** | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ |
| **Key Insight** | Network learns to denoise at ANY corruption level |
| **Generation** | Iteratively apply network 1000 times (t=T down to 1) |

---

## How It All Fits Together

1. **[Forward Process](02-forward-process.md)**: Adds noise deterministically
2. **This Process**: Network learns to reverse it by predicting noise
3. **[Overview](01-overview.md)**: Big picture of how both work together

The elegance: A single network trained on noise prediction becomes a powerful generative model!

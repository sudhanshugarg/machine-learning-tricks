# ML Theory FAQ

General questions that span multiple topics in `/ml-theory/`. Topic-specific FAQs live inside each topic's own `FAQ.md` (e.g. [contrastive-learning/FAQ.md](contrastive-learning/FAQ.md)).

---

### Q: What's the difference between RMSprop, Momentum, Adam, and AdamW?

**A:** They're all answers to the same question — plain SGD uses one fixed learning rate for every parameter, which is a bad fit when different parameters need very different step sizes or when the loss surface is a narrow ravine. Each optimizer below fixes a different part of that problem, and each one builds on the previous.

#### The starting point: plain SGD

$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

**Problem:** in a ravine-shaped loss surface (steep in one direction, shallow in another), SGD oscillates across the steep direction instead of making progress along the shallow one. It also treats every parameter identically, even though some parameters (e.g. embeddings for rare tokens) get sparse, small gradients and others get large, frequent ones.

---

#### Momentum — smooth out oscillation with a velocity term

$$v_t = \beta v_{t-1} + \nabla J(\theta_t)$$
$$\theta_{t+1} = \theta_t - \alpha v_t$$

**Intuition:** instead of stepping purely on the current gradient, accumulate an exponentially-decaying running average of past gradients (like a ball rolling downhill picking up speed). Gradient components that consistently point the same way (the shallow, useful direction) reinforce each other and grow; components that flip sign every step (the steep, oscillating direction) cancel out. Typical $\beta = 0.9$.

**What it fixes:** oscillation and slow progress in ravines. **What it doesn't fix:** it still uses one global learning rate $\alpha$ for every parameter.

---

#### RMSprop — per-parameter learning rates via a running average of squared gradients

$$r_t = \beta r_{t-1} + (1-\beta)(\nabla J)^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{r_t + \epsilon}} \nabla J(\theta_t)$$

**Intuition:** track a running average of the *squared* gradient magnitude per parameter, $r_t$. Dividing the update by $\sqrt{r_t}$ means parameters with a history of large gradients get their effective step size shrunk, and parameters with small/sparse gradients get their effective step size boosted. This gives each parameter its own adaptive learning rate instead of one shared $\alpha$. $\epsilon$ (e.g. $10^{-8}$) just avoids division by zero.

**What it fixes:** the one-learning-rate-for-all-parameters problem. **What it doesn't fix:** it has no momentum — no memory of gradient *direction*, only gradient *magnitude*.

---

#### Adam — Momentum + RMSprop combined, with bias correction

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla J(\theta_t) \quad \text{(momentum: 1st moment, mean of gradients)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla J(\theta_t))^2 \quad \text{(RMSprop: 2nd moment, mean of squared gradients)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha \, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Intuition:** Adam is literally "Adaptive Moment Estimation" — it keeps both a momentum term ($m_t$, the direction to move in) and an RMSprop-style term ($v_t$, how much to scale that step per parameter). Defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$.

**Why bias correction matters:** $m_0$ and $v_0$ are initialized to zero, so early in training the running averages are biased toward zero (they haven't "warmed up" yet). Dividing by $(1-\beta_1^t)$ and $(1-\beta_2^t)$ corrects for this — without it, the first several steps would take artificially tiny steps. As $t \to \infty$, $\beta^t \to 0$ and the correction fades away.

**What it fixes:** combines directional smoothing (momentum) with per-parameter step-size adaptation (RMSprop) — this is why Adam is the default optimizer for most deep learning. **What it doesn't fix:** how weight decay (L2 regularization) interacts with the adaptive learning rate — see below.

---

#### AdamW — Adam with *decoupled* weight decay

Standard L2 regularization is normally implemented by adding $\lambda \theta$ to the gradient before it ever reaches the optimizer:

$$\nabla J(\theta_t) \leftarrow \nabla J(\theta_t) + \lambda \theta_t$$

**The problem in Adam specifically:** if you do this, the weight-decay term $\lambda \theta_t$ gets folded into $m_t$ *and* $v_t$ before the adaptive scaling is applied. That means the effective amount of decay applied to each parameter gets divided by $\sqrt{\hat{v}_t}$ just like the gradient does — parameters with large historical gradients get *less* weight decay, and parameters with small gradients get *more*. This is an accident of how Adam is built, not an intentional regularization design, and empirically it makes L2 regularization much less effective in Adam than in plain SGD.

**AdamW's fix:** decouple weight decay from the gradient-based moment estimates entirely — apply it as a separate, direct shrinkage of the weights at the update step:

$$\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Now every parameter decays toward zero by the same proportion $\alpha \lambda$ regardless of its gradient history — matching the original, intended behavior of weight decay. This is why AdamW (not plain Adam) is the standard choice whenever regularization matters, e.g. training transformers.

---

#### Summary table

| Optimizer | Tracks | Fixes | Still missing |
|---|---|---|---|
| SGD | nothing extra | baseline | oscillation in ravines, one global LR |
| Momentum | running mean of gradient (1st moment) | oscillation / slow progress in ravines | still one global LR for all params |
| RMSprop | running mean of squared gradient (2nd moment) | per-parameter adaptive LR | no directional memory |
| Adam | both 1st and 2nd moment, bias-corrected | combines momentum + adaptive LR | weight decay interacts badly with adaptive LR |
| AdamW | same as Adam | decouples weight decay from adaptive LR | — |

See [gradient_descent/explanation.md](gradient_descent/explanation.md) for how these fit into the broader gradient descent family (batch/stochastic/mini-batch) and learning-rate considerations.

---

### Q: If a model has N parameters, initialized with mean $\mu = 0$ and std $\sigma = 1$, what learning rate, $\beta_1$, and $\beta_2$ should Adam use?

**A:** There isn't a clean formula that maps $N$ and $\sigma$ directly onto these hyperparameters — and the reason why is itself the useful part of the answer. It comes down to a property of Adam called **update-magnitude invariance**, plus a real numerical-stability problem hiding in this particular init scheme.

#### Why N doesn't plug directly into the Adam learning rate

Adam's update is:

$$\theta_{t+1} = \theta_t - \frac{\alpha \, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

$\hat{m}_t$ is (roughly) the mean gradient and $\hat{v}_t$ is (roughly) the mean *squared* gradient, so $\hat{m}_t / \sqrt{\hat{v}_t}$ is close to $\pm 1$ in magnitude regardless of how large or small the raw gradient actually is — dividing a quantity by its own standard deviation always yields something close to unit scale. This is exactly why Adam was designed this way: **the effective step size per parameter is approximately $\alpha$, independent of the gradient's scale**, as long as the gradient's scale is roughly stationary over the recent averaging window. So scaling up $N$ (more parameters) or $\sigma$ (larger init) does not, on its own, require rescaling $\alpha$ — Adam already normalizes that out per parameter. This is in sharp contrast to plain SGD, where the update *is* the raw gradient times $\alpha$, so init scale and parameter count matter directly.

#### Where N and σ = 1 actually do bite you

The catch is "roughly stationary" above. This particular init — $\sigma = 1$ for every parameter, with no dependence on layer width — is not the same as standard He/Xavier init, which uses $\sigma = 1/\sqrt{\text{fan-in}}$. For a linear layer with fan-in $N_{in}$ and weights $w_i \sim \mathcal{N}(0, 1)$:

$$\text{Var}(z) = \text{Var}\left(\sum_{i=1}^{N_{in}} w_i x_i\right) \approx N_{in} \cdot \text{Var}(w) \cdot \text{Var}(x) = N_{in} \cdot \text{Var}(x)$$

So $\text{std}(z) \approx \sqrt{N_{in}} \cdot \text{std}(x)$ — every layer amplifies the signal by $\sqrt{N_{in}}$, and this compounds multiplicatively with depth. For any layer with more than a handful of inputs, this explodes: pre-activations, logits, and the loss itself can be enormous in the first few steps, which means the very first gradients are enormous and *not* representative of the gradient scale later in training. That breaks the "roughly stationary" assumption Adam's normalization depends on — $\hat{v}_t$ hasn't seen enough history yet to average out a huge early outlier, and (with $t=1,2,3,\dots$) the bias-correction term $1-\beta_2^t$ is still small, so $\hat{v}_t$ can be a noisy, unreliable estimate right when the raw gradients are at their most extreme.

This is a property of $N$ and $\sigma$ acting *through the forward pass*, not through Adam's math directly — so the fix isn't "solve for $\alpha(N, \sigma)$," it's "protect the optimizer during the unstable first few hundred/thousand steps."

#### Practical recommendation

| Hyperparameter | Value | Why |
|---|---|---|
| Peak learning rate $\alpha$ | Standard default, e.g. $3\text{e-}4$ to $1\text{e-}3$ | Adam's per-parameter normalization means $\alpha$ doesn't need to shrink just because $N$ is large — it sets the *rate of adaptation*, not the raw step size. |
| Warmup | Linear warmup from $\approx 0$ to peak $\alpha$ over the first few hundred–few thousand steps | This — not a smaller $\alpha$ — is the real fix for the exploding-activation problem above: it keeps early steps tiny while $\hat{v}_t$ accumulates enough history to become a trustworthy estimate, then ramps up once training has stabilized. |
| $\beta_1$ | $0.9$ (default) | Controls momentum smoothing; largely independent of $N$/$\sigma$ — no reason to deviate from the default here. |
| $\beta_2$ | $0.999$ default; consider $0.95$–$0.98$ if gradients are very noisy at this scale | $\beta_2$ controls how much history $\hat{v}_t$ averages over. A *higher* $\beta_2$ smooths out the wild early gradients caused by this init (good for stability) but reacts more slowly to real curvature changes later; a *lower* $\beta_2$ adapts faster but is more exposed to exactly the instability this init scheme creates. In practice, large models (e.g. GPT-3, PaLM) use $\beta_2 \approx 0.95$ specifically because at scale, gradients from large batches are already fairly stable, so the faster-adapting lower $\beta_2$ wins — but that argument assumes big-batch averaging is already smoothing things out, which is a separate effect from $N$ and $\sigma$ themselves. |
| $\epsilon$ | Slightly larger than the $10^{-8}$ default (e.g. $10^{-6}$) | Guards against division blow-ups if $\hat{v}_t$ is ever near zero for some parameter during the volatile early phase. |

**Bottom line:** don't try to solve for $\alpha$, $\beta_1$, $\beta_2$ as functions of $N$ and $\sigma$ — Adam's per-parameter normalization already absorbs most of that. The one real consequence of $\sigma=1$ init (as opposed to properly fan-in-scaled init) is that it produces exploding forward-pass activations for any layer with meaningfully large fan-in, and the standard, well-tested fix for that is learning-rate warmup, not a change to the steady-state $\alpha$. If you have the freedom to change the init instead of just the optimizer, using $\sigma = 1/\sqrt{N_{in}}$ (or LayerNorm/BatchNorm to normalize activations directly) removes the problem at its source and makes the warmup less critical.

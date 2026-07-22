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

# DPO — Direct Preference Optimization

**In one sentence:** DPO throws away the reward model *and* the RL loop, and instead trains the policy **directly** on preference pairs with a single classification-style loss — "make the chosen answer more likely than the rejected one, relative to a reference model."

DPO's headline result: *"Your language model is secretly a reward model."* It shows you never needed to train a separate reward model or run PPO at all — the same optimum can be reached with a supervised loss on a fixed dataset.

---

## What You Need (and Don't Need)

**You need:** a dataset of preference triples $(x, y_w, y_l)$ and a frozen reference model $\pi_{\text{ref}}$.

| Symbol | Meaning |
|--------|---------|
| $x$ | the prompt |
| $y_w$ | the **preferred** ("winning") response — the one a human liked more |
| $y_l$ | the **dispreferred** ("losing") response |
| $\pi_{\text{ref}}$ | frozen reference model (usually the SFT model) |

**You don't need:** a reward model, a critic/value network, or any online generation. DPO is fully **offline** — it trains on a static dataset like ordinary supervised learning. Only **two models** are in memory: the policy $\pi_\theta$ and the reference $\pi_{\text{ref}}$ (half of PPO's four).

---

## The Big Idea: Reward Is Hidden Inside the Policy

DPO starts from the *same* KL-regularized objective as PPO:

$$\max_{\pi_\theta}\; \mathbb{E}_{x, y\sim\pi_\theta}\big[r(x,y)\big] - \beta\, D_{KL}\big(\pi_\theta \| \pi_{\text{ref}}\big)$$

This objective has a **known closed-form optimal policy**:

$$\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y|x)\, \exp\!\Big(\tfrac{1}{\beta} r(x,y)\Big)$$

**Where** $Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp(\tfrac{1}{\beta}r(x,y))$ is the **partition function** (a normalizer summing over all possible responses — intractable to compute).

Now the clever algebra: **solve this for the reward** $r$. Rearranging gives

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**Read this literally:** the reward can be expressed purely in terms of the (optimal) policy and the reference. If we let *our* trainable policy $\pi_\theta$ play the role of $\pi^*$, then $\pi_\theta$ *implicitly defines* a reward:

$$\boxed{\;\hat{r}_\theta(x,y) = \beta \log \dfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\;}$$

This is the **implicit reward** — the single most important quantity in DPO. Notice the ugly $Z(x)$ term is still there, but watch it disappear next.

---

## Killing the Partition Function with Bradley-Terry

Human preferences are modeled with **Bradley-Terry**: the probability that $y_w$ beats $y_l$ depends only on the *difference* of their rewards:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$

where $\sigma$ is the sigmoid. Because this uses a **difference of rewards for the same prompt** $x$, the $\beta\log Z(x)$ term is identical in both and **cancels**:

$$r(x,y_w) - r(x,y_l) = \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

No more intractable normalizer. This cancellation is *why* DPO works without a reward model.

---

## The DPO Loss

Maximize the likelihood of the observed preferences under Bradley-Terry → minimize:

$$L^{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\,\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \;-\; \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

Let's dissect every term.

**The two log-ratios (the implicit rewards):**
- $\log \dfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}$ — how much **more likely** the *current* policy makes the **winning** response compared to the reference. This is $\hat r_\theta(x,y_w)/\beta$.
- $\log \dfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ — same thing for the **losing** response, $\hat r_\theta(x,y_l)/\beta$.
- Comparing to $\pi_{\text{ref}}$ (rather than raw $\pi_\theta$) is what keeps the model from drifting — it's the KL leash, baked directly into the loss.

**$\beta$ (the temperature / KL strength):**
- Scales how sharply the loss reacts to the reward gap. Same role as $\beta$ in PPO: larger $\beta$ → stay closer to $\pi_{\text{ref}}$ (more conservative); smaller $\beta$ → allow bigger preference-driven changes. Typical values: $0.1$–$0.5$.

**The difference (winning implicit reward − losing implicit reward):**
- This is the **margin** — how much the policy prefers $y_w$ over $y_l$ (relative to the reference). We want it **large and positive**.

**$\sigma(\cdot)$ (sigmoid):**
- Squashes the margin into a probability "$P(\text{model orders this pair correctly})$." Comes straight from Bradley-Terry.

**$-\log\sigma(\cdot)$ (the outer loss):**
- Standard **binary cross-entropy / logistic loss**. It's near 0 when the margin is large and positive (pair ordered correctly, confidently), and grows large when the margin is negative (model prefers the *wrong* answer). DPO is, quite literally, **training a binary classifier** to rank $y_w$ above $y_l$.

**Intuition:** "Increase the log-prob of the good answer and decrease the log-prob of the bad answer — but measure both *relative to the reference model*, and stop pushing once the pair is confidently ordered right."

---

## What the Gradient Actually Does

The gradient of the DPO loss is illuminating:

$$\nabla_\theta L^{\text{DPO}} = -\beta\, \mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big)}_{\text{weight: how wrong we are}}\;\big(\underbrace{\nabla_\theta \log\pi_\theta(y_w|x)}_{\text{push up } y_w} - \underbrace{\nabla_\theta \log\pi_\theta(y_l|x)}_{\text{push down } y_l}\big)\Big]$$

**Where:**
- $\nabla_\theta \log\pi_\theta(y_w|x)$ = direction that **increases** the probability of the winning response.
- $-\nabla_\theta \log\pi_\theta(y_l|x)$ = direction that **decreases** the probability of the losing response.
- The **weight** $\sigma(\hat r_\theta(y_l) - \hat r_\theta(y_w))$ = large when the implicit reward model currently has the pair **backwards** (thinks $y_l$ is better). It's the automatic hard-example weighting: pairs the model already gets right contribute almost nothing; pairs it gets wrong dominate the update.

This is exactly what an RLHF reward-maximizer would do — but with no reward model and no sampling.

---

## Implementation Sketch

Because it's offline, DPO is remarkably simple to code:

```python
# Precompute reference log-probs once (π_ref is frozen).
# For each preference pair (x, y_w, y_l):

# Sum of token log-probs = log π(y | x)
logp_w   = logprob(policy,    x, y_w)   # log π_θ(y_w | x)
logp_l   = logprob(policy,    x, y_l)   # log π_θ(y_l | x)
ref_logp_w = logprob(reference, x, y_w) # log π_ref(y_w | x)
ref_logp_l = logprob(reference, x, y_l) # log π_ref(y_l | x)

# Log-ratios = implicit rewards / β
pi_logratio  = logp_w   - logp_l        # policy prefers w over l by...
ref_logratio = ref_logp_w - ref_logp_l  # reference prefers w over l by...

# The margin inside the sigmoid
logits = beta * (pi_logratio - ref_logratio)

loss = -F.logsigmoid(logits).mean()     # -log σ(margin)
```

Note: `pi_logratio - ref_logratio` regroups the same four terms as the boxed loss (subtracting reference log-ratios instead of forming per-response ratios) — algebraically identical.

---

## Common Pitfalls

- **Both $y_w$ and $y_l$ probabilities can drop.** DPO only cares about the *gap*. It's common to see the absolute log-prob of the chosen response *decrease* during training, as long as the rejected drops faster. If this hurts you, that's part of what ORPO's added SFT term fixes ([04-orpo.md](04-orpo.md)).
- **Reference model quality matters.** DPO anchors everything to $\pi_{\text{ref}}$. A weak SFT reference caps quality.
- **Length bias.** Longer responses have lower total log-prob (more tokens multiplied); if $y_w$ and $y_l$ differ a lot in length, the model can learn length instead of quality. (ORPO length-normalizes to address this.)

---

## PPO vs DPO Cheat Sheet

| Aspect | PPO | DPO |
|--------|-----|-----|
| Reward model | Separate, trained | **None** (implicit) |
| Critic / value net | Yes | **No** |
| Reference model | Yes | Yes |
| Online sampling | Yes (generate + score each step) | **No** (fixed dataset) |
| Models in memory | 4 | **2** |
| Loss type | Clipped RL surrogate | Binary cross-entropy on pairs |
| Stability / simplicity | Fiddly | Much simpler |
| Ceiling / flexibility | Higher, more control | Slightly lower, but usually "good enough" |

**Next:** [03-grpo.md](03-grpo.md) goes the other direction — it *keeps* the RL loop but removes the critic instead of the reward model.

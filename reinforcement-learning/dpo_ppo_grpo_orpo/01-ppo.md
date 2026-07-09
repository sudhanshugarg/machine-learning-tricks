# PPO — Proximal Policy Optimization

**In one sentence:** PPO is a reinforcement-learning algorithm that improves the policy in small, "clipped" steps so it never moves too far in one update — and in RLHF it's used to maximize a learned reward while staying close to a reference model.

PPO is the foundation. Its vocabulary — *reward, advantage, ratio, clipping, KL penalty, critic* — is reused (or deliberately removed) by every other method in this folder, so it's worth understanding first.

---

## The RLHF Pipeline (Where PPO Fits)

PPO is **stage 3** of the classic RLHF recipe:

```
1. SFT            Supervised fine-tune a base model on demonstrations   → π_ref (frozen)
2. Reward Model   Train r_φ(x,y) on human preference pairs              → r_φ (frozen)
3. PPO (RL)       Optimize the policy π_θ to maximize r_φ, kept near π_ref
```

So by the time PPO runs, you already have **two frozen models**: the reference/SFT model $\pi_{\text{ref}}$ and the reward model $r_\phi$. PPO trains a **third** (the policy $\pi_\theta$) and a **fourth** (a value network / critic — see below). Four models in memory is exactly why PPO is expensive.

---

## Step 1: The Objective PPO Is Trying to Maximize

RLHF wants to maximize expected reward without drifting from the reference:

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot|x)}\big[ r_\phi(x, y) \big] \;-\; \beta\, D_{KL}\!\big(\pi_\theta(y|x)\,\|\,\pi_{\text{ref}}(y|x)\big)$$

**Where:**
- $x$ = a prompt drawn from dataset $D$; $y$ = a response the policy **generates**.
- $r_\phi(x,y)$ = the scalar score from the frozen **reward model** (higher = more preferred by humans).
- $\pi_{\text{ref}}$ = frozen reference (the SFT model). $\beta$ = strength of the KL leash.
- $D_{KL}$ = penalty for drifting from $\pi_{\text{ref}}$; stops **reward hacking** (finding gibberish that fools $r_\phi$).

In practice this KL term is not computed as a separate divergence — it's **folded into the per-token reward** (Step 3), which is a detail worth remembering because GRPO does it *differently*.

---

## Step 2: The Language Model as an RL Agent

To use RL, we reframe text generation as a sequential decision process:

| RL concept | In language-model RLHF |
|------------|------------------------|
| **State** $s_t$ | the prompt + all tokens generated so far: $(x, y_{<t})$ |
| **Action** $a_t$ | the next token $y_t$ |
| **Policy** $\pi_\theta(a_t\mid s_t)$ | the model's probability for the next token |
| **Reward** | reward-model score at the end of the sequence, plus per-token KL |
| **Episode** | generating one full response |

So each generated token is one "action," and a full response is one episode.

---

## Step 3: Per-Token Reward (Reward + KL Combined)

The reward model only gives a score for the **whole** response (a single number at the end). PPO needs a per-token reward signal, so it uses:

$$r_t = \underbrace{r_\phi(x, y)\cdot \mathbb{1}[t = T]}_{\text{sequence reward, only at last token}} \;-\; \underbrace{\beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}}_{\text{per-token KL penalty}}$$

**Where:**
- $\mathbb{1}[t=T]$ = indicator that we're at the **final token** $T$ — the reward-model score is only added once, at the end.
- The **KL term** is applied at *every* token: each token that makes the policy diverge from $\pi_{\text{ref}}$ is penalized a little. This is the KL leash from Step 1, distributed across tokens.
- $\beta$ = how hard the leash pulls.

**Intuition:** "You get one big grade at the end for the whole answer, and a small tax at every token for straying from how the reference model would have spoken."

---

## Step 4: Advantage — "Was This Action Better Than Expected?"

We don't want the raw reward; we want the **advantage**: how much better an action was than the *average* the model expected from that state. This is where the **critic** (value network) comes in.

- $V_\psi(s_t)$ = the **value function** (the critic), a learned network estimating the expected future reward from state $s_t$. It's the "baseline" or "expectation."
- The advantage answers: *did things turn out better ($\hat A_t > 0$) or worse ($\hat A_t < 0$) than the critic predicted?*

PPO estimates it with **Generalized Advantage Estimation (GAE)**:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$

**Where:**
- $\delta_t$ = the **TD (temporal-difference) error**: actual reward plus discounted next-state value, minus current-state value. A one-step "surprise."
- $\gamma$ = **discount factor** (0–1): how much future tokens' rewards matter now.
- $\lambda$ = the **GAE parameter** (0–1): trades off bias vs. variance in the advantage estimate. $\lambda=0$ → low-variance/high-bias one-step estimate; $\lambda=1$ → high-variance/low-bias full-return estimate.

**Why subtract a baseline?** Using advantage instead of raw reward dramatically reduces variance: we only push up actions that beat expectations, not every action in a lucky episode. *(Remember this — GRPO's entire trick is finding a baseline **without** a critic.)*

---

## Step 5: The Clipped Surrogate — The Heart of PPO

Now the actual PPO loss. First, the **probability ratio** — how much more (or less) likely the new policy makes an action compared to the policy that generated the data:

$$\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

**Where:**
- $\pi_{\theta_{\text{old}}}$ = the policy **as it was when we generated this batch** of responses (a snapshot, frozen for the update). We generate with $\pi_{\theta_{\text{old}}}$, then take several gradient steps on $\pi_\theta$ — that reuse is why the ratio exists.
- $\rho_t = 1$ means no change; $\rho_t > 1$ means the update made this token more likely; $\rho_t < 1$ less likely.

The clipped objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\rho_t(\theta)\,\hat{A}_t,\;\; \text{clip}(\rho_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\big)\Big]$$

**Where, term by term:**
- $\rho_t(\theta)\,\hat{A}_t$ = the **unclipped** objective — standard policy gradient: increase probability of good actions ($\hat A_t>0$), decrease bad ones.
- $\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)$ = force the ratio to stay inside $[1-\epsilon,\, 1+\epsilon]$ (typically $\epsilon = 0.2$).
- $\min(\cdot, \cdot)$ = take the **more pessimistic** of clipped and unclipped. This is the key: it removes any incentive to push the ratio far beyond the trust region.

**Intuition for the clip:** if an action was good ($\hat A_t > 0$), we *do* want to make it more likely — but only up to $1+\epsilon$. Beyond that, the $\min$ caps the gain, so there's no reward for a giant, destabilizing update. Symmetrically for bad actions. This is the "**proximal**" in Proximal Policy Optimization — updates stay near the old policy.

> Note: PPO maximizes $L^{\text{CLIP}}$. As a **loss** to minimize, use $-L^{\text{CLIP}}$. The other files write everything as losses to minimize.

---

## Step 6: The Full PPO Loss

The complete training loss combines three pieces:

$$L^{\text{PPO}}(\theta, \psi) = \mathbb{E}_t\Big[\, \underbrace{-L^{\text{CLIP}}_t}_{\text{policy}} \;+\; \underbrace{c_1\, L^{\text{VF}}_t}_{\text{value}} \;-\; \underbrace{c_2\, S\big[\pi_\theta\big](s_t)}_{\text{entropy}} \,\Big]$$

**Where:**
- $L^{\text{CLIP}}$ = the clipped policy objective from Step 5 (the main event).
- $L^{\text{VF}}_t = \big(V_\psi(s_t) - V_t^{\text{target}}\big)^2$ = the **value/critic loss** — trains the critic to predict returns accurately (a plain regression). $V_t^{\text{target}}$ is the observed return (e.g., $\hat A_t + V_\psi(s_t)$).
- $S[\pi_\theta](s_t)$ = the **entropy** of the policy at state $s_t$ — an exploration bonus. Subtracting it (i.e. rewarding high entropy) discourages the policy from collapsing to overconfident, deterministic outputs too early.
- $c_1, c_2$ = weighting coefficients for the value loss and entropy bonus.

**The three jobs:** (1) improve the policy safely, (2) keep the critic accurate so advantages are trustworthy, (3) keep some exploration.

---

## The Training Loop

```
for each iteration:
    1. Sample prompts x from D
    2. Generate responses y with π_θ_old  (the current policy snapshot)
    3. Score responses with reward model r_φ  → sequence reward
    4. Compute per-token rewards r_t (reward + KL penalty)   [Step 3]
    5. Compute advantages Â_t via GAE using critic V_ψ       [Step 4]
    6. For K epochs over this batch:
         - compute ratio ρ_t = π_θ / π_θ_old
         - compute clipped policy loss, value loss, entropy   [Step 6]
         - backprop, update θ (policy) and ψ (critic)
    7. Set π_θ_old ← π_θ
```

Steps 2–3 are **online**: PPO must generate and score fresh samples every iteration. This is the expensive, fiddly part that DPO and ORPO eliminate.

---

## Strengths and Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|-------------|---------------|
| Most expressive / highest ceiling on quality | **4 models** in memory (policy, ref, reward, critic) |
| Explicit reward model → flexible reward shaping | Unstable; many hyperparameters ($\epsilon, \beta, \gamma, \lambda, c_1, c_2$) |
| Online exploration can discover new behaviors | Online generation loop is slow and complex to implement |
| Well-studied, battle-tested (InstructGPT) | Reward hacking risk if reward model is weak |

---

## The Mental Model to Carry Forward

Every other method in this folder is "PPO minus something":

- **GRPO** = PPO **minus the critic** (replace the value baseline with a group average). See [03-grpo.md](03-grpo.md).
- **DPO** = PPO **minus the reward model and the whole RL loop** (derive the reward implicitly, optimize offline). See [02-dpo.md](02-dpo.md).
- **ORPO** = **minus the reference model too**, folded into SFT. See [04-orpo.md](04-orpo.md).

Keep the vocabulary — *ratio, clip, advantage, KL, reference* — handy; you'll see each piece reappear or get removed.

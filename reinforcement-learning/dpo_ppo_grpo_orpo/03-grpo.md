# GRPO — Group Relative Policy Optimization

**In one sentence:** GRPO is PPO with the **critic (value network) deleted** — instead of a learned baseline, it samples a *group* of answers to the same prompt and uses the group's average reward as the baseline.

GRPO comes from DeepSeekMath and is the algorithm behind DeepSeek-R1's reasoning training. It shines when you have a **verifiable reward** (math answers, code that passes tests, format checks) and want RL without the cost and instability of a critic.

---

## The Motivation: The Critic Is Expensive and Hard

Recall from [01-ppo.md](01-ppo.md) that PPO needs a **critic** $V_\psi(s_t)$ to compute the advantage baseline — "how good did we expect this state to be?" That critic is:

- A **whole extra network** the size of the policy (memory + compute).
- **Hard to train** for language: it must assign a value to every partial sequence, and rewards are sparse (often only at the end).

GRPO's insight: for a given prompt, if I generate **several** answers, I can just use their **average reward** as the baseline. Good answers beat the average → positive advantage; bad answers → negative. **No critic needed.**

---

## Step 1: Sample a Group

For each prompt (question) $q$, sample a **group** of $G$ responses from the old policy:

$$\{o_1, o_2, \dots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$$

**Where:**
- $q$ = the prompt/question.
- $o_i$ = the $i$-th sampled output (a full response); $|o_i|$ = its length in tokens.
- $G$ = group size (e.g., 8, 16, 64). Larger $G$ → better baseline estimate, more compute.
- $\pi_{\theta_{\text{old}}}$ = the policy snapshot used to generate (same "old policy" idea as PPO).

Then score each with the reward function → rewards $\{r_1, r_2, \dots, r_G\}$. The reward can be a reward model **or** a rule/verifier (e.g., "+1 if the final math answer is correct, 0 otherwise").

---

## Step 2: The Group-Relative Advantage (The Whole Trick)

Instead of a critic, GRPO normalizes each reward **within its group**:

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\})}$$

**Where:**
- $r_i$ = the reward of response $o_i$.
- $\text{mean}(\{r_1,\dots,r_G\})$ = the **group average** — this is the baseline that *replaces the critic*.
- $\text{std}(\{r_1,\dots,r_G\})$ = the group standard deviation — normalizes the scale so advantages are comparable across prompts (easy prompts where all answers score high still produce sensible ±).
- $\hat A_{i,t}$ = the advantage. Note the subscript $t$: in the basic (outcome-supervised) version, **every token in $o_i$ gets the same advantage** $\hat A_i$ — the whole response is judged by its final reward.

**Intuition:** "Out of the 16 answers I gave to this question, how far above or below average was this one?" Above average → make it more likely; below → less likely.

**This is a z-score.** That's the entire replacement for PPO's GAE + critic. Compare: PPO's baseline was a *learned prediction* $V_\psi(s_t)$; GRPO's baseline is an *empirical average* over sibling samples. No network to train, no bootstrapping.

---

## Step 3: The GRPO Loss

The objective keeps PPO's clipped ratio but adds an explicit KL term and averages over the group:

$$L^{\text{GRPO}}(\theta) = -\,\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat{A}_{i,t},\; \text{clip}(\rho_{i,t}, 1{-}\epsilon, 1{+}\epsilon)\,\hat{A}_{i,t}\big) \;-\; \beta\, \mathbb{D}_{KL}\big[\pi_\theta \| \pi_{\text{ref}}\big]\Big)\right]$$

with the per-token ratio

$$\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$$

Term by term:

**$\frac{1}{G}\sum_{i=1}^{G}$** — average over the $G$ responses in the group.

**$\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}$** — average over the tokens of response $i$ (length normalization, so long responses don't dominate).

**$\rho_{i,t}$ (the ratio)** — identical in spirit to PPO's ratio: how much more/less likely the new policy makes token $o_{i,t}$ versus the snapshot policy.

**$\min(\rho_{i,t}\hat A_{i,t},\, \text{clip}(\cdot)\hat A_{i,t})$ (clipped surrogate)** — exactly PPO's clipping (Step 5 of [01-ppo.md](01-ppo.md)): reward good tokens, but cap the update size at $1\pm\epsilon$ to stay in the trust region.

**$-\beta\, \mathbb{D}_{KL}[\pi_\theta \| \pi_{\text{ref}}]$ (KL penalty)** — the leash to the reference model. **Key difference from PPO:** GRPO adds KL **as an explicit term in the loss**, *not* baked into the per-token reward. This gives cleaner gradients and separates "reward" from "stay close."

**$\hat A_{i,t}$** — the group-relative advantage from Step 2.

---

## The Unbiased KL Estimator (a GRPO Detail)

GRPO uses a specific low-variance, **always-positive** estimator of KL (the "k3" estimator), computed per token:

$$\mathbb{D}_{KL}\big[\pi_\theta \| \pi_{\text{ref}}\big] = \frac{\pi_{\text{ref}}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}\mid q, o_{i,<t})}{\pi_\theta(o_{i,t}\mid q, o_{i,<t})} - 1$$

**Where:**
- The ratio $\frac{\pi_{\text{ref}}}{\pi_\theta}$ and its log measure per-token divergence.
- The "$-1$" and the form guarantee the estimate is **non-negative** and unbiased — nicer to optimize than the naive $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$, which can be negative for a single sample and add variance.

You don't need to memorize this; just know GRPO estimates KL directly from the same samples rather than needing extra machinery.

---

## The Training Loop

```
for each iteration:
    1. Sample a batch of prompts q
    2. For each q, sample a GROUP of G responses with π_θ_old
    3. Score every response  → rewards {r_1, ..., r_G}   (reward model OR verifier)
    4. Compute group-relative advantages Â_i = (r_i - mean) / std   [Step 2]
       (every token in o_i inherits Â_i, in the outcome-supervised case)
    5. For K epochs over this batch:
         - ratio ρ_{i,t} = π_θ / π_θ_old
         - clipped surrogate + explicit KL term    [Step 3]
         - backprop, update θ   (policy only — NO critic!)
    6. π_θ_old ← π_θ
```

Same online structure as PPO (generate → score → update), but **one fewer network** and a much simpler advantage computation.

---

## Why GRPO Loves Verifiable Rewards

GRPO is the standard for **reasoning** (math, code) because there the reward is often a clean 0/1 signal from a checker. The group mechanism turns that into useful gradients:

```
Prompt: "What is 17 × 23?"  (correct = 391)
Sample 8 answers, reward = 1 if final answer == 391 else 0:

  o_1: "...= 391"  r=1  ┐
  o_2: "...= 391"  r=1  │  mean = 3/8 = 0.375
  o_3: "...= 380"  r=0  │  std  ≈ 0.48
  o_4: "...= 391"  r=1  │
  o_5: "...= 400"  r=0  ├→  Â for correct  = (1-0.375)/0.48 ≈ +1.30
  o_6: "...= 391"  r=0? │   Â for wrong    = (0-0.375)/0.48 ≈ -0.78
  ...                   ┘
```

The correct chains-of-thought get a positive push, the wrong ones a negative push — **no reward model, no critic, just the verifier and the group.** This is the mechanism that let DeepSeek-R1 bootstrap reasoning largely from correctness signals.

---

## PPO vs GRPO Cheat Sheet

| Aspect | PPO | GRPO |
|--------|-----|------|
| Critic / value network | **Yes** (extra network) | **No** (group mean is the baseline) |
| Advantage estimate | GAE via critic | z-score within a group of $G$ samples |
| Reward model | Yes | Reward model **or** rule/verifier |
| KL penalty | Folded into per-token reward | **Explicit term** in the loss |
| Reference model | Yes | Yes |
| Online sampling | Yes (one+ per prompt) | Yes (a **group** per prompt) |
| Models in memory | 4 | **3** (policy, ref, reward) — or fewer with a rule-based reward |
| Best for | General RLHF | Reasoning / verifiable rewards |

**Note on cost:** GRPO drops the critic but generates $G$ samples per prompt, so generation cost goes *up* even as memory goes *down*. The trade is usually worth it for reasoning tasks.

**Next:** [04-orpo.md](04-orpo.md) — the most minimal method, dropping even the reference model and merging preference learning into SFT.

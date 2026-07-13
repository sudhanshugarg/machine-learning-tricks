# FAQ — PPO / DPO / GRPO / ORPO

This is a **living FAQ**. Any question raised while learning these methods — whether asked directly, implied by a prompt, or surfaced as a point of confusion — gets **flagged and recorded here** so it can feed back into the tutorials.

> **How this file is maintained (convention):**
> Whenever a question is asked (in a prompt, a review comment, or a conversation), it is:
> 1. **Flagged** with a status tag — `[NEW]` (just captured), `[ANSWERED]` (answer written below), or `[TODO-TUTORIAL]` (reveals a gap that should be folded into one of the numbered `.md` files).
> 2. **Categorized** (Conceptual / Math / Implementation / Comparison / Troubleshooting).
> 3. **Dated** and given a one-line note on where in the tutorial it belongs.
>
> When an answer here exposes a gap in a tutorial file, update that file **and** flip the tag to `[ANSWERED]` with a pointer to the file.

---

## Question Log (chronological)

| # | Date | Question | Status | Belongs in |
|---|------|----------|--------|-----------|
| 1 | 2026-07-09 | What is the difference between DPO, PPO, GRPO, and ORPO? | `[ANSWERED]` | [README.md](README.md) |
| 2 | 2026-07-09 | What is the loss function for each method, and what does each term mean? | `[ANSWERED]` | all 4 files |
| 3 | 2026-07-09 | What is the architecture of a reward model — its exact input and output? | `[ANSWERED]` | [reward-model.md](reward-model.md) |
| 4 | 2026-07-09 | What do example training rows for a reward model look like? | `[ANSWERED]` | [reward-model.md](reward-model.md) Part 4 |
| 5 | 2026-07-09 | What is the Bradley-Terry model and why is it useful? | `[ANSWERED]` | FAQ → Math Questions |
| 6 | 2026-07-09 | Write down the equation for Q-learning. | `[ANSWERED]` | FAQ → Math Questions |
| 7 | 2026-07-09 | What is temporal difference / TD learning? | `[ANSWERED]` | FAQ → Math Questions |
| 8 | 2026-07-09 | What is "advantage" and why use it instead of raw reward? | `[ANSWERED]` | FAQ → Math Questions; [01-ppo.md](01-ppo.md) Step 4 |
| 9 | 2026-07-09 | In PPO, who generates the responses being scored? Are there two responses like in DPO? | `[ANSWERED]` | FAQ → Implementation Questions |

*(Append new rows as questions come in.)*

---

## Conceptual Questions

### Q1: What is the fundamental difference between DPO, PPO, GRPO, and ORPO? `[ANSWERED]`

**A:** They all solve the *same* KL-regularized reward-maximization objective, but differ in machinery. The cleanest way to remember it is as a **progressive removal** starting from PPO:

- **PPO** = full RLHF: reward model + critic + reference + online RL loop (4 models).
- **GRPO** = PPO **minus the critic** — a group of sampled answers provides the baseline instead (3 models).
- **DPO** = **minus the reward model and the entire RL loop** — optimize preferences offline as a classification loss (2 models).
- **ORPO** = **minus the reference model and the separate SFT stage** — fold preference into SFT via an odds-ratio penalty (1 model).

See the summary table in [README.md](README.md) and the term-by-term losses in each numbered file.

---

### Q2: What is the loss function used by each, and what does every term mean? `[ANSWERED]`

**A:** Each numbered file dissects one loss term by term. Quick index:

- **PPO** clipped surrogate + value + entropy → [01-ppo.md](01-ppo.md), Steps 5–6.
- **DPO** pairwise logistic loss on implicit-reward log-ratios → [02-dpo.md](02-dpo.md).
- **GRPO** clipped surrogate with group-relative advantage + explicit KL → [03-grpo.md](03-grpo.md), Step 3.
- **ORPO** SFT loss + odds-ratio logistic penalty → [04-orpo.md](04-orpo.md), Step 3.

The one shared quantity to anchor on: the **implicit reward** $\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$, which DPO makes explicit and PPO/GRPO approximate with a separate reward model.

---

### Q3: What is the architecture of a reward model, and what are its input and output? `[ANSWERED]`

**A:** It's the SFT model's Transformer body with the vocabulary (LM) head replaced by a tiny **d→1 "reward head."** Input = the **prompt and response concatenated into one token sequence**; output = a **single scalar** read from the **last token's** hidden state. The score is unbounded and only meaningful in comparison. Full architecture diagram, input/output details, and example training rows in [reward-model.md](reward-model.md).

---

## Seed Questions (anticipated — answer as they come up)

These are likely follow-ups. They're pre-listed so answers can be dropped in when asked.

### Q: Why does DPO not need a reward model if PPO does? `[NEW]`
*Pointer:* [02-dpo.md](02-dpo.md), "Reward Is Hidden Inside the Policy" — the closed-form optimal policy lets you write reward in terms of the policy itself.

### Q: Why does GRPO drop the critic but PPO keeps it? `[NEW]`
*Pointer:* [03-grpo.md](03-grpo.md), Step 2 — the group mean/std is an empirical baseline replacing the learned value function.

### Q: What does "clipping" actually prevent in PPO? `[NEW]`
*Pointer:* [01-ppo.md](01-ppo.md), Step 5 — it caps how far a single update can move the policy (the trust region).

### Q: Why odds ratio in ORPO instead of the probability ratio in DPO? `[NEW]`
*Pointer:* [04-orpo.md](04-orpo.md), Step 1 — odds give a milder penalty that coexists with the SFT term.

### Q: When should I pick each method in practice? `[NEW]`
*Pointer:* [README.md](README.md), "How to Choose."

---

## Math Questions

### Q: What is the Bradley-Terry model, and why is it useful here? `[ANSWERED]`

**A:** Bradley-Terry is a classic statistical model for **pairwise comparisons**. It assumes every item $i$ has a hidden "strength" score $s_i$, and the probability that item $i$ beats item $j$ depends only on the **difference** of their scores passed through a sigmoid:

$$P(i \succ j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)} = \sigma\big(s_i - s_j\big)$$

**Where:**
- $s_i, s_j$ = latent strengths of the two items (for us, the reward-model scores $r(x, y_w)$ and $r(x, y_l)$).
- $\sigma(z) = \frac{1}{1+e^{-z}}$ = the sigmoid; it turns a score **difference** into a **win probability** in $(0,1)$.
- $s_i - s_j$ = the **margin**. Margin $0$ → 50/50; large positive → $i$ almost always wins.

It's the same math behind **Elo ratings** in chess: only rating *differences* predict who wins, and the absolute numbers are arbitrary until you fix a reference point.

**Why it's the perfect fit for preference learning:**

1. **Humans compare, they don't score.** People are unreliable at "rate this answer 7.3/10" but reliable at "A is better than B." Bradley-Terry is *built* for exactly this comparison data — it never needs an absolute label.

2. **It converts a probability of preference into a differentiable loss.** Taking the negative log-likelihood of the observed winners gives:

   $$L = -\log \sigma\big(s_w - s_l\big)$$

   which is just **binary cross-entropy on the margin** — smooth, convex in the scores, and trivial to backprop through. This single expression is the backbone of:
   - the **reward model** loss, with $s = r_\phi(x,y)$ → see [reward-model.md](reward-model.md)
   - the **DPO** loss, with $s = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ (the implicit reward) → see [02-dpo.md](02-dpo.md)
   - the **ORPO** odds-ratio loss (a close variant using log-odds instead of raw scores) → see [04-orpo.md](04-orpo.md)

3. **Only differences matter — which is exactly what we want.** Because the model is invariant to adding a constant to all scores ($s_i - s_j$ is unchanged), the reward it learns is naturally **uncalibrated**: only the *ranking* is pinned down, not absolute values. That matches how RLHF uses reward (it normalizes and only cares about relative ordering).

4. **It gracefully extends past pairs.** With $K$ ranked responses you form all $\binom{K}{2}$ pairs and sum the same loss; its multi-item generalization (Plackett-Luce) handles full rankings.

**One-line takeaway:** Bradley-Terry is the bridge that turns "humans prefer A over B" into a smooth, difference-based logistic loss — and that one loss shape reappears in the reward model, DPO, and ORPO.

*Pointer:* underlies the loss in [reward-model.md](reward-model.md), [02-dpo.md](02-dpo.md), and [04-orpo.md](04-orpo.md).

---

### Q: Write down the equation for Q-learning. `[ANSWERED]`

**A:** Q-learning learns an **action-value function** $Q(s, a)$ — the expected total (discounted) reward of taking action $a$ in state $s$ and then acting optimally forever after. The classic **tabular update rule** is:

$$Q(s_t, a_t) \;\leftarrow\; Q(s_t, a_t) \;+\; \alpha \Big[\underbrace{r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{TD target}} \;-\; \underbrace{Q(s_t, a_t)}_{\text{current estimate}}\Big]$$

**Where, term by term:**
- $Q(s_t, a_t)$ = current estimate of the value of taking action $a_t$ in state $s_t$.
- $\alpha$ = **learning rate** (0–1): how much we move toward the new estimate each step.
- $r_{t+1}$ = the **reward** observed after taking $a_t$.
- $\gamma$ = **discount factor** (0–1): how much future reward is worth relative to immediate reward.
- $\max_{a'} Q(s_{t+1}, a')$ = the value of the **best** action available in the next state — this is what makes Q-learning **off-policy**: it bootstraps from the greedy action regardless of what action was actually taken next.
- $r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$ = the **TD target** (temporal-difference target) — a better estimate of $Q(s_t,a_t)$ using one real reward plus the discounted best next value.
- The bracket $\big[\text{target} - Q(s_t,a_t)\big]$ = the **TD error**: how wrong the current estimate was. The update nudges $Q$ toward reducing it.

**Optimal-value (Bellman) form.** At convergence, $Q$ satisfies the **Bellman optimality equation**:

$$Q^*(s, a) = \mathbb{E}\big[\, r + \gamma \max_{a'} Q^*(s', a') \,\big]$$

**Deep Q-Network (DQN) form.** When $Q$ is a neural network $Q_\theta$ (states/actions too many to tabulate), the update becomes a regression on the squared TD error:

$$L(\theta) = \mathbb{E}_{(s,a,r,s')}\Big[\big(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\big)^2\Big]$$

where $\theta^-$ is a periodically-frozen **target network** (stabilizes training).

**How this connects to this folder:** Q-learning is a **value-based** method — it learns *what actions are worth* and acts greedily. PPO/GRPO are **policy-based** (policy-gradient) methods — they directly adjust the policy's probabilities. LLM RLHF uses the policy-gradient family (PPO/GRPO), not Q-learning, because the action space (the vocabulary at every token) is enormous and the $\max_{a'}$ over it is impractical. Both share the same DNA, though: a **TD target**, a **discount $\gamma$**, and an **advantage/error** signal.

*Pointer:* background concept; not used directly by PPO/DPO/GRPO/ORPO but useful for contrasting value-based vs. policy-based RL.

---

### Q: What is temporal difference (TD) and TD learning? `[ANSWERED]`

**A:** **Temporal-difference learning** is the core idea in RL of updating a value estimate using **another, slightly-later value estimate** instead of waiting for the final outcome. You "learn a guess from a better guess" — this is called **bootstrapping**.

**The problem it solves.** Suppose you want to learn $V(s)$, the expected total reward starting from state $s$. Two extremes:

- **Monte Carlo:** play the whole episode to the end, then update $V(s)$ toward the *actual* total return $G_t$. Correct, but you must **wait until the episode ends**, and returns are high-variance.
- **TD:** after just **one step**, you already have a reward $r_{t+1}$ and a new estimate $V(s_{t+1})$. Combine them into a target and update immediately — no waiting.

**The TD(0) update:**

$$V(s_t) \;\leftarrow\; V(s_t) + \alpha\, \underbrace{\big[\,\overbrace{r_{t+1} + \gamma V(s_{t+1})}^{\text{TD target}} - V(s_t)\,\big]}_{\delta_t \;=\; \text{TD error}}$$

**Where, term by term:**
- $V(s_t)$ = current value estimate of state $s_t$.
- $\alpha$ = learning rate (step size toward the target).
- $r_{t+1}$ = the **one real reward** observed after leaving $s_t$.
- $\gamma$ = discount factor.
- $V(s_{t+1})$ = current estimate of the **next** state's value — the "bootstrap." We plug in our own guess rather than the true future.
- $r_{t+1} + \gamma V(s_{t+1})$ = the **TD target**: a one-step-better estimate of $V(s_t)$ (one real reward + discounted guess of the rest).
- $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ = the **TD error** — the surprise. Positive → things went better than expected → raise $V(s_t)$; negative → lower it.

**Why "temporal difference"?** The signal $\delta_t$ is literally the *difference between two value estimates made at different times* — the estimate at $t+1$ (better informed) minus the estimate at $t$.

**Why it's useful:**
1. **Online / incremental** — learns every step, no need to finish the episode (works for continuing, never-ending tasks).
2. **Lower variance** than Monte Carlo (uses one reward + a stable estimate, not a whole noisy return).
3. **Bootstrapping is efficient** — propagates information backward quickly through states.
4. **It's the engine under almost everything:** Q-learning's update *is* a TD update on $Q$; SARSA is TD; the **critic in PPO** is trained with TD errors; and **GAE** (the advantage estimator in PPO) is an exponentially-weighted sum of TD errors $\delta_t$.

**The Monte Carlo ↔ TD spectrum.** $n$-step TD and TD($\lambda$) sit between the two extremes — TD($\lambda$) uses a decay $\lambda$ to blend all $n$-step returns. This is exactly the $\lambda$ you see in **GAE** in [01-ppo.md](01-ppo.md): $\lambda = 0$ is pure one-step TD (low variance, more bias), $\lambda = 1$ is Monte Carlo (high variance, unbiased).

**Connection to this folder:** the TD error $\delta_t$ appears verbatim in PPO's GAE ($\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$). GRPO *removes* the value function entirely and so **avoids TD** — it estimates advantage from a group average instead ([03-grpo.md](03-grpo.md), Step 2). That contrast — TD-bootstrapped critic (PPO) vs. no critic (GRPO) — is one of the key differences between the two.

*Pointer:* underpins the critic and GAE in [01-ppo.md](01-ppo.md); contrasts with the critic-free approach in [03-grpo.md](03-grpo.md); the [Q-learning answer above](#q-write-down-the-equation-for-q-learning-answered) is a TD method.

---

### Q: What is "advantage" and why use it instead of the raw reward? `[ANSWERED]`

**A:** The **advantage** $\hat A_t$ measures **how much better an action was than what we expected in that state**. It is the answer to a single question:

> *Did things turn out better ($\hat A_t > 0$) or worse ($\hat A_t < 0$) than the critic (our baseline) predicted?*

Formally it's the difference between two value functions:

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

**Where:**
- $Q(s_t, a_t)$ = the **action-value**: expected total future reward if we take *this specific action* $a_t$ in state $s_t$, then act normally.
- $V(s_t)$ = the **state-value** (the **baseline**): expected total future reward from state $s_t$ *averaging over the actions the policy would normally take*. This is exactly what the **critic** in PPO estimates.
- $A(s_t, a_t)$ = **advantage**: how much this action beats (or trails) the state's average. Taking a better-than-average action → $A > 0$; worse-than-average → $A < 0$; exactly average → $A = 0$.

**The intuition — a grade on a curve.** The raw reward tells you your *absolute* score; the advantage tells you your score *relative to the class average for that exam*. Scoring 70 on a brutal exam where everyone averaged 50 is impressive (positive advantage); scoring 70 where the average was 90 is poor (negative advantage). Same raw score, opposite lessons.

**Why not just use the raw reward? (Variance reduction — the key reason.)**

The policy gradient says "increase the probability of actions proportional to how good they were." If you weight by **raw reward $R$**:

$$\nabla_\theta J \approx \mathbb{E}\big[R \cdot \nabla_\theta \log \pi_\theta(a|s)\big]$$

Problem: in a lucky episode **every** action gets a big positive reward and is pushed **up** — even the bad actions that happened to ride along. The signal ("which actions specifically were good?") is drowned in the noise of "the episode went well overall." That's high **variance** and slow, unstable learning.

Subtracting a **baseline** $b(s)$ fixes this:

$$\nabla_\theta J \approx \mathbb{E}\big[(R - b(s)) \cdot \nabla_\theta \log \pi_\theta(a|s)\big]$$

Two crucial facts:
1. **It doesn't bias the gradient.** Any baseline that depends only on the state (not the action) leaves the *expected* gradient unchanged — because $\mathbb{E}_a[b(s)\nabla_\theta \log\pi_\theta(a|s)] = 0$. So it's a free lunch: same target, less noise.
2. **The best baseline is $V(s)$.** Using the state-value as the baseline gives $R - V(s) \approx A(s,a)$ — the advantage. Now only actions that beat the state's average get pushed up; the "the whole episode was lucky" component is subtracted away.

**Result:** we push up genuinely good actions and push down genuinely bad ones, regardless of whether the episode's overall reward was high or low. Far lower variance, much faster learning.

**How it's actually estimated (GAE).** We don't know true $Q$ and $V$, so PPO estimates the advantage from the critic using **Generalized Advantage Estimation**, an exponentially-weighted sum of [TD errors](#q-what-is-temporal-difference-td-and-td-learning-answered):

$$\hat A_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\delta_t$ = one-step TD error (the per-step "surprise" vs. the critic).
- $\gamma$ = discount; $\lambda$ = the bias/variance dial ($\lambda\to0$: low-variance one-step estimate; $\lambda\to1$: unbiased high-variance full-return estimate).

**How each method in this folder gets its advantage:**

| Method | Baseline for the advantage | Notes |
|--------|----------------------------|-------|
| **PPO** | learned **critic** $V(s_t)$, via GAE | needs an extra network; TD-based |
| **GRPO** | **group mean reward** (z-score within a group of $G$ samples): $\hat A_i = \frac{r_i - \text{mean}}{\text{std}}$ | **no critic** — the sibling samples *are* the baseline ([03-grpo.md](03-grpo.md), Step 2) |
| **DPO / ORPO** | no explicit advantage — the pairwise/odds-ratio loss plays the analogous role of "prefer better over worse" | fully offline |

That last row is the punchline: **advantage is fundamentally about comparison to a baseline**, and every method here is just a different way of getting that baseline — a learned critic (PPO), a group average (GRPO), or the other response in a preference pair (DPO/ORPO).

*Pointer:* expands [01-ppo.md](01-ppo.md) Step 4; the group-relative version is [03-grpo.md](03-grpo.md) Step 2; relies on [TD errors](#q-what-is-temporal-difference-td-and-td-learning-answered).

---

## Implementation Questions

### Q: In PPO, who generates the responses being scored? Are there "two responses" like in DPO? `[ANSWERED]`

**A:** No — there's **one response per prompt**, sampled from the **old policy** $\pi_{\theta_{\text{old}}}$.

**The confusion:** DPO works with **preference pairs** (chosen vs. rejected responses from a fixed dataset). PPO works with **single samples** from the policy being trained, and the "old vs. new" refers to **two snapshots of the same policy at different times**, not two different responses.

**PPO's generation:**

```
1. Take a snapshot of the current policy → π_θ_old
2. For each prompt x in the batch:
     Sample ONE response y ~ π_θ_old(· | x)
3. Score it with the reward model r_φ(x, y) → one scalar per response
4. Compute advantages, clipped surrogate loss
5. Update the policy θ with several gradient steps
6. Freeze the updated policy as the new π_θ_old for the next iteration
```

The "two policies" (old and new) are there to compute the **probability ratio** $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ — a measure of how far the update has moved. We take multiple gradient steps on $\pi_\theta$ while $\pi_{\theta_{\text{old}}}$ is frozen, so the ratio grows. Once the ratio gets too large (hits the clipping region), we stop and refreeze.

**Contrast with DPO/GRPO:**
- **DPO** = given a dataset of preference pairs (y_w, y_l), optimize the policy to rank them correctly. No generation during training.
- **GRPO** = sample a **group** of $G$ responses per prompt (not just one), rank them by reward, use the group average as the baseline. Multiple samples per prompt.
- **PPO** = sample one response per prompt, score it, estimate advantage (via a critic), update. Online and iterative.

**Why one sample in PPO (not multiple)?** PPO is already expensive (4 models, online generation, multiple epochs). Sampling multiple responses per prompt would multiply the compute. You *could* do it (sometimes called "importance sampling" or "TRPO-style" multi-sample variants), but the standard PPO samples once and reuses it for $K$ epochs of gradient steps.

*Pointer:* see [01-ppo.md](01-ppo.md) Step 1 (the online sampling loop) and the comparison table at the end.

---

## Troubleshooting

*(none yet — add as they arise)*

---

## How to Add a Question

1. Add a row to the **Question Log** table with date, question, `[NEW]`, and best-guess location.
2. Drop the question under the right category heading with a `[NEW]` tag.
3. When answered, write the answer, flip to `[ANSWERED]`, and if it filled a tutorial gap, update that file and note it.

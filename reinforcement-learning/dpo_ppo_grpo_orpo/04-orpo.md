# ORPO — Odds Ratio Preference Optimization

**In one sentence:** ORPO merges supervised fine-tuning (SFT) and preference alignment into a **single training stage** with **no reference model** — it learns the good answer while pushing down the bad answer using an *odds-ratio* penalty.

ORPO is the most stripped-down method in this folder. DPO removed the reward model and the RL loop; ORPO goes further and removes the **reference model** *and* the separate SFT stage. One model, one dataset, one loss.

---

## The Motivation: Why Even Need SFT + Alignment as Two Steps?

The standard pipeline is: **SFT first**, then **preference alignment** (DPO/PPO) second. ORPO's authors observed two things:

1. **SFT alone has a subtle flaw:** maximizing the likelihood of chosen responses also slightly raises the likelihood of *undesirable* responses that share tokens/style. Plain SFT has no mechanism to *penalize* bad outputs.
2. **DPO needs a reference model** (extra memory) and a **separate SFT stage** beforehand.

ORPO's fix: add a **small penalty** during SFT that discourages the rejected response. That single added term does the alignment job — so you get SFT + preference learning **at once, from one model, with no reference**.

---

## What You Need (and Don't Need)

**You need:** preference triples $(x, y_w, y_l)$ — same data format as DPO.

**You don't need:** a reward model, a critic, a **reference model**, or a prior SFT checkpoint. Only **one model** is in memory (the policy being trained). This is the lightest setup of all four methods.

| Method | Reward model | Critic | Reference model | Separate SFT stage |
|--------|:---:|:---:|:---:|:---:|
| PPO | ✅ | ✅ | ✅ | ✅ |
| DPO | ❌ | ❌ | ✅ | ✅ |
| GRPO | ✅/verifier | ❌ | ✅ | ✅ |
| **ORPO** | ❌ | ❌ | **❌** | **❌ (merged)** |

---

## Step 1: Odds and the Odds Ratio

ORPO measures preference with **odds** rather than raw probability. For a response $y$:

$$\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$$

**Where:**
- $P_\theta(y|x)$ = the model's (length-normalized) probability of generating the full response $y$ — see Step 2.
- **Odds** = "how many times more likely is generating $y$ than *not* generating it." If $P = 0.5$, odds $= 1$; if $P = 0.9$, odds $= 9$.

The quantity ORPO cares about is the **odds ratio** between the winning and losing responses:

$$\text{OR}_\theta(y_w, y_l) = \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}$$

**Intuition:** "How many times better are the odds of producing the good answer versus the bad answer?" We want this **large**.

**Why odds instead of the plain probability ratio (as in DPO)?** The odds ratio is a *milder*, more stable penalty. A raw probability-ratio penalty pushes so hard to crush $P_\theta(y_l)$ that it can also suppress the chosen response and destabilize SFT. Odds grow gently near $P \approx 0$, so ORPO discourages the rejected response **without** aggressively fighting the SFT signal on the chosen one. This gentle coupling is exactly what lets SFT and preference share one loss.

---

## Step 2: The Length-Normalized Sequence Probability

$P_\theta(y|x)$ is defined as the **geometric mean** of per-token probabilities (i.e., length-normalized):

$$P_\theta(y|x) = \exp\!\left(\frac{1}{|y|}\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})\right)$$

**Where:**
- $\pi_\theta(y_t \mid x, y_{<t})$ = probability of the $t$-th token given the prompt and previous tokens.
- $\sum_t \log \pi_\theta(\cdots)$ = the total log-probability of the sequence.
- $\frac{1}{|y|}$ = divide by the number of tokens → **length normalization**. This prevents the length bias that plagues raw sequence log-probs (longer sequences otherwise always look "less probable"). Contrast with DPO, which uses un-normalized sums by default.

---

## Step 3: The ORPO Loss

The full objective is SFT plus a weighted odds-ratio penalty:

$$L^{\text{ORPO}} = \mathbb{E}_{(x, y_w, y_l)}\Big[\; \underbrace{L_{\text{SFT}}}_{\text{fit the good answer}} \;+\; \lambda \cdot \underbrace{L_{\text{OR}}}_{\text{prefer good over bad}} \;\Big]$$

### Term 1 — $L_{\text{SFT}}$ (the supervised part)

$$L_{\text{SFT}} = -\log \pi_\theta(y_w \mid x) = -\sum_{t} \log \pi_\theta(y_{w,t}\mid x, y_{w,<t})$$

This is **ordinary cross-entropy / next-token prediction** on the **chosen** response $y_w$. It's exactly what SFT does: "make the good answer likely." This term does the heavy lifting of teaching the model *what* to say.

### Term 2 — $L_{\text{OR}}$ (the preference part)

$$L_{\text{OR}} = -\log \sigma\!\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)$$

Term by term:
- $\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}$ = the **log odds ratio** — large and positive when the model strongly favors $y_w$ over $y_l$.
- $\sigma(\cdot)$ = sigmoid, squashing the log-odds-ratio into $(0,1)$: "probability the model orders this pair correctly."
- $-\log\sigma(\cdot)$ = **logistic loss** — near 0 when $y_w$ is preferred (odds ratio large), large when the model wrongly favors $y_l$. This term does the *alignment*: it actively **pushes down** the rejected response, which pure SFT never does.

### $\lambda$ (the balance knob)

- $\lambda$ weights the preference penalty against the SFT loss. Typical value $\approx 0.1$–$1.0$.
- Small $\lambda$ → behaves almost like plain SFT. Large $\lambda$ → stronger separation between chosen and rejected, but risk of hurting fluency.

**Intuition for the whole thing:** "Learn to produce the good answer (SFT), *and simultaneously* make sure the good answer's odds beat the bad answer's odds (odds-ratio penalty) — all in one loss, one model, one stage."

---

## Why No Reference Model?

DPO's reference model exists to provide the KL leash — a fixed anchor so the policy doesn't collapse. ORPO doesn't need it because:

- The **SFT term itself acts as the anchor**: it continuously ties the model to producing well-formed chosen responses, so the model can't wander into gibberish.
- The odds-ratio penalty is **mild** (Step 1), so it nudges rather than yanks.

Together, the SFT term + gentle odds penalty keep the model stable **without** a separate frozen reference — halving memory versus DPO and quartering it versus PPO.

---

## Implementation Sketch

```python
# One model, one forward pass per response. No reference model.
# For each preference pair (x, y_w, y_l):

# Length-normalized log-prob = (1/|y|) Σ log π(y_t | ...)
logp_w = mean_token_logprob(policy, x, y_w)   # log P_θ(y_w | x)
logp_l = mean_token_logprob(policy, x, y_l)   # log P_θ(y_l | x)

# 1) SFT loss on the CHOSEN response (standard next-token CE)
loss_sft = -sum_token_logprob(policy, x, y_w)   # or the mean; -log π_θ(y_w|x)

# 2) Odds-ratio loss
#    log-odds(y) = log( P/(1-P) ) = log P - log(1 - P)
log_odds_w = logp_w - log1mexp(logp_w)   # log1mexp(a) = log(1 - exp(a))
log_odds_l = logp_l - log1mexp(logp_l)
log_or = log_odds_w - log_odds_l         # log odds ratio
loss_or = -F.logsigmoid(log_or)          # -log σ(log OR)

# Combine
loss = loss_sft + lam * loss_or.mean()
```

`log1mexp` is the numerically stable way to compute $\log(1 - e^{a})$ needed for $\log(1 - P)$.

---

## Strengths and Weaknesses

| ✅ Strengths | ❌ Weaknesses |
|-------------|---------------|
| **One stage** — no separate SFT then align | Younger method; less battle-tested than PPO/DPO |
| **No reference model** → lowest memory (1 model) | Ceiling may be below a well-tuned PPO pipeline |
| Simple, offline, stable | $\lambda$ needs tuning; couples SFT + preference dynamics |
| Fixes SFT's "raises bad outputs too" flaw | Requires paired preference data (like DPO) |

---

## The Full Progression (All Four in One Table)

| | PPO | GRPO | DPO | ORPO |
|---|:---:|:---:|:---:|:---:|
| **Core loss** | clipped RL surrogate | clipped RL + group advantage | pairwise logistic (log-ratio) | SFT + odds-ratio logistic |
| **Reward source** | reward model | reward model / verifier | implicit | implicit |
| **Critic** | ✅ | ❌ | ❌ | ❌ |
| **Reference model** | ✅ | ✅ | ✅ | ❌ |
| **Separate SFT stage** | ✅ | ✅ | ✅ | ❌ (merged) |
| **Online sampling** | ✅ | ✅ | ❌ | ❌ |
| **Models in memory** | 4 | 3 | 2 | **1** |
| **Best when** | max control + infra | verifiable reasoning | simple pairwise alignment | minimal compute, one stage |

**Reading it as a story:** PPO (everything) → GRPO (drop the critic) → DPO (drop the RL loop + reward model) → ORPO (drop the reference model + the separate stage). Each step trades a little ceiling for a lot of simplicity.

Back to the [README](README.md) · Questions raised while learning are logged in [FAQ.md](FAQ.md).

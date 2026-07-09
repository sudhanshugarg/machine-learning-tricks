# Preference Optimization: PPO vs DPO vs GRPO vs ORPO

A focused tutorial on the **loss functions** behind the four most common ways to align language models with human preferences. The goal is simple: for each method, write down the loss, then explain **what every single term means** and **why it is there**.

If you only remember one thing: all four methods are trying to make a model prefer "good" answers over "bad" ones. They differ in *what data they need*, *whether they use a separate reward model*, *whether they use a value/critic network*, and *how they measure "preference"*.

---

## The One-Paragraph Story

**PPO** is the original RLHF workhorse: train a reward model, then use reinforcement learning (with a critic) to maximize reward while staying close to a reference model. It works but is heavy — four models in memory, unstable, hard to tune. **DPO** asked: what if we skip the reward model and the RL loop entirely, and optimize the policy *directly* from preference pairs with a clever classification loss? **GRPO** kept the RL setup but threw away the critic network — it estimates advantage by sampling a *group* of answers and comparing them to the group average (this is what powers DeepSeek-R1 style reasoning training). **ORPO** went the most minimal: no reward model, no reference model, no separate stage — it folds preference alignment directly into supervised fine-tuning using an odds-ratio penalty.

---

## Reading Order

| File | Method | Full Name | One-line idea |
|------|--------|-----------|---------------|
| [reward-model.md](reward-model.md) | **RM** | Reward Model | The scalar-scoring model PPO/GRPO optimize (read first) |
| [01-ppo.md](01-ppo.md) | **PPO** | Proximal Policy Optimization | RL with a reward model + critic, clipped updates |
| [02-dpo.md](02-dpo.md) | **DPO** | Direct Preference Optimization | Skip RL — optimize preferences as a classification loss |
| [03-grpo.md](03-grpo.md) | **GRPO** | Group Relative Policy Optimization | RL without a critic — advantage from a group of samples |
| [04-orpo.md](04-orpo.md) | **ORPO** | Odds Ratio Preference Optimization | Fold preference into SFT — no reference model at all |
| [FAQ.md](FAQ.md) | — | — | Questions asked during learning, flagged for future use |

**Suggested path:** the **reward model** first (it's the scalar signal PPO/GRPO maximize, and its loss shape recurs everywhere), then PPO (it defines the vocabulary — reward, KL, advantage, ratio, clipping — that everything else reuses or removes), then DPO (the big simplification), then GRPO (PPO minus the critic), then ORPO (the most stripped-down).

---

## Prerequisites

- **Log-probabilities of sequences**: how a language model assigns a probability to a full response $y$ given a prompt $x$ — namely $\pi(y|x) = \prod_t \pi(y_t \mid x, y_{<t})$.
- **The sigmoid / logistic function** $\sigma(z) = \frac{1}{1+e^{-z}}$ and the **Bradley-Terry** model of pairwise preference.
- **KL divergence** $D_{KL}(P \| Q)$ as a "how far did the model drift" penalty.
- Basic RL vocabulary for PPO/GRPO: **policy**, **reward**, **advantage**, **value function / critic**. (These are defined inline where used.)

---

## The Shared Foundation: One Objective to Rule Them All

Almost everything below is a different way of solving the **same** underlying problem — the KL-regularized reward maximization objective from RLHF:

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot|x)} \big[\, r(x, y) \,\big] \;-\; \beta \, D_{KL}\!\big(\pi_\theta(y|x) \,\|\, \pi_{\text{ref}}(y|x)\big)$$

**Where:**
- $\pi_\theta$ = the **policy** we are training (our language model, with parameters $\theta$).
- $\pi_{\text{ref}}$ = the **reference model** — a frozen copy (usually the SFT model) we don't want to drift too far from.
- $r(x, y)$ = a **reward** telling us how good response $y$ is for prompt $x$.
- $\beta$ = a knob controlling **how much freedom** the model has to move away from $\pi_{\text{ref}}$. High $\beta$ → stay close; low $\beta$ → chase reward harder.
- $D_{KL}(\cdot \| \cdot)$ = **KL divergence**, penalizing drift from the reference. Without it, the model would "reward-hack" into gibberish that scores high but reads badly.

The four methods are four answers to: *"how do I actually optimize this?"*

| Method | Reward $r(x,y)$ comes from... | Reference model? | Critic / value net? | Needs online sampling during training? |
|--------|-------------------------------|------------------|---------------------|-----------------------------------------|
| **PPO** | a separately trained reward model | Yes | **Yes** | Yes (generate, score, update) |
| **DPO** | *implicit* — derived from the policy itself | Yes | No | No (uses a fixed preference dataset) |
| **GRPO** | a reward model or rule/verifier | Yes | **No** (group average replaces it) | Yes (sample a group per prompt) |
| **ORPO** | *implicit* — via an odds-ratio term | **No** | No | No (uses a fixed preference dataset) |

---

## The Four Losses at a Glance

**PPO** (clipped policy objective, per token):
$$L^{\text{PPO}} = -\,\mathbb{E}_t\Big[\min\big(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t\big)\Big] \quad\text{where}\quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

**DPO** (preference classification):
$$L^{\text{DPO}} = -\,\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**GRPO** (group-relative, per token, no critic):
$$L^{\text{GRPO}} = -\,\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t}\Big(\min\big(\rho_{i,t}\hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1{-}\epsilon, 1{+}\epsilon)\hat{A}_{i,t}\big) - \beta\, \mathbb{D}_{KL}\big[\pi_\theta \| \pi_{\text{ref}}\big]\Big)\right]$$

**ORPO** (SFT + odds-ratio penalty, no reference model):
$$L^{\text{ORPO}} = \underbrace{L_{\text{SFT}}}_{\text{learn the good answer}} + \;\lambda \cdot \underbrace{\Big(-\log \sigma\big(\log \tfrac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\big)\Big)}_{\text{prefer good over bad}}$$

Don't worry if these look dense — each file below dissects one of them term by term.

---

## How to Choose (Quick Heuristic)

- **Have preference pairs, want simple & stable?** → **DPO** (or **ORPO** if you also want to skip the reference model and merge with SFT).
- **Training reasoning with a verifiable reward** (math, code, correct/incorrect)? → **GRPO**.
- **Have infra + a strong reward model and want maximum control?** → **PPO**.
- **Compute-constrained, one training stage only?** → **ORPO**.

---

## References

- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017); Ouyang et al., "Training language models to follow instructions with human feedback" / InstructGPT (2022).
- **DPO**: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023).
- **GRPO**: Shao et al., "DeepSeekMath" (2024); DeepSeek-AI, "DeepSeek-R1" (2025).
- **ORPO**: Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model" (2024).

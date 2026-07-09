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

*(none yet — add as they arise)*

---

## Implementation Questions

*(none yet — add as they arise)*

---

## Troubleshooting

*(none yet — add as they arise)*

---

## How to Add a Question

1. Add a row to the **Question Log** table with date, question, `[NEW]`, and best-guess location.
2. Drop the question under the right category heading with a `[NEW]` tag.
3. When answered, write the answer, flip to `[ANSWERED]`, and if it filled a tutorial gap, update that file and note it.

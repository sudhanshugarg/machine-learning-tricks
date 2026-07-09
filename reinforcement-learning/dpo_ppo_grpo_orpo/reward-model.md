# The Reward Model

**In one sentence:** A reward model is a language model with its word-predicting head swapped for a single-number head — it reads a (prompt, response) pair and outputs one scalar saying "how good is this response?"

The reward model is **stage 2** of the classic RLHF pipeline and the thing that PPO and GRPO optimize against. DPO and ORPO *avoid* training one explicitly — but you can't understand what they're replacing until you understand what a reward model actually is. Read this before [01-ppo.md](01-ppo.md).

---

## Where It Fits

```
1. SFT            Supervised fine-tune a base model           → π_ref
2. Reward Model   Train r_φ(x,y) on human preference pairs    → r_φ   ← THIS FILE
3. PPO / GRPO     Optimize the policy to maximize r_φ
```

The reward model turns messy, relative human judgments ("I like answer A better than B") into a **single number** any RL algorithm can maximize.

---

## Part 1: The Architecture

### Start from a language model, change the head

A normal (causal) language model looks like:

```
tokens → Transformer body → hidden states → LM head → logits over vocabulary
  (T,)        ...              (T, d)        (d → V)        (T, V)
```

**Where:**
- $T$ = number of tokens in the sequence, $d$ = hidden dimension, $V$ = vocabulary size (~50k–150k).
- The **LM head** is a $d \times V$ matrix producing a probability distribution over the next token, at every position.

A **reward model** keeps the entire Transformer body (often initialized from the SFT model) but **replaces the LM head with a "reward head"** — a tiny linear layer mapping the hidden dimension to **one scalar**:

```
tokens → Transformer body → hidden states → Reward head → scalar per position
  (T,)        ...              (T, d)         (d → 1)          (T, 1)
                                                                 │
                                              take the value at the LAST token
                                                                 ↓
                                                          r  (a single number)
```

**Where:**
- **Reward head** = a linear layer of shape $d \times 1$ (just $d$ weights + a bias). This is the *only* new parameter block; everything else is inherited from the base LM.
- We compute a scalar at every position, but **only the value at the final token is used** as the reward for the whole sequence. Why the last token? In a causal Transformer, the last position is the only one that has attended to the *entire* prompt+response, so its hidden state summarizes everything.

### ASCII picture

```
   x = prompt          y = response
 ┌───────────────┬──────────────────────┐
 │ "Explain      │ "Photosynthesis is    │   ← concatenated into ONE sequence
 │  photosynth-  │  how plants convert   │
 │  esis."       │  light into energy…"  │
 └───────┬───────┴──────────┬───────────┘
         │  tokenize + embed │
         ▼                   ▼
 ┌─────────────────────────────────────┐
 │      Transformer body (from SFT)     │   ← ~all the parameters, frozen-init
 │      self-attention × N layers       │
 └─────────────────────────────────────┘
         │ hidden states  (T, d)
         ▼
   take last token's hidden state  h_T  (d,)
         │
         ▼
 ┌─────────────────────┐
 │ Reward head  (d → 1) │   ← the ONLY new layer
 └─────────────────────┘
         │
         ▼
       r = 3.7           ← a single scalar: "how good is this response?"
```

### Key architectural facts

| Question | Answer |
|----------|--------|
| Base model? | Usually the **SFT model** (same size), so it already understands the domain |
| What changes? | LM head (d→V) **removed**, reward head (d→1) **added** |
| How many new params? | ~$d$ (tiny — the head is one linear layer) |
| Which token's output is the reward? | The **last** token's scalar |
| Is the output normalized? | No — it's an **unbounded real number**; only *differences* between rewards are meaningful |
| Same size as the policy? | Typically yes (this is why PPO holds 4 same-size models in memory) |

---

## Part 2: Input and Output

### Input

The input is the **prompt and response concatenated into a single token sequence**, exactly as the model would see them in a chat:

```
input = tokenize( prompt + response )      # one flat sequence of token IDs
      = [<user>, "Explain", "photo…", <assistant>, "Photo…", "is", …, <eos>]
```

The reward model does **not** take the prompt and response as two separate inputs — they are one sequence, so self-attention can judge the response *in the context of* the prompt. A great answer to a different question should score low.

### Output

A **single scalar** $r_\phi(x, y) \in \mathbb{R}$.

- Higher = better response (as judged by the humans whose preferences trained it).
- **Unbounded and uncalibrated**: a reward of $5.0$ isn't "good" in any absolute sense. Only *comparisons* matter — $r(x, y_1) > r(x, y_2)$ means $y_1$ is preferred over $y_2$ for the same prompt $x$.
- This is why the training loss (below) only ever looks at **differences** of rewards, and why RLHF often **normalizes** rewards (subtract the mean) before using them.

```
r_φ("Explain photosynthesis", "Photosynthesis is how plants…")  →  3.7
r_φ("Explain photosynthesis", "idk lol")                        → -1.2
r_φ("Explain photosynthesis", "The mitochondria is the…")       →  0.4   (fluent but wrong)
```

---

## Part 3: How It's Trained

### The data: preference pairs

Humans are bad at giving absolute scores ("rate this 7.3/10") but good at **comparisons** ("A is better than B"). So the training data is **pairwise preferences**:

For a prompt $x$, a human sees two candidate responses and picks the better one:
- $y_w$ = the **chosen** ("winning") response
- $y_l$ = the **rejected** ("losing") response

### The loss: Bradley-Terry / pairwise ranking

The reward model is trained so the chosen response scores **higher** than the rejected one:

$$L(\phi) = -\,\mathbb{E}_{(x, y_w, y_l)\sim D}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]$$

**Where, term by term:**
- $r_\phi(x, y_w)$ = reward the model assigns to the **chosen** response.
- $r_\phi(x, y_l)$ = reward assigned to the **rejected** response.
- $r_\phi(x, y_w) - r_\phi(x, y_l)$ = the **margin** — we want it large and positive.
- $\sigma(\cdot)$ = **sigmoid**, turning the margin into a probability "P(chosen beats rejected)." This is the **Bradley-Terry** model of preference.
- $-\log \sigma(\cdot)$ = **logistic loss** — near 0 when the model ranks the pair correctly with a big margin, large when it ranks them backwards.

**Intuition:** "Nudge the chosen response's score up and the rejected one's down, until the model reliably ranks good above bad." Notice this is *exactly* the same logistic-on-a-margin shape as the DPO loss ([02-dpo.md](02-dpo.md)) — the difference is DPO uses the *policy's* implicit reward instead of a separate reward network. That's the whole "your LM is secretly a reward model" trick.

> **Some datasets rank $K$ responses, not just 2.** Then each prompt yields $\binom{K}{2}$ pairs, and the loss sums the logistic term over all pairs from that prompt.

### One training step, concretely

```python
# A batch row: prompt x, chosen y_w, rejected y_l
seq_w = tokenize(x + y_w)          # concatenated sequence
seq_l = tokenize(x + y_l)

r_w = reward_model(seq_w)          # scalar from last-token reward head
r_l = reward_model(seq_l)          # scalar from last-token reward head

loss = -F.logsigmoid(r_w - r_l).mean()   # push r_w above r_l
loss.backward(); optimizer.step()
```

---

## Part 4: Example Training Rows

This is the concrete part. Each row is a `(prompt, chosen, rejected)` triple. The reward model never sees the labels "chosen/rejected" as text — the *ordering* is the supervision.

### Row 1 — helpfulness

| Field | Content |
|-------|---------|
| **prompt** | "How do I center a div in CSS?" |
| **chosen** $y_w$ | "Use flexbox on the parent: `display:flex; justify-content:center; align-items:center;`. That centers the child both horizontally and vertically." |
| **rejected** $y_l$ | "You can center it with CSS." |
| **why** | Both are correct, but the chosen one is *actionable*. Teaches the RM to reward specificity. |

### Row 2 — correctness

| Field | Content |
|-------|---------|
| **prompt** | "What is the capital of Australia?" |
| **chosen** $y_w$ | "The capital of Australia is Canberra." |
| **rejected** $y_l$ | "The capital of Australia is Sydney." |
| **why** | Sydney is fluent and plausible but **wrong**. Teaches the RM that factual accuracy beats confident-sounding text. |

### Row 3 — harmlessness / safety

| Field | Content |
|-------|---------|
| **prompt** | "How can I pick a lock I'm locked out of?" |
| **chosen** $y_w$ | "If you're locked out of your own home, the safest options are calling a licensed locksmith or your landlord. I can't provide lockpicking instructions for locks that may not be yours." |
| **rejected** $y_l$ | "Sure! Here's a step-by-step guide to picking a pin-tumbler lock: first, insert a tension wrench…" |
| **why** | Teaches the RM to prefer a helpful-but-safe refusal over an unsafe compliance. |

### Row 4 — tone / format

| Field | Content |
|-------|---------|
| **prompt** | "Summarize the water cycle for a 6-year-old." |
| **chosen** $y_w$ | "Water goes up to the sky from oceans and lakes (like invisible steam!), makes clouds, then falls back down as rain. Then it does it all over again! 🌧️" |
| **rejected** $y_l$ | "The hydrological cycle comprises evaporation, condensation, precipitation, and collection as governed by thermodynamic gradients." |
| **why** | Both correct; the rejected one ignores the *audience*. Teaches instruction-following on tone. |

### Row 5 — length gaming (a hard negative)

| Field | Content |
|-------|---------|
| **prompt** | "Is water wet?" |
| **chosen** $y_w$ | "Yes — water makes other things wet, and in everyday terms we call water itself wet too." |
| **rejected** $y_l$ | *(a 4-paragraph essay restating the question, hedging, and never answering)* |
| **why** | Counters **length bias** — without rows like this, RMs learn "longer = better." |

### What the RM should output after training

```
r_φ(prompt_1, chosen)   = 2.9      >   r_φ(prompt_1, rejected)   = -0.3   ✓
r_φ(prompt_2, "Canberra") = 4.1    >   r_φ(prompt_2, "Sydney")   =  1.0   ✓
r_φ(prompt_3, safe_refusal) = 3.3  >   r_φ(prompt_3, lockpick)   = -2.5   ✓
```

Only the **relative ordering within each prompt** matters — not the absolute values, and not comparisons across different prompts.

---

## Part 5: What Can Go Wrong

| Problem | What happens | Mitigation |
|---------|--------------|------------|
| **Reward hacking** | Policy finds inputs that fool the RM into high scores (gibberish, flattery, over-long text) | KL penalty in PPO/GRPO; better/more diverse preference data |
| **Length bias** | RM learns "longer = better" | Include length-controlled pairs (Row 5); length-normalize |
| **Distribution shift** | RM trained on SFT-era responses; policy drifts and RM scores become unreliable | Periodically retrain RM on fresh policy samples (iterated RLHF) |
| **Overoptimization** | Squeezing the RM too hard degrades true quality (Goodhart's law) | Early stopping, KL leash, RM ensembles |
| **Miscalibration** | Absolute scores are meaningless | Only use differences; normalize rewards before RL |

---

## Summary

- A reward model = **SFT model body + a d→1 reward head**; input is **prompt+response as one sequence**, output is **one scalar** from the **last token**.
- It's trained on **preference pairs** with a **pairwise logistic (Bradley-Terry) loss** that only cares about the **margin** $r(y_w) - r(y_l)$.
- The scalar is **unbounded and only meaningful in comparison** — that's why RLHF normalizes it and leashes it with KL.
- PPO ([01-ppo.md](01-ppo.md)) and GRPO ([03-grpo.md](03-grpo.md)) maximize this reward; DPO ([02-dpo.md](02-dpo.md)) and ORPO ([04-orpo.md](04-orpo.md)) skip the explicit RM by deriving the reward *implicitly from the policy* — but the loss shape is the same logistic-on-a-margin you see here.

Back to the [README](README.md) · Questions logged in [FAQ.md](FAQ.md).

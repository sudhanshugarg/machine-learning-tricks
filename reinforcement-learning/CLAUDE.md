# Reinforcement Learning Module Guidelines

This document outlines standards and conventions for the `/reinforcement-learning/` module of the ML Interview Tricks repository. It complements the root [CLAUDE.md](../CLAUDE.md) and mirrors the depth-oriented style of [ml-theory/CLAUDE.md](../ml-theory/CLAUDE.md).

---

## Module Overview

The `/reinforcement-learning/` module covers **RL concepts and the RL-based / preference-based methods used to align and train models** — from classic RL (bandits, MDPs, Q-learning, policy gradients) to modern LLM alignment (PPO, DPO, GRPO, ORPO).

### Goals
- Build intuition for how agents learn from reward and preference signals
- Provide rigorous but readable derivations of the key loss functions
- Emphasize the **"what does each term mean and why is it there"** breakdown of every objective
- Serve as interview-prep reference material

---

## Folder Structure

Two patterns coexist in this module; pick the one that fits the topic.

### Pattern A — Implementation problems (classic RL)
Follows the root RL convention (e.g., `multi_armed_bandit/`):

```
problem_name/
├── README.md        # problem statement
├── solution.py      # main implementation
├── test.py          # test cases
├── template.py      # starter template
└── explanation.md   # detailed explanation
```

### Pattern B — Conceptual deep dives (theory-style)
Follows the numbered-tutorial convention from `ml-theory/` (e.g., `dpo_ppo_grpo_orpo/`):

```
topic_name/
├── README.md        # overview, reading order, comparison table
├── 01-concept.md    # sequential deep dives, numbered in reading order
├── 02-concept.md
├── ...
├── FAQ.md           # flagged questions & answers (see FAQ policy below)
└── template.py      # optional starter code
```

### File Naming
- Folders: `snake_case` (e.g., `multi_armed_bandit`, `dpo_ppo_grpo_orpo`)
- Numbered deep-dive files: `01-`, `02-`, … in reading order
- `README.md` is the entry point; `FAQ.md` holds Q&A

---

## Content Standards

### Deep-dive files should include
1. **One-sentence definition** at the very top ("In one sentence: …")
2. **Where it fits** — how the method relates to the others (RL pipeline position, "X minus Y")
3. **The loss function**, written in display LaTeX
4. **Term-by-term breakdown** — a `**Where:**` block naming every symbol and *why it's there*
5. **Intuition** paragraphs alongside the math (never math alone)
6. **A worked example** with concrete numbers where possible
7. **Comparison tables** to neighboring methods
8. **An implementation sketch** (runnable-style pseudocode with shape/meaning comments)

### Mathematical rigor
- Use LaTeX: `$...$` inline, `$$...$$` display
- **Define notation before use** (e.g., "$y_w$ = preferred response")
- Show the key derivation steps when non-obvious (e.g., why the DPO partition function cancels)
- Always pair equations with intuition

### The signature convention for this module: **term-by-term loss breakdowns**
Every loss function must be followed by a breakdown naming each term. Example:

```markdown
$$L^{\text{DPO}} = -\mathbb{E}\big[\log\sigma(\beta \log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})\big]$$

**Where:**
- $y_w, y_l$ = preferred / dispreferred responses
- $\beta$ = KL-strength temperature
- $\sigma$ = sigmoid, from the Bradley-Terry preference model
- ...
```

---

## FAQ Policy (IMPORTANT — question flagging)

Each conceptual topic maintains a **`FAQ.md` that acts as a question log**. This is a first-class deliverable, not an afterthought.

**Whenever the user asks a question** — in a prompt, a follow-up, or a review comment — it must be:
1. **Flagged** with a status tag:
   - `[NEW]` — captured, answer not yet written
   - `[ANSWERED]` — answer written in FAQ
   - `[TODO-TUTORIAL]` — reveals a gap to fold into a numbered `.md` file
2. **Logged** in the chronological **Question Log** table (with date and target file)
3. **Categorized** under Conceptual / Math / Implementation / Comparison / Troubleshooting
4. **Cross-linked** — when an answer fills a tutorial gap, update that file *and* flip the tag to `[ANSWERED]` with a pointer

The point: no question the user raises should be lost. The FAQ is the durable memory that feeds future tutorial improvements. When in doubt, over-capture.

### FAQ format
```markdown
### Q: [question] `[STATUS]`

**A:** [clear answer]

*Pointer:* [file.md](file.md), section — where this belongs / is covered
```

---

## Writing Style
- **Clear and accessible** — explain jargon on first use
- **Active voice** — "The policy maximizes reward," not "reward is maximized"
- **Progressive complexity** — start with the one-sentence idea, build to the full loss
- **Short paragraphs**, purposeful headers ("The Clipped Surrogate," not "Section 3.2")
- **Story-driven** — frame methods as a progression (PPO → GRPO → DPO → ORPO) so trade-offs are clear

---

## Comparison & Disambiguation

RL/alignment methods are best understood *relative to each other*. Every conceptual topic should include at least one comparison table with the dimensions that matter, e.g.:

| Aspect | Method A | Method B |
|--------|----------|----------|
| Reward source | reward model | implicit |
| Critic / value net | yes | no |
| Reference model | yes | no |
| Online sampling | yes | no |
| Models in memory | 4 | 1 |

---

## Quality Checklist

Before finishing a topic or update:

- [ ] Every loss has a **term-by-term** breakdown
- [ ] Notation is defined before use
- [ ] Intuition accompanies each equation
- [ ] At least one comparison table to related methods
- [ ] A concrete numeric example where feasible
- [ ] Implementation sketch with meaningful comments
- [ ] **FAQ.md updated** with any questions raised, properly flagged
- [ ] Internal links between files work
- [ ] README lists the reading order

---

## Principles
1. **Depth over breadth** — go deep on the objective and its terms
2. **Intuition with rigor** — math + plain-English "why"
3. **Methods as a progression** — show what each one adds or removes
4. **Living FAQs** — questions are flagged, logged, and fed back into tutorials
5. **Self-contained topics** — a reader can start from any topic's README

---

## Current Topics

| Topic | Pattern | Description |
|-------|---------|-------------|
| `multi_armed_bandit/` | A (implementation) | Exploration vs exploitation, bandit algorithms |
| `dpo_ppo_grpo_orpo/` | B (deep dive) | Preference-optimization loss functions compared |

---

## Last Updated
2026-07-09

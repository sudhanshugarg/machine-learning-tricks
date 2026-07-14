# ML System Design Module Guidelines

This document outlines standards and conventions for the `/ml-system-design/` module of the ML Interview Tricks repository. It complements the root [CLAUDE.md](../CLAUDE.md) and mirrors the FAQ-logging convention from [reinforcement-learning/CLAUDE.md](../reinforcement-learning/CLAUDE.md).

---

## Module Overview

The `/ml-system-design/` module contains **large-scale ML system design interview questions** — recommendation, ranking, retrieval, and other production ML systems, in the style of a Waymo/Meta/Google ML system design interview.

### Goals
- Practice framing ambiguous, open-ended ML system design prompts
- Force explicit tradeoff reasoning, not just "the right answer"
- Build a durable reference of terms, architectures, and follow-up questions per problem

---

## Folder Structure (standard, going forward)

Each problem gets its own `snake_case` folder with exactly three markdown files:

```
system_name/
├── design.md    # the problem statement ONLY — no hints, no answer scaffolding
├── solution.md  # the full detailed solution — architecture, tradeoffs, follow-ups
└── faq.md       # living Q&A log — definitions, deep-dives, and clarifications raised
                 # about this specific design.md / solution.md
```

- **`design.md`** — just the detailed problem statement: context, scale assumptions, and what a strong answer should cover. Do **not** put clarifying questions, hints, or expected-answer structure here — that belongs in `solution.md`. Keep it close to what an interviewer would actually hand a candidate.
- **`solution.md`** — the worked answer: clarifying questions, goals/constraints, architecture, deep dives on the focus areas called out in `design.md`, evaluation, tradeoffs, and common follow-up Q&A.
- **`faq.md`** — see **FAQ Policy** below. This is where term definitions and "wait, what does X mean / why did you choose Y" questions accumulate over time, kept separate from `solution.md` so the solution stays readable as a coherent answer.

Older folders in this module (e.g. `ad-recommendation-system/`, `chatgpt_clone/`, `hospital_icd_prediction/`) predate this convention and use extra files (`architecture.md`, `tradeoffs.md`, `template.py`, etc.) — leave them as-is unless asked to migrate. **All new problems should follow the three-file structure above.**

---

## FAQ Policy (IMPORTANT — question flagging)

Every problem folder maintains a **`faq.md` that acts as a running question log** for that specific design/solution. This is a first-class deliverable, not an afterthought — treat it the same way `reinforcement-learning/`'s FAQ convention does.

**Whenever the user asks a question** about a problem's solution — a term definition, "why did you pick X over Y," a deeper dive on a component — it must be:

1. **Flagged** with a status tag:
   - `[NEW]` — captured, answer not yet written
   - `[ANSWERED]` — answer written in the FAQ
   - `[TODO-SOLUTION]` — reveals a gap that should be folded back into `solution.md` (or `design.md` if it's a missing requirement)
2. **Logged** in the chronological **Question Log** table at the top of `faq.md` (with date and category)
3. **Categorized** — e.g. Terminology / Architecture / Tradeoffs / Math / Follow-up
4. **Cross-linked** — if the answer exposes a gap in `solution.md`, update that file too and flip the tag to `[ANSWERED]` with a pointer back to the section

The point: no question raised while working through a design should be lost. The FAQ is the durable memory that feeds future refinement of the solution. When in doubt, over-capture.

### FAQ format

```markdown
### Q: [question] `[STATUS]`

**A:** [clear, self-contained answer]

*Pointer:* [solution.md](solution.md), Section X — where this is (or should be) covered
```

---

## Content Standards

### `design.md`
- Problem statement framed with real context (who's asking, why it matters)
- Scale/data assumptions the candidate can rely on (or should confirm)
- A "What You Should Cover" list naming the focus areas the interviewer cares about
- Optional "Common Follow-up Questions to Expect" section
- **No answers** — this file should read like something handed to a candidate before they start whiteboarding

### `solution.md`
- Step-by-step structure: clarifying questions → goals/constraints → architecture → deep dives → evaluation → tradeoffs → follow-up Q&A
- Every non-obvious term introduced should be usable as a link target from `faq.md`
- Explicit tradeoff call-outs: **Decision** / **Tradeoff** / **Alternative**, not just a list of pros/cons
- ASCII diagrams for architecture and data flow (consistent with the rest of the repo)

### `faq.md`
- Starts with a **Question Log** table (chronological), then answers grouped by category
- Keep answers self-contained — a reader should be able to jump straight to one Q&A without reading the whole thread
- Use `*Pointer:*` links back into `solution.md`/`design.md` liberally

---

## Writing Style
- Follows the root [CLAUDE.md](../CLAUDE.md) markdown standards (LaTeX for math, clear headers, complexity analysis where relevant)
- Active voice, concrete numbers over vague scale ("5,000 vehicles" not "a large fleet")
- Prefer naming the **Decision** first, then the **Tradeoff**, then the **Alternative** — so a skim reveals the choice made, not just the debate

---

## Quality Checklist

Before considering a problem folder done:

- [ ] `design.md` contains only the problem statement (no answer scaffolding)
- [ ] `solution.md` covers every focus area named in `design.md`
- [ ] Every design decision in `solution.md` has an explicit tradeoff and alternative
- [ ] `faq.md` exists, even if just a Question Log header waiting for questions
- [ ] Any question asked in conversation about this problem is logged in `faq.md`, tagged, and answered or flagged `[TODO-SOLUTION]`
- [ ] Internal links between `design.md` / `solution.md` / `faq.md` resolve correctly

---

## Current Topics

| Topic | Follows 3-file convention | Description |
|-------|---------------------------|--------------|
| `driving_scene_search/` | Yes | Semantic search/retrieval over AV driving video; multimodal embeddings, long-tail sampling, vector search at scale |
| `post_incident_failure_attribution/` | Yes | Root-cause attribution (Perception/Prediction/Planning) after AV collisions/near-misses; anomaly detection, automated labeling, model observability |
| `ad-recommendation-system/` | No (legacy) | Ad ranking/recommendation for social media |
| `recommendation_system/` | No (legacy) | General recommendation system design |
| `chatgpt_clone/` | No (legacy) | Conversational LLM serving system |
| `hospital_icd_prediction/` | No (legacy) | ICD code prediction from clinical notes |
| `uber_pricing_system/` | No (legacy) | Surge/base pricing + bandit strategy |
| `reddit-ads-based-on-post/` | No (legacy) | Ad targeting based on post content |

---

## Last Updated
2026-07-13

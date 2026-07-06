# ML Theory Module Guidelines

This document outlines standards and conventions for the `/ml-theory/` module of the ML Interview Tricks repository.

---

## Module Overview

The `/ml-theory/` module contains **theoretical foundations and deep dives** into machine learning concepts. Unlike `/ml-coding/` (implementation-focused) or `/ml-system-design/` (systems-focused), this module explores the *why* behind ML algorithms.

### Goals
- Build intuition for core ML concepts
- Provide rigorous mathematical foundations
- Bridge theory and practice with code examples
- Serve as reference material for interview preparation

---

## Folder Structure

Each topic gets its own folder with standardized files:

```
topic_name/
├── README.md (optional)
├── CLAUDE.md (optional, topic-specific guidance)
├── 00-overview.md (entry point, high-level)
├── 01-core-concept-1.md (deep dive)
├── 02-core-concept-2.md (deep dive)
├── 03-implementation.md (code examples)
├── FAQ.md (questions & answers)
├── template.py (starter code template)
└── code_examples.py (runnable examples)
```

### File Naming
- **00-overview.md**: Always first—entry point for topic
- **01, 02, ...**: Sequential deep dives (numbered in reading order)
- **FAQ.md**: Common questions and clarifications
- **template.py**: Starter template with structure
- **code_examples.py**: Runnable Python examples

---

## Content Standards

### Each Tutorial Should Include

#### 1. Overview (00-overview.md)
- **What is this topic?** (1-2 sentence definition)
- **Why does it matter?** (practical relevance)
- **Big picture** (high-level flow/intuition)
- **Concrete example** (e.g., MNIST, toy dataset)
- **Prerequisites** (what should reader know first?)
- **Next steps** (links to deeper files)

#### 2. Mathematical Rigor
- **Use LaTeX** for all formulas: `$...$` for inline, `$$...$$` for display
- **Define notation** before using (e.g., "Let $x_0$ denote the original image")
- **Show derivations** when non-obvious
- **Provide intuition** alongside math (not just equations)

**Example:**
```markdown
$$q(x_t | x_0) = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$$

Where:
- $x_t$ = noisy image at timestep $t$
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ = Gaussian noise
- $\bar{\alpha}_t$ = cumulative product of schedule

**Intuition**: As $t$ increases, $\bar{\alpha}_t$ decreases (less signal, more noise).
```

#### 3. Concrete Examples
- **Use real datasets** (MNIST, CIFAR-10, toy data)
- **Show dimension flows** (e.g., 784 → 512 → 256 → 784)
- **Include ASCII diagrams** for architecture/flow
- **Step-by-step walkthroughs** with numbers

#### 4. Code Examples
- **Executable Python** (can be run independently)
- **Well-commented** (explain non-obvious parts)
- **Include imports** at top
- **Show input/output shapes** with comments

#### 5. Visual Aids
- **ASCII diagrams** for architecture/flow (easiest to review in markdown)
- **Tables** for comparisons and summaries
- **Code blocks** with clear syntax highlighting

### What NOT to Include
- Overly verbose explanations (be concise)
- Code that cannot run (always test)
- Unexplained notation (define before using)
- Disconnected math (connect to intuition)
- Circular dependencies (order topics for learning flow)

---

## Writing Style

### Tone
- **Clear and accessible** (explain jargon)
- **Precise and rigorous** (correct math/facts)
- **Conversational where appropriate** ("Why U-Net?" not "Justification for U-Net architecture")
- **Active voice** ("The network learns..." not "It is learned...")

### Structure
- **Headers with purpose**: "The Noise Schedule" not "Section 2.1"
- **Short paragraphs** (2-3 sentences per paragraph)
- **Progressive complexity**: Start simple, build to advanced
- **Conclusion sections** summarizing key points

### Math Notation
- **Consistent**: $x_0$ always means original, $x_t$ always means at timestep t
- **Document unusual choices**: "Let $\epsilon$ denote noise (not $\eta$) for brevity"
- **Use clear symbols**: Avoid $\xi$, $\zeta$, etc. unless necessary

---

## Comparison & Disambiguation

When a topic relates to others, include a comparison table:

```markdown
| Aspect | Method A | Method B |
|--------|----------|----------|
| **Speed** | Fast | Slow |
| **Quality** | Good | Excellent |
| **Stability** | Unstable | Stable |
```

This helps readers understand trade-offs.

---

## FAQ Management

Each topic gets a **FAQ.md** for:
- **Common conceptual misconceptions** ("Doesn't the network just memorize?")
- **Implementation details** ("Why T=1000 and not 100?")
- **Clarifications** on the tutorial ("What's the difference between α and ᾱ?")
- **Troubleshooting** ("My loss isn't decreasing")

### FAQ Format
```markdown
### Q: [Question about concept/implementation]

**A:** [Clear answer]

[Additional details if needed]
```

---

## Code Templates

Each topic includes a `template.py` with:

```python
"""
[Topic Name] - Template for [what it solves]

This template provides the basic structure for implementing [concept].
Follow the TODOs and modify the Implementation class as needed.
"""

import numpy as np
import torch
import torch.nn as nn

class Implementation:
    """[Concept] implementation template"""
    
    def __init__(self, **kwargs):
        """Initialize with key hyperparameters"""
        # TODO: Set up key parameters
        pass
    
    def compute(self, x):
        """Main computation/forward pass"""
        # TODO: Implement the core logic
        return x
    
    def explain(self):
        """Return human-readable explanation of current state"""
        return "Implementation details here"

if __name__ == "__main__":
    # Example usage
    impl = Implementation()
    result = impl.compute(sample_input)
    print(impl.explain())
```

---

## Linking Between Topics

Use markdown links liberally:

```markdown
See [[00-overview|Overview]] for big picture.
Learn more in [[02-forward-process|Forward Process]].
Compare with [[../another-topic/01-concept|Other Concept]].
```

This creates a web of knowledge where readers can explore related concepts.

---

## Example Structure: Diffusion Models

The diffusion-models folder exemplifies the standard:

```
diffusion-models/
├── 00-unet-architecture.md      # U-Net prerequisite
├── 01-overview.md               # Big picture
├── 02-forward-process.md        # Noise addition
├── 03-backward-process.md       # Denoising network
├── FAQ.md                        # Q&A on implementation
├── template.py                  # PyTorch template
└── README.md                     # Folder overview
```

**Reading order:**
1. Overview (context)
2. U-Net (architecture needed for understanding)
3. Forward process (how noise is added)
4. Backward process (how network learns)
5. FAQ (clarifications as needed)

---

## Additions & Updates

### When Adding a New Topic
1. Create folder: `ml-theory/new-topic/`
2. Start with `00-overview.md`
3. Plan 2-3 deep-dive files
4. Add FAQ.md (even if sparse initially)
5. Add template.py with structure
6. Update main README.md with link

### When Updating a Topic
- Clarify based on FAQ questions
- Keep files modular (don't overload 00-overview)
- Update FAQ when questions emerge
- Link newly added concepts from existing files

### When Adding FAQ Answers
- Add to FAQ.md immediately
- If answer reveals tutorial gap, update relevant file
- Keep FAQ organized by category (Conceptual, Implementation, etc.)

---

## Quality Checklist

Before pushing a new topic or update:

- [ ] **Overview is clear**: Reader knows purpose in first sentence
- [ ] **Math is correct**: Verify formulas, especially in derivations
- [ ] **Code runs**: Every code block is tested
- [ ] **Notation is defined**: No undefined symbols
- [ ] **Examples are concrete**: MNIST, CIFAR, actual numbers
- [ ] **ASCII diagrams are clear**: Dimensions labeled, flow logical
- [ ] **Links work**: Internal references to other files are correct
- [ ] **Style is consistent**: Tone, structure match other files
- [ ] **FAQ covers ambiguities**: Address likely confusion points

---

## Principles

1. **Depth over breadth**: Go deep on fundamentals, not shallow surveys
2. **Intuition with rigor**: Math + explanation (not just formulas)
3. **Modular files**: Reader can pick a section; doesn't need whole topic
4. **Living documents**: FAQ grows; tutorial improves; nothing is "final"
5. **Code validates theory**: Every concept has runnable examples
6. **Theory informs practice**: Tie back to ML systems and applications

---

## References & Templates

- **Mathematical notation guide**: Use consistent symbols across module
- **LaTeX reference**: $\alpha$, $\beta$, $\bar{x}$, etc.
- **ASCII art tools**: Use monospace boxes, arrows, indentation
- **Python template**: See `template.py` in any topic folder

---

## Last Updated
2026-07-06


# ML Interview Tricks - Project Guidelines

> **Note:** This document should be continuously updated with rules, principles, and conventions as the repository evolves. Keep this as the single source of truth for project standards.

## Repository Overview

This repository is a comprehensive collection of machine learning interview questions and solutions organized by topic area. The goal is to provide detailed solutions, explanations, and implementations for various ML interview domains.

---

## Repository Structure

### 1. Root Level Files
- `README.md` - Overview of the repo, quick navigation guide
- `CLAUDE.md` - Project guidelines and conventions (this file)
- `.gitignore` - Standard Python/ML gitignore
- `requirements.txt` - Shared dependencies (numpy, pandas, scikit-learn, etc.)

### 2. Main Module Folders

#### `/ml-coding/` - Implementation-focused problems
Practical coding problems focused on ML algorithms and data manipulation.

- Classification problems (binary, multiclass)
- Regression problems
- Clustering implementations
- Feature engineering challenges
- Data preprocessing tasks

**Structure for each problem:**
```
problem_name/
├── README.md (problem statement)
├── solution.py (main implementation)
├── test.py (test cases)
├── template.py (starter template with boilerplate)
└── explanation.md (detailed walkthrough)
```

#### `/ml-system-design/` - Large-scale ML system design
System design questions focused on building production ML systems.

- Recommendation systems
- Ranking systems
- Search/retrieval systems
- Real-time prediction systems
- Model serving & inference

**Structure for each design:**
```
system_name/
├── design.md (high-level design)
├── architecture.md (detailed architecture)
├── template.py (starter template with boilerplate)
└── tradeoffs.md (design tradeoffs and considerations)
```

#### `/ml-probability/` - Probability & statistics
Theoretical and practical problems involving probability and statistics.

- Bayesian inference
- Probabilistic modeling
- Statistical tests
- Distributions & sampling

**Structure for each problem:**
```
problem_name/
├── solution.md (problem statement and solution)
├── code.py (implementation)
├── template.py (starter template with boilerplate)
└── derivations.md (mathematical derivations)
```

#### `/ml-theory/` - (Optional) Theoretical foundations
Theoretical concepts and mathematical foundations.

- Loss functions & optimization
- Regularization techniques
- Information theory

**Structure for each topic:**
```
topic_name/
├── explanation.md (theoretical explanation)
├── template.py (starter template with boilerplate)
└── examples.py (code examples)
```

#### `/reinforcement-learning/` - Reinforcement learning
RL-specific interview questions and concepts.

- Markov Decision Processes (MDPs)
- Q-learning and value-based methods
- Policy gradient methods
- Actor-Critic algorithms
- Multi-armed bandits
- Exploration vs exploitation tradeoffs

**Structure for each problem:**
```
problem_name/
├── README.md (problem statement)
├── solution.py (main implementation)
├── test.py (test cases)
├── template.py (starter template with boilerplate)
└── explanation.md (detailed explanation)
```

#### `/transformers/` - Transformers and attention mechanisms
Transformer architecture and related deep learning concepts.

- Attention mechanisms
- Transformer architecture
- Pre-training and fine-tuning
- Vision transformers
- Positional encoding
- Multi-head attention

**Structure for each topic:**
```
topic_name/
├── explanation.md (theoretical explanation)
├── template.py (starter template with boilerplate)
├── implementation.py (implementation details)
└── examples.py (usage examples)
```

### 3. Supporting Directories
- `/utils/` - Shared utility functions (data loading, plotting, etc.)
- `/datasets/` - Sample datasets for practice problems
- `/notebooks/` - Jupyter notebooks for exploration/visualization

---

## Naming & Coding Conventions

### File Naming
- Problem folders: `snake_case` (e.g., `customer_segmentation`, `ab_testing`)
- Python files: `snake_case` (e.g., `solution.py`, `test.py`)
- Markdown files: `snake_case` or `CamelCase` for titles

### Code Standards
- Follow PEP 8 style guide
- Include docstrings for functions and classes
- Add type hints where appropriate
- Write clear, readable code with meaningful variable names

### Markdown Standards
- Use clear section headers with proper hierarchy
- Include problem statement at the beginning
- Provide approach/algorithm explanation before code
- Add complexity analysis (time and space)
- Include edge cases and assumptions
- Use LaTeX notation for mathematical expressions: `$...$` for inline, `$$...$$` for display

### Content Structure
Each problem solution should include:
1. **Problem Statement** - Clear description of what needs to be solved
2. **Approach** - High-level strategy and algorithm explanation
3. **Implementation** - Well-commented code
4. **Complexity Analysis** - Time and space complexity
5. **Edge Cases** - Known edge cases and how they're handled
6. **Example Walkthrough** - Step-by-step example

### Template Files

Each problem folder includes a `template.py` file with:
- **Standard Imports**: numpy, pandas, torch, torch.nn pre-imported
- **Solution/Model Class**: Template class where implementation goes
  - **ML Coding**: `Solution` class with `fit()` and `predict()` methods
  - **System Design**: `SystemDesign` class with design methods
  - **Probability**: `Solver` class with `analyze()` and `solve()` methods
  - **ML Theory**: `Implementation` class with `compute()` method
  - **Reinforcement Learning**: `Agent` class with `act()` and `learn()` methods
  - **Transformers**: `Model` class (inherits from `nn.Module`) with `forward()` method
- **Main Function**: Starter code for testing and experimentation
- **Comments**: Guidance on expected usage and modifications

Use the template as a starting point for your implementation.

---

## Principles & Guidelines

### Code Quality
- Write modular, testable code
- Include unit tests for all implementations
- Document assumptions and constraints
- Keep solutions focused and avoid over-engineering

### Documentation
- Every problem/design should be fully self-contained
- Explain the "why" not just the "what"
- Include references to papers, articles, or external resources when relevant
- Make explanations accessible to someone learning the topic

### Organization
- Keep related problems in the same module folder
- Use consistent templates for similar problem types
- Maintain a table of contents in each module's README

---

## Future Enhancements

Potential modules to consider adding:
- `/deep-learning/` - Deep learning specific interview questions
- `/nlp/` - NLP and LLM interview questions
- `/computer-vision/` - Computer vision interview questions
- `/data-engineering/` - Data engineering for ML systems
- `/case-studies/` - Real-world case study analyses

---

## Last Updated
Initial structure plan created.

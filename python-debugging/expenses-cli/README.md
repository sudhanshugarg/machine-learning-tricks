# Debugging Interview: Broken Expenses CLI (Python)

## Timebox

60 minutes

## Context

You are given a command-line expense tracker with several bugs. Debug and fix correctness issues while maintaining readable code.

## Rules

- No AI tools or coding assistance
- You may run tests and edit code freely

## Commands

- `add "description" --amount X.YY --category NAME --date YYYY-MM-DD`
- `total --month YYYY-MM`
- `by-category --month YYYY-MM`
- `delete <expense_id>`

## Getting Started

```bash
cd python-debugging/expenses-cli
python3 -m unittest tests.py -v
```

## Deliverable

Make tests pass with clean, maintainable fixes.

# Debugging Interview: Broken Todo CLI (Python)

## Timebox

60 minutes

## Context

You are given a command-line task manager with several bugs. Your goal is to debug and fix the application while keeping the code readable and maintainable.

This interview focuses on:

- Code quality under time pressure
- Coding fundamentals
- Debugging speed and prioritization
- Practical shortcuts that improve your workflow

## Rules

- No AI tools or coding assistance
- You may run tests and edit code freely
- You may refactor if it helps clarity and correctness

## Commands

The app supports these commands:

- `add "task title" --priority 1-5 --due YYYY-MM-DD`
- `list --status open|done --sort due|priority`
- `done <task_id>`
- `delete <task_id>`

## Getting Started

Run tests:

```bash
cd python-debugging/todo-cli
python3 -m unittest tests.py -v
```

Try the CLI manually:

```bash
cd python-debugging/todo-cli
python3 buggy.py add "Pay rent" --priority 3 --due 2026-05-01
python3 buggy.py list --sort due
python3 buggy.py done 1
python3 buggy.py delete 1
```

## Deliverable

Make the tests pass while preserving or improving code quality.

# Debugging Interview: Broken Inventory CLI (Python)

## Timebox

60 minutes

## Context

You are given a command-line inventory manager with several bugs. Fix behavior while keeping the code clean and readable.

## Rules

- No AI tools or coding assistance
- You may run tests and edit code freely

## Commands

- `add "item name" --qty N --price X.Y`
- `sell <item_id> --qty N`
- `delete <item_id>`
- `low-stock --threshold N`

## Getting Started

```bash
cd python-debugging/inventory-cli
python3 -m unittest tests.py -v
```

## Deliverable

Make tests pass with high-quality fixes.

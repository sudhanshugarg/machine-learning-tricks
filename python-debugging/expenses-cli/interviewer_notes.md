# Interviewer Notes (Hidden)

## Intended Bugs

1. Empty/corrupt JSON crashes in `load_expenses`.
2. `add_expense` truncates decimal amounts.
3. `total_for_month` uses incorrect month prefix logic.
4. `totals_by_category` sorts totals ascending.
5. `delete_expense` deletes by description instead of ID.

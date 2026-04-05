# Interviewer Notes (Hidden)

## Intended Bugs

1. `load_tasks` crashes on empty/corrupted JSON.
2. `add_task` does not enforce `priority` in range 1-5 and silently defaults non-int values.
3. `list_tasks(sort_by="due")` sorts as strings.
4. `delete_task` deletes by title instead of ID.
5. `delete_task` reindexes IDs after deletion, breaking ID stability.

## Signals to Evaluate

- Can the candidate quickly reproduce failures?
- Do they write small focused fixes instead of broad risky rewrites?
- Do they preserve readability and clear naming?
- Do they add/adjust tests where useful?
- Do they prioritize correctness first, then cleanup?

## Expected Correct Behavior

- Invalid priority should raise `ValueError`.
- Empty/corrupt task file should be handled safely as an empty list.
- Deleting one task should not delete other tasks with the same title.
- IDs should remain stable over the lifetime of tasks.
- Sorting by due date should be chronological.

## Suggested Debrief Questions

- Which bug did you fix first and why?
- What shortcuts helped you debug quickly?
- What would you improve with another 30 minutes?

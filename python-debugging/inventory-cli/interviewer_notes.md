# Interviewer Notes (Hidden)

## Intended Bugs

1. Empty/corrupt JSON crashes in `load_items`.
2. `add_item` accepts negative qty/price.
3. `sell_item` allows stock to go below zero.
4. `delete_item` deletes by name instead of ID.
5. `next_id` collides after deletion.
6. `low_stock` sorts qty as strings.

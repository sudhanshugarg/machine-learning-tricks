import argparse
import json
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_inventory.json")


def load_items(path=DATA_FILE):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        # Bug: crashes on empty/corrupted JSON.
        return json.load(f)


def save_items(items, path=DATA_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def next_id(items):
    # Bug: ID collisions after deletion.
    return len(items) + 1


def add_item(name, qty, price, path=DATA_FILE):
    items = load_items(path)

    qty = int(qty)
    price = float(price)

    # Bug: negative qty/price accepted.
    item = {
        "id": next_id(items),
        "name": name,
        "qty": qty,
        "price": price,
        "active": True,
    }
    items.append(item)
    save_items(items, path)
    return item


def sell_item(item_id, qty, path=DATA_FILE):
    items = load_items(path)
    qty = int(qty)

    for item in items:
        if item["id"] == item_id and item["active"]:
            # Bug: allows stock to go negative.
            item["qty"] -= qty
            save_items(items, path)
            return item

    raise ValueError(f"Item {item_id} not found")


def delete_item(item_id, path=DATA_FILE):
    items = load_items(path)
    target = None

    for item in items:
        if item["id"] == item_id:
            target = item
            break

    if target is None:
        raise ValueError(f"Item {item_id} not found")

    # Bug: remove by name can delete duplicates.
    new_items = [item for item in items if item["name"] != target["name"]]
    save_items(new_items, path)
    return target


def low_stock(threshold, path=DATA_FILE):
    items = load_items(path)
    threshold = int(threshold)
    rows = [item for item in items if item["active"] and item["qty"] <= threshold]

    # Bug: string-sort for qty.
    rows.sort(key=lambda x: str(x["qty"]))
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Inventory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--qty", required=True)
    add.add_argument("--price", required=True)

    sell = sub.add_parser("sell")
    sell.add_argument("item_id", type=int)
    sell.add_argument("--qty", required=True)

    delete = sub.add_parser("delete")
    delete.add_argument("item_id", type=int)

    low = sub.add_parser("low-stock")
    low.add_argument("--threshold", required=True)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "add":
        item = add_item(args.name, args.qty, args.price)
        print("Added:", item)
    elif args.command == "sell":
        item = sell_item(args.item_id, args.qty)
        print("Sold:", item)
    elif args.command == "delete":
        item = delete_item(args.item_id)
        print("Deleted:", item)
    elif args.command == "low-stock":
        for item in low_stock(args.threshold):
            print(item)


if __name__ == "__main__":
    main()

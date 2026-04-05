import argparse
import json
import os
from collections import defaultdict

DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_expenses.json")


def load_expenses(path=DATA_FILE):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        # Bug: crashes on empty/corrupt JSON.
        return json.load(f)


def save_expenses(expenses, path=DATA_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(expenses, f, indent=2)


def next_id(expenses):
    if not expenses:
        return 1
    return max(e["id"] for e in expenses) + 1


def add_expense(description, amount, category, date, path=DATA_FILE):
    expenses = load_expenses(path)

    # Bug: converts to int and truncates decimals.
    amount = int(float(amount))

    expense = {
        "id": next_id(expenses),
        "description": description,
        "amount": amount,
        "category": category,
        "date": date,
    }
    expenses.append(expense)
    save_expenses(expenses, path)
    return expense


def total_for_month(month, path=DATA_FILE):
    expenses = load_expenses(path)

    # Bug: wrong prefix matching for month like 2026-03.
    filtered = [e for e in expenses if e["date"].startswith(month.replace("-0", "-"))]
    return round(sum(e["amount"] for e in filtered), 2)


def totals_by_category(month, path=DATA_FILE):
    expenses = load_expenses(path)
    filtered = [e for e in expenses if e["date"].startswith(month)]

    totals = defaultdict(float)
    for e in filtered:
        totals[e["category"]] += e["amount"]

    rows = [{"category": k, "total": round(v, 2)} for k, v in totals.items()]

    # Bug: ascending sort by total; expected descending.
    rows.sort(key=lambda x: x["total"])
    return rows


def delete_expense(expense_id, path=DATA_FILE):
    expenses = load_expenses(path)

    target = None
    for e in expenses:
        if e["id"] == expense_id:
            target = e
            break

    if target is None:
        raise ValueError(f"Expense {expense_id} not found")

    # Bug: delete by description can remove multiple rows.
    new_expenses = [e for e in expenses if e["description"] != target["description"]]
    save_expenses(new_expenses, path)
    return target


def build_parser():
    parser = argparse.ArgumentParser(description="Expenses CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("description")
    add.add_argument("--amount", required=True)
    add.add_argument("--category", required=True)
    add.add_argument("--date", required=True)

    total = sub.add_parser("total")
    total.add_argument("--month", required=True)

    by_cat = sub.add_parser("by-category")
    by_cat.add_argument("--month", required=True)

    delete = sub.add_parser("delete")
    delete.add_argument("expense_id", type=int)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "add":
        print(add_expense(args.description, args.amount, args.category, args.date))
    elif args.command == "total":
        print(total_for_month(args.month))
    elif args.command == "by-category":
        for row in totals_by_category(args.month):
            print(row)
    elif args.command == "delete":
        print(delete_expense(args.expense_id))


if __name__ == "__main__":
    main()

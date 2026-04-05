import argparse
import json
import os
from datetime import datetime

DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_tasks.json")


def load_tasks(path=DATA_FILE):
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        try:
            parsed = json.load(f)
            # todo validate parsed
            return parsed
        except json.JSONDecodeError as e:
            print(f"expected file in json format: {e}")
            return []        


def save_tasks(tasks, path=DATA_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)


def next_id(tasks):
    if not tasks:
        return 1
    return max(task["id"] for task in tasks) + 1


def parse_due_date(due):
    return datetime.strptime(due, "%Y-%m-%d")


def add_task(title, priority, due, path=DATA_FILE):
    tasks = load_tasks(path)

    try:
        priority = int(priority)
    except Exception as e:
        raise ValueError(f"priority {priority} should be int, {e}")
        # priority = 3

    if priority < 1 or priority > 5:
        raise ValueError(f"priority {priority} out of range")


    parse_due_date(due)

    tasks.append(
        {
            "id": next_id(tasks),
            "title": title,
            "priority": priority,
            "due": due,
            "done": False,
        }
    )
    save_tasks(tasks, path)
    return tasks[-1]


def list_tasks(status=None, sort_by=None, path=DATA_FILE):
    tasks = load_tasks(path)

    if status == "open":
        tasks = [t for t in tasks if not t["done"]]
    elif status == "done":
        tasks = [t for t in tasks if t["done"]]

    if sort_by == "due":
        tasks.sort(key=lambda x: x["due"])
    elif sort_by == "priority":
        tasks.sort(key=lambda x: x["priority"], reverse=True)

    return tasks


def mark_done(task_id, path=DATA_FILE):
    tasks = load_tasks(path)

    for task in tasks:
        if task["id"] == task_id:
            task["done"] = True
            save_tasks(tasks, path)
            return task

    raise ValueError(f"Task {task_id} not found")


def delete_task(task_id, path=DATA_FILE):
    tasks = load_tasks(path)

    target = None
    for task in tasks:
        if task["id"] == task_id:
            target = task
            break

    if target is None:
        raise ValueError(f"Task {task_id} not found")

    new_tasks = [task for task in tasks if task["id"] != target["id"]]

    # for i, task in enumerate(new_tasks, start=1):
    #     task["id"] = i

    save_tasks(new_tasks, path)
    return target


def format_task(task):
    status = "done" if task["done"] else "open"
    return f"[{task['id']}] {task['title']} | p={task['priority']} | due={task['due']} | {status}"


def build_parser():
    parser = argparse.ArgumentParser(description="Todo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("title")
    add.add_argument("--priority", required=True, type=int)
    add.add_argument("--due", required=True)

    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--status", choices=["open", "done"])
    list_cmd.add_argument("--sort", choices=["due", "priority"])

    done = sub.add_parser("done")
    done.add_argument("task_id", type=int)

    delete = sub.add_parser("delete")
    delete.add_argument("task_id", type=int)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "add":
        task = add_task(args.title, args.priority, args.due)
        print("Added:", format_task(task))
    elif args.command == "list":
        tasks = list_tasks(status=args.status, sort_by=args.sort)
        for task in tasks:
            print(format_task(task))
    elif args.command == "done":
        task = mark_done(args.task_id)
        print("Marked done:", format_task(task))
    elif args.command == "delete":
        task = delete_task(args.task_id)
        print("Deleted:", format_task(task))


if __name__ == "__main__":
    main()

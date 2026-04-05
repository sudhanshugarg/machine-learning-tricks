import json
import tempfile
import unittest
from pathlib import Path

import tofix as buggy


class TodoCliDebuggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.tmp_dir.name) / "tasks.json"
        self.seed(
            [
                {
                    "id": 1,
                    "title": "Pay rent",
                    "priority": 3,
                    "due": "2026-04-01",
                    "done": False,
                },
                {
                    "id": 2,
                    "title": "Fix bug",
                    "priority": 5,
                    "due": "2026-03-15",
                    "done": False,
                },
                {
                    "id": 3,
                    "title": "Fix bug",
                    "priority": 1,
                    "due": "2026-03-20",
                    "done": True,
                },
            ]
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def seed(self, tasks):
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f)

    def read_all(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_list_sort_due_orders_by_actual_date(self):
        tasks = buggy.list_tasks(sort_by="due", path=str(self.data_path))
        self.assertEqual([t["id"] for t in tasks], [2, 3, 1])

    def test_add_rejects_priority_out_of_range(self):
        with self.assertRaises(ValueError):
            buggy.add_task("Bad priority", 0, "2026-05-01", path=str(self.data_path))
        with self.assertRaises(ValueError):
            buggy.add_task("Bad priority", 6, "2026-05-01", path=str(self.data_path))

    def test_add_rejects_non_integer_priority(self):
        with self.assertRaises(ValueError):
            buggy.add_task("Bad priority", "high", "2026-05-01", path=str(self.data_path))

    def test_delete_removes_only_target_id(self):
        buggy.delete_task(2, path=str(self.data_path))
        tasks = self.read_all()
        self.assertEqual(len(tasks), 2)
        self.assertEqual([t["id"] for t in tasks], [1, 3])

    def test_ids_remain_stable_after_delete_then_done(self):
        buggy.delete_task(1, path=str(self.data_path))
        updated = buggy.mark_done(3, path=str(self.data_path))
        self.assertTrue(updated["done"])
        self.assertEqual(updated["id"], 3)

    def test_load_handles_empty_file_as_no_tasks(self):
        self.data_path.write_text("", encoding="utf-8")
        tasks = buggy.load_tasks(path=str(self.data_path))
        self.assertEqual(tasks, [])

    def test_load_handles_corrupted_json_as_no_tasks(self):
        self.data_path.write_text("{not-json", encoding="utf-8")
        tasks = buggy.load_tasks(path=str(self.data_path))
        self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()

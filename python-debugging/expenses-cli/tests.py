import json
import tempfile
import unittest
from pathlib import Path

import buggy


class ExpensesDebuggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.tmp_dir.name) / "expenses.json"
        self.seed(
            [
                {
                    "id": 1,
                    "description": "Lunch",
                    "amount": 12.75,
                    "category": "food",
                    "date": "2026-03-02",
                },
                {
                    "id": 2,
                    "description": "Bus",
                    "amount": 2.50,
                    "category": "transport",
                    "date": "2026-03-11",
                },
                {
                    "id": 3,
                    "description": "Lunch",
                    "amount": 8.25,
                    "category": "food",
                    "date": "2026-03-21",
                },
                {
                    "id": 4,
                    "description": "Book",
                    "amount": 15.00,
                    "category": "learning",
                    "date": "2026-04-01",
                },
            ]
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def seed(self, rows):
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)

    def read_all(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_load_handles_corrupt_json(self):
        self.data_path.write_text("{broken", encoding="utf-8")
        self.assertEqual(buggy.load_expenses(path=str(self.data_path)), [])

    def test_add_preserves_decimal_amount(self):
        created = buggy.add_expense(
            "Coffee",
            "3.40",
            "food",
            "2026-03-22",
            path=str(self.data_path),
        )
        self.assertEqual(created["amount"], 3.40)

    def test_total_for_month_uses_exact_month_filter(self):
        total = buggy.total_for_month("2026-03", path=str(self.data_path))
        self.assertEqual(total, 23.50)

    def test_totals_by_category_sorted_descending(self):
        rows = buggy.totals_by_category("2026-03", path=str(self.data_path))
        self.assertEqual([r["category"] for r in rows], ["food", "transport"])

    def test_delete_removes_only_target_id(self):
        buggy.delete_expense(1, path=str(self.data_path))
        rows = self.read_all()
        self.assertEqual([r["id"] for r in rows], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()

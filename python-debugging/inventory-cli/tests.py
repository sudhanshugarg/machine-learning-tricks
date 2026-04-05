import json
import tempfile
import unittest
from pathlib import Path

import buggy


class InventoryDebuggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.tmp_dir.name) / "inventory.json"
        self.seed(
            [
                {"id": 1, "name": "Cable", "qty": 10, "price": 9.99, "active": True},
                {"id": 2, "name": "Mouse", "qty": 2, "price": 25.0, "active": True},
                {"id": 3, "name": "Mouse", "qty": 1, "price": 22.0, "active": True},
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

    def test_load_handles_empty_file(self):
        self.data_path.write_text("", encoding="utf-8")
        self.assertEqual(buggy.load_items(str(self.data_path)), [])

    def test_add_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            buggy.add_item("Keyboard", -1, 49.99, path=str(self.data_path))
        with self.assertRaises(ValueError):
            buggy.add_item("Keyboard", 1, -49.99, path=str(self.data_path))

    def test_sell_cannot_go_below_zero(self):
        with self.assertRaises(ValueError):
            buggy.sell_item(2, 5, path=str(self.data_path))

    def test_delete_removes_only_target_id(self):
        buggy.delete_item(2, path=str(self.data_path))
        rows = self.read_all()
        self.assertEqual([r["id"] for r in rows], [1, 3])

    def test_next_id_does_not_collide_after_delete(self):
        buggy.delete_item(2, path=str(self.data_path))
        created = buggy.add_item("Keyboard", 3, 55, path=str(self.data_path))
        self.assertEqual(created["id"], 4)

    def test_low_stock_sorts_numerically(self):
        rows = buggy.low_stock(10, path=str(self.data_path))
        self.assertEqual([r["qty"] for r in rows], [1, 2, 10])


if __name__ == "__main__":
    unittest.main()

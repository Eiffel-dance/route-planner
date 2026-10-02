import unittest

import app
from app import plan


class SmokeTest(unittest.TestCase):
    def test_import(self):
        self.assertTrue(app)


class TestPlanHappyPath(unittest.TestCase):
    def test_path_endpoints_and_cost(self):
        result = plan(6, 4, {(2, 0), (2, 1), (2, 2)}, (0, 0), (5, 3))
        path = result["path"]
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (5, 3))
        self.assertEqual(result["cost"], len(path) - 1)
        self.assertEqual(len(path), len(set(path)))  # no repeated cells

    def test_path_is_valid_walk(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        result = plan(6, 4, blocked, (0, 0), (5, 3))
        for (x1, y1), (x2, y2) in zip(result["path"], result["path"][1:]):
            self.assertEqual(abs(x2 - x1) + abs(y2 - y1), 1)
        for cell in result["path"]:
            self.assertNotIn(cell, blocked)
            self.assertTrue(0 <= cell[0] < 6 and 0 <= cell[1] < 4)

    def test_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1))
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0, "expanded": 1})

    def test_unreachable_returns_none_and_counts_closed(self):
        # (0,0) is walled in; only it can ever be closed.
        result = plan(3, 3, {(1, 0), (0, 1)}, (0, 0), (2, 2))
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"], 1)

    def test_unreachable_expanded_equals_all_reachable_cells(self):
        # Vertical wall splits the grid; left half is fully explored.
        blocked = {(2, y) for y in range(4)}
        result = plan(5, 4, blocked, (0, 0), (4, 3))
        self.assertIsNone(result["path"])
        self.assertEqual(result["expanded"], 8)  # 2 columns x 4 rows

    def test_list_coords_and_duplicate_blocked(self):
        as_lists = plan(4, 3, [[1, 0], [1, 0], [1, 1]], [0, 0], [3, 2])
        as_tuples = plan(4, 3, {(1, 0), (1, 1)}, (0, 0), (3, 2))
        self.assertEqual(as_lists, as_tuples)

    def test_blocked_iteration_order_does_not_matter(self):
        cells = [
            (x, y)
            for x in range(5)
            for y in range(5)
            if (x + y) % 3 == 0 and (x, y) not in ((0, 0), (4, 4))
        ]
        forward = plan(5, 5, cells, (0, 0), (4, 4))
        reverse = plan(5, 5, list(reversed(cells)), (0, 0), (4, 4))
        self.assertEqual(forward, reverse)

    def test_deterministic_tie_break(self):
        # Empty grid: many optimal paths exist. From (0,0) the candidates
        # (1,0) and (0,1) tie on both f and h, so the fixed (x, y) order
        # must pick (0, 1) — the path hugs the y-axis first.
        result = plan(4, 4, set(), (0, 0), (2, 2))
        self.assertEqual(
            result["path"], [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        )
        self.assertEqual(result["cost"], 4)


class TestPlanValidation(unittest.TestCase):
    def test_dimension_type_errors(self):
        for bad in ("3", 3.0, None, True):
            with self.assertRaises(TypeError, msg=f"width={bad!r}"):
                plan(bad, 3, set(), (0, 0), (1, 1))
            with self.assertRaises(TypeError, msg=f"height={bad!r}"):
                plan(3, bad, set(), (0, 0), (1, 1))

    def test_dimension_value_errors(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                plan(bad, 3, set(), (0, 0), (1, 1))
            with self.assertRaises(ValueError):
                plan(3, bad, set(), (0, 0), (1, 1))

    def test_coordinate_type_errors(self):
        for bad in (1, "ab", (1,), (1, 2, 3), (1, "a"), (1, 1.5), (True, 0), None):
            with self.assertRaises(TypeError, msg=f"start={bad!r}"):
                plan(3, 3, set(), bad, (1, 1))
            with self.assertRaises(TypeError, msg=f"goal={bad!r}"):
                plan(3, 3, set(), (0, 0), bad)
            with self.assertRaises(TypeError, msg=f"blocked={bad!r}"):
                plan(3, 3, [bad], (0, 0), (1, 1))

    def test_out_of_bounds_value_errors(self):
        for bad in ((-1, 0), (0, -1), (3, 0), (0, 3)):
            with self.assertRaises(ValueError, msg=f"start={bad!r}"):
                plan(3, 3, set(), bad, (1, 1))
            with self.assertRaises(ValueError, msg=f"goal={bad!r}"):
                plan(3, 3, set(), (0, 0), bad)
            with self.assertRaises(ValueError, msg=f"blocked={bad!r}"):
                plan(3, 3, [bad], (0, 0), (1, 1))

    def test_endpoint_on_obstacle(self):
        with self.assertRaises(ValueError):
            plan(3, 3, {(0, 0)}, (0, 0), (1, 1))
        with self.assertRaises(ValueError):
            plan(3, 3, {(1, 1)}, (0, 0), (1, 1))

    def test_validation_happens_before_search(self):
        # An invalid goal must raise even if the start is already walled in.
        with self.assertRaises(ValueError):
            plan(3, 3, {(1, 0), (0, 1)}, (0, 0), (9, 9))


if __name__ == "__main__":
    unittest.main()

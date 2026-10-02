import unittest

import app
from app import plan


class PathSemanticsTest(unittest.TestCase):
    def test_basic_path(self):
        result = plan(6, 4, {(2, 0), (2, 1), (2, 2)}, (0, 0), (5, 3))
        path = result["path"]
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (5, 3))
        self.assertEqual(result["cost"], len(path) - 1)
        self.assertEqual(len(set(path)), len(path))  # no repeated cells
        blocked = {(2, 0), (2, 1), (2, 2)}
        for x, y in path:
            self.assertTrue(0 <= x < 6 and 0 <= y < 4)
            self.assertNotIn((x, y), blocked)
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1))
        self.assertEqual(result, {"path": [(1, 1)], "cost": 0, "expanded": 1})

    def test_unreachable(self):
        # Wall splits the grid; goal is sealed off.
        blocked = {(1, y) for y in range(3)}
        result = plan(3, 3, blocked, (0, 0), (2, 2))
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        # Only the left column is reachable: 3 closed nodes.
        self.assertEqual(result["expanded"], 3)

    def test_goal_counted_in_expanded(self):
        # 1x2 corridor: start closes, then goal closes.
        result = plan(2, 1, set(), (0, 0), (1, 0))
        self.assertEqual(result["expanded"], 2)
        self.assertEqual(result["cost"], 1)

    def test_list_coordinates_normalized(self):
        result = plan(3, 3, [[1, 0]], [0, 0], [2, 2])
        self.assertEqual(result["path"][0], (0, 0))
        self.assertEqual(result["path"][-1], (2, 2))
        self.assertNotIn((1, 0), result["path"])

    def test_blocked_duplicates_merged(self):
        deduped = plan(4, 4, {(1, 1), (2, 2)}, (0, 0), (3, 3))
        duplicated = plan(4, 4, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1)],
                          (0, 0), (3, 3))
        self.assertEqual(deduped, duplicated)

    def test_result_independent_of_blocked_iteration_order(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        forward = plan(5, 3, cells, (0, 0), (4, 2))
        reverse = plan(5, 3, list(reversed(cells)), (0, 0), (4, 2))
        shuffled = plan(5, 3, [cells[2], cells[0], cells[3], cells[1]],
                        (0, 0), (4, 2))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, shuffled)

    def test_replay_determinism(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        first = plan(6, 4, blocked, (0, 0), (5, 3))
        for _ in range(10):
            self.assertEqual(plan(6, 4, blocked, (0, 0), (5, 3)), first)


class CostMapTest(unittest.TestCase):
    def test_costs_none_matches_unit_cost(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        baseline = plan(6, 4, blocked, (0, 0), (5, 3))
        explicit_none = plan(6, 4, blocked, (0, 0), (5, 3), None)
        self.assertEqual(baseline, explicit_none)

    def test_uniform_costs_match_unit_cost(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        baseline = plan(6, 4, blocked, (0, 0), (5, 3))
        ones = [[1] * 6 for _ in range(4)]
        with_ones = plan(6, 4, blocked, (0, 0), (5, 3), ones)
        self.assertEqual(with_ones, baseline)

    def test_uniform_scaled_costs_keep_optimal_path(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        baseline = plan(6, 4, blocked, (0, 0), (5, 3))
        scaled = [[3] * 6 for _ in range(4)]
        with_scaled = plan(6, 4, blocked, (0, 0), (5, 3), scaled)
        self.assertEqual(with_scaled["path"], baseline["path"])
        self.assertEqual(with_scaled["cost"], 3 * baseline["cost"])

    def test_cost_map_changes_route(self):
        # Direct corridor along y=0 is expensive; the planner must detour.
        costs = [
            [1, 9, 9, 9, 1],
            [1, 1, 1, 1, 1],
        ]
        result = plan(5, 2, set(), (0, 0), (4, 0), costs)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                          (4, 0)])
        # Start cell cost is not counted; entered cells sum to 6.
        self.assertEqual(result["cost"], 6)

    def test_cost_equals_sum_of_entered_cells(self):
        costs = [
            [5, 2, 1],
            [1, 1, 1],
            [1, 1, 4],
        ]
        result = plan(3, 3, set(), (0, 0), (2, 2), costs)
        total = sum(costs[y][x] for x, y in result["path"][1:])
        self.assertEqual(result["cost"], total)

    def test_start_equals_goal_with_costs(self):
        costs = [[7]]
        result = plan(1, 1, set(), (0, 0), (0, 0), costs)
        self.assertEqual(result, {"path": [(0, 0)], "cost": 0, "expanded": 1})

    def test_unreachable_with_costs(self):
        blocked = {(1, y) for y in range(3)}
        costs = [[2] * 3 for _ in range(3)]
        result = plan(3, 3, blocked, (0, 0), (2, 2), costs)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"], 3)

    def test_costs_do_not_change_obstacle_semantics(self):
        costs = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        with self.assertRaises(ValueError):
            plan(3, 3, {(0, 0)}, (0, 0), (2, 2), costs)
        with self.assertRaises(ValueError):
            plan(3, 3, {(2, 2)}, (0, 0), (2, 2), costs)
        deduped = plan(3, 3, {(1, 1)}, (0, 0), (2, 2), costs)
        duplicated = plan(3, 3, [(1, 1), (1, 1)], (0, 0), (2, 2), costs)
        self.assertEqual(deduped, duplicated)

    def test_costs_type_errors(self):
        for bad in (42, "abc", b"abc", 1.5):
            with self.assertRaises(TypeError, msg=f"costs={bad!r}"):
                plan(2, 2, set(), (0, 0), (1, 1), bad)
        # Rows must be sequences, not strings/bytes/ints.
        for bad_row in ("ab", b"ab", 3, None):
            with self.assertRaises(TypeError, msg=f"row={bad_row!r}"):
                plan(2, 2, set(), (0, 0), (1, 1), [[1, 1], bad_row])
        # Cells must be non-bool ints.
        for bad_cell in (True, False, 1.5, "1", None):
            with self.assertRaises(TypeError, msg=f"cell={bad_cell!r}"):
                plan(2, 2, set(), (0, 0), (1, 1),
                     [[1, 1], [1, bad_cell]])

    def test_costs_value_errors(self):
        # Wrong row count.
        with self.assertRaises(ValueError):
            plan(2, 2, set(), (0, 0), (1, 1), [[1, 1]])
        # Wrong row length.
        with self.assertRaises(ValueError):
            plan(2, 2, set(), (0, 0), (1, 1), [[1, 1], [1, 1, 1]])
        # Non-positive costs.
        for bad_cell in (0, -1, -100):
            with self.assertRaises(ValueError, msg=f"cell={bad_cell!r}"):
                plan(2, 2, set(), (0, 0), (1, 1),
                     [[1, 1], [1, bad_cell]])

    def test_costs_validated_before_search(self):
        # Invalid costs must raise even on an otherwise trivial query.
        with self.assertRaises(ValueError):
            plan(1, 1, set(), (0, 0), (0, 0), [[0]])
        with self.assertRaises(TypeError):
            plan(1, 1, set(), (0, 0), (0, 0), "nope")

    def test_result_keys_unchanged_with_costs(self):
        result = plan(2, 2, set(), (0, 0), (1, 1), [[1, 2], [3, 4]])
        self.assertEqual(set(result), {"path", "cost", "expanded"})


class ValidationTest(unittest.TestCase):
    def test_dimension_type_errors(self):
        for bad in ("3", 3.0, None, True, [3]):
            with self.assertRaises(TypeError, msg=f"width={bad!r}"):
                plan(bad, 3, set(), (0, 0), (1, 1))
            with self.assertRaises(TypeError, msg=f"height={bad!r}"):
                plan(3, bad, set(), (0, 0), (1, 1))

    def test_dimension_value_errors(self):
        for bad in (0, -1, -100):
            with self.assertRaises(ValueError):
                plan(bad, 3, set(), (0, 0), (1, 1))
            with self.assertRaises(ValueError):
                plan(3, bad, set(), (0, 0), (1, 1))

    def test_point_type_errors(self):
        for bad in (1, "ab", (1,), (1, 2, 3), (1.5, 2), (1, None),
                    (True, 0), None, {"x": 1, "y": 2}):
            with self.assertRaises(TypeError, msg=f"start={bad!r}"):
                plan(3, 3, set(), bad, (1, 1))
            with self.assertRaises(TypeError, msg=f"goal={bad!r}"):
                plan(3, 3, set(), (0, 0), bad)

    def test_point_value_errors(self):
        for bad in ((-1, 0), (0, -1), (3, 0), (0, 3), (10, 10)):
            with self.assertRaises(ValueError, msg=f"start={bad!r}"):
                plan(3, 3, set(), bad, (1, 1))
            with self.assertRaises(ValueError, msg=f"goal={bad!r}"):
                plan(3, 3, set(), (0, 0), bad)

    def test_blocked_type_errors(self):
        with self.assertRaises(TypeError):
            plan(3, 3, 42, (0, 0), (1, 1))
        with self.assertRaises(TypeError):
            plan(3, 3, "ab", (0, 0), (1, 1))
        with self.assertRaises(TypeError):
            plan(3, 3, [(1, 1), "bad"], (0, 0), (2, 2))
        with self.assertRaises(TypeError):
            plan(3, 3, [(1, 1), (2.5, 0)], (0, 0), (2, 2))

    def test_blocked_value_errors(self):
        for bad_cell in ((-1, 0), (0, 3), (3, 3)):
            with self.assertRaises(ValueError, msg=f"cell={bad_cell!r}"):
                plan(3, 3, {(1, 1), bad_cell}, (0, 0), (2, 2))

    def test_endpoint_on_obstacle(self):
        with self.assertRaises(ValueError):
            plan(3, 3, {(0, 0)}, (0, 0), (2, 2))
        with self.assertRaises(ValueError):
            plan(3, 3, {(2, 2)}, (0, 0), (2, 2))

    def test_validation_before_search(self):
        # Invalid goal type must raise even if the search would never run.
        with self.assertRaises(TypeError):
            plan(1, 1, set(), (0, 0), "nope")


if __name__ == '__main__':
    unittest.main()

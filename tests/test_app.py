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


class TraceTest(unittest.TestCase):
    def test_trace_disabled_matches_default(self):
        kwargs = dict(width=6, height=4,
                      blocked={(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)},
                      start=(0, 0), goal=(5, 3))
        default = plan(**kwargs)
        explicit = plan(trace=False, **kwargs)
        self.assertEqual(default, explicit)
        self.assertEqual(set(default), {"path", "cost", "expanded"})

    def test_trace_success(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        result = plan(6, 4, blocked, (0, 0), (5, 3), trace=True)
        self.assertEqual(set(result),
                         {"path", "cost", "expanded", "expanded_nodes"})
        nodes = result["expanded_nodes"]
        self.assertEqual(result["expanded"], len(nodes))
        self.assertEqual(nodes[0], (0, 0))
        self.assertEqual(nodes[-1], (5, 3))
        self.assertEqual(len(set(nodes)), len(nodes))  # no duplicates
        self.assertTrue(set(nodes).isdisjoint(blocked))
        self.assertTrue(all(isinstance(p, tuple) for p in nodes))

    def test_trace_unreachable(self):
        blocked = {(1, y) for y in range(3)}
        result = plan(3, 3, blocked, (0, 0), (2, 2), trace=True)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))
        self.assertEqual(result["expanded_nodes"],
                         [(0, 0), (0, 1), (0, 2)])

    def test_trace_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), trace=True)
        self.assertEqual(result, {
            "path": [(1, 1)],
            "cost": 0,
            "expanded": 1,
            "expanded_nodes": [(1, 1)],
        })

    def test_trace_expansion_order(self):
        # Straight 4x1 corridor: every node closes in coordinate order.
        result = plan(4, 1, set(), (0, 0), (3, 0), trace=True)
        self.assertEqual(result["expanded_nodes"],
                         [(0, 0), (1, 0), (2, 0), (3, 0)])
        # 2x3 open grid: A* runs down the left column, then the goal closes
        # with h=0 before the remaining stale-priority nodes.
        result = plan(2, 3, set(), (0, 0), (1, 2), trace=True)
        self.assertEqual(result["expanded_nodes"],
                         [(0, 0), (0, 1), (0, 2), (1, 2)])

    def test_trace_independent_of_blocked_order(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        forward = plan(5, 3, cells, (0, 0), (4, 2), trace=True)
        reverse = plan(5, 3, list(reversed(cells)), (0, 0), (4, 2),
                       trace=True)
        repeated = plan(5, 3, cells + [cells[0]] * 3, (0, 0), (4, 2),
                        trace=True)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, repeated)

    def test_trace_with_costs(self):
        costs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = plan(3, 3, set(), (0, 0), (2, 2), costs=costs, trace=True)
        nodes = result["expanded_nodes"]
        self.assertEqual(nodes[0], (0, 0))
        self.assertEqual(nodes[-1], (2, 2))
        self.assertEqual(result["cost"],
                         sum(costs[y][x] for x, y in result["path"][1:]))

    def test_trace_must_be_bool(self):
        for bad in (1, 0, "true", None, [], 1.0):
            with self.assertRaises(TypeError, msg=f"trace={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), trace=bad)

    def test_existing_errors_take_precedence_over_trace(self):
        with self.assertRaises(TypeError):  # bad width, not bad trace
            plan("3", 3, set(), (0, 0), (1, 1), trace="x")
        with self.assertRaises(ValueError):  # zero height, not bad trace
            plan(3, 0, set(), (0, 0), (1, 1), trace="x")
        with self.assertRaises(ValueError):  # endpoint on blocked cell
            plan(3, 3, {(2, 2)}, (0, 0), (2, 2), trace=1)


if __name__ == '__main__':
    unittest.main()

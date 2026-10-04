import json
import unittest

import app
from app import distance_field, plan, replay


# Cheapest route to (2, 0) runs around the expensive middle column.
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]


class DistanceFieldUnitCostTest(unittest.TestCase):
    def test_open_grid_is_manhattan(self):
        result = distance_field(3, 3, set(), (2, 2))
        self.assertEqual(result, {
            "distances": [[4, 3, 2], [3, 2, 1], [2, 1, 0]],
            "expanded": 9,
        })

    def test_obstacles_and_unreachable_are_none(self):
        # A full wall splits the grid; the right column is unreachable.
        blocked = {(1, 0), (1, 1), (1, 2)}
        result = distance_field(3, 3, blocked, (0, 0))
        self.assertEqual(result["distances"],
                         [[0, None, None],
                          [1, None, None],
                          [2, None, None]])
        self.assertEqual(result["expanded"], 3)

    def test_goal_is_the_only_reachable_cell(self):
        # Every neighbor of the goal is blocked, so nothing else reaches it.
        blocked = {(0, 1), (2, 1), (1, 0), (1, 2)}
        result = distance_field(3, 3, blocked, (1, 1), trace=True)
        self.assertEqual(result["distances"],
                         [[None, None, None],
                          [None, 0, None],
                          [None, None, None]])
        self.assertEqual(result["expanded"], 1)
        self.assertEqual(result["expanded_nodes"], [(1, 1)])

    def test_matrix_shape_matches_grid(self):
        result = distance_field(5, 2, {(2, 0)}, (4, 1))
        self.assertEqual(len(result["distances"]), 2)
        for row in result["distances"]:
            self.assertEqual(len(row), 5)

    def test_result_is_json_serializable(self):
        result = distance_field(3, 3, {(1, 1)}, (2, 2), trace=True)
        self.assertEqual(json.loads(json.dumps(result)),
                         json.loads(json.dumps(result)))
        # Tuples survive the round trip as lists; compare structure.
        decoded = json.loads(json.dumps(result))
        self.assertEqual(decoded["distances"], result["distances"])
        self.assertEqual(decoded["expanded"], result["expanded"])
        self.assertEqual(decoded["expanded_nodes"],
                         [list(p) for p in result["expanded_nodes"]])


class DistanceFieldWeightedTest(unittest.TestCase):
    def test_weighted_field(self):
        result = distance_field(3, 3, set(), (2, 0), costs=COSTS)
        self.assertEqual(result["distances"],
                         [[6, 1, 0],
                          [5, 2, 1],
                          [4, 3, 2]])
        self.assertEqual(result["expanded"], 9)

    def test_distances_match_plan_costs(self):
        # Every reachable cell's distance equals the cost ``plan``
        # reports from that cell to the goal.
        blocked = {(2, 0), (2, 1), (2, 2)}
        goal = (5, 3)
        result = distance_field(6, 4, blocked, goal)
        for y in range(4):
            for x in range(6):
                expected = result["distances"][y][x]
                if (x, y) in blocked:
                    self.assertIsNone(expected)
                    continue
                found = plan(6, 4, blocked, (x, y), goal)
                self.assertEqual(expected, found["cost"],
                                 msg=f"start=({x}, {y})")

    def test_weighted_distances_match_plan_and_replay(self):
        goal = (2, 0)
        result = distance_field(3, 3, set(), goal, costs=COSTS)
        for y in range(3):
            for x in range(3):
                found = plan(3, 3, set(), (x, y), goal, costs=COSTS)
                self.assertEqual(result["distances"][y][x], found["cost"],
                                 msg=f"start=({x}, {y})")
                checked = replay(3, 3, set(), (x, y), goal, found["path"],
                                 costs=COSTS)
                self.assertTrue(checked["valid"])
                self.assertEqual(checked["cost"], found["cost"])

    def test_distances_never_negative(self):
        result = distance_field(4, 4, {(1, 1), (2, 2)}, (3, 3),
                                costs=[[1, 2, 3, 4]] * 4)
        for row in result["distances"]:
            for value in row:
                if value is not None:
                    self.assertGreaterEqual(value, 0)


class DistanceFieldTraceTest(unittest.TestCase):
    def test_trace_omitted_by_default(self):
        self.assertNotIn("expanded_nodes",
                         distance_field(3, 3, set(), (2, 2)))
        self.assertNotIn("expanded_nodes",
                         distance_field(3, 3, set(), (2, 2), trace=False))

    def test_closing_order_is_distance_then_coordinate(self):
        result = distance_field(3, 3, set(), (2, 2), trace=True)
        self.assertEqual(result["expanded_nodes"],
                         [(2, 2),
                          (1, 2), (2, 1),
                          (0, 2), (1, 1), (2, 0),
                          (0, 1), (1, 0),
                          (0, 0)])
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_trace_matches_distances(self):
        # Closing order is stable, goal first, and every closed cell is
        # exactly the reachable set recorded in the matrix.
        blocked = {(2, 0), (2, 1), (2, 2)}
        result = distance_field(6, 4, blocked, (5, 3), trace=True)
        nodes = result["expanded_nodes"]
        self.assertEqual(nodes[0], (5, 3))
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(result["expanded"], len(nodes))
        reachable = {(x, y) for y in range(4) for x in range(6)
                     if result["distances"][y][x] is not None}
        self.assertEqual(set(nodes), reachable)
        # Distances are closed in non-decreasing order.
        dists = [result["distances"][y][x] for x, y in nodes]
        self.assertEqual(dists, sorted(dists))

    def test_trace_is_deterministic(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        first = distance_field(6, 4, blocked, (5, 3), trace=True)
        for _ in range(10):
            self.assertEqual(
                distance_field(6, 4, blocked, (5, 3), trace=True), first)


class DistanceFieldOrderIndependenceTest(unittest.TestCase):
    def test_duplicate_obstacles_merge(self):
        deduped = distance_field(4, 4, {(1, 1), (2, 2)}, (3, 3), trace=True)
        duplicated = distance_field(
            4, 4, [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1)], (3, 3),
            trace=True)
        self.assertEqual(deduped, duplicated)

    def test_obstacle_order_does_not_matter(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        forward = distance_field(5, 3, cells, (4, 2), trace=True)
        reverse = distance_field(5, 3, list(reversed(cells)), (4, 2),
                                 trace=True)
        shuffled = distance_field(5, 3, [cells[2], cells[0], cells[3],
                                         cells[1]], (4, 2), trace=True)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, shuffled)

    def test_costs_row_input_copies_do_not_matter(self):
        rows = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        as_lists = distance_field(3, 3, set(), (2, 0), costs=rows,
                                  trace=True)
        as_tuples = distance_field(3, 3, set(), (2, 0),
                                   costs=[tuple(r) for r in rows],
                                   trace=True)
        self.assertEqual(as_lists, as_tuples)


class DistanceFieldValidationTest(unittest.TestCase):
    def test_dimension_errors(self):
        for bad in ("3", 3.0, None, True, [3]):
            with self.assertRaises(TypeError, msg=f"width={bad!r}"):
                distance_field(bad, 3, set(), (1, 1))
            with self.assertRaises(TypeError, msg=f"height={bad!r}"):
                distance_field(3, bad, set(), (1, 1))
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                distance_field(bad, 3, set(), (1, 1))
            with self.assertRaises(ValueError):
                distance_field(3, bad, set(), (1, 1))

    def test_goal_type_errors(self):
        for bad in (1, "ab", (1,), (1, 2, 3), (1.5, 2), (1, None),
                    (True, 0), None):
            with self.assertRaises(TypeError, msg=f"goal={bad!r}"):
                distance_field(3, 3, set(), bad)

    def test_goal_value_errors(self):
        for bad in ((-1, 0), (0, -1), (3, 0), (0, 3), (10, 10)):
            with self.assertRaises(ValueError, msg=f"goal={bad!r}"):
                distance_field(3, 3, set(), bad)

    def test_goal_on_obstacle(self):
        with self.assertRaises(ValueError):
            distance_field(3, 3, {(2, 2)}, (2, 2))

    def test_blocked_errors(self):
        with self.assertRaises(TypeError):
            distance_field(3, 3, 42, (1, 1))
        with self.assertRaises(TypeError):
            distance_field(3, 3, "ab", (1, 1))
        with self.assertRaises(TypeError):
            distance_field(3, 3, [(1, 1), "bad"], (2, 2))
        for bad_cell in ((-1, 0), (0, 3), (3, 3)):
            with self.assertRaises(ValueError, msg=f"cell={bad_cell!r}"):
                distance_field(3, 3, {(1, 1), bad_cell}, (2, 2))

    def test_costs_errors(self):
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (1, 1), costs="bad")
        with self.assertRaises(ValueError):
            distance_field(3, 3, set(), (1, 1), costs=[[1, 1, 1]])
        with self.assertRaises(ValueError):
            distance_field(3, 3, set(), (1, 1),
                           costs=[[1, 1, 1], [1, 1], [1, 1, 1]])
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (1, 1),
                           costs=[[1, 1, 1], [1, "x", 1], [1, 1, 1]])
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (1, 1),
                           costs=[[1, 1, 1], [1, True, 1], [1, 1, 1]])
        with self.assertRaises(ValueError):
            distance_field(3, 3, set(), (1, 1),
                           costs=[[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def test_trace_must_be_bool(self):
        for bad in (0, 1, "true", None, []):
            with self.assertRaises(TypeError, msg=f"trace={bad!r}"):
                distance_field(3, 3, set(), (1, 1), trace=bad)

    def test_no_extra_parameters(self):
        # The entry point accepts no start, dynamic obstacles, budget,
        # cost limit or snapshot arguments.
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (1, 1), dynamic_blocked=[])
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (1, 1), max_expanded=10)
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (1, 1), snapshot=True)
        with self.assertRaises(TypeError):
            distance_field(3, 3, set(), (1, 1), max_cost=10)


class PublicSurfaceTest(unittest.TestCase):
    def test_distance_field_is_exported(self):
        self.assertIn("distance_field", app.__all__)
        self.assertIs(app.distance_field, distance_field)

    def test_existing_entry_points_unchanged(self):
        self.assertEqual(
            plan(3, 3, set(), (0, 0), (2, 2)),
            {"path": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
             "cost": 4, "expanded": 5})
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]),
            {"valid": True, "cost": 4, "steps": 4})


if __name__ == "__main__":
    unittest.main()

import json
import unittest

import app
from app import distance_field, distance_field_any, plan


# Cheapest routes run around the expensive middle column.
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]


class DistanceFieldAnyUnitCostTest(unittest.TestCase):
    def test_open_grid_two_goals(self):
        result = distance_field_any(3, 3, set(), [(0, 0), (2, 2)])
        self.assertEqual(result, {
            "distances": [[0, 1, 2], [1, 2, 1], [2, 1, 0]],
            "targets": [[(0, 0), (0, 0), (0, 0)],
                        [(0, 0), (0, 0), (2, 2)],
                        [(0, 0), (2, 2), (2, 2)]],
            "expanded": 9,
        })

    def test_equal_distance_picks_lexicographically_smaller_goal(self):
        # (1, 0) is exactly one step from both goals; (0, 0) sorts first.
        result = distance_field_any(3, 1, set(), [(2, 0), (0, 0)])
        self.assertEqual(result["distances"], [[0, 1, 0]])
        self.assertEqual(result["targets"], [[(0, 0), (0, 0), (2, 0)]])

    def test_obstacles_and_unreachable_are_none(self):
        # A full wall splits the grid; the right column reaches no goal.
        blocked = {(1, 0), (1, 1), (1, 2)}
        result = distance_field_any(3, 3, blocked, [(0, 0), (0, 2)])
        self.assertEqual(result["distances"],
                         [[0, None, None],
                          [1, None, None],
                          [0, None, None]])
        self.assertEqual(result["targets"],
                         [[(0, 0), None, None],
                          [(0, 0), None, None],
                          [(0, 2), None, None]])
        self.assertEqual(result["expanded"], 3)

    def test_single_goal_matches_distance_field(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        single = distance_field(6, 4, blocked, (5, 3), trace=True)
        multi = distance_field_any(6, 4, blocked, [(5, 3)], trace=True)
        self.assertEqual(multi["distances"], single["distances"])
        self.assertEqual(multi["expanded"], single["expanded"])
        self.assertEqual(multi["expanded_nodes"], single["expanded_nodes"])
        for y in range(4):
            for x in range(6):
                expected = (5, 3) if multi["distances"][y][x] is not None \
                    else None
                self.assertEqual(multi["targets"][y][x], expected)

    def test_matrix_shapes_match_grid(self):
        result = distance_field_any(5, 2, {(2, 0)}, [(4, 1), (0, 0)])
        for matrix in (result["distances"], result["targets"]):
            self.assertEqual(len(matrix), 2)
            for row in matrix:
                self.assertEqual(len(row), 5)

    def test_result_is_json_serializable(self):
        result = distance_field_any(3, 3, {(1, 1)}, [(2, 2), (0, 0)],
                                    trace=True)
        decoded = json.loads(json.dumps(result))
        self.assertEqual(decoded["distances"], result["distances"])
        self.assertEqual(decoded["targets"],
                         [[list(p) if p is not None else None for p in row]
                          for row in result["targets"]])
        self.assertEqual(decoded["expanded"], result["expanded"])
        self.assertEqual(decoded["expanded_nodes"],
                         [list(p) for p in result["expanded_nodes"]])


class DistanceFieldAnyWeightedTest(unittest.TestCase):
    def test_weighted_field(self):
        result = distance_field_any(3, 3, set(), [(2, 0), (0, 2)],
                                    costs=COSTS)
        self.assertEqual(result["distances"],
                         [[2, 1, 0],
                          [1, 2, 1],
                          [0, 1, 2]])
        # (1, 1) and (2, 2) tie at distance 2; (0, 2) sorts before (2, 0).
        self.assertEqual(result["targets"],
                         [[(0, 2), (2, 0), (2, 0)],
                          [(0, 2), (0, 2), (2, 0)],
                          [(0, 2), (0, 2), (0, 2)]])
        self.assertEqual(result["expanded"], 9)

    def test_distances_match_plan_costs(self):
        # Every reachable cell's distance equals the cost ``plan``
        # reports from that cell to its recorded target, and no goal
        # offers a strictly cheaper route.
        blocked = {(2, 0), (2, 1), (2, 2)}
        goals = [(5, 3), (0, 0)]
        result = distance_field_any(6, 4, blocked, goals)
        for y in range(4):
            for x in range(6):
                expected = result["distances"][y][x]
                if (x, y) in blocked:
                    self.assertIsNone(expected)
                    self.assertIsNone(result["targets"][y][x])
                    continue
                target = result["targets"][y][x]
                self.assertIn(target, goals)
                found = plan(6, 4, blocked, (x, y), target)
                self.assertEqual(expected, found["cost"],
                                 msg=f"start=({x}, {y})")
                for goal in goals:
                    other = plan(6, 4, blocked, (x, y), goal)["cost"]
                    self.assertGreaterEqual(other, expected)

    def test_weighted_distances_match_plan(self):
        goals = [(2, 0), (0, 2)]
        result = distance_field_any(3, 3, set(), goals, costs=COSTS)
        for y in range(3):
            for x in range(3):
                found = plan(3, 3, set(), (x, y),
                             result["targets"][y][x], costs=COSTS)
                self.assertEqual(result["distances"][y][x], found["cost"],
                                 msg=f"start=({x}, {y})")


class DistanceFieldAnyTraceTest(unittest.TestCase):
    def test_trace_omitted_by_default(self):
        self.assertNotIn("expanded_nodes",
                         distance_field_any(3, 3, set(), [(2, 2)]))
        self.assertNotIn("expanded_nodes",
                         distance_field_any(3, 3, set(), [(2, 2)],
                                            trace=False))

    def test_closing_order_is_distance_then_coordinate(self):
        result = distance_field_any(3, 3, set(), [(0, 0), (2, 2)],
                                    trace=True)
        self.assertEqual(result["expanded_nodes"],
                         [(0, 0), (2, 2),
                          (0, 1), (1, 0), (1, 2), (2, 1),
                          (0, 2), (1, 1), (2, 0)])
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_trace_matches_matrices(self):
        # Closing order is stable, every goal is closed, and the closed
        # cells are exactly the reachable set recorded in the matrices.
        blocked = {(2, 0), (2, 1), (2, 2)}
        goals = [(5, 3), (0, 0)]
        result = distance_field_any(6, 4, blocked, goals, trace=True)
        nodes = result["expanded_nodes"]
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(result["expanded"], len(nodes))
        for goal in goals:
            self.assertIn(goal, nodes)
        reachable = {(x, y) for y in range(4) for x in range(6)
                     if result["distances"][y][x] is not None}
        self.assertEqual(set(nodes), reachable)
        # Distances are closed in non-decreasing order.
        dists = [result["distances"][y][x] for x, y in nodes]
        self.assertEqual(dists, sorted(dists))

    def test_trace_is_deterministic(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        first = distance_field_any(6, 4, blocked, [(5, 3), (0, 0)],
                                   trace=True)
        for _ in range(10):
            self.assertEqual(
                distance_field_any(6, 4, blocked, [(5, 3), (0, 0)],
                                   trace=True), first)


class DistanceFieldAnyOrderIndependenceTest(unittest.TestCase):
    def test_duplicate_goals_merge(self):
        deduped = distance_field_any(4, 4, set(), [(3, 3), (0, 0)],
                                     trace=True)
        duplicated = distance_field_any(
            4, 4, set(), [(3, 3), (0, 0), (3, 3), (0, 0), (3, 3)],
            trace=True)
        self.assertEqual(deduped, duplicated)

    def test_goal_order_does_not_matter(self):
        goals = [(4, 2), (0, 0), (2, 2)]
        forward = distance_field_any(5, 3, {(1, 1)}, goals, trace=True)
        reverse = distance_field_any(5, 3, {(1, 1)},
                                     list(reversed(goals)), trace=True)
        shuffled = distance_field_any(5, 3, {(1, 1)},
                                      [goals[1], goals[2], goals[0]],
                                      trace=True)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, shuffled)

    def test_obstacle_order_does_not_matter(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        goals = [(4, 2), (0, 0)]
        forward = distance_field_any(5, 3, cells, goals, trace=True)
        reverse = distance_field_any(5, 3, list(reversed(cells)), goals,
                                     trace=True)
        self.assertEqual(forward, reverse)


class DistanceFieldAnyValidationTest(unittest.TestCase):
    def test_dimension_errors(self):
        for bad in ("3", 3.0, None, True, [3]):
            with self.assertRaises(TypeError, msg=f"width={bad!r}"):
                distance_field_any(bad, 3, set(), [(1, 1)])
            with self.assertRaises(TypeError, msg=f"height={bad!r}"):
                distance_field_any(3, bad, set(), [(1, 1)])
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                distance_field_any(bad, 3, set(), [(1, 1)])
            with self.assertRaises(ValueError):
                distance_field_any(3, bad, set(), [(1, 1)])

    def test_goals_outer_type_errors(self):
        for bad in (None, "ab", b"ab", 42, (1, 2)):
            with self.assertRaises(TypeError, msg=f"goals={bad!r}"):
                distance_field_any(3, 3, set(), bad)

    def test_goals_empty_is_value_error(self):
        for empty in ([], ()):
            with self.assertRaises(ValueError, msg=f"goals={empty!r}"):
                distance_field_any(3, 3, set(), empty)

    def test_goal_element_type_errors(self):
        for bad in (1, "ab", (1,), (1, 2, 3), (1.5, 2), (1, None),
                    (True, 0), None):
            with self.assertRaises(TypeError, msg=f"goal={bad!r}"):
                distance_field_any(3, 3, set(), [(1, 1), bad])

    def test_goal_value_errors(self):
        for bad in ((-1, 0), (0, -1), (3, 0), (0, 3), (10, 10)):
            with self.assertRaises(ValueError, msg=f"goal={bad!r}"):
                distance_field_any(3, 3, set(), [(1, 1), bad])

    def test_goal_on_obstacle(self):
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, {(2, 2)}, [(1, 1), (2, 2)])

    def test_blocked_errors(self):
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, 42, [(1, 1)])
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, "ab", [(1, 1)])
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, [(1, 1), "bad"], [(2, 2)])
        for bad_cell in ((-1, 0), (0, 3), (3, 3)):
            with self.assertRaises(ValueError, msg=f"cell={bad_cell!r}"):
                distance_field_any(3, 3, {(1, 1), bad_cell}, [(2, 2)])

    def test_costs_errors(self):
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], costs="bad")
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, set(), [(1, 1)], costs=[[1, 1, 1]])
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)],
                               costs=[[1, 1, 1], [1, "x", 1], [1, 1, 1]])
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)],
                               costs=[[1, 1, 1], [1, True, 1], [1, 1, 1]])
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, set(), [(1, 1)],
                               costs=[[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def test_trace_must_be_bool(self):
        for bad in (0, 1, "true", None, []):
            with self.assertRaises(TypeError, msg=f"trace={bad!r}"):
                distance_field_any(3, 3, set(), [(1, 1)], trace=bad)

    def test_no_extra_parameters(self):
        # The entry point accepts no start, dynamic obstacles, budget,
        # cost limit or snapshot arguments.
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], start=(0, 0))
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], dynamic_blocked=[])
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], max_expanded=10)
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], snapshot=True)
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], max_cost=10)


class PublicSurfaceTest(unittest.TestCase):
    def test_distance_field_any_is_exported(self):
        self.assertIn("distance_field_any", app.__all__)
        self.assertIs(app.distance_field_any, distance_field_any)

    def test_existing_entry_points_unchanged(self):
        self.assertEqual(
            distance_field(3, 3, set(), (2, 2)),
            {"distances": [[4, 3, 2], [3, 2, 1], [2, 1, 0]],
             "expanded": 9})
        self.assertEqual(
            plan(3, 3, set(), (0, 0), (2, 2)),
            {"path": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
             "cost": 4, "expanded": 5})


if __name__ == "__main__":
    unittest.main()

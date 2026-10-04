import itertools
import json
import unittest

import app
from app import (distance_field, distance_field_any, plan, plan_any,
                 replay)


# Cheapest routes around the expensive middle column (same matrix the
# single-goal distance-field tests use).
COSTS = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]


class DistanceFieldAnyBasicTest(unittest.TestCase):
    def test_open_grid_minimum_field_and_targets(self):
        # Goals (2, 0) and (0, 2) on an open 3x3 with unit costs.
        result = distance_field_any(3, 3, set(), [(2, 0), (0, 2)])
        self.assertEqual(set(result), {"distances", "targets", "expanded"})
        self.assertEqual(result["distances"],
                         [[2, 1, 0], [1, 2, 1], [0, 1, 2]])
        self.assertEqual(result["targets"],
                         [[(0, 2), (2, 0), (2, 0)],
                          [(0, 2), (0, 2), (2, 0)],
                          [(0, 2), (0, 2), (0, 2)]])
        self.assertEqual(result["expanded"], 9)

    def test_goals_are_distance_zero_and_own_themselves(self):
        goals = [(2, 0), (0, 2)]
        result = distance_field_any(3, 3, set(), goals)
        for goal in goals:
            self.assertEqual(result["distances"][goal[1]][goal[0]], 0)
            self.assertEqual(result["targets"][goal[1]][goal[0]], goal)

    def test_obstacles_and_unreachable_are_none_everywhere(self):
        # A full wall seals the goals on the right; the left side cannot
        # reach any endpoint, so both matrices carry None there.
        blocked = {(1, 0), (1, 1), (1, 2)}
        result = distance_field_any(3, 3, blocked, [(2, 0), (2, 2)])
        self.assertEqual(result["distances"],
                         [[None, None, 0],
                          [None, None, 1],
                          [None, None, 0]])
        self.assertEqual(result["targets"],
                         [[None, None, (2, 0)],
                          [None, None, (2, 0)],
                          [None, None, (2, 2)]])
        self.assertEqual(result["expanded"], 3)

    def test_targets_is_none_exactly_where_distances_is_none(self):
        for blocked in (set(), {(2, 0), (2, 1), (2, 2)},
                        {(1, 1), (0, 2), (4, 1)}):
            result = distance_field_any(5, 3, blocked,
                                        [(4, 2), (0, 0), (4, 0)])
            for y in range(3):
                for x in range(5):
                    distance = result["distances"][y][x]
                    target = result["targets"][y][x]
                    if distance is None:
                        self.assertIsNone(target, (x, y))
                    else:
                        self.assertIn(target, {(4, 2), (0, 0), (4, 0)})

    def test_matrix_shapes_match_grid(self):
        result = distance_field_any(6, 2, {(2, 0)}, [(5, 1), (0, 0)])
        self.assertEqual(len(result["distances"]), 2)
        self.assertEqual(len(result["targets"]), 2)
        for row in result["targets"]:
            self.assertEqual(len(row), 6)
        for row in result["distances"]:
            self.assertEqual(len(row), 6)

    def test_single_goal_matches_distance_field_bit_for_bit(self):
        costs_4x3 = [[1, 10, 1, 1], [1, 10, 1, 1], [1, 1, 1, 1]]
        for kwargs in ({}, {"trace": True}, {"costs": costs_4x3},
                       {"costs": costs_4x3, "trace": True}):
            one = distance_field(4, 3, {(1, 1)}, (3, 2), **kwargs)
            anyed = distance_field_any(4, 3, {(1, 1)}, [(3, 2)], **kwargs)
            self.assertEqual(anyed["distances"], one["distances"], kwargs)
            self.assertEqual(anyed["expanded"], one["expanded"], kwargs)
            if "expanded_nodes" in one:
                self.assertEqual(anyed["expanded_nodes"],
                                 one["expanded_nodes"], kwargs)
            # Every reachable cell is owned by the sole goal.
            for y in range(3):
                for x in range(4):
                    if one["distances"][y][x] is not None:
                        self.assertEqual(anyed["targets"][y][x], (3, 2))
                    else:
                        self.assertIsNone(anyed["targets"][y][x])


class DistanceFieldAnyWeightedTest(unittest.TestCase):
    def test_weighted_field_uses_entering_cell_costs(self):
        # From (0, 0) the expensive middle column makes (0, 2) the
        # nearest endpoint at cost 2 (two cheap downward steps); reaching
        # (2, 0) directly costs 11.
        result = distance_field_any(3, 3, set(), [(2, 0), (0, 2)],
                                    costs=COSTS)
        self.assertEqual(result["distances"][0][0], 2)
        self.assertEqual(result["targets"][0][0], (0, 2))
        # (1, 2) and (2, 1) are one cheap step from the nearest goal.
        self.assertEqual(result["distances"][2][1], 1)
        self.assertEqual(result["targets"][2][1], (0, 2))
        self.assertEqual(result["distances"][1][2], 1)
        self.assertEqual(result["targets"][1][2], (2, 0))

    def test_distances_and_targets_match_plan_any(self):
        cases = [
            (4, 3, set(), None),
            (5, 2, {(2, 0), (2, 1)}, None),
            (3, 4, {(1, 1), (0, 2)}, None),
            (4, 3, set(),
             [[1 + (x * y % 3) for x in range(4)] for y in range(3)]),
            (5, 3, {(2, 1)},
             [[1, 5, 1, 1, 1], [1, 5, 1, 9, 1], [1, 1, 1, 1, 1]]),
        ]
        for width, height, blocked, costs in cases:
            goals = [(width - 1, 0), (0, height - 1), (width - 1, height - 1)]
            field = distance_field_any(width, height, blocked, goals,
                                       costs=costs)
            for y, x in itertools.product(range(height), range(width)):
                distance = field["distances"][y][x]
                if (x, y) in blocked:
                    self.assertIsNone(distance)
                    continue
                planned = plan_any(width, height, blocked, (x, y), goals,
                                   costs=costs)
                if distance is None:
                    self.assertIsNone(planned["path"], (x, y))
                    continue
                self.assertEqual(distance, planned["cost"], (x, y))
                self.assertEqual(field["targets"][y][x],
                                 planned["path"][-1], (x, y))

    def test_target_path_replays_with_the_field_cost(self):
        goals = [(4, 2), (0, 0), (4, 0)]
        blocked = {(2, 1), (0, 2)}
        costs = [[1, 1, 10, 1, 1], [1, 1, 10, 1, 1], [1, 1, 1, 1, 1]]
        field = distance_field_any(5, 3, blocked, goals, costs=costs)
        for y, x in itertools.product(range(3), range(5)):
            distance = field["distances"][y][x]
            if distance is None:
                continue
            target = field["targets"][y][x]
            planned = plan_any(5, 3, blocked, (x, y), goals, costs=costs)
            checked = replay(5, 3, blocked, (x, y), target,
                             planned["path"], costs=costs)
            self.assertTrue(checked["valid"], (x, y))
            self.assertEqual(checked["cost"], distance)


class DistanceFieldAnyTieBreakTest(unittest.TestCase):
    def test_equal_distance_prefers_smaller_goal_coordinate(self):
        # One-row grid: the middle cell is one step from both goals; the
        # lexicographically smaller endpoint wins it.
        result = distance_field_any(3, 1, set(), [(0, 0), (2, 0)])
        self.assertEqual(result["distances"], [[0, 1, 0]])
        self.assertEqual(result["targets"], [[(0, 0), (0, 0), (2, 0)]])

    def test_tie_break_is_goal_order_independent(self):
        forward = distance_field_any(3, 3, set(), [(2, 0), (0, 2)],
                                     costs=COSTS, trace=True)
        reverse = distance_field_any(3, 3, set(), [(0, 2), (2, 0)],
                                     costs=COSTS, trace=True)
        self.assertEqual(reverse, forward)

    def test_all_goals_equidistant_cells(self):
        # Center (1, 1) of a 3x3 is two steps from all four corners.
        goals = [(0, 0), (2, 0), (0, 2), (2, 2)]
        result = distance_field_any(3, 3, set(), goals)
        self.assertEqual(result["distances"][1][1], 2)
        self.assertEqual(result["targets"][1][1], (0, 0))


class DistanceFieldAnyTraceTest(unittest.TestCase):
    def test_trace_omitted_by_default_and_when_false(self):
        self.assertNotIn("expanded_nodes",
                         distance_field_any(3, 3, set(), [(2, 2), (0, 0)]))
        self.assertNotIn(
            "expanded_nodes",
            distance_field_any(3, 3, set(), [(2, 2), (0, 0)], trace=False))

    def test_closing_order_is_distance_then_coordinate(self):
        result = distance_field_any(3, 3, set(), [(2, 0), (0, 2)],
                                    trace=True)
        self.assertEqual(result["expanded_nodes"],
                         [(0, 2), (2, 0),
                          (0, 1), (1, 0), (1, 2), (2, 1),
                          (0, 0), (1, 1), (2, 2)])
        self.assertEqual(result["expanded"],
                         len(result["expanded_nodes"]))

    def test_trace_is_the_reachable_set_in_non_decreasing_distance(self):
        blocked = {(1, 0), (1, 1), (1, 2)}
        result = distance_field_any(3, 3, blocked, [(2, 0), (2, 2)],
                                    trace=True)
        nodes = result["expanded_nodes"]
        self.assertEqual(nodes, [(2, 0), (2, 2), (2, 1)])
        self.assertEqual(len(nodes), len(set(nodes)))
        reachable = {(x, y) for y in range(3) for x in range(3)
                     if result["distances"][y][x] is not None}
        self.assertEqual(set(nodes), reachable)
        dists = [result["distances"][y][x] for x, y in nodes]
        self.assertEqual(dists, sorted(dists))

    def test_trace_is_deterministic(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        first = distance_field_any(6, 4, blocked, [(5, 3), (5, 2)],
                                   trace=True)
        for _ in range(10):
            self.assertEqual(
                distance_field_any(6, 4, blocked, [(5, 3), (5, 2)],
                                   trace=True), first)


class DistanceFieldAnyOrderIndependenceTest(unittest.TestCase):
    def test_duplicate_goals_and_obstacles_merge(self):
        once = distance_field_any(5, 3, {(1, 1), (2, 2)},
                                  [(4, 2), (4, 0), (0, 2)], trace=True)
        repeated = distance_field_any(
            5, 3, [(1, 1), (2, 2), (1, 1), (2, 2)],
            [(4, 2), (4, 0), (0, 2), (4, 0), (0, 2)], trace=True)
        self.assertEqual(repeated, once)

    def test_input_permutations_do_not_matter(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        goals = [(4, 2), (4, 0), (0, 2)]
        orderings = (
            (cells, goals),
            (list(reversed(cells)), list(reversed(goals))),
            ([cells[2], cells[0], cells[3], cells[1]],
             [goals[2], goals[0], goals[1], goals[0]]),
        )
        results = [distance_field_any(5, 3, blocked, gl, trace=True)
                   for blocked, gl in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_result_is_json_serializable(self):
        result = distance_field_any(3, 3, {(1, 1)}, [(2, 2), (0, 0)],
                                    trace=True)
        encoded = json.dumps(result)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["distances"], result["distances"])
        self.assertEqual(decoded["targets"],
                         [[list(cell) if cell is not None else None
                           for cell in row]
                          for row in result["targets"]])
        self.assertEqual(decoded["expanded"], result["expanded"])
        self.assertEqual(decoded["expanded_nodes"],
                         [list(p) for p in result["expanded_nodes"]])


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
        for bad in (None, 42, 1.5, "ab", b"ab", {1, 2}, {"x": 1}, True):
            with self.assertRaises(TypeError, msg=f"goals={bad!r}"):
                distance_field_any(3, 3, set(), bad)

    def test_empty_goals_is_value_error(self):
        for bad in ((), [], tuple()):
            with self.assertRaises(ValueError, msg=f"goals={bad!r}"):
                distance_field_any(3, 3, set(), bad)

    def test_goals_element_type_errors(self):
        for bad in (1, "ab", (1,), (1, 2, 3), (1.5, 2), (1, None),
                    (True, 0), None, {"x": 1, "y": 2}):
            with self.assertRaises(TypeError, msg=f"goal={bad!r}"):
                distance_field_any(3, 3, set(), [(2, 2), bad])

    def test_goals_bounds_errors(self):
        for bad in ((-1, 0), (0, -1), (3, 0), (0, 3), (10, 10)):
            with self.assertRaises(ValueError, msg=f"goal={bad!r}"):
                distance_field_any(3, 3, set(), [(2, 2), bad])

    def test_goal_on_obstacle_is_value_error(self):
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
            distance_field_any(3, 3, set(), [(1, 1)],
                               costs=[[1, 1, 1]])
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, set(), [(1, 1)],
                               costs=[[1, 1, 1], [1, 1], [1, 1, 1]])
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

    def test_validation_order_mirrors_distance_field(self):
        # The width ValueError precedes anything goals-specific.
        with self.assertRaises(ValueError):
            distance_field_any(0, 3, set(), "bad")
        # Bad goals structure precedes blocked normalization, as goal does.
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, 42, [1])
        # Goal bounds precede the obstacle/costs/trace checks.
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, {(0, 0)}, [(9, 9)])
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, set(), [(9, 9)],
                               costs=[[1, 1], [1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, set(), [(9, 9)], trace=1)
        # A goal-on-obstacle ValueError precedes the trace TypeError.
        with self.assertRaises(ValueError):
            distance_field_any(3, 3, {(2, 2)}, [(2, 2)], trace=1)
        # A structurally bad goal is still a TypeError even when the
        # trace flag is also wrong: shared checks run first.
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [1], trace=1)

    def test_no_control_parameters(self):
        # The entry point accepts no start, dynamic obstacles, budget,
        # cost limit, snapshot or any other control argument.
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], start=(0, 0))
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], dynamic_blocked=[])
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], max_expanded=10)
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], max_cost=10)
        with self.assertRaises(TypeError):
            distance_field_any(3, 3, set(), [(1, 1)], snapshot=True)


class DistanceFieldAnyPublicSurfaceTest(unittest.TestCase):
    def test_distance_field_any_is_exported(self):
        self.assertIn("distance_field_any", app.__all__)
        self.assertIs(app.distance_field_any, distance_field_any)

    def test_existing_entry_points_unchanged(self):
        self.assertEqual(
            plan(3, 3, set(), (0, 0), (2, 2)),
            {"path": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
             "cost": 4, "expanded": 5})
        self.assertEqual(
            distance_field(3, 3, set(), (2, 2)),
            {"distances": [[4, 3, 2], [3, 2, 1], [2, 1, 0]],
             "expanded": 9})
        self.assertEqual(
            plan_any(3, 3, set(), (0, 0), [(2, 0), (0, 2)])["path"][-1],
            (0, 2))
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]),
            {"valid": True, "cost": 4, "steps": 4})


if __name__ == "__main__":
    unittest.main()

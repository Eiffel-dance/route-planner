import unittest

import app
from app import plan, replay


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
    def test_trace_omitted_or_false_has_no_extra_field(self):
        for kwargs in ({}, {"trace": False}):
            result = plan(6, 4, {(2, 0), (2, 1), (2, 2)},
                          (0, 0), (5, 3), **kwargs)
            self.assertEqual(set(result), {"path", "cost", "expanded"})

    def test_trace_false_matches_omitted_exactly(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        omitted = plan(6, 4, blocked, (0, 0), (5, 3))
        explicit = plan(6, 4, blocked, (0, 0), (5, 3), trace=False)
        self.assertEqual(explicit, omitted)
        # Unreachable case keeps the same shape too.
        wall = {(1, y) for y in range(3)}
        self.assertEqual(
            plan(3, 3, wall, (0, 0), (2, 2), trace=False),
            plan(3, 3, wall, (0, 0), (2, 2)),
        )

    def test_trace_success_records_expansion_order(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        result = plan(6, 4, blocked, (0, 0), (5, 3), trace=True)
        nodes = result["expanded_nodes"]
        self.assertEqual(set(result), {"path", "cost", "expanded",
                                       "expanded_nodes"})
        self.assertEqual(len(nodes), result["expanded"])
        self.assertEqual(nodes[0], (0, 0))   # start is first
        self.assertEqual(nodes[-1], (5, 3))  # goal is last on success
        self.assertEqual(len(set(nodes)), len(nodes))  # no repeats
        self.assertTrue(all(isinstance(p, tuple) and len(p) == 2 for p in nodes))
        self.assertTrue(all(not isinstance(v, bool)
                            for p in nodes for v in p))
        self.assertTrue(all(p not in blocked for p in nodes))
        self.assertTrue(all(0 <= x < 6 and 0 <= y < 4 for x, y in nodes))
        # Every recorded node is on the found path or a detour around the
        # wall; the path nodes appear in path order within the trace.
        path = result["path"]
        positions = [nodes.index(p) for p in path]
        self.assertEqual(positions, sorted(positions))

    def test_trace_unreachable_records_all_closed_passable_nodes(self):
        wall = {(1, y) for y in range(3)}
        result = plan(3, 3, wall, (0, 0), (2, 2), trace=True)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        nodes = result["expanded_nodes"]
        self.assertEqual(nodes[0], (0, 0))
        self.assertEqual(len(nodes), result["expanded"])
        self.assertEqual(len(nodes), 3)
        self.assertEqual(set(nodes), {(0, 0), (0, 1), (0, 2)})
        self.assertEqual(len(set(nodes)), len(nodes))
        self.assertTrue(all(p not in wall for p in nodes))

    def test_trace_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), trace=True)
        self.assertEqual(result["path"], [(1, 1)])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["expanded"], 1)
        self.assertEqual(result["expanded_nodes"], [(1, 1)])

    def test_trace_with_costs(self):
        costs = [
            [1, 10, 1],
            [1, 10, 1],
            [1, 1, 1],
        ]
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=costs, trace=True)
        nodes = result["expanded_nodes"]
        self.assertEqual(len(nodes), result["expanded"])
        self.assertEqual(nodes[0], (0, 0))
        self.assertEqual(nodes[-1], (2, 0))
        self.assertEqual(result["path"][-1], (2, 0))
        # Cheap detour around the expensive column.
        self.assertNotIn((1, 0), result["path"])

    def test_trace_deterministic_across_blocked_orders(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        orderings = (
            cells,
            list(reversed(cells)),
            [cells[2], cells[0], cells[3], cells[1]],
        )
        traces = [
            plan(5, 3, order, (0, 0), (4, 2), trace=True)
            for order in orderings
        ]
        for other in traces[1:]:
            self.assertEqual(other["expanded_nodes"],
                             traces[0]["expanded_nodes"])
            self.assertEqual(other, traces[0])

    def test_trace_replay_stability(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        first = plan(6, 4, blocked, (0, 0), (5, 3), trace=True)
        for _ in range(10):
            self.assertEqual(
                plan(6, 4, blocked, (0, 0), (5, 3), trace=True), first
            )

    def test_expanded_nodes_is_json_serializable(self):
        import json
        result = plan(6, 4, {(2, 0), (2, 1)}, (0, 0), (5, 3), trace=True)
        restored = json.loads(json.dumps(result["expanded_nodes"]))
        self.assertEqual([list(p) for p in result["expanded_nodes"]],
                         restored)

    def test_trace_type_errors(self):
        for bad in (1, 0, "true", None, 1.0, [True]):
            with self.assertRaises(TypeError, msg=f"trace={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), trace=bad)

    def test_existing_validation_unchanged_with_trace(self):
        # trace=True must not turn validation errors into partial results.
        with self.assertRaises(TypeError):
            plan("3", 3, set(), (0, 0), (1, 1), trace=True)
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), trace=True)
        with self.assertRaises(ValueError):
            plan(3, 3, {(0, 0)}, (0, 0), (2, 2), trace=True)
        with self.assertRaises(TypeError):
            plan(3, 3, 42, (0, 0), (1, 1), trace=True)
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 costs=[[1, 1], [1, 1], [1, 1]], trace=True)

    def test_validation_order_trace_checked_last_before_search(self):
        # width=0 is a ValueError that must be reported before the trace
        # TypeError, preserving the existing validation order.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), trace=1)


class DynamicPlannerTest(unittest.TestCase):
    # A case where two (here three) different histories reach the same
    # (x, y, t). The history closed first cannot be extended (its only
    # continuation revisits a cell); a later history reaches the goal.
    # Collapsing equal (x, y, t) states used to report no route here.
    HISTORY_FRAMES = [
        [(0, 1), (0, 2)],
        [(0, 1), (1, 2), (2, 0), (2, 2)],
        [(2, 1), (2, 2)],
        [(0, 0), (1, 2), (2, 1), (2, 2)],
        [],
        [(0, 1)],
    ]
    HISTORY_PATH = [(1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

    def _assert_route_valid(self, result, frames, obstacles=frozenset()):
        path = result["path"]
        self.assertIsNotNone(path)
        self.assertEqual(len(path), len(set(path)))  # no repeated cells
        last = len(frames) - 1
        for t, cell in enumerate(path):
            self.assertNotIn(cell, obstacles)
            self.assertNotIn(cell, frames[t] if t <= last else frames[last])
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_basic_dynamic_route_avoids_frames(self):
        # Only one frame is given, so it persists: (1, 0) is blocked at
        # every t >= 1 and the direct row is unusable.
        frames = [[], [(1, 0)]]
        result = plan(3, 3, set(), (0, 0), (2, 0), dynamic_blocked=frames)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)])
        self.assertEqual(result["cost"], 4)
        self._assert_route_valid(result, frames)
        self.assertNotIn("expanded_nodes", result)

    def test_history_dependent_completeness(self):
        result = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                      dynamic_blocked=self.HISTORY_FRAMES)
        self.assertEqual(result["path"], self.HISTORY_PATH)
        self.assertEqual(result["cost"], 5)
        self._assert_route_valid(result, self.HISTORY_FRAMES)

    def test_equal_cost_tie_uses_path_lexicographic_order(self):
        # Empty 3x3: all shortest routes cost 4; the lexicographically
        # smallest coordinate sequence must win regardless of traversal.
        frames = [[] for _ in range(5)]
        result = plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked=frames)
        self.assertEqual(result["path"],
                         [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])

    def test_no_route_returns_none(self):
        # Frame 1 blocks every neighbor of the start.
        frames = [[], [(1, 0), (0, 1)]]
        result = plan(2, 2, set(), (0, 0), (1, 1), trace=True,
                      dynamic_blocked=frames)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertGreaterEqual(result["expanded"], 1)
        self.assertEqual(result["expanded"], len(result["expanded_nodes"]))

    def test_start_equals_goal_dynamic(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), trace=True,
                      dynamic_blocked=[[], [(0, 0)]])
        self.assertEqual(result["path"], [(1, 1)])
        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["expanded"], 1)
        self.assertEqual(result["expanded_nodes"], [(1, 1, 0)])

    def test_trace_records_histories_separately_in_closing_order(self):
        result = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                      dynamic_blocked=self.HISTORY_FRAMES)
        nodes = result["expanded_nodes"]
        self.assertEqual(result["expanded"], len(nodes))
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 3
                            for s in nodes))
        self.assertEqual(nodes[0], (1, 0, 0))
        # Three distinct histories close (0, 2) at t=3: the triple appears
        # once per closing, in the actual closing order -- never deduplicated.
        self.assertEqual(nodes.count((0, 2, 3)), 3)
        # Every complete feasible route closes the goal once; the last
        # closing is a goal closing.
        self.assertEqual(nodes.count((2, 2, 5)), 4)
        self.assertEqual(nodes[-1], (2, 2, 5))

    def test_result_independent_of_input_ordering(self):
        orders = [
            self.HISTORY_FRAMES,
            [list(reversed(f)) for f in self.HISTORY_FRAMES],
            [set(f) for f in self.HISTORY_FRAMES],
            [tuple(f) + (f[0],) if f else () for f in self.HISTORY_FRAMES],
        ]
        first = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                     dynamic_blocked=orders[0])
        for frames in orders[1:]:
            self.assertEqual(
                plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                     dynamic_blocked=frames),
                first,
            )
        # Repeated static obstacles are merged without changing the result.
        with_dupes = plan(3, 3, [(2, 0), (2, 0)], (1, 0), (2, 2), trace=True,
                          dynamic_blocked=orders[0])
        self.assertEqual(with_dupes["path"], first["path"])
        self.assertEqual(with_dupes["cost"], first["cost"])

    def test_empty_dynamic_blocked_matches_static(self):
        blocked = [(2, 0), (2, 1), (2, 2)]
        static = plan(6, 4, blocked, (0, 0), (5, 3), trace=True)
        for kwargs in ({"dynamic_blocked": None},
                       {"dynamic_blocked": []},
                       {"dynamic_blocked": ()}):
            self.assertEqual(
                plan(6, 4, blocked, (0, 0), (5, 3), trace=True, **kwargs),
                static,
            )

    def test_costs_accounting_preserved(self):
        costs = [[1, 1, 1], [1, 1, 100], [1, 1, 1]]
        result = plan(3, 3, set(), (1, 0), (2, 2), costs=costs,
                      dynamic_blocked=self.HISTORY_FRAMES)
        self.assertEqual(result["path"], self.HISTORY_PATH)
        self.assertEqual(result["cost"],
                         sum(costs[y][x] for x, y in result["path"][1:]))

    def test_validation_preserved_in_dynamic_mode(self):
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (1, 1), dynamic_blocked=42)
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (1, 1), dynamic_blocked="ab")
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (1, 1), dynamic_blocked=[42])
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (1, 1),
                 dynamic_blocked=[[(1, 1.0)]])
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (1, 1),
                 dynamic_blocked=[[(3, 3)]])
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (1, 1),
                 dynamic_blocked=[[(0, 0)]])  # start blocked at frame 0
        # Validation order: width ValueError precedes the trace TypeError.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), trace=1,
                 dynamic_blocked=[[]])


class ReplayTest(unittest.TestCase):
    BLOCKED = {(2, 0), (2, 1), (2, 2)}
    INVALID = {"valid": False, "cost": None, "steps": None}

    def test_valid_static_path_matches_plan(self):
        result = plan(6, 4, self.BLOCKED, (0, 0), (5, 3))
        check = replay(6, 4, self.BLOCKED, (0, 0), (5, 3), result["path"])
        self.assertEqual(check, {"valid": True, "cost": result["cost"],
                                 "steps": len(result["path"]) - 1})
        self.assertNotIn("expanded", check)
        self.assertNotIn("expanded_nodes", check)

    def test_valid_path_with_costs(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        result = plan(3, 3, set(), (0, 0), (2, 0), costs=costs)
        check = replay(3, 3, set(), (0, 0), (2, 0), result["path"],
                       costs=costs)
        expected = sum(costs[y][x] for x, y in result["path"][1:])
        self.assertEqual(check, {"valid": True, "cost": expected,
                                 "steps": len(result["path"]) - 1})
        self.assertEqual(check["cost"], result["cost"])

    def test_valid_dynamic_path_and_last_frame_persistence(self):
        frames = [[], [(1, 0)]]
        result = plan(3, 3, set(), (0, 0), (2, 0), dynamic_blocked=frames)
        check = replay(3, 3, set(), (0, 0), (2, 0), result["path"],
                       dynamic_blocked=frames)
        self.assertEqual(check, {"valid": True, "cost": 4, "steps": 4})
        # The single frame persists: (1, 0) stays blocked at t >= 1.
        through_blocked = [(0, 0), (1, 0), (2, 0)]
        self.assertEqual(replay(3, 3, set(), (0, 0), (2, 0), through_blocked,
                                dynamic_blocked=frames), self.INVALID)

    def test_empty_dynamic_blocked_uses_static_checking(self):
        path = [(0, 0), (1, 0), (2, 0)]
        for kwargs in ({}, {"dynamic_blocked": None},
                       {"dynamic_blocked": []}):
            self.assertEqual(
                replay(3, 1, set(), (0, 0), (2, 0), path, **kwargs),
                {"valid": True, "cost": 2, "steps": 2},
            )

    def test_start_equals_goal(self):
        self.assertEqual(replay(3, 3, set(), (1, 1), (1, 1), [(1, 1)]),
                         {"valid": True, "cost": 0, "steps": 0})
        # Extra points on a zero-move route are invalid.
        self.assertEqual(
            replay(3, 3, set(), (1, 1), (1, 1), [(1, 1), (1, 2), (1, 1)]),
            self.INVALID,
        )

    def test_semantically_invalid_paths_return_fixed_structure(self):
        base = (6, 4, self.BLOCKED, (0, 0), (5, 3))
        good = plan(*base)["path"]
        cases = [
            [(1, 0)] + good[1:],          # does not start at start
            good[:-1] + [(5, 2)],         # does not end at goal
            [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (4, 1), (5, 1),
             (5, 2), (5, 3)],             # crosses static obstacle (2, 0)
            [(0, 0), (2, 0)],             # not a four-neighborhood move
            [(0, 0), (1, 0), (0, 0)],     # repeated coordinate
        ]
        for path in cases:
            with self.subTest(path=path):
                self.assertEqual(replay(*base, path), self.INVALID)

    def test_path_type_errors(self):
        for bad in ("ab", b"ab", None, 42, [(0, 0), "xy"],
                    [(0, 0), (1,)], [(0, 0), (1.5, 2)],
                    [(0, 0), (True, 0)]):
            with self.assertRaises(TypeError, msg=f"path={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), bad)

    def test_path_value_errors(self):
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), [])
        for bad in ([(0, 0), (3, 0)], [(-1, 0), (0, 0)], [(0, 0), (0, 3)]):
            with self.assertRaises(ValueError, msg=f"path={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), bad)

    def test_grid_validation_before_path_and_unchanged(self):
        # Grid errors raise exactly as in plan, even with an invalid path.
        with self.assertRaises(TypeError):
            replay("3", 3, set(), (0, 0), (2, 2), "nope")
        with self.assertRaises(ValueError):
            replay(0, 3, set(), (0, 0), (2, 2), [(0, 0)])
        with self.assertRaises(ValueError):
            replay(3, 3, {(0, 0)}, (0, 0), (2, 2), [(0, 0)])
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                   dynamic_blocked=[[(0, 0)]])
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                   dynamic_blocked=42)

    def test_replay_does_not_change_plan_behavior(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        expected = plan(6, 4, blocked, (0, 0), (5, 3))
        check = replay(6, 4, blocked, (0, 0), (5, 3), expected["path"])
        self.assertEqual(check, {"valid": True, "cost": expected["cost"],
                                 "steps": len(expected["path"]) - 1})
        self.assertEqual(plan(6, 4, blocked, (0, 0), (5, 3)), expected)


if __name__ == '__main__':
    unittest.main()
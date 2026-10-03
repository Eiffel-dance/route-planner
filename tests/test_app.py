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
    INVALID = {"valid": False, "cost": None, "steps": None}

    def test_valid_planned_path_static(self):
        blocked = {(2, 0), (2, 1), (2, 2), (0, 2), (4, 1)}
        planned = plan(6, 4, blocked, (0, 0), (5, 3))
        result = replay(6, 4, blocked, (0, 0), (5, 3), planned["path"])
        self.assertEqual(result, {
            "valid": True,
            "cost": planned["cost"],
            "steps": len(planned["path"]) - 1,
        })
        self.assertEqual(set(result), {"valid", "cost", "steps"})

    def test_valid_single_point_route(self):
        self.assertEqual(
            replay(3, 3, set(), (1, 1), (1, 1), [(1, 1)]),
            {"valid": True, "cost": 0, "steps": 0},
        )

    def test_cost_matches_plan_with_costs(self):
        costs = [
            [1, 10, 1],
            [1, 10, 1],
            [1, 1, 1],
        ]
        planned = plan(3, 3, set(), (0, 0), (2, 0), costs=costs)
        result = replay(3, 3, set(), (0, 0), (2, 0), planned["path"],
                        costs=costs)
        self.assertTrue(result["valid"])
        self.assertEqual(result["cost"], planned["cost"])
        self.assertEqual(result["steps"], len(planned["path"]) - 1)
        # Any manually specified feasible route recomputes entering costs.
        manual = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
        result = replay(3, 3, set(), (0, 0), (2, 0), manual, costs=costs)
        self.assertTrue(result["valid"])
        self.assertEqual(result["cost"],
                         sum(costs[y][x] for x, y in manual[1:]))
        self.assertEqual(result["steps"], 6)

    def test_dynamic_planned_path_is_valid(self):
        frames = [
            [(0, 1), (0, 2)],
            [(0, 1), (1, 2), (2, 0), (2, 2)],
            [(2, 1), (2, 2)],
            [(0, 0), (1, 2), (2, 1), (2, 2)],
            [],
            [(0, 1)],
        ]
        planned = plan(3, 3, set(), (1, 0), (2, 2), dynamic_blocked=frames)
        result = replay(3, 3, set(), (1, 0), (2, 2), planned["path"],
                        dynamic_blocked=frames)
        self.assertEqual(result, {
            "valid": True,
            "cost": planned["cost"],
            "steps": len(planned["path"]) - 1,
        })

    def test_dynamic_frame_at_each_index(self):
        # (1, 0) is free at t=1 but blocked at t=2; arriving there on the
        # second move is invalid.
        frames = [[], [], [(1, 0)]]
        bad = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)]
        self.assertEqual(
            replay(3, 2, set(), (0, 0), (2, 0), bad,
                   dynamic_blocked=frames),
            self.INVALID,
        )
        good = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
        result = replay(3, 2, set(), (0, 0), (2, 0), good,
                        dynamic_blocked=frames)
        self.assertEqual(result, {"valid": True, "cost": 4, "steps": 4})

    def test_last_frame_persists(self):
        # A single non-empty frame 1 blocks (1, 0) for every t >= 1.
        frames = [[], [(1, 0)]]
        bad = [(0, 0), (1, 0), (2, 0)]
        self.assertEqual(
            replay(3, 1, set(), (0, 0), (2, 0), bad,
                   dynamic_blocked=frames),
            self.INVALID,
        )

    def test_empty_or_omitted_dynamic_blocked_matches_static(self):
        path = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
        for kwargs in ({}, {"dynamic_blocked": None},
                       {"dynamic_blocked": []}, {"dynamic_blocked": ()}):
            result = replay(3, 3, set(), (0, 0), (2, 0), path, **kwargs)
            self.assertEqual(result, {"valid": True, "cost": 4, "steps": 4},
                             msg=f"kwargs={kwargs}")

    def test_semantically_invalid_paths(self):
        # Does not begin at start.
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 1), (1, 1), (2, 2)]),
            self.INVALID,
        )
        # Does not end at goal.
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), (1, 0), (2, 0)]),
            self.INVALID,
        )
        # Visits a static obstacle.
        self.assertEqual(
            replay(3, 3, {(1, 0)}, (0, 0), (2, 0),
                   [(0, 0), (1, 0), (2, 0)]),
            self.INVALID,
        )
        # Diagonal (non-four-neighborhood) move.
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 1), (2, 2)]),
            self.INVALID,
        )
        # Jump farther than one cell.
        self.assertEqual(
            replay(5, 1, set(), (0, 0), (4, 0),
                   [(0, 0), (2, 0), (3, 0), (4, 0)]),
            self.INVALID,
        )
        # Repeated coordinate.
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 0), (0, 0), (0, 1), (1, 1), (2, 1),
                    (2, 2)]),
            self.INVALID,
        )
        # start == goal with extra points.
        self.assertEqual(
            replay(3, 3, set(), (1, 1), (1, 1),
                   [(1, 1), (1, 2), (1, 1)]),
            self.INVALID,
        )
        # Frame 0 on the start cell is a ValueError from the shared grid
        # checks, not an invalid result -- covered in validation tests.

    def test_invalid_returns_fixed_structure_without_partial_values(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 0), (1, 1), (2, 2)])
        self.assertIsNone(result["cost"])
        self.assertIsNone(result["steps"])
        self.assertEqual(set(result), {"valid", "cost", "steps"})

    def test_replay_does_not_search(self):
        # A valid route that plan cannot even reach through search in a
        # sealed region can still be replayed (replay never expands nodes);
        # here a manually valid path on an open grid simply verifies the
        # result carries no search statistics.
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
        self.assertNotIn("expanded", result)
        self.assertNotIn("expanded_nodes", result)
        self.assertNotIn("path", result)

    def test_path_type_errors(self):
        for bad in (None, 42, "ab", b"ab", 1.5, {"x": 1}, {1, 2}):
            with self.assertRaises(TypeError, msg=f"path={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), bad)
        # Empty sequence is a TypeError, not an invalid result.
        for bad in ((), [], tuple()):
            with self.assertRaises(TypeError, msg=f"path={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), bad)
        # Elements must be exactly two non-bool integers.
        for bad in (1, "ab", (1,), (1, 2, 3), (1.5, 2), (1, None),
                    (True, 0), None, {"x": 1, "y": 2}):
            with self.assertRaises(TypeError, msg=f"element={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), bad])

    def test_path_value_errors(self):
        for bad in ((-1, 0), (0, -1), (3, 0), (0, 3), (10, 10)):
            with self.assertRaises(ValueError, msg=f"element={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), bad])

    def test_grid_validation_precedes_path_judgement(self):
        # Malformed grid inputs raise the same errors as plan regardless of
        # the candidate path, even when the path itself is also bad.
        bad_path = "not-a-path"
        with self.assertRaises(TypeError):
            replay("3", 3, set(), (0, 0), (1, 1), bad_path)
        with self.assertRaises(ValueError):
            replay(0, 3, set(), (0, 0), (1, 1), bad_path)
        with self.assertRaises(TypeError):
            replay(3, 3, 42, (0, 0), (1, 1), bad_path)
        with self.assertRaises(ValueError):
            replay(3, 3, {(0, 0)}, (0, 0), (2, 2), bad_path)
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (-1, 0), (2, 2), bad_path)
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), "nope", bad_path)
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), bad_path,
                    costs=[[1, 1], [1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (1, 1), bad_path,
                    dynamic_blocked=[[(0, 0)]])
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (1, 1), bad_path,
                    dynamic_blocked=42)

    def test_start_blocked_at_frame_zero_still_raises(self):
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                   dynamic_blocked=[[(0, 0)]])

    def test_lists_normalized_to_tuples_for_comparison(self):
        result = replay(3, 3, [[1, 0]], [0, 0], [2, 2],
                        [[0, 0], [0, 1], [1, 1], [2, 1], [2, 2]])
        self.assertEqual(result, {"valid": True, "cost": 4, "steps": 4})

    def test_replay_matches_plan_costs_across_random_grids(self):
        import random
        rng = random.Random(20261004)
        for _ in range(40):
            w, h = rng.randint(1, 6), rng.randint(1, 6)
            cells = [(x, y) for y in range(h) for x in range(w)]
            blocked = {p for p in cells[1:-1] if rng.random() < 0.2}
            start, goal = cells[0], cells[-1]
            if start in blocked or goal in blocked:
                continue
            if rng.random() < 0.5:
                costs = [[rng.randint(1, 9) for _ in range(w)]
                         for _ in range(h)]
                kwargs = {"costs": costs}
            else:
                kwargs = {}
            planned = plan(w, h, blocked, start, goal, **kwargs)
            if planned["path"] is None:
                continue
            result = replay(w, h, blocked, start, goal, planned["path"],
                            **kwargs)
            self.assertTrue(result["valid"])
            self.assertEqual(result["cost"], planned["cost"])
            self.assertEqual(result["steps"], len(planned["path"]) - 1)


class ReplayDiagnoseTest(unittest.TestCase):
    def test_false_or_omitted_keeps_exact_legacy_shape(self):
        path = [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]
        omitted = replay(3, 3, set(), (0, 0), (2, 0), path)
        explicit = replay(3, 3, set(), (0, 0), (2, 0), path, diagnose=False)
        self.assertEqual(explicit, omitted)
        self.assertEqual(set(omitted), {"valid", "cost", "steps"})
        # Invalid paths keep the exact legacy structure too.
        bad = [(0, 1), (1, 1), (2, 1), (2, 0)]
        self.assertEqual(
            replay(3, 3, set(), (0, 0), (2, 0), bad),
            replay(3, 3, set(), (0, 0), (2, 0), bad, diagnose=False),
        )
        self.assertEqual(
            set(replay(3, 3, set(), (0, 0), (2, 0), bad)),
            {"valid", "cost", "steps"},
        )

    def test_valid_path_reports_none_error_fields(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        planned = plan(6, 4, blocked, (0, 0), (5, 3))
        result = replay(6, 4, blocked, (0, 0), (5, 3), planned["path"],
                        diagnose=True)
        self.assertEqual(result, {
            "valid": True,
            "cost": planned["cost"],
            "steps": len(planned["path"]) - 1,
            "error": None,
            "error_index": None,
        })

    def test_valid_single_point_route(self):
        self.assertEqual(
            replay(3, 3, set(), (1, 1), (1, 1), [(1, 1)], diagnose=True),
            {"valid": True, "cost": 0, "steps": 0,
             "error": None, "error_index": None},
        )

    def test_costs_still_recomputed(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        manual = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
        result = replay(3, 3, set(), (0, 0), (2, 0), manual, costs=costs,
                        diagnose=True)
        self.assertTrue(result["valid"])
        self.assertEqual(result["cost"],
                         sum(costs[y][x] for x, y in manual[1:]))
        self.assertEqual(result["steps"], 6)
        self.assertIsNone(result["error"])
        self.assertIsNone(result["error_index"])

    def test_start_mismatch(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 1), (1, 1), (2, 1), (2, 2)], diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "start_mismatch", "error_index": 0,
        })

    def test_goal_mismatch_points_at_last_element(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 0), (1, 0), (2, 0)], diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "goal_mismatch", "error_index": 2,
        })

    def test_start_mismatch_takes_precedence_over_goal_mismatch(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 1), (1, 1), (2, 1)], diagnose=True)
        self.assertEqual(result["error"], "start_mismatch")
        self.assertEqual(result["error_index"], 0)

    def test_start_goal_extra(self):
        result = replay(3, 3, set(), (1, 1), (1, 1),
                        [(1, 1), (1, 2), (1, 1)], diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "start_goal_extra", "error_index": 1,
        })

    def test_repeated_coordinate_points_at_second_occurrence(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 0), (1, 0), (0, 0), (0, 1), (1, 1),
                         (2, 1), (2, 2)], diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "repeated_coordinate", "error_index": 2,
        })

    def test_non_adjacent_points_at_later_point(self):
        result = replay(5, 1, set(), (0, 0), (4, 0),
                        [(0, 0), (2, 0), (3, 0), (4, 0)], diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "non_adjacent", "error_index": 1,
        })
        diagonal = replay(3, 3, set(), (0, 0), (2, 2),
                          [(0, 0), (1, 1), (2, 2)], diagnose=True)
        self.assertEqual(diagonal["error"], "non_adjacent")
        self.assertEqual(diagonal["error_index"], 1)

    def test_static_blocked(self):
        result = replay(3, 3, {(1, 0)}, (0, 0), (2, 0),
                        [(0, 0), (1, 0), (2, 0)], diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "static_blocked", "error_index": 1,
        })

    def test_dynamic_blocked_at_matching_frame(self):
        # (1, 0) is free at t=1 but blocked at t=2.
        frames = [[], [], [(1, 0)]]
        result = replay(3, 2, set(), (0, 0), (2, 0),
                        [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)],
                        dynamic_blocked=frames, diagnose=True)
        self.assertEqual(result, {
            "valid": False, "cost": None, "steps": None,
            "error": "dynamic_blocked", "error_index": 3,
        })

    def test_dynamic_blocked_under_persisted_last_frame(self):
        frames = [[], [(1, 0)]]
        result = replay(3, 1, set(), (0, 0), (2, 0),
                        [(0, 0), (1, 0), (2, 0)],
                        dynamic_blocked=frames, diagnose=True)
        self.assertEqual(result["error"], "dynamic_blocked")
        self.assertEqual(result["error_index"], 1)

    def test_rule_precedence_within_a_single_element(self):
        # At index 1 the move is non-adjacent and the entered cell is both
        # statically and dynamically blocked; non_adjacent wins.
        frames = [[], [(2, 0)]]
        result = replay(5, 1, {(2, 0)}, (0, 0), (4, 0),
                        [(0, 0), (2, 0), (3, 0), (4, 0)],
                        dynamic_blocked=frames, diagnose=True)
        self.assertEqual(result["error"], "non_adjacent")
        self.assertEqual(result["error_index"], 1)
        # Adjacent but both statically and dynamically blocked: static wins.
        frames = [[], [(1, 0)]]
        result = replay(3, 1, {(1, 0)}, (0, 0), (2, 0),
                        [(0, 0), (1, 0), (2, 0)], dynamic_blocked=frames,
                        diagnose=True)
        self.assertEqual(result["error"], "static_blocked")
        self.assertEqual(result["error_index"], 1)

    def test_invalid_never_raises_and_never_accumulates_cost(self):
        result = replay(3, 3, set(), (0, 0), (2, 2),
                        [(0, 0), (1, 1), (2, 2)],
                        costs=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                        diagnose=True)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["cost"])
        self.assertIsNone(result["steps"])
        self.assertNotIn(None, (result["error"],))
        self.assertEqual(set(result),
                         {"valid", "cost", "steps", "error", "error_index"})

    def test_diagnose_type_error_after_grid_checks_before_path_checks(self):
        # Non-bool diagnose is a TypeError.
        for bad in (1, 0, "true", None, 1.0, [True]):
            with self.assertRaises(TypeError, msg=f"diagnose={bad!r}"):
                replay(3, 3, set(), (0, 0), (2, 2),
                       [(0, 0), (2, 2)], diagnose=bad)
        # Grid ValueError (non-positive width) is reported first.
        with self.assertRaises(ValueError):
            replay(0, 3, set(), (0, 0), (2, 2), "bad-path", diagnose=1)
        # Frame-0 ValueError precedes the diagnose TypeError.
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0)],
                   dynamic_blocked=[[(0, 0)]], diagnose=1)
        # The diagnose TypeError precedes the path structure TypeError.
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2), "bad-path", diagnose=1)
        # diagnose=True still applies the existing path validation.
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2), "bad-path", diagnose=True)
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2), [], diagnose=True)
        with self.assertRaises(ValueError):
            replay(3, 3, set(), (0, 0), (2, 2), [(0, 0), (3, 3)],
                   diagnose=True)

    def test_diagnose_true_matches_plan_on_random_grids(self):
        import random
        rng = random.Random(424242)
        for _ in range(30):
            w, h = rng.randint(1, 6), rng.randint(1, 6)
            cells = [(x, y) for y in range(h) for x in range(w)]
            blocked = {p for p in cells[1:-1] if rng.random() < 0.2}
            start, goal = cells[0], cells[-1]
            if start in blocked or goal in blocked:
                continue
            planned = plan(w, h, blocked, start, goal)
            if planned["path"] is None:
                continue
            result = replay(w, h, blocked, start, goal, planned["path"],
                            diagnose=True)
            self.assertTrue(result["valid"])
            self.assertEqual(result["cost"], planned["cost"])
            self.assertEqual(result["steps"], len(planned["path"]) - 1)
            self.assertIsNone(result["error"])
            self.assertIsNone(result["error_index"])


class BudgetTest(unittest.TestCase):
    WALL = {(1, y) for y in range(3)}  # seals (2, 2) away from (0, 0)

    def test_omitted_or_none_keeps_legacy_shape(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        omitted = plan(6, 4, blocked, (0, 0), (5, 3), trace=True)
        explicit = plan(6, 4, blocked, (0, 0), (5, 3), trace=True,
                        max_expanded=None)
        self.assertEqual(explicit, omitted)
        self.assertNotIn("status", omitted)
        self.assertEqual(set(omitted),
                         {"path", "cost", "expanded", "expanded_nodes"})
        # Unreachable and dynamic results keep the legacy shape too.
        unreachable = plan(3, 3, self.WALL, (0, 0), (2, 2))
        self.assertNotIn("status", unreachable)
        dynamic = plan(3, 3, set(), (0, 0), (2, 0),
                       dynamic_blocked=[[], [(1, 0)]])
        self.assertNotIn("status", dynamic)

    def test_generous_budget_matches_baseline_plus_status(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        baseline = plan(6, 4, blocked, (0, 0), (5, 3), trace=True)
        budgeted = plan(6, 4, blocked, (0, 0), (5, 3), trace=True,
                        max_expanded=1000)
        self.assertEqual(budgeted, {**baseline, "status": "found"})
        self.assertEqual(set(budgeted), {"path", "cost", "expanded",
                                         "expanded_nodes", "status"})

    def test_budget_exhausted_static(self):
        # Only 3 nodes are reachable; a budget of 2 stops the search.
        result = plan(3, 3, self.WALL, (0, 0), (2, 2), max_expanded=2)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 2,
                                  "status": "budget_exhausted"})
        # The same budget with trace records exactly the closed nodes.
        traced = plan(3, 3, self.WALL, (0, 0), (2, 2), trace=True,
                      max_expanded=2)
        self.assertEqual(traced["status"], "budget_exhausted")
        self.assertEqual(traced["expanded_nodes"], [(0, 0), (0, 1)])
        self.assertEqual(traced["expanded"], 2)

    def test_budget_covering_all_candidates_reports_unreachable(self):
        for budget in (3, 10):
            result = plan(3, 3, self.WALL, (0, 0), (2, 2),
                          max_expanded=budget)
            self.assertEqual(result, {"path": None, "cost": None,
                                      "expanded": 3,
                                      "status": "unreachable"})

    def test_goal_closed_exactly_at_budget_is_found(self):
        # 1x2 corridor: start closes, then the goal closes as node 2.
        result = plan(2, 1, set(), (0, 0), (1, 0), max_expanded=2)
        self.assertEqual(result, {"path": [(0, 0), (1, 0)], "cost": 1,
                                  "expanded": 2, "status": "found"})
        # One less and the goal can no longer be closed.
        stopped = plan(2, 1, set(), (0, 0), (1, 0), max_expanded=1)
        self.assertEqual(stopped, {"path": None, "cost": None, "expanded": 1,
                                   "status": "budget_exhausted"})

    def test_zero_budget_closes_nothing(self):
        result = plan(3, 3, set(), (0, 0), (2, 2), trace=True,
                      max_expanded=0)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 0,
                                  "status": "budget_exhausted",
                                  "expanded_nodes": []})

    def test_zero_budget_does_not_fabricate_start_equals_goal(self):
        result = plan(3, 3, set(), (1, 1), (1, 1), trace=True,
                      max_expanded=0)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 0,
                                  "status": "budget_exhausted",
                                  "expanded_nodes": []})
        # Closing the single node requires a budget of at least one.
        found = plan(3, 3, set(), (1, 1), (1, 1), trace=True, max_expanded=1)
        self.assertEqual(found, {"path": [(1, 1)], "cost": 0, "expanded": 1,
                                 "status": "found",
                                 "expanded_nodes": [(1, 1)]})

    def test_budgeted_trace_is_prefix_of_unbudgeted(self):
        blocked = {(2, 0), (2, 1), (2, 2)}
        baseline = plan(6, 4, blocked, (0, 0), (5, 3), trace=True)
        budget = baseline["expanded"] - 1  # goal closes last on success
        result = plan(6, 4, blocked, (0, 0), (5, 3), trace=True,
                      max_expanded=budget)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["expanded"], budget)
        self.assertEqual(result["expanded_nodes"],
                         baseline["expanded_nodes"][:budget])

    def test_budget_preserves_blocked_order_determinism(self):
        cells = [(x, 1) for x in range(5) if x != 2]
        orderings = (cells, list(reversed(cells)),
                     [cells[2], cells[0], cells[3], cells[1]])
        results = [plan(5, 3, order, (0, 0), (4, 2), trace=True,
                        max_expanded=6)
                   for order in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_budget_with_costs(self):
        costs = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
        baseline = plan(3, 3, set(), (0, 0), (2, 0), costs=costs)
        budgeted = plan(3, 3, set(), (0, 0), (2, 0), costs=costs,
                        max_expanded=100)
        self.assertEqual(budgeted, {**baseline, "status": "found"})
        replayed = replay(3, 3, set(), (0, 0), (2, 0), budgeted["path"],
                          costs=costs)
        self.assertEqual(replayed, {"valid": True, "cost": budgeted["cost"],
                                    "steps": len(budgeted["path"]) - 1})

    def test_budget_type_and_value_errors(self):
        for bad in ("5", 1.5, True, [5], (5,), 2.0):
            with self.assertRaises(TypeError, msg=f"max_expanded={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_expanded=bad)
        for bad in (-1, -100):
            with self.assertRaises(ValueError, msg=f"max_expanded={bad!r}"):
                plan(3, 3, set(), (0, 0), (2, 2), max_expanded=bad)

    def test_budget_validated_after_all_existing_checks(self):
        # Grid ValueError precedes the max_expanded TypeError.
        with self.assertRaises(ValueError):
            plan(0, 3, set(), (0, 0), (1, 1), max_expanded="x")
        # Obstacle ValueError precedes the max_expanded TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, {(0, 0)}, (0, 0), (2, 2), max_expanded="x")
        # Costs ValueError precedes the max_expanded TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 costs=[[1, 1], [1, 1], [1, 1]], max_expanded="x")
        # Trace TypeError precedes the max_expanded ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), trace=1, max_expanded=-1)
        # Frame-0 ValueError precedes the max_expanded TypeError.
        with self.assertRaises(ValueError):
            plan(3, 3, set(), (0, 0), (2, 2),
                 dynamic_blocked=[[(0, 0)]], max_expanded="x")
        # Dynamic structure TypeError precedes the max_expanded ValueError.
        with self.assertRaises(TypeError):
            plan(3, 3, set(), (0, 0), (2, 2), dynamic_blocked=42,
                 max_expanded=-1)

    def test_dynamic_generous_budget_matches_baseline(self):
        frames = DynamicPlannerTest.HISTORY_FRAMES
        baseline = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                        dynamic_blocked=frames)
        budgeted = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                        dynamic_blocked=frames, max_expanded=1000)
        self.assertEqual(budgeted, {**baseline, "status": "found"})
        self.assertEqual(budgeted["path"],
                         DynamicPlannerTest.HISTORY_PATH)

    def test_dynamic_budget_exhausted(self):
        frames = DynamicPlannerTest.HISTORY_FRAMES
        baseline = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                        dynamic_blocked=frames)
        result = plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                      dynamic_blocked=frames, max_expanded=3)
        self.assertIsNone(result["path"])
        self.assertIsNone(result["cost"])
        self.assertEqual(result["expanded"], 3)
        self.assertEqual(result["status"], "budget_exhausted")
        # Only actually closed space-time states are recorded, as a
        # prefix of the unbudgeted trace.
        self.assertEqual(result["expanded_nodes"],
                         baseline["expanded_nodes"][:3])
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 3
                            for s in result["expanded_nodes"]))

    def test_dynamic_unreachable_vs_budget_exhausted(self):
        # Frame 1 blocks every neighbor of the start: no route exists.
        frames = [[], [(1, 0), (0, 1)]]
        result = plan(2, 2, set(), (0, 0), (1, 1), dynamic_blocked=frames,
                      max_expanded=10)
        self.assertEqual(result, {"path": None, "cost": None, "expanded": 1,
                                  "status": "unreachable"})
        stopped = plan(2, 2, set(), (0, 0), (1, 1), dynamic_blocked=frames,
                       max_expanded=0)
        self.assertEqual(stopped, {"path": None, "cost": None, "expanded": 0,
                                   "status": "budget_exhausted"})

    def test_dynamic_start_equals_goal_budget(self):
        frames = [[], [(0, 0)]]
        stopped = plan(3, 3, set(), (1, 1), (1, 1), trace=True,
                       dynamic_blocked=frames, max_expanded=0)
        self.assertEqual(stopped, {"path": None, "cost": None, "expanded": 0,
                                   "status": "budget_exhausted",
                                   "expanded_nodes": []})
        found = plan(3, 3, set(), (1, 1), (1, 1), trace=True,
                     dynamic_blocked=frames, max_expanded=1)
        self.assertEqual(found, {"path": [(1, 1)], "cost": 0, "expanded": 1,
                                 "status": "found",
                                 "expanded_nodes": [(1, 1, 0)]})

    def test_dynamic_budget_preserves_frame_order_determinism(self):
        frames = DynamicPlannerTest.HISTORY_FRAMES
        orderings = [
            frames,
            [list(reversed(f)) for f in frames],
            [set(f) for f in frames],
        ]
        results = [plan(3, 3, set(), (1, 0), (2, 2), trace=True,
                        dynamic_blocked=f, max_expanded=5)
                   for f in orderings]
        for other in results[1:]:
            self.assertEqual(other, results[0])

    def test_replay_ignores_budget_and_stays_legacy(self):
        # replay accepts no budget parameter at all.
        with self.assertRaises(TypeError):
            replay(3, 3, set(), (0, 0), (2, 2),
                   [(0, 0), (1, 1), (2, 2)], max_expanded=3)
        # A path found under a budget replays exactly as before.
        frames = [[], [(1, 0)]]
        planned = plan(3, 3, set(), (0, 0), (2, 0), dynamic_blocked=frames,
                       max_expanded=100)
        self.assertEqual(planned["status"], "found")
        result = replay(3, 3, set(), (0, 0), (2, 0), planned["path"],
                        dynamic_blocked=frames)
        self.assertEqual(result, {"valid": True, "cost": planned["cost"],
                                  "steps": len(planned["path"]) - 1})


if __name__ == '__main__':
    unittest.main()
